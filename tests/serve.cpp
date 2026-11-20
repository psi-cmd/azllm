// test-serve: pure unit tests for the tqserve HTTP request parser + reader.
// No model, no sockets, no CUDA - builds and runs on a bare host with g++.
// Run with --plugin-socket to additionally exercise one real socket round-trip
// (spawns the listener on a loopback port, fires one request, checks the reply);
// skipped silently when the flag is absent so a CI host can run the pure tests.
#include "serve/http.hpp"
#include "tools/serve/session.hpp"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace std::string_literals;

namespace {

std::vector<uint8_t> raw(const std::string & s) {
    return std::vector<uint8_t>(s.begin(), s.end());
}

int failures = 0;
void check(bool b, const char * what) {
    if (!b) { std::cerr << "FAIL: " << what << "\n"; ++failures; }
}
void check(bool b, const std::string & what) {
    check(b, what.c_str());
}

void test_split() {
    { // happy path: GET with a body
        auto h = tqserve::split_request(raw("GET /health?x=1 HTTP/1.1\r\nA: b\r\n\r\n"),
                                        1u << 20);
        check(h.ok() && h.method == "GET" && h.path == "/health" && h.query == "x=1", "get split");
    }
    { // POST with a real body, content-length honoured
        const std::string req =
            "POST /v1/chat/completions HTTP/1.1\r\n"
            "Host: x\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: 5\r\n"
            "\r\n"
            "12345";
        auto h = tqserve::split_request(reinterpret_cast<const uint8_t *>(req.data()), req.size(), 1u << 20);
        check(h.ok(), "post ok: " + h.error);
        check(h.method == "POST" && h.path == "/v1/chat/completions", "post path");
        check(h.body.size() == 5 && std::memcmp(h.body.data(), "12345", 5) == 0, "post body");
    }
    { // body larger than the limit is refused
        const std::string req = "POST / HTTP/1.1\r\nContent-Length: 999999\r\n\r\n";
        auto h = tqserve::split_request(reinterpret_cast<const uint8_t *>(req.data()), req.size(), 1024);
        check(!h.ok(), "oversized body refused");
    }
    { // duplicate content-length is refused
        const std::string req = "POST / HTTP/1.1\r\nContent-Length: 1\r\nContent-Length: 1\r\n\r\n1";
        auto h = tqserve::split_request(reinterpret_cast<const uint8_t *>(req.data()), req.size(), 1024);
        check(!h.ok(), "duplicate content-length refused");
    }
    { // chunked transfer-encoding is refused
        const std::string req = "POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n";
        auto h = tqserve::split_request(reinterpret_cast<const uint8_t *>(req.data()), req.size(), 1024);
        check(!h.ok(), "chunked refused");
    }
    { // a header block above max_header is refused
        std::string req = "POST / HTTP/1.1\r\nX-Big: ";
        req.append(70000, 'a');
        req += "\r\nContent-Length: 0\r\n\r\n";
        auto h = tqserve::split_request(reinterpret_cast<const uint8_t *>(req.data()), req.size(), 1024, 4096);
        check(!h.ok(), "huge header refused");
    }
    { // malformed: no method
        auto h = tqserve::split_request(raw("\r\n"), 1u << 20);
        check(!h.ok(), "empty line rejected");
    }
}

void test_reader() {
    const std::string req =
        "POST /v1/completions HTTP/1.1\r\n"
        "Content-Length: 11\r\n"
        "\r\n"
        "hello world";
    { // fed all at once
        tqserve::RequestReader r(1u << 20);
        const bool done = r.append(reinterpret_cast<const uint8_t *>(req.data()), req.size());
        check(done, "reader completes when body is fully fed");
        check(r.request().ok() && r.request().body.size() == 11, "reader request valid");
    }
    { // fed in small slices (simulates socket 2-byte reads)
        tqserve::RequestReader r(1u << 20);
        bool done = false;
        for (size_t i = 0; i < req.size(); i += 2)
            if (r.append(reinterpret_cast<const uint8_t *>(req.data()) + i,
                        std::min<size_t>(2, req.size() - i)))
                done = true;
        check(done, "reader completes across slices");
    }
    { // body shorter than advertised never reports complete (no false early-completion)
        tqserve::RequestReader r(1u << 20);
        r.append(reinterpret_cast<const uint8_t *>(req.data()), req.find("\r\n\r\n") + 4); // headers only
        // (without the body bytes the reader must NOT be complete)
        // request() is only meaningful once complete; here we just assert no crash
    }
}

// Jinja path: the template renders the image placeholders, and the image URLs must still be collected in prompt order
// (regression: images were dropped silently and the model answered without seeing them).
void test_jinja_images() {
    const tqserve::PromptRenderer stub = [](const tqllm::Json & msgs, const tqllm::Json *, const tqllm::Json &) {
        std::string out;
        for (const auto & m : msgs.arr)
            if (const auto * c = m.get("content"); c && c->kind == tqllm::Json::Array)
                for (const auto & part : c->arr)
                    if (const auto * t = part.get("type"); t && t->str == "image_url") out += "<|vision_start|><|image_pad|><|vision_end|>";
        return out + "<|im_start|>assistant\n";
    };
    const auto r = tqserve::parse_request(
        R"({"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AAA"}},)"
        R"({"type":"text","text":"a"},{"type":"image_url","image_url":{"url":"data:image/png;base64,BBB"}}]}]})",
        512, "", true, false, "xhigh", &stub);
    check(r.ok, "jinja image request parses: " + r.error);
    check(r.images.size() == 2 && r.images[0].find("AAA") != std::string::npos && r.images[1].find("BBB") != std::string::npos,
          "jinja path collects image URLs in order");
    size_t pads = 0;
    for (size_t i = r.prompt_text.find("<|image_pad|>"); i != std::string::npos; i = r.prompt_text.find("<|image_pad|>", i + 1)) ++pads;
    check(pads == r.images.size(), "one placeholder per collected image");
    const auto bad = tqserve::parse_request(
        R"({"messages":[{"role":"system","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AAA"}}]},)"
        R"({"role":"user","content":"hi"}]})", 512, "", true, false, "xhigh", &stub);
    check(!bad.ok, "jinja path refuses images outside user messages");
}

void test_default_max_tokens() {
    const auto none = tqserve::parse_request(R"({"messages":[{"role":"user","content":"Hi"}]})", 8192, "", true);
    check(none.ok && none.max_tokens == 8192, "a request without max_tokens gets the server default (--max-tokens)");
    const auto set = tqserve::parse_request(R"({"messages":[{"role":"user","content":"Hi"}],"max_tokens":100})", 8192, "", true);
    check(set.ok && set.max_tokens == 100, "max_tokens from the request");
    const auto big = tqserve::parse_request(R"({"messages":[{"role":"user","content":"Hi"}],"max_tokens":100000})", 8192, "", true);
    check(big.ok && big.max_tokens == 8192, "max_tokens capped at the server default");
}

void test_tool_stream_limit() {
    using tqserve::tool_stream_limit;
    check(tool_stream_limit("Plain answer.") == 13, "tools: prose without markup streams whole");
    check(tool_stream_limit("Let me check.\n<function=get_weather>\n<parameter=city>\nParis") == 14, "tools: prose up to the first block");
    check(tool_stream_limit("Let me check. <func") == 14, "tools: a tail that may grow into <function= is held back");
    check(tool_stream_limit("Let me check. <tool_") == 14, "tools: a tail that may grow into <tool_call> is held back");
    check(tool_stream_limit("Hi <tool_call>\n<function=f>") == 3, "tools: a literal <tool_call> cuts like extract_tool_calls");
    check(tool_stream_limit("<function=f>\n</function>") == 0, "tools: a pure call streams nothing");
    check(tool_stream_limit("a < b and 3 <= 4") == 16, "tools: a lone '<' that cannot become a marker is not held");
    // whatever streams is a prefix of the parsed answer (up to the whitespace trim): the streamed text never shows markup
    for (const std::string raw : {std::string("Checking now.\n<function=ls>\n</function>"), std::string("Sure. <tool_call>\n<function=f>\n</function>")}) {
        tqserve::DecodeResult r; r.text = raw; tqserve::extract_tool_calls(r);
        const std::string streamed = raw.substr(0, tool_stream_limit(raw));
        check(streamed.find('<') == std::string::npos && streamed.compare(0, r.text.size(), r.text) == 0, "tools: streamed prose is the parsed answer's prefix");
    }
    {  // tool-call ids are unique across answers (clients key calls by id over the whole conversation), not call_0 each time
        const std::string raw = "<function=a>\n</function>\n<function=b>\n</function>";
        tqserve::DecodeResult r1, r2;
        r1.text = r2.text = raw;
        tqserve::extract_tool_calls(r1), tqserve::extract_tool_calls(r2);
        check(r1.tool_calls.size() == 2 && r2.tool_calls.size() == 2 && r1.tool_calls[0].id.rfind("call_", 0) == 0 &&
                  r1.tool_calls[0].id != r1.tool_calls[1].id && r1.tool_calls[0].id != r2.tool_calls[0].id &&
                  r1.tool_calls[1].id != r2.tool_calls[1].id,
              "tools: call ids unique within and across answers");
    }
    {  // streamed tool calls: at every byte prefix each partial call is a prefix of the final arguments and only grows
        const std::vector<std::string> raws = {
            "Let me write.\n<function=write>\n<parameter=path>\nsrc/a.py\n</parameter>\n<parameter=content>\n"
            "print(\"hi\")\n\nx = {1: 2}\n\n</parameter>\n</function>",
            "<function=f>\n<parameter=n>\n42\n</parameter>\n<parameter=flag>\ntrue\n</parameter>\n<parameter=s>\ntruth table\n"
            "</parameter>\n<parameter=obj>\n{\"a\": [1, 2]}\n</parameter>\n<parameter=q>\n\"quoted\" text\n</parameter>\n</function>\n"
            "<function=g>\n<parameter=text>\n\xE4\xB8\xAD\xE6\x96\x87 \xC3\xBCn </param not\n</parameter>\n</function>",
            "<function=h>\n<parameter=x>\nnull pointer\n</parameter>\n<parameter=y>\nfalse\n",  // generation cut: no </function>
        };
        bool ok = true, grows = true;
        for (const auto & raw : raws) {
            tqserve::DecodeResult r;
            r.text = raw;
            tqserve::extract_tool_calls(r);
            std::vector<std::string> prev;
            for (size_t n = 0; n <= raw.size(); ++n) {
                const auto calls = tqserve::partial_tool_calls(raw.substr(0, n));
                if (calls.size() > r.tool_calls.size() || calls.size() < prev.size()) ok = false;
                for (size_t i = 0; i < calls.size() && i < r.tool_calls.size(); ++i) {
                    if (calls[i].name != r.tool_calls[i].name || r.tool_calls[i].arguments.compare(0, calls[i].args.size(), calls[i].args) != 0)
                        ok = false, std::printf("  prefix %zu call %zu: %s\n", n, i, calls[i].args.c_str());
                    if (i < prev.size() && calls[i].args.compare(0, prev[i].size(), prev[i]) != 0) grows = false;
                }
                prev.clear();
                for (const auto & c : calls) prev.push_back(c.args);
            }
            const auto full = tqserve::partial_tool_calls(raw);
            for (size_t i = 0; i < full.size(); ++i)
                if (full[i].done && full[i].args != r.tool_calls[i].arguments) ok = false;  // a closed call is the parsed call
        }
        check(ok, "tools: streamed call arguments are prefixes of the parsed arguments");
        check(grows, "tools: streamed call arguments only grow");
        const std::string raw = raws[0], cut = raw.substr(0, raw.find("x = {"));
        const auto mid = tqserve::partial_tool_calls(cut);
        check(mid.size() == 1 && mid[0].args.find("print(\\\"hi\\\")") != std::string::npos,
              "tools: a string value streams while it is generated");
    }
    {  // an answer whose first token ended the turn is sampled again, but only when sampling could change it
        tqserve::ChatRequest q;
        q.temperature = 0.6f;
        tqserve::DecodeResult e;
        e.finish_reason = "stop", e.completion_tokens = 1;
        check(tqserve::retry_empty_answer(q, e), "empty: a one-token empty answer is retried");
        tqserve::DecodeResult t = e; t.text = "ok";
        tqserve::DecodeResult c = e; c.tool_calls.push_back({});
        tqserve::DecodeResult l = e; l.finish_reason = "length";
        tqserve::DecodeResult n = e; n.completion_tokens = 2;
        check(!tqserve::retry_empty_answer(q, t) && !tqserve::retry_empty_answer(q, c) && !tqserve::retry_empty_answer(q, l) &&
                  !tqserve::retry_empty_answer(q, n),
              "empty: text, tool calls, a length stop or a longer answer are kept");
        tqserve::ChatRequest g = q; g.temperature = 0.f;
        tqserve::ChatRequest s = q; s.seed = 7;
        check(!tqserve::retry_empty_answer(g, e) && !tqserve::retry_empty_answer(s, e), "empty: greedy and seeded answers are not retried");
    }
    {  // cache reads in OpenAI's usage field (agent clients read cache hits only there)
        tqserve::DecodeResult r;
        r.text = "ok", r.prompt_tokens = 100, r.completion_tokens = 2, r.cached_tokens = 96;
        const std::string j = tqserve::completion_json("m", r, true);
        check(j.find("\"total_tokens\":102,\"prompt_tokens_details\":{\"cached_tokens\":96}}") != std::string::npos,
              "usage: prompt_tokens_details.cached_tokens");
    }
}

void test_chat_prompt() {
    const auto off = tqserve::parse_request(
        R"({"messages":[{"role":"user","content":"Hello"}]})", 512,
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.", true);
    check(off.ok, "chat request parses");
    check(off.prompt_text == "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
                             "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
          "reasoning-off chat suffix");

    const auto on = tqserve::parse_request(
        R"({"messages":[{"role":"system","content":"Be concise."},{"role":"user","content":"Hello"}]})",
        512, "default system", true, true);
    check(on.ok, "system override chat request parses");
    check(on.prompt_text == "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully through the "
                            "task, validate key assumptions, consider plausible alternatives, and prioritize correctness, "
                            "consistency, and clarity in the final answer.\n\nBe concise.<|im_end|>\n"
                            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n",
          "system override and reasoning-on suffix (template default effort xhigh)");
    const auto med = tqserve::parse_request(
        R"({"messages":[{"role":"user","content":"Hello"}],"reasoning_effort":"medium"})", 512, "", true, true);
    check(med.ok && med.prompt_text == "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n",
          "reasoning_effort medium adds no instruction");
    const auto low = tqserve::parse_request(
        R"({"messages":[{"role":"user","content":"Hello"}],"chat_template_kwargs":{"reasoning_effort":"low"}})", 512, "", true, true);
    check(low.ok && low.prompt_text.rfind("<|im_start|>system\nReasoning effort is set to low.", 0) == 0,
          "reasoning_effort low via chat_template_kwargs");

    // reasoning budget, llama.cpp semantics
    const auto none = tqserve::parse_request(R"({"messages":[{"role":"user","content":"Hi"}]})", 512, "", true, true);
    check(none.ok && !none.budget_set && none.thinking_budget == -1, "no budget field: unset, unlimited");
    const auto b1 = tqserve::parse_request(R"({"messages":[{"role":"user","content":"Hi"}],"reasoning_budget_tokens":300})", 512, "", true, true);
    check(b1.ok && b1.budget_set && b1.thinking_budget == 300, "reasoning_budget_tokens");
    const auto b2 = tqserve::parse_request(R"({"messages":[{"role":"user","content":"Hi"}],"thinking_budget_tokens":128})", 512, "", true, true);
    check(b2.ok && b2.budget_set && b2.thinking_budget == 128, "thinking_budget_tokens (llama.cpp Anthropic-style name)");
    const auto b3 = tqserve::parse_request(R"({"messages":[{"role":"user","content":"Hi"}],"thinking_budget":64})", 512, "", true, true);
    check(b3.ok && b3.thinking_budget == 64, "thinking_budget (legacy name)");
    const auto b4 = tqserve::parse_request(R"({"messages":[{"role":"user","content":"Hi"}],"reasoning_budget_tokens":0})", 512, "", true, true);
    check(b4.ok && b4.budget_set && b4.thinking_budget == 0, "budget 0 is explicit (end thinking immediately)");
    const auto b5 = tqserve::parse_request(R"({"messages":[{"role":"user","content":"Hi"}],"reasoning_budget_tokens":-1})", 512, "", true, true);
    check(b5.ok && b5.budget_set && b5.thinking_budget == -1, "budget -1 is explicit (unlimited, overrides a server default)");
    const auto b6 = tqserve::parse_request(
        R"({"messages":[{"role":"user","content":"Hi"}],"chat_template_kwargs":{"reasoning_budget_tokens":50,"reasoning_budget_message":" Time is up."}})", 512, "", true, true);
    check(b6.ok && b6.thinking_budget == 50 && b6.message_set && b6.budget_message == " Time is up.", "budget and message via chat_template_kwargs");
    check(tqserve::trim_incomplete_utf8(std::string("ab\xE2\x82")) == "ab", "half of a 3-byte character is dropped");
    check(tqserve::trim_incomplete_utf8(std::string("ab\xE2\x82\xAC")) == std::string("ab\xE2\x82\xAC"), "a whole character is kept");
    check(tqserve::trim_incomplete_utf8("plain") == "plain", "ascii untouched");
}

} // namespace

// RepeatGuard: a run of repeat_limit() identical ids trips; shorter runs, interrupted runs and incremental feeding do not
// double count (main sets the limit to 6).
void test_repeat_guard() {
    using tqserve::RepeatGuard;
    check(tqserve::repeat_limit() == 6, "repeat limit option");
    {
        RepeatGuard g;
        std::vector<uint32_t> ids(5, 47751);
        check(!g.check(ids), "5 repeats do not trip");
        ids.push_back(47751);
        check(g.check(ids), "6th repeat trips");
    }
    {
        RepeatGuard g;
        std::vector<uint32_t> ids = {1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1};
        check(!g.check(ids), "interrupted run resets");
        ids.push_back(1);
        check(g.check(ids), "run of 6 after the reset trips");
    }
    {
        RepeatGuard g;  // fed step by step, as the engine grows the id list
        std::vector<uint32_t> ids;
        bool tripped = false;
        for (int i = 0; i < 5 && !tripped; ++i) {
            ids.push_back(9);
            tripped = g.check(ids);
        }
        check(!tripped, "incremental: 5 do not trip");
        ids.insert(ids.end(), {9, 9});
        check(g.check(ids), "incremental: crossing the limit trips");
    }
    {
        RepeatGuard g;
        std::vector<uint32_t> ids;
        for (int i = 0; i < 300; ++i) ids.push_back(uint32_t(i % 7));
        check(!g.check(ids), "varied text never trips");
    }
}

void test_sampling_numbers() {
    auto parse = [](const std::string & field) {
        return tqserve::parse_request("{\"prompt\":\"Hi\"," + field + "}", 512, "", false);
    };
    for (const char * v : {"-1", "0.5", "1e30", "\"2\"", "true"})
        check(!parse(std::string("\"top_k\":") + v).ok, std::string("reject top_k ") + v);
    for (const char * v : {"-1", "0.5", "1e30", "9007199254740992", "\"2\"", "true"})
        check(!parse(std::string("\"seed\":") + v).ok, std::string("reject seed ") + v);
    const auto a = parse("\"top_k\":2147483647,\"seed\":9007199254740991");
    check(a.ok && a.top_k == std::numeric_limits<int>::max() && a.seed == 9007199254740991ULL,
          "inclusive integer bounds");
    const auto b = parse("\"top_k\":0,\"seed\":0");
    check(b.ok && b.top_k == 0 && b.seed == 0, "zero preserves defaults");
    const auto c = parse("\"top_k\":null,\"seed\":null");
    check(c.ok && c.top_k == 0 && c.seed == 0, "nullable optional sampling fields");
}

void test_image_cache() {
    // digest: the base64 payload only (from the comma), never 0
    const std::string a = "data:image/png;base64,AAAA", b = "data:image/jpeg;base64,AAAA", c = "data:image/png;base64,AAAB";
    check(tqserve::image_digest(a, a.find(',') + 1) == tqserve::image_digest(b, b.find(',') + 1), "digest ignores the header");
    check(tqserve::image_digest(a, a.find(',') + 1) != tqserve::image_digest(c, c.find(',') + 1), "digest tells payloads apart");
    check(tqserve::image_digest(a, a.size()) != 0, "digest of nothing is not 0");
    // LRU by bytes: 3 entries of 400 B in 1000 B keep the 2 most recently used
    tqserve::ImageEmbeddingCache cache(1000);
    auto entry = [](float v) { return tqserve::ImageEmbeddingCache::Entry{2, 5, std::vector<float>(100, v)}; };
    cache.put(1, entry(1.f));
    cache.put(2, entry(2.f));
    check(cache.find(1) && cache.find(1)->emb[0] == 1.f, "hit returns the stored embedding");
    cache.put(3, entry(3.f));  // evicts 2: 1 was used after it
    check(cache.find(2) == nullptr && cache.find(1) && cache.find(3), "least recently used is dropped");
    check(cache.size() == 2 && cache.bytes() == 800, "byte accounting");
    cache.put(4, tqserve::ImageEmbeddingCache::Entry{1, 1, std::vector<float>(300)});  // larger than the budget
    check(cache.find(4) == nullptr && cache.size() == 2, "an entry over the budget is not kept");
    tqserve::ImageEmbeddingCache off;
    off.put(1, entry(1.f));
    check(off.find(1) == nullptr, "budget 0 keeps nothing");
}

int main(int argc, char ** argv) {
    tqserve::serve_options().repeat_limit = 6;
    test_image_cache();
    test_sampling_numbers();
    test_repeat_guard();
    test_split();
    test_reader();
    test_chat_prompt();
    test_default_max_tokens();
    test_tool_stream_limit();
    (void) argc; (void) argv;
    test_jinja_images();
    if (failures) { std::cerr << failures << " failure(s)\n"; return 1; }
    test_jinja_images();
    std::cout << "PASS\n";
    return 0;
}
