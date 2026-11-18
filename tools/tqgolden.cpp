// tqgolden: validation dump for the golden-fixture gates (Q4).
// Usage: tqgolden --model-dir DIR (--prompt TEXT | --text FORMATTED) --out PREFIX [--greedy N] [--context N]
// --prompt wraps TEXT in the Qwen2.5 chat template; --text tokenizes the given string as-is.
// Writes PREFIX.ids (chat-formatted token ids, one line), PREFIX.hidden.f32 ([layers][hidden]),
// PREFIX.logits.f32 ([vocab]) and PREFIX.greedy (generated ids, one line).
#include "tqllm/options.hpp"
#include "tqllm/model.hpp"
#include "tqllm/tokenizer.hpp"

#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
void write_ids(const std::string & path, const std::vector<uint32_t> & ids) {
    std::ofstream o(path);
    for (size_t i = 0; i < ids.size(); ++i) o << (i ? " " : "") << ids[i];
    o << '\n';
}

void write_f32(const std::string & path, const std::vector<float> & v) {
    std::ofstream o(path, std::ios::binary);
    o.write(reinterpret_cast<const char *>(v.data()), std::streamsize(v.size() * sizeof(float)));
}

void write_u16(const std::string & path, const std::vector<uint16_t> & v) {
    std::ofstream o(path, std::ios::binary);
    o.write(reinterpret_cast<const char *>(v.data()), std::streamsize(v.size() * sizeof(uint16_t)));
}
} // namespace

int main(int argc, char ** argv) {
    tqllm::check_env_flags("tqgolden");
    try {
        std::string model_file, weights, model_dir, tok_override, prompt, text, out;
        uint32_t greedy = 32, context = 2048;
        tqllm::EngineOptions engine;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto value = [&]() -> std::string {
                if (++i >= argc) throw std::invalid_argument("missing value for " + arg);
                return argv[i];
            };
            if (arg == "--model") model_file = value();
            else if (arg == "--tokenizer") tok_override = value();
            else if (arg == "--model-dir") model_dir = value();
            else if (arg == "--weights") weights = value();  // e.g. a .gguf; tokenizer still from --model-dir
            else if (arg == "--prompt") prompt = value();
            else if (arg == "--text") text = value();
            else if (arg == "--out") out = value();
            else if (arg == "--greedy") greedy = std::stoul(value());
            else if (arg == "--context") context = std::stoul(value());
            else if (tqllm::parse_engine_option(argc, argv, i, engine)) continue;
            else throw std::invalid_argument("unknown argument " + arg);
        }
        if ((model_dir.empty() && model_file.empty()) || (prompt.empty() == text.empty()) || out.empty())
            throw std::invalid_argument("need (--model-dir DIR | --model FILE), exactly one of --prompt/--text, --out");
        const std::string wpath = !model_file.empty() ? model_file
                                : (!weights.empty() ? weights : (std::filesystem::path(model_dir) / "weights.tqbin").string());
        const std::string tpath = !tok_override.empty() ? tok_override
                                : (!model_file.empty() ? model_file : (std::filesystem::path(model_dir) / "tokenizer.tqtok").string());
        tqllm::Tokenizer tokenizer;
        tokenizer.load(tpath);
        constexpr const char * system = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";
        const auto ids = !text.empty() ? tokenizer.encode(text)
                                       : tokenizer.encode(std::string("<|im_start|>system\n") + system +
                                                          "<|im_end|>\n<|im_start|>user\n" + prompt +
                                                          "<|im_end|>\n<|im_start|>assistant\n");
        write_ids(out + ".ids", ids);
        if (!model_file.empty()) {
            // GGUF verify/prefill golden mode (supports TQ4, DFlash2 hidden-state features, full-vocab logits & speculative greedy)
            tqllm::Model model(wpath, tpath, context, 1, engine);
            model.set_full_logits(true);
            const size_t V = model.vocab_size();
            const uint32_t H = model.hidden_size();
            if (model.has_drafter()) {
                std::vector<uint16_t> feats;
                const int nf = model.drafter_features(ids, feats);
                if (nf > 0) {
                    write_u16(out + ".dflash_feats.bf16", feats);
                    std::ofstream meta(out + ".dflash_meta.json");
                    meta << "{\"num_tokens\":" << ids.size() << ",\"nf\":" << nf << ",\"hidden_size\":" << H << "}\n";
                }
            }
            if (ids.size() > 1) {
                std::vector<uint32_t> prefix(ids.begin(), ids.end() - 1);
                if (!model.prefill_state(prefix, 0))
                    throw std::runtime_error("prefill_state unsupported on this model");
            }
            std::vector<uint32_t> last_tok{ids.back()}, next;
            model.verify(last_tok, uint32_t(ids.size() - 1), next);
            std::vector<float> logits(V);
            if (cudaMemcpy(logits.data(), model.verify_logits(), V * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
                throw std::runtime_error("cudaMemcpy of verify_logits failed");
            model.commit(1);
            write_f32(out + ".logits.f32", logits);

            write_ids(out + ".greedy", model.generate(ids, greedy));
            if (model.has_drafter()) {
                uint64_t stats[5]{};
                model.set_sampling(0.f);
                write_ids(out + ".greedy_draft", model.generate_spec(ids, greedy, 7, stats));
                std::ofstream st(out + ".spec_stats.json");
                st << "{\"steps\":" << stats[0] << ",\"drafted\":" << stats[1] << ",\"accepted\":" << stats[2]
                   << ",\"decode_us\":" << stats[3] << ",\"prefill_us\":" << stats[4] << "}\n";
            }
            return 0;
        }
        tqllm::EngineOptions base_engine = engine;
        base_engine.draft_model.clear();
        base_engine.draft_vocab.clear();
        std::vector<float> hidden, logits;
        {
            tqllm::Model model(wpath, tpath, context, 1, base_engine);
            model.capture(ids, hidden, logits);
        }
        write_f32(out + ".hidden.f32", hidden);
        write_f32(out + ".logits.f32", logits);
        {
            tqllm::Model model(wpath, tpath, context, 1, base_engine);
            write_ids(out + ".greedy", model.generate(ids, greedy));
        }
        return 0;
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
