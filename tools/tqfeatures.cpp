// tqfeatures: DFlash2 training feature server. Streams the drafter target-layer features of token sequences.
// Usage: tqfeatures --model TARGET.gguf -md DRAFTER.gguf [--context N] [engine options]
// Protocol (one request per line on stdin, one binary frame per request on stdout):
//   request:  space-separated token ids, newline-terminated (an empty line is skipped; EOF ends the server)
//   response: "TQF1" u32 P, u32 nf, u32 hidden, then P * nf * hidden BF16 values (little endian),
//             row t = the features [nf][hidden] the drafter reads for position t (residual input of each target layer).
//   A request that fails (too long, bad id) answers P = 0 with nf = hidden = 0 and logs the reason on stderr.
// The model is loaded once; the drafter is only needed for its target layer list (its weights stay unused here).
// Anything the engine prints on stdout is diverted to stderr so the frame stream stays clean.
#include "tqllm/options.hpp"
#include "tqllm/model.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <array>
#include <condition_variable>
#include <exception>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {
int frame_fd = -1;

void write_all(const void * p, size_t n) {
    const auto * b = static_cast<const char *>(p);
    while (n) {
        const ssize_t w = ::write(frame_fd, b, n);
        if (w <= 0) throw std::runtime_error("tqfeatures: output stream closed");
        b += w, n -= size_t(w);
    }
}

void write_frame(uint32_t P, uint32_t nf, uint32_t H, const std::vector<uint16_t> & feats) {
    const uint32_t hdr[4] = {0x31465154u /* "TQF1" */, P, nf, H};
    write_all(hdr, sizeof hdr);
    if (P) write_all(feats.data(), feats.size() * sizeof(uint16_t));
}

// The producer and writer own separate buffers, so large frames can be drained
// to the pipe while the GPU computes the next request. At most two frames may
// be outstanding: the third waits for the first write to finish.
class FrameWriter {
public:
    struct Frame {
        uint32_t p = 0, nf = 0, hidden = 0;
        std::vector<uint16_t> data;
    };

    FrameWriter() : writer_([this] { run(); }) {}
    FrameWriter(const FrameWriter &) = delete;
    FrameWriter & operator=(const FrameWriter &) = delete;

    Frame & acquire() {
        std::unique_lock lock(mu_);
        cv_.wait(lock, [&] { return queued_ - sent_ < frames_.size() || error_; });
        if (error_) std::rethrow_exception(error_);
        return frames_[queued_ % frames_.size()];
    }

    void submit() {
        {
            std::lock_guard lock(mu_);
            if (error_) std::rethrow_exception(error_);
            ++queued_;
        }
        cv_.notify_all();
    }

    void finish() {
        {
            std::lock_guard lock(mu_);
            closing_ = true;
        }
        cv_.notify_all();
        if (writer_.joinable()) writer_.join();
        if (error_) std::rethrow_exception(error_);
    }

    ~FrameWriter() {
        if (writer_.joinable()) {
            {
                std::lock_guard lock(mu_);
                closing_ = true;
            }
            cv_.notify_all();
            writer_.join();
        }
    }

private:
    void run() {
        while (true) {
            size_t index;
            {
                std::unique_lock lock(mu_);
                cv_.wait(lock, [&] { return sent_ < queued_ || closing_; });
                if (sent_ == queued_) return;
                index = sent_ % frames_.size();
            }
            try {
                const Frame & frame = frames_[index];
                write_frame(frame.p, frame.nf, frame.hidden, frame.data);
            } catch (...) {
                {
                    std::lock_guard lock(mu_);
                    error_ = std::current_exception();
                    closing_ = true;
                }
                cv_.notify_all();
                return;
            }
            {
                std::lock_guard lock(mu_);
                ++sent_;
            }
            cv_.notify_all();
        }
    }

    std::array<Frame, 2> frames_;
    std::mutex mu_;
    std::condition_variable cv_;
    size_t queued_ = 0, sent_ = 0;
    bool closing_ = false;
    std::exception_ptr error_;
    std::thread writer_;
};
} // namespace

int main(int argc, char ** argv) {
    tqllm::check_env_flags("tqfeatures");
    // Keep the real stdout for frames only; route everything else printed to fd 1 to stderr.
    frame_fd = ::dup(STDOUT_FILENO);
    if (frame_fd < 0 || ::dup2(STDERR_FILENO, STDOUT_FILENO) < 0) {
        std::perror("tqfeatures: dup");
        return 1;
    }
    try {
        std::string model_file;
        uint32_t context = 4096;
        tqllm::EngineOptions engine;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto value = [&]() -> std::string {
                if (++i >= argc) throw std::invalid_argument("missing value for " + arg);
                return argv[i];
            };
            if (arg == "--model") model_file = value();
            else if (arg == "--context") context = std::stoul(value());
            else if (tqllm::parse_engine_option(argc, argv, i, engine)) continue;
            else throw std::invalid_argument("unknown argument " + arg);
        }
        if (model_file.empty() || engine.draft_model.empty())
            throw std::invalid_argument("need --model TARGET.gguf and -md DRAFTER.gguf");

        tqllm::Model model(model_file, model_file, context, 1, engine);
        if (!model.has_drafter()) throw std::runtime_error("drafter failed to load");
        const uint32_t H = model.hidden_size(), V = model.vocab_size(), ctx = model.context();
        std::cerr << "tqfeatures: ready (hidden " << H << ", vocab " << V << ", context " << ctx << ")\n";

        std::string line;
        std::vector<uint32_t> ids;
        FrameWriter frames;
        while (std::getline(std::cin, line)) {
            ids.clear();
            std::istringstream in(line);
            for (uint64_t v; in >> v;) ids.push_back(uint32_t(v));
            if (ids.empty()) continue;
            auto & frame = frames.acquire();
            try {
                for (auto t : ids)
                    if (t >= V) throw std::invalid_argument("token id " + std::to_string(t) + " >= vocab");
                if (ids.size() + 1 > ctx)
                    throw std::invalid_argument(std::to_string(ids.size()) + " tokens exceed context " + std::to_string(ctx));
                const int nf = model.drafter_features(ids, frame.data);
                frame.p = uint32_t(ids.size());
                frame.nf = uint32_t(nf);
                frame.hidden = H;
            } catch (const std::invalid_argument & e) {
                std::cerr << "tqfeatures: request rejected: " << e.what() << '\n';
                frame.p = frame.nf = frame.hidden = 0;
            }
            frames.submit();
        }
        frames.finish();
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "tqfeatures: " << e.what() << '\n';
        return 1;
    }
}
