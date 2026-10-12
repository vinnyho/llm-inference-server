#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>
#include <string>
#include <future>
#include <chrono>
#include <optional>
#include <atomic>
#include <memory>

struct TokenChunk {
    std::string text;
    bool is_final = false;
    std::string finish_reason;   // "stop" | "length" | "timeout" | "cancelled" | "error"
    int n_generated = 0;
};

class TokenChannel {
public:
    void push_token(std::string piece) {
        std::lock_guard<std::mutex> lock(mutex_);
        TokenChunk chunk;
        chunk.text = std::move(piece);
        chunks_.push(std::move(chunk));
        cv_.notify_one();
    }

    void close(std::string finish_reason, int n_generated) {
        std::lock_guard<std::mutex> lock(mutex_);
        TokenChunk chunk;
        chunk.is_final = true;
        chunk.finish_reason = std::move(finish_reason);
        chunk.n_generated = n_generated;
        chunks_.push(std::move(chunk));
        cv_.notify_one();
    }

    std::optional<TokenChunk> wait_next(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!cv_.wait_for(lock, timeout, [this] { return !chunks_.empty(); })) {
            return std::nullopt;
        }
        TokenChunk chunk = std::move(chunks_.front());
        chunks_.pop();
        return chunk;
    }

private:
    std::queue<TokenChunk> chunks_;
    std::mutex mutex_;
    std::condition_variable cv_;
};

// Final result of a request, handed back to whichever frontend is waiting on it.
struct Completion {
    std::string text;
    std::string finish_reason;
    int n_generated = 0;
};

struct Request {
    std::string prompt;
    int n_predict;
    std::promise<Completion> result_promise;
    std::shared_ptr<TokenChannel> token_channel;      // null unless streaming
    std::shared_ptr<std::atomic<bool>> cancelled;     // null unless streaming
    std::chrono::steady_clock::time_point enqueue_time;
    std::chrono::steady_clock::time_point deadline;
    bool first_token_seen = false;
};

// Resolves a request exactly once: fulfils the promise, closes the token
// stream if there is one, and records completion metrics.
void complete_request(Request & req, std::string finish_reason, std::string text, int n_generated);

// Bounded FIFO shared by the HTTP/gRPC frontends (producers) and the
// scheduler thread (consumer). When it is full, try_push fails immediately so
// the frontend can reject the request instead of letting latency grow without
// bound.
class RequestQueue {
public:
    explicit RequestQueue(size_t max_depth) : max_depth_(max_depth) {}

    // Moves `req` into the queue and returns true, or leaves `req` untouched
    // and returns false if the queue is at capacity.
    bool try_push(Request & req) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.size() >= max_depth_) {
            return false;
        }
        queue_.push(std::move(req));
        cv_.notify_one();
        return true;
    }

    Request pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return !queue_.empty(); });

        Request req = std::move(queue_.front());
        queue_.pop();
        return req;
    }

    std::optional<Request> try_pop_for(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!cv_.wait_for(lock, timeout, [this] { return !queue_.empty(); })) {
            return std::nullopt;
        }

        Request req = std::move(queue_.front());
        queue_.pop();
        return req;
    }

    size_t size() {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    size_t capacity() const { return max_depth_; }

private:
    std::queue<Request> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    size_t max_depth_;
};
