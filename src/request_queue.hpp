#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>
#include <string>
#include <future>

struct Request {
    std::string prompt;
    int n_predict;
    std::promise<std::string> result_promise;
};

class RequestQueue {
public:
    void push(Request req) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(std::move(req));
        cv_.notify_one();
    }

    Request pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return !queue_.empty(); });

        Request req = std::move(queue_.front());
        queue_.pop();
        return req;
    }

private:
    std::queue<Request> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
};