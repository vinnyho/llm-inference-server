#pragma once
#include <array>
#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <vector>

namespace metrics {

// Prometheus-style histogram: cumulative bucket counts plus sum and count.
// Percentiles are computed on the Prometheus side with histogram_quantile().
class Histogram {
public:
    explicit Histogram(std::vector<double> upper_bounds);
    void observe(double value);
    void render(std::string & out, const std::string & name, const std::string & help) const;

private:
    std::vector<double> bounds_;
    std::vector<uint64_t> counts_;   // one per bound, plus +Inf at the end
    double sum_ = 0.0;
    uint64_t count_ = 0;
    mutable std::mutex mutex_;
};

struct Registry {
    Registry();

    std::atomic<uint64_t> requests_accepted{0};
    std::atomic<uint64_t> requests_rejected{0};
    std::atomic<int64_t>  requests_in_flight{0};

    // Completed requests, by finish reason.
    std::atomic<uint64_t> finished_stop{0};
    std::atomic<uint64_t> finished_length{0};
    std::atomic<uint64_t> finished_timeout{0};
    std::atomic<uint64_t> finished_cancelled{0};
    std::atomic<uint64_t> finished_error{0};

    std::atomic<uint64_t> tokens_generated{0};
    std::atomic<uint64_t> batches_run{0};

    Histogram request_latency_seconds;   // enqueue -> completion
    Histogram queue_wait_seconds;        // enqueue -> picked up by a batch
    Histogram ttft_seconds;              // enqueue -> first generated token
    Histogram batch_size;

    // Renders everything in Prometheus text exposition format.
    std::string render(size_t queue_depth, size_t queue_capacity) const;
};

Registry & get();

inline double seconds_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

}
