#include "metrics.hpp"
#include <cstdio>

namespace metrics {

namespace {

// Latency buckets from 5ms to 60s, roughly doubling.
std::vector<double> latency_buckets() {
    return {0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60};
}

std::string fmt(double v) {
    char buf[64];
    snprintf(buf, sizeof(buf), "%.6g", v);
    return buf;
}

void scalar(std::string & out, const char * type, const std::string & name,
            const std::string & help, const std::string & value) {
    out += "# HELP " + name + " " + help + "\n";
    out += "# TYPE " + name + " " + type + "\n";
    out += name + " " + value + "\n";
}

}

Histogram::Histogram(std::vector<double> upper_bounds)
    : bounds_(std::move(upper_bounds)), counts_(bounds_.size() + 1, 0) {}

void Histogram::observe(double value) {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t i = 0;
    while (i < bounds_.size() && value > bounds_[i]) {
        i++;
    }
    counts_[i]++;
    sum_ += value;
    count_++;
}

void Histogram::render(std::string & out, const std::string & name, const std::string & help) const {
    std::lock_guard<std::mutex> lock(mutex_);
    out += "# HELP " + name + " " + help + "\n";
    out += "# TYPE " + name + " histogram\n";
    uint64_t cumulative = 0;
    for (size_t i = 0; i < bounds_.size(); i++) {
        cumulative += counts_[i];
        out += name + "_bucket{le=\"" + fmt(bounds_[i]) + "\"} " + std::to_string(cumulative) + "\n";
    }
    cumulative += counts_.back();
    out += name + "_bucket{le=\"+Inf\"} " + std::to_string(cumulative) + "\n";
    out += name + "_sum " + fmt(sum_) + "\n";
    out += name + "_count " + std::to_string(count_) + "\n";
}

Registry::Registry()
    : request_latency_seconds(latency_buckets()),
      queue_wait_seconds(latency_buckets()),
      ttft_seconds(latency_buckets()),
      batch_size({1, 2, 4, 8, 16, 32}) {}

std::string Registry::render(size_t queue_depth, size_t queue_capacity) const {
    std::string out;

    scalar(out, "counter", "llm_requests_accepted_total",
           "Requests admitted to the queue", std::to_string(requests_accepted.load()));
    scalar(out, "counter", "llm_requests_rejected_total",
           "Requests rejected because the queue was full", std::to_string(requests_rejected.load()));
    scalar(out, "gauge", "llm_requests_in_flight",
           "Requests admitted but not yet completed", std::to_string(requests_in_flight.load()));
    scalar(out, "gauge", "llm_queue_depth",
           "Requests waiting in the queue", std::to_string(queue_depth));
    scalar(out, "gauge", "llm_queue_capacity",
           "Maximum queue depth before rejecting", std::to_string(queue_capacity));

    out += "# HELP llm_requests_completed_total Completed requests by finish reason\n";
    out += "# TYPE llm_requests_completed_total counter\n";
    out += "llm_requests_completed_total{reason=\"stop\"} " + std::to_string(finished_stop.load()) + "\n";
    out += "llm_requests_completed_total{reason=\"length\"} " + std::to_string(finished_length.load()) + "\n";
    out += "llm_requests_completed_total{reason=\"timeout\"} " + std::to_string(finished_timeout.load()) + "\n";
    out += "llm_requests_completed_total{reason=\"cancelled\"} " + std::to_string(finished_cancelled.load()) + "\n";
    out += "llm_requests_completed_total{reason=\"error\"} " + std::to_string(finished_error.load()) + "\n";

    scalar(out, "counter", "llm_tokens_generated_total",
           "Tokens generated across all requests", std::to_string(tokens_generated.load()));
    scalar(out, "counter", "llm_batches_total",
           "Inference batches run", std::to_string(batches_run.load()));

    request_latency_seconds.render(out, "llm_request_latency_seconds",
                                   "End-to-end latency of admitted requests (enqueue to completion)");
    queue_wait_seconds.render(out, "llm_queue_wait_seconds",
                              "Time a request waited in the queue before its batch started");
    ttft_seconds.render(out, "llm_time_to_first_token_seconds",
                        "Time from enqueue to the first generated token");
    batch_size.render(out, "llm_batch_size", "Number of requests per inference batch");

    return out;
}

Registry & get() {
    static Registry registry;
    return registry;
}

}
