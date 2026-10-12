#include "scheduler.hpp"
#include "inference.hpp"
#include "metrics.hpp"
#include <thread>
#include <vector>

namespace scheduler {

static void worker_loop(RequestQueue & queue, Config cfg) {
    while (true) {
        std::vector<Request> batch;
        batch.push_back(queue.pop());

        auto deadline = std::chrono::steady_clock::now() + cfg.max_wait;

        while ((int) batch.size() < cfg.batch_size) {
            auto remaining = deadline - std::chrono::steady_clock::now();
            if (remaining <= std::chrono::milliseconds(0)) {
                break;
            }

            auto item = queue.try_pop_for(std::chrono::duration_cast<std::chrono::milliseconds>(remaining));
            if (!item) {
                break;
            }
            batch.push_back(std::move(*item));
        }

        // Requests that expired or were abandoned while queued never reach the
        // model, so they don't take a slot away from requests that can still
        // finish in time.
        auto now = std::chrono::steady_clock::now();
        std::vector<Request> runnable;
        runnable.reserve(batch.size());
        for (auto & req : batch) {
            if (now >= req.deadline) {
                complete_request(req, "timeout", "", 0);
            } else if (req.cancelled && req.cancelled->load()) {
                complete_request(req, "cancelled", "", 0);
            } else {
                metrics::get().queue_wait_seconds.observe(metrics::seconds_since(req.enqueue_time));
                runnable.push_back(std::move(req));
            }
        }

        if (runnable.empty()) {
            continue;
        }

        metrics::get().batches_run++;
        metrics::get().batch_size.observe((double) runnable.size());
        run_inference_batch(runnable);
    }
}

void start(RequestQueue & queue, Config cfg) {
    std::thread worker(worker_loop, std::ref(queue), cfg);
    worker.detach();
}

}
