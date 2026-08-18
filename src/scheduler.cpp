#include "scheduler.hpp"
#include "inference.hpp"
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

        // TODO (section 3): drop any request in `batch` whose deadline has
        // already passed, before calling run_inference_batch — needs
        // Request::deadline, added in section 2.
        run_inference_batch(batch);
    }
}

void start(RequestQueue & queue, Config cfg) {
    std::thread worker(worker_loop, std::ref(queue), cfg);
    worker.detach();
}

}
