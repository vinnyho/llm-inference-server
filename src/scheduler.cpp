#include "scheduler.hpp"
#include "inference.hpp"
#include <thread>
#include <vector>
#include <chrono>

static const int BATCH_SIZE = 4;
static const auto MAX_WAIT = std::chrono::milliseconds(10);

namespace scheduler {

static void worker_loop(RequestQueue & queue) {
    while (true) {
        std::vector<Request> batch;
        batch.push_back(queue.pop());

        auto deadline = std::chrono::steady_clock::now() + MAX_WAIT;

        while (batch.size() < BATCH_SIZE) {
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

        run_inference_batch(batch);
    }
}

void start(RequestQueue & queue) {
    std::thread worker(worker_loop, std::ref(queue));
    worker.detach();
}

}
