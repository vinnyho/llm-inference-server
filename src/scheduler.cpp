#include "scheduler.hpp"
#include "inference.hpp"
#include <thread>

static const int BATCH_SIZE = 4;

namespace scheduler {

static void worker_loop(RequestQueue & queue) {
    while (true) {
        std::vector<Request> batch;
        for (int i = 0; i < BATCH_SIZE; i++) {
            batch.push_back((queue.pop()));
                
        }
        run_inference_batch(batch);
    }
}

void start(RequestQueue & queue) {
    std::thread worker(worker_loop, std::ref(queue));
    worker.detach();
}

}
