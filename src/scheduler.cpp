#include "scheduler.hpp"
#include "inference.hpp"
#include <thread>

namespace scheduler {

static void worker_loop(RequestQueue & queue) {
    while (true) {
        Request req = queue.pop();
        std::string result = run_inference(req.prompt, req.n_predict);
        req.result_promise.set_value(result);
    }
}

void start(RequestQueue & queue) {
    std::thread worker(worker_loop, std::ref(queue));
    worker.detach();
}

}
