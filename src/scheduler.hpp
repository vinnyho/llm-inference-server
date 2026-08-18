#pragma once
#include "request_queue.hpp"
#include <chrono>

namespace scheduler {

struct Config {
    int batch_size = 4;
    std::chrono::milliseconds max_wait{10};
};

void start(RequestQueue & queue, Config cfg);

}
