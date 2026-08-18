#pragma once
#include "scheduler.hpp"

namespace server {
    void start(int port, scheduler::Config sched_cfg);
}
