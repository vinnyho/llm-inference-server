#pragma once
#include "config.hpp"
#include "request_queue.hpp"

namespace server {
    // Serves the HTTP API on cfg.port. Blocks until the server stops.
    void start(RequestQueue & queue, const ServerConfig & cfg);
}
