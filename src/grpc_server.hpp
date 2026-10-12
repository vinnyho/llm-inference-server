#pragma once
#include "config.hpp"
#include "request_queue.hpp"

namespace grpc_frontend {
    // Starts the gRPC API on cfg.grpc_port in a background thread.
    void start(RequestQueue & queue, const ServerConfig & cfg);
}
