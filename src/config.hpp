#pragma once
#include <string>

struct ServerConfig {
    std::string model_path = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    int port = 8080;
    int grpc_port = 50051;
    int http_threads = 64;
    int batch_size = 4;
    int batch_wait_ms = 10;
    int max_queue_depth = 64;
    int request_timeout_ms = 30000;
};
