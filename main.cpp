#include "src/inference.hpp"
#include "src/server.hpp"
#include "src/scheduler.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <chrono>

struct ServerConfig {
    std::string model_path = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    int port = 8080;
    int batch_size = 4;
    int batch_wait_ms = 10;
    int max_queue_depth = 256;
    int request_timeout_ms = 30000;
};

static ServerConfig parse_args(int argc, char ** argv) {
    ServerConfig cfg;

    auto next_value = [&](int & i) -> std::string {
        if (i + 1 >= argc) {
            fprintf(stderr, "error: missing value for %s\n", argv[i]);
            exit(1);
        }
        return argv[++i];
    };

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if      (arg == "--model")              cfg.model_path         = next_value(i);
        else if (arg == "--port")               cfg.port               = std::stoi(next_value(i));
        else if (arg == "--batch-size")         cfg.batch_size         = std::stoi(next_value(i));
        else if (arg == "--batch-wait-ms")      cfg.batch_wait_ms      = std::stoi(next_value(i));
        else if (arg == "--max-queue-depth")    cfg.max_queue_depth    = std::stoi(next_value(i));
        else if (arg == "--request-timeout-ms") cfg.request_timeout_ms = std::stoi(next_value(i));
        else {
            fprintf(stderr, "error: unknown argument %s\n", arg.c_str());
            exit(1);
        }
    }
    return cfg;
}

int main(int argc, char ** argv) {
    ServerConfig cfg = parse_args(argc, argv);

    printf("config: model=%s port=%d batch_size=%d batch_wait_ms=%d max_queue_depth=%d request_timeout_ms=%d\n",
           cfg.model_path.c_str(), cfg.port, cfg.batch_size, cfg.batch_wait_ms,
           cfg.max_queue_depth, cfg.request_timeout_ms);
    fflush(stdout);

    if (!init_model(cfg.model_path)) {
        fprintf(stderr, "error: unable to load model\n");
        return 1;
    }

    scheduler::Config sched_cfg;
    sched_cfg.batch_size = cfg.batch_size;
    sched_cfg.max_wait = std::chrono::milliseconds(cfg.batch_wait_ms);

    server::start(cfg.port, sched_cfg);
    return 0;
}
