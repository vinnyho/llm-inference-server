#include "src/config.hpp"
#include "src/inference.hpp"
#include "src/request_queue.hpp"
#include "src/scheduler.hpp"
#include "src/server.hpp"
#ifdef LLM_SERVER_GRPC
#include "src/grpc_server.hpp"
#endif
#include <cstdio>
#include <cstdlib>
#include <string>
#include <chrono>

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
        else if (arg == "--grpc-port")          cfg.grpc_port          = std::stoi(next_value(i));
        else if (arg == "--http-threads")       cfg.http_threads       = std::stoi(next_value(i));
        else if (arg == "--batch-size")         cfg.batch_size         = std::stoi(next_value(i));
        else if (arg == "--batch-wait-ms")      cfg.batch_wait_ms      = std::stoi(next_value(i));
        else if (arg == "--max-queue-depth")    cfg.max_queue_depth    = std::stoi(next_value(i));
        else if (arg == "--request-timeout-ms") cfg.request_timeout_ms = std::stoi(next_value(i));
        else {
            fprintf(stderr, "error: unknown argument %s\n", arg.c_str());
            exit(1);
        }
    }

    if (cfg.batch_size < 1 || cfg.max_queue_depth < 1 || cfg.http_threads < 1) {
        fprintf(stderr, "error: --batch-size, --max-queue-depth and --http-threads must be >= 1\n");
        exit(1);
    }
    return cfg;
}

int main(int argc, char ** argv) {
    ServerConfig cfg = parse_args(argc, argv);

    printf("config: model=%s port=%d grpc_port=%d http_threads=%d batch_size=%d batch_wait_ms=%d "
           "max_queue_depth=%d request_timeout_ms=%d\n",
           cfg.model_path.c_str(), cfg.port, cfg.grpc_port, cfg.http_threads, cfg.batch_size,
           cfg.batch_wait_ms, cfg.max_queue_depth, cfg.request_timeout_ms);
    fflush(stdout);

    if (!init_model(cfg.model_path)) {
        fprintf(stderr, "error: unable to load model\n");
        return 1;
    }

    RequestQueue queue(cfg.max_queue_depth);

    scheduler::Config sched_cfg;
    sched_cfg.batch_size = cfg.batch_size;
    sched_cfg.max_wait = std::chrono::milliseconds(cfg.batch_wait_ms);
    scheduler::start(queue, sched_cfg);

#ifdef LLM_SERVER_GRPC
    grpc_frontend::start(queue, cfg);
#endif

    server::start(queue, cfg);
    return 0;
}
