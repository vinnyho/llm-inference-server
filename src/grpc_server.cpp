#include "grpc_server.hpp"
#include "metrics.hpp"
#include "inference.grpc.pb.h"
#include <grpcpp/grpcpp.h>
#include <atomic>
#include <memory>
#include <thread>

namespace grpc_frontend {

namespace {

class InferenceService final : public llmserver::Inference::Service {
public:
    InferenceService(RequestQueue & queue, int request_timeout_ms)
        : queue_(queue), timeout_(request_timeout_ms) {}

    grpc::Status Generate(grpc::ServerContext * context,
                          const llmserver::GenerateRequest * in,
                          grpc::ServerWriter<llmserver::GenerateReply> * writer) override {
        int n_predict = in->n_predict() > 0 ? in->n_predict() : 32;

        auto token_channel = std::make_shared<TokenChannel>();
        auto cancelled = std::make_shared<std::atomic<bool>>(false);

        auto now = std::chrono::steady_clock::now();
        Request request;
        request.prompt = in->prompt();
        request.n_predict = n_predict;
        request.token_channel = token_channel;
        request.cancelled = cancelled;
        request.enqueue_time = now;
        request.deadline = now + timeout_;

        auto & m = metrics::get();
        m.requests_in_flight++;
        if (!queue_.try_push(request)) {
            m.requests_in_flight--;
            m.requests_rejected++;
            return grpc::Status(grpc::StatusCode::RESOURCE_EXHAUSTED, "request queue is full");
        }
        m.requests_accepted++;

        // Poll in short steps so a client that disconnects is noticed quickly
        // and its slot in the batch is released.
        auto give_up_at = request.deadline + std::chrono::milliseconds(5000);
        while (true) {
            auto chunk = token_channel->wait_next(std::chrono::milliseconds(100));
            if (!chunk) {
                if (context->IsCancelled()) {
                    cancelled->store(true);
                    return grpc::Status(grpc::StatusCode::CANCELLED, "client cancelled");
                }
                if (std::chrono::steady_clock::now() > give_up_at) {
                    cancelled->store(true);
                    return grpc::Status(grpc::StatusCode::DEADLINE_EXCEEDED, "no response from scheduler");
                }
                continue;
            }

            llmserver::GenerateReply reply;
            if (chunk->is_final) {
                reply.set_done(true);
                reply.set_finish_reason(chunk->finish_reason);
                reply.set_n_generated(chunk->n_generated);
                writer->Write(reply);
                if (chunk->finish_reason == "timeout") {
                    return grpc::Status(grpc::StatusCode::DEADLINE_EXCEEDED, "request timed out");
                }
                if (chunk->finish_reason == "error") {
                    return grpc::Status(grpc::StatusCode::INTERNAL, "inference failed");
                }
                return grpc::Status::OK;
            }

            reply.set_token(chunk->text);
            if (!writer->Write(reply)) {
                cancelled->store(true);
                return grpc::Status(grpc::StatusCode::CANCELLED, "client went away");
            }
        }
    }

private:
    RequestQueue & queue_;
    std::chrono::milliseconds timeout_;
};

}

void start(RequestQueue & queue, const ServerConfig & cfg) {
    std::string address = "0.0.0.0:" + std::to_string(cfg.grpc_port);
    int timeout_ms = cfg.request_timeout_ms;

    std::thread([&queue, address, timeout_ms] {
        InferenceService service(queue, timeout_ms);
        grpc::ServerBuilder builder;
        builder.AddListeningPort(address, grpc::InsecureServerCredentials());
        builder.RegisterService(&service);
        std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
        if (!server) {
            fprintf(stderr, "error: failed to start gRPC server on %s\n", address.c_str());
            return;
        }
        printf("grpc listening on %s\n", address.c_str());
        fflush(stdout);
        server->Wait();
    }).detach();
}

}
