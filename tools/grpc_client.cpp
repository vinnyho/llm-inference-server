// Minimal gRPC client for manual testing:
//   ./build/grpc_client "The capital of France is" 16 [localhost:50051]
#include "inference.grpc.pb.h"
#include <grpcpp/grpcpp.h>
#include <cstdio>
#include <string>

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <prompt> [n_predict] [host:port]\n", argv[0]);
        return 1;
    }
    std::string prompt = argv[1];
    int n_predict = argc > 2 ? std::stoi(argv[2]) : 32;
    std::string target = argc > 3 ? argv[3] : "localhost:50051";

    auto channel = grpc::CreateChannel(target, grpc::InsecureChannelCredentials());
    auto stub = llmserver::Inference::NewStub(channel);

    llmserver::GenerateRequest request;
    request.set_prompt(prompt);
    request.set_n_predict(n_predict);

    grpc::ClientContext context;
    auto reader = stub->Generate(&context, request);

    llmserver::GenerateReply reply;
    while (reader->Read(&reply)) {
        if (reply.done()) {
            printf("\n[done: %s, %d tokens]\n", reply.finish_reason().c_str(), reply.n_generated());
        } else {
            printf("%s", reply.token().c_str());
            fflush(stdout);
        }
    }

    grpc::Status status = reader->Finish();
    if (!status.ok()) {
        fprintf(stderr, "rpc failed: %s (%d)\n", status.error_message().c_str(), (int) status.error_code());
        return 1;
    }
    return 0;
}
