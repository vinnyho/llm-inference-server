#include "src/inference.hpp"
#include "src/server.hpp"
#include <cstdio>

int main() {
    if (!init_model("models/qwen2.5-0.5b-instruct-q4_k_m.gguf")) {
        fprintf(stderr, "error: unable to load model\n");
        return 1;
    }
    server::start(8080);
    return 0;
}
