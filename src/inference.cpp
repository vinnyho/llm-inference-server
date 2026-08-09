#include "inference.hpp"
#include "llama.h"
#include <vector>

static llama_model * g_model = nullptr;
static const llama_vocab * g_vocab = nullptr;

bool init_model(const std::string & model_path) {
    ggml_backend_load_all();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 99;

    g_model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (g_model == nullptr) {
        return false;
    }

    g_vocab = llama_model_get_vocab(g_model);
    return true;
}

std::string run_inference(const std::string & prompt, int n_predict) {
    int n_prompt = -llama_tokenize(g_vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);
    std::vector<llama_token> prompt_tokens(n_prompt);
    llama_tokenize(g_vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_prompt + n_predict;
    ctx_params.n_batch = n_prompt;
    llama_context * ctx = llama_init_from_model(g_model, ctx_params);

    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    std::string result = prompt;
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + n_predict; ) {
        if (llama_decode(ctx, batch)) {
            break;
        }
        n_pos += batch.n_tokens;

        llama_token new_token_id = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(g_vocab, new_token_id)) {
            break;
        }

        char buf[128];
        int n = llama_token_to_piece(g_vocab, new_token_id, buf, sizeof(buf), 0, true);
        result += std::string(buf, n);

        batch = llama_batch_get_one(&new_token_id, 1);
    }

    llama_sampler_free(smpl);
    llama_free(ctx);
    return result;
}
