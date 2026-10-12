#include "inference.hpp"
#include "llama.h"
#include "metrics.hpp"
#include <algorithm>

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

void complete_request(Request & req, std::string finish_reason, std::string text, int n_generated) {
    auto & m = metrics::get();
    m.request_latency_seconds.observe(metrics::seconds_since(req.enqueue_time));
    m.tokens_generated += n_generated;
    m.requests_in_flight--;
    if      (finish_reason == "stop")      m.finished_stop++;
    else if (finish_reason == "length")    m.finished_length++;
    else if (finish_reason == "timeout")   m.finished_timeout++;
    else if (finish_reason == "cancelled") m.finished_cancelled++;
    else                                   m.finished_error++;

    if (req.token_channel) {
        req.token_channel->close(finish_reason, n_generated);
    }
    req.result_promise.set_value(Completion{std::move(text), std::move(finish_reason), n_generated});
}

namespace {

struct Slot {
    int seq_id;
    std::vector<llama_token> prompt_tokens;
    int n_past = 0;
    int n_generated = 0;
    int n_predict;
    std::string result;
    bool finished = false;
    int logits_index = -1;
};

}

void run_inference_batch(std::vector<Request> & requests) {
    int n_seq = requests.size();

    std::vector<Slot> slots(n_seq);
    int max_tokens = 0;

    for (int i = 0; i < n_seq; i++) {
        const std::string & prompt = requests[i].prompt;
        int n_prompt = -llama_tokenize(g_vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);

        slots[i].seq_id = i;
        slots[i].prompt_tokens.resize(n_prompt);
        llama_tokenize(g_vocab, prompt.c_str(), prompt.size(), slots[i].prompt_tokens.data(), n_prompt, true, true);
        slots[i].n_predict = requests[i].n_predict;

        max_tokens = std::max(max_tokens, n_prompt + requests[i].n_predict);
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_seq_max = n_seq;
    ctx_params.n_ctx = n_seq * max_tokens;
    ctx_params.n_batch = std::min(ctx_params.n_ctx, 2048u);

    llama_context * ctx = llama_init_from_model(g_model, ctx_params);
    if (ctx == nullptr) {
        for (auto & req : requests) {
            complete_request(req, "error", "failed to create llama context", 0);
        }
        return;
    }

    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    int total_prompt_tokens = 0;
    for (auto & s : slots) total_prompt_tokens += s.prompt_tokens.size();

    llama_batch batch = llama_batch_init(total_prompt_tokens, 0, n_seq);
    batch.n_tokens = 0;

    for (auto & s : slots) {
        for (size_t j = 0; j < s.prompt_tokens.size(); j++) {
            int i = batch.n_tokens;
            batch.token[i]     = s.prompt_tokens[j];
            batch.pos[i]       = (llama_pos) j;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = s.seq_id;
            bool is_last       = (j == s.prompt_tokens.size() - 1);
            batch.logits[i]    = is_last;
            if (is_last) {
                s.logits_index = i;
            }
            batch.n_tokens++;
        }
        s.n_past = (int) s.prompt_tokens.size();
    }

    int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        for (auto & req : requests) {
            complete_request(req, "error", "prefill decode failed", 0);
        }
        llama_sampler_free(smpl);
        llama_free(ctx);
        return;
    }

    int n_active = n_seq;

    while (n_active > 0) {
        llama_batch step_batch = llama_batch_init(n_active, 0, n_seq);
        step_batch.n_tokens = 0;

        for (auto & s : slots) {
            if (s.finished) continue;

            Request & req = requests[s.seq_id];

            if (req.cancelled && req.cancelled->load()) {
                s.finished = true;
                n_active--;
                complete_request(req, "cancelled", s.result, s.n_generated);
                continue;
            }

            if (std::chrono::steady_clock::now() >= req.deadline) {
                s.finished = true;
                n_active--;
                complete_request(req, "timeout", s.result, s.n_generated);
                continue;
            }

            llama_token new_token = llama_sampler_sample(smpl, ctx, s.logits_index);

            if (llama_vocab_is_eog(g_vocab, new_token)) {
                s.finished = true;
                n_active--;
                complete_request(req, "stop", s.result, s.n_generated);
                continue;
            }

            char buf[128];
            int n = llama_token_to_piece(g_vocab, new_token, buf, sizeof(buf), 0, true);
            std::string piece(buf, n);
            s.result += piece;
            s.n_generated++;

            if (!req.first_token_seen) {
                req.first_token_seen = true;
                metrics::get().ttft_seconds.observe(metrics::seconds_since(req.enqueue_time));
            }

            if (req.token_channel) req.token_channel->push_token(piece);

            if (s.n_generated >= s.n_predict) {
                s.finished = true;
                n_active--;
                complete_request(req, "length", s.result, s.n_generated);
                continue;
            }

            int i = step_batch.n_tokens;
            step_batch.token[i]     = new_token;
            step_batch.pos[i]       = s.n_past;
            step_batch.n_seq_id[i]  = 1;
            step_batch.seq_id[i][0] = s.seq_id;
            step_batch.logits[i]    = true;
            s.logits_index = i;

            s.n_past++;
            step_batch.n_tokens++;
        }

        int step_rc = 0;
        if (step_batch.n_tokens > 0) {
            step_rc = llama_decode(ctx, step_batch);
        }
        llama_batch_free(step_batch);

        if (step_rc != 0) {
            for (auto & s : slots) {
                if (s.finished) continue;
                s.finished = true;
                n_active--;
                complete_request(requests[s.seq_id], "error", s.result, s.n_generated);
            }
        }
    }

    llama_sampler_free(smpl);
    llama_free(ctx);
}
