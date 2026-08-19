#include "server.hpp"
#include "request_queue.hpp"
#include "scheduler.hpp"
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <atomic>
#include <memory>

using json = nlohmann::json;

namespace server {

void start(int port, scheduler::Config sched_cfg, int request_timeout_ms) {
    httplib::Server svr;
    RequestQueue queue;

    scheduler::start(queue, sched_cfg);

    svr.Post("/generate", [&](const httplib::Request & req, httplib::Response & res) {
        try {
            json body = json::parse(req.body);
            std::string prompt = body.at("prompt").get<std::string>();
            int n_predict = body.value("n_predict", 32);
            bool stream = body.value("stream", false);

            auto now = std::chrono::steady_clock::now();
            auto deadline = now + std::chrono::milliseconds(request_timeout_ms);

            if (stream) {
                auto token_channel = std::make_shared<TokenChannel>();
                auto cancelled = std::make_shared<std::atomic<bool>>(false);

                Request request;
                request.prompt = prompt;
                request.n_predict = n_predict;
                request.token_channel = token_channel;
                request.cancelled = cancelled;
                request.enqueue_time = now;
                request.deadline = deadline;

                queue.push(std::move(request));

                res.set_chunked_content_provider("application/x-ndjson",
                    [token_channel, cancelled](size_t, httplib::DataSink & sink) -> bool {
                        if (!sink.is_writable()) {
                            cancelled->store(true);
                            return false;
                        }

                        auto chunk = token_channel->wait_next(std::chrono::milliseconds(30000));
                        if (!chunk) {
                            cancelled->store(true);
                            return false;
                        }

                        if (chunk->is_final) {
                            json done_line = {
                                {"done", true},
                                {"finish_reason", chunk->finish_reason},
                                {"n_generated", chunk->n_generated}
                            };
                            std::string line = done_line.dump() + "\n";
                            bool ok = sink.write(line.data(), line.size());
                            sink.done();
                            return ok;
                        }

                        json token_line = {{"token", chunk->text}, {"done", false}};
                        std::string line = token_line.dump() + "\n";
                        return sink.write(line.data(), line.size());
                    });
            } else {
                std::promise<std::string> result_promise;
                std::future<std::string> result_future = result_promise.get_future();

                Request request;
                request.prompt = prompt;
                request.n_predict = n_predict;
                request.result_promise = std::move(result_promise);
                request.enqueue_time = now;
                request.deadline = deadline;

                queue.push(std::move(request));

                std::string result = result_future.get();

                json response = {{"response", result}};
                res.set_content(response.dump(), "application/json");
            }
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    printf("listening on http://localhost:%d\n", port);
    svr.listen("0.0.0.0", port);
}

}
