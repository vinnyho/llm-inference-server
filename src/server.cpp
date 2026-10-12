#include "server.hpp"
#include "metrics.hpp"
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <atomic>
#include <memory>

using json = nlohmann::json;

namespace server {

namespace {

// Counts the request as in flight and tries to enqueue it. On rejection the
// caller still owns `req` and must answer the client itself.
bool admit(RequestQueue & queue, Request & req) {
    auto & m = metrics::get();
    m.requests_in_flight++;
    if (!queue.try_push(req)) {
        m.requests_in_flight--;
        m.requests_rejected++;
        return false;
    }
    m.requests_accepted++;
    return true;
}

void reject_overloaded(httplib::Response & res) {
    res.status = 503;
    res.set_header("Retry-After", "1");
    res.set_content(json{{"error", "server overloaded: request queue is full"}}.dump(), "application/json");
}

}

void start(RequestQueue & queue, const ServerConfig & cfg) {
    httplib::Server svr;

    // Each non-streaming request holds an HTTP worker thread while it waits for
    // its result, so the pool must be larger than the expected client count;
    // otherwise connections queue inside httplib and never reach our bounded
    // queue, which is where overload is supposed to be detected.
    int http_threads = cfg.http_threads;
    svr.new_task_queue = [http_threads] { return new httplib::ThreadPool(http_threads); };

    int request_timeout_ms = cfg.request_timeout_ms;

    svr.Post("/generate", [&queue, request_timeout_ms](const httplib::Request & req, httplib::Response & res) {
        json body;
        std::string prompt;
        int n_predict;
        bool stream;
        try {
            body = json::parse(req.body);
            prompt = body.at("prompt").get<std::string>();
            n_predict = body.value("n_predict", 32);
            stream = body.value("stream", false);
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(json{{"error", e.what()}}.dump(), "application/json");
            return;
        }
        if (n_predict <= 0) {
            res.status = 400;
            res.set_content(json{{"error", "n_predict must be positive"}}.dump(), "application/json");
            return;
        }

        auto now = std::chrono::steady_clock::now();
        auto timeout = std::chrono::milliseconds(request_timeout_ms);

        Request request;
        request.prompt = prompt;
        request.n_predict = n_predict;
        request.enqueue_time = now;
        request.deadline = now + timeout;

        if (stream) {
            auto token_channel = std::make_shared<TokenChannel>();
            auto cancelled = std::make_shared<std::atomic<bool>>(false);
            request.token_channel = token_channel;
            request.cancelled = cancelled;

            if (!admit(queue, request)) {
                reject_overloaded(res);
                return;
            }

            // Wait a little past the request deadline: the scheduler closes the
            // stream with finish_reason "timeout" once the deadline passes.
            auto max_wait = timeout + std::chrono::milliseconds(5000);

            res.set_chunked_content_provider("application/x-ndjson",
                [token_channel, cancelled, max_wait](size_t, httplib::DataSink & sink) -> bool {
                    if (!sink.is_writable()) {
                        cancelled->store(true);
                        return false;
                    }

                    auto chunk = token_channel->wait_next(max_wait);
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
                    if (!sink.write(line.data(), line.size())) {
                        cancelled->store(true);
                        return false;
                    }
                    return true;
                });
            return;
        }

        std::future<Completion> result_future = request.result_promise.get_future();
        if (!admit(queue, request)) {
            reject_overloaded(res);
            return;
        }

        Completion result = result_future.get();
        if (result.finish_reason == "timeout") {
            res.status = 504;
        } else if (result.finish_reason == "error") {
            res.status = 500;
        }
        json response = {
            {"response", result.text},
            {"finish_reason", result.finish_reason},
            {"n_generated", result.n_generated}
        };
        res.set_content(response.dump(), "application/json");
    });

    svr.Get("/metrics", [&queue](const httplib::Request &, httplib::Response & res) {
        res.set_content(metrics::get().render(queue.size(), queue.capacity()),
                        "text/plain; version=0.0.4");
    });

    svr.Get("/healthz", [](const httplib::Request &, httplib::Response & res) {
        res.set_content("ok\n", "text/plain");
    });

    printf("http listening on http://localhost:%d\n", cfg.port);
    fflush(stdout);
    svr.listen("0.0.0.0", cfg.port);
}

}
