#include "server.hpp"
#include "request_queue.hpp"
#include "scheduler.hpp"
#include <httplib.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace server {

void start(int port, scheduler::Config sched_cfg) {
    httplib::Server svr;
    RequestQueue queue;

    scheduler::start(queue, sched_cfg);

    svr.Post("/generate", [&](const httplib::Request & req, httplib::Response & res) {
        try {
            json body = json::parse(req.body);
            std::string prompt = body.at("prompt").get<std::string>();
            int n_predict = body.value("n_predict", 32);

            std::promise<std::string> result_promise;
            std::future<std::string> result_future = result_promise.get_future();

            queue.push(Request{prompt, n_predict, std::move(result_promise)});

            std::string result = result_future.get();

            json response = {{"response", result}};
            res.set_content(response.dump(), "application/json");
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    printf("listening on http://localhost:%d\n", port);
    svr.listen("0.0.0.0", port);
}

}
