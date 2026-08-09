#include "server.hpp"
#include "inference.hpp"
#include <httplib.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace server {

void start(int port) {
    httplib::Server svr;

    svr.Post("/generate", [](const httplib::Request & req, httplib::Response & res) {
        try {
            json body = json::parse(req.body);
            std::string prompt = body.at("prompt").get<std::string>();
            int n_predict = body.value("n_predict", 32);

            std::string result = run_inference(prompt, n_predict);

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