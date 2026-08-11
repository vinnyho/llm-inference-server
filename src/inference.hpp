#pragma once
#include <string>
#include <vector>
#include "request_queue.hpp"

bool init_model(const std::string & model_path);
void run_inference_batch(std::vector<Request> & requests);
