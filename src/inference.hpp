#pragma once
#include <string>

bool init_model(const std::string & model_path);
std::string run_inference(const std::string & prompt, int n_predict);