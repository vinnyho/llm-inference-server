cmake_minimum_required(VERSION 3.16)

include(FetchContent)

FetchContent_Declare(
    httplib
  GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
  GIT_TAG v0.15.3
)
FetchContent_MakeAvailable(httplib)

FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(json)

project(llm_server CXX)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(llama.cpp)

add_executable(main main.cpp src/inference.cpp src/server.cpp)

target_link_libraries(main PRIVATE llama httplib nlohmann_json::nlohmann_json)
