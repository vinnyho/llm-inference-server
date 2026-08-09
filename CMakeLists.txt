cmake_minimum_required(VERSION 3.16)

project(llm_server CXX)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(llama.cpp)

add_executable(main main.cpp)

target_link_libraries(main PRIVATE llama)
