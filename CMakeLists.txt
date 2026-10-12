cmake_minimum_required(VERSION 3.16)

project(llm_server CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_GRPC "Build the gRPC frontend (needs gRPC + protobuf installed)" ON)

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

# Backend is chosen when configuring llama.cpp, e.g. -DGGML_CUDA=ON for NVIDIA
# GPUs. Metal is enabled by default on macOS.
add_subdirectory(llama.cpp)

add_executable(main
    main.cpp
    src/inference.cpp
    src/server.cpp
    src/scheduler.cpp
    src/metrics.cpp
)

target_link_libraries(main PRIVATE llama httplib nlohmann_json::nlohmann_json)

if(ENABLE_GRPC)
    find_package(Protobuf CONFIG REQUIRED)
    find_package(gRPC CONFIG REQUIRED)

    set(PROTO_FILE ${CMAKE_CURRENT_SOURCE_DIR}/proto/inference.proto)
    set(GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
    file(MAKE_DIRECTORY ${GEN_DIR})
    set(GEN_SRCS
        ${GEN_DIR}/inference.pb.cc
        ${GEN_DIR}/inference.grpc.pb.cc
    )

    add_custom_command(
        OUTPUT ${GEN_SRCS} ${GEN_DIR}/inference.pb.h ${GEN_DIR}/inference.grpc.pb.h
        COMMAND $<TARGET_FILE:protobuf::protoc>
            --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/proto
            --cpp_out=${GEN_DIR}
            --grpc_out=${GEN_DIR}
            --plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>
            ${PROTO_FILE}
        DEPENDS ${PROTO_FILE}
    )

    add_library(inference_proto STATIC ${GEN_SRCS})
    target_include_directories(inference_proto PUBLIC ${GEN_DIR})
    target_link_libraries(inference_proto PUBLIC gRPC::grpc++ protobuf::libprotobuf)

    target_sources(main PRIVATE src/grpc_server.cpp)
    target_compile_definitions(main PRIVATE LLM_SERVER_GRPC=1)
    target_link_libraries(main PRIVATE inference_proto)

    add_executable(grpc_client tools/grpc_client.cpp)
    target_link_libraries(grpc_client PRIVATE inference_proto)
endif()
