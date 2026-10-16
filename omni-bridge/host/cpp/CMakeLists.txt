cmake_minimum_required(VERSION 3.20)
project(OmniBridge CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Wasmtime precompiled library
set(WASMTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/wasmtime-v24.0.0-x86_64-linux-c-api")
include_directories(${WASMTIME_DIR}/include)
link_directories(${WASMTIME_DIR}/lib)

set(CMAKE_CXX_FLAGS_RELEASE "-O2 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_ASAN "-fsanitize=address,undefined -g -O1 -fno-omit-frame-pointer")

find_package(SQLite3 REQUIRED)
find_package(nlohmann_json 3.9 REQUIRED)

add_library(omni_bridge_core INTERFACE)
target_include_directories(omni_bridge_core INTERFACE include)

# Tests
enable_testing()
add_executable(omni_bridge_tests tests/test_core.cpp)
target_link_libraries(omni_bridge_tests PRIVATE omni_bridge_core SQLite::SQLite3 blake3 wasmtime nlohmann_json::nlohmann_json)
add_test(NAME CoreTests COMMAND omni_bridge_tests)

# Daemon
add_executable(omni_bridge_daemon src/omni_bridge_daemon.cpp)
target_link_libraries(omni_bridge_daemon PRIVATE omni_bridge_core SQLite::SQLite3 blake3 wasmtime nlohmann_json::nlohmann_json)

# RPATH so the daemon finds libwasmtime without LD_LIBRARY_PATH juggling
set_target_properties(omni_bridge_daemon omni_bridge_tests PROPERTIES
    BUILD_RPATH "${WASMTIME_DIR}/lib"
    INSTALL_RPATH "${WASMTIME_DIR}/lib")
