cmake_minimum_required(VERSION 3.24)

# GPU architecture (e.g. 86 = sm_86, Ampere). Must be set BEFORE project(): project() fills in nvcc's
# default (sm_75) if it's unset, and that default would win.
# Override with -DCMAKE_CUDA_ARCHITECTURES=native if needed.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

project(fa_dev LANGUAGES CXX CUDA)

# ---- Options (toggled by presets or -D on the command line) --------------
option(WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(ENABLE_SANITIZERS "Build host code with ASan + UBSan" OFF)

# ---- Global settings -----------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # -std=c++20, not -std=gnu++20 (no GNU-only extensions)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # compile_commands.json for clangd / clang-tidy

find_package(CUDAToolkit REQUIRED) # provides CUDA::cudart, CUDA::cublas

# By default CMake hides nvcc's include paths in a response file (--options-file x.rsp).
# .clangd has to strip --options-file (clang doesn't know it), and the -I paths would go
# with it: clangd then can't find common.cuh or gtest.h. Put them on the command line.
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)

include(cmake/CompilerWarnings.cmake)
include(cmake/Sanitizers.cmake)

# ---- Targets -------------------------------------------------------------
# Flags shared by every target below.
function(fa_target_options target)
  target_include_directories(${target} PRIVATE csrc)
  # Same math flags as setup.py, so these builds run the same machine code as Python.
  target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
  # Non-Debug builds: -lineinfo maps SASS back to source lines in ncu /
  # compute-sanitizer. Costs nothing at runtime, so it's always on there.
  # (Debug builds use -G below, which already includes line info and conflicts
  # with -lineinfo.)
  target_compile_options(${target} PRIVATE $<$<AND:$<NOT:$<CONFIG:Debug>>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo>)
  # Debug builds: -G keeps device debug info and turns off device optimizations,
  # so cuda-gdb can set breakpoints in kernels and show local variables.
  # Kernels get MUCH slower with -G: never benchmark a Debug build.
  target_compile_options(${target} PRIVATE $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>)
  set_project_warnings(${target} ${WARNINGS_AS_ERRORS})
  enable_sanitizers(${target} ${ENABLE_SANITIZERS})
endfunction()

# The kernel, compiled once and linked by the driver and the tests.
# The Python extension is built separately by setup.py from the same source file.
add_library(fa_kernel STATIC csrc/kernels/flash_fwd.cu)
fa_target_options(fa_kernel)
target_link_libraries(fa_kernel PUBLIC CUDA::cudart)

# Standalone driver (no Python) for compute-sanitizer, ncu and the strict-warning build.
add_executable(fa_dev csrc/dev/main.cu)
fa_target_options(fa_dev)
target_link_libraries(fa_dev PRIVATE fa_kernel)

# ---- C++ tests (GoogleTest, run with ctest) --------------------------------
# GoogleTest is downloaded at configure time. SYSTEM: its headers are included with
# -isystem, so our strict warnings (-Werror) don't fire inside them.
option(FA_BUILD_TESTS "Build the GoogleTest suite in tests/cpp" ON)
if(FA_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SYSTEM)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  enable_testing()
  add_executable(fa_tests tests/cpp/test_ptx.cu tests/cpp/test_flash_fwd.cu)
  fa_target_options(fa_tests)
  target_link_libraries(fa_tests PRIVATE fa_kernel GTest::gtest_main)
  include(GoogleTest)
  # One ctest entry per TEST(...), discovered by running `fa_tests --gtest_list_tests`.
  gtest_discover_tests(fa_tests DISCOVERY_TIMEOUT 60)
endif()
