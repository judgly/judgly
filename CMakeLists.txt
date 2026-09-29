cmake_minimum_required(VERSION 3.20)

# Workaround for one local toolchain: with Xcode 26.1 on macOS 27 the linker cannot link against
# the macOS 27 SDK. When the 26.5 Command Line Tools SDK is present and no sysroot was given, use
# it. This must run before project(). Remove once the toolchain links against the default SDK.
set(JUDGLY_FALLBACK_SDK /Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk)
if(APPLE AND NOT CMAKE_OSX_SYSROOT AND NOT DEFINED ENV{SDKROOT} AND EXISTS ${JUDGLY_FALLBACK_SDK})
  set(CMAKE_OSX_SYSROOT ${JUDGLY_FALLBACK_SDK} CACHE PATH "macOS SDK (judgly toolchain workaround)")
endif()

# Build for macOS 14 and later unless a target is given (MACOSX_DEPLOYMENT_TARGET or
# -DCMAKE_OSX_DEPLOYMENT_TARGET), so the library matches the wheel tag macosx_14_0 (pyproject.toml
# sets the variable for uv builds) instead of the build machine's version. Must precede project().
if(APPLE AND NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET AND NOT DEFINED ENV{MACOSX_DEPLOYMENT_TARGET})
  set(CMAKE_OSX_DEPLOYMENT_TARGET 14.0 CACHE STRING "minimum macOS version")
endif()

project(judgly C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)          # static llama/ggml go into the shared libjudgly
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ---- versions ---------------------------------------------------------------------------------

# The judgly version has one source: pyproject.toml. scikit-build-core passes it in; a plain
# CMake build (the CLI tools) reads it from the file.
if(DEFINED SKBUILD_PROJECT_VERSION_FULL)
  set(JUDGLY_VERSION ${SKBUILD_PROJECT_VERSION_FULL})
else()
  file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml JUDGLY_VERSION_LINE REGEX "^version = ")
  string(REGEX REPLACE "^version = \"([^\"]+)\".*" "\\1" JUDGLY_VERSION "${JUDGLY_VERSION_LINE}")
endif()

# The llama.cpp commit judgly is built and tested against. It is
# written here so an sdist, which has no git metadata, still reports it. When the submodule is
# a git checkout, a different commit is an error.
set(JUDGLY_LLAMA_COMMIT 6f41ac59e0a49a00483a316a22ada6b04edd2950)
set(LLAMA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp)
if(NOT EXISTS ${LLAMA_DIR}/CMakeLists.txt)
  message(FATAL_ERROR "third_party/llama.cpp is empty: run git submodule update --init")
endif()
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS ${LLAMA_DIR}/.git)
  execute_process(COMMAND ${GIT_EXECUTABLE} -C ${LLAMA_DIR} rev-parse HEAD
                  OUTPUT_VARIABLE LLAMA_HEAD OUTPUT_STRIP_TRAILING_WHITESPACE
                  RESULT_VARIABLE LLAMA_HEAD_RC ERROR_QUIET)
  if(LLAMA_HEAD_RC EQUAL 0 AND NOT LLAMA_HEAD STREQUAL JUDGLY_LLAMA_COMMIT)
    message(FATAL_ERROR "third_party/llama.cpp is at ${LLAMA_HEAD}, judgly pins ${JUDGLY_LLAMA_COMMIT}")
  endif()
endif()
set(LLAMA_BUILD_COMMIT ${JUDGLY_LLAMA_COMMIT})

# ---- third-party ------------------------------------------------------------------------------

# Source and build paths (__FILE__ in assertions, debug info) are written relative, so a wheel
# does not carry the directory it was built in: the checkout, or the temporary directory an
# sdist is unpacked into. Set before add_subdirectory so llama.cpp and ggml get it too.
include(CheckCCompilerFlag)
check_c_compiler_flag("-ffile-prefix-map=/a=b" JUDGLY_HAS_FILE_PREFIX_MAP)
if(JUDGLY_HAS_FILE_PREFIX_MAP)
  add_compile_options(-ffile-prefix-map=${CMAKE_BINARY_DIR}=build
                      -ffile-prefix-map=${CMAKE_SOURCE_DIR}=.)
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)    # llama and ggml are linked into libjudgly
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)   # common/ (and its vendored json.hpp) unused
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)          # a wheel must run on other machines
add_subdirectory(third_party/llama.cpp EXCLUDE_FROM_ALL)   # Metal on by default on macOS, embedded

add_library(yyjson STATIC third_party/yyjson/yyjson.c)
target_include_directories(yyjson SYSTEM PUBLIC third_party/yyjson)

# ---- judgly -----------------------------------------------------------------------------------

# Warnings as errors for judgly's own sources; third-party code is exempt.
set(JUDGLY_WARNINGS -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Werror)
set(JUDGLY_SANITIZE "" CACHE STRING "sanitizers for judgly sources: undefined, or address,undefined")
if(JUDGLY_SANITIZE)
  list(APPEND JUDGLY_WARNINGS -fsanitize=${JUDGLY_SANITIZE} -fno-sanitize-recover=all -fno-omit-frame-pointer)
  add_link_options(-fsanitize=${JUDGLY_SANITIZE})
endif()

# The engine (the s1_ sources).
add_library(s1 STATIC csrc/s1_engine.c csrc/s1_prompt.c csrc/s1_rows.c csrc/s1_decide.c
                      csrc/s1_request.c csrc/s1_sha256.c csrc/s1_lbfgs.c csrc/s1_head.c
                      csrc/s1_feat.c csrc/s1_args.c csrc/s1_headfile.c csrc/s1_metrics.c
                      csrc/s1_temperature.c)
target_include_directories(s1 PUBLIC csrc)
target_compile_options(s1 PRIVATE ${JUDGLY_WARNINGS})
target_compile_definitions(s1 PRIVATE _POSIX_C_SOURCE=200809L)   # pread, strdup under strict -std=c11
target_link_libraries(s1 PUBLIC llama ggml yyjson m)
if(APPLE)
  target_link_libraries(s1 PUBLIC "-framework Accelerate")
endif()

# libjudgly: the C API the Python package loads through ctypes.
add_library(judgly SHARED csrc/judgly.c)
target_include_directories(judgly PUBLIC csrc)
target_compile_options(judgly PRIVATE ${JUDGLY_WARNINGS})
target_compile_definitions(judgly PRIVATE JUDGLY_VERSION="${JUDGLY_VERSION}"
                                          JUDGLY_LLAMA_COMMIT="${JUDGLY_LLAMA_COMMIT}")
target_link_libraries(judgly PRIVATE s1)
set_target_properties(judgly PROPERTIES LINKER_LANGUAGE CXX C_VISIBILITY_PRESET hidden)

# CLI tools (s1-*). Built by default in a plain CMake build, skipped in a wheel build.
if(SKBUILD)
  set(JUDGLY_TOOLS_DEFAULT OFF)
else()
  set(JUDGLY_TOOLS_DEFAULT ON)
endif()
option(JUDGLY_BUILD_TOOLS "build the s1-* command-line tools" ${JUDGLY_TOOLS_DEFAULT})
if(JUDGLY_BUILD_TOOLS)
  foreach(t s1-selftest s1-run s1-features s1-train s1-eval s1-probe s1-compare)
    add_executable(${t} tools/${t}.c)
    target_compile_options(${t} PRIVATE ${JUDGLY_WARNINGS})
    target_link_libraries(${t} PRIVATE s1)
    set_target_properties(${t} PROPERTIES LINKER_LANGUAGE CXX)   # libllama is C++ inside
  endforeach()
endif()

if(SKBUILD)
  install(TARGETS judgly LIBRARY DESTINATION ${SKBUILD_PLATLIB_DIR}/judgly
                         RUNTIME DESTINATION ${SKBUILD_PLATLIB_DIR}/judgly)
endif()
