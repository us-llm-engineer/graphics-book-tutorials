cmake_minimum_required(VERSION 3.24)
project(wavefront_study LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(WF_SANITIZE "Build host code with ASan+UBSan" OFF)
option(WF_BUILD_VIEWER "Build the Vulkan/GLFW viewer if available" ON)

# Identical float semantics on every arm: no fused multiply-add contraction anywhere.
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall>)
if(WF_SANITIZE)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address,undefined>
                      $<$<COMPILE_LANGUAGE:CXX>:-fno-omit-frame-pointer>)
  add_link_options(-fsanitize=address,undefined)
endif()

include_directories(${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src/gpu)

find_package(CUDAToolkit REQUIRED)

# ---- host library (CPU arm, scenes, BVH build, I/O, config, report) ----
file(GLOB WFCORE_SRC CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/*.cpp)
list(FILTER WFCORE_SRC EXCLUDE REGEX ".*/(main|bench)\\.cpp$")
add_library(wfcore STATIC ${WFCORE_SRC})
find_package(Threads REQUIRED)
target_link_libraries(wfcore PUBLIC Threads::Threads)

# ---- CUDA library (megakernel, wavefront, queues) ----
file(GLOB WFGPU_SRC CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/gpu/*.cu)
if(EXISTS ${CMAKE_SOURCE_DIR}/src/bench.cpp)
  list(APPEND WFGPU_SRC ${CMAKE_SOURCE_DIR}/src/bench.cpp)   # benchmark driver needs GpuRenderer
endif()
add_library(wfgpu STATIC ${WFGPU_SRC})
set_target_properties(wfgpu PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
target_link_libraries(wfgpu PUBLIC wfcore CUDA::cudart_static)

# ---- command line renderer ----
add_executable(wf src/main.cpp)
target_link_libraries(wf PRIVATE wfgpu wfcore)

# ---- tools ----
foreach(tool wfbench imgdiff svgplot ncusum wfstats svgstats wfgif)
  if(EXISTS ${CMAKE_SOURCE_DIR}/tools/${tool}.cpp)
    add_executable(${tool} tools/${tool}.cpp)
    target_link_libraries(${tool} PRIVATE wfgpu wfcore)
  endif()
endforeach()

# ---- viewer (optional) ----
if(WF_BUILD_VIEWER AND EXISTS ${CMAKE_SOURCE_DIR}/src/viewer/viewer.cpp)
  find_package(glfw3 QUIET)
  find_package(Vulkan QUIET)
  if(glfw3_FOUND AND Vulkan_FOUND)
    add_executable(wfviewer src/viewer/viewer.cpp)
    target_link_libraries(wfviewer PRIVATE wfgpu wfcore glfw Vulkan::Vulkan)
  else()
    message(STATUS "GLFW or Vulkan not found: wfviewer not built")
  endif()
endif()

# ---- tests: tests/test_*.cpp|cu use tests/test_framework.hpp; tests/catch/*.cpp|cu use Catch2 ----
enable_testing()
file(GLOB TEST_CPP CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/tests/test_*.cpp)
file(GLOB TEST_CU CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/tests/test_*.cu)
foreach(src ${TEST_CPP} ${TEST_CU})
  get_filename_component(name ${src} NAME_WE)
  add_executable(${name} ${src})
  target_link_libraries(${name} PRIVATE wfgpu wfcore)
  target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/tests)
  add_test(NAME ${name} COMMAND ${name} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endforeach()

# Catch2 3 from vcpkg: every tests/catch/*.cpp and tests/catch/*.cu is one executable.
list(APPEND CMAKE_PREFIX_PATH $ENV{HOME}/vcpkg/installed/x64-linux)
find_package(Catch2 3 CONFIG QUIET)
if(Catch2_FOUND)
  file(GLOB CATCH_SRC CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/tests/catch/*.cpp ${CMAKE_SOURCE_DIR}/tests/catch/*.cu)
  foreach(src ${CATCH_SRC})
    get_filename_component(name ${src} NAME_WE)
    add_executable(catch_${name} ${src})
    target_link_libraries(catch_${name} PRIVATE wfgpu wfcore Catch2::Catch2WithMain)
    target_include_directories(catch_${name} PRIVATE ${CMAKE_SOURCE_DIR}/tests)
    add_test(NAME catch_${name} COMMAND catch_${name} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
  endforeach()
else()
  message(STATUS "Catch2 3 not found: tests/catch suites not built")
endif()
