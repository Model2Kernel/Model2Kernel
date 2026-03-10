cmake_minimum_required(VERSION 3.13)
project(Model2Kernel)

find_package(LLVM CONFIG)

# includes headers from LLVM - update the include paths accordingly
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})
add_definitions(${LLVM_DEFINITIONS})
add_compile_definitions(PROJECT_ROOT="${CMAKE_SOURCE_DIR}")

add_library(ReplaceDeviceStubCallsPass SHARED src/ReplaceDeviceStubCallsPass.cpp)

include_directories(${CMAKE_SOURCE_DIR}/include)
add_executable(Model2Kernel src/main.cpp)

# Link the necessary LLVM libraries
llvm_map_components_to_libnames(llvm_libs
    support
    core
    irreader
)

target_link_libraries(Model2Kernel ${llvm_libs})
