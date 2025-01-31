cmake_minimum_required (VERSION 3.21)

project(TP_HPC_Convection LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/modules")

set(Kokkos_DIR "/home/aubin/Téléchargements/kokkos/build_sycl")

find_package(Kokkos REQUIRED)

add_executable(exe01 convection.cpp)
target_link_libraries(exe01 Kokkos::kokkos)
