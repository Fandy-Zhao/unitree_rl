cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(test_libtorch)
set(Torch_DIR "/home/zzf/install_source/libtorch-shared-with-deps-1.10.0+cpu/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

add_executable(v_test v_test.cpp)
target_link_libraries(v_test "${TORCH_LIBRARIES}")
set_property(TARGET v_test PROPERTY CXX_STANDARD 14)


add_executable(test_libtorch testlibtorch.cpp)
target_link_libraries(test_libtorch "${TORCH_LIBRARIES}")
set_property(TARGET test_libtorch PROPERTY CXX_STANDARD 14)