cmake_minimum_required(VERSION 3.24)
project(test_boost)
find_package(Boost 1.36.0)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
if(Boost_FOUND)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "-g")
    include_directories(${Boost_INCLUDE_DIRS})

    add_executable(test_vanderpol_time vanderpol_time.cpp Dual.h AutoJacobian.h Clock.h FiniteDifferenceJacobian.h)
    target_link_libraries(test_vanderpol_time Eigen3::Eigen)

    add_executable(test_vanderpol_precision vanderpol_precision.cpp Dual.h AutoJacobian.h Clock.h FiniteDifferenceJacobian.h)
    target_link_libraries(test_vanderpol_precision Eigen3::Eigen)

    add_executable(test_hires_time hires_time.cpp Dual.h AutoJacobian.h Clock.h FiniteDifferenceJacobian.h)
    target_link_libraries(test_hires_time Eigen3::Eigen)

    add_executable(test_hires_precision hires_precision.cpp Dual.h AutoJacobian.h Clock.h FiniteDifferenceJacobian.h)
    target_link_libraries(test_hires_precision Eigen3::Eigen)

    add_executable(test_bruss_time bruss_time.cpp Dual.h AutoJacobian.h Clock.h FiniteDifferenceJacobian.h)
    target_link_libraries(test_bruss_time Eigen3::Eigen)

    add_executable(test_bruss_precision bruss_precision.cpp Dual.h AutoJacobian.h Clock.h FiniteDifferenceJacobian.h)
    target_link_libraries(test_bruss_precision Eigen3::Eigen)
endif()
set(CMAKE_CXX_STANDARD 17)

