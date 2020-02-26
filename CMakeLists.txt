cmake_minimum_required(VERSION 2.6)
project(uesmann)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-pg -std=c++11")


add_executable(uesmann_test test.cpp)
target_link_libraries(uesmann_test -lm)

