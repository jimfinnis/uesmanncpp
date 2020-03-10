cmake_minimum_required(VERSION 2.6)
project(uesmann)

find_package(Boost COMPONENTS unit_test_framework REQUIRED)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-pg -std=c++11")

set(UESMANN_LIBS -lm)

add_executable(uesmann_test testBasic.cpp testTrainBasic.cpp
    testTrainBooleans.cpp)

target_link_libraries(uesmann_test
    ${UESMANN_LIBS}
    ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    )

