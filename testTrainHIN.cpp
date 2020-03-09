/**
 * @file testTrainHIN.cpp
 * @brief Simple boolean test for modulation in h-as-input (XOR->AND)
 *
 */

#include <iostream>
#include <boost/test/unit_test.hpp>

#include "test.hpp"

BOOST_AUTO_TEST_SUITE(hin)


BOOST_AUTO_TEST_CASE(obxorand) {
    BooleanExampleSet e;
    // xor examples
    e.add0(0,1,1,0);
    // and examples
    e.add1(0,0,0,1);
    
    // make a net with 2 hidden nodes
    Net *net = NetFactory::makeNet(NetType::OUTPUTBLENDING,e,2);
    // train it
    Net::SGDParams params(0.01,1000000);
    params.storeBest().crossValidation(e,0.5,10000,1,false).setSeed(0);
    
    double mse = net->trainSGD(e,params);
    printf("%f\n",mse);
    BOOST_REQUIRE(mse<0.002);
    
    // now test
    BOOST_REQUIRE(booleanTest(net,0,  0,0,0)<0.1);
    BOOST_REQUIRE(booleanTest(net,0,  0,1,1)<0.1);
    BOOST_REQUIRE(booleanTest(net,0,  1,0,1)<0.1);
    BOOST_REQUIRE(booleanTest(net,0,  1,1,0)<0.1);
    BOOST_REQUIRE(booleanTest(net,1,  0,0,0)<0.1);
    BOOST_REQUIRE(booleanTest(net,1,  0,1,0)<0.1);
    BOOST_REQUIRE(booleanTest(net,1,  1,0,0)<0.1);
    BOOST_REQUIRE(booleanTest(net,1,  1,1,1)<0.1);
}

BOOST_AUTO_TEST_SUITE_END()
