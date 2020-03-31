/**
 * @file testTrainBooleans.cpp
 * @brief Simple boolean tests for modulation
 *
 */

#include <iostream>
#include <boost/test/unit_test.hpp>

#include "test.hpp"

BOOST_AUTO_TEST_SUITE(booleans)

/** \addtogroup booleantests Testing modulatory networks transitioning from XOR to AND
 * \ingroup tests
 * @{
 */



/**
 * \brief test function for booleans.
 * Creates a network with 2 hidden nodes, 2 inputs and 1 output
 * of the appropriate type, and trains it to transition between
 * XOR at h=0 and AND at h=1.
 */

static void dotest(NetType tp){
    static const double threshold=0.4;
    BooleanExampleSet e;
    // xor examples
    e.add0(0,1,1,0);
    // and examples
    e.add1(0,0,0,1);
    
    // make a net with 2 hidden nodes of the right type
    Net *net;
    try {
        net = NetFactory::makeNet(tp,e,2);
    } catch(std::runtime_error *e) {
        BOOST_FAIL(e->what());
    }
    // train it
    Net::SGDParams params(0.1,1000000);
    params.storeBest().crossValidation(e,0.5,10000,1,false).setSeed(1);
    
    double mse = net->trainSGD(e,params);
    printf("%f\n",mse);
    BOOST_REQUIRE(mse<0.002);
    
    // now test
    BOOST_REQUIRE(booleanTest(net,0,  0,0,0)<threshold);
    BOOST_REQUIRE(booleanTest(net,0,  0,1,1)<threshold);
    BOOST_REQUIRE(booleanTest(net,0,  1,0,1)<threshold);
    BOOST_REQUIRE(booleanTest(net,0,  1,1,0)<threshold);
    BOOST_REQUIRE(booleanTest(net,1,  0,0,0)<threshold);
    BOOST_REQUIRE(booleanTest(net,1,  0,1,0)<threshold);
    BOOST_REQUIRE(booleanTest(net,1,  1,0,0)<threshold);
    BOOST_REQUIRE(booleanTest(net,1,  1,1,1)<threshold);
    delete net;
}

/**
 * \brief Test of output blending on XOR->AND modulation
 */
BOOST_AUTO_TEST_CASE(obxorand) {
    dotest(NetType::OUTPUTBLENDING);
}
/**
 * \brief Test of h-as-input on XOR->AND modulation
 */
BOOST_AUTO_TEST_CASE(hinxorand) {
    dotest(NetType::HINPUT);
}
/**
 * \brief Test of UESMANN on XOR->AND modulation
 */
BOOST_AUTO_TEST_CASE(uesxorand) {
    dotest(NetType::UESMANN);
}

/** 
 * @}
 */

BOOST_AUTO_TEST_SUITE_END()
