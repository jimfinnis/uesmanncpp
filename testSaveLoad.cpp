/**
 * @file testSaveLoad.cpp
 * @brief Tests of loading and saving. These work by
 * generating random networks, running them, and load/save cycling
 * them to see if the params are the same
 *
 */

#include <iostream>

#include <boost/test/unit_test.hpp>

#include "test.hpp"

/** \addtogroup saveloadtests save and load tests.
 * \ingroup tests
 * @{
 */


BOOST_AUTO_TEST_SUITE(saveload)


void testSaveLoad(NetType tp){
    // generate a new network
    int layers[3];
    layers[0]=4;
    layers[1]=3;
    layers[2]=2;
    Net *n = NetFactory::makeNet(tp,3,layers);
    
    // generate a toy example. Doesn't matter what it is.
    ExampleSet e(1,4,2,1);
    double *p = e.getInputs(0);
    *p++=0;
    *p++=2;
    *p++=3;
    *p=1;
    p = e.getOutputs(0);
    *p++=100;
    *p=20;
    e.setH(0,0);
    
    // train it a little.
    Net::SGDParams parms(10,e,100);
    n->trainSGD(e,parms);
    
    // save the net to memory
    double *oldData = new double[n->getDataSize()];
    n->save(oldData);
    
    // now save the net to disk
    NetFactory::save("foo.net",n);
    
    // and load
    Net *saved = NetFactory::load("foo.net");
    
    BOOST_REQUIRE(n->type == saved->type);
    BOOST_REQUIRE(n->getDataSize() == saved->getDataSize());
    
    // save the newly loaded net params to memory
    double *savedData = new double[saved->getDataSize()];
    saved->save(savedData); 
    
    // and compare params
    for(int i=0;i<n->getDataSize();i++){
        BOOST_REQUIRE(oldData[i]==savedData[i]);
    }
    
    delete [] savedData;
    delete [] oldData;
    delete n;
}


/**
 * \brief Test that saving and loading a plain network
 * leaves the weights and biases unchanged
 */
BOOST_AUTO_TEST_CASE(saveloadplain) {
    testSaveLoad(NetType::PLAIN);
}
/**
 * \brief Test that saving and loading an output blending network
 * leaves the weights and biases unchanged
 */
BOOST_AUTO_TEST_CASE(saveloadob) {
    testSaveLoad(NetType::OUTPUTBLENDING);
}
/**
 * \brief Test that saving and loading an h-as-input network
 * leaves the weights and biases unchanged
 */
BOOST_AUTO_TEST_CASE(saveloadhin) {
    testSaveLoad(NetType::HINPUT);
}
/**
 * \brief Test that saving and loading a UESMANN network
 * leaves the weights and biases unchanged
 */
BOOST_AUTO_TEST_CASE(saveloadues) {
    testSaveLoad(NetType::UESMANN);
}


/** 
 * @}
 */


BOOST_AUTO_TEST_SUITE_END()

