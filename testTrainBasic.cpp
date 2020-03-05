/**
 * @file testTrainBasic.cpp
 * @brief Tests of basic training
 *
 */

#include <iostream>
#include <boost/test/unit_test.hpp>

#include "test.hpp"

BOOST_AUTO_TEST_SUITE(basictrain)

/**
 * \brief Test training.
 * This just checks that the network trains.
 */

BOOST_AUTO_TEST_CASE(trainparams) {
    const int NUMEXAMPLES=1000;
    ExampleSet e(NUMEXAMPLES,1,1 ,1); // 100 examples at 1 input, 1 output, 1 mod level
    
    double recipNE = 1.0/(double)NUMEXAMPLES;
    // generate examples of the identity function y=x, from 0 to 1.
    // This will train but will be a bit iffy at the ends!
    
    for(double i=0;i<NUMEXAMPLES;i++){
        double v = i*recipNE;
        *(e.getInputs(i)) = v;
        *(e.getOutputs(i)) = v;
        e.setH(i,0);
    }
    
    // set up a net which conforms to those examples with 3 hidden nodes.
    Net *net = NetFactory::makeNet(NetType::PLAIN,e,3);
    
    // eta=1, lots of  iterations
    Net::SGDParams params(1,10000000);
    
    // use half of the data as CV examples, 1000 CV cycles, 3 slices.
    // Don't shuffle the CV examples on epoch. Also, store the best net
    // and make sure we end up with that.
    params.crossValidation(e,0.5,1000,10,true).storeBest(*net).setSeed(0);
    
    // do the training and get the MSE of the best net.
    double mse = net->trainSGD(e,params);
    printf("%f\n",mse);
    // assert that it's a sensible value
    BOOST_REQUIRE(mse>0);
    BOOST_REQUIRE(mse<0.005);
#if 0
    for(double i=0;i<NUMEXAMPLES;i++){
        double v = i*recipNE;
        double o = *(net->run(&v));
        printf("    %f,%f\n",v,o);
    }
#endif
    delete net;
}

/**
 * \brief another test without cross-validation which attempts to emulate the Angort
 * test.ang program.
 */

BOOST_AUTO_TEST_CASE(trainparams2) {
    const int NUMEXAMPLES=100;
    ExampleSet e(NUMEXAMPLES*2,1,1,1); // 100 examples at 1 input, 1 output, 1 modulator level
    
    double recipNE = 1.0/(double)NUMEXAMPLES;
    
    for(double i=0;i<NUMEXAMPLES*2;i+=2){
        double v = (i/2)*recipNE;
        *(e.getInputs(i)) = v;
        *(e.getOutputs(i)) = v;
        *(e.getInputs(i+1)) = v;
        *(e.getOutputs(i+1)) = v;
        e.setH(i,0);
        e.setH(i+1,1);
    }
    
    Net *net = NetFactory::makeNet(NetType::PLAIN,e,2);
    
    // eta=1, 10000000 iterations. No CV.
    Net::SGDParams params(1,10000000);
    params.storeBest(*net);
    
    // do the training and get the MSE of the best net.
    double mse = net->trainSGD(e,params);
    printf("%f\n",mse);
#if 0
    for(double i=0;i<NUMEXAMPLES;i++){
        double v = i*recipNE;
        double o = *(net->run(&v));
        printf("%f -> %f\n",v,o);
    }
#endif
    // assert that it's a sensible value
    BOOST_REQUIRE(mse>0);
    BOOST_REQUIRE(mse<0.005);
    
    delete net;
}

BOOST_AUTO_TEST_SUITE_END()
