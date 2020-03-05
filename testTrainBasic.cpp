/**
 * @file testTrainBasic.cpp
 * @brief Tests of basic training.
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

/**
 * \brief Loading MNIST data and converting to an example set.
 * Ensure we can load MNIST data into an example set, and that
 * the image and its label are correct. The former is hard to test
 * automatically, so I'll rely on having eyeballed it.
 */
BOOST_AUTO_TEST_CASE(loadmnist) {
    MNIST m("../testdata/t10k-labels-idx1-ubyte","../testdata/t10k-images-idx3-ubyte");
    ExampleSet e(m);
    
    // in this data set, example 1233 should be a 5.
    double *in = e.getInputs(1233);
    for(int y=0;y<28;y++){
        for(int x=0;x<28;x++){
            uint8_t qq = *in++ * 10;
            if(qq>9)putchar('?');
            else putchar(qq?qq+'0':'.');
        }
        putchar('\n');
    }
    double *out = e.getOutputs(1233);
    BOOST_REQUIRE(e.getOutputCount()==10);
    for(int i=0;i<10;i++){
        if(i==5)
            BOOST_REQUIRE(out[i]==1.0);
        else
            BOOST_REQUIRE(out[i]==0.0);
    }
}

BOOST_AUTO_TEST_CASE(trainmnist){
    // load training set and make examples
    MNIST m("../testdata/train-labels-idx1-ubyte","../testdata/train-images-idx3-ubyte");
    ExampleSet e(m);
    // create a network conforming to the examples' input and output counts with 16 hidden nodes
    Net *n = NetFactory::makeNet(NetType::PLAIN,e,16);
    
    Net::SGDParams params(0.1,10000); // eta,iterations
    
    // use half of the data as CV examples, 1000 CV cycles, 10 slices.
    // Shuffle the CV examples on epoch. Also, store the best net
    // and make sure we end up with that. Set the seed to 10.
    // Shuffle mode is stride, the default (not that it matters here)
    
    params.crossValidation(e,0.5,1000,10,true)
          .storeBest(*n)
          .setSeed(10);
    
    double mse = n->trainSGD(e,params);
    
    // load test set
    MNIST mtest("../testdata/t10k-labels-idx1-ubyte","../testdata/t10k-images-idx3-ubyte");
    ExampleSet testSet(mtest);
    
    // and test against the test set, recording how many are good.
    int correct=0;
    for(int i=0;i<testSet.getCount();i++){
        double *ins = testSet.getInputs(i);
        double *o = n->run(ins);
        int correctLabel = getHighest(testSet.getOutputs(i),testSet.getOutputCount());
        int netLabel = getHighest(o,testSet.getOutputCount());
        if(correctLabel==netLabel)correct++;
    }
    
    // we've not trained for long so this isn't going to be awesome.
    double ratio = ((double)correct)/(double)testSet.getCount();
    printf("MSE=%f, correct=%d/%d=%f\n",mse,correct,testSet.getCount(),ratio);
    BOOST_REQUIRE(mse<0.03);
    BOOST_REQUIRE(ratio>0.85);
}



BOOST_AUTO_TEST_SUITE_END()
