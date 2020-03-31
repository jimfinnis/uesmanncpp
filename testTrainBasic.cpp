/**
 * @file testTrainBasic.cpp
 * @brief Tests of basic training.
 *
 */

#include <iostream>
#include <boost/test/unit_test.hpp>

#include "test.hpp"

BOOST_AUTO_TEST_SUITE(basictrain)

/** \addtogroup testtrainbasic Training a plain unmodulated network
 * \ingroup tests
 * @{
 */


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
    
    // use half of the data as CV examples, 1000 CV cycles, 10 slices.
    // Don't shuffle the CV examples on epoch. Also, store the best net
    // and make sure we end up with that.
    params.crossValidation(e,0.5,1000,10,false).storeBest().setSeed(0);
    
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
    params.storeBest();
    
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

//! [addition]

/**
 * \brief Construct an addition model from scratch and try to learn it
 * with backprop
 */

BOOST_AUTO_TEST_CASE(addition) {
    // 1000 examples, 2 inputs, 1 output, 1 modulator level (i.e. no modulation)
    ExampleSet e(1000,2,1,1);
    
    // initialise a PRNG
    drand48_data rd;
    srand48_r(10,&rd);
    
    // create the examples
    for(int i=0;i<1000;i++){
        // get a pointer to the inputs for this example
        double *ins = e.getInputs(i);
        // and a pointer to the outputs (only one of them in this case)
        double *out = e.getOutputs(i);
        
        // use the PRNG to generate the operands in the range [0,0.5) to ensure
        // that the result is <1.
        double a,b;
        drand48_r(&rd,&a);a*=0.5;
        drand48_r(&rd,&b);b*=0.5;
        
        // write the inputs and the output
        ins[0] = a;
        ins[1] = b;
        *out = a+b;
    }
    
    // create a plain backprop network
    // which conforms to those examples with 2 hidden nodes.
    Net *net = NetFactory::makeNet(NetType::PLAIN,e,2);
    
    // set up training parameters:
    // eta=1, lots of  iterations
    Net::SGDParams params(1,10000000);
    // cross validation etc.:
    // use half of the data as CV examples. This is divided into 10 slices,
    // and we do cross-validation 1000 times during the run.
    // Don't shuffle the CV examples on epoch. Also, store the best net
    // and make sure we end up with that. We also set a PRNG seed to
    // ensure reproducibility.
    params.crossValidation(e, // example set
                           0.5, // proportion of set to hold back for CV
                           1000, // number of CV cycles
                           10, // number of CV slices
                           false // don't shuffle the entire CV set on completing an epoch
                           )
          .storeBest() // store the best net inside this param block
          .setSeed(0); // initialise PRNG for net, used for initial weights and shuffling.
    
    // do the training and get the MSE of the best net, found by cross-validation.
    double mse = net->trainSGD(e,params);
    printf("%f\n",mse);
    // check the MSE
    BOOST_REQUIRE(mse<0.03);
    
    // test the actual performance - loop through lots of pairs
    // of numbers <0.5 (the only numbers we can do given the range of the function)
    for(double a=0;a<0.5;a+=0.02){
        for(double b=0;b<0.5;b+=0.02){
            // set up an array holding the inputs
            double runIns[2];
            runIns[0]=a;
            runIns[1]=b;
            // run the net and get the output
            double out = *(net->run(runIns));
            // check the difference (with a line commented out to print it)
            double diff = fabs(out-(a+b));
//            printf("%f+%f=%f (%f)\n",a,b,out,diff);
            BOOST_REQUIRE(diff<0.05);
        }
    }
    delete net;
}

//! [addition]

//! [additionmod]

/**
 * \brief Construct an addition/addition+scaling model from scratch and try to learn it
 * with UESMANN. The h=0 is \f$y=a+b\f$, the h=1 function is \f$y=(a+b)*0.3\f$.
 */

BOOST_AUTO_TEST_CASE(additionmod) {
    // 2000 examples, 2 inputs, 1 output, 2 modulator levels. We need to know
    // the number of modulator levels so that example shuffling will shuffle in
    // blocks of that count, or will shuffle normally and then fix up to ensure that
    // example modulator levels alternate in the net (ExampleSet::STRIDE and
    // ExampleNet::ALTERNATE respectively).
    
    ExampleSet e(2000,2,1,2);
    
    // initialise a PRNG
    drand48_data rd;
    srand48_r(10,&rd);
    
    // create the examples
    int idx=0; // example index
    for(int i=0;i<1000;i++){
        // use the PRNG to generate the operands in the range [0,0.5) to ensure
        // that the result is <1.
        double a,b;
        drand48_r(&rd,&a);a*=0.5;
        drand48_r(&rd,&b);b*=0.5;
        
        // get a pointer to the inputs for the h=0 example and write to it
        double *ins = e.getInputs(idx);
        double *out = e.getOutputs(idx);
        ins[0] = a;
        ins[1] = b;
        *out = a+b;
        e.setH(idx,0); // set modulator for this example
        idx++; // increment the example index
        
        // Do the same for the h=1 example, but here we're writing 0.3(a+b),
        // and the modulator is 1.
        ins = e.getInputs(idx);
        out = e.getOutputs(idx);
        ins[0] = a;
        ins[1] = b;
        *out = (a+b)*0.3;
        e.setH(idx,1);
        idx++;
        
    }
    
    // create a UESMANN network which conforms to those examples with 2 hidden nodes.
    Net *net = NetFactory::makeNet(NetType::UESMANN,e,2);
    
    // set up training parameters:
    // eta=1, lots of iterations
    Net::SGDParams params(1,1000000);
    // cross validation etc.:
    // use half of the data as CV examples. This is divided into 10 slices,
    // and we do cross-validation 1000 times during the run.
    // DO shuffle the CV examples on epoch. Also, store the best net
    // and make sure we end up with that. We also set a PRNG seed to
    // ensure reproducibility.
    params.crossValidation(e, // example set
                           0.5, // proportion of set to hold back for CV
                           1000, // number of CV cycles
                           10, // number of CV slices
                           true // shuffle the entire CV set on completing an epoch
                           )
          .storeBest() // store the best net inside this param block
          .setSeed(0); // initialise PRNG for net, used for initial weights and shuffling.
    
    // do the training and get the MSE of the best net, found by cross-validation.
    double mse = net->trainSGD(e,params);
    printf("%f\n",mse);
    // check the MSE
    BOOST_REQUIRE(mse<0.03);
    
    // test the actual performance - loop through lots of pairs
    // of numbers. We limit the range here; performance is known to fall
    // off at the ends due to node saturation (probably)
    for(double a=0.1;a<0.4;a+=0.02){
        for(double b=0.1;b<0.4;b+=0.02){
            // set up an array holding the inputs
            double runIns[2];
            runIns[0]=a;
            runIns[1]=b;
            // check for H=0
            net->setH(0);
            double out = *(net->run(runIns));
            // check the difference (with a line commented out to print it)
            double diff = fabs(out-(a+b));
            printf("%f+%f=%f (%f)\n",a,b,out,diff);
            BOOST_REQUIRE(diff<0.07);
            
            // check for H=1
            net->setH(1);
            out = *(net->run(runIns));
            diff = fabs(out-(a+b)*0.3);
            printf("%f+%f=%f (%f)\n",a,b,out,diff);
            BOOST_REQUIRE(diff<0.07); 
        }
    }
    delete net;
}
//! [additionmod]


//! [trainmnist]
/**
 * \brief Train for MNIST handwriting recognition in a plain backprop network.
 * This doesn't do a huge number of iterations.
 */
BOOST_AUTO_TEST_CASE(trainmnist){
    // Create an MNIST object, which consists of labelled data in the standard MNIST
    // format (http://yann.lecun.com/exdb/mnist/). This is in two files, one containing
    // the images and one containing the data.
    
    MNIST m("../testdata/train-labels-idx1-ubyte","../testdata/train-images-idx3-ubyte");
    
    // This ExampleSet constructor builds the examples directly from the MNIST data,
    // with a large number of inputs (28x28) and a number of outputs equal to the maximum
    // label value + 1. The outputs of the examples are in a one-hot format: for handwritten
    // digits, there will be 10 in which the output corresponding to the label will
    // be 1 with the others 0.
    ExampleSet e(m);
    
    // create a plain MLP network conforming to the examples' input and output counts
    // with 16 hidden nodes
    Net *n = NetFactory::makeNet(NetType::PLAIN,e,16);
    
    // set up the parameters
    Net::SGDParams params(0.1,10000); // eta,iterations
    
    // use half of the data as CV examples, 1000 CV cycles, 10 slices.
    // Shuffle the CV examples on epoch. Also, store the best net
    // and make sure we end up with that. Set the seed to 10.
    // Shuffle mode is stride, the default (not that it matters here)
    
    params.crossValidation(e,0.5,1000,10,true)
          .storeBest()
          .setSeed(10);
    
    // train and get the MSE, and test it is low.
    double mse = n->trainSGD(e,params);
    BOOST_REQUIRE(mse<0.03);
    
    // now load the test set of 10000 images and labels and construct
    // examples in a similar way.
    MNIST mtest("../testdata/t10k-labels-idx1-ubyte","../testdata/t10k-images-idx3-ubyte");
    ExampleSet testSet(mtest);
    
    // and test against the test set, recording how many are good.
    int correct=0;
    for(int i=0;i<testSet.getCount();i++){
        // for each test, get the inputs
        double *ins = testSet.getInputs(i);
        // run them through the network and get the outputs
        double *o = n->run(ins);
        // find the correct label by getting the highest output in the example
        int correctLabel = getHighest(testSet.getOutputs(i),testSet.getOutputCount());
        // find the network's result by getting its highest output 
        int netLabel = getHighest(o,testSet.getOutputCount());
        // and increment the count if they agree
        if(correctLabel==netLabel)correct++;
    }
    
    // get the ratio of correct answers -
    // we've not trained for long so this isn't going to be brilliant performance.
    double ratio = ((double)correct)/(double)testSet.getCount();
    printf("MSE=%f, correct=%d/%d=%f\n",mse,correct,testSet.getCount(),ratio);
    // assert that it's at least 85%
    BOOST_REQUIRE(ratio>0.85);
    delete n;
}
//! [trainmnist]

/** 
 * @}
 */


BOOST_AUTO_TEST_SUITE_END()
