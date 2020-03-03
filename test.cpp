/**
 * @file test.cpp
 * @brief  Brief description of file.
 *
 */

#include <iostream>

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "bpnet.hpp"

/**
 * \brief Utility test class.
 * Constructs a standard set: 10 examples, 5 ins, 2 outs:
 * 
 * * input j of example i is i*100+j
 * * output j of example i is i*200+j
 * * h of example i is i*1000
 * 
 */

class TestExampleSet : public ExampleSet {
public:
    TestExampleSet() : ExampleSet(10,5,2){
        for(int i=0;i<getCount();i++){
            double *d = getInputs(i);
            for(int j=0;j<getInputCount();j++)
                d[j] = i*10+j;
            d= getOutputs(i);
            for(int j=0;j<getOutputCount();j++)
                d[j] = i*20+j;
            setH(i,i*1000);
        }
    }
};

/**
 * \brief A simple test network, wraps a network
 * based on a given example set with a single hidden layer of 2 nodes.
 * It also has a method to zero all the parameters (weights and biases), used
 * in testing MSE.
 */

class TestNet {
public:
    BPNet *n;
    TestNet(const ExampleSet& e,int hnodes) {
        int layers[3];
        layers[0] = e.getInputCount();
        layers[1] = hnodes;
        layers[2] = e.getOutputCount();
        n = new BPNet(2,layers);
    }
    
    void zero(){
        int ct = n->getDataSize();
        double *buf = new double[ct];
        for(int i=0;i<ct;i++)buf[i]=0;
        n->load(buf);
        delete[] buf;
    }
    
    ~TestNet(){
        delete n;
    }
};



/**
 * \brief Test the basic example
 */

BOOST_AUTO_TEST_CASE(example) {
    TestExampleSet e;
    
    // just check the retrieved values are correct
    for(int i=0;i<e.getCount();i++){
        double *d = e.getInputs(i);
        for(int j=0;j<e.getInputCount();j++)
            BOOST_REQUIRE(d[j]== i*10+j);
        d= e.getOutputs(i);
        for(int j=0;j<e.getOutputCount();j++)
            BOOST_REQUIRE(d[j]== i*20+j);
        BOOST_REQUIRE(e.getH(i)==i*1000);
    }
}

/**
 * \brief Test that subsetting examples works
 */

BOOST_AUTO_TEST_CASE(subset) {
    TestExampleSet parent;
    
    // check that bad values throw
    BOOST_REQUIRE_THROW(ExampleSet bad(parent,5,6),std::out_of_range);
    BOOST_REQUIRE_THROW(ExampleSet bad(parent,-1,6),std::out_of_range);
    BOOST_REQUIRE_THROW(ExampleSet bad(parent,5,6),std::out_of_range);
    BOOST_REQUIRE_THROW(ExampleSet bad(parent,11,6),std::out_of_range);
    
    ExampleSet e(parent,5,5);
    BOOST_REQUIRE(e.getCount()==5);
    for(int i=0;i<e.getCount();i++){
        int parentIndex = i+5;
        double *d = e.getInputs(i);
        for(int j=0;j<e.getInputCount();j++)
            BOOST_REQUIRE(d[j]== parentIndex*10+j);
        d= e.getOutputs(i);
        for(int j=0;j<e.getOutputCount();j++)
            BOOST_REQUIRE(d[j]== parentIndex*20+j);
        BOOST_REQUIRE(e.getH(i)==parentIndex*1000);
    }
    
}

/**
   \brief simple shuffle for testing
 */
template <class T> void sshuffle(T *x, int ct){
    T tmp;
    for(int i=ct-1;i>=1;i--){
        long lr;
        int j = rand()%(i+1);
        tmp=x[i];
        x[i]=x[j];
        x[j]=tmp;
    }
}


/**
 * \brief Test the alternate() function
 */

BOOST_AUTO_TEST_CASE(alt) {
    long t;
    srand(time(&t));
    // make a bunch of numbers and shuffle them
    int arr[100];
    for(int i=0;i<100;i++){
        arr[i] = i;
    }
    sshuffle<int>(arr,100);
    
    // make them alternate odd and even
    alternate<int>(arr,100,[](int *v){return (*v)%2==0;});
    
    // make sure each item is there only once and that the sequence
    // alternates odd and even
    bool seen[100];
    for(int i=0;i<100;i++)seen[i]=false;
    for(int i=0;i<100;i++){
        int n = arr[i];
        BOOST_REQUIRE(!seen[n]);
        seen[n]=true;
        BOOST_REQUIRE((n%2) == (i%2));
    }
}

/**
 * \brief Test training.
 * This just checks that the network trains.
 */

BOOST_AUTO_TEST_CASE(trainparams) {
    const int NUMEXAMPLES=1000;
    ExampleSet e(NUMEXAMPLES,1,1); // 100 examples at 1 input, 1 output
    
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
    TestNet n(e,3);
    
    // eta=0.1, 10000 iterations
    Net::SGDParams params(0.1,10000);
    // use half of the data as CV examples, 1000 CV cycles, 3 slices.
    // Don't shuffle the CV examples on epoch. Also, store the best net
    // and make sure we end up with that.
    params.crossValidation(e,0.5,1000,10,false).storeBest(*n.n);
    
    // do the training and get the MSE of the best net.
    double mse = n.n->trainSGD(e,params);
    printf("%f\n",mse);
    
    // assert that it's a sensible value
    BOOST_REQUIRE(mse>0);
    BOOST_REQUIRE(mse<0.005);
}


/**
 * \brief Test mean sum squared error of outputs.
 * This test finds the mean of the sum of the squared errors on the outputs
 * across all examples in the test set, in the case of a network where
 * all the parameters are zero. In this case, all outputs will be 0.5 given
 * the logistic sigmoid activation function. We test for the correct value
 * determined by lots of print statements during development.
 */

BOOST_AUTO_TEST_CASE(testmse) {
    TestExampleSet e;
    // make a standard net
    TestNet n(e,2);
    // zero it so all the nodes produce 0.5 as their output
    n.zero();
    // get the MSE on all examples, given the example values
    // in the test set
    double t = n.n->test(e);
    
    // value determined by running the network with a lot of
    // debug printing
    BOOST_REQUIRE(t==11400.25);
    
}
