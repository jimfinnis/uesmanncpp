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
                d[j] = i*100+j;
            d= getOutputs(i);
            for(int j=0;j<getOutputCount();j++)
                d[j] = i*200+j;
            setH(i,i*1000);
        }
    }
};

/**
 * Test the basic example
 */

BOOST_AUTO_TEST_CASE(example) {
    TestExampleSet e;
    
    // just check the retrieved values are correct
    for(int i=0;i<e.getCount();i++){
        double *d = e.getInputs(i);
        for(int j=0;j<e.getInputCount();j++)
            BOOST_REQUIRE(d[j]== i*100+j);
        d= e.getOutputs(i);
        for(int j=0;j<e.getOutputCount();j++)
            BOOST_REQUIRE(d[j]== i*200+j);
        BOOST_REQUIRE(e.getH(i)==i*1000);
    }
}

/**
 * Test that subsetting examples works
 */

BOOST_AUTO_TEST_CASE(subset) {
    TestExampleSet parent;
    
    // check that bad values throw
    BOOST_REQUIRE_THROW(ExampleSet bad(parent,5,6),std::out_of_range);
    BOOST_REQUIRE_THROW(ExampleSet bad(parent,0,6),std::out_of_range);
    BOOST_REQUIRE_THROW(ExampleSet bad(parent,5,6),std::out_of_range);
    BOOST_REQUIRE_THROW(ExampleSet bad(parent,11,6),std::out_of_range);
    
    ExampleSet e(parent,5,5);
    BOOST_REQUIRE(e.getCount()==5);
    for(int i=0;i<e.getCount();i++){
        int parentIndex = i+5;
        double *d = e.getInputs(i);
        for(int j=0;j<e.getInputCount();j++)
            BOOST_REQUIRE(d[j]== parentIndex*100+j);
        d= e.getOutputs(i);
        for(int j=0;j<e.getOutputCount();j++)
            BOOST_REQUIRE(d[j]== parentIndex*200+j);
        BOOST_REQUIRE(e.getH(i)==parentIndex*1000);
    }
    
}

// simple shuffle for testing
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
 * Test the alternate() function
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


BOOST_AUTO_TEST_CASE(trainparams) {
    TestExampleSet e;
    
    int layers[3];
    layers[0] = e.getInputCount();
    layers[1] = 2;
    layers[2] = e.getOutputCount();
    BPNet n(0.1,3,layers);
    
    n.trainSGD(&e,
               1000,
               2,
               2,
               1);
}
