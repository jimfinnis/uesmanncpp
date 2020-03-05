/**
 * @file test.cpp
 * @brief  Basic tests of underlying functionality, or things which only take
 * a short time to run!
 *
 */

#include <iostream>

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "test.hpp"

/**
 * \brief Utility test class.
 * Constructs a standard set: 10 examples, 5 ins, 2 outs:
 * 
 * * input j of example i is i*100+j
 * * output j of example i is i*200+j
 * * h of example i is i*1000
 * 
 * While the set says there are 2 h levels, this is
 * untrue (however, a later test resets the H to match this)
 */

class TestExampleSet : public ExampleSet {
public:
    TestExampleSet() : ExampleSet(10,5,2, 2){
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

BOOST_AUTO_TEST_SUITE(basic)

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
 * \brief Test the alternation function on examples
 */

BOOST_AUTO_TEST_CASE(altex){
    TestExampleSet e;
    for(int i=0;i<e.getCount();i++){
        e.setH(i, i<e.getCount()/2 ? 1:0);
    }
    drand48_data rd;
    srand48_r(10,&rd);
    e.shuffle(&rd,ExampleSet::ALTERNATE);
    for(int i=0;i<e.getCount();i++){
        BOOST_REQUIRE((e.getH(i)<0.5 ? 0 : 1) == i%2);
    }
}

/**
 * \brief test strided examples
 */

BOOST_AUTO_TEST_CASE(shufflestride){
    static const int NEX=32;
    ExampleSet e(NEX,2,1, 4); // 16 examples, 2 inputs, 1 output, 4 h-points
    for(int i=0;i<NEX/4;i++){
        int exbase = i*4;
        for(int j=0;j<4;j++){
            int ex = exbase+j;
            double *d = e.getInputs(ex);
            for(int k=0;k<e.getInputCount();k++)
                d[k] = k*10+j;
            *e.getOutputs(ex) = ex;
            e.setH(ex,i);
        }
    }
    
    drand48_data rd;
    srand48_r(10,&rd);
    e.shuffle(&rd,ExampleSet::STRIDE);
    for(int i=0;i<NEX;i++){
        double *d = e.getInputs(i);
        double o = *e.getOutputs(i);
        double h = e.getH(i);
        printf("(%f,%f)->%f, h=%f\n",d[0],d[1],o,h);
    }
    /**\bug write actual test! */
}


/** 
 * \brief set all parameters (weights and biases) in a network to zero
 * \param n the network to zero
 */

void zero(Net *n){
    int ct = n->getDataSize();
    double *buf = new double[ct];
    for(int i=0;i<ct;i++)buf[i]=0;
    n->load(buf);
    delete[] buf;
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
    Net *n = NetFactory::makeNet(NetType::PLAIN,e,2);
    // zero it so all the nodes produce 0.5 as their output
    zero(n);
    // get the MSE on all examples, given the example values
    // in the test set
    double t = n->test(e);
    
    // value determined by running the network with a lot of
    // debug printing
    BOOST_REQUIRE(t==11400.25);
    
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

BOOST_AUTO_TEST_SUITE_END()

