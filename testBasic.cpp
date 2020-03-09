/**
 * @file testBasic.cpp
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
    static const int NUMEXAMPLES = 100;
    static const int CYCLE=5;
    
    srand(time(&t));
    // make a bunch of numbers and shuffle them
    int arr[NUMEXAMPLES];
    for(int i=0;i<NUMEXAMPLES;i++){
        arr[i] = i;
    }
    sshuffle<int>(arr,NUMEXAMPLES);
    
    // make them alternate odd and even
    alternate<int>(arr,NUMEXAMPLES,CYCLE,[](int v){return v;});
    
//    for(int i=0;i<NUMEXAMPLES;i++)printf("%d ",arr[i]%CYCLE);
    
    // make sure each item is there only once and that the sequence
    // alternates odd and even
    bool seen[NUMEXAMPLES];
    for(int i=0;i<NUMEXAMPLES;i++)seen[i]=false;
    for(int i=0;i<NUMEXAMPLES;i++){
        int n = arr[i];
        BOOST_REQUIRE(!seen[n]);
        seen[n]=true;
        BOOST_REQUIRE((n%CYCLE) == (i%CYCLE));
    }
}

/**
 * \brief Test the alternation function on examples, simple version
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
 * \brief test strided example shuffle, 4 different modulator levels
 */

BOOST_AUTO_TEST_CASE(shufflestride){
    static const int NUMBEREXAMPLES=32;
    // 32 examples (or however many) with 2 inputs and 1 output, and 4
    // different modulator values. We then create 4 different groups.
    // Each group has 4 examples with the same inputs, but with different outputs
    // between the groups. Examples within each group have the same modulator value.
    
    ExampleSet e(NUMBEREXAMPLES,2,1, 4); 
    for(int i=0;i<NUMBEREXAMPLES/4;i++){
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
    
    // shuffle the data with STRIDE, so that the four groups are each considered as
    // a whole and moved en-bloc.
    
    drand48_data rd;
    srand48_r(10,&rd);
    e.shuffle(&rd,ExampleSet::STRIDE);
    
    // check the results, both that the output is non-monotonic (i.e. data is shuffled)
    // and that the inputs appear to show that the blocks are intact.
    
    int lasto = -1;
    bool monotonic_increasing=true;
    for(int i=0;i<NUMBEREXAMPLES;i++){
        double *d = e.getInputs(i);
        int i0 = (int)d[0];
        int i1 = (int)d[1];
        int o = (int)*e.getOutputs(i);
        int h = (int)e.getH(i);
        if(o<lasto)monotonic_increasing=false;
        lasto=o;
        BOOST_REQUIRE(i0 == i%4);
        BOOST_REQUIRE(o/4  == h);
    }
    BOOST_REQUIRE(!monotonic_increasing);
}

/**
 * \brief test strided example shuffle, 4 different modulator levels
 */

BOOST_AUTO_TEST_CASE(altex4){
    static const int NUMBEREXAMPLES=32;
    // 32 examples (or however many) with 2 inputs and 1 output, and 4
    // different modulator values. We then create 4 different groups.
    // Each group has 4 examples with the same inputs, but with different outputs
    // between the groups. Examples within each group have the same modulator value.
    // Concretely, examples (index,in0,in1,out,h) will be
    // (0 0 10 0 0) (1 1 11 1 0) (2 2 12 2 0) (3 3 13 3 0) (4 0 10 4 0) (5 1 11 5 0)
    //       (6 2 12 6 0) (7 3 13 7 0)
    // (8 0 10 8 1) (9 1 11 9 1) (10 2 12 10 1) (11 3 13 11 1) (12 0 10 12 1) (13 1 11 13 1)
    //       (14 2 12 14 1) (15 3 13 15 1)
    // and so on. These should be shuffled randomly, and then rearranged so that the 
    // h-value (the last value in the brackets) goes (0,1,2,3,0,1,2,3...)
          
    ExampleSet e(NUMBEREXAMPLES,2,1, 4); 
    for(int i=0;i<NUMBEREXAMPLES/4;i++){
        int exbase = i*4;
        for(int j=0;j<4;j++){
            int ex = exbase+j;
            double *d = e.getInputs(ex);
            for(int k=0;k<e.getInputCount();k++)
                d[k] = k*10+j;
            *e.getOutputs(ex) = ex;
            e.setH(ex,i/2); // gives four hormone levels
            e.setHRange(0,i/2); // reset max h-level (will end up at 3)
        }
    }
    
    // shuffle the data with ALTERNATE, so everything is shuffled freely, but
    // ensure afterwards that the h-levels go 01230123....
    
    drand48_data rd;
    srand48_r(10,&rd);
    e.shuffle(&rd,ExampleSet::ALTERNATE);
    
    // check the results, both that the output is non-monotonic (i.e. data is shuffled)
    // and that the h-sequence is correct
    
    int lasto = -1;
    bool monotonic_increasing=true;
    for(int i=0;i<NUMBEREXAMPLES;i++){
        double *d = e.getInputs(i);
        int i0 = (int)d[0];
        int i1 = (int)d[1];
        int o = (int)*e.getOutputs(i);
        int h = (int)e.getH(i);
        if(o<lasto)monotonic_increasing=false;
        lasto=o;
        BOOST_REQUIRE(i%4==h);
        BOOST_REQUIRE(o%4==i0); // check internal integrity of datum
    }
    BOOST_REQUIRE(!monotonic_increasing);
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
    // set up a net which conforms to those examples with 2 hidden nodes.
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
    BOOST_REQUIRE(mse<0.03);
}
//! [addition]


BOOST_AUTO_TEST_SUITE_END()

