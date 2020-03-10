/**
 * @file test.hpp
 * @brief Useful stuff for testing
 *
 */

#ifndef __TEST_HPP
#define __TEST_HPP

#include "netFactory.hpp"

/** \defgroup tests Unit tests */

namespace utf = boost::unit_test;

/**
 * \brief get index of max value in an array
 * \param o array ptr
 * \param n length of array
 */

static int getHighest(double *o,int n){
    int h=0;
    double maxval=-10;
    for(int i=0;i<n;i++,o++){
        if(*o > maxval){
            maxval= *o;
            h=i;
        }
    }
    return h;
}


/**
 * \brief boolean example set: 16 examples, 2 inputs, 1 output, 2 mod levels.
 * There are 4 examples for each function and they're repeated twice,
 * so we can do "cross-validation" on the identical second half.
 */

class BooleanExampleSet : public ExampleSet {
    
    /**
     * \brief Helper for setting up boolean examples
     * \param i   index of example
     * \param h   h of example
     * \param in0 input 0
     * \param in1 input 1
     * \param out required output
     */
    void setExample(int i,double h,double in0,double in1,double out){
        double *ins = getInputs(i);
        ins[0]=in0;
        ins[1]=in1;
        *getOutputs(i) = out;
        setH(i,h);
    }
public:
    
    BooleanExampleSet() : ExampleSet(16,2,1,2) {}
    
    /**
     * \brief set the 4 examples at modulator=0
     * \param o00 output for 0,0
     * \param o01 output for 0,1
     * \param o10 output for 1,0
     * \param o11 output for 1,1
     */
    void add0(double o00,double o01,double o10,double o11){
        setExample(0,0, 0,0, o00);
        setExample(2,0, 0,1, o01);
        setExample(4,0, 1,0, o10);
        setExample(6,0, 1,1, o11);
        setExample(8,0, 0,0, o00);
        setExample(10,0, 0,1, o01);
        setExample(12,0, 1,0, o10);
        setExample(14,0, 1,1, o11);
    }
    /**
     * \brief set the 4 examples at modulator=1
     * \param o00 output for 0,0
     * \param o01 output for 0,1
     * \param o10 output for 1,0
     * \param o11 output for 1,1
     */
    void add1(double o00,double o01,double o10,double o11){
        setExample(1,1, 0,0, o00);
        setExample(3,1, 0,1, o01);
        setExample(5,1, 1,0, o10);
        setExample(7,1, 1,1, o11);
        setExample(9,1, 0,0, o00);
        setExample(11,1, 0,1, o01);
        setExample(13,1, 1,0, o10);
        setExample(15,1, 1,1, o11);
    }
};

/**
 * \brief boolean checker. Net should produce v given a,b at modulator h.
 * Returns the squared error.
 */
static double booleanTest(Net *net,double h,int a,int b,double v){
    double in[2];
    in[0] = a;
    in[1] = b;
    net->setH(h);
    double out = *net->run(in);
    BOOST_TEST_MESSAGE("  At " << h << ", " << a << " " << b << 
                       " gives " << out << ", should be " << v);
    return (v-out)*(v-out);
}


    



#endif /* __TEST_HPP */
