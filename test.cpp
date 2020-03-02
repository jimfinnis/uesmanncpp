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

/*
int main(int argc,char *argv[]){
    int layers[] = {2,2,1};
    BPNet foo(0.1,3,layers);
    return 0;
}


 */

BOOST_AUTO_TEST_CASE(example) {
    ExampleSet e(10,5,2); // 10 examples, 5 ins, 2 outs
    
    for(int i=0;i<e.getCount();i++){
        double *d = e.getInputs(i);
        for(int j=0;j<e.getInputCount();j++)
            d[j] = i*100+j;
        d= e.getOutputs(i);
        for(int j=0;j<e.getOutputCount();j++)
            d[j] = i*200+j;
        e.setH(i,i*1000);
    }
    
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
