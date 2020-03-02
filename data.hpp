/**
 * @file data.h
 * @brief  Contains formats for example data
 *
 */

#ifndef __DATA_H
#define __DATA_H

#include <assert.h>
#include <stdint.h>

/**
 * \brief Ensure array has alternating values of boolean predicate.
 * Given an array, this function will rearrange the values to ensure that
 * the boolean predicate passed in has values true and false alternately.
 * This is done in-place.
 */

template <class T,class TestFunc> void alternate(T *arr,int n,TestFunc f){
    // for each item, if it is not the appropriate value,
    // scan forward until we find one which is and swap with that.
    // Leave if we can't find one.
    for(int i=0;i<n;i++){
        if(f(arr+i)!=(i%2==0)){
            // doesn't match; swap.
            for(int j=i;;j++){
                if(j>n)return; // can't find a match, exit.
                // scan for one that does
                if(f(arr+j)==(i%2==0)){
                    // and swap and leave loop
                    T v=arr[i];
                    arr[i]=arr[j];
                    arr[j]=v;
                    break;
                }
            }
        }
    }
}


/**
 * \brief
 * A set of example data. Each datum consists of 
 * hormone (i.e. modulator value), inputs and outputs.
 */

class ExampleSet {
    
    /**
     * \brief An example consists of three integer offsets into a large double array
     * of raw data.
     */
    
    struct Example {
        uint32_t ins; //!< offset to start of inputs
        uint32_t outs; //!< offset to start of outputs
        uint32_t h; //!< offset to hormone (modulator)
    };
        
    Example *x; //!< pointer to array of examples
    double *data; //!< pointer to block of floats containing all example data
    
    int ninputs; //!< number of inputs 
    int noutputs; //!< number of outputs
    int ct; //!< number of examples
    
public:
    
    
    /**
     * \brief
     * Constructor - creates but doesn't fill in the data
     * and offset arrays
     * \param n    number of examples
     * \param nin  number of inputs to each example
     * \param nout number of outputs from each example
     */
    ExampleSet(int n,int nin,int nout){
        ninputs=nin;
        noutputs=nout;
        ct=n;
        
        printf("Allocating new set %d*(%d,%d)\n",
               n,ninputs,noutputs);
        
        // size of a single example: number of inputs plus number of outputs
        // plus one for the modulator.
        
        uint32_t exampleSize = nin+nout+1;
        
        data = new double[exampleSize*ct]; // allocate data
        x = new Example[ct]; // allocate offset structures
        
        for(int i=0;i<ct;i++){
            // fill in the example pointers
            x[i].ins = exampleSize*i;
            x[i].outs = x[i].ins+ninputs;
            x[i].h = x[i].outs+noutputs;
        }
    }
    
    /**
     * \brief
     * Destructor - deletes data and offset array
     */
    
    ~ExampleSet(){
        delete [] x;
        delete [] data;
    }
    
    /**
     * \brief
     * Shuffle the example using a PRNG and a Fisher-Yates shuffle.
     * \param rd  pointer to a PRNG data block
     * \param preserveHAlternation if true, the data is fixed after the shuffle to preserve
     * strict alternation between h=0 and h=1.
     */
    
    void shuffle(drand48_data *rd,bool preserveHAlternation){
        Example tmp; // makes a copy of the structures
        for(int i=ct-1;i>=1;i--){
            long lr;
            lrand48_r(rd,&lr);
            int j = lr%(i+1);
            tmp=x[i];
            x[i]=x[j];
            x[j]=tmp;
        }
        // if this flat is set, rearrange the shuffled data so that they go in the sequence
        // h<0.5, h>=0.5, h>0.5 etc.
        if(preserveHAlternation){
            alternate<Example>(x, ct, [](Example *e){return e->h<0.5;});
        }
    }
    
    /**
     * \brief get the number of inputs in all examples
     * \return number of inputs into each example
     */
    int getInputCount(){
        return ninputs;
    }
    
    /**
     * \brief get the number of outputs in all examples
     * \return number of outputs from each example
     */
    int getOutputCount(){
        return noutputs;
    }
    
    /**
     * \brief get the number of examples
     * \return number of examples
     */
    int getCount(){
        return ct;
    }
    
    
    /**
     * \brief
     * Get a pointer to the inputs for a given example, for reading or writing
     * \param example   index of the example
     */
    
    double *getInputs(int example){
        assert(example<ct);
        return data+x[example].ins;
    }
    
    /**
     * \brief
     * Get a pointer to the outputs for a given example, for reading or writing
     * \param example   index of the example
     */
    
    double *getOutputs(int example){
        assert(example<ct);
        return data+x[example].outs;
    }
    
    /**
     * \brief
     * Get the h (modulator) for a given example
     * \param example   index of the example
     */
    double getH(int example){
        assert(example<ct);
        return data[x[example].h];
    }
    
    /**
     * \brief
     * Set the h (modulator) for a given example
     * \param example   index of the example
     * \param h         modulator to use
     */
    void setH(int example, double h){
        assert(example<ct);
        data[x[example].h]=h;
    }
        
};



#endif /* __DATA_H */
