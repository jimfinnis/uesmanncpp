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
     * Destructor - deletes data and offset array
     */
    
    ~ExampleSet(){
        delete [] x;
        delete [] data;
    }
    
    /**
     * Shuffle the example using a PRNG and a Fisher-Yates shuffle
     * \param rd  pointer to a PRNG data block
     */
    
    void shuffle(drand48_data *rd){
        Example tmp; // makes a copy of the structures
        for(int i=ct-1;i>=1;i--){
            long lr;
            lrand48_r(rd,&lr);
            int j = lr%(i+1);
            tmp=x[i];
            x[i]=x[j];
            x[j]=tmp;
        }
    }
    
    /**
     * \return number of inputs into each example
     */
    int getInputCount(){
        return ninputs;
    }
    
    /**
     * \return number of outputs from each example
     */
    int getOutputCount(){
        return noutputs;
    }
    
    /**
     * \return number of examples
     */
    int getCount(){
        return ct;
    }
    
    
    /**
     * Get a pointer to the inputs for a given example, for reading or writing
     * \param example   index of the example
     */
    
    double *getInputs(int example){
        assert(example<ct);
        return data+x[example].ins;
    }
    
    /**
     * Get a pointer to the outputs for a given example, for reading or writing
     * \param example   index of the example
     */
    
    double *getOutputs(int example){
        assert(example<ct);
        return data+x[example].outs;
    }
    
    /**
     * Get the h (modulator) for a given example
     * \param example   index of the example
     */
    double getH(int example){
        assert(example<ct);
        return data[x[example].h];
    }
    
    /**
     * Set the h (modulator) for a given example
     * \param example   index of the example
     * \param h         modulator to use
     */
    void setH(int example, double v){
        assert(example<ct);
        data[x[example].h]=v;
    }
        
};



#endif /* __DATA_H */
