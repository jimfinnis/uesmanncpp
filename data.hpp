/**
 * @file data.hpp
 * @brief  Contains formats for example data
 *
 */

#ifndef __DATA_H
#define __DATA_H

#include <assert.h>
#include <stdint.h>

#include "mnist.hpp"

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
                if(j>=n)return; // can't find a match, exit.
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
 * The data is stored as a single double array, with each example made up
 * of inputs, followed by outputs, followed by modulator value (h).
 */

class ExampleSet {
    double **examples; //!< pointers to each example, stored as inputs, then outputs, then h.
    double *data; //!< pointer to block of floats containing all example data
    
    int ninputs; //!< number of inputs 
    int noutputs; //!< number of outputs
    int ct; //!< number of examples
    
    uint32_t outputOffset; //!< offset of outputs in example data
    uint32_t hOffset; //!< offset of h in example data
    
    /**
     * \brief Does this set own its data?
     * A little bit of a hack. An example set can be constructed as part of another
     * set, in which case it shouldn't delete its memory. This is used in constructing
     * cross-validation sets. If an example set is created in such a way, this
     * should be false.
     */
    bool ownsData;
    
public:
    
    
    /**
     * \brief
     * Constructor - creates but doesn't fill in the data
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
        
        uint32_t exampleSize = ninputs+noutputs+1;
        
        // calculate the offsets
        outputOffset = ninputs;
        hOffset = ninputs+noutputs;
        
        data = new double[exampleSize*ct]; // allocate data
        examples = new double*[ct]; // allocate example pointers
        
        for(int i=0;i<ct;i++){
            // work out and store the example pointer
            examples[i] = data+i*exampleSize;
            
        }
        ownsData = true;
    }
    
    /**
     * \brief Constructor for making a subset of another set.
     * This uses the actual data in the parent, but creates a fresh
     * set of offset structures which can be independently shuffled.
     * \param parent the set which holds our data.
     * \param start the start index of the data in the parent.
     * \param length the length of the subset.
     */
    ExampleSet(const ExampleSet &parent,int start,int length){
        if(length > parent.ct - start || start<0 || length<1)
            throw std::out_of_range("subset out of range");
        ownsData = false;
        ninputs = parent.ninputs;
        noutputs = parent.noutputs;
        outputOffset = ninputs;
        hOffset = ninputs+noutputs;
        data = parent.data;
        examples = new double*[length];
        ct = length;
        
        for(int i=0;i<ct;i++){
            examples[i] = parent.examples[start+i];
        }
    }
    
    /**
     * \brief Special constructor for generating a data set
     * from an MNIST database with a single labelling (i.e.
     * for use in non-modulatory training). We copy the data
     * from the MNIST object. The outputs will use a one-hot encoding.
     */
    ExampleSet(const MNIST& mnist) : ExampleSet(
                                                mnist.getCount(), // number of examples
                                                mnist.r()*mnist.c(), // input count
                                                mnist.getMaxLabel()+1 // output count
                                                ){
        // fill in the data
        for(int i=0;i<ct;i++){
            // convert each pixel into a 0-1 double and store
            uint8_t *imgpix = mnist.getImg(i);
            double *inpix = getInputs(i);
            for(int i=0;i<ninputs;i++){
                double pixval = *imgpix++;
                pixval /= 255.0;
                *inpix++ = pixval; 
            }
            // fill in the one-hot encoded output
            double *out = getOutputs(i);
            for(int outIdx=0;outIdx<noutputs;outIdx++){
                out[outIdx] = mnist.getLabel(i)==outIdx?1:0;
            }
        }
        ownsData=true;
    }
    
    /**
     * \brief
     * Destructor - deletes data and offset array
     */
    
    ~ExampleSet(){
        if(ownsData){ // only delete the data if we aren't a subset
            delete [] data;
        }
    }
    
public:
    
    /**
     * \brief
     * Shuffle the example using a PRNG and a Fisher-Yates shuffle.
     * \param rd  pointer to a PRNG data block
     * \param preserveHAlternation if true, the data is fixed after the shuffle to preserve
     * strict alternation between h=0 and h=1.
     */
    
    void shuffle(drand48_data *rd,bool preserveHAlternation){
        double *tmp; // makes a copy of the structures
        for(int i=ct-1;i>=1;i--){
            long lr;
            lrand48_r(rd,&lr);
            int j = lr%(i+1);
            tmp=examples[i];
            examples[i]=examples[j];
            examples[j]=tmp;
        }
        // if this flat is set, rearrange the shuffled data so that they go in the sequence
        // h<0.5, h>=0.5, h>0.5 etc.
        if(preserveHAlternation){
            alternate<double*>(examples, ct, 
                               // abominations like this are why I used an overcomplicated
                               // example system at first...
                               [this](double **e){return (*e)[hOffset]<0.5;});
        }
    }
    
    /**
     * \brief get the number of inputs in all examples
     * \return number of inputs into each example
     */
    int getInputCount() const {
        return ninputs;
    }
    
    /**
     * \brief get the number of outputs in all examples
     * \return number of outputs from each example
     */
    int getOutputCount() const {
        return noutputs;
    }
    
    /**
     * \brief get the number of examples
     * \return number of examples
     */
    int getCount() const {
        return ct;
    }
    
    
    /**
     * \brief
     * Get a pointer to the inputs for a given example, for reading or writing
     * \param example   index of the example
     */
    
    double *getInputs(int example) {
        assert(example<ct);
        return examples[example]; // inputs are first in each block
    }
    
    /**
     * \brief
     * Get a pointer to the outputs for a given example, for reading or writing
     * \param example   index of the example
     */
    
    double *getOutputs(int example) {
        assert(example<ct);
        return examples[example] + outputOffset;
    }
    
    /**
     * \brief
     * Get the h (modulator) for a given example
     * \param example   index of the example
     */
    double getH(int example) const {
        assert(example<ct);
        return *(examples[example] + hOffset);
    }
    
    /**
     * \brief
     * Set the h (modulator) for a given example
     * \param example   index of the example
     * \param h         modulator to use
     */
    void setH(int example, double h){
        assert(example<ct);
        *(examples[example] + hOffset) = h;
    }
    
};



#endif /* __DATA_H */
