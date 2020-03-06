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
 * \brief Ensure array has cycling values of some function f mod n.
 * Given an array, this function will rearrange the values to ensure that
 * the integer function passed in has values which cycle. For example, if
 * a cycle length of four is specified, the values will be made to run 0,1,2,3,0,1,2,3.
 * This is done in-place.
 * 
 * The input function has the signature (int)(T). In the shuffling code we use it
 * takes a pointer to the data of the example.
 */

template <class T,class TestFunc> void alternate(T *arr,int nitems,int cycle,TestFunc f){
    // for each item, if it is not the appropriate value,
    // scan forward until we find one which is and swap with that.
    // Leave if we can't find one.
    for(int i=0;i<nitems;i++){
        if(f(arr[i])%cycle!=(i%cycle)){
            // doesn't match; swap.
            for(int j=i;;j++){
                if(j>=nitems)return; // can't find a match, exit.
                // scan for one that does
                if(f(arr[j])%cycle==i%cycle){
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
    
    /**
     * \brief If there are discrete modulator levels, this is how many there
     * are - if not, it should be 1.
     */
    int numHLevels;
    
    /**
     * \brief minimum H level, 0 by default, set with setHRange()
     */
    double minH;
    /**
     * \brief maximum H level, 1 by default, set with setHRange()
     */
    double maxH;
    
    
public:
    
    
    /**
     * \brief
     * Constructor - creates but doesn't fill in the data
     * \param n    number of examples
     * \param nin  number of inputs to each example
     * \param nout number of outputs from each example
     * \param levels number of modulator levels (see numHLevels)
     */
    ExampleSet(int n,int nin,int nout,int levels){
        ninputs=nin;
        noutputs=nout;
        ct=n;
        numHLevels = levels;
        minH=0;
        maxH=1;
        
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
        numHLevels = parent.numHLevels;
        minH = parent.minH;
        maxH = parent.maxH;
        
        for(int i=0;i<ct;i++){
            examples[i] = parent.examples[start+i];
        }
    }
    
    /**
     * \brief Special constructor for generating a data set
     * from an MNIST database with a single labelling (i.e.
     * for use in non-modulatory training). We copy the data
     * from the MNIST object. The outputs will use a one-hot encoding.
     * This example set will have no modulation.
     */
    ExampleSet(const MNIST& mnist) : ExampleSet(
                                                mnist.getCount(), // number of examples
                                                mnist.r()*mnist.c(), // input count
                                                mnist.getMaxLabel()+1, // output count
                                                1 // single modulation level
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
            setH(i,0); // set nominal modulator value
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
     * \brief Shuffling mode for shuffle()
     */
    enum ShuffleMode { 
        /**
         * \brief Shuffle blocks of numHLevels examples, rather than single examples.
         * This is intended for cases where examples with the same inputs are added contiguously
         * at different modulator levels. 
         * For this to work correctly, the modulator levels must be distributed evenly
         * across their range. For example, for four modulator levels from 2-3:
         * 
         * * ensure that numHLevels is 4 
         * * ensure that the values for 2,2.25,2.5 and 3 are equally represented in the data.
         * * ensure that the data is provided in equally sized groups cycling through the
         * modulator (similar to the output of the ALTERNATE mode)
         * 
         * It is possible to run a shuffle(rd,ALTERNATE) on the data after input, followed
         * by training with this mode.
         */
        STRIDE,
              /**
               * \brief Shuffle single examples, but follow up by running a pass over the examples
               * to ensure that they alternate by modulator level. This is useful where there are 
               * discrete modulator levels but the examples are mixed
               * up (as happens in the robot experiments). This doesn't require equal distribution
               * of modulator levels, but the levels should be evenly spaced across the range.
               * If the distribution is unequal, a portion at the end of the set will not alternate
               * correctly.
               */
              ALTERNATE,
              /**
               * \brief Shuffle single examples, no matter the value of numHLevels.
               */
              NONE
    };
        
    
    /**
     * \brief
     * Shuffle the example using a PRNG and a Fisher-Yates shuffle.
     * \param rd  pointer to a PRNG data block
     * \param mode ShuffleMode::STRIDE to keep blocks of size numHLevels together, ShuffleMode::ALTERNATE to
     * shuffle all examples but ensure that h-levels alternate after shuffling, or ShuffleMode::NONE to just shuffle.
     */
    
    void shuffle(drand48_data *rd,ShuffleMode mode){
        int blockSize; // size of the blocks we are shuffling, in bytes
        if(mode == STRIDE)
            blockSize = numHLevels;
        else
            blockSize = 1;
        double **tmp = new double*[blockSize]; // temporary storage for swapping
        for(int i=(ct/blockSize)-1;i>=1;i--){
            long lr;
            lrand48_r(rd,&lr);
            int j = lr%(i+1);
            memcpy(tmp,examples+i*blockSize,blockSize*sizeof(double*));
            memcpy(examples+i*blockSize,examples+j*blockSize,blockSize*sizeof(double*));
            memcpy(examples+j*blockSize,tmp,blockSize*sizeof(double*));
        }
        // if this mode is set, rearrange the shuffled data so that the h-levels cycle
        if(mode == ALTERNATE){
            alternate<double*>(examples, ct, numHLevels,
                               // abominations like this are why I used an overcomplicated
                               // example system at first...
                               [this](double *e){
                               double d = (e[hOffset]-minH)/(maxH-minH);
                               int i = (int)(d*(numHLevels-1));
                               return i;
                           });
        }
        delete [] tmp;
    }
    
    /**
     * Modify the min/max h range, which is 0<=h<=1 by default.
     * \param mn minimum H value in set domain
     * \param mx maximum H value in set domain
     */
    ExampleSet& setHRange(double mn,double mx){
        minH = mn;
        maxH = mx;
        return *this;
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
     * \brief return the number of different H-levels
     */
    int getNumHLevels(){
        return numHLevels;
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
