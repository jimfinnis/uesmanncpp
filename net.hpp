/**
 * @file net.hpp
 * @brief  This is the abstract basic network class - the training
 * methods are in each subclass.
 */


#ifndef __NET_HPP
#define __NET_HPP

#include <math.h>

#include "data.h"

/**
 * Logistic sigmoid function, which is our activation function
 */

inline double sigmoid(double x){
    return 1.0/(1.0+exp(-x));
}

/**
 * The derivative of the sigmoid function
 */

inline double sigmoidDiff(double x){
    double s = sigmoid(x);
    return (1.0-s)*s;
}

/**
 * \brief 
 * The abstract network type upon which all others are based.
 * It's not pure virtual, in that it encapsulates some high
 * level operations (such as the top-level training algorithm).
 */
class Net {
protected:
    
    double eta; //!< learning rate
public:
    
    /**
     * \brief get the learning rate
     */
    
    double getEta() {
        return eta;
    }
    
    drand48_data rd; //!< PRNG data (thread safe)
    
    /**
     * \brief Set the inputs to the network before running or training
     * \param d array of doubles, the size of the input layer
     */
    
    virtual void setInputs(double *d) = 0;
    
    /**
     * \brief Get the outputs after running
     * \return pointer to the output layer outputs
     */
    
    virtual double *getOutputs() = 0;
    
    /**
     * \brief Run the network on some data.
     * \param in pointer to the input double array
     * \return pointer to the output double array
     */
    double *run(double *in){
        setInputs(in);
        update();
        return getOutputs();
    }
    
    /**
     * Train using stochastic gradient descent.
     * Note that cross-validation parameters are slightly different from those
     * given in the thesis. Here we give the number of slices and number of examples
     * per slice; in the thesis we give the total number of examples to be held out
     * and the number of slices.
     * \pre Network has weights initialised to random values
     * \throws std::out_of_range Too many CV examples
     * 
     * @param training training set (including cross-validation data)
     * @param iterations number of training iterations (pair-presentations for UESMANN,
     * h-as-input and output blending)
     * @param nSlices number of cross-val slices
     * @param nPerSlice number of examples in each slice
     * @param nCV cross-validation interval (1 means CV every for every training example)
     * @param initrange range of initial weights/biases [-n,n], or -1 for Bishop's rule.
     * @param selectBestWithCV if true, use the minimum CV error to find the best net,
     * otherwise use the training error. Note that if true, networks will only be tested
     * when the cross-validation runs.
     */
    
    void trainSGD(ExampleSet *training,
                  int nSlices,
                  int nPerSlice,
                  int nCV,
                  double initrange=-1,
                  bool selectBestWithCV=false
                  ){
        // separate out the training examples from the cross-validation samples
        int nCV = nSlices*nPerSlice;
        // it's an error if there are too many CV examples
        if(nCV>=training->getCount())
            throw std::out_of_range("Too many cross-validation examples");
        // get the number of actual training examples
        int nTraining = training->getCount() - nCV;
        
        // initialise the network
        initWeights(initrange);
        
        // initialise minimum error to rogue value
        double minError = -1;
        
        // ah, now it gets awkward; we need to shuffle the examples on each
        // iteration but for UESMANN they really need to be h=0/1 alternating.
        // In the thesis this was done for everything but the robot work, because
        // examples were stored as h=0/1 pairs. For the robot work, the data was
        // generated by a simple controller in a separate program which switched
        // between h=0 and h=1 every now and then. These were shuffled and used
        // as is, so the data is not alternating. In fact, it sort of can't be
        // with that data because there might be more of one h level than the other.
        // Need to give this some thought. It doesn't invalidate the work in the PhD
        // although this is one of those cases where I could have been clearer!
        
        shuffle data
        
        
        for(int i=0;i<iterations;i++){
            // find the example number
            int exampleIndex = i % nTraining;
            // get the inputs and outputs for this example
            double *in = training->getInputs(exampleIndex);
            double *out = training->getOutputs(exampleIndex);
            
            trainBatch(...)
              }
        
    }
    
protected:
    
    /**
     * \brief Run a single update of the network
     * \pre input layer must be filled with values
     * \post output layer contains result
     */
    
    virtual void update() = 0;
    
    
    /**
     * \brief Constructor - protected because others inherit it and it's not used
     * directly
     * \param _eta the learning rate
     */
    Net(double _eta){
        eta = _eta;
    }
    
    /**
     * \brief get a random number using this net's PRNG data
     * \param mn minimum value (inclusive)
     * \param mx maximum value (inclusive)
     */
    
    inline double drand(double mn,double mx){
        double res;
        drand48_r(&rd,&res);
        return res*(mx-mn)+mn;
    }
    
    /**
     * \brief initialise weights to random values
     * \param initr range of weights [-n,n], or -1 for Bishop's rule.
     */
    
    virtual void initWeights(double initr) = 0;
    
    /**
     * \brief
     * Train a network for batch (or mini-batch) (or single example).
     * 
     * This will 
     * - zero the average gradient variables for all weights and biases
     * - zero the total error
     * - for each example
     *    - calculate the error with calcError() which itself calls update()
     *    - add to the total mean squared error (see NOTE below)
     * - for each weight and bias
     *    - calculate the means across all provided examples
     *    - apply the mean to the weight or bias
     * - return the total of the mean squared errors (NOTE: different from original, which returned
     *   mean absolute error) for each output. This is for all examples:
     * \f[
     * \sum_{e \in Examples} \sum_{i=0}^{N_{outs}} (e_o(i) - e_y(i))^2
     * \f]
     * where \f$e_o(i)\f$ is network's output \f$i\f$ for example \f$e\f$ and \f$e_y(i)\f$ is the desired output
     * for the same example.
     * 
     * 
     * \param num     number of examples. For a single example, you'd just use 1.
     * \param in      for an array of pointers, one for each example. Each points to an array
     *                of doubles which constitute the inputs. 
     * \param out     an array of pointers to doubles to write the output layer on completion.
     * \return        the sum of mean squared errors in the output layer (see formula in method documentation)
     */
    
    double trainBatch(int num,double **in, double **out) = 0;
    
};    


#endif /* __NET_HPP */
