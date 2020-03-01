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
     * Train using stochastic gradient descent.
     * Note that cross-validation parameters are slightly different from those
     * given in the thesis. Here we give the number of slices and number of examples
     * per slice; in the thesis we give the total number of examples to be held out
     * and the number of slices.
     * 
     */
    
    void trainSGD(ExampleSet *training, //!< training set (including cross-validation data
                  int nslices, //!< number of cross-val slices
                  int nperslice //!< number of examples in each slice
                  ){}
                  
};    
    
    
#endif /* __NET_HPP */
