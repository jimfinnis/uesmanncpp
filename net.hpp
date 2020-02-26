/**
 * @file net.hpp
 * @brief  This is the abstract basic network class - the training
 * methods are in each subclass.
 */

#ifndef __NET_HPP
#define __NET_HPP

#include <math.h>

inline double sigmoid(double x){
    return 1.0/(1.0+exp(-x));
}

inline double sigmoidDiff(double x){
    double s = sigmoid(x);
    return (1.0-s)*s;
}

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
    
protected:
    
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
    
    
    
};    
    
    
#endif /* __NET_HPP */
