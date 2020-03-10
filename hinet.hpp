/**
 * @file hinet.hpp
 * @brief h-as-input network - only 
 *
 */

#ifndef __HINET_HPP
#define __HINET_HPP

#include "data.hpp"

/**
 * \brief A modulatory network architecture which uses a plain backprop network
 * with an extra input to carry the modulator.
 */

class HInputNet : public BPNet {
    /**
     * \brief The current modulator value, which is sent to the last input
     * when we train/run the network
     */
    double modulator;

public:
    /**
     * \brief Constructor -  does not initialise the weights to random values so
     * that we can reinitialise networks. Uses the non-initialising constructor
     * BPNet::BPNet(), changes the layer count and initialises.
     * \param nlayers number of layers
     * \param layerCounts array of layer counts
     */
    HInputNet(int layers,const int *layerCounts) : BPNet() {
        // you may have noticed that I tend to use arrays a lot rather than
        // std::vector. Sorry, I do this without realising because I'm very,
        // very old.
        
        int *ll = new int[layers];
        for(int i=0;i<layers;i++){
            ll[i] = layerCounts[i]; // copy the layers array
        }
        ll[0]++; // add an extra input
        
        init(layers,ll);
    }
    
    /**
     * \brief destructor
     */
    virtual ~HInputNet(){
    }
    
    virtual int getLayerSize(int n) const {
        int ct = layerSizes[0];
        // subtract one if it's the input layer, so we
        // don't see the hidden input.
        return (n==0)?ct-1:ct;
    }
    
    virtual void setH(double h){
        modulator = h;
    }
    
    virtual double getH() const {
        return modulator;
    }
    
    virtual void setInputs(double *d) {
        // get the number of input which are not the modulator input
        int nins = layerSizes[0]-1;
        for(int i=0;i<nins;i++){
            setInput(i,*d++); // set manually
        }
        
        // now set the final input
        setInput(nins,modulator);
    }
};
    
    
    
#endif /* __HINET_HPP */
