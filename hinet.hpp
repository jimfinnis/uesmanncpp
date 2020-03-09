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

class HInputNet : public Net {
    /**
     * \brief the underlying plain backprop net, which has to be allocated
     * on the heap so that we can manipulate the layer count before initialisation.
     */
    
    Net *bpnet; 
    
    /**
     * \brief The current modulator value, which is sent to the last input
     * when we train/run the network
     */
    double modulator;

public:
    /**
     * \brief Constructor -  does not initialise the weights to random values so
     * that we can reinitialise networks.
     * \param nlayers number of layers
     * \param layerCounts array of layer counts
     */
    HInputNet(int layers,const int *layerCounts) : Net() {
        // you may have noticed that I tend to use arrays a lot rather than
        // std::vector. Sorry, I do this without realising because I'm very,
        // very old.
        
        int *ll = new int[layers];
        for(int i=0;i<layers;i++){
            ll[i] = layerCounts[i]; // copy the layers array
        }
        ll[0]++; // add an extra input
        
        bpnet = new BPNet(layers,ll);
    }
    
    /**
     * \brief destructor, to delete the underlying net
     */
    virtual ~HInputNet(){
        delete bpnet;
    }
    
    virtual int getLayerSize(int n) const {
        int ct = bpnet->getLayerSize(n);
        // subtract one if it's the input layer, so we
        // don't see the hidden input.
        return (n==0)?ct-1:ct;
    }
    
    virtual int getLayerCount() const {
        return bpnet->getLayerCount();
    }
    
    virtual void setH(double h){
        modulator = h;
    }
    
    virtual double getH() const {
        return modulator;
    }
    
    virtual double *getOutputs() const {
        return bpnet->getOutputs();
    }
    
    virtual void setInputs(double *d) {
        // this is a little ugly - to do this, we can't use setInputs in
        // the underlying net because it will also try to set the extra h
        // input, which would be out of range. We have to set the inputs
        // (actually the outputs of layer 0) by hand.
        
        // cast here; we need access to the BPNet proper for setInput.
        BPNet *bpn = static_cast<BPNet *>(bpnet);
        
        // getInputCount() on this class will return the number of 
        // real inputs: that's the net's inputs, -1 for the h input.
        int nins = getInputCount();
        for(int i=0;i<nins;i++){
            bpn->setInput(i,*d++);
        }
        
        // now set the final input
        bpn->setInput(nins,modulator);
    }
    
    virtual int getDataSize() const {
        return bpnet->getDataSize();
    }
    
    virtual void save(double *buf) const {
        bpnet->save(buf);
    }
    
    virtual void load(double *buf){
        bpnet->load(buf);
    }
    
protected:
    
    virtual void update() {
        bpnet->update();
    }
    
    virtual void initWeights(double initr){
        bpnet->initWeights(initr);
    }
    
    virtual double trainBatch(ExampleSet& ex,int start,int num,double eta) {
        bpnet->trainBatch(ex,start,num,eta);
    }
};
    
    
    
#endif /* __HINET_HPP */
