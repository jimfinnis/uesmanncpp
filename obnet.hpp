/**
 * @file obnet.hpp
 * @brief Output blending network - only works with 2 h-levels, 0 and 1.
 * \bug more doc required
 *
 */

#ifndef __OBNET_HPP
#define __OBNET_HPP

#include "data.hpp"


class OutputBlendingNet : public Net {
private:
    /**
     * \brief the modulator (or h)
     */
    double modulator;
    
    
public:
    /**
     * \brief Constructor -  does not initialise the weights to random values so
     * that we can reinitialise networks.
     * \param nlayers number of layers
     * \param layerCounts array of layer counts
     */
    OutputBlendingNet(int nlayers,const int *layerCounts) : Net() {
        // we create two networks, one for each modulator level.
        net0 = new BPNet(nlayers,layerCounts);
        net1 = new BPNet(nlayers,layerCounts);
        interpolatedOutputs = new double [net0->getOutputCount()];
    }
    
    /**
     * \brief destructor to delete subnets and outputs
     */
    virtual ~OutputBlendingNet(){
        delete net0;
        delete net1;
        delete [] interpolatedOutputs;
    }
    
    virtual int getLayerSize(int n) {
        return net0->getLayerSize(n);
    }
    
    virtual int getLayerCount(){
        return net0->getLayerCount();
    }
    
    virtual void setH(double h){
        modulator = h;
    }
    
    virtual double getH(){
        return modulator;
    }
    
    
    
    virtual void setInputs(double *d) {
        // a bit inefficient, since we should only need to do this 
        // for the network currently being trained.
        net0->setInputs(d);
        net1->setInputs(d);
    }
    
    virtual double *getOutputs() const {
        // constructed during the update
        return interpolatedOutputs;
    }
    
    virtual int getDataSize() const {
        // need room for the two (equally-sized) nets
        return net0->getDataSize()*2;
    }
    
    virtual void save(double *buf) const {
        // just save the two networks, one after the other
        net0->save(buf);
        buf+=net0->getDataSize();
        net1->save(buf);
    }
    
    virtual void load(double *buf){
        net0->load(buf);
        buf+=net0->getDataSize();
        net1->load(buf);
    }
    
protected:
    
    Net *net0; //!< the network trained by h=0 examples
    Net *net1; //!< the network trained by h=1 examples
    double *interpolatedOutputs;
    
    virtual void initWeights(double initr){
        net0->initWeights(initr);
        net1->initWeights(initr);
    }
    
    virtual void update(){
        net0->update();
        net1->update();
        
        // interpolate the outputs
        double *o0 = net0->getOutputs();
        double *o1 = net1->getOutputs();
        double h = getH();
        for(int i=0;i<getOutputCount();i++){
            interpolatedOutputs[i] = h*o1[i] + (1.0-h)*o0[i];
        }
    }
    
    double lastError = -1;
    
    virtual double trainBatch(ExampleSet& ex,int start,int num,double eta){
        /** \bug can only use SGD for now; how this works in batching
           could be tricky. */
        if(num!=1)
            std::runtime_error("num!=1 (i.e. batch training) not implemented");
        
        // what we do here depends on the modulator for the first and only
        // example
        double hzero = (ex.getH(start)<0.5);
        Net *net = hzero ? net0 : net1;
        
        double e = net->trainBatch(ex,start,1,eta);
        // return avg of 0/1 error rate, so this will change once every two cycles;
        // but the first one will just be the error for h=0
        double rv;
        if(lastError<0){
            // nothing yet, we've done one h=0, return it.
            lastError=e;
            rv=e;
        } else if(hzero) {
            // this is the h=0 the second and subsequent times; return the last mean.
            rv=lastError;
        } else {
            // this is the h=1 - calculate the new mean and return it. Set this to
            // also be the value that will be returned on the next h=0 run.
            lastError = rv = (e+lastError)*0.5;
        }
        return rv;
    }
};    
    


#endif /* __OBNET_HPP */
