/**
 * @file uesnet.hpp
 * @brief This file contains the implementation of the UESMANN network itself - at least, those
 * parts which are different from a standard Rumelhart/Hinton/Williams MLP.
 *
 */

#ifndef __UESNET_HPP
#define __UESNET_HPP


/**
 * \brief The UESMANN network, which it itself based on the BPNet code as it has
 * the same architecture as the plain MLP.
 */

class UESNet: public BPNet {
    /**
     * \brief the modulator value, initially 0
     */
    
    double modulator; 
public:
    /**
     * \brief The constructor is mostly identical to the BPNet constructor
     */
    
    UESNet(int nlayers,const int *layerCounts) : BPNet(nlayers,layerCounts),
          modulator(0)
    {
        // replace the net type, it's not a plain net any more
        type = NetType::UESMANN;
        
    }
    
    virtual void setH(double h){
        modulator = h;
    }
    
    virtual double getH() const {
        return modulator;
    }
    
protected:
    
    void calcError(double *in,double *out){
        // first run the network forwards
        setInputs(in);
        update();
        
        // first, calculate the error in the output layer
        // This does the THIRD of the backprop equations, Eq. 4.15, giving dLj.
        int ol = numLayers-1;
        for(int i=0;i<layerSizes[ol];i++){
            double o = outputs[ol][i];
            errors[ol][i] = o*(1-o)*(o-out[i]);
        }
        
        // then work out the errors in all the other layers
        // factoring in (rather inefficiently) the hormone.
        // This is the FOURTH backprop equation, Eq. 4.16.
        for(int l=1;l<numLayers-1;l++){
            for(int j=0;j<layerSizes[l];j++){
                double e = 0;
                for(int i=0;i<layerSizes[l+1];i++)
                    e += errors[l+1][i]*getw(l+1,i,j);
                
                // produce the \delta^l_i term where l is the layer and i
                // the index of the node. Here is where we factor in the modulator.
                
                errors[l][j] = e * (modulator+1.0) * outputs[l][j] * (1-outputs[l][j]); 
            }
        }
    }
    
    virtual void update(){
        double hfactor = modulator+1.0;
        for(int i=1;i<numLayers;i++){
            for(int j=0;j<layerSizes[i];j++){
                double v = 0.0;
                for(int k=0;k<layerSizes[i-1];k++){
                    v += getw(i,j,k) * outputs[i-1][k];
                }
                // factor in the hormone here
                outputs[i][j]=sigmoid(v*hfactor+biases[i][j]);
            }
        }
    }
    
    virtual double trainBatch(ExampleSet& ex,int start,int num,double eta){
        // zero average gradients
        for(int j=0;j<numLayers;j++){
            for(int k=0;k<layerSizes[j];k++)
                gradAvgsBiases[j][k]=0;
            for(int i=0;i<largestLayerSize*largestLayerSize;i++)
                gradAvgsWeights[j][i]=0;
        }
        
        // reset total error
        double totalError=0;
        // iterate over examples
        for(int nn=0;nn<num;nn++){
            int exampleIndex = nn+start;
            // set modulator
            setH(ex.getH(exampleIndex));
            // get outputs for this example
            double *outs = ex.getOutputs(exampleIndex);
            // build errors for each example
            calcError(ex.getInputs(exampleIndex),outs);
            
            // accumulate errors
            for(int l=1;l<numLayers;l++){
                for(int i=0;i<layerSizes[l];i++){
                    // this does the FIRST of the backprop equations, 
                    // Eq. 4.13, calculating dC/dw(h+1), but the modulator
                    // is dealt with below.
                    for(int j=0;j<layerSizes[l-1];j++)
                        getavggradw(l,i,j) += errors[l][i]*outputs[l-1][j];
                    // this does the SECOND of the backprop equations,
                    // Eq. 4.14.
                    gradAvgsBiases[l][i] += errors[l][i];
                }
            }
            // count up the total error
            int ol = numLayers-1;
            for(int i=0;i<layerSizes[ol];i++){
                double o = outputs[ol][i];
                double e = (o-outs[i]);
                totalError += e*e;
            }
        }
        
        // get modulator factor
        double hfactor = modulator+1.0;
        
        
        // for calculating average error - 1/number of examples trained
        double factor = 1.0/(double)num;
        // we now have a full set of running averages. Time to apply them.
        for(int l=1;l<numLayers;l++){
            for(int i=0;i<layerSizes[l];i++){
                for(int j=0;j<layerSizes[l-1];j++){
                    // this does the modulation part of Eq. 4.13, but a little
                    // later than in the thesis.
                    double wdelta = eta*getavggradw(l,i,j)*factor*hfactor;
                    //                    printf("WCORR: %f factor %f\n",wdelta,getavggradw(l,i,j));
                    getw(l,i,j) -= wdelta;
                }
                // biases are not modulated
                double bdelta = eta*gradAvgsBiases[l][i]*factor;
                biases[l][i] -= bdelta;
            }
        }
        // and return total error - this is the SUM of the MSE of each output
        return totalError*factor;
    }
};

#endif /* __UESNET_HPP */
