/**
 * @file bpnet.hpp
 * @brief This implements a plain backprop network
 *
 */

#ifndef __BPNET_HPP
#define __BPNET_HPP

#include "net.hpp"

class BPNet : public Net {
public:
    /**
     * \brief Constructor with initial range
     * \param _eta the learning rate
     * \param nlayers number of layers
     * \param layerCounts array of layer counts
     * \param initrange initial weight range [-n,n]
     */
    BPNet(double _eta,int nlayers,const int *layerCounts,double initrange) : Net(eta) {
        init(nlayers,layerCounts,initrange);
    }
    
    /**
     * \brief Constructor for Bishop's rule weights
     * \param _eta the learning rate
     * \param nlayers number of layers
     * \param layerCounts array of layer counts
     */
    BPNet(double _eta,int nlayers,const int *layerCounts) : Net(eta) {
        init(nlayers,layerCounts,-1);
    }
    
    /**
     * \brief destructor
     */
    
    virtual ~BPNet(){
        for(int i=0;i<numLayers;i++){
            delete [] weights[i];
            delete [] biases[i];
            delete [] gradAvgsWeights[i];
            delete [] gradAvgsBiases[i];
            delete [] outputs[i];
            delete [] errors[i];
        }
        delete [] weights;
        delete [] biases;
        delete [] gradAvgsWeights;
        delete [] gradAvgsBiases;
        delete [] outputs;
        delete [] errors;
        delete [] layerSizes;
    }
    
    /**
     * \brief Set the inputs to the network before running or training
     * \param d array of doubles, the size of the input layer
     */
    
    virtual void setInputs(double *d){
        for(int i=0;i<layerSizes[0];i++)
            outputs[0][i]=d[i];
    }
    
    /**
     * \brief Get the outputs after running
     * \return pointer to the output layer outputs
     */
    
    virtual double *getOutputs(){
        return outputs[numLayers-1];
    }
    
    
    
protected:
    int numLayers; //!< number of layers, including input and output
    int *layerSizes; //!< array of layer sizes
    int largestLayerSize; //!< number of nodes in largest layer
    
    /// Weights stored as a square matrix, even though less than
    /// half is used. Less than that, if not all layers are the same
    /// size, since the dimension of the matrix must be the size of
    /// the largest layer. Each array has its own matrix,
    /// so index by [layer][i+largestLayerSize*j], where
    /// - layer is the "TO" layer
    /// - layer-1 is the FROM layer
    /// - i is the TO neuron (i.e. the end of the connection)
    /// - j is the FROM neuron (the start)
    double **weights;
    
    double **biases; // node biases [layer][i]
    
    // data generated during training and running
    
    double **outputs; //!< outputs of each layer: one array of doubles for each
    double **errors; //!< the error for each node, calculated by calcError()
    
    double **gradAvgsWeights; //!< average gradient for each weight (built during training)
    double **gradAvgsBiases; //!< average gradient for each bias (built during training)
    
    
    
    /**
     * \brief Default initialisation routine; other things may override it. 
     * Called from each net class' constructor because non-standard net types
     * do strange things.
     * \param nlayers number of layers
     * \param layerCounts array of layer counts
     * \param initrange initial weight range [-n,n], or -ve for Bishop's rule.
     */
    
    virtual void init(int nlayers,const int *layerCounts,double initrange=-1){
        numLayers = nlayers;
        outputs = new double* [numLayers];
        errors = new double* [numLayers];
        layerSizes = new int [numLayers];
        largestLayerSize=0;
        for(int i=0;i<numLayers;i++){
            int n = layerCounts[i];
            outputs[i] = new double[n];
            errors[i] = new double[n];
            for(int k=0;k<n;k++)
                outputs[i][k]=0;
            layerSizes[i]=n;
            if(n>largestLayerSize)
                largestLayerSize=n;
        }
        
        weights = new double * [numLayers];
        gradAvgsWeights = new double* [numLayers];
        biases = new double* [numLayers];
        gradAvgsBiases = new double* [numLayers];
        for(int i=0;i<numLayers;i++){
            int n = layerCounts[i];
            weights[i] = new double[largestLayerSize*largestLayerSize];
            gradAvgsWeights[i] = new double[largestLayerSize*largestLayerSize];
            biases[i] = new double[n];
            gradAvgsBiases[i] = new double[n];
        }
        initWeights(initrange);
    }
    
    /**
     * \brief initialise weights to random values
     * \param initr range of weights [-n,n], or -1 for Bishop's rule.
     */
    
    virtual void initWeights(double initr){
        for(int i=0;i<numLayers;i++){
            double initrange;
            if(i){
                double ct = layerSizes[i-1];
                if(initr>0)
                    initrange = initr;
                else
                    initrange = 1.0/sqrt(ct); // from Bishop
            } else 
                initrange = 0.1; // on input layer, should mean little.
            //        printf("Layer %d - count %d - initrange %f\n",i,layerSizes[i],initrange);
            for(int j=0;j<layerSizes[i];j++)
                biases[i][j]=drand(-initrange,initrange);
            for(int j=0;j<largestLayerSize*largestLayerSize;j++){
                weights[i][j]=drand(-initrange,initrange);
            }
        }
        // zero the input layer weights, which should be unused.
        for(int j=0;j<layerSizes[0];j++)
            biases[0][j]=0;
        for(int j=0;j<largestLayerSize*largestLayerSize;j++)
            weights[0][j]=0;
    }
    
    /**
     * \brief get the value of a weight.
     * \param tolayer    the layer of the destination node (from is assumed to be previous layer)
     * \param toneuron   the index of the destination node in that layer
     * \param fromneuron the index of the source node
     */
    
    inline double& getw(int tolayer,int toneuron,int fromneuron){
        return weights[tolayer][toneuron+largestLayerSize*fromneuron];
    }
    
    /**
     * \brief get the value of a bias
     * \param layer   index of layer
     * \param neuron  index of neuron within layer
     */
    
    inline double& getb(int layer,int neuron){
        return biases[layer][neuron];
    }
    
    
    /**
     * \brief get the value of the gradient for a given weight 
     * \pre gradients must have been calculated as part of training step
     * \param tolayer    the layer of the destination node (from is assumed to be previous layer)
     * \param toneuron   the index of the destination node in that layer
     * \param fromneuron the index of the source node
     */
    
    inline double& getavggradw(int tolayer,int toneuron,int fromneuron){
        return gradAvgsWeights[tolayer][toneuron+largestLayerSize*fromneuron];
    }
    
    /**
     * \brief get the value of a bias gradient
     * \pre gradients must have been calculated as part of training step
     * \param layer   index of layer
     * \param neuron  index of neuron within layer
     */
    
    inline double getavggradb(int l,int n){
        return gradAvgsBiases[l][n];
    }
    
    /**
     * \brief run a single example and calculate the errors; used in training.
     * \param in inputs
     * \param out required outputs
     * \post the errors will be in the errors variable
     */
    
    void calcError(double *in,double *out){
        // perform the actual backprop algorithm
        // first run the network forwards
        setInputs(in);
        update();
        
        // first, calculate the error in the output layer
        int ol = numLayers-1;
        for(int i=0;i<layerSizes[ol];i++){
            double o = outputs[ol][i];
            errors[ol][i] = o*(1-o)*(o-out[i]);
        }
        
        // then work out the errors in all the other layers
        for(int l=1;l<numLayers-1;l++){
            for(int j=0;j<layerSizes[l];j++){
                double e = 0;
                for(int i=0;i<layerSizes[l+1];i++)
                    e += errors[l+1][i]*getw(l+1,i,j);
                
                // produce the \delta^l_i term where l is the layer and i
                // the index of the node
                errors[l][j] = e * outputs[l][j] * (1-outputs[l][j]); 
            }
        }
    }
    
    /**
     * \brief update the network
     * \pre inputs must be set
     * \post the output member will hold the output values
     */
    
    void update(){
        for(int i=1;i<numLayers;i++){
            for(int j=0;j<layerSizes[i];j++){
                double v = biases[i][j];
                for(int k=0;k<layerSizes[i-1];k++){
                    //                printf("Layer %d, Node %d-%d weight*output = %f*%f\n",
                    //                       i,k,j,getw(i,j,k),outputs[i-1][k]);
                    v += getw(i,j,k) * outputs[i-1][k];
                }
                outputs[i][j]=sigmoid(v);
            }
        }
    }
};


#endif /* __BPNET_HPP */
