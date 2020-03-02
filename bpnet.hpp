/**
 * @file bpnet.hpp
 * @brief This implements a plain backprop network
 *
 */

#ifndef __BPNET_HPP
#define __BPNET_HPP

#include "net.hpp"

/**
 * \brief The "basic" back-propagation network using a logistic sigmoid,
 * as described by Rumelhart, Hinton and Williams (and many others).
 * This class is used by output blending and h-as-input networks.
 */

class BPNet : public Net {
public:
    /**
     * \brief Constructor
     * \param _eta the learning rate
     * \param nlayers number of layers
     * \param layerCounts array of layer counts
     */
    BPNet(double _eta,int nlayers,const int *layerCounts) : Net(eta) {
        init(nlayers,layerCounts);
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
    
    virtual void setInputs(double *d) {
        for(int i=0;i<layerSizes[0];i++)
            outputs[0][i]=d[i];
    }
    
    /**
     * \brief Get the outputs after running
     * \return pointer to the output layer outputs
     */
    
    virtual double *getOutputs() const {
        return outputs[numLayers-1];
    }
    
    /**
     * \brief Get the length of the serialised data block
     * for this network.
     * \return the size in bytes
     */
    virtual int getDataSize() const {
        // number of weights+biases for each layer is
        // the number of nodes in that layer (bias count)
        // times the number of nodes in the previous layer.
        int pc=0;
        int total=0;
        for(int i=0;i<numLayers;i++){
            int c = layerSizes[i];
            total += c*(1+pc);
            pc = c;
        }
        return total;
    }
    
    /**
     * \brief Serialize the data (not including any network type magic number or
     * layer/node counts) to the given memory (which must be of sufficient size).
     * \param buf the buffer to save the data, must be at least getDataSize() bytes
     */
    virtual void save(double *buf) const {
        double *g=buf;
        // data is ordered by layers, with nodes within
        // layers, and each node is bias then weights.
        for(int i=0;i<numLayers;i++){
            for(int j=0;j<layerSizes[i];j++){
                *g++ = biases[i][j];
                if(i){
                    for(int k=0;k<layerSizes[i-1];k++){
                        *g++ = getw(i,j,k);
                    }
                }
            }
        }
    }
    
    /**
     * \brief Given that the pointer points to a data block of the correct size
     * for the current network, copy the parameters from that data block into
     * the current network overwriting the current parameters.
     * \param buf the buffer to load the data from, must be at least getDataSize() bytes
     */
    virtual void load(double *buf){
        double *g=buf;
        // genome is ordered by layers, with nodes within
        // layers, and each node is bias then weights.
        for(int i=0;i<numLayers;i++){
            for(int j=0;j<layerSizes[i];j++){
                biases[i][j]=*g++;
                if(i){
                    for(int k=0;k<layerSizes[i-1];k++){
                        getw(i,j,k) = *g++;
                    }
                }
            }
        }
    }
    
protected:
    int numLayers; //!< number of layers, including input and output
    int *layerSizes; //!< array of layer sizes
    int largestLayerSize; //!< number of nodes in largest layer
    
    /// \brief Array of weights as [tolayer][tonode+largestLayerSize*fromnode]
    ///
    /// Weights are stored as a square matrix, even though less than
    /// half is used. Less than that, if not all layers are the same
    /// size, since the dimension of the matrix must be the size of
    /// the largest layer. Each array has its own matrix,
    /// so index by [layer][i+largestLayerSize*j], where
    /// - layer is the "TO" layer
    /// - layer-1 is the FROM layer
    /// - i is the TO neuron (i.e. the end of the connection)
    /// - j is the FROM neuron (the start)
    double **weights;
    
    /// array of biases, stored as a rectangular array of [layer][node]
    double **biases;
    
    // data generated during training and running
    
    double **outputs; //!< outputs of each layer: one array of doubles for each
    double **errors; //!< the error for each node, calculated by calcError()
    
    double **gradAvgsWeights; //!< average gradient for each weight (built during training)
    double **gradAvgsBiases; //!< average gradient for each bias (built during training)
    
    
    
    /**
     * \brief Default initialisation routine; other things may override it. 
     * Called from each net class' constructor because non-standard net types
     * do strange things. Does not initialise the weights; this is done as the
     * first step in training.
     * \param nlayers number of layers
     * \param layerCounts array of layer counts
     */
    
    virtual void init(int nlayers,const int *layerCounts){
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
    
    inline double& getw(int tolayer,int toneuron,int fromneuron) const {
        return weights[tolayer][toneuron+largestLayerSize*fromneuron];
    }
    
    /**
     * \brief get the value of a bias
     * \param layer   index of layer
     * \param neuron  index of neuron within layer
     */
    
    inline double& getb(int layer,int neuron) const {
        return biases[layer][neuron];
    }
    
    
    /**
     * \brief get the value of the gradient for a given weight 
     * \pre gradients must have been calculated as part of training step
     * \param tolayer    the layer of the destination node (from is assumed to be previous layer)
     * \param toneuron   the index of the destination node in that layer
     * \param fromneuron the index of the source node
     */
    
    inline double& getavggradw(int tolayer,int toneuron,int fromneuron) const {
        return gradAvgsWeights[tolayer][toneuron+largestLayerSize*fromneuron];
    }
    
    /**
     * \brief get the value of a bias gradient
     * \pre gradients must have been calculated as part of training step
     * \param l  index of layer
     * \param n  index of neuron within layer
     */
    
    inline double getavggradb(int l,int n) const {
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
    
    virtual void update(){
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
    
    double trainBatch(int num,double **in, double **out) {
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
            // build errors for each example
            calcError(in[nn],out[nn]);
            
            // accumulate errors
            for(int l=1;l<numLayers;l++){
                for(int i=0;i<layerSizes[l];i++){
                    for(int j=0;j<layerSizes[l-1];j++)
                        getavggradw(l,i,j) += errors[l][i]*outputs[l-1][j];
                    gradAvgsBiases[l][i] += errors[l][i];
                }
            }
            // count up the total error
            int ol = numLayers-1;
            for(int i=0;i<layerSizes[ol];i++){
                double o = outputs[ol][i];
                double e = (o-out[nn][i]);
                totalError += e*e;
            }
        }
        
        // for calculating average error - 1/number of examples trained
        double factor = 1.0/(double)num;
        // we now have a full set of running averages. Time to apply them.
        for(int l=1;l<numLayers;l++){
            for(int i=0;i<layerSizes[l];i++){
                for(int j=0;j<layerSizes[l-1];j++){
                    double wdelta = eta*getavggradw(l,i,j)*factor;
                    getw(l,i,j) -= wdelta;
                }
                double bdelta = eta*gradAvgsBiases[l][i]*factor;
                biases[l][i] -= bdelta;
            }
        }
        // and return total error - this is the SUM of the MSE of each output
        return totalError*factor;
    }
};


#endif /* __BPNET_HPP */
