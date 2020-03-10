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
protected:
    /**
     * \brief Special constructor for subclasses which need to manipulate layer
     * count before initialisation (e.g. HInputNet).
     */
    BPNet() {
    }
    
    /**
     * \brief Initialiser for use by the main constructor and the ctors of those
     * subclasses mentioned in BPNet()
     */
    
    void init(int nlayers,const int *layerCounts){
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
        
public:
    /**
     * \brief Constructor -  does not initialise the weights to random values so
     * that we can reinitialise networks.
     * \param nlayers number of layers
     * \param layerCounts array of layer counts
     */
    BPNet(int nlayers,const int *layerCounts) : Net() {
        init(nlayers,layerCounts);
    }
    
    virtual void setH(double h){
        // does nothing, because this is an unmodulated net.
    }
    
    virtual double getH() const {
        return 0;
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
    
    virtual void setInputs(double *d) {
        for(int i=0;i<layerSizes[0];i++){
            outputs[0][i]=d[i];
        }
    }
    
    /**
     * \brief Used to set inputs manually, typically in
     * HInputNet.
     */
    
    void setInput(int n, double d){
        outputs[0][n] = d;
    }
        
    
    virtual double *getOutputs() const {
        return outputs[numLayers-1];
    }
    
    virtual int getLayerSize(int n) const {
        return layerSizes[n];
    }
    
    virtual int getLayerCount() const {
        return numLayers;
    }
    
        
    
    virtual int getDataSize() const {
        // number of weights+biases for each layer is
        // the number of nodes in that layer (bias count)
        // times the number of nodes in the previous layer.
        // 
        // NOTE THAT this uses the true layer size rather than
        // the fake version returned in the subclass HInputNet
        int pc=0;
        int total=0;
        for(int i=0;i<numLayers;i++){
            int c = layerSizes[i];
            total += c*(1+pc);
            pc = c;
        }
        return total;
    }
    
    virtual void save(double *buf) const {
        double *g=buf;
        // data is ordered by layers, with nodes within
        // layers, and each node is bias then weights.
        // 
        // NOTE THAT this uses the true layer size rather than
        // the fake version returned in the subclass HInputNet
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
    
    virtual void load(double *buf){
        double *g=buf;
        // genome is ordered by layers, with nodes within
        // layers, and each node is bias then weights.
        // 
        // NOTE THAT this uses the true layer size rather than
        // the fake version returned in the subclass HInputNet
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
    
    virtual void update(){
        for(int i=1;i<numLayers;i++){
            for(int j=0;j<layerSizes[i];j++){
                double v = biases[i][j];
                for(int k=0;k<layerSizes[i-1];k++){
                    v += getw(i,j,k) * outputs[i-1][k];
                }
                outputs[i][j]=sigmoid(v);
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
                    for(int j=0;j<layerSizes[l-1];j++)
                        getavggradw(l,i,j) += errors[l][i]*outputs[l-1][j];
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
        
        // for calculating average error - 1/number of examples trained
        double factor = 1.0/(double)num;
        // we now have a full set of running averages. Time to apply them.
        for(int l=1;l<numLayers;l++){
            for(int i=0;i<layerSizes[l];i++){
                for(int j=0;j<layerSizes[l-1];j++){
                    double wdelta = eta*getavggradw(l,i,j)*factor;
//                    printf("WCORR: %f factor %f\n",wdelta,getavggradw(l,i,j));
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
