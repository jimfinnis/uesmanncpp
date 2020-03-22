/**
 * @file net.hpp
 * @brief  This is the abstract basic network class - the training
 * methods are in each subclass.
 */


#ifndef __NET_HPP
#define __NET_HPP

#include <math.h>

#include "netType.hpp"
#include "data.hpp"

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
    friend class OutputBlendingNet;
    friend class HInputNet;
public:
    
    /**
     * \brief virtual destructor which does nothing
     */
    virtual ~Net() {} 
    
    NetType type; //!< type of the network, used for load/save
    drand48_data rd; //!< PRNG data (thread safe)
    
    /** 
     * \brief Set this network's random number generator, which is
     * used for weight initialisation done at the start of training.
     */
    
    void setSeed(long seed){
        srand48_r(seed,&rd);
    }
    
    /**
     * \brief Get the number of nodes in a given layer
     * \param n layer number
     */
    virtual int getLayerSize(int n) const =0;
    
    /**
     * \brief Get the number of layers
     */
    virtual int getLayerCount() const =0;
    
    /**
     * \brief get the number of inputs
     */
    int getInputCount() const {
        return getLayerSize(0);
    }
    
    /**
     * \brief get the number of outputs
     */
    int getOutputCount() const {
        return getLayerSize(getLayerCount()-1);
    }
        
    
    
    /**
     * \brief Set the inputs to the network before running or training
     * \param d array of doubles, the size of the input layer
     */
    
    virtual void setInputs(double *d) = 0;
    
    /**
     * \brief Get the outputs after running
     * \return pointer to the output layer outputs
     */
    
    virtual double *getOutputs() const = 0;
    
    /**
     * \brief Run the network on some data.
     * \param in pointer to the input double array
     * \return pointer to the output double array
     */
    double *run(double *in) {
        setInputs(in);
        update();
        return getOutputs();
    }
    
    /**
     * \brief Set the modulator level for subsequent runs and training of this
     * network.
     */
    virtual void setH(double h)=0;
    
    /**
     * \brief get the modulator level
     */
    virtual double getH() const =0;
    
    /**
     * \brief Test a network.
     * Runs the network over a set of examples and returns the mean MSE for all outputs
     * \f[
     * \frac{1}{N\cdot N_{outs}}\sum^N_{e \in Examples} \sum_{i=0}^{N_{outs}} (e_o(i) - e_y(i))^2
     * \f]
     * where
     * \f$N\f$ is the number of examples, 
     * \f$N_{outs}\f$ is the number of outputs,
     * \f$e_o(i)\f$ is network's output for example \f$e\f$,
     * and
     *  \f$e_y(i)\f$ is the desired output for the same example.
     * 
     * \param examples Example set to test (or partially test).
     * \param start    index of example to start test at.
     * \param num      number of examples to test (or -1 for all after start point).
     * 
     */
    double test(ExampleSet& examples,int start=0,int num=-1){
        double mseSum = 0;
        // have to do this here, too, although runExamples does it, so we can
        // get the denominator for the mse.
        if(num<0)num=examples.getCount()-start;
        
        // for each example, run it and accumulate the sum of squared errors
        // on all outputs 
        
        for(int i=0;i<num;i++){
            int idx = start+i;
            setH(examples.getH(idx));
            double *netout = run(examples.getInputs(idx));
            double *exout = examples.getOutputs(idx);
            for(int j=0;j<examples.getOutputCount();j++){
                double d = netout[j]-exout[j];
                mseSum += d*d;
            }
        }
        
        // we then divide by the number of examples and the output count.
        return mseSum / (num * examples.getOutputCount());
    }
    
    /**
     * \brief Training parameters for trainSGD().
     * This structure holds the parameters for the trainSGD() method, and serves
     * as a better way of passing them than a long parameter list. All values
     * have defaults set up by the constructor, which are given as constants.
     * You can set parameters by hand, but there are fluent (chainable) setters for many members.
     */
    struct SGDParams {
        friend class Net;
        
        /**
         * \brief number of iterations to run: an iteration is the presentation of a single example, NOT
         * an epoch (or occasionally pair-presentation) as is the case in the thesis when discussing the modulatory
         * network types.
         */
        int iterations;
        
        /**
         * The learning rate to use
         */
        double eta;
        
        
        /**
         * \brief The number of cross-validation slices to use
         */
        int nSlices;
        
        /**
         * \brief the number of example per cross-validation slice
         */
        int nPerSlice;
        
        /**
         * \brief how often to cross-validate given as the interval between CV events:
         * 1 is every iteration, 2 is every other iteration and so on.
         */
        int cvInterval;
        
        /** \brief fluent setter for cross-validation parameters manually; consider using crossValidation instead 
         * \param slices number of slices
         * \param nperslice number of examples per slice
         * \param interval iteration interval for cross-validation events
         */
        SGDParams& crossValidationManual(int slices,int nperslice,int interval){
            nSlices = slices;
            nPerSlice = nperslice;
            cvInterval = interval;
            return *this;
        }
        
        /**
         * \brief The shuffle mode to use - see the ExampleSet::ShuffleMode
         * enum for details.
         */
        ExampleSet::ShuffleMode shuffleMode;
        
        /** \brief fluent setter for preserveHAlternation */
        SGDParams& setShuffle(ExampleSet::ShuffleMode m){
            shuffleMode = m;
            return *this;
        }
        
        
        
        /**
         * \brief if true, use the minimum CV error to find the best net,
         * otherwise use the training error. Note that if true, networks will only be tested
         * when the cross-validation runs.
         */
        bool selectBestWithCV;
        
        /** \brief fluent setter for selectBestWithCV */
        SGDParams& setSelectBestWithCV(bool v=true){
            selectBestWithCV=v;
            return *this;
        }
        
        
        /**
         * \brief if true, shuffle the entire CV data set when all slices have been done so
         * that the cross-validation has (effectively) a new set of slices each time.
         */
        bool cvShuffle;
        
        /** \brief fluent setter for cvShuffle */
        SGDParams& setCVShuffle(bool v=true){
            cvShuffle=v;
            return *this;
        }
        
        /**
         * \brief range of initial weights/biases [-n,n], or -1 for Bishop's rule.
         */
        int initrange;
        
        /** \brief fluent setter for initrange */
        SGDParams& setInitRange(double range=-1){
            initrange = range;
            return *this;
        }
        
        /**
         * \brief seed for random number generator used to initialise weights and also
         * perform shuffling
         */
        long seed;
        
        /** \brief fluent setter for seed */
        SGDParams& setSeed(long v){
            seed = v;
            return *this;
        }
        
        /**
         * \brief a buffer of at least getDataSize() bytes for the best network. If NULL,
         * the best network is not saved.
         */
        double *bestNetBuffer;
        
        /**
         * \brief true if we should store the best net data
         */
        bool storeBestNet;
        
    private:
        /**
         * \brief Private construction helper method used by all constructors.
         * Done this way rather than calling a more basic constructor in case
         * we need to do anything clever before calling that more basic constructor.
         */
        
        void init(double _eta,int _iters){
            seed = 0L;
            eta = _eta;
            iterations = _iters;
            initrange = -1;
            bestNetBuffer = NULL;
            ownsBestNetBuffer = false;
            storeBestNet = false;
            nSlices=0;
            nPerSlice=0;
            cvInterval=1;
            shuffleMode = ExampleSet::STRIDE;
            selectBestWithCV=false; // there might not be CV!
            cvShuffle = true; // do shuffle CV at the end of an epoch
        }
    public:        
        /** \brief Constructor which sets up defaults with no information about examples - 
         * cross-validation is not set up by default, but can be done by calling
         * crossValidation() or crossValidationManual().
         * \param _eta learning rate to use
         * \param _iters number of iterations to run: an iteration is the presentation
         * of a single example, NOT a pair-presentation as is the case in the thesis when
         * discussing the modulatory network types.
         */
        
        SGDParams(double _eta, int _iters) {
            init(_eta,_iters);
        }
        
        /**
         * Alternative constructor which uses the examples to calculate
         * the number of iterations from an epoch count
         */
        
        SGDParams(double _eta,const ExampleSet& examples,int _iters){
            init(_eta,examples.getCount()*_iters);
        }
        
        /**
         * \brief Destructor
         */
        
        ~SGDParams(){
            if(ownsBestNetBuffer)delete[] bestNetBuffer;
        }
        
        /**
         * \brief Set up the cross-validation parameters given the full training set,
         * the proportion to be used for CV, the number of CV events in the training
         * run, and the number of CV slices to use. 
         * @param examples the example set we will train with
         * @param propCV the proportion of the training set to use for cross-validation
         * @param cvCount the desired number of cross-validation events across the training run
         * @param cvSlices the desired number of cross-validation slices
         * @param cvShuf should cvShuffle be true?
         * @return a reference to this, so we can do fluent chains.
         */
        
        SGDParams &crossValidation(const ExampleSet& examples,
                                   double propCV,
                                   int cvCount,
                                   int cvSlices,
                                   bool cvShuf=true
                                   ){
            cvShuffle = cvShuf;
            // calculate the number of CV examples
            int nCV = (int)round(propCV*examples.getCount());
            if(nCV==0 || nCV>examples.getCount())
                throw std::out_of_range("Bad cross-validation count");
            if(cvSlices<=0)
                throw std::out_of_range("Zero (or fewer) CV slices is a bad thing");
            // calculate the number of examples per slice and check it's not zero.
            // The resulting number of CV examples may not agree with nCV above due
            // to the integer division
            nPerSlice = nCV/cvSlices;
            nSlices = cvSlices;
            if(!nPerSlice)
                throw std::logic_error("Too many slices");
            // calculate the cvInterval
            cvInterval = iterations/cvCount;
            if(cvInterval<=0)
                throw std::logic_error("Too many CV events");
            // we want to pick the best network by CV rather than training error
            selectBestWithCV=true;
            
            printf("Cross-validation: %d slices, %d items per slice, %d total\n",
                   nSlices,nPerSlice,nSlices*nPerSlice);
            return *this;
        }
        
        /**
         * \brief set up a "best net buffer" to store the best network found,
         * to which the network will be set on completion of training.
         * @return a reference to this, so we can do fluent chains.
         */
        
        SGDParams &storeBest(){
            ownsBestNetBuffer = true;
            storeBestNet = true;
            return *this;
        }
    private:
        /**
         * \brief true if we own the best net data buffer bestNetBuffer, and should delete
         * it on destruction.
         */
        bool ownsBestNetBuffer;
    };
    
    
    
    /**
     * \brief Train using stochastic gradient descent.
     * Note that cross-validation parameters are slightly different from those
     * given in the thesis. Here we give the number of slices and number of examples
     * per slice; in the thesis we give the total number of examples to be held out
     * and the number of slices.
     * \pre Network has weights initialised to random values
     * \post The network will be set to the best network found if bestNetBuffer is set,
     * otherwise the final network will be used.
     * \throws std::out_of_range Too many CV examples
     * \throws std::logic_error Trying to select best by CV when there's no CV done
     * 
     * @param examples training set (including cross-validation data)
     * @param params a filled-in SGDParams structure giving the parameters for the training.
     * @return If storeBestNet is null, the MSE of the final network; otherwise the MSE
     * of the best network found. This is done across the entire
     * validation set if provided, or the entire training set if not.
     */
    
    double trainSGD(ExampleSet &examples,SGDParams& params){
        
        // set seed for PRNG
        setSeed(params.seed);
        
        // separate out the training examples from the cross-validation examples
        int nCV = params.nSlices*params.nPerSlice;
        // it's an error if there are too many CV examples
        if(nCV>=examples.getCount())
            throw std::out_of_range("Too many cross-validation examples");
        
        if(!nCV && params.selectBestWithCV)
            throw std::logic_error("cannot use CV to select best when no CV is done");
        
        // get the number of actual training examples
        int nExamples = examples.getCount() - nCV;
        
        // initialise the network
        initWeights(params.initrange);
        
        // initialise minimum error to rogue value
        double minError = -1;
        
        // We don't shuffle before getting the cross-validation examples,
        // because in some cases there's a kind of "fake" cv going on where the
        // training portion and cv portion have to have similar (or identical)
        // distributions. See the boolean test code for an example.
        //        examples.shuffle(&rd,params.shuffleMode);
        
        // build a temporary subset for the CV examples. This still needs to exist
        // even if we're not using CV, so in that case we'll just
        // use a dummy of one example.
        
        ExampleSet cvExamples(examples,nCV?examples.getCount()-nCV:0,nCV?nCV:1);
        
        
        // setup a countdown for when we cross-validate
        int cvCountdown = params.cvInterval;
        // and which slice we are doing
        int cvSlice = 0;
        
        // now actually do the training
        
        FILE *log = fopen("foo","w");
        fprintf(log,"x,slice,y\n");
        for(int i=0;i<params.iterations;i++){
            // find the example number
            int exampleIndex = i % nExamples;
            
            // at the start of each epoch, reshuffle. This will effectively do an extra shuffle
            // as we've already done it once at the start, before splitting out the CV examples.
            
            if(exampleIndex == 0)
                examples.shuffle(&rd,params.shuffleMode,nExamples);
                
            // train here, just one example, no batching.
            double trainingError = trainBatch(examples,exampleIndex,1,params.eta);
            
            if(!params.selectBestWithCV){
                // now test the error and keep the best net. This works differently
                // if we're doing this by cross-validation or training error. Here
                // we're using the training error.
                if(minError < 0 || trainingError < minError){
                    if(params.storeBestNet){
                        if(!params.bestNetBuffer)
                            params.bestNetBuffer = new double[getDataSize()];
                        save(params.bestNetBuffer);
                    }
                    minError = trainingError;
                }
            }
            
            // is there cross-validation? If so, do it.
            
            if(nCV && !--cvCountdown){
                cvCountdown = params.cvInterval; // reset
                
                // test the appropriate slice, from example cvSlice*nPerSlice, length nPerSlice,
                // and get the MSE
                double error = test(cvExamples,cvSlice*params.nPerSlice,
                                    params.nPerSlice);
                fprintf(log,"%d,%d,%f\n",i,cvSlice,error);
                
                // test this against the min error as was done above
                if(params.selectBestWithCV){
                    if(minError < 0 || trainingError < minError){
                        if(params.storeBestNet){
                        if(!params.bestNetBuffer)
                            params.bestNetBuffer = new double[getDataSize()];
                            save(params.bestNetBuffer);
                        }
                        minError = trainingError;
                    }
                }
                
                // increment the slice index
                cvSlice = (cvSlice+1)%params.nSlices;
                // if we are now on the first slice, shuffle the entire CV set
                if(!cvSlice && params.cvShuffle)
                    cvExamples.shuffle(&rd,params.shuffleMode);
            }
        }
        
        fclose(log);
        
        // at the end, finalise the network to the best found if we can
        if(params.bestNetBuffer)
            load(params.bestNetBuffer);
        
        // test on either the entire CV set or the training set and return result
        return test(nCV?cvExamples:examples);
    }
    
    /**
     * \brief Get the length of the serialised data block
     * for this network.
     * \return the size in doubles
     */
    virtual int getDataSize() const = 0;
    
    /**
     * \brief Serialize the data (not including any network type magic number or
     * layer/node counts) to the given memory (which must be of sufficient size).
     * \param buf the buffer to save the data, must be at least getDataSize() doubles
     */
    virtual void save(double *buf) const = 0;
    
    /**
     * \brief Given that the pointer points to a data block of the correct size
     * for the current network, copy the parameters from that data block into
     * the current network overwriting the current parameters.
     * \param buf the buffer to load the data from, must be at least getDataSize() doubles
     */
    virtual void load(double *buf) = 0;
    
protected:
    
    
    
    /**
     * \brief Run a single update of the network
     * \pre input layer must be filled with values
     * \post output layer contains result
     */
    
    virtual void update() = 0;
    
    /**
     * \brief Constructor - protected because others inherit it and it's not used
     * directly.
     * \param tp network type enumeration
     */
    Net(NetType tp){
        type = tp;
        setSeed(0);
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
     * - return the mean squared error (NOTE: different from original, which returned
     *   mean absolute error) for all outputs and examples:
     * \f[
     * \frac{1}{N\cdot N_{outs}}\sum^N_{e \in Examples} \sum_{i=0}^{N_{outs}} (e_o(i) - e_y(i))^2
     * \f]
     * where
     * \f$N\f$ is the number of examples, 
     * \f$N_{outs}\f$ is the number of outputs,
     * \f$e_o(i)\f$ is network's output for example \f$e\f$,
     * and
     *  \f$e_y(i)\f$ is the desired output for the same example.
     * \param ex      example set
     * \param start   index of first example to use
     * \param num     number of examples. For a single example, you'd just use 1.
     * \param eta     learning rate
     * \return        the sum of mean squared errors in the output layer (see formula in method documentation)
     */
    virtual double trainBatch(ExampleSet& ex,int start,int num,double eta) = 0;
    
};    


#endif /* __NET_HPP */
