/**
 * @file net.hpp
 * @brief  This is the abstract basic network class - the training
 * methods are in each subclass.
 */


#ifndef __NET_HPP
#define __NET_HPP

#include <math.h>

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
protected:
    
    /// \brief learning rate, set up as part of each training function
    double eta;
public:
    
    /**
     * \brief get the learning rate
     */
    
    double getEta() const {
        return eta;
    }
    
    drand48_data rd; //!< PRNG data (thread safe)
    
    /** 
     * \brief Set this network's random number generator, which is
     * used for weight initialisation done at the start of training.
     */
    
    void setSeed(long seed){
        srand48_r(seed,&rd);
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
    virtual void setH(double h){
        // default does nothing.
    }
    
    /**
     * \brief Run a function on some or all examples in an example set.
     * This function runs a function, typically provided as a lambda,
     * on an example set. It's used in testing the network.
     * The function has the signature
     * \code
     *  void f(double *netout,ExampleSet& examples,int index)
     * \endcode
     * where 
     * 
     * * *netout* is the array of outputs from the network after the example has run,
     * * *examples* is the ExampleSet in which the example resides, and
     * * *index* is the index of the example within the set.
     * 
     * See test() for an example.
     */

    template <class TestFunc> void runExamples(ExampleSet& examples,
                                                       int start,int num,
                                                       TestFunc f){
        if(num<0)num=examples.getCount()-start;
        for(int i=0;i<num;i++){
            int idx = start+i;
            // run the example
            setH(examples.getH(idx));
            double *netout = run(examples.getInputs(idx));
            // perform the function
            f(netout,examples,i);
        }
    }
    
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
        // we use runExamples, which performs a function on all examples. The function
        // here accumulates the sum of squared errors on all outputs.
        runExamples(examples,start,num,
                    [&mseSum](double *out,ExampleSet& examples,int idx){
                    double *netOuts = examples.getOutputs(idx);
                    for(int i=0;i<examples.getOutputCount();i++){
                        double d = out[i]-netOuts[i];
                        mseSum+=d*d;
                    }
                });
        // we then divide by the number of examples and the output count.
//        printf("SUM: %f, dividing by %d\n", mseSum, num*examples.getOutputCount());
        return mseSum / (num * examples.getOutputCount());
    }
    
    /**
     * \brief Training parameters for trainSGD().
     * This structure holds the parameters for the trainSGD() method, and serves
     * as a better way of passing them than a long parameter list. All values
     * have defaults set up by the constructor, which are given as constants.
     */
    struct SGDParams {
        
        /// @brief default value of iterations
        constexpr static int DEF_ITERATIONS=10000;
        /**
         * \brief number of iterations to run: an iteration is the presentation of a single example, NOT
         * a pair-presentation as is the case in the thesis when discussing the modulatory network types.
         */
        int iterations;
        
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
        
        /// @brief default value of preserveHAlternation (do preserve)
        constexpr static bool DEF_PRESERVEHALTERNATION=true;
        /**
         * \brief if true, the shuffled examples are rearranged so that
         * they alternate h<0.5 and h>=0.5
         */
        bool preserveHAlternation;
        
        /// @brief default value of selectBestWithCV (don't select with cross-validation;
        /// there isn't any by default)
        constexpr static bool DEF_SELECTBESTWITHCV=false;
        /**
         * \brief if true, use the minimum CV error to find the best net,
         * otherwise use the training error. Note that if true, networks will only be tested
         * when the cross-validation runs.
         */
        bool selectBestWithCV;
        
        /// @brief default value of cvShuffle (do shuffle)
        constexpr static bool DEF_CVSHUFFLE=true;
        /**
         * \brief if true, shuffle the entire CV data set when all slices have been done so
         * that the cross-validation has (effectively) a new set of slices each time.
         */
        bool cvShuffle = DEF_CVSHUFFLE;
        
        /**
         * \brief range of initial weights/biases [-n,n], or -1 for Bishop's rule.
         */
        int initrange;
        
        /**
         * \brief a buffer of at least getDataSize() bytes for the best network. If NULL,
         * the best network is not saved.
         */
        double *bestNetData;
        
        /**
         * \brief true if we own the best net data buffer bestNetData, and should delete
         * it on destruction.
         */
        bool ownsBestNetData;
        
        /**
         * The learning rate to use
         */
        double eta;
        
        /** \brief Constructor which sets up defaults with no information about examples - 
         * cross-validation is not set up by default, but can be done by calling
         * setupCrossValidation().
         * \param _eta learning rate to use
         * \param _iters number of iterations to run: an iteration is the presentation
         * of a single example, NOT a pair-presentation as is the case in the thesis when
         * discussing the modulatory network types.
         */
        
        SGDParams(double _eta, int _iters) {
            eta = _eta;
            iterations = _iters;
            initrange = -1;
            bestNetData = NULL;
            ownsBestNetData = false;
            nSlices=0;
            nPerSlice=0;
            cvInterval=1;
            preserveHAlternation=DEF_PRESERVEHALTERNATION;
            selectBestWithCV=DEF_SELECTBESTWITHCV;
            cvShuffle = DEF_CVSHUFFLE;
        }
        
        /**
         * \brief Destructor
         */
        
        ~SGDParams(){
            if(ownsBestNetData)delete[] bestNetData;
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
        
        SGDParams &storeBest(const Net& net){
            bestNetData = new double[net.getDataSize()];
            ownsBestNetData = true;
            return *this;
        }
    };
        
    
    
    /**
     * \brief Train using stochastic gradient descent.
     * Note that cross-validation parameters are slightly different from those
     * given in the thesis. Here we give the number of slices and number of examples
     * per slice; in the thesis we give the total number of examples to be held out
     * and the number of slices.
     * \pre Network has weights initialised to random values
     * \post The network will be set to the best network found if bestNetData is set,
     * otherwise the final network will be used.
     * \throws std::out_of_range Too many CV examples
     * \throws std::logic_error Trying to select best by CV when there's no CV done
     * 
     * @param examples training set (including cross-validation data)
     * @param params a filled-in SGDParams structure giving the parameters for the training.
     * @return If bestNetData is null, the MSE of the final network; otherwise the MSE
     * of the best network found. This is done across the entire
     * validation set if provided, or the entire training set if not.
     */
    
    double trainSGD(ExampleSet &examples,SGDParams& params){
        // set up learning rate
        eta = params.eta;
        
        // separate out the training examples from the cross-validation examples
        int nCV = params.nSlices*params.nPerSlice;
        // it's an error if there are too many CV examples
        if(nCV>=examples.getCount())
            throw std::out_of_range("Too many cross-validation examples");
        
        if(!nCV && params.selectBestWithCV)
            throw std::logic_error("cannot use CV to select best when no CV is done");
        
        // shuffle before getting the cross-validation examples
        examples.shuffle(&rd,true);
        
        // build a temporary subset for the CV examples. This still needs to exist
        // even if we're not using CV, so in that case we'll just
        // use a dummy of one example.
        
        ExampleSet cvExamples(examples,nCV?examples.getCount()-nCV:0,nCV?nCV:1);
        
        // get the number of actual training examples
        int nExamples = examples.getCount() - nCV;
        
        // initialise the network
        initWeights(params.initrange);
        
        // initialise minimum error to rogue value
        double minError = -1;
        
        // ah, now it gets awkward; we need to shuffle the examples on each
        // iteration but for UESMANN they really need to be h=0/1 alternating.
        // In the thesis this was done for everything but the robot work, because
        // examples were stored as h=0/1 pairs. For the robot work, the data was
        // generated by a simple controller in a separate program which switched
        // between h=0 and h=1 every now and then. These were shuffled and used
        // as is, so the data is not alternating. In fact, it sort of can't be
        // with that data because there might be more of one h level than the other.
        // Need to give this some thought. It doesn't invalidate the work in the PhD
        // although this is one of those cases where I could have been clearer!
        
        examples.shuffle(&rd,params.preserveHAlternation);
        
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
            
            // train here, just one example, no batching.
            double trainingError = trainBatch(examples,exampleIndex,1);
            
            if(!params.selectBestWithCV){
                // now test the error and keep the best net. This works differently
                // if we're doing this by cross-validation or training error. Here
                // we're using the training error.
                if(minError < 0 || trainingError < minError){
                    if(params.bestNetData)
                        save(params.bestNetData);
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
                if(minError < 0 || trainingError < minError){
                    if(params.bestNetData)
                        save(params.bestNetData);
                    minError = trainingError;
                }
                
                // increment the slice index
                cvSlice = (cvSlice+1)%params.nSlices;
                // if we are now on the first slice, shuffle the entire CV set
                if(!cvSlice && params.cvShuffle)
                    cvExamples.shuffle(&rd,params.preserveHAlternation);
            }
        }
        
        fclose(log);
        
        // at the end, finalise the network to the best found if we can
        if(params.bestNetData)
            load(params.bestNetData);
        
        // test on either the entire CV set or the training set and return result
        return test(nCV?cvExamples:examples);
    }
    
    /**
     * \brief Get the length of the serialised data block
     * for this network.
     * \return the size in bytes
     */
    virtual int getDataSize() const = 0;
    
    /**
     * \brief Serialize the data (not including any network type magic number or
     * layer/node counts) to the given memory (which must be of sufficient size).
     * \param buf the buffer to save the data, must be at least getDataSize() bytes
     */
    virtual void save(double *buf) const = 0;
    
    /**
     * \brief Given that the pointer points to a data block of the correct size
     * for the current network, copy the parameters from that data block into
     * the current network overwriting the current parameters.
     * \param buf the buffer to load the data from, must be at least getDataSize() bytes
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
     * directly
     */
    Net(){}
    
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
     * \return        the sum of mean squared errors in the output layer (see formula in method documentation)
     */
    virtual double trainBatch(ExampleSet& ex,int start,int num) = 0;
    
};    


#endif /* __NET_HPP */
