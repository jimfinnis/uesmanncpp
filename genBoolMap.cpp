/**
 * @file genBoolMap.cpp
 * @brief  Generate a 2D grid (well, a table from which such
 * a grid can be generated) of how many trials of UESMANN on
 * every combination of binary boolean functions succeed.
 * 
 * This should (and does) generate approximately the same
 * data as in Fig. 5.3a of the thesis (p.100). The variation is
 * no greater than 0.001 (i.e. a single network) in each pairing
 * tested.
 */

#include "netFactory.hpp"

/** \brief How many networks to attempt for each pairing in genBoolMap */
#define NUM_ATTEMPTS 1000

/** \brief the learning rate for genBoolMap */
#define ETA 0.1


/** \brief
 * how many epochs to train each genBoolMap network for -  at 8 examples per
 * epoch, this is 600000 training iterations (single examples)
 */
#define EPOCHS 75000

/**
 * \brief possible inputs to boolean functions
 */

double ins[][2]={
    {0,0},
    {0,1},
    {1,0},
    {1,1}};

/**
 * \brief names of functions performed by boolFunc()
 */
const char *simpleNames[] = {
 "f","and","x and !y","x","!x and y","y","xor","or","nor","xnor",
    "!y","x or !y","!x","!x or y","nand","t"};

/**
 * \brief given a function index, perform the appropriate boolean. 
 * The index is actually the truth table: four bits in order 00,01,10,11
 */

bool boolFunc(int f,bool a,bool b){
    // which bit do we want?
    int bit = 1<<((a?0:2)+(b?0:1));
//    printf("Bit set is %d, & %d = %d\n",bit,f,bit&f);
    return (f&bit)!=0;
}

/**
 * \brief Set an example in the example set to the output of a given
 * function (as an index into simpleNames), given the inputs and
 * the modulator level.
 */

static void setExample(ExampleSet& e,int exampleIdx,
                       int functionIdx,int xbit,int ybit,double mod){
    double *ins = e.getInputs(exampleIdx);
    double *outs = e.getOutputs(exampleIdx);
    e.setH(exampleIdx,mod);
    ins[0] = xbit;
    ins[1] = ybit;
    bool val = boolFunc(functionIdx,xbit!=0,ybit!=0);
    *outs = val ? 1 : 0;
    
}

/**
 * \brief test if a given network successfully performs a given pair
 * of boolean functions, modulating from f1 to f2. The functions are
 * indices into the simpleNames array.
 */
bool success(int f1,int f2,Net *n){
    double in[2];
    double out;
    for(int a=0;a<2;a++){
        for(int b=0;b<2;b++){
            bool shouldBeHigh1 = boolFunc(f1,a!=0,b!=0);
            bool shouldBeHigh2 = boolFunc(f2,a!=0,b!=0);
            in[0]=a;
            in[1]=b;
            n->setH(0);
            out = *(n->run(in));
//            printf("%d %d at 0 -> %f (should be %d)\n",a,b,out,shouldBeHigh1);
            if(out>0.5 != shouldBeHigh1)return false;
            n->setH(1);
            out = *(n->run(in));
//            printf("%d %d at 1 -> %f (should be %d)\n",a,b,out,shouldBeHigh2);
            if(out>0.5 != shouldBeHigh2)return false;
        }
    }
    return true;
}


/**
 * \brief Train a large number of networks to do a particular
 * pairing of boolean functions (provided as indices into simpleNames)
 * and return what proportion successfully perform that pairing under
 * modulation
 */

double doPairing(int f1,int f2){
    // first we need to build the examples.
    // 8 examples (4 at each mod level), 2 in, 1 out, 2 mod levels
    ExampleSet e(8,2,1,2);
    // add examples for 0,0
    setExample(e,0,f1,0,0,0);
    setExample(e,1,f2,0,0,1);
    // add examples for 0,1
    setExample(e,2,f1,0,1,0);
    setExample(e,3,f2,0,1,1);
    // add examples for 1,0
    setExample(e,4,f1,1,0,0);
    setExample(e,5,f2,1,0,1);
    // add examples for 1,1
    setExample(e,6,f1,1,1,0);
    setExample(e,7,f2,1,1,1);
    
    // training parameters.
    Net::SGDParams params(ETA,e,EPOCHS);
    // pick the best network by training MSE (not cross-validation
    // as we're not doing it) and keep it as we go along.
    // Shuffle the network by stride, which is the number of examples:
    // on each epoch, pairs of examples will be shuffled rather than
    // single examples. This is true to the method given in the thesis,
    // alternating training between h=0 and h=1 examples.
    
    params.storeBest().setShuffle(ExampleSet::STRIDE);
    
    
    int successful = 0; // number of networks which worked
    for(int i=0;i<NUM_ATTEMPTS;i++){
        // make a new network
        Net *n = NetFactory::makeNet(NetType::UESMANN,e,2);
        // set a new PRNG and train it (this will also set the init
        // weights)
        params.setSeed(i);
        // train the network
        n->trainSGD(e,params);
        // and increment the count if it was good
        if(success(f1,f2,n))
            successful++;
        delete n; // remember to delete the network
    }
    // return successful proportion
    return ((double)successful)/(double)NUM_ATTEMPTS;
}

/**
 * \brief The main function for genBoolMap
 */
int main(int argc,char *argv[]){
    // output is function 1, function 2, and correct network
    // proportion
    printf("a,b,correct\n");
    // run the 256 pairings and output their correctness proportion.
    for(int f1=0;f1<16;f1++){
        for(int f2=0;f2<16;f2++){
            printf("%d,%d,%f\n",f1,f2, doPairing(f1,f2));
        }
    }
}
