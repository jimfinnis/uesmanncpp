Tests   {#tests}
=====
This page describes the various unit tests, some of which are more time-consuming
than others. For this reason, only the **basic** and **booleans** suites are
performed during the Travis build process. The test suites and tests are:

* **basic** : suite for underlying functionality tests
    * **example** : test that ExampleSet can construct and retrieve example data.
    * **alt** : test that the alternate() function works.
    * **altex** : test ExampleSet::ALTERNATE shuffling on examples.
    * **stride** : test ExampleSet::STRIDE shuffling.
    * **altex4** : test ExampleSet::ALTERNATE with 4 modulator levels.
    * **testmse** : test mean squared error sum of outputs on a zero parameter net
    * **loadmnist** : test that MNIST data sets can be loaded.
    and confirm the MSE is low on training complete. This test is described in
    [this section](##Addition).
* **basictrain** : test training of backprop nets
    * **trainparams** : test that we can train the identity function.
    * **trainparams2** : as trainparams, but with more examples and no crossvalidation;
    it aims to be identical to an existing program written using Angort.
    * **addition** : train a plain backprop network to perform addition.
    * **additionmod** : train a UESMANN network to perform addition and scaled addition:
    at *h*=0 the generated function will be *y*= *a* + *b*, while at *h*=1 it becomes
    *y*=0.3( *a* + *b* ).
    * **trainmnist** train a plain backpropagation network to recognise MNIST digits
    using a low number of iterations; we aim for a success rate of at least 85%.
* **booleans** : test training of a boolean modulatory pairing (XOR/AND) in all 3 modulatory network
types - the network should modulate from XOR to AND as the modulator moves from 0 to 1.
    * **obxorand** : output blending
    * **hinxorand** : h-as-input
    * **uesmann** : UESMANN
    
    

## Example code
Some of the tests are described below in more detail with commented source code.
These should give you some idea of how to use the system.

### basictrain/addition
This test constructs an ExampleSet consisting of 1000 examples of pairs of random numbers
as input with their sums as output. It then builds a BPNet - 
a plain multilayer perceptron, traininable by backpropagation with no modulation. 
This is done by calling NetFactory::makeNet() with the NetType::PLAIN
argument. 

A Net::SGDParams structure is set up with suitable training parameters for stochastic
gradient descent, and the network is trained using this structure. The result is the mean
squared error for all outputs:
    \f[
    \frac{1}{N\cdot N_{outs}}\sum^N_{e \in Examples} \sum_{i=0}^{N_{outs}} (e_o(i) - e_y(i))^2
    \f]
    where
    \f$N\f$ is the number of examples, 
    \f$N_{outs}\f$ is the number of outputs,
    \f$e_o(i)\f$ is network's output for example \f$e\f$,
    and
     \f$e_y(i)\f$ is the desired output for the same example.

\snippet testTrainBasic.cpp addition


### basictrain/additionmod
This test is similar to **basictrain/addition**, but builds an ExampleSet consisting of
2000 examples. Evenly numbered examples are of *y*= *a* + *b*, and odd-numbered have
*y*=0.3( *a* + *b* ). The modulator on each example is set at 0 for even and 1 for odd.
Thus these should train a modulated network to transition from the former to the latter
function as the modulator goes from 0 to 1.

\snippet testTrainBasic.cpp additionmod


### basictrain/trainmnist
This test loads the %MNIST handwritten digits dataset and trains an ordinary unmodulated
network to recognise them. It makes use of the MNIST class and the special %MNIST constructor
for ExampleSet, and runs another test set through the network to see how well it performs.

\snippet testTrainBasic.cpp trainmnist
