Tests
=====

## Addition 
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

\snippet testBasic.cpp addition


## %MNIST handwriting recognition
This test loads the %MNIST handwritten digits dataset and trains an ordinary unmodulated
network to recognise them. It makes use of the MNIST class and the special %MNIST constructor
for ExampleSet, and runs another test set through the network to see how well it performs.

\snippet testTrainBasic.cpp trainmnist
