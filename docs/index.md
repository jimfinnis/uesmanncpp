Main Page   {#mainpage}
=========

This is the new C++ implementation of UESMANN, based on the original
code used in my thesis. That code was written as a ```.angso```
library for the Angort language, and has evolved to become
rather unwieldly (as well as being intended for use from a language
no-one but me uses).

I originally intended to write this implementation using Keras/Tensorflow,
but would have been limited to using the low-level Tensorflow operations
because of the somewhat peculiar nature of optimisation in UESMANN:
we descend the gradient relative to the weights for one function,
and the gradient relative to the weights times some constant for the other.
Because the existing code is in C++, this implementation will be written
in C++ with other languages and platforms as future work.

Implementations of the other network types mentioned in the thesis
are also included.

The top level class is @ref Net, which is an virtual type describing the neural net interface
and performing some basic operations. 

## Tests

The files named *test* have various unit tests in them which can be useful to 
examine. The [Tests](@ref tests) page shows some of these tests in more detail.


