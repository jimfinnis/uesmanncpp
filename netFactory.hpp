/**
 * @file netFactory.hpp
 * @brief I'm not a fan of factories, but here's one - this makes 
 * a network of the appropriate type which conforms to an example
 * set, and is a namespace.
 *
 */

#ifndef __NETFACTORY_HPP
#define __NETFACTORY_HPP

#include "bpnet.hpp"

/**
 * \brief The different types of network - each has an associated integer
 * for saving/loading file data.
 */
enum class NetType {
    PLAIN=1000, /// \brief plain back-propagation
          
    MAX = PLAIN /// \brief max
};

namespace NetFactory {

/**
 * Construct a single hidden layer network of a given type
 * which conforms to the example set.
 */

inline static Net *makeNet(NetType t,ExampleSet &e,int hnodes){
    Net *net;
    
    int layers[3];
    layers[0] = e.getInputCount();
    layers[1] = hnodes;
    layers[2] = e.getOutputCount();
    
    switch(t){
    case NetType::PLAIN:
        return new BPNet(3,layers);
    default:break;
    }
}

}



#endif /* __NETFACTORY_HPP */
