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
#include "obnet.hpp"
#include "hinet.hpp"

/**
 * \brief The different types of network - each has an associated integer
 * for saving/loading file data.
 */
enum class NetType {
    PLAIN=1000, /// \brief plain back-propagation
          OUTPUTBLENDING, /// \brief output blending
          HINPUT, /// \brief h-as-input
          UESMANN, /// \brief UESMANN
          
          MAX = PLAIN /// \brief max
};

/**
 * \brief
 * This class - really a namespace -  contains functions which create,
 *  load or save networks of all types.
 */

class NetFactory { // not a namespace because Doxygen gets confused.
public:
/**
 * \brief
 * Construct a single hidden layer network of a given type
 * which conforms to the example set.
 */

static Net *makeNet(NetType t,ExampleSet &e,int hnodes){
    Net *net;
    
    int layers[3];
    layers[0] = e.getInputCount();
    layers[1] = hnodes;
    layers[2] = e.getOutputCount();
    
    switch(t){
    case NetType::PLAIN:
        return new BPNet(3,layers);
    case NetType::OUTPUTBLENDING:
        return new OutputBlendingNet(3,layers);
    case NetType::HINPUT:
        return new HInputNet(3,layers);
    case NetType::UESMANN:
        throw new std::runtime_error("UESMANN not yet implemented");
    default:break;
    }
}

/**
 * \brief Load a network of any type from a file
 */

inline static Net *loadNet(char *fn){
    throw new std::runtime_error("loadNet not yet implemented");
}

/**
 * \brief Save a net of any type to a file
 */

inline static void saveNet(char *fn,Net *n){
    throw new std::runtime_error("saveNet not yet implemented");
}

};



#endif /* __NETFACTORY_HPP */
