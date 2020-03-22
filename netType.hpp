/**
 * @file netType.hpp
 * @brief Contains integer enum for network types.
 *
 */

#ifndef __NETTYPE_HPP
#define __NETTYPE_HPP

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


#endif /* __NETTYPE_HPP */
