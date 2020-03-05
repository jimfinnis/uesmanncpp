/**
 * @file test.hpp
 * @brief Useful stuff for testing
 *
 */

#ifndef __TEST_HPP
#define __TEST_HPP

#include "netFactory.hpp"

namespace utf = boost::unit_test;

/**
 * \brief get index of max value in an array
 * \param o array ptr
 * \param n length of array
 */

static int getHighest(double *o,int n){
    int h=0;
    double maxval=-10;
    for(int i=0;i<n;i++,o++){
        if(*o > maxval){
            maxval= *o;
            h=i;
        }
    }
    return h;
}
    



#endif /* __TEST_HPP */
