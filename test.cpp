/**
 * @file test.cpp
 * @brief  Brief description of file.
 *
 */

#include <iostream>

#include "bpnet.hpp"

int main(int argc,char *argv[]){
    int layers[] = {2,2,1};
    BPNet foo(0.1,3,layers);
    return 0;
}
