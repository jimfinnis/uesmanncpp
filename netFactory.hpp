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
#include "uesnet.hpp"


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
        
        return makeNet(t,3,layers);
    }
    
    static Net *makeNet(NetType t,int layercount, int *layers){
        switch(t){
        case NetType::PLAIN:
            return new BPNet(layercount,layers);
        case NetType::OUTPUTBLENDING:
            return new OutputBlendingNet(layercount,layers);
        case NetType::HINPUT:
            return new HInputNet(layercount,layers);
        case NetType::UESMANN:
            return new UESNet(layercount,layers);
        default:break;
        }
    }
    
    /**
     * \brief Load a network of any type from a file - note, endianness not checked!
     */
    
    inline static Net *loadNet(char *fn){
        FILE *a = fopen(fn,"rb");
        if(!a)
            throw new std::runtime_error("cannot open file");
        
        // get type
        uint32_t magic;
        if(!fread(&magic,sizeof(uint32_t),1,a)){
            fclose(a);
            throw new std::runtime_error("bad net save file");
        }
            
        NetType t = static_cast<NetType>(magic);
        
        // build layer specification reading the layer count and then
        // the layer sizes
        uint32_t layercount,tmp;
        if(!fread(&layercount,sizeof(uint32_t),1,a)){
            fclose(a);
            throw new std::runtime_error("bad net save file");
        }
        int *layers = new int[layercount];
        for(int i=0;i<layercount;i++){
            if(!fread(&tmp,sizeof(uint32_t),1,a)){
                delete [] layers;
                fclose(a);
                throw new std::runtime_error("bad net save file");
            }
            layers[i]=tmp;
        }
        
        // build the net
        Net *n = makeNet(t,layercount,layers);
        
        // get the parameter data 
        int size = n->getDataSize();
        double *buf = new double[size];
        // and read it
        if(fread(buf,sizeof(double),size,a)!=size){
            delete [] buf;
            delete [] layers;
            fclose(a);
            throw new std::runtime_error("bad net save file");
        }
        n->load(buf);
        
        delete [] buf;
        delete [] layers;
        fclose(a);
        
        
        
        
    }
    
    /**
     * \brief Save a net of any type to a file - note, endianness not checked!
     */
    
    inline static void saveNet(const char *fn,Net *n) {
        FILE *a = fopen(fn,"wb");
        if(!a)
            throw new std::runtime_error("cannot open file");
        
        // get and write the magic number
        uint32_t magic=static_cast<uint32_t>(n->type); // magic number
        fwrite(&magic,sizeof(uint32_t),1,a);
        
        // write the layer count and layer sizes, all as 32-bit.
        uint32_t layercount = n->getLayerCount();
        fwrite(&layercount,sizeof(uint32_t),1,a);
        for(int i=0;i<layercount;i++){
            uint32_t layersize = n->getLayerSize(i);
            fwrite(&layercount,sizeof(uint32_t),1,a);
        }
        
        // get the parameter data 
        int size = n->getDataSize();
        double *buf = new double[size];
        n->save(buf);
        // and write it
        fwrite(buf,sizeof(double),size,a);
        delete [] buf;
        
        fclose(a);
    }
    
};



#endif /* __NETFACTORY_HPP */
