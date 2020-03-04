/**
 * @file mnist.hpp
 * @brief Code for converting MNIST data into example sets
 *
 */

#ifndef __MNIST_HPP
#define __MNIST_HPP


#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <arpa/inet.h>
#include <stdexcept>

/**
 * \brief This class encapsulates and loads data in the standard MNIST format.
 * The data resides in two files, an image file and a label file. 
 */

class MNIST {
public:
    /**
     * \brief constructor which loads the data from the given file, and can load
     * only part of the data in a file.
     * \param labelFile the name of the file containing the labels
     * \param imgFile the name of the file containing the image data
     * \param start the image number to start loading from
     * \param len how many images to load (0 means all)
     */
    
    MNIST(const char *labelFile,const char *imgFile,int start=0,int len=0){
        valid = false;
        int rd;
        FILE *a = fopen(labelFile,"rb");
        if(!a){
            printf("Error opening label file %s: %s\n",labelFile,strerror(errno));;
            throw std::runtime_error("cannot open label file: " + std::string(labelFile));
            exit(1);
        }
        uint32_t magic;
        rd=fread(&magic,sizeof(uint32_t),1,a);
        magic = htonl(magic);
        if(magic!=2049){
            printf("incorrect magic number in label file %s: %x\n",labelFile,magic);
            throw std::runtime_error("bad magic number in label file");
        }
        rd=fread(&ct,sizeof(uint32_t),1,a);
        ct = htonl(ct);
        if(ct>100000){
            printf("unfeasibly large count in label file %s: %x\n",labelFile,ct);
            throw std::runtime_error("bad count in label file");
        }
        
        if(!len)len=ct;
        if(start+len>ct){
            printf("specified range [%d-%d], only %d in file %s\n",start,start+len,ct,labelFile);
            throw std::runtime_error("bad range in label file");
        }
        fseek(a,start*sizeof(uint8_t),SEEK_CUR); // skip some
        
        labels = new uint8_t[len];
        rd = fread(labels,sizeof(uint8_t),len,a);
        if(rd!=len){
            printf("not enough items in label file %s: %d\n",labelFile,rd);
            throw std::runtime_error("not enough elements in label file");
        }
        fclose(a);
        
        
        a = fopen(imgFile,"rb");
        if(!a){
            printf("Error opening image file %s: %s\n",imgFile,strerror(errno));
            throw std::runtime_error("cannot open image file: " + std::string(imgFile));
        }        
        rd=fread(&magic,sizeof(uint32_t),1,a);
        magic=htonl(magic);
        if(magic!=2051){
            printf("incorrect magic number in image file %s: %d\n",imgFile,magic);
            throw std::runtime_error("bad magic in image file");
        }
        uint32_t ct2;
        rd=fread(&ct2,sizeof(uint32_t),1,a);
        ct2=htonl(ct2);
        if(ct2!=ct){
            printf("image file count does not agree with label file count:\n"
                   "%s:%d != %s:%d\n",
                   imgFile,ct2,labelFile,ct);
            throw std::runtime_error("bad count in image file");
        }
        
        rd=fread(&rows,sizeof(uint32_t),1,a);
        rows = htonl(rows);
        rd=fread(&cols,sizeof(uint32_t),1,a);
        cols = htonl(cols);
        if(rows > 128 || cols > 128){
            printf("Bad dimensions in image file %s: %dx%d\n",imgFile,rows,cols);
            throw std::runtime_error("bad dimensions in image file");
        }
        
        fseek(a,start*sizeof(uint8_t)*rows*cols,SEEK_CUR); // skip some
        imgs = new uint8_t[len*rows*cols];
        rd = fread(imgs,sizeof(uint8_t),rows*cols*len,a);
        if(rd!=len*rows*cols){
            printf("wrong amount of pixels in image file %s: %d\n",imgFile,rd);
            throw std::runtime_error("bad filesize in image file");
        }
        fclose(a);
        ct=len;
        
        // get the max label
        maxLabel=0;
        for(int i=0;i<ct;i++){
            if(getLabel(i)>maxLabel)
                maxLabel=getLabel(i);
        }
        
    }
    
    
    /**
     * \brief Destructor
     */
    ~MNIST(){
        delete [] labels;
        delete [] imgs;
    }
    
    /**
     * \brief returns the number of examples
     */
    int getCount() const {
        return ct;
    }
    
    /**
     * \brief returns the number of rows in each image
     */
    int r() const {
        return rows;
    }
    
    /**
     * \brief returns the number of columns in each image
     */
    
    int c() const {
        return cols;
    }
    
    /**
     * \brief get the label for a given example
     */
    
    uint8_t getLabel(int n) const {
        return labels[n];
    }
    
    /**
     * \brief get the maximum label value (0 to 9 in the original
     * data but different in other tests)
     */
    uint8_t getMaxLabel() const {
        return maxLabel;
    }
    
    /**
     * \brief get the bitmap for a given example
     * \return a pointer to the first pixel in the image
     */
    
    uint8_t *getImg(int n) const {
        return imgs+rows*cols*n;
    }
    
    /**
     * \brief get a pixel for a given example
     * \return the pixel as a byte
     */
    
    uint8_t getPix(int n,int x,int y) const {
        int idx = x+y*cols;
        return getImg(n)[idx];
    }
    
    /**
     * \brief dump the image data to standard out
     */
    void dump(int i) const {
        if(i>=getCount())
            printf("Out of range\n");
        else {
            printf("Label: %d\n",getLabel(i));
            uint8_t *d = getImg(i);
            for(int x=0;x<r();x++){
                for(int y=0;y<c();y++){
                    uint8_t qq = *d++ / 25;
                    if(qq>9)qq=9;
                    putchar(qq ? qq+'0': '.');
                }
                putchar('\n');
            }
        }
    }        
    
private:
    /**
     * \brief is the data valid - false if there was a problem in loading
     */
    bool valid;
    
    /**
     * \brief the number of rows in each image
     */
    uint32_t rows;
    
    /**
     * \brief the number of columns in each image
     */
    uint32_t cols;
    
    /**
     * \brief the number of images in the data set
     */
    uint32_t ct;
    
    /**
     * \brief the maximum label value
     */
    uint8_t maxLabel;
    
    /**
     * \brief pointer to the label data
     */
    
    uint8_t *labels;
    
    /**
     * \brief pointer to the image data
     */
    uint8_t *imgs;
};


#endif /* __MNIST_HPP */
