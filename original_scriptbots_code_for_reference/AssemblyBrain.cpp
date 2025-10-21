#include "AssemblyBrain.h"
#include <stdio.h>
using namespace std;

AssemblyBrain::AssemblyBrain()
{
    
    //constructor
    w.resize(BRAINSIZE, 0);
    for (int i=0;i<BRAINSIZE;i++) {
        w[i] = randf(-3,3);
        if(randf(0,1)<0.1) w[i] = randf(0,0.5);
        if(randf(0,1)<0.1) w[i] = randf(0.8,1);
    }
    
    //other inits
    init();
}
void AssemblyBrain::init()
{
}

AssemblyBrain::AssemblyBrain(const AssemblyBrain& other)
{
    w = other.w;
}

AssemblyBrain& AssemblyBrain::operator=(const AssemblyBrain& other)
{
    if( this != &other ) {
        w = other.w;
    }
    return *this;
}

void AssemblyBrain::tick(vector< float >& in, vector< float >& out)
{
    //do a single tick of the brain

    //take first few boxes and set their out to in[].
    for (int i=0;i<INPUTSIZE;i++) {
        w[i]= in[i];
    }
    
    //TICK!
    for (int i=INPUTSIZE;i<BRAINSIZE-OUTPUTSIZE;i++) {
        float v = w[i];
        
        //this is an operand
        if(v>=2 && v<3){
            float v1 = w[i+1];
            float v2 = w[i+2];
            float v3 = w[i+3];
            int d1 = (int)BRAINSIZE*(fabs(v1)-((int)fabs(v1)));
            int d2 = (int)BRAINSIZE*(fabs(v2)-((int)fabs(v2)));
            int d3 = (int)BRAINSIZE*(fabs(v3)-((int)fabs(v3)));
            
           // printf("%f d1=%d d2=%d d3=%d\n", v, d1, d2, d3);
            
            if(v<2.1) { w[d3] = w[d1] + w[d2]; continue;}
            if(v<2.2) { w[d3] = w[d1] - w[d2]; continue;}
            if(v<2.3) { w[d3] = w[d1] * w[d2]; continue;}
            if(v<2.4) { if(w[d3]>0) w[d1] = 0; continue;}            
            if(v<2.5) { if(w[d3]>0) w[d1] = -w[d1]; continue;}
            if(v<2.7) { if(w[d3]>0) w[d1] += v2; continue;}
            if(v<3) { if(w[d3]>0) w[d1] = w[d2]; continue;}
        }
    }
    
    //cap all to -10,10
    for(int i=INPUTSIZE;i<BRAINSIZE-OUTPUTSIZE;i++){
        float v = w[i];
        if(v>10)w[i]=10;
        if(v<-10)w[i]=-10;
    }
    
    //finally set out[] to the last few boxes output
    for (int i=0;i<OUTPUTSIZE;i++) {
        float v =w[BRAINSIZE-1-i];
        if(v>1)v=1;
        if(v<0)v=0;
        out[i]= v;
    }
}

void AssemblyBrain::mutate(float MR, float MR2)
{
    for (int j=0;j<BRAINSIZE;j++) {
        if(randf(0,1)<MR) {
            w[j] = randf(-3,3);
        }
    }
}

AssemblyBrain AssemblyBrain::crossover(const AssemblyBrain& other)
{
    //this could be made faster by returning a pointer
    //instead of returning by value
    AssemblyBrain newbrain(*this);
    for (int i=0;i<newbrain.w.size(); i++) {
        if(randf(0,1)<0.5){
            newbrain.w[i] = this->w[i];
        } else {
            newbrain.w[i] = other.w[i];
        }
    }
    return newbrain;
}

