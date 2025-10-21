#include "Agent.h"

#include "settings.h"
#include "helpers.h"
#include <stdio.h>
#include <iostream>
#include <string>
#include "DWRAONBrain.h"
#include "MLPBrain.h"
#include "AssemblyBrain.h"

using namespace std;
Agent::Agent()
{
    pos= Vector2f(randf(0,conf::WIDTH),randf(0,conf::HEIGHT));
    angle= randf(-M_PI,M_PI);
    health= 1.0+randf(0,0.1);
    age=0;
    spikeLength=0;
    red= 0;
    gre= 0;
    blu= 0;
    w1=0;
    w2=0;
    soundmul=1;
    give=0;
    clockf1= randf(5,100);
    clockf2= randf(5,100);
    boost=false;
    indicator=0;
    gencount=0;
    selectflag=0;
    ir=0;
    ig=0;
    ib=0;
    temperature_preference=randf(0,1);
    hybrid= false;
    herbivore= randf(0,1);
    repcounter= herbivore*randf(conf::REPRATEH-0.1,conf::REPRATEH+0.1) + (1-herbivore)*randf(conf::REPRATEC-0.1,conf::REPRATEC+0.1);

    id=0;
    
    smellmod= randf(0.1, 0.5);
    soundmod= randf(0.2, 0.6);
    hearmod= randf(0.7, 1.3);
    eyesensmod= randf(1, 3);
    bloodmod= randf(1, 3);
    
    MUTRATE1= randf(0.001, 0.005);
    MUTRATE2= randf(0.03, 0.07);

    spiked= false;
    
    in.resize(INPUTSIZE, 0);
    out.resize(OUTPUTSIZE, 0);
    
    eyefov.resize(NUMEYES, 0);
    eyedir.resize(NUMEYES, 0);
    for(int i=0;i<NUMEYES;i++) {
        eyefov[i] = randf(0.5, 2);
        eyedir[i] = randf(0, 2*M_PI);
    }
}

void Agent::printSelf()
{
    printf("Agent age=%i\n", age);
    for (int i=0;i<mutations.size();i++) {
        cout << mutations[i];
    }
}

void Agent::initEvent(float size, float r, float g, float b)
{
    indicator=size;
    ir=r;
    ig=g;
    ib=b;
}

void Agent::tick()
{
    brain.tick(in, out);
}
Agent Agent::reproduce(float MR, float MR2)
{
    bool BDEBUG = false;
    if(BDEBUG) printf("New birth---------------\n");
    Agent a2;

    //spawn the baby somewhere closeby behind agent
    //we want to spawn behind so that agents dont accidentally eat their young right away
    Vector2f fb(conf::BOTRADIUS,0);
    fb.rotate(-a2.angle);
    a2.pos= this->pos + fb + Vector2f(randf(-conf::BOTRADIUS*2,conf::BOTRADIUS*2), randf(-conf::BOTRADIUS*2,conf::BOTRADIUS*2));
    if (a2.pos.x<0) a2.pos.x= conf::WIDTH+a2.pos.x;
    if (a2.pos.x>=conf::WIDTH) a2.pos.x= a2.pos.x-conf::WIDTH;
    if (a2.pos.y<0) a2.pos.y= conf::HEIGHT+a2.pos.y;
    if (a2.pos.y>=conf::HEIGHT) a2.pos.y= a2.pos.y-conf::HEIGHT;

    a2.gencount= this->gencount+1;
    a2.repcounter= a2.herbivore*randf(conf::REPRATEH-0.1,conf::REPRATEH+0.1) + (1-a2.herbivore)*randf(conf::REPRATEC-0.1,conf::REPRATEC+0.1);

    //noisy attribute passing
    a2.MUTRATE1= this->MUTRATE1;
    a2.MUTRATE2= this->MUTRATE2;
    if (randf(0,1)<0.1) a2.MUTRATE1= randn(this->MUTRATE1, conf::METAMUTRATE1);
    if (randf(0,1)<0.1) a2.MUTRATE2= randn(this->MUTRATE2, conf::METAMUTRATE2);
    if (this->MUTRATE1<0.001) this->MUTRATE1= 0.001;
    if (this->MUTRATE2<0.02) this->MUTRATE2= 0.02;
    a2.herbivore= cap(randn(this->herbivore, 0.03));
    if (randf(0,1)<MR*5) a2.clockf1= randn(a2.clockf1, MR2);
    if (a2.clockf1<2) a2.clockf1= 2;
    if (randf(0,1)<MR*5) a2.clockf2= randn(a2.clockf2, MR2);
    if (a2.clockf2<2) a2.clockf2= 2;
    
    a2.smellmod = this->smellmod;
    a2.soundmod = this->soundmod;
    a2.hearmod = this->hearmod;
    a2.eyesensmod = this->eyesensmod;
    a2.bloodmod = this->bloodmod;
    if(randf(0,1)<MR*5) {float oo = a2.smellmod; a2.smellmod = randn(a2.smellmod, MR2); if(BDEBUG) printf("smell mutated from %f to %f\n", oo, a2.smellmod);}
    if(randf(0,1)<MR*5) {float oo = a2.soundmod; a2.soundmod = randn(a2.soundmod, MR2); if(BDEBUG) printf("sound mutated from %f to %f\n", oo, a2.soundmod);}
    if(randf(0,1)<MR*5) {float oo = a2.hearmod; a2.hearmod = randn(a2.hearmod, MR2); if(BDEBUG) printf("hear mutated from %f to %f\n", oo, a2.hearmod);}
    if(randf(0,1)<MR*5) {float oo = a2.eyesensmod; a2.eyesensmod = randn(a2.eyesensmod, MR2); if(BDEBUG) printf("eyesens mutated from %f to %f\n", oo, a2.eyesensmod);}
    if(randf(0,1)<MR*5) {float oo = a2.bloodmod; a2.bloodmod = randn(a2.bloodmod, MR2); if(BDEBUG) printf("blood mutated from %f to %f\n", oo, a2.bloodmod);}
    
    a2.eyefov = this->eyefov;
    a2.eyedir = this->eyedir;
    for(int i=0;i<NUMEYES;i++){
        if(randf(0,1)<MR*5) a2.eyefov[i] = randn(a2.eyefov[i], MR2);
        if(a2.eyefov[i]<0) a2.eyefov[i] = 0;
        
        if(randf(0,1)<MR*5) a2.eyedir[i] = randn(a2.eyedir[i], MR2);
        if(a2.eyedir[i]<0) a2.eyedir[i] = 0;
        if(a2.eyedir[i]>2*M_PI) a2.eyedir[i] = 2*M_PI;
    }
    
    a2.temperature_preference= cap(randn(this->temperature_preference, 0.005));
//    a2.temperature_preference= this->temperature_preference;
    
    //mutate brain here
    a2.brain= this->brain;
    a2.brain.mutate(MR,MR2);
    
    return a2;

}

Agent Agent::crossover(const Agent& other)
{
    //this could be made faster by returning a pointer
    //instead of returning by value
    Agent anew;
    anew.hybrid=true; //set this non-default flag
    anew.gencount= this->gencount;
    if (other.gencount<anew.gencount) anew.gencount= other.gencount;

    //agent heredity attributes
    anew.clockf1= randf(0,1)<0.5 ? this->clockf1 : other.clockf1;
    anew.clockf2= randf(0,1)<0.5 ? this->clockf2 : other.clockf2;
    anew.herbivore= randf(0,1)<0.5 ? this->herbivore : other.herbivore;
    anew.MUTRATE1= randf(0,1)<0.5 ? this->MUTRATE1 : other.MUTRATE1;
    anew.MUTRATE2= randf(0,1)<0.5 ? this->MUTRATE2 : other.MUTRATE2;
    anew.temperature_preference = randf(0,1)<0.5 ? this->temperature_preference : other.temperature_preference;
    
    anew.smellmod= randf(0,1)<0.5 ? this->smellmod : other.smellmod;
    anew.soundmod= randf(0,1)<0.5 ? this->soundmod : other.soundmod;
    anew.hearmod= randf(0,1)<0.5 ? this->hearmod : other.hearmod;
    anew.eyesensmod= randf(0,1)<0.5 ? this->eyesensmod : other.eyesensmod;
    anew.bloodmod= randf(0,1)<0.5 ? this->bloodmod : other.bloodmod;
    
    anew.eyefov= randf(0,1)<0.5 ? this->eyefov : other.eyefov;
    anew.eyedir= randf(0,1)<0.5 ? this->eyedir : other.eyedir;
    
    anew.brain= this->brain.crossover(other.brain);
    
    return anew;
}
