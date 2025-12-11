/**********************************************************************
 Copyright (c) 2020-2023, Unitree Robotics.Co.Ltd. All rights reserved.
***********************************************************************/
#ifndef TIMEMARKER_H
#define TIMEMARKER_H

#include <iostream>
#include <sys/time.h>
#include <unistd.h>


// static long long success = 0;
// static long long total = 0;

//时间戳  微秒级， 需要#include <sys/time.h> 
inline long long getSystemTime(){
    struct timeval t;  
    gettimeofday(&t, NULL);
    return 1000000 * t.tv_sec + t.tv_usec;  
}
//时间戳  秒级， 需要getSystemTime()
inline double getTimeSecond(){
    double time = getSystemTime() * 0.000001;
    return time;
}
//等待函数，微秒级，从startTime开始等待waitTime微秒
inline int absoluteWait(long long startTime, long long waitTime){
    if(getSystemTime() - startTime > waitTime){
        // total++;
        // std::cout << "[WARNING] The waitTime=" << waitTime << " of function absoluteWait is not enough!" << std::endl
        // << "The program has already cost " << getSystemTime() - startTime << "us." << std::endl;
        // std::cout << "Warning"<<std::endl;
        return 0;
    }
    while(getSystemTime() - startTime < waitTime){
        // std::cout << "Safe"<<std::endl;
        // success++;
        // total++;
        usleep(50);
    }
    return 1;

    // std::cout << "absoluteWait success rate: " << (double)success/total*100 << "%"<< std::endl;
    
}

#endif //TIMEMARKER_H