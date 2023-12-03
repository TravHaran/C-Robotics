#include "KalmanFilter.h"
#include "Eigen/Dense"
#include <iostream>

int main(int argc, char** argv){
    
    constexpr int32_t DIM_X{ 2 };
    constexpr int32_t DIM_Z{ 1 };
    kf::KalmanFilter<DIM_X, DIM_Z> kalmanFilter; // estimating state of 2 dimensions, and measrung one state which is position

    kalmanFilter.vecX() << 0.0F, 1.0F; // set position to 0 and velocity to 1
    kalmanFilter.matP() << 1.0F, 0.0F, 
                            0.0F, 1.0F; // set state covariance to identity matrix
    
    Eigen::Matrix<float, DIM_X, DIM_X> matF;
    // set diagonals to 1 to map to previous state, 
    // set top right to 1 so that we integrate the velocity at every prediction step to the previous position state to accumulate it and get the new position
    matF << 1.0F, 1.0F, 
            0.0F, 1.0F;
    
    Eigen::Matrix<float, DIM_X, DIM_X> matQ;
    matQ << 0.5F, 0.0F, 
            0.0F, 0.5F;
    
    kalmanFilter.prediction(matF, matQ);

    std::cout<<"After Prediction Step"<<std::endl;
    std::cout<<"x = \n"<<kalmanFilter.vecX()<<std::endl;
    std::cout<<"P = \n"<<kalmanFilter.matP()<<std::endl;
    std::cout<<std::endl;

    Eigen::Matrix<float, DIM_Z, DIM_Z> matR;
    // 1x1 matrix
    matR << 0.1F;
    
    Eigen::Vector<float, DIM_Z> vecZ;
    vecZ << 1.2F; // set measurement to 1.2

    Eigen::Matrix<float, DIM_Z, DIM_X> matH; // measurement model
    // since we're only looking at position we set the first value to 1 as that maps to the first element in the state vector
    matH << 1.0F, 0.0F;

    kalmanFilter.correction(vecZ, matH, matR);

    std::cout<<"After Correction Step"<<std::endl;
    std::cout<<"x = \n"<<kalmanFilter.vecX()<<std::endl;
    std::cout<<"P = \n"<<kalmanFilter.matP()<<std::endl;
    std::cout<<std::endl;

    return 0;
}

/// Expected Result
// After Prediction Step
// x = 
// 1
// 1
// P = 
// 2.5   1
//   1 1.5

// After Correction Step
// x = 
// 1.19231
// 1.07692
// P = 
// 0.0961538 0.0384615
// 0.0384615   1.11538
