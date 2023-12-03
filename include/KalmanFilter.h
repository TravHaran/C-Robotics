#include <stdint.h>
#include "Eigen/Dense"

namespace kf {
    template<int32_t DIM_X, int32_t DIM_Z> // two main arguments: dimension of state vector and dimension of measurement vector
    class KalmanFilter{
        public:
            KalmanFilter() = default;

            void prediction(const Eigen::Matrix<float, DIM_X, DIM_X>& matF, const Eigen::Matrix<float, DIM_X, DIM_X>& matQ){
                m_vecX = matF * m_vecX;
                m_matP = matF * m_matP * matF.transpose() + matQ;
            }

            void correction(const Eigen::Vector<float, DIM_Z>& vecZ, const Eigen::Matrix<float, DIM_Z, DIM_X>& matH, const Eigen::Matrix<float, DIM_Z, DIM_Z>& matR){
                const Eigen::Vector<float, DIM_Z> vecY{ vecZ - (matH * m_vecX) }; // innovation vector
                const Eigen::Matrix<float, DIM_Z, DIM_Z> matS{ matH * m_matP * matH.transpose() + matR }; //innovation covariance
                const Eigen::Matrix<float, DIM_X, DIM_Z> matK{ m_matP * matH.transpose() * matS.inverse() }; // kalmann gain
                const Eigen::Matrix<float, DIM_X, DIM_X> matI{ Eigen::Matrix<float, DIM_X, DIM_X>::Identity() }; // identity matrix
                //correction step
                m_vecX += matK * vecY;
                m_matP = (matI - matK * matH) * m_matP;
            }

            Eigen::Vector<float, DIM_X>& vecX() { return m_vecX; }
            const Eigen::Vector<float, DIM_X>& vecX() const { return m_vecX; }

            Eigen::Matrix<float, DIM_X, DIM_X>& matP() { return m_matP; }
            const Eigen::Matrix<float, DIM_X, DIM_X> matP() const { return m_matP; }
        private:
            ///
            /// @brief state vector
            ///
            Eigen::Vector<float, DIM_X> m_vecX;

            ///
            /// @brief state covariance matrix
            ///
            Eigen::Matrix<float, DIM_X, DIM_X> m_matP; \
    };
}