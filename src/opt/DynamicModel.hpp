#ifndef PQ_DYNAMIC_MODEL_HPP
#define PQ_DYNAMIC_MODEL_HPP
#include <Eigen/Core>

#include "src/params.hpp"

namespace pq
{
    Eigen::Vector3d dynamic_model_predict(const Eigen::Vector<double, 6> &state, const Eigen::Vector2d &controls, const Eigen::Vector4d &model_params)
    {
        Eigen::Vector3d ddq;
        ddq[0] = -(controls[0] + controls[1]) * sin(state[2]) / model_params[0];
        ddq[1] = (controls[0] + controls[1]) * cos(state[2]) / model_params[0] - model_params[3];
        ddq[2] = (controls[1] - controls[0]) * model_params[2] / (2 * model_params[1]);
        return ddq;
    }
}

#endif
