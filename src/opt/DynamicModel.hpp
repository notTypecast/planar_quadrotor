#ifndef DYNAMIC_MODEL_HPP
#define DYNAMIC_MODEL_HPP
#include <Eigen/Core>

#include "src/params.hpp"

namespace pq
{
    namespace opt
    {
        Eigen::Vector3d dynamic_model_predict(const Eigen::Vector<double, 6> &state, const Eigen::Vector2d &controls)
        {
            Eigen::Vector3d ddq;
            ddq[0] = -(controls[0] + controls[1]) * sin(state[2]) / pq::Value::Param::Opt::mass;
            ddq[1] = (controls[0] + controls[1]) * cos(state[2]) / pq::Value::Param::Opt::mass - pq::Value::Param::Opt::g;
            ddq[2] = (controls[1] - controls[0]) * pq::Value::Param::Opt::length / (2 * pq::Value::Param::Opt::inertia);
            return ddq;
        }
    }
}

#endif
