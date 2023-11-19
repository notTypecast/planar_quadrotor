#ifndef INDIVIDUAL_HPP
#define INDIVIDUAL_HPP
#include <Eigen/Core>

#include "src/params.hpp"
#include "src/opt/DynamicModel.hpp"
#include "src/opt/LearnedModel.hpp"

namespace pq
{
    namespace opt
    {
        struct ControlIndividual
        {
            static constexpr unsigned int dim = 2 * pq::Value::Param::Opt::horizon;

            // Individual: [u11, u12, ..., uh1, uh2]  (size 2h)
            double eval(const Eigen::Matrix<double, 1, dim> &x)
            {
                Eigen::Vector<double, 6> state = pq::Value::init_state;
                double cost = 0;
                for (int i = 0; i < dim; i += 2)
                {
                    Eigen::Vector2d controls = x.block<1, 2>(0, i).transpose();
                    Eigen::Vector3d ddq = pq::opt::dynamic_model_predict(state, controls);
                    if (pq::Value::learned_model->trained())
                    {
                        ddq += pq::Value::learned_model->predict((Eigen::Vector<double, 8>() << state, controls).finished());
                    }

                    state.segment(0, 3) += state.segment(3, 3) * pq::Value::Param::Sim::dt + 0.5 * ddq * pq::Value::Param::Sim::dt * pq::Value::Param::Sim::dt;
                    state.segment(3, 3) += ddq * pq::Value::Param::Sim::dt;
                    cost += (pq::Value::target.segment(0, 3) - state.segment(0, 3)).squaredNorm() + 3.5 * i / dim * state.segment(3, 3).squaredNorm();
                }

                return -cost;
            }
        };
    }
}

#endif
