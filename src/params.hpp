#include <Eigen/Core>

#ifndef PARAMS_HPP
#define PARAMS_HPP

namespace pq
{
    namespace opt
    {
        class NNModel;
    }
    namespace Value
    {
        // Constants / predefined values
        namespace Constant
        {
            constexpr double g = 9.81;
            constexpr double gm = 1.61;
            constexpr double mass = 1.0;
            constexpr double length = 0.3;
            constexpr double inertia = 0.2 * mass * length * length;
        }

        // Parameters for simulation and optimization
        namespace Param
        {
            namespace Sim
            {
                constexpr double dt = 0.05;
                constexpr bool sync_with_real_time = false;
            }
            namespace Opt
            {
                constexpr int target_x = 10;
                constexpr int target_y = 10;
                constexpr double g = Constant::gm;
                constexpr double mass = Constant::mass;
                constexpr double length = Constant::length;
                constexpr double inertia = Constant::inertia;
                constexpr int steps = 50;
                constexpr int horizon = 20;
                constexpr int pop_size = 200;
                constexpr int num_elites = 32;
                constexpr double max_value = Constant::mass * Constant::g;
                constexpr double min_value = 0.0;
                constexpr double init_mu = 0.5 * mass * Constant::g; // TODO: should use Opt::g here?
                constexpr double init_std = 0.3;
            }
            namespace NN
            {
                constexpr bool use = true;
                constexpr int epochs = 10000;
                constexpr int collection_steps = 1000;
            }
        }

        Eigen::Vector<double, 6> init_state;
        Eigen::Vector<double, 6> target;

        std::unique_ptr<pq::opt::NNModel> learned_model;
    }
}

#endif