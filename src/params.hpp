#ifndef PQ_PARAMS_HPP
#define PQ_PARAMS_HPP
#include <Eigen/Core>
#include <memory>

namespace symnn
{
    class SymNN;
}

namespace pq
{
    namespace cem_opt
    {
        class NNModel;
    }
    namespace Value
    {
        // Constants / predefined values
        namespace Constant
        {
            constexpr double g = 9.81;                               // earth gravity
            constexpr double gm = 1.61;                              // moon gravity
            constexpr double mass = 1.0;                             // default quadrotor mass
            constexpr double length = 0.3;                           // default total quadrotor length
            constexpr double inertia = 0.2 * mass * length * length; // default quadrotor inertia
        }

        // Parameters for simulation and optimization
        namespace Param
        {
            namespace Sim
            {
                constexpr double dt = 0.05;                 // simulation time step
                constexpr bool sync_with_real_time = true; // whether to sync simulation with real time (ratio <= 1)
            }
            namespace CEMOpt
            {
                constexpr int target_x = 10;                                   // target x position
                constexpr int target_y = 10;                                   // target y position
                constexpr double g = Constant::g;                              // MPC gravity
                double mass;                                                   // MPC mass
                constexpr double length = Constant::length;                    // MPC length
                double inertia;                                                // MPC inertia
                constexpr int steps = 80;                                      // MPC steps
                constexpr int horizon = 20;                                    // MPC horizon (number of control inputs per individual)
                constexpr int pop_size = 200;                                  // population size
                constexpr int num_elites = 32;                                 // number of elites
                constexpr double max_value = Constant::mass * Constant::g;     // maximum control input force
                constexpr double min_value = 0.0;                              // minimum control input force
                constexpr double init_mu = 0.5 * Constant::mass * Constant::g; // initial mean for CEM
                constexpr double init_std = 0.3;                               // initial standard deviation for CEM
                Eigen::Vector4d model_params;
            }
            namespace SimpleNN
            {
#ifdef CEM_OPT
                std::unique_ptr<pq::cem_opt::NNModel> learned_model;
#endif
                constexpr int epochs = 10000; // number of epochs for training
            }
            namespace NumOpt
            {
                constexpr int target_x = 10;                           // target x position
                constexpr int target_y = 10;                           // target y position
                constexpr int horizon = 50;                           // Horizon
                constexpr double dt = Sim::dt;                         // optimization time step
                constexpr double F_max = 16 * Constant::mass * Constant::g; // maximum force
            }
            namespace SymNN
            {
#ifdef NUM_OPT
                std::unique_ptr<symnn::SymNN> learned_model;
#endif
                bool use = false;
                constexpr int epochs = 100;
            }
            namespace Train
            {
                constexpr int collection_steps = 50; // number of steps to collect data for training (per episode)
                constexpr int episodes = 10;          // number of episodes to train
                constexpr int runs = 5;               // number of runs to train (for averaging)
            }
        }

        Eigen::Vector<double, 6> init_state;
        Eigen::Vector<double, 6> target;
    }
}

#endif