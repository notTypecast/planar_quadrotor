#ifndef PQ_EPISODE_HPP
#define PQ_EPISODE_HPP
#include <Eigen/Core>
#include <iostream>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <memory>

#include "src/params.hpp"
#include "src/opt/Optimizer.hpp"
#include "src/opt/DynamicModel.hpp"
#include "src/sim/PlanarQuadrotor.hpp"
#include "src/sim/Visualizer.hpp"

#include "algevo/src/algevo/algo/cem.hpp"

using Algo = algevo::algo::CrossEntropyMethod<pq::cem_opt::ControlIndividual>;

namespace pq
{
    namespace train
    {
        class Episode
        {
        public:
            Episode()
            {
                _train_input = Eigen::MatrixXd(8, pq::Value::Param::Train::collection_steps);
                _train_target = Eigen::MatrixXd(3, pq::Value::Param::Train::collection_steps);
            }

            std::vector<double> run(Optimizer &optimizer, pq::sim::Visualizer &v)
            {
                _visualize = true;
                _v = v;
                return _run(optimizer);
            }

            std::vector<double> run(Optimizer &optimizer)
            {
                _visualize = false;
                return _run(optimizer);
            }

            Eigen::MatrixXd get_train_input()
            {
                return _train_input;
            }

            Eigen::MatrixXd get_train_target()
            {
                return _train_target;
            }

            void set_run(int run)
            {
                _run_iter = run;
            }

        protected:
            Eigen::MatrixXd _train_input;
            Eigen::MatrixXd _train_target;
            int _episode = 1;
            int _run_iter = -1;
            bool _visualize = false;
            pq::sim::Visualizer _v;

            std::vector<double> _run(Optimizer &optimizer)
            {
                optimizer.reinit();

                pq::sim::PlanarQuadrotor p(pq::Value::Constant::mass, pq::Value::Constant::inertia, pq::Value::Constant::length);

                double control_freq = 0;
                int count = 0;
                std::chrono::duration<double> elapsed = std::chrono::duration<double>::zero();
                std::chrono::duration<double> total_time = std::chrono::duration<double>::zero();
                auto real_start = std::chrono::high_resolution_clock::now();

                int episode_idx = (_run_iter - 1) * pq::Value::Param::Train::episodes * pq::Value::Param::Train::collection_steps + (_episode - 1) * pq::Value::Param::Train::collection_steps;

                std::vector<double> errors(pq::Value::Param::Train::collection_steps, 0);

                for (int i = 0; i < pq::Value::Param::Train::collection_steps; ++i)
                {
                    total_time += std::chrono::high_resolution_clock::now() - real_start;
                    real_start = std::chrono::high_resolution_clock::now();
                    if (pq::Value::Param::Sim::sync_with_real_time)
                    {
                        if (p.get_sim_time() > total_time.count())
                        {
                            std::this_thread::sleep_for(std::chrono::milliseconds((int)(1000 * (p.get_sim_time() - total_time.count()))));
                        }
                    }

                    pq::Value::init_state = p.get_state();

                    auto start = std::chrono::high_resolution_clock::now();

                    Eigen::VectorXd controls = optimizer.next(p.get_state(), pq::Value::target);
                    std::cout << controls.transpose() << std::endl;

                    elapsed += std::chrono::high_resolution_clock::now() - start;

                    p.update(controls, pq::Value::Param::Sim::dt);
                    errors[i] = (pq::Value::target.segment(0, 6) - p.get_state().segment(0, 6)).squaredNorm();

                    _train_input.col(i) = (Eigen::Vector<double, 8>() << pq::Value::init_state, controls).finished();
                    _train_target.col(i) = p.get_last_ddq() - pq::dynamic_model_predict(pq::Value::init_state, controls, optimizer.model_params());

                    std::stringstream ss;
                    ss << std::fixed << std::setprecision(2) << "Control frequency: " << control_freq
                       << " Hz, angle: " << p.get_state()[2] * 360 / M_PI
                       << " deg, time: " << p.get_sim_time()
                       << " sec (ratio " << pq::Value::Param::Sim::dt / std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - real_start).count()
                       << "), MPC mass: " << pq::Value::Param::CEMOpt::mass
                       << " kg, episode: " << _episode;
                    if (_run_iter != -1)
                    {
                        ss << ", run: " << _run_iter;
                    }
                    if (_visualize)
                    {
                        _v.set_message(ss.str());
                        _v.show(p, {pq::Value::Param::CEMOpt::target_x, pq::Value::Param::CEMOpt::target_y});
                    }

                    ++count;
                    if (elapsed.count() > 1)
                    {
                        control_freq = count / elapsed.count();
                        count = 0;
                        elapsed = std::chrono::duration<double>::zero();
                    }
                }

                ++_episode;

                return errors;
            }
        };
    }
}

#endif
