#ifndef EPISODE_HPP
#define EPISODE_HPP
#include <Eigen/Core>
#include <iostream>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <memory>

#include "src/params.hpp"
#include "src/sim/PlanarQuadrotor.hpp"
#include "src/sim/Visualizer.hpp"
#include "src/opt/Individual.hpp"

#include "algevo/src/algevo/algo/cem.hpp"

using Algo = algevo::algo::CrossEntropyMethod<pq::opt::ControlIndividual>;

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
                _params.dim = pq::opt::ControlIndividual::dim;
                _params.pop_size = pq::Value::Param::Opt::pop_size;
                _params.num_elites = pq::Value::Param::Opt::num_elites;
                _params.max_value = Algo::x_t::Constant(_params.dim, pq::Value::Param::Opt::max_value);
                _params.min_value = Algo::x_t::Constant(_params.dim, pq::Value::Param::Opt::min_value);
                _params.init_std = Algo::x_t::Constant(_params.dim, pq::Value::Param::Opt::init_std);
            }

            void run(pq::sim::Visualizer &v)
            {
                _visualize = true;
                _v = v;
                _run();
            }

            void run()
            {
                _visualize = false;
                _run();
            }

            Eigen::MatrixXd get_train_input()
            {
                return _train_input;
            }

            Eigen::MatrixXd get_train_target()
            {
                return _train_target;
            }

            void set_error_array(std::shared_ptr<double> errors)
            {
                _errors = errors;
            }

            void set_run(int run)
            {
                _run_iter = run;
            }

        private:
            Eigen::MatrixXd _train_input;
            Eigen::MatrixXd _train_target;
            Algo::Params _params;
            int _episode = 1;
            int _run_iter = -1;
            bool _visualize = false;
            pq::sim::Visualizer _v;
            std::shared_ptr<double> _errors = nullptr;

            void _run()
            {
                _params.init_mu = Algo::x_t::Constant(_params.dim, pq::Value::Param::Opt::init_mu);

                pq::sim::PlanarQuadrotor p(pq::Value::Constant::mass, pq::Value::Constant::inertia, pq::Value::Constant::length);

                double control_freq = 0;
                int count = 0;
                std::chrono::duration<double> elapsed = std::chrono::duration<double>::zero();
                std::chrono::duration<double> total_time = std::chrono::duration<double>::zero();
                auto real_start = std::chrono::high_resolution_clock::now();

                int episode_idx = (_run_iter - 1) * pq::Value::Param::Train::episodes * pq::Value::Param::Train::collection_steps + (_episode - 1) * pq::Value::Param::Train::collection_steps;

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

                    Algo cem(_params);

                    for (int j = 0; j < pq::Value::Param::Opt::steps; ++j)
                    {
                        cem.step();
                    }
                    auto end = std::chrono::high_resolution_clock::now();

                    elapsed += end - start;

                    _params.init_mu = cem.best();
                    Eigen::Vector2d controls = cem.best().segment(0, 2);

                    p.update(controls, pq::Value::Param::Sim::dt);
                    _train_input.col(i) = (Eigen::Vector<double, 8>() << pq::Value::init_state, controls).finished();
                    _train_target.col(i) = p.get_last_ddq() - pq::opt::dynamic_model_predict(pq::Value::init_state, controls);

                    std::stringstream ss;
                    ss << std::fixed << std::setprecision(2) << "Control frequency: " << control_freq
                       << " Hz, angle: " << p.get_state()[2] * 360 / M_PI
                       << " deg, time: " << p.get_sim_time()
                       << " sec (ratio " << pq::Value::Param::Sim::dt / std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - real_start).count()
                       << "), MPC gravity: " << pq::Value::Param::Opt::g
                       << " m/s^2, episode: " << _episode;
                    if (_run_iter != -1)
                    {
                        ss << ", run: " << _run_iter;
                        _errors.get()[episode_idx + i] = (pq::Value::target.segment(0, 6) - p.get_state().segment(0, 6)).squaredNorm();
                    }
                    if (_visualize)
                    {
                        _v.set_message(ss.str());
                        _v.show(p, {pq::Value::Param::Opt::target_x, pq::Value::Param::Opt::target_y});
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
            }
        };
    }
}

void run_episode(Eigen::MatrixXd &train_input, Eigen::MatrixXd &train_target)
{
    Algo::Params params;
    params.dim = pq::opt::ControlIndividual::dim;
    params.pop_size = pq::Value::Param::Opt::pop_size;
    params.num_elites = pq::Value::Param::Opt::num_elites;
    params.max_value = Algo::x_t::Constant(params.dim, pq::Value::Param::Opt::max_value);
    params.min_value = Algo::x_t::Constant(params.dim, pq::Value::Param::Opt::min_value);
    params.init_mu = Algo::x_t::Constant(params.dim, pq::Value::Param::Opt::init_mu);
    params.init_std = Algo::x_t::Constant(params.dim, pq::Value::Param::Opt::init_std);
}

#endif
