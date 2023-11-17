#include <iostream>
#include <chrono>
#include <sstream>
#include <iomanip>

#include "src/params.hpp"
#include "src/sim/PlanarQuadrotor.hpp"
#include "src/sim/Visualizer.hpp"
#include "src/opt/Individual.hpp"
#include "src/opt/LearnedModel.hpp"

#include "algevo/cem.hpp"

using Algo = algevo::algo::CrossEntropyMethod<pq::opt::ControlIndividual>;

int main(int argc, char **argv)
{
    Algo::Params params;
    params.dim = pq::opt::ControlIndividual::dim;
    params.pop_size = pq::Value::Param::Opt::pop_size;
    params.num_elites = pq::Value::Param::Opt::num_elites;
    params.max_value = Algo::x_t::Constant(params.dim, pq::Value::Param::Opt::max_value);
    params.min_value = Algo::x_t::Constant(params.dim, pq::Value::Param::Opt::min_value);
    params.init_mu = Algo::x_t::Constant(params.dim, pq::Value::Param::Opt::init_mu);
    params.init_std = Algo::x_t::Constant(params.dim, pq::Value::Param::Opt::init_std);

    pq::sim::PlanarQuadrotor p(pq::Value::Constant::mass, pq::Value::Constant::inertia, pq::Value::Constant::length);
    pq::sim::Visualizer v;

    pq::Value::target << pq::Value::Param::Opt::target_x, pq::Value::Param::Opt::target_y, 0, 0, 0, 0;

    Eigen::MatrixXd train_input;
    Eigen::MatrixXd train_target;

    if (pq::Value::Param::NN::use)
    {
        pq::Value::learned_model = std::make_unique<pq::opt::NNModel>(std::vector<int>{12, 6, 4});
        train_input.resize(8, pq::Value::Param::NN::collection_steps);
        train_target.resize(3, pq::Value::Param::NN::collection_steps);
    }

    double control_freq = 0;
    int count = 0;
    std::chrono::duration<double> elapsed = std::chrono::duration<double>::zero();
    std::chrono::duration<double> total_time = std::chrono::duration<double>::zero();
    auto real_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < INT_MAX; ++i)
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

        Algo cem(params);

        for (int j = 0; j < pq::Value::Param::Opt::steps; ++j)
        {
            cem.step();
        }
        auto end = std::chrono::high_resolution_clock::now();

        elapsed += end - start;

        Eigen::Vector2d controls = cem.best().segment(0, 2);

        p.update(controls, pq::Value::Param::Sim::dt);
        if (pq::Value::Param::NN::use)
        {
            if (i < pq::Value::Param::NN::collection_steps)
            {
                train_input.col(i) = (Eigen::Vector<double, 8>() << pq::Value::init_state, controls).finished();
                train_target.col(i) = p.get_last_ddq() - pq::opt::dynamic_model_predict(pq::Value::init_state, controls);
            }
            else if (i == pq::Value::Param::NN::collection_steps)
            {
                system("clear");
                std::cout << "Training model..." << std::endl;
                pq::Value::learned_model->train(train_input, train_target);
            }
        }

        std::stringstream ss;
        ss << std::fixed << std::setprecision(2) << "Control frequency: " << control_freq
           << " Hz, angle: " << p.get_state()[2] * 360 / M_PI
           << " deg, time: " << p.get_sim_time()
           << " sec (ratio " << pq::Value::Param::Sim::dt / std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - real_start).count()
           << "), MPC gravity: " << pq::Value::Param::Opt::g << " m/s^2";
        if (pq::Value::Param::NN::use)
        {
            ss << ", NN: " << (i < pq::Value::Param::NN::collection_steps ? "collecting data" : "trained");
        }
        v.set_message(ss.str());
        v.show(p, {pq::Value::Param::Opt::target_x, pq::Value::Param::Opt::target_y});

        ++count;
        if (elapsed.count() > 1)
        {
            control_freq = count / elapsed.count();
            count = 0;
            elapsed = std::chrono::duration<double>::zero();
        }
    }

    return 0;
}