#include <iostream>
#include <chrono>
#include "src/PlanarQuadrotor.hpp"
#include "src/Visualizer.hpp"

#include "algevo/cem.hpp"

namespace Value
{
    // Parameters for simulation and optimization
    namespace Param
    {
        namespace Sim
        {
            constexpr double dt = 0.05;
        };
        namespace Opt
        {
            constexpr double g = Value::Constant::gm;
            constexpr double mass = Value::Constant::mass;
            constexpr double length = Value::Constant::length;
            constexpr double inertia = Value::Constant::inertia;
            constexpr int horizon = 20;
            constexpr int pop_size = 128;
            constexpr int num_elites = 32;
            constexpr double max_value = Value::Constant::mass * Value::Constant::g;
            constexpr double min_value = 0.0;
            constexpr double init_mu = 0.5 * mass * g;
            constexpr double init_std = 0.3;
            constexpr int target_x = 10;
            constexpr int target_y = 10;
        };
    };

    Eigen::Vector<double, 6> init_state;
    Eigen::Vector<double, 6> target;
};

struct ControlIndividual
{
    static constexpr unsigned int dim = 2 * Value::Param::Opt::horizon;

    // Individual: [u11, u12, ..., uh1, uh2]  (size 2h)
    double eval(const Eigen::Matrix<double, 1, dim> &x)
    {
        Eigen::Vector<double, 6> state = Value::init_state;
        int inv = 0;
        for (int i = 0; i < dim; i += 2)
        {
            Eigen::Vector3d ddq;
            ddq[0] = -(x[i] + x[i + 1]) * sin(state[2]) / Value::Param::Opt::mass;
            ddq[1] = (x[i] + x[i + 1]) * cos(state[2]) / Value::Param::Opt::mass - Value::Param::Opt::g;
            ddq[2] = (x[i + 1] - x[i]) * Value::Param::Opt::length / (2 * Value::Param::Opt::inertia);

            state.segment(0, 3) += state.segment(3, 3) * Value::Param::Sim::dt + 0.5 * ddq * Value::Param::Sim::dt * Value::Param::Sim::dt;
            state.segment(3, 3) += ddq * Value::Param::Sim::dt;
            if (abs(Value::target[2] - state[2]) > M_PI / 4)
            {
                inv += 5;
            }
        }

        return -(Value::target.segment(0, 2) - state.segment(0, 2)).norm() - 0.1 * (Value::target.segment(3, 2) - state.segment(3, 2)).norm() - inv;
    }
};

using Algo = algevo::algo::CrossEntropyMethod<ControlIndividual>;

int main(int argc, char **argv)
{
    Algo::Params params;
    params.dim = ControlIndividual::dim;
    params.pop_size = Value::Param::Opt::pop_size;
    params.num_elites = Value::Param::Opt::num_elites;
    params.max_value = Algo::x_t::Constant(params.dim, Value::Param::Opt::max_value);
    params.min_value = Algo::x_t::Constant(params.dim, Value::Param::Opt::min_value);
    params.init_mu = Algo::x_t::Constant(params.dim, Value::Param::Opt::init_mu);
    params.init_std = Algo::x_t::Constant(params.dim, Value::Param::Opt::init_std);

    PlanarQuadrotor p(Value::Constant::mass, Value::Constant::inertia, Value::Constant::length);
    Visualizer v;

    Value::target << Value::Param::Opt::target_x, Value::Param::Opt::target_y, 0, 0, 0, 0;

    int count = 0;
    std::chrono::duration<double> elapsed = std::chrono::duration<double>::zero();

    for (int i = 0; i < 100000; ++i)
    {
        Value::init_state = p.get_state();

        auto start = std::chrono::high_resolution_clock::now();

        Algo cem(params);

        for (int j = 0; j < 100; ++j)
        {
            cem.step();
        }
        auto end = std::chrono::high_resolution_clock::now();

        elapsed += end - start;

        Eigen::Vector2d controls = cem.best().segment(0, 2);

        p.update(controls, Value::Param::Sim::dt);
        v.show(p, {Value::Param::Opt::target_x, Value::Param::Opt::target_y});

        ++count;
        if (elapsed.count() > 1)
        {
            v.set_message("Control frequency: " + std::to_string(count / elapsed.count()) + " Hz, angle: " + std::to_string(p.get_state()[2] * 360 / M_PI) + " deg, time: " + std::to_string(p.get_sim_time()) + " sec");
            count = 0;
            elapsed = std::chrono::duration<double>::zero();
        }
    }

    return 0;
}