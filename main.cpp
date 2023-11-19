#include <Eigen/Core>
#include <iostream>
#include <chrono>
#include <sstream>
#include <iomanip>

#include "src/params.hpp"
#include "src/sim/PlanarQuadrotor.hpp"
#include "src/sim/Visualizer.hpp"
#include "src/opt/Individual.hpp"
#include "src/opt/LearnedModel.hpp"
#include "src/train/Episode.hpp"

#include "algevo/src/algevo/algo/cem.hpp"

using Algo = algevo::algo::CrossEntropyMethod<pq::opt::ControlIndividual>;

int main(int argc, char **argv)
{
    pq::Value::target << pq::Value::Param::Opt::target_x, pq::Value::Param::Opt::target_y, 0, 0, 0, 0;

    pq::Value::learned_model = std::make_unique<pq::opt::NNModel>(std::vector<int>{12, 6, 4});

    pq::train::Episode episode;

    for (int i = 0; i < pq::Value::Param::NN::episodes; ++i)
    {
        episode.run();
        pq::Value::learned_model->train(episode.get_train_input(), episode.get_train_target());
    }

    return 0;
}