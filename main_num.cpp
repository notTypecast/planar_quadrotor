#include <Eigen/Core>
#include <iostream>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <memory>

#define NUM_OPT 1

#include "src/params.hpp"
#include "src/sim/PlanarQuadrotor.hpp"
#include "src/sim/Visualizer.hpp"
#include "src/opt/num-opt/SymNN.hpp"
#include "src/opt/num-opt/NumOptimizer.hpp"
#include "src/train/Episode.hpp"

int main(int argc, char **argv)
{
    pq::Value::target << pq::Value::Param::CEMOpt::target_x, pq::Value::Param::CEMOpt::target_y, 0, 0, 0, 0;
    pq::Value::Param::SymNN::learned_model = std::make_unique<symnn::SymNN>(8, 3, std::vector<int>{4, 3, 2});
    

    double masses[] = {4, 8, 16};
    std::vector<std::vector<double>> errors_per_episode(pq::Value::Param::Train::runs * pq::Value::Param::Train::episodes);
    pq::sim::Visualizer v;

    for (int i = 0; i < 3; ++i)
    {
        pq::num_opt::Params params;
        params.m = masses[i];
        params.I = 0.2 * masses[i] * pq::Value::Param::CEMOpt::length * pq::Value::Param::CEMOpt::length;
        params.init = Eigen::Vector3d(0, 0, 0);
        params.target << pq::Value::Param::CEMOpt::target_x, pq::Value::Param::CEMOpt::target_y, 0;

        pq::num_opt::NumOptimizer optimizer(params);

        std::cout << "Running with mass = " << masses[i] << std::endl;

        for (int j = 0; j < pq::Value::Param::Train::runs; ++j)
        {
            std::srand(std::time(NULL));
            std::cout << "Run " << j << std::endl;
            pq::Value::Param::SymNN::learned_model->reset();
            pq::Value::Param::SymNN::use = false;
            pq::train::Episode episode;
            episode.set_run(j + 1);

            int run_idx = j * pq::Value::Param::Train::episodes;
            std::cout << "Episode: ";
            for (int k = 0; k < pq::Value::Param::Train::episodes; ++k)
            {
                std::cout << k << " " << std::flush;
                errors_per_episode[run_idx + k] = episode.run(optimizer);
                /*
                // Early stopping if the change in error is smaller than a threshold
                // This will not work if writing to a file
                if (k > 0 && errors_per_episode[run_idx + k - 1][pq::Value::Param::Train::collection_steps - 1] - errors_per_episode[run_idx + k][pq::Value::Param::Train::collection_steps - 1] < 10e-4)
                {
                    std::cout << ":done";
                    break;
                }
                */

                pq::Value::Param::SymNN::learned_model->train(episode.get_train_input(), episode.get_train_target());
                pq::Value::Param::SymNN::use = true;
                std::cout << episode.get_train_target().col(0).transpose() << std::endl;
                std::cout << "NN sample: " << pq::Value::Param::SymNN::learned_model->forward(episode.get_train_input().col(0)).transpose() << std::endl;
            }
            std::cout << std::endl;
        }

        std::ofstream out("sample_error/error_" + std::to_string(pq::Value::Param::CEMOpt::mass) + ".txt");
        out << pq::Value::Param::CEMOpt::mass << " "
            << pq::Value::Param::Train::collection_steps << " "
            << pq::Value::Param::Train::episodes << " "
            << pq::Value::Param::Train::runs << " "
            << std::endl;
        for (int j = 0; j < pq::Value::Param::Train::runs; ++j)
        {
            int run_idx = j * pq::Value::Param::Train::episodes;
            for (int k = 0; k < pq::Value::Param::Train::episodes; ++k)
            {
                for (int l = 0; l < pq::Value::Param::Train::collection_steps; ++l)
                {
                    out << errors_per_episode[run_idx + k][l] << " ";
                }
                out << std::endl;
            }
        }
        out.close();
    }

    return 0;
}