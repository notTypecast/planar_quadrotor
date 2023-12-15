#include <Eigen/Core>
#include <iostream>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <memory>

#include "src/params.hpp"
#include "src/sim/PlanarQuadrotor.hpp"
#include "src/sim/Visualizer.hpp"
#include "src/opt/Individual.hpp"
#include "src/opt/LearnedModel.hpp"
#include "src/train/Episode.hpp"

using Algo = algevo::algo::CrossEntropyMethod<pq::opt::ControlIndividual>;

int main(int argc, char **argv)
{
    pq::Value::target << pq::Value::Param::Opt::target_x, pq::Value::Param::Opt::target_y, 0, 0, 0, 0;
    pq::Value::learned_model = std::make_unique<pq::opt::NNModel>(std::vector<int>{12, 6, 4});

    double gs[] = {18, 39, 100};
    std::shared_ptr<double> errors(new double[pq::Value::Param::Train::collection_steps * pq::Value::Param::Train::episodes * pq::Value::Param::Train::runs]);
    memset(errors.get(), 0, pq::Value::Param::Train::collection_steps * pq::Value::Param::Train::episodes * pq::Value::Param::Train::runs * sizeof(double));
    // pq::sim::Visualizer v;

    for (int i = 0; i < 3; ++i)
    {
        pq::Value::Param::Opt::g = gs[i];
        std::cout << "Running with g = " << pq::Value::Param::Opt::g << std::endl;

        for (int j = 0; j < pq::Value::Param::Train::runs; ++j)
        {
            std::srand(std::time(NULL));
            std::cout << "Run " << j << std::endl;
            pq::Value::learned_model->reset();
            pq::train::Episode episode;
            episode.set_error_array(errors);
            episode.set_run(j + 1);

            std::cout << "Episode: ";
            for (int k = 0; k < pq::Value::Param::Train::episodes; ++k)
            {
                std::cout << k << " " << std::flush;
                episode.run();
                if (k > 0 && errors.get()[k * pq::Value::Param::Train::collection_steps - 1] - errors.get()[(k + 1) * pq::Value::Param::Train::collection_steps - 1] < 10e-4)
                {
                    std::cout << ":done";
                    break;
                }

                pq::Value::learned_model->train(episode.get_train_input(), episode.get_train_target());
            }
            std::cout << std::endl;
        }

        std::ofstream out("sample_error/error_" + std::to_string(pq::Value::Param::Opt::g) + ".txt");
        out << pq::Value::Param::Opt::g << " "
            << pq::Value::Param::Train::collection_steps << " "
            << pq::Value::Param::Train::episodes << " "
            << pq::Value::Param::Train::runs << " "
            << pq::Value::Param::Opt::target_x << " "
            << pq::Value::Param::Opt::target_y << " "
            << std::endl;
        for (int j = 0; j < pq::Value::Param::Train::runs; ++j)
        {
            int run_idx = j * pq::Value::Param::Train::episodes * pq::Value::Param::Train::collection_steps;
            for (int k = 0; k < pq::Value::Param::Train::episodes; ++k)
            {
                int episode_idx = run_idx + k * pq::Value::Param::Train::collection_steps;
                for (int l = 0; l < pq::Value::Param::Train::collection_steps; ++l)
                {
                    out << errors.get()[episode_idx + l] << " ";
                }
                out << std::endl;
            }
        }
        out.close();
    }

    return 0;
}