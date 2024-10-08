#ifndef PQ_SYMNN_HPP
#define PQ_SYMNN_HPP

#include <vector>

#include <Eigen/Core>
#include <casadi/casadi.hpp>

#include "src/params.hpp"

using namespace casadi;

namespace symnn
{
    MX relu(const MX &x)
    {
        return fmax(0, x);
    }

    // Fully connected NN
    class SymNN
    {
    public:
        SymNN(int input_size, int output_size, const std::vector<int> &hidden_layers) : _input_size(input_size), _output_size(output_size)
        {
            _X = MX::sym("X", input_size);
            _Y = MX::sym("Y", output_size);

            std::vector<MX> all_params;
            all_params.push_back(_X);

            _W.push_back(MX::sym("W0", hidden_layers[0], input_size));
            _b.push_back(MX::sym("b0", hidden_layers[0]));
            MX prev = relu(mtimes(_W[0], _X) + _b[0]);

            all_params.push_back(_W[0]);
            all_params.push_back(_b[0]);

            for (int i = 1; i < hidden_layers.size(); ++i)
            {
                _W.push_back(MX::sym("W" + std::to_string(i), hidden_layers[i], hidden_layers[i - 1]));
                _b.push_back(MX::sym("b" + std::to_string(i), hidden_layers[i]));
                prev = relu(mtimes(_W[i], prev) + _b[i]);

                all_params.push_back(_W[i]);
                all_params.push_back(_b[i]);
            }

            _W.push_back(MX::sym("Wout", output_size, hidden_layers.back()));
            _b.push_back(MX::sym("bout", output_size));

            all_params.push_back(_W.back());
            all_params.push_back(_b.back());

            _out = mtimes(_W.back(), prev) + _b.back();
            _out_fn = Function("out", all_params, {_out});

            _loss = sumsqr(_Y - _out);

            for (int i = 1; i < all_params.size(); ++i)
            {
                _nn_values.push_back(DM::rand(all_params[i].size1(), all_params[i].size2()));
            }

            std::vector<MX> opt_vars(all_params.size() - 1);
            for (int i = 0; i < _W.size(); ++i)
            {
                opt_vars[2*i] = reshape(_W[i], -1, 1);
                opt_vars[2*i + 1] = _b[i];

            }

            _opt_vars = vertcat(opt_vars);
        }

        Eigen::VectorXd forward(const Eigen::VectorXd &input)
        {
            DM X = DM(input.size(), 1);
            for (int i = 0; i < input.size(); ++i)
            {
                X(i) = input(i);
            }

            std::vector<DM> params(_nn_values.begin(), _nn_values.end());
            params.insert(params.begin(), X);

            DM out = _out_fn(params)[0];

            Eigen::VectorXd output(_output_size);

            for (int i = 0; i < _output_size; ++i)
            {
                output(i) = static_cast<double>(out(i));
            }

            return output;
        }

        MX forward(MX &input)
        {
            return substitute(_out_substituted, _X, input);
        }

        void backward(const Eigen::VectorXd &input, const Eigen::VectorXd &target)
        {
            DM X = DM(input.size(), 1);
            for (int i = 0; i < input.size(); ++i)
            {
                X(i) = input(i);
            }

            DM Y = DM(target.size(), 1);
            for (int i = 0; i < target.size(); ++i)
            {
                Y(i) = target(i);
            }

            MX loss_sub = substitute(_loss, _X, X);
            loss_sub = substitute(loss_sub, _Y, Y);

            MXDict nlp = {
                {"x", _opt_vars},
                {"f", loss_sub}
            };

            Dict opts;
            opts["ipopt.print_level"] = 0;
            opts["print_time"] = false;
            opts["ipopt.tol"] = 1e-2;

            Function solver = nlpsol("solver", "ipopt", nlp, opts);

            std::vector<DM> params(_nn_values.size());
            for (int i = 0; i < _nn_values.size(); i += 2)
            {
                params[i] = reshape(_nn_values[i], -1, 1);
                params[i + 1] = _nn_values[i + 1];
            }

            DMDict args;
            args["x0"] = vertcat(params);

            DMDict result = solver(args);

            DM out = result.at("x");

            int offset = 0;
            for (int i = 0; i < _nn_values.size(); ++i)
            {
                int param_size = _nn_values[i].size1() * _nn_values[i].size2();
                _nn_values[i] = reshape(out(Slice(offset, offset + param_size), 0), _nn_values[i].size1(), _nn_values[i].size2());
                offset += param_size;
            }

            _out_substituted = _out;
            for (int i = 0; i < _W.size(); ++i)
            {
                _out_substituted = substitute(_out_substituted, _W[i], _nn_values[2*i]);
                _out_substituted = substitute(_out_substituted, _b[i], _nn_values[2*i + 1]);
            }
        }

        void train(const Eigen::MatrixXd &input, const Eigen::MatrixXd &target)
        {
            for (int j = 0; j < input.cols(); ++j)
            {
                backward(input.col(j), target.col(j));
            }
        }

        void reset()
        {
            for (int i = 0; i < _nn_values.size(); ++i)
            {
                _nn_values[i] = DM::rand(_nn_values[i].size1(), _nn_values[i].size2());
            }

            _out_substituted = _out;
            for (int i = 0; i < _W.size(); ++i)
            {
                _out_substituted = substitute(_out_substituted, _W[i], _nn_values[2*i]);
                _out_substituted = substitute(_out_substituted, _b[i], _nn_values[2*i + 1]);
            }
        }

    protected:
        int _input_size, _output_size;
        MX _X, _Y;
        std::vector<MX> _W, _b;
        MX _out;
        Function _out_fn;
        MX _loss;

        std::vector<DM> _nn_values;
        MX _out_substituted;

        MX _opt_vars;
    };
}

#endif