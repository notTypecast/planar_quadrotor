#ifndef PQ_NUM_OPTIMIZER_HPP
#define PQ_NUM_OPTIMIZER_HPP

#include <iostream>

#include <casadi/casadi.hpp>

#include "src/params.hpp"
#include "src/opt/Optimizer.hpp"

using namespace casadi;

namespace pq
{
    namespace num_opt
    {
        struct Params
        {
            int horizon = pq::Value::Param::NumOpt::horizon;
            double dt = pq::Value::Param::NumOpt::dt;
            double m = pq::Value::Constant::mass;
            double I = pq::Value::Constant::inertia;
            double l = pq::Value::Constant::length;
            double g = pq::Value::Constant::g;
            Eigen::Vector3d init;
            Eigen::Vector3d target;
        };

        class NumOptimizer : public pq::Optimizer
        {
        public:
            NumOptimizer(Params params) : _H(params.horizon),
                                       dt(params.dt),
                                       _m(params.m),
                                       _I(params.I),
                                       _l(params.l),
                                       _g(params.g)
            {
                _model_params = Eigen::VectorXd::Zero(4);
                _model_params << params.m, params.I, params.l, params.g;

                _setup(params.init, params.target);
            }

            Eigen::MatrixXd solve()
            {
                OptiSol sol = _opti.solve();

                DM result = sol.value(_F);

                Eigen::MatrixXd forces(2, _H);

                for (int i = 0; i < _H; ++i)
                {
                    forces(0, i) = static_cast<double>(result(0, i));
                    forces(1, i) = static_cast<double>(result(1, i));
                }

                return forces;
            }

            virtual void reinit() {}

            virtual Eigen::VectorXd next(const Eigen::VectorXd &init, const Eigen::VectorXd &target)
            {
                if (_offset == _H)
                {
                    _setup(init, target);
                    _offset = -1;
                }
                if (_offset == -1)
                {
                    _offset = 0;
                    _forces = solve();
                }

                return _forces.col(_offset++);
            }

            virtual Eigen::VectorXd model_params()
            {
                return _model_params;
            }

        protected:
            int _H;
            double dt;
            double _m, _I, _l;
            double _g;
            Opti _opti;
            MX _x, _u, _a;
            MX _F;

            int _offset = -1;
            Eigen::MatrixXd _forces;

            Eigen::VectorXd _model_params;

            void _setup(const Eigen::VectorXd &init, const Eigen::VectorXd &target)
            {                
                _opti = Opti();
                _x = _opti.variable(3, _H + 1);
                _u = _opti.variable(3, _H + 1);
                _a = _opti.variable(3, _H);
                _F = _opti.variable(2, _H);

                _opti.minimize(sumsqr(_x) + sumsqr(_u) + sumsqr(_a));

                _opti.subject_to(0 <= _F <= pq::Value::Param::NumOpt::F_max);

                DM init_dm = DM::vertcat({init[0], init[1], init[2]});
                DM target_dm = DM::vertcat({target[0], target[1], target[2]});

                _opti.subject_to(_x(Slice(), 0) == init_dm);
                _opti.subject_to(_x(Slice(), _H) == target_dm);
                _opti.subject_to(_u(Slice(), 0) == 0);
                _opti.subject_to(_u(Slice(), _H) == 0);

                if (pq::Value::Param::SymNN::use)
                {
                    for (int i = 0; i < _H; ++i)
                    {
                        MX state = vertcat(_x(Slice(), i), _u(Slice(), i), _F(Slice(), i));
                        MX l = pq::Value::Param::SymNN::learned_model->forward(state);

                        _opti.subject_to(_a(0, i) == -(_F(0, i) + _F(1, i)) * sin(_x(2, i)) / _m + l(0));
                        _opti.subject_to(_a(1, i) == (_F(0, i) + _F(1, i)) * cos(_x(2, i)) / _m - _g + l(1));
                        _opti.subject_to(_a(2, i) == (_F(1, i) - _F(0, i)) * _l / (2 * _I) + l(2));
                    }
                }
                else {
                    for (int i = 0; i < _H; ++i)
                    {
                        _opti.subject_to(_a(0, i) == -(_F(0, i) + _F(1, i)) * sin(_x(2, i)) / _m);
                        _opti.subject_to(_a(1, i) == (_F(0, i) + _F(1, i)) * cos(_x(2, i)) / _m - _g);
                        _opti.subject_to(_a(2, i) == (_F(1, i) - _F(0, i)) * _l / (2 * _I));
                    }
                }

                for (int i = 1; i < _H + 1; ++i)
                {
                    _opti.subject_to(_x(Slice(), i) == _x(Slice(), i - 1) + _u(Slice(), i - 1) * dt + 0.5 * _a(Slice(), i - 1) * dt * dt);
                    _opti.subject_to(_u(Slice(), i) == _u(Slice(), i - 1) + _a(Slice(), i - 1) * dt);
                }

                Dict opts;
                opts["ipopt.print_level"] = 0;
                opts["print_time"] = false;

                _opti.solver("ipopt", opts);
            }
        };
    }
}

#endif