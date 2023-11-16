#ifndef PQ_PLANAR_QUADROTOR_CPP
#define PQ_PLANAR_QUADROTOR_CPP
#include <Eigen/Core>

namespace Value
{
    namespace Constant
    {
        constexpr double g = 9.81;
        constexpr double gm = 1.61;
        constexpr double mass = 1.0;
        constexpr double length = 0.3;
        constexpr double inertia = 0.2 * mass * length * length;
    };
};

class PlanarQuadrotor
{
public:
    PlanarQuadrotor(double mass, double moment_of_inertia, double length) : _m(mass),
                                                                            _I(moment_of_inertia),
                                                                            _l(length),
                                                                            _x(Eigen::Vector<double, 6>::Zero())
    {
    }

    void update(Eigen::Vector2d controls, double dt)
    {
        Eigen::Vector3d ddq;
        ddq[0] = -(controls[0] + controls[1]) * sin(_x[2]) / _m;
        ddq[1] = (controls[0] + controls[1]) * cos(_x[2]) / _m - Value::Constant::g;
        ddq[2] = (controls[1] - controls[0]) * _l / (2 * _I);

        _x.segment(0, 3) += _x.segment(3, 3) * dt + 0.5 * ddq * dt * dt;
        _x.segment(3, 3) += ddq * dt;
        _total += dt;
    }

    Eigen::Vector<double, 6> get_state()
    {
        return _x;
    }

    double get_sim_time()
    {
        return _total;
    }

private:
    double _m, _I, _l;
    Eigen::Vector<double, 6> _x;
    double _total = 0.0;
};

#endif