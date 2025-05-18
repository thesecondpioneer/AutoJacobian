#include <boost/numeric/odeint.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/phoenix.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include "Dual.h"
#include "AutoJacobian.h"
#include "FiniteDifferenceJacobian.h"
#include "Clock.h"

const int N = 3;

using namespace std;
using namespace boost::numeric::odeint;
namespace phoenix = boost::phoenix;
typedef boost::numeric::ublas::vector<double> vector_type;
typedef boost::numeric::ublas::matrix<double> matrix_type;

typedef dual::Dual<double, N> dual_scalar_type;
typedef boost::numeric::ublas::vector<dual_scalar_type> dual_vector_type;

void vanderpol(const vector_type &x, vector_type &dxdt, double /* t */) {
    dxdt[0] = x[1];
    dxdt[1] = ((1.0 - x[0] * x[0]) * x[1] - x[0]) * 1e6;
}

void vanderpol_jac(const vector_type &x, matrix_type &J, const double & /* t */ , vector_type &dfdt) {
    J(0, 0) = 0;
    J(0, 1) = 1;
    J(1, 0) = -2e6 * x[0] * x[1] - 1e6;
    J(1, 1) = 1e6 * (1.0 - x[0] * x[0]);
}

void dual_vanderpol(const dual_vector_type &x, dual_vector_type &dxdt, dual_scalar_type /* t */) {
    dxdt[0] = x[1];
    dxdt[1] = ((1.0 - x[0] * x[0]) * x[1] - x[0]) * 1e6;
}

stopwatch<std::chrono::microseconds> j_time_analytical, j_time_automatical, j_time_finite_differences;

int main() {
    double an_time_sum = 0, au_time_sum = 0, fd_time_sum = 0;
    int repeats = 1000;
    for (int i = 0; i < repeats; i++) {
        stopwatch<std::chrono::microseconds> time_analytical, time_automatical, time_finite_differences;
        vector_type x1(N - 1);
        x1[0] = 2;
        x1[1] = 0;

        auto a_pair = make_pair([&](const vector_type &x, vector_type &dxdt, double t) {
            j_time_analytical.start();
            vanderpol(x, dxdt, t);
            j_time_analytical.stop();
        }, [&](const vector_type &x, matrix_type &J, const double &t, vector_type &dfdt) {
            j_time_analytical.start();
            vanderpol_jac(x, J, t, dfdt);
            j_time_analytical.stop();
        });
        time_analytical.start();
        size_t num_of_steps = integrate_adaptive(rosenbrock4_controller<rosenbrock4<double>>(1.0e-6, 1.0e-6),
                                                 a_pair,
                                                 x1, 0.0, 10.0, 0.01,
                                                 [](const vector_type &, double) {});
        time_analytical.stop();

        vector_type x2(N - 1);
        x2[0] = 2;
        x2[1] = 0;
        AutoJacobian<double, N> aj(dual_vanderpol);
        auto aj_pair = make_pair([&](const vector_type &x, vector_type &dxdt, double t) {
            j_time_automatical.start();
            aj.system_function(x, dxdt, t);
            j_time_automatical.stop();
        }, [&](const vector_type &x, matrix_type &J, const double &t, vector_type &dfdt) {
            j_time_automatical.start();
            aj.jacobi_function(x, J, t, dfdt);
            j_time_automatical.stop();
        });
        time_automatical.start();
        size_t dual_num_of_steps = integrate_adaptive(rosenbrock4_controller<rosenbrock4<double>>(1.0e-6, 1.0e-6),
                                                      aj_pair,
                                                      x2, 0.0, 10.0, 0.01,
                                                      [](const vector_type &, double) {});
        time_automatical.stop();

        vector_type x3(N - 1);
        x3[0] = 2;
        x3[1] = 0;
        FiniteDifferenceJacobian<double> fd(vanderpol);
        auto fd_pair = make_pair([&](const vector_type &x, vector_type &dxdt, double t) {
            j_time_finite_differences.start();
            vanderpol(x, dxdt, t);
            j_time_finite_differences.stop();
        }, [&](const vector_type &x, matrix_type &J, const double &t, vector_type &dfdt) {
            j_time_finite_differences.start();
            fd(x, J, t, dfdt);
            j_time_finite_differences.stop();
        });
        time_finite_differences.start();
        size_t fd_num_of_steps = integrate_adaptive(rosenbrock4_controller<rosenbrock4<double>>(1.0e-6, 1.0e-6),
                                                    fd_pair,
                                                    x3, 0.0, 10.0, 0.01,
                                                    [](const vector_type &, double) {});
        time_finite_differences.stop();

        an_time_sum += time_analytical.total_time();
        au_time_sum += time_automatical.total_time();
        fd_time_sum += time_finite_differences.total_time();
    }
    cout << "Analytical average time:          " << an_time_sum / repeats << endl
         << "AutoJacobian average time:        " << au_time_sum / repeats << endl
         << "Finite differences average time:  " << fd_time_sum / repeats << endl
         << "Of this time, system and Jacobian calculation took on average: " << endl
         << "For analytical:         " << j_time_analytical.total_time() / repeats << endl
         << "For AutoJacobian:       " << j_time_automatical.total_time() / repeats << endl
         << "For finite differences: " << j_time_finite_differences.total_time() / repeats << endl;
    return 0;
}