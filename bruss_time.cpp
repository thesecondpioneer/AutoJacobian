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

const int N = 501; //dimensions + time
const int M = N - 1; //dimensions

using namespace std;
using namespace boost::numeric::odeint;
namespace phoenix = boost::phoenix;
typedef boost::numeric::ublas::vector<double> vector_type;
typedef boost::numeric::ublas::matrix<double> matrix_type;

typedef dual::Dual<double, N> dual_scalar_type;
double alpha = 0.02, delta_x = 1.0 / (M + 1);
typedef boost::numeric::ublas::vector<dual_scalar_type> dual_vector_type;

void bruss(const vector_type &x, vector_type &dxdt, double /* t */) {
    dxdt[0] = 1.0 + x[0] * x[0] * x[1] - 4.0 * x[0] + alpha / (delta_x * delta_x) * (1.0 - 2.0 * x[0] + x[2]);
    dxdt[1] = 3.0 * x[0] - x[0] * x[0] * x[1] + alpha / (delta_x * delta_x) * (3.0 - 2.0 * x[1] + x[3]);
    for (int i = 1; i < M / 2 - 1; i++) {
        dxdt[i * 2] = 1.0 + x[i * 2] * x[i * 2] * x[i * 2 + 1] - 4.0 * x[i * 2] +
                      alpha / (delta_x * delta_x) * (x[(i - 1) * 2] - 2.0 * x[i * 2] + x[(i + 1) * 2]);
        dxdt[i * 2 + 1] = 3.0 * x[i * 2] - x[i * 2] * x[i * 2] * x[i * 2 + 1] +
                          alpha / (delta_x * delta_x) * (x[(i - 1) * 2 + 1] - 2.0 * x[i * 2 + 1] + x[(i + 1) * 2 + 1]);
    }
    dxdt[M - 2] = 1.0 + x[M - 2] * x[M - 2] * x[M - 1] - 4.0 * x[M - 2] +
                  alpha / (delta_x * delta_x) * (x[M - 4] - 2.0 * x[M - 2] + 1.0);
    dxdt[M - 1] = 3.0 * x[M - 2] - x[M - 2] * x[M - 2] * x[M - 1] +
                  alpha / (delta_x * delta_x) * (x[M - 3] - 2.0 * x[M - 1] + 3.0);
}

void bruss_jac(const vector_type &x, matrix_type &J, const double & /* t */ , vector_type &dfdt) {
    J.clear();
    J(0, 0) = 2.0 * x(0) * x(1) - 4.0 - 2.0 * alpha / (delta_x * delta_x); // ∂f_0/∂x_0
    J(0, 1) = x(0) * x(0);                        // ∂f_0/∂x_1
    J(0, 2) = alpha / (delta_x * delta_x);                                   // ∂f_0/∂x_2
    J(1, 0) = 3.0 - 2.0 * x(0) * x(1);            // ∂f_1/∂x_0
    J(1, 1) = -x(0) * x(0) - 2.0 * alpha / (delta_x * delta_x);             // ∂f_1/∂x_1
    J(1, 3) = alpha / (delta_x * delta_x);                                   // ∂f_1/∂x_3

    for (int i = 1; i < M / 2 - 1; i++) {
        J(2 * i, 2 * i) = 2.0 * x(2 * i) * x(2 * i + 1) - 4.0 - 2.0 * alpha / (delta_x * delta_x); // ∂f_{2i}/∂x_{2i}
        J(2 * i, 2 * i + 1) = x(2 * i) * x(2 * i);                       // ∂f_{2i}/∂x_{2i+1}
        J(2 * i, 2 * (i - 1)) =
                alpha / (delta_x * delta_x);                                       // ∂f_{2i}/∂x_{2(i-1)}
        J(2 * i, 2 * (i + 1)) =
                alpha / (delta_x * delta_x);                                       // ∂f_{2i}/∂x_{2(i+1)}

        J(2 * i + 1, 2 * i) = 3.0 - 2.0 * x(2 * i) * x(2 * i + 1);       // ∂f_{2i+1}/∂x_{2i}
        J(2 * i + 1, 2 * i + 1) =
                -x(2 * i) * x(2 * i) - 2.0 * alpha / (delta_x * delta_x);        // ∂f_{2i+1}/∂x_{2i+1}
        J(2 * i + 1, 2 * (i - 1) + 1) =
                alpha / (delta_x * delta_x);                               // ∂f_{2i+1}/∂x_{2(i-1)+1}
        J(2 * i + 1, 2 * (i + 1) + 1) =
                alpha / (delta_x * delta_x);                               // ∂f_{2i+1}/∂x_{2(i+1)+1}
    }

    J(M - 2, M - 2) = 2.0 * x(M - 2) * x(M - 1) - 4.0 - 2.0 * alpha / (delta_x * delta_x); // ∂f_{M-2}/∂x_{M-2}
    J(M - 2, M - 1) = x(M - 2) * x(M - 2);                       // ∂f_{M-2}/∂x_{M-1}
    J(M - 2, M - 4) = alpha / (delta_x * delta_x);                                          // ∂f_{M-2}/∂x_{M-4}
    J(M - 1, M - 2) = 3.0 - 2.0 * x(M - 2) * x(M - 1);           // ∂f_{M-1}/∂x_{M-2}
    J(M - 1, M - 1) = -x(M - 2) * x(M - 2) - 2.0 * alpha / (delta_x * delta_x);            // ∂f_{M-1}/∂x_{M-1}
    J(M - 1, M - 3) = alpha / (delta_x * delta_x);                                          // ∂f_{M-1}/∂x_{M-3}
}

void dual_bruss(const dual_vector_type &x, dual_vector_type &dxdt, dual_scalar_type /* t */) {
    dxdt[0] = 1.0 + x[0] * x[0] * x[1] - 4.0 * x[0] + alpha / (delta_x * delta_x) * (1.0 - 2.0 * x[0] + x[2]);
    dxdt[1] = 3.0 * x[0] - x[0] * x[0] * x[1] + alpha / (delta_x * delta_x) * (3.0 - 2.0 * x[1] + x[3]);
    for (int i = 1; i < M / 2 - 1; i++) {
        dxdt[i * 2] = 1.0 + x[i * 2] * x[i * 2] * x[i * 2 + 1] - 4.0 * x[i * 2] +
                      alpha / (delta_x * delta_x) * (x[(i - 1) * 2] - 2.0 * x[i * 2] + x[(i + 1) * 2]);
        dxdt[i * 2 + 1] = 3.0 * x[i * 2] - x[i * 2] * x[i * 2] * x[i * 2 + 1] +
                          alpha / (delta_x * delta_x) * (x[(i - 1) * 2 + 1] - 2.0 * x[i * 2 + 1] + x[(i + 1) * 2 + 1]);
    }
    dxdt[M - 2] = 1.0 + x[M - 2] * x[M - 2] * x[M - 1] - 4.0 * x[M - 2] +
                  alpha / (delta_x * delta_x) * (x[M - 4] - 2.0 * x[M - 2] + 1.0);
    dxdt[M - 1] = 3.0 * x[M - 2] - x[M - 2] * x[M - 2] * x[M - 1] +
                  alpha / (delta_x * delta_x) * (x[M - 3] - 2.0 * x[M - 1] + 3.0);
}

stopwatch<std::chrono::microseconds> j_time_analytical, j_time_automatical, j_time_finite_differences;

int main() {
    double an_time_sum = 0, au_time_sum = 0, fd_time_sum = 0;
    int repeats = 20;
    for (int i = 0; i < repeats; i++) {
        stopwatch<std::chrono::microseconds> time_analytical, time_automatical, time_finite_differences;
        vector_type x1(M);
        for (int i = 0; i < M / 2; i++) {
            x1[i * 2] = 1.0 + sin(M_2_PI * (i + 1) / (M + 1));
            x1[i * 2 + 1] = 3.0;
        }

        auto a_pair = make_pair([&](const vector_type &x, vector_type &dxdt, double t) {
            j_time_analytical.start();
            bruss(x, dxdt, t);
            j_time_analytical.stop();
        }, [&](const vector_type &x, matrix_type &J, const double &t, vector_type &dfdt) {
            j_time_analytical.start();
            bruss_jac(x, J, t, dfdt);
            j_time_analytical.stop();
        });
        time_analytical.start();
        size_t num_of_steps = integrate_adaptive(rosenbrock4_controller<rosenbrock4<double>>(1.0e-6, 1.0e-6),
                                                 a_pair,
                                                 x1, 0.0, 10.0, 0.01,
                                                 [](const vector_type &, double) {});
        time_analytical.stop();

        vector_type x2(M);
        for (int i = 0; i < M / 2; i++) {
            x2[i * 2] = 1.0 + sin(M_2_PI * (i + 1) / (M + 1));
            x2[i * 2 + 1] = 3.0;
        }
        AutoJacobian<double, N> aj(dual_bruss);
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

        vector_type x3(M);
        for (int i = 0; i < M / 2; i++) {
            x3[i * 2] = 1.0 + sin(M_2_PI * (i + 1) / (M + 1));
            x3[i * 2 + 1] = 3.0;
        }
        FiniteDifferenceJacobian<double> fd(bruss);
        auto fd_pair = make_pair([&](const vector_type &x, vector_type &dxdt, double t) {
            j_time_finite_differences.start();
            bruss(x, dxdt, t);
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