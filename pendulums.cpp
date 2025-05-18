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

const int N = 41; // dimensions + time (20 pendulums * 2 + time)
const int M = N - 1; // dimensions (2 * 20 = 40)

using namespace std;
using namespace boost::numeric::odeint;
namespace phoenix = boost::phoenix;
typedef boost::numeric::ublas::vector<double> vector_type;
typedef boost::numeric::ublas::matrix<double> matrix_type;

typedef dual::Dual<double, N> dual_scalar_type;
const double omega0 = 6.95; // rad/s
const double c_over_a_sq = 2.34; // rad^2/s^2
const double eta = 1.2; // rad/s^2
const double omega_d = 6.1; // rad/s
const double p_gamma = 0.05792; // rad/s
const int num_pendulums = M / 2; // 20 pendulums
typedef boost::numeric::ublas::vector<dual_scalar_type> dual_vector_type;

void pendulum_array(const vector_type &x, vector_type &dxdt, double t) {
    // First pendulum (n=1, i=0)
    dxdt[0] = x[1];
    dxdt[1] = -omega0 * omega0 * sin(x[0]) + c_over_a_sq * (x[2] - 2.0 * x[0]) - eta * cos(omega_d * t) * cos(x[0]) - p_gamma * x[1];

    // Middle pendulums (n=2 to n=19, i=2,4,...,36)
    for (int n = 1; n < num_pendulums - 1; ++n) {
        int i = 2 * n;
        dxdt[i] = x[i + 1];
        dxdt[i + 1] = -omega0 * omega0 * sin(x[i]) + c_over_a_sq * (x[i + 2] + x[i - 2] - 2.0 * x[i]) - eta * cos(omega_d * t) * cos(x[i]) - p_gamma * x[i + 1];
    }

    // Last pendulum (n=20, i=38)
    dxdt[M - 2] = x[M - 1];
    dxdt[M - 1] = -omega0 * omega0 * sin(x[M - 2]) + c_over_a_sq * (x[M - 4] - 2.0 * x[M - 2]) - eta * cos(omega_d * t) * cos(x[M - 2]) - p_gamma * x[M - 1];
}

void pendulum_array_jac(const vector_type &x, matrix_type &J, const double &t, vector_type &dfdt) {
    J.clear();

    // First pendulum (n=1, i=0)
    J(0, 1) = 1.0; // d(dxdt[0])/d(x[1])
    J(1, 0) = -omega0 * omega0 * cos(x[0]) + eta * cos(omega_d * t) * sin(x[0]) - 2.0 * c_over_a_sq; // d(dxdt[1])/d(x[0])
    J(1, 1) = -p_gamma; // d(dxdt[1])/d(x[1])
    J(1, 2) = c_over_a_sq; // d(dxdt[1])/d(x[2])
    dfdt[1] = eta * omega_d * sin(omega_d * t) * cos(x[0]); // d(dxdt[1])/dt

    // Middle pendulums (n=2 to n=19, i=2,4,...,36)
    for (int n = 1; n < num_pendulums - 1; ++n) {
        int i = 2 * n;
        J(i, i + 1) = 1.0; // d(dxdt[i])/d(x[i+1])
        J(i + 1, i - 2) = c_over_a_sq; // d(dxdt[i+1])/d(x[i-2])
        J(i + 1, i) = -omega0 * omega0 * cos(x[i]) + eta * cos(omega_d * t) * sin(x[i]) - 2.0 * c_over_a_sq; // d(dxdt[i+1])/d(x[i])
        J(i + 1, i + 1) = -p_gamma; // d(dxdt[i+1])/d(x[i+1])
        J(i + 1, i + 2) = c_over_a_sq; // d(dxdt[i+1])/d(x[i+2])
        dfdt[i + 1] = eta * omega_d * sin(omega_d * t) * cos(x[i]); // d(dxdt[i+1])/dt
    }

    // Last pendulum (n=20, i=38)
    J(M - 2, M - 1) = 1.0; // d(dxdt[M-2])/d(x[M-1])
    J(M - 1, M - 4) = c_over_a_sq; // d(dxdt[M-1])/d(x[M-4])
    J(M - 1, M - 2) = -omega0 * omega0 * cos(x[M - 2]) + eta * cos(omega_d * t) * sin(x[M - 2]) - 2.0 * c_over_a_sq; // d(dxdt[M-1])/d(x[M-2])
    J(M - 1, M - 1) = -p_gamma; // d(dxdt[M-1])/d(x[M-1])
    dfdt[M - 1] = eta * omega_d * sin(omega_d * t) * cos(x[M - 2]); // d(dxdt[M-1])/dt
}

void dual_pendulum_array(const dual_vector_type &x, dual_vector_type &dxdt, dual_scalar_type t) {
    // First pendulum (n=1, i=0)
    dxdt[0] = x[1];
    dxdt[1] = -omega0 * omega0 * dual::sin(x[0]) + c_over_a_sq * (x[2] - 2.0 * x[0]) - eta * dual::cos(omega_d * t) * dual::cos(x[0]) - p_gamma * x[1];

    // Middle pendulums (n=2 to n=19, i=2,4,...,36)
    for (int n = 1; n < num_pendulums - 1; ++n) {
        int i = 2 * n;
        dxdt[i] = x[i + 1];
        dxdt[i + 1] = -omega0 * omega0 * dual::sin(x[i]) + c_over_a_sq * (x[i + 2] + x[i - 2] - 2.0 * x[i]) - eta * dual::cos(omega_d * t) * dual::cos(x[i]) - p_gamma * x[i + 1];
    }

    // Last pendulum (n=20, i=38)
    dxdt[M - 2] = x[M - 1];
    dxdt[M - 1] = -omega0 * omega0 * dual::sin(x[M - 2]) + c_over_a_sq * (x[M - 4] - 2.0 * x[M - 2]) - eta * dual::cos(omega_d * t) * dual::cos(x[M - 2]) - p_gamma * x[M - 1];
}

int main() {
    double an_time_sum = 0, au_time_sum = 0, fd_time_sum = 0;
    int repeats = 1;
    srand(42); // Seed for reproducible random noise
    for (int i = 0; i < repeats; ++i) {
        stopwatch<std::chrono::microseconds> time_analytical, time_automatical, time_finite_differences;
        string csvDir = "csvs/pendulum/";
        ofstream analyticalOut(csvDir + "analytical.csv"), automaticOut(csvDir + "automatic.csv"), divdifOut(csvDir + "divdif.csv");

        // Generate one set of initial conditions
        vector_type init_x(M);
        for (int i = 0; i < num_pendulums; ++i) {
            init_x[i * 2] = 0.1 + 0.01 * (rand() / (double)RAND_MAX); // theta_n
            init_x[i * 2 + 1] = 0.0; // dot_theta_n
        }

        // Copy to all state vectors
        vector_type x1 = init_x, x2 = init_x, x3 = init_x;

        time_analytical.start();
        auto a_pair = make_pair(pendulum_array, pendulum_array_jac);
        size_t num_of_steps = integrate_const(
                make_dense_output<rosenbrock4<double>>(1.0e-12, 1.0e-12),
                a_pair,
                x1, 0.0, 1.0, 0.1,
                analyticalOut << fixed << setprecision(16)
                              << phoenix::arg_names::arg1[0] << ","
                              << phoenix::arg_names::arg1[1] << "\n"
        );
        time_analytical.stop();

        cout << endl << "AutoJacobian solution" << endl;

        AutoJacobian<double, N> aj(dual_pendulum_array);
        auto aj_pair = make_pair(aj.system_function, aj.jacobi_function);
        time_automatical.start();
        size_t dual_num_of_steps = integrate_const(
                make_dense_output<rosenbrock4<double>>(1.0e-12, 1.0e-12),
                aj_pair,
                x2, 0.0, 1.0, 0.1,
                automaticOut << fixed << setprecision(16)
                             << phoenix::arg_names::arg1[0] << ","
                             << phoenix::arg_names::arg1[1] << "\n"
        );
        time_automatical.stop();

        cout << endl << "Finite differences solution" << endl;

        auto fd_pair = make_pair(pendulum_array, FiniteDifferenceJacobian<double>(pendulum_array));
        time_finite_differences.start();
        size_t fd_num_of_steps = integrate_const(
                make_dense_output<rosenbrock4<double>>(1.0e-12, 1.0e-12),
                fd_pair,
                x3, 0.0, 1.0, 0.1,
                divdifOut << fixed << setprecision(16)
                          << phoenix::arg_names::arg1[0] << ","
                          << phoenix::arg_names::arg1[1] << "\n"
        );
        time_finite_differences.stop();

        cout << "Analytical time:          " << time_analytical.total_time() << endl
             << "AutoJacobian time:        " << time_automatical.total_time() << endl
             << "Finite differences time:  " << time_finite_differences.total_time() << endl
             << "Analytical steps:         " << num_of_steps << endl
             << "Dual steps:               " << dual_num_of_steps << endl
             << "Finite differences steps: " << fd_num_of_steps << endl;

        an_time_sum += time_analytical.total_time();
        au_time_sum += time_automatical.total_time();
        fd_time_sum += time_finite_differences.total_time();
    }
    cout << "Analytical average time:          " << an_time_sum / repeats << endl
         << "AutoJacobian average time:        " << au_time_sum / repeats << endl
         << "Finite differences average time:  " << fd_time_sum / repeats << endl;
}