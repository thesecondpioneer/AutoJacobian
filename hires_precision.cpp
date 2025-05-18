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

const int N = 9;
using namespace std;
using namespace boost::numeric::odeint;
namespace phoenix = boost::phoenix;

typedef boost::numeric::ublas::vector<double> vector_type;
typedef boost::numeric::ublas::matrix<double> matrix_type;

typedef dual::Dual<double, N> dual_scalar_type;
typedef boost::numeric::ublas::vector<dual_scalar_type> dual_vector_type;

void hires(const vector_type &x, vector_type &dxdt, double /* t */) {
    dxdt[0] = -1.71 * x[0] + 0.43 * x[1] + 8.32 * x[2] + 0.0007;
    dxdt[1] = 1.71 * x[0] - 8.75 * x[1];
    dxdt[2] = -10.03 * x[2] + 0.43 * x[3] + 0.035 * x[4];
    dxdt[3] = 8.32 * x[1] + 1.71 * x[2] - 1.12 * x[3];
    dxdt[4] = -1.745 * x[4] + 0.43 * x[5] + 0.43 * x[6];
    dxdt[5] = -280.0 * x[5] * x[7] + 0.69 * x[3] + 1.71 * x[4] - 0.43 * x[5] + 0.69 * x[6];
    dxdt[6] = 280.0 * x[5] * x[7] - 1.81 * x[6];
    dxdt[7] = -280.0 * x[5] * x[7] + 1.81 * x[6];
}

void hires_jac(const vector_type &x, matrix_type &J, const double & /* t */, vector_type &dfdt) {
    J.clear();
    J(0, 0) = -1.71;
    J(0, 1) = 0.43;
    J(0, 2) = 8.32;

    J(1, 0) = 1.71;
    J(1, 1) = -8.75;

    J(2, 2) = -10.03;
    J(2, 3) = 0.43;
    J(2, 4) = 0.035;

    J(3, 1) = 8.32;
    J(3, 2) = 1.71;
    J(3, 3) = -1.12;

    J(4, 4) = -1.745;
    J(4, 5) = 0.43;
    J(4, 6) = 0.43;

    J(5, 3) = 0.69;
    J(5, 4) = 1.71;
    J(5, 5) = -0.43 - 280.0 * x[7];
    J(5, 6) = 0.69;
    J(5, 7) = -280.0 * x[5];

    J(6, 5) = 280.0 * x[7];
    J(6, 6) = -1.81;
    J(6, 7) = 280.0 * x[5];

    J(7, 5) = -280.0 * x[7];
    J(7, 6) = 1.81;
    J(7, 7) = -280.0 * x[5];
}

void dual_hires(const dual_vector_type &x, dual_vector_type &dxdt, dual_scalar_type /* t */) {
    dxdt[0] = -1.71 * x[0] + 0.43 * x[1] + 8.32 * x[2] + 0.0007;
    dxdt[1] = 1.71 * x[0] - 8.75 * x[1];
    dxdt[2] = -10.03 * x[2] + 0.43 * x[3] + 0.035 * x[4];
    dxdt[3] = 8.32 * x[1] + 1.71 * x[2] - 1.12 * x[3];
    dxdt[4] = -1.745 * x[4] + 0.43 * x[5] + 0.43 * x[6];
    dxdt[5] = -280.0 * x[5] * x[7] + 0.69 * x[3] + 1.71 * x[4] - 0.43 * x[5] + 0.69 * x[6];
    dxdt[6] = 280.0 * x[5] * x[7] - 1.81 * x[6];
    dxdt[7] = -280.0 * x[5] * x[7] + 1.81 * x[6];
}

int main() {
    string csvDir = "csvs/hires/";
    ofstream analyticalOut(csvDir + "analytical.csv"), automaticOut(csvDir + "automatic.csv"), divdifOut(
            csvDir + "divdif.csv");
    vector_type x1(N - 1);
    x1[0] = 1;
    x1[1] = x1[2] = x1[3] = x1[4] = x1[5] = x1[6] = 0;
    x1[7] = 0.0057;

    auto a_pair = make_pair(hires, hires_jac);
    size_t num_of_steps = integrate_adaptive(rosenbrock4_controller<rosenbrock4<double>>(std::numeric_limits<double>::epsilon()),
                                             a_pair,
                                             x1, 321.8122, 421.8122, 0.01,
                                             analyticalOut << fixed
                                                           << phoenix::bind([](const vector_type &x) {
                                                               stringstream ss;
                                                               for (size_t i = 0; i < x.size(); ++i) {
                                                                   ss << fixed << setprecision(16) << x[i];
                                                                   if (i < x.size() - 1) ss << ",";
                                                               }
                                                               return ss.str();
                                                           }, phoenix::arg_names::arg1) << "\n");

    vector_type x2(N - 1);
    x2[0] = 1;
    x2[1] = x2[2] = x2[3] = x2[4] = x2[5] = x2[6] = 0;
    x2[7] = 0.0057;
    AutoJacobian<double, N> aj(dual_hires);
    auto aj_pair = make_pair(aj.system_function, aj.jacobi_function);
    size_t dual_num_of_steps = integrate_adaptive(rosenbrock4_controller<rosenbrock4<double>>(std::numeric_limits<double>::epsilon()),
                                                  aj_pair,
                                                  x2, 321.8122, 421.8122, 0.01,
                                                  automaticOut << fixed
                                                               << phoenix::bind([](const vector_type &x) {
                                                                   stringstream ss;
                                                                   for (size_t i = 0; i < x.size(); ++i) {
                                                                       ss << fixed << setprecision(16) << x[i];
                                                                       if (i < x.size() - 1) ss << ",";
                                                                   }
                                                                   return ss.str();
                                                               }, phoenix::arg_names::arg1) << "\n");

    vector_type x3(N - 1);
    x3[0] = 1;
    x3[1] = x3[2] = x3[3] = x3[4] = x3[5] = x3[6] = 0;
    x3[7] = 0.0057;
    FiniteDifferenceJacobian<double> fd(hires);
    auto fd_pair = make_pair(hires, fd);
    size_t fd_num_of_steps = integrate_adaptive(rosenbrock4_controller<rosenbrock4<double>>(std::numeric_limits<double>::epsilon()),
                                                fd_pair,
                                                x3, 321.8122, 421.8122, 0.01,
                                                divdifOut << fixed
                                                          << phoenix::bind([](const vector_type &x) {
                                                              stringstream ss;
                                                              for (size_t i = 0; i < x.size(); ++i) {
                                                                  ss << fixed << setprecision(16) << x[i];
                                                                  if (i < x.size() - 1) ss << ",";
                                                              }
                                                              return ss.str();
                                                          }, phoenix::arg_names::arg1) << "\n");

    return 0;
}