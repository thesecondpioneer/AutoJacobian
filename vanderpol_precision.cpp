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

int main() {
    string csvDir = "csvs/vanderpol/";
    ofstream analyticalOut(csvDir + "analytical.csv"), automaticOut(csvDir + "automatic.csv"), divdifOut(
            csvDir + "divdif.csv");
    vector_type x1(N - 1);
    x1[0] = 2;
    x1[1] = 0;

    auto a_pair = make_pair(vanderpol, vanderpol_jac);
    size_t num_of_steps = integrate_adaptive(rosenbrock4_controller<rosenbrock4<double>>(std::numeric_limits<double>::epsilon()),
                                             a_pair,
                                             x1, 0.0, 10.0, 0.01,
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
    x2[0] = 2;
    x2[1] = 0;
    AutoJacobian<double, N> aj(dual_vanderpol);
    auto aj_pair = make_pair(aj.system_function, aj.jacobi_function);
    size_t dual_num_of_steps = integrate_adaptive(rosenbrock4_controller<rosenbrock4<double>>(std::numeric_limits<double>::epsilon()),
                                                  aj_pair,
                                                  x2, 0.0, 10.0, 0.01,
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
    x3[0] = 2;
    x3[1] = 0;
    FiniteDifferenceJacobian<double> fd(vanderpol);
    auto fd_pair = make_pair(vanderpol, fd);
    size_t fd_num_of_steps = integrate_adaptive(rosenbrock4_controller<rosenbrock4<double>>(std::numeric_limits<double>::epsilon()),
                                                fd_pair,
                                                x3, 0.0, 10.0, 0.01,
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