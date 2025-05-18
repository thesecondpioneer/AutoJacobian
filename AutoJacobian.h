#ifndef TEST_BOOST_AUTOJACOBIAN_H
#define TEST_BOOST_AUTOJACOBIAN_H

#include <functional>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include "Dual.h"
#include "Clock.h"

template<typename Scalar, size_t N>
class AutoJacobian {
    typedef boost::numeric::ublas::vector<Scalar> vector_type;
    typedef dual::Dual<Scalar, N> dual_scalar_type;
    typedef boost::numeric::ublas::vector<dual_scalar_type> dual_vector_type;
    typedef boost::numeric::ublas::matrix<Scalar> matrix_type;

    dual_vector_type rhs_with_jacobian;
    dual_vector_type state;
    dual_scalar_type time;
    std::function<void(const dual_vector_type &, dual_vector_type &, dual_scalar_type)> dual_system_function;

    static constexpr size_t ODE_DIM = N - 1;

    //The scalar system function needed for boost implicit methods
    void sys(const vector_type &x, vector_type &dxdt, Scalar t) {
        for (size_t i = 0; i < ODE_DIM; i++) {
            state[i].x = x[i];
            state[i].y.setZero();
            state[i].y[i] = Scalar(1);
        }

        time.x = t;
        time.y.setZero();
        time.y[ODE_DIM] = Scalar(1);

        dual_system_function(state, rhs_with_jacobian, time);
        for (size_t i = 0; i < ODE_DIM; i++) {
            dxdt[i] = rhs_with_jacobian[i].x;
        }
    }

    //The scalar jacobian function needed for boost implicit methods
    void jac(const vector_type &x, matrix_type &J, const Scalar &t, vector_type &dfdt) {
        Scalar *J_data = &J(0, 0);
        const size_t row_stride = J.size2();
        for (size_t j = 0; j < ODE_DIM; j++) {
            const auto &dy_eigen = rhs_with_jacobian[j].y;
            Eigen::Map<Eigen::RowVectorXd>(J_data + j * row_stride, ODE_DIM) =
                    dy_eigen.head(ODE_DIM).transpose();
            dfdt[j] = dy_eigen[ODE_DIM];
        }
    }

public:
    AutoJacobian(void f(const dual_vector_type &, dual_vector_type &, dual_scalar_type)) :
            rhs_with_jacobian(ODE_DIM),
            state(ODE_DIM),
            dual_system_function(f) {};

    std::function<void(const vector_type &, vector_type &, Scalar)> system_function =
            [this](const vector_type &x, vector_type &dxdt, Scalar t) {
                this->sys(x, dxdt, t);
            };

    std::function<void(const vector_type &, matrix_type &, const Scalar &, vector_type &)> jacobi_function =
            [this](const vector_type &x, matrix_type &J, const Scalar &t, vector_type &dfdt) {
                this->jac(x, J, t, dfdt);
            };
};

#endif //TEST_BOOST_AUTOJACOBIAN_H
