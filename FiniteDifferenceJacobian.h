// FiniteDifferenceJacobian.h
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <cmath>
#include <limits>

namespace ublas = boost::numeric::ublas;

template<typename Scalar>
class FiniteDifferenceJacobian {
public:
    using vector_type = ublas::vector<Scalar>;
    using matrix_type = ublas::matrix<Scalar>;
    using system_function = std::function<void(const vector_type&, vector_type&, Scalar)>;

    explicit FiniteDifferenceJacobian(system_function sys) : system_(sys) {}

    void operator()(const vector_type& x, matrix_type& J, const Scalar& t, vector_type& dfdt) {
        const size_t n = x.size();
        vector_type f_orig(n), f_pert(n);
        vector_type x_pert = x;

        system_(x, f_orig, t);

        for (size_t j = 0; j < n; ++j) {
            const Scalar h = compute_step_size(x[j]);
            x_pert[j] += h;
            system_(x_pert, f_pert, t);

            for (size_t i = 0; i < n; ++i) {
                J(i, j) = (f_pert[i] - f_orig[i]) / h;
            }
            x_pert[j] = x[j];  // Reset
        }

        const Scalar h_t = compute_step_size(t);
        system_(x, dfdt, t + h_t);
        dfdt = (dfdt - f_orig) / h_t;
    }

private:
    system_function system_;

    static Scalar compute_step_size(Scalar x) {
        const Scalar eps = std::numeric_limits<Scalar>::epsilon();
        return std::sqrt(eps) * (1.0 + std::abs(x));
    }
};