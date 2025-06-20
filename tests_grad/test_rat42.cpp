#include <gtest/gtest.h>
#include "Dual.h"
#include "DualScalar.h"

typedef dual::DualScalar<double> d_scalar;
typedef dual::Dual<d_scalar, 3> dir_hessian_type;

class Rat42Test : public ::testing::Test {
protected:
    void SetUp() override {
        b_dual = {
                dir_hessian_type(d_scalar(7.2462237576E+01, 1), 0),
                dir_hessian_type(d_scalar(2.6180768402E+00, 1), 1),
                dir_hessian_type(d_scalar(6.7359200066E-02, 1), 2)
        };

        b_scalar = {
                7.2462237576E+01,
                2.6180768402E+00,
                6.7359200066E-02
        };

        x_values = {
                9.000E0,
                14.000E0,
                21.000E0,
                28.000E0,
                42.000E0,
                57.000E0,
                63.000E0,
                70.000E0,
                79.000E0
        };
        y_values = {
                8.930E0,
                10.800E0,
                18.590E0,
                22.330E0,
                39.350E0,
                56.110E0,
                61.730E0,
                64.620E0,
                67.080E0
        };
    }

    std::vector<dir_hessian_type> b_dual;
    std::vector<long double> b_scalar;
    std::vector<long double> x_values;
    std::vector<long double> y_values;

    dir_hessian_type rat42(double x, double y, std::vector<dir_hessian_type> b) {
        return y - b[0] / (1.0 + dual::exp(b[1] - b[2] * x));
    }

    std::vector<long double> rat42_grad(long double x, long double y, std::vector<long double> b) {
        return {
                -1.0L / (1.0L + exp(b[1] - b[2] * x)),
                b[0] * exp(b[1] - b[2] * x) / std::pow((1.0L + exp(b[1] - b[2] * x)), 2),
                -b[0] * exp(b[1] - b[2] * x) * x / std::pow((1.0L + exp(b[1] - b[2] * x)), 2)
        };
    }
};

TEST_F(Rat42Test, GradientComparison) {
    constexpr long double tolerance = 6e-16;

    long double total_error = 0.0L;
    for (size_t i = 0; i < x_values.size(); ++i) {
        dir_hessian_type eps = rat42(x_values[i], y_values[i], b_dual);
        dual::getGradient(eps);
        auto grad = rat42_grad(x_values[i], y_values[i], b_scalar);

        for (int j = 0; j < 3; ++j) {
            long double dual_value = eps.y[j].x;
            long double ref_value = grad[j];
            long double max_denominator = std::max(1.0L, std::abs(ref_value));
            long double relative_error = std::abs(dual_value - ref_value) / max_denominator;
            total_error += relative_error;
            EXPECT_LT(relative_error, tolerance)
                                << "For x=" << x_values[i] << ", y=" << y_values[i]
                                << ", parameter " << j
                                << ": Dual value " << dual_value
                                << " vs reference " << ref_value
                                << " (relative error: " << relative_error << ")";
        }
    }
    GTEST_LOG_(INFO) << "Average relative error: " << total_error / (x_values.size() * 3);
}