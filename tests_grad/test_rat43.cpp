#include <gtest/gtest.h>
#include "Dual.h"
#include "DualScalar.h"

typedef dual::DualScalar<double> d_scalar;
typedef dual::Dual<d_scalar, 4> dir_hessian_type;

class Rat43Test : public ::testing::Test {
protected:
    void SetUp() override {
        b_dual = {
                dir_hessian_type(d_scalar(6.9964151270E+02, 1), 0),
                dir_hessian_type(d_scalar(5.2771253025E+00, 1), 1),
                dir_hessian_type(d_scalar(7.5962938329E-01, 1), 2),
                dir_hessian_type(d_scalar(1.2792483859E+00, 1), 3)
        };

        b_scalar = {
                6.9964151270E+02,
                5.2771253025E+00,
                7.5962938329E-01,
                1.2792483859E+00
        };

        x_values = {
                1.0E0,
                2.0E0,
                3.0E0,
                4.0E0,
                5.0E0,
                6.0E0,
                7.0E0,
                8.0E0,
                9.0E0,
                10.0E0,
                11.0E0,
                12.0E0,
                13.0E0,
                14.0E0,
                15.0E0
        };
        y_values = {
                16.08E0,
                33.83E0,
                65.80E0,
                97.20E0,
                191.55E0,
                326.20E0,
                386.87E0,
                520.53E0,
                590.03E0,
                651.92E0,
                724.93E0,
                699.56E0,
                689.96E0,
                637.56E0,
                717.41E0
        };
    }

    std::vector<dir_hessian_type> b_dual;
    std::vector<long double> b_scalar;
    std::vector<long double> x_values;
    std::vector<long double> y_values;

    dir_hessian_type rat43(double x, double y, std::vector<dir_hessian_type> b) {
        return y - b[0] / pow(1.0 + exp(b[1] - b[2] * x), 1.0 / b[3]);
    }

    std::vector<long double> rat43_grad(long double x, long double y, std::vector<long double> b) {
        return {
                -1.0L / std::pow(1.0L + std::exp(b[1] - b[2] * x), 1.0L / b[3]),
                b[0] * std::exp(b[1] - b[2] * x) / (b[3] * std::pow(1.0L + std::exp(b[1] - b[2] * x), (1.0L + b[3]) / b[3])),
                -b[0] * std::exp(b[1] - b[2] * x) * x / (b[3] * std::pow(1.0L + std::exp(b[1] - b[2] * x), (1.0L + b[3]) / b[3])),
                -b[0] * std::log(1.0L + std::exp(b[1] - b[2] * x)) / (pow(1.0L + exp(b[1] - b[2] * x), 1.0L / b[3]) * b[3] * b[3])
        };
    }
};

TEST_F(Rat43Test, GradientComparison) {
    constexpr long double tolerance = 1e-13;

    long double total_error = 0.0L;
    for (size_t i = 0; i < x_values.size(); ++i) {
        dir_hessian_type eps = rat43(x_values[i], y_values[i], b_dual);
        dual::getGradient(eps);
        auto grad = rat43_grad(x_values[i], y_values[i], b_scalar);

        for (int j = 0; j < 4; ++j) {
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
    GTEST_LOG_(INFO) << "Average relative error: " << total_error / (x_values.size() * 4);
}