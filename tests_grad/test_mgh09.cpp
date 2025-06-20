#include <gtest/gtest.h>
#include "Dual.h"
#include "DualScalar.h"

typedef dual::DualScalar<double> d_scalar;
typedef dual::Dual<d_scalar, 4> dir_hessian_type;

class Mgh09Test : public ::testing::Test {
protected:
    void SetUp() override {
        b_dual = {
                dir_hessian_type(d_scalar(1.9280693458E-01, 1), 0),
                dir_hessian_type(d_scalar(1.9128232873E-01, 1), 1),
                dir_hessian_type(d_scalar(1.2305650693E-01, 1), 2),
                dir_hessian_type(d_scalar(1.3606233068E-01, 1), 3)
        };

        b_scalar = {
                1.9280693458E-01,
                1.9128232873E-01,
                1.2305650693E-01,
                1.3606233068E-01
        };

        x_values = {
                4.000000E+00,
                2.000000E+00,
                1.000000E+00,
                5.000000E-01,
                2.500000E-01,
                1.670000E-01,
                1.250000E-01,
                1.000000E-01,
                8.330000E-02,
                7.140000E-02,
                6.250000E-02
        };
        y_values = {
                1.957000E-01,
                1.947000E-01,
                1.735000E-01,
                1.600000E-01,
                8.440000E-02,
                6.270000E-02,
                4.560000E-02,
                3.420000E-02,
                3.230000E-02,
                2.350000E-02,
                2.460000E-02
        };
    }

    std::vector<dir_hessian_type> b_dual;
    std::vector<long double> b_scalar;
    std::vector<long double> x_values;
    std::vector<long double> y_values;

    dir_hessian_type mgh09(double x, double y, std::vector<dir_hessian_type> b) {
        return y - b[0] * (x * x + x * b[1]) / (x * x + x * b[2] + b[3]);
    }

    std::vector<long double> mgh09_grad(long double x, long double y, std::vector<long double> b) {
        return {
                -(x * x + b[1] * x) / (x * x + b[2] * x + b[3]),
                -b[0] * x / (x * x + b[2] * x + b[3]),
                (b[0] * std::pow(x, 3) + b[0] * b[1] * x * x) / std::pow((x * x + b[2] * x + b[3]), 2),
                (b[0] * x * x + b[0] * b[1] * x) / std::pow((x * x + b[2] * x + b[3]), 2)
        };
    }
};

TEST_F(Mgh09Test, GradientComparison) {
    constexpr long double tolerance = 6e-16;

    long double total_error = 0.0L;
    for (size_t i = 0; i < x_values.size(); ++i) {
        dir_hessian_type eps = mgh09(x_values[i], y_values[i], b_dual);
        dual::getGradient(eps);
        auto grad = mgh09_grad(x_values[i], y_values[i], b_scalar);

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