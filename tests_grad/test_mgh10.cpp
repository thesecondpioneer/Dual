#include <gtest/gtest.h>
#include "Dual.h"
#include "DualScalar.h"

typedef dual::DualScalar<double> d_scalar;
typedef dual::Dual<d_scalar, 3> dir_hessian_type;

class Mgh10Test : public ::testing::Test {
protected:
    void SetUp() override {
        b_dual = {
                dir_hessian_type(d_scalar(5.6096364710E-03, 1), 0),
                dir_hessian_type(d_scalar(6.1813463463E+03, 1), 1),
                dir_hessian_type(d_scalar(3.4522363462E+02, 1), 2)
        };

        b_scalar = {
                5.6096364710E-03,
                6.1813463463E+03,
                3.4522363462E+02
        };

        x_values = {
                5.000000E+01,
                5.500000E+01,
                6.000000E+01,
                6.500000E+01,
                7.000000E+01,
                7.500000E+01,
                8.000000E+01,
                8.500000E+01,
                9.000000E+01,
                9.500000E+01,
                1.000000E+02,
                1.050000E+02,
                1.100000E+02,
                1.150000E+02,
                1.200000E+02,
                1.250000E+02
        };
        y_values = {
                3.478000E+04,
                2.861000E+04,
                2.365000E+04,
                1.963000E+04,
                1.637000E+04,
                1.372000E+04,
                1.154000E+04,
                9.744000E+03,
                8.261000E+03,
                7.030000E+03,
                6.005000E+03,
                5.147000E+03,
                4.427000E+03,
                3.820000E+03,
                3.307000E+03,
                2.872000E+03
        };
    }

    std::vector<dir_hessian_type> b_dual;
    std::vector<long double> b_scalar;
    std::vector<long double> x_values;
    std::vector<long double> y_values;

    dir_hessian_type mgh10(double x, double y, std::vector<dir_hessian_type> b) {
        return y - b[0] * exp(b[1] / (x + b[2]));
    }

    std::vector<long double> mgh10_grad(long double x, long double y, std::vector<long double> b) {
        return {
                -exp(b[1] / (x + b[2])),
                -b[0] * exp(b[1] / (x + b[2])) / (x + b[2]),
                b[0] * exp(b[1] / (x + b[2])) * b[1] / ((x + b[2]) * (x + b[2]))
        };
    }
};

TEST_F(Mgh10Test, GradientComparison) {
    constexpr long double tolerance = 1e-14;

    long double total_error = 0.0L;
    for (size_t i = 0; i < x_values.size(); ++i) {
        dir_hessian_type eps = mgh10(x_values[i], y_values[i], b_dual);
        dual::getGradient(eps);
        auto grad = mgh10_grad(x_values[i], y_values[i], b_scalar);

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