#include <gtest/gtest.h>
#include "Dual.h"
#include "DualScalar.h"

typedef dual::DualScalar<double> d_scalar;
typedef dual::Dual<d_scalar, 2> dir_hessian_type;

class BoxBodTest : public ::testing::Test {
protected:
    void SetUp() override {
        b_dual = {dir_hessian_type(d_scalar(2.1380940889E+02, 1), 0), dir_hessian_type(d_scalar(5.4723748542E-01, 1), 1)};

        b_scalar = {2.1380940889E+02, 5.4723748542E-01};

        x_values = {1.0, 2.0, 3.0, 5.0, 7.0, 10.0};
        y_values = {109.0, 149.0, 149.0, 191.0, 213.0, 224.0};
    }

    std::vector<dir_hessian_type> b_dual;
    std::vector<long double> b_scalar;
    std::vector<long double> x_values;
    std::vector<long double> y_values;

    dir_hessian_type boxbod(double x, double y, std::vector<dir_hessian_type> b) {
        return y - b[0] * (1.0 - dual::exp(-b[1] * x));
    }

    std::vector<long double> boxbod_grad(long double x, long double y, std::vector<long double> b) {
        return {
                -1.0 + std::exp(-b[1] * x),
                -(b[0] * x) * std::exp(-b[1] * x)
        };
    }
};

TEST_F(BoxBodTest, GradientComparison) {
    constexpr long double tolerance = 6e-16;

    long double total_error = 0.0L;
    for (size_t i = 0; i < x_values.size(); ++i) {
        dir_hessian_type eps = boxbod(x_values[i], y_values[i], b_dual);
        dual::getGradient(eps);
        auto grad = boxbod_grad(x_values[i], y_values[i], b_scalar);

        for (int j = 0; j < 2; ++j) {
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
    GTEST_LOG_(INFO) << "Average relative error: " << total_error / (x_values.size() * 2);
}