#include <gtest/gtest.h>
#include "Clock.h"
#include "Dual.h"
#include "DualScalar.h"
#include "DualDirectional.h"
#include <numeric>

const int dim = 4;
typedef dual::DualScalar<double> d_scalar;
typedef dual::Dual<d_scalar, dim> nested_dual_type;
typedef dual::DualDirectional<double, dim> generalized_dual_type;

class Rat43HessianTest : public ::testing::Test {
protected:
    void SetUp() override {
        direction = Eigen::Vector<double, dim>::Ones().normalized();

        for (int i = 0; i < dim; ++i) {
            b_nested.emplace_back(d_scalar(b_scalar[i], direction[i]), i);
            b_generalized.emplace_back(b_scalar[i], direction[i], i);
        }

        x_values = {1.0E0,
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
                    15.0E0};
        y_values = {16.08E0,
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
                    717.41E0};
    }

    std::vector<nested_dual_type> b_nested;
    std::vector<generalized_dual_type> b_generalized;
    Eigen::Vector<double, dim> direction;
    const std::vector<double> b_scalar = {6.9964151270E+02,
                                          5.2771253025E+00,
                                          7.5962938329E-01,
                                          1.2792483859E+00};
    std::vector<double> x_values;
    std::vector<double> y_values;

    template<typename DualType>
    auto rat43(double x, double y, const std::vector<DualType>& b) {
        return y - b[0] / pow(1.0 + exp(b[1] - b[2] * x), 1.0 / b[3]);
    }
};

TEST_F(Rat43HessianTest, DirectionalHessianAgreement) {
    constexpr double tolerance = 1e-14;

    for (size_t i = 0; i < x_values.size(); ++i) {
        auto nested_result = rat43(x_values[i], y_values[i], b_nested);
        auto generalized_result = rat43(x_values[i], y_values[i], b_generalized);

        auto nested_hessian = dual::getDirectionalHessian(nested_result);
        auto generalized_hessian = generalized_result.v;

        double relative_error = (nested_hessian - generalized_hessian).norm() /
                                std::max(1.0, std::abs(nested_hessian.norm()));

        EXPECT_LT(relative_error, tolerance)
                            << "At x=" << x_values[i]
                            << ": Nested=" << nested_hessian.transpose() << std::endl
                            << ", Generalized=" << generalized_hessian.transpose();
    }
}

TEST_F(Rat43HessianTest, Performance) {
    constexpr int runs = 10000;
    stopwatch<std::chrono::nanoseconds> timer;
    double control_sum = 0;

    // 1. Benchmark nested version
    timer.start();
    for (int i = 0; i < runs; ++i) {
        control_sum += rat43(x_values[i%x_values.size()], y_values[i%y_values.size()], b_nested).x.x;
    }
    timer.stop();
    double nested_time = timer.total_time();
    timer = stopwatch<std::chrono::nanoseconds>();
    std::cout << control_sum << std::endl;

    control_sum = 0;

    // 2. Benchmark generalized version
    timer.start();
    for (int i = 0; i < runs; ++i) {
        control_sum += rat43(x_values[i%x_values.size()], y_values[i%y_values.size()], b_generalized).x;
    }
    timer.stop();
    double generalized_time = timer.total_time();
    std::cout << control_sum << std::endl;
    std::cout << "\nPerformance Results (avg ns/op):\n"
              << "Nested:      " << nested_time/runs << "\n"
              << "Generalized: " << generalized_time/runs << "\n"
              << "Speedup:     " << nested_time/generalized_time << "x\n";
}