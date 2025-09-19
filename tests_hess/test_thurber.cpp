#include <gtest/gtest.h>
#include "Clock.h"
#include "Dual.h"
#include "DualScalar.h"
#include "DualDirectional.h"
#include <numeric>

const int dim = 7;
typedef dual::DualScalar<double> d_scalar;
typedef dual::Dual<d_scalar, dim> nested_dual_type;
typedef dual::DualDirectional<double, dim> generalized_dual_type;

class ThurberHessianTest : public ::testing::Test {
protected:
    void SetUp() override {
        direction = Eigen::Vector<double, dim>::Ones().normalized();

        for (int i = 0; i < dim; ++i) {
            b_nested.emplace_back(d_scalar(b_scalar[i], direction[i]), i);
            b_generalized.emplace_back(b_scalar[i], direction[i], i);
        }

        x_values = {-3.067E0,
                    -2.981E0,
                    -2.921E0,
                    -2.912E0,
                    -2.840E0,
                    -2.797E0,
                    -2.702E0,
                    -2.699E0,
                    -2.633E0,
                    -2.481E0,
                    -2.363E0,
                    -2.322E0,
                    -1.501E0,
                    -1.460E0,
                    -1.274E0,
                    -1.212E0,
                    -1.100E0,
                    -1.046E0,
                    -0.915E0,
                    -0.714E0,
                    -0.566E0,
                    -0.545E0,
                    -0.400E0,
                    -0.309E0,
                    -0.109E0,
                    -0.103E0,
                    0.010E0,
                    0.119E0,
                    0.377E0,
                    0.790E0,
                    0.963E0,
                    1.006E0,
                    1.115E0,
                    1.572E0,
                    1.841E0,
                    2.047E0,
                    2.200E0};
        y_values = {80.574E0,
                    84.248E0,
                    87.264E0,
                    87.195E0,
                    89.076E0,
                    89.608E0,
                    89.868E0,
                    90.101E0,
                    92.405E0,
                    95.854E0,
                    100.696E0,
                    101.060E0,
                    401.672E0,
                    390.724E0,
                    567.534E0,
                    635.316E0,
                    733.054E0,
                    759.087E0,
                    894.206E0,
                    990.785E0,
                    1090.109E0,
                    1080.914E0,
                    1122.643E0,
                    1178.351E0,
                    1260.531E0,
                    1273.514E0,
                    1288.339E0,
                    1327.543E0,
                    1353.863E0,
                    1414.509E0,
                    1425.208E0,
                    1421.384E0,
                    1442.962E0,
                    1464.350E0,
                    1468.705E0,
                    1447.894E0,
                    1457.628E0};
    }

    std::vector<nested_dual_type> b_nested;
    std::vector<generalized_dual_type> b_generalized;
    Eigen::Vector<double, dim> direction;
    const std::vector<double> b_scalar = {1.2881396800E+03,
                                          1.4910792535E+03,
                                          5.8323836877E+02,
                                          7.5416644291E+01,
                                          9.6629502864E-01,
                                          3.9797285797E-01,
                                          4.9727297349E-02};
    std::vector<double> x_values;
    std::vector<double> y_values;

    template<typename DualType>
    auto thurber(double x, double y, const std::vector<DualType>& b) {
        return y - (b[0] + b[1] * x + b[2] * x * x + b[3] * std::pow(x, 3)) / (1.0 + b[4] * x + b[5] * x * x + b[6] * std::pow(x, 3));
    }
};

TEST_F(ThurberHessianTest, DirectionalHessianAgreement) {
    constexpr double tolerance = 1e-14;

    for (size_t i = 0; i < x_values.size(); ++i) {
        auto nested_result = thurber(x_values[i], y_values[i], b_nested);
        auto generalized_result = thurber(x_values[i], y_values[i], b_generalized);

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

TEST_F(ThurberHessianTest, Performance) {
    constexpr int runs = 10000;
    stopwatch<std::chrono::nanoseconds> timer;
    double control_sum = 0;

    // 1. Benchmark nested version
    timer.start();
    for (int i = 0; i < runs; ++i) {
        control_sum += thurber(x_values[i % x_values.size()], y_values[i % y_values.size()], b_nested).x.x;
    }
    timer.stop();
    double nested_time = timer.total_time();
    timer = stopwatch<std::chrono::nanoseconds>();
    //std::cout << control_sum << std::endl;

    control_sum = 0;

    // 2. Benchmark generalized version
    timer.start();
    for (int i = 0; i < runs; ++i) {
        control_sum += thurber(x_values[i % x_values.size()], y_values[i % y_values.size()], b_generalized).x;
    }
    timer.stop();
    double generalized_time = timer.total_time();
    //std::cout << control_sum << std::endl;
    GTEST_LOG_(INFO) << "\nPerformance Results for Thurber (avg ns/op):\n"
              << "Nested:      " << nested_time/runs << "\n"
              << "Generalized: " << generalized_time/runs << "\n"
              << "Speedup:     " << nested_time/generalized_time << "x\n";
}//
// Created by pio on 9/19/25.
//
