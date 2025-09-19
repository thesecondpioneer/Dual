#include <gtest/gtest.h>
#include "Clock.h"
#include "Dual.h"
#include "DualScalar.h"
#include "DualDirectional.h"
#include <numeric>

const int dim = 3;
typedef dual::DualScalar<double> d_scalar;
typedef dual::Dual<d_scalar, dim> nested_dual_type;
typedef dual::DualDirectional<double, dim> generalized_dual_type;

class Rat42HessianTest : public ::testing::Test {
protected:
    void SetUp() override {
        direction = Eigen::Vector<double, dim>::Ones().normalized();

        for (int i = 0; i < dim; ++i) {
            b_nested.emplace_back(d_scalar(b_scalar[i], direction[i]), i);
            b_generalized.emplace_back(b_scalar[i], direction[i], i);
        }

        x_values = {9.000E0,
                    14.000E0,
                    21.000E0,
                    28.000E0,
                    42.000E0,
                    57.000E0,
                    63.000E0,
                    70.000E0,
                    79.000E0};
        y_values = {8.930E0,
                    10.800E0,
                    18.590E0,
                    22.330E0,
                    39.350E0,
                    56.110E0,
                    61.730E0,
                    64.620E0,
                    67.080E0};
    }

    std::vector<nested_dual_type> b_nested;
    std::vector<generalized_dual_type> b_generalized;
    Eigen::Vector<double, dim> direction;
    const std::vector<double> b_scalar = {7.2462237576E+01,
                                          2.6180768402E+00,
                                          6.7359200066E-02};
    std::vector<double> x_values;
    std::vector<double> y_values;

    template<typename DualType>
    auto rat42(double x, double y, const std::vector<DualType>& b) {
        return y - b[0] / (1.0 + dual::exp(b[1] - b[2] * x));
    }
};

TEST_F(Rat42HessianTest, DirectionalHessianAgreement) {
    constexpr double tolerance = 1e-14;

    for (size_t i = 0; i < x_values.size(); ++i) {
        auto nested_result = rat42(x_values[i], y_values[i], b_nested);
        auto generalized_result = rat42(x_values[i], y_values[i], b_generalized);

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

TEST_F(Rat42HessianTest, Performance) {
    constexpr int runs = 10000;
    stopwatch<std::chrono::nanoseconds> timer;
    double control_sum = 0;

    // 1. Benchmark nested version
    timer.start();
    for (int i = 0; i < runs; ++i) {
        control_sum += rat42(x_values[i%x_values.size()], y_values[i%y_values.size()], b_nested).x.x;
    }
    timer.stop();
    double nested_time = timer.total_time();
    timer = stopwatch<std::chrono::nanoseconds>();
    //std::cout << control_sum << std::endl;

    control_sum = 0;

    // 2. Benchmark generalized version
    timer.start();
    for (int i = 0; i < runs; ++i) {
        control_sum += rat42(x_values[i%x_values.size()], y_values[i%y_values.size()], b_generalized).x;
    }
    timer.stop();
    double generalized_time = timer.total_time();
    //std::cout << control_sum << std::endl;
    GTEST_LOG_(INFO) << "\nPerformance Results for Rat42 (avg ns/op):\n"
              << "Nested:      " << nested_time/runs << "\n"
              << "Generalized: " << generalized_time/runs << "\n"
              << "Speedup:     " << nested_time/generalized_time << "x\n";
}