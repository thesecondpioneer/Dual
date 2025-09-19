#include <gtest/gtest.h>
#include "Clock.h"
#include "Dual.h"
#include "DualScalar.h"
#include "DualDirectional.h"
#include <numeric>

const int dim = 2;
typedef dual::DualScalar<double> d_scalar;
typedef dual::Dual<d_scalar, dim> nested_dual_type;
typedef dual::DualDirectional<double, dim> generalized_dual_type;

class BoxbodHessianTest : public ::testing::Test {
protected:
    void SetUp() override {
        direction = Eigen::Vector<double, dim>::Ones().normalized();

        for (int i = 0; i < dim; ++i) {
            b_nested.emplace_back(d_scalar(b_scalar[i], direction[i]), i);
            b_generalized.emplace_back(b_scalar[i], direction[i], i);
        }

        x_values = {1.0, 2.0, 3.0, 5.0, 7.0, 10.0};
        y_values = {109.0, 149.0, 149.0, 191.0, 213.0, 224.0};
    }

    std::vector<nested_dual_type> b_nested;
    std::vector<generalized_dual_type> b_generalized;
    Eigen::Vector<double, dim> direction;
    const std::vector<double> b_scalar = {2.1380940889E+02, 5.4723748542E-01};
    std::vector<double> x_values;
    std::vector<double> y_values;

    template<typename DualType>
    auto boxbod(double x, double y, const std::vector<DualType>& b) {
        return y - b[0] * (1.0 - dual::exp(-b[1] * x));
    }
};

TEST_F(BoxbodHessianTest, DirectionalHessianAgreement) {
    constexpr double tolerance = 1e-14;

    for (size_t i = 0; i < x_values.size(); ++i) {
        auto nested_result = boxbod(x_values[i], y_values[i], b_nested);
        auto generalized_result = boxbod(x_values[i], y_values[i], b_generalized);

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

TEST_F(BoxbodHessianTest, Performance) {
    constexpr int runs = 10000;
    stopwatch<std::chrono::nanoseconds> timer;
    double control_sum = 0;

    // 1. Benchmark nested version
    timer.start();
    for (int i = 0; i < runs; ++i) {
        control_sum += boxbod(x_values[i % x_values.size()], y_values[i % y_values.size()], b_nested).x.x;
    }
    timer.stop();
    double nested_time = timer.total_time();
    timer = stopwatch<std::chrono::nanoseconds>();
    //std::cout << control_sum << std::endl;

    control_sum = 0;

    // 2. Benchmark generalized version
    timer.start();
    for (int i = 0; i < runs; ++i) {
        control_sum += boxbod(x_values[i % x_values.size()], y_values[i % y_values.size()], b_generalized).x;
    }
    timer.stop();
    double generalized_time = timer.total_time();
    //std::cout << control_sum << std::endl;
    GTEST_LOG_(INFO) << "\nPerformance Results for Boxbod (avg ns/op):\n"
              << "Nested:      " << nested_time/runs << "\n"
              << "Generalized: " << generalized_time/runs << "\n"
              << "Speedup:     " << nested_time/generalized_time << "x\n";
}