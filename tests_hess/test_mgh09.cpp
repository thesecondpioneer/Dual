#include <gtest/gtest.h>
#include "Clock.h"
#include "Dual.h"
#include "DualScalar.h"
#include "DualDirectional.h"
#include <numeric>

// Portable optimization barrier
template<class T>
void KeepOptimizerOut(T const& value) {
    asm volatile("" : : "r,m"(value) : "memory");
}

typedef dual::DualScalar<double> d_scalar;
typedef dual::Dual<d_scalar, 4> nested_dual_type;
typedef dual::DualDirectional<double, 4> generalized_dual_type;

class Mgh09HessianTest : public ::testing::Test {
protected:
    void SetUp() override {
        direction = Eigen::Vector4d::Ones().normalized();

        for (int i = 0; i < 4; ++i) {
            b_nested.emplace_back(d_scalar(b_scalar[i], direction[i]), i);
            b_generalized.emplace_back(b_scalar[i], direction[i], i);
        }

        x_values = {4.000000E+00, 2.000000E+00, 1.000000E+00, 5.000000E-01,
                    2.500000E-01, 1.670000E-01, 1.250000E-01, 1.000000E-01,
                    8.330000E-02, 7.140000E-02, 6.250000E-02};
        y_values = {1.957000E-01, 1.947000E-01, 1.735000E-01, 1.600000E-01,
                    8.440000E-02, 6.270000E-02, 4.560000E-02, 3.420000E-02,
                    3.230000E-02, 2.350000E-02, 2.460000E-02};
    }

    std::vector<nested_dual_type> b_nested;
    std::vector<generalized_dual_type> b_generalized;
    Eigen::Vector4d direction;
    const std::vector<double> b_scalar = {1.9280693458E-01, 1.9128232873E-01,
                                         1.2305650693E-01, 1.3606233068E-01};
    std::vector<double> x_values;
    std::vector<double> y_values;

    template<typename DualType>
    auto mgh09(double x, double y, const std::vector<DualType>& b) {
        return y - b[0] * (x * x + x * b[1]) / (x * x + x * b[2] + b[3]);
    }
};

TEST_F(Mgh09HessianTest, DirectionalHessianAgreement) {
    constexpr double tolerance = 1e-14;

    for (size_t i = 0; i < x_values.size(); ++i) {
        auto nested_result = mgh09(x_values[i], y_values[i], b_nested);
        auto generalized_result = mgh09(x_values[i], y_values[i], b_generalized);

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

TEST_F(Mgh09HessianTest, Performance) {
    constexpr int runs = 10000;
    stopwatch<std::chrono::nanoseconds> timer;
    double control_sum = 0;

    // 1. Benchmark nested version
    timer.start();
    for (int i = 0; i < runs; ++i) {
        control_sum += mgh09(x_values[i%11], y_values[i%11], b_nested).x.x;
    }
    timer.stop();
    double nested_time = timer.total_time();
    timer = stopwatch<std::chrono::nanoseconds>();
    std::cout << control_sum << std::endl;

    // 2. Benchmark generalized version
    timer.start();
    for (int i = 0; i < runs; ++i) {
        control_sum += mgh09(x_values[i%11], y_values[i%11], b_generalized).x;
    }
    timer.stop();
    double generalized_time = timer.total_time();
    std::cout << control_sum << std::endl;
    std::cout << "\nPerformance Results (avg ns/op):\n"
              << "Nested:      " << nested_time/runs << "\n"
              << "Generalized: " << generalized_time/runs << "\n"
              << "Speedup:     " << nested_time/generalized_time << "x\n";
}