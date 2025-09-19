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

class Eckerle4HessianTest : public ::testing::Test {
protected:
    void SetUp() override {
        direction = Eigen::Vector<double, dim>::Ones().normalized();

        for (int i = 0; i < dim; ++i) {
            b_nested.emplace_back(d_scalar(b_scalar[i], direction[i]), i);
            b_generalized.emplace_back(b_scalar[i], direction[i], i);
        }

        x_values = {400.000000E0,
                    405.000000E0,
                    410.000000E0,
                    415.000000E0,
                    420.000000E0,
                    425.000000E0,
                    430.000000E0,
                    435.000000E0,
                    436.500000E0,
                    438.000000E0,
                    439.500000E0,
                    441.000000E0,
                    442.500000E0,
                    444.000000E0,
                    445.500000E0,
                    447.000000E0,
                    448.500000E0,
                    450.000000E0,
                    451.500000E0,
                    453.000000E0,
                    454.500000E0,
                    456.000000E0,
                    457.500000E0,
                    459.000000E0,
                    460.500000E0,
                    462.000000E0,
                    463.500000E0,
                    465.000000E0,
                    470.000000E0,
                    475.000000E0,
                    480.000000E0,
                    485.000000E0,
                    490.000000E0,
                    495.000000E0,
                    500.000000E0};
        y_values = {0.0001575E0,
                    0.0001699E0,
                    0.0002350E0,
                    0.0003102E0,
                    0.0004917E0,
                    0.0008710E0,
                    0.0017418E0,
                    0.0046400E0,
                    0.0065895E0,
                    0.0097302E0,
                    0.0149002E0,
                    0.0237310E0,
                    0.0401683E0,
                    0.0712559E0,
                    0.1264458E0,
                    0.2073413E0,
                    0.2902366E0,
                    0.3445623E0,
                    0.3698049E0,
                    0.3668534E0,
                    0.3106727E0,
                    0.2078154E0,
                    0.1164354E0,
                    0.0616764E0,
                    0.0337200E0,
                    0.0194023E0,
                    0.0117831E0,
                    0.0074357E0,
                    0.0022732E0,
                    0.0008800E0,
                    0.0004579E0,
                    0.0002345E0,
                    0.0001586E0,
                    0.0001143E0,
                    0.0000710E0};
    }

    std::vector<nested_dual_type> b_nested;
    std::vector<generalized_dual_type> b_generalized;
    Eigen::Vector<double, dim> direction;
    const std::vector<double> b_scalar = {1.5543827178E+00,
                                          4.0888321754E+00,
                                          4.5154121844E+02};
    std::vector<double> x_values;
    std::vector<double> y_values;

    template<typename DualType>
    auto eckerle4(double x, double y, const std::vector<DualType> &b) {
        return y - b[0] / b[1] * exp(-(x - b[2]) * (x - b[2]) / (2 * b[1] * b[1]));
    }
};

TEST_F(Eckerle4HessianTest, DirectionalHessianAgreement) {
    constexpr double tolerance = 1e-14;

    for (size_t i = 0; i < x_values.size(); ++i) {
        auto nested_result = eckerle4(x_values[i], y_values[i], b_nested);
        auto generalized_result = eckerle4(x_values[i], y_values[i], b_generalized);

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

TEST_F(Eckerle4HessianTest, Performance) {
    constexpr int runs = 10000;
    stopwatch<std::chrono::nanoseconds> timer;
    double control_sum = 0;

    // 1. Benchmark nested version
    timer.start();
    for (int i = 0; i < runs; ++i) {
        control_sum += eckerle4(x_values[i % x_values.size()], y_values[i % y_values.size()], b_nested).x.x;
    }
    timer.stop();
    double nested_time = timer.total_time();
    timer = stopwatch<std::chrono::nanoseconds>();
    //std::cout << control_sum << std::endl;

    control_sum = 0;

    // 2. Benchmark generalized version
    timer.start();
    for (int i = 0; i < runs; ++i) {
        control_sum += eckerle4(x_values[i % x_values.size()], y_values[i % y_values.size()], b_generalized).x;
    }
    timer.stop();
    double generalized_time = timer.total_time();
    //std::cout << control_sum << std::endl;
    GTEST_LOG_(INFO) << "\nPerformance Results for Eckerle4 (avg ns/op):\n"
              << "Nested:      " << nested_time / runs << "\n"
              << "Generalized: " << generalized_time / runs << "\n"
              << "Speedup:     " << nested_time / generalized_time << "x\n";
}