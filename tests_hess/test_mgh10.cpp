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

class Mgh10HessianTest : public ::testing::Test {
protected:
    void SetUp() override {
        direction = Eigen::Vector<double, dim>::Ones().normalized();

        for (int i = 0; i < dim; ++i) {
            b_nested.emplace_back(d_scalar(b_scalar[i], direction[i]), i);
            b_generalized.emplace_back(b_scalar[i], direction[i], i);
        }

        x_values = {5.000000E+01,5.500000E+01,6.000000E+01,6.500000E+01,7.000000E+01,7.500000E+01,8.000000E+01,
                    8.500000E+01,9.000000E+01,9.500000E+01,1.000000E+02,1.050000E+02,1.100000E+02,1.150000E+02,
                    1.200000E+02,1.250000E+02};
        y_values = {3.478000E+04,2.861000E+04,2.365000E+04,1.963000E+04,1.637000E+04,1.372000E+04,1.154000E+04,
                    9.744000E+03,8.261000E+03,7.030000E+03,6.005000E+03,5.147000E+03,4.427000E+03,3.820000E+03,
                    3.307000E+03,2.872000E+03};
    }

    std::vector<nested_dual_type> b_nested;
    std::vector<generalized_dual_type> b_generalized;
    Eigen::Vector<double, dim> direction;
    const std::vector<double> b_scalar = {5.6096364710E-03, 6.1813463463E+03, 3.4522363462E+02};
    std::vector<double> x_values;
    std::vector<double> y_values;

    template<typename DualType>
    auto mgh10(double x, double y, const std::vector<DualType>& b) {
        return y - b[0] * exp(b[1] / (x + b[2]));
    }
};

TEST_F(Mgh10HessianTest, DirectionalHessianAgreement) {
    constexpr double tolerance = 1e-14;

    for (size_t i = 0; i < x_values.size(); ++i) {
        auto nested_result = mgh10(x_values[i], y_values[i], b_nested);
        auto generalized_result = mgh10(x_values[i], y_values[i], b_generalized);

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

TEST_F(Mgh10HessianTest, Performance) {
    constexpr int runs = 10000;
    stopwatch<std::chrono::nanoseconds> timer;
    double control_sum = 0;

    // 1. Benchmark nested version
    timer.start();
    for (int i = 0; i < runs; ++i) {
        control_sum += mgh10(x_values[i%x_values.size()], y_values[i%y_values.size()], b_nested).x.x;
    }
    timer.stop();
    double nested_time = timer.total_time();
    timer = stopwatch<std::chrono::nanoseconds>();
    //std::cout << control_sum << std::endl;

    control_sum = 0;

    // 2. Benchmark generalized version
    timer.start();
    for (int i = 0; i < runs; ++i) {
        control_sum += mgh10(x_values[i%x_values.size()], y_values[i%y_values.size()], b_generalized).x;
    }
    timer.stop();
    double generalized_time = timer.total_time();
    //std::cout << control_sum << std::endl;
    GTEST_LOG_(INFO) << "\nPerformance Results for MGH10 (avg ns/op):\n"
              << "Nested:      " << nested_time/runs << "\n"
              << "Generalized: " << generalized_time/runs << "\n"
              << "Speedup:     " << nested_time/generalized_time << "x\n";
}