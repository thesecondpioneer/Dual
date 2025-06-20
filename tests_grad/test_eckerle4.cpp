#include <gtest/gtest.h>
#include "Dual.h"
#include "DualScalar.h"

typedef dual::DualScalar<double> d_scalar;
typedef dual::Dual<d_scalar, 3> dir_hessian_type;

class Eckerle4Test : public ::testing::Test {
protected:
    void SetUp() override {
        b_dual = {
                dir_hessian_type(d_scalar(1.5543827178E+00, 1), 0),
                dir_hessian_type(d_scalar(4.0888321754E+00, 1), 1),
                dir_hessian_type(d_scalar(4.5154121844E+02, 1), 2)
        };

        b_scalar = {
                1.5543827178E+00,
                4.0888321754E+00,
                4.5154121844E+02
        };

        x_values = {
                400.000000E0,
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
                500.000000E0
        };
        y_values = {
                0.0001575E0,
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
                0.0000710E0
        };
    }

    std::vector<dir_hessian_type> b_dual;
    std::vector<long double> b_scalar;
    std::vector<long double> x_values;
    std::vector<long double> y_values;

    dir_hessian_type eckerle4(double x, double y, std::vector<dir_hessian_type> b) {
        return y - b[0] / b[1] * exp(-(x - b[2]) * (x - b[2]) / (2 * b[1] * b[1]));
    }

    std::vector<long double> eckerle4_grad(long double x, long double y, std::vector<long double> b) {
        return {
                -exp(-(x - b[2]) * (x - b[2]) / (2 * b[1] * b[1])) / b[1],
                -(b[0] * x * x - 2.0L * b[0] * b[2] * x + b[0] * b[2] * b[2] - b[0] * b[1] * b[1]) * exp(-(x - b[2]) * (x - b[2]) / (2 * b[1] * b[1])) / (std::pow(b[1], 4.0L)),
                (b[0] * b[2] - b[0] * x) * exp(-(x - b[2]) * (x - b[2]) / (2 * b[1] * b[1])) / std::pow(b[1], 3.0L)
        };
    }
};

TEST_F(Eckerle4Test, GradientComparison) {
    constexpr long double tolerance = 6e-16;

    long double total_error = 0.0L;
    for (size_t i = 0; i < x_values.size(); ++i) {
        dir_hessian_type eps = eckerle4(x_values[i], y_values[i], b_dual);
        dual::getGradient(eps);
        auto grad = eckerle4_grad(x_values[i], y_values[i], b_scalar);

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