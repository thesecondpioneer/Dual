//#include <eigen3/Eigen/Core>
#include "Eigen/Core"
#include <iostream>
#include "Dual.h"
#include "DualScalar.h"
typedef dual::DualScalar<long double> d_scalar;
typedef dual::Dual<d_scalar, 3> dir_hessian_type;

dir_hessian_type
eckerle4(long double x, long double y, dir_hessian_type b[3]){
    return y - b[0] / b[1] * exp(-(x - b[2]) * (x - b[2]) / (2 * b[1] * b[1]));
}
long double *eckerle4_grad(long double x, long double y, long double b[3]){
    long double *result = new long double[3] {
            -exp(-(x - b[2]) * (x - b[2]) / (2 * b[1] * b[1])) / b[1],
            -(b[0] * x * x - 2.0L * b[0] * b[2] * x + b[0] * b[2] * b[2] - b[0] * b[1] * b[1]) * exp(-(x - b[2]) * (x - b[2]) / (2 * b[1] * b[1])) / (std::pow(b[1], 4.0L)),
            (b[0] * b[2] - b[0] * x) * exp(-(x - b[2]) * (x - b[2]) / (2 * b[1] * b[1])) / std::pow(b[1], 3.0L)
    };
    return result;
}

int main() {
    dir_hessian_type b[3] = {
            dir_hessian_type(d_scalar(1.5543827178E+00, 1), 0),
            dir_hessian_type(d_scalar(4.0888321754E+00, 1), 1),
            dir_hessian_type(d_scalar(4.5154121844E+02, 1), 2)
    };
    long double b_scal[3] = {
            1.5543827178E+00,
            4.0888321754E+00,
            4.5154121844E+02
    };

    long double x[35] = {
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
    },
            y[35] = {
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
    bool equiv = true;
    for(int i = 0; i < 35; i++){
        dir_hessian_type eps = eckerle4(x[i], y[i], b);
        long double *grad = eckerle4_grad(x[i], y[i], b_scal);
        for(int j = 0; j < 3; j++){
            equiv *= std::abs(eps.y[j].x - grad[j]) / std::max(1.0L, grad[j]) < 1e-15;
        }
    }
    std::cout << (equiv ? "The gradients match! Passed! " : "The gradients don't match. Failed. ");
    return 0;
}