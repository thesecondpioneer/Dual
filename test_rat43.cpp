//#include <eigen3/Eigen/Core>
#include "Eigen/Core"
#include <iostream>
#include "Dual.h"
#include "DualScalar.h"
typedef dual::DualScalar<long double> d_scalar;
typedef dual::Dual<d_scalar, 4> dir_hessian_type;

dir_hessian_type
rat43(long double x, long double y, dir_hessian_type b[4]){
    return y - b[0] / pow(1.0L + exp(b[1] - b[2] * x), 1.0L / b[3]);
}
long double *rat43_grad(long double x, long double y, long double b[4]){
    long double *result = new long double[4] {
            -1.0L / std::pow(1.0L + std::exp(b[1] - b[2] * x), 1.0L / b[3]),
            b[0] * std::exp(b[1] - b[2] * x) / (b[3] * std::pow(1.0L + std::exp(b[1] - b[2] * x), (1.0L + b[3]) / b[3])),
            -b[0] * std::exp(b[1] - b[2] * x) * x / (b[3] * std::pow(1.0L + std::exp(b[1] - b[2] * x), (1.0L + b[3]) / b[3])),
            -b[0] * std::log(1.0L + std::exp(b[1] - b[2] * x)) / (pow(1.0L + exp(b[1] - b[2] * x), 1.0L / b[3]) * b[3] * b[3])
    };
    return result;
}

int main() {
    dir_hessian_type b[4] = {
            dir_hessian_type(d_scalar(6.9964151270E+02, 1), 0),
            dir_hessian_type(d_scalar(5.2771253025E+00, 1), 1),
            dir_hessian_type(d_scalar(7.5962938329E-01, 1), 2),
            dir_hessian_type(d_scalar(1.2792483859E+00, 1), 3)
    };

    long double b_scal[4] = {
            6.9964151270E+02,
            5.2771253025E+00,
            7.5962938329E-01,
            1.2792483859E+00
    };

    long double x[15] = {
            1.0E0,
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
            15.0E0
    },

            y[15] = {
            16.08E0,
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
            717.41E0
    };
    bool equiv = true;
    for(int i = 0; i < 15; i++){
        dir_hessian_type eps = rat43(x[i], y[i], b);
        long double *grad = rat43_grad(x[i], y[i], b_scal);
        for(int j = 0; j < 3; j++){
            equiv *= std::abs(eps.y[j].x - grad[j]) / std::max(1.0L, grad[j]) < 1e-15;
        }
    }
    std::cout << (equiv ? "The gradients match! Passed! " : "The gradients don't match. Failed. ");
    return 0;
}