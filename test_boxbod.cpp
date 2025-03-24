#include <eigen3/Eigen/Core>
#include <iostream>
#include "Dual.h"
#include "DualScalar.h"
typedef dual::DualScalar<double> d_scalar;
typedef dual::Dual<d_scalar, 2> dir_hessian_type;

dir_hessian_type
boxbod(double x, double y, dir_hessian_type b[2]){
    return y - b[0] * (1.0 - dual::exp(-b[1] * x));
}
long double *boxbod_grad(long double x, long double y, long double b[2]){
    long double *result = new long double[2] {
            -1.0 + std::exp(-b[1] * x),
            -(b[0] * x) * std::exp(-b[1] * x)
    };
    return result;
}

int main() {
    dir_hessian_type b[2] = {
            dir_hessian_type(d_scalar(2.1380940889E+02, 1), 0),
            dir_hessian_type(d_scalar(5.4723748542E-01, 1), 1),
    };
    long double b_scal[2] = {
            2.1380940889E+02,
            5.4723748542E-01
    };
    double x[6] = {
            1.0,
            2.0,
            3.0,
            5.0,
            7.0,
            10.0
    }, y[6] = {
            109.0,
            149.0,
            149.0,
            191.0,
            213.0,
            224.0
    };
    bool equiv = true;
    for(int i = 0; i < 6; i++){
        dir_hessian_type eps = boxbod(x[i], y[i], b);
        dual::getGradient(eps);
        long double *grad = boxbod_grad(x[i], y[i], b_scal);
        for(int j = 0; j < 2; j++){
            equiv *= std::abs(eps.y[j].x - grad[j]) / std::max(1.0L, std::abs(grad[j])) < 6e-16;
        }
    }
    std::cout << (equiv ? "The gradients match! Passed! " : "The gradients don't match. Failed. ");
    return 0;
}
