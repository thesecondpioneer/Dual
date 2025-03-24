
#include <eigen3/Eigen/Core>
#include <iostream>
#include "Dual.h"
#include "DualScalar.h"
typedef dual::DualScalar<double> d_scalar;
typedef dual::Dual<d_scalar, 4> dir_hessian_type;

dir_hessian_type
thurber(double x, double y, dir_hessian_type b[7]){
    return y - ();
}
long double *thurber_grad(long double x, long double y, long double b[7]){
    long double *result = new long double[7] {
            -(x * x + b[1] * x) / (x * x + b[2] * x + b[3]),
            -b[0] * x / (x * x + b[2] * x + b[3]),
            (b[0] * std::pow(x, 3) + b[0] * b[1] * x * x) / std::pow((x * x + b[2] * x + b[3]), 2),
            (b[0] * x * x + b[0] * b[1] * x) / std::pow((x * x + b[2] * x + b[3]), 2)
    };
    return result;
}

int main() {
    dir_hessian_type b[7] = {
            dir_hessian_type(d_scalar(1.9280693458E-01, 1), 0),
            dir_hessian_type(d_scalar(1.9128232873E-01, 1), 1),
            dir_hessian_type(d_scalar(1.2305650693E-01, 1), 2),
            dir_hessian_type(d_scalar(1.3606233068E-01, 1), 3),
    };
    long double b_scal[4] = {
            1.9280693458E-01,
            1.9128232873E-01,
            1.2305650693E-01,
            1.3606233068E-01
    };
    double x[11] = {
            4.000000E+00,
            2.000000E+00,
            1.000000E+00,
            5.000000E-01,
            2.500000E-01,
            1.670000E-01,
            1.250000E-01,
            1.000000E-01,
            8.330000E-02,
            7.140000E-02,
            6.250000E-02

    }, y[11] = {
            1.957000E-01,
            1.947000E-01,
            1.735000E-01,
            1.600000E-01,
            8.440000E-02,
            6.270000E-02,
            4.560000E-02,
            3.420000E-02,
            3.230000E-02,
            2.350000E-02,
            2.460000E-02

    };
    bool equiv = true;
    for(int i = 0; i < 11; i++){
        dir_hessian_type eps = thurber(x[i], y[i], b);
        dual::getGradient(eps);
        long double *grad = thurber_grad(x[i], y[i], b_scal);
        for(int j = 0; j < 4; j++){
            equiv = std::abs(eps.y[j].x - grad[j]) / std::max(1.0L, grad[j]) < 1e-16;
        }
        delete[] grad;
    }
    std::cout << (equiv ? "The gradients match! Passed! " : "The gradients don't match. Failed. ");
    return 0;
}
