//#include <eigen3/Eigen/Core>
#include "Eigen/Core"
#include <iostream>
#include "Dual.h"
#include "DualScalar.h"
typedef dual::DualScalar<long double> d_scalar;
typedef dual::Dual<d_scalar, 3> dir_hessian_type;

dir_hessian_type
mgh10(long double x, long double y, dir_hessian_type b[3]){
    return y - b[0] * exp(b[1] / (x + b[2]));
}
long double *mgh10_grad(long double x, long double y, long double b[3]){
    long double *result = new long double[3] {
            -exp(b[1] / (x + b[2])),
            -b[0] * exp(b[1] / (x + b[2])) / (x + b[2]),
            b[0] * exp(b[1] / (x + b[2])) * b[1] / ((x + b[2]) * (x + b[2]))
    };
    return result;
}

int main() {
    dir_hessian_type b[3] = {
            dir_hessian_type(d_scalar(5.6096364710E-03, 1), 0),
            dir_hessian_type(d_scalar(6.1813463463E+03, 1), 1),
            dir_hessian_type(d_scalar(3.4522363462E+02, 1), 2)
    };
    long double b_scal[3] = {
            5.6096364710E-03,
            6.1813463463E+03,
            3.4522363462E+02
    };

    long double x[16] = {
            5.000000E+01,
            5.500000E+01,
            6.000000E+01,
            6.500000E+01,
            7.000000E+01,
            7.500000E+01,
            8.000000E+01,
            8.500000E+01,
            9.000000E+01,
            9.500000E+01,
            1.000000E+02,
            1.050000E+02,
            1.100000E+02,
            1.150000E+02,
            1.200000E+02,
            1.250000E+02
    },

            y[16] = {
            3.478000E+04,
            2.861000E+04,
            2.365000E+04,
            1.963000E+04,
            1.637000E+04,
            1.372000E+04,
            1.154000E+04,
            9.744000E+03,
            8.261000E+03,
            7.030000E+03,
            6.005000E+03,
            5.147000E+03,
            4.427000E+03,
            3.820000E+03,
            3.307000E+03,
            2.872000E+03
    };
    bool equiv = true;
    for(int i = 0; i < 16; i++){
        dir_hessian_type eps = mgh10(x[i], y[i], b);
        long double *grad = mgh10_grad(x[i], y[i], b_scal);
        for(int j = 0; j < 3; j++){
            equiv *= std::abs(eps.y[j].x - grad[j]) / std::max(1.0L, grad[j]) < 1e-8;
        }
    }
    std::cout << (equiv ? "The gradients match! Passed! " : "The gradients don't match. Failed. ");
    return 0;
}