#include <eigen3/Eigen/Core>
#include <iostream>
#include "Dual.h"
#include "DualScalar.h"
typedef dual::DualScalar<double> d_scalar;
typedef dual::Dual<d_scalar, 3> dir_hessian_type;

dir_hessian_type
rat42(double x, double y, dir_hessian_type b[3]){
    return y - b[0] / (1.0 + dual::exp(b[1] - b[2] * x));
}
long double *rat42_grad(long double x, long double y, long double b[3]){
    long double *result = new long double[3] {
            -1.0L / (1.0L + exp(b[1] - b[2] * x)),
            b[0] * exp(b[1] - b[2] * x) / std::pow((1.0L + exp(b[1] - b[2] * x)), 2),
            -b[0] * exp(b[1] - b[2] * x) * x / std::pow((1.0L + exp(b[1] - b[2] * x)), 2)
    };
    return result;
}

int main() {
    dir_hessian_type b[3] = {
            dir_hessian_type(d_scalar(7.2462237576E+01, 1), 0),
            dir_hessian_type(d_scalar(2.6180768402E+00, 1), 1),
            dir_hessian_type(d_scalar(6.7359200066E-02, 1), 2),
    };
    long double b_scal[3] = {
            7.2462237576E+01,
            2.6180768402E+00,
            6.7359200066E-02
    };
    double x[9] = {
            9.000E0,
            14.000E0,
            21.000E0,
            28.000E0,
            42.000E0,
            57.000E0,
            63.000E0,
            70.000E0,
            79.000E0
    }, y[9] = {
            8.930E0,
            10.800E0,
            18.590E0,
            22.330E0,
            39.350E0,
            56.110E0,
            61.730E0,
            64.620E0,
            67.080E0
    };
    bool equiv = true;
    for(int i = 0; i < 11; i++){
        dir_hessian_type eps = rat42(x[i], y[i], b);
        dual::getGradient(eps);
        long double *grad = rat42_grad(x[i], y[i], b_scal);
        for(int j = 0; j < 3; j++){
            equiv *= std::abs(eps.y[j].x - grad[j]) / std::max(1.0L, grad[j]) < 1e-10;
        }
    }
    std::cout << (equiv ? "The gradients match! Passed! " : "The gradients don't match. Failed. ");
    return 0;
}
