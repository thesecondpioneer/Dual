#include <eigen3/Eigen/Core>
#include <iostream>
#include "Dual.h"
#include "DualScalar.h"
typedef dual::DualScalar<double> d_scalar;
typedef dual::Dual<d_scalar, 7> dir_hessian_type;

dir_hessian_type
thurber(double x, double y, dir_hessian_type b[7]){
    return y - (b[0] + b[1] * x + b[2] * x * x + b[3] * std::pow(x, 3)) / (1.0 + b[4] * x + b[5] * x * x + b[6] * std::pow(x, 3));
}
long double *thurber_grad(long double x, long double y, long double b[7]){
    long double *result = new long double[7] {
            -(1.0) / (1.0 + b[4] * x + b[5] * x * x + b[6] * std::pow(x, 3)),
            -(x) / (1.0 + b[4] * x + b[5] * x * x + b[6] * std::pow(x, 3)),
            -(x * x) / (1.0 + b[4] * x + b[5] * x * x + b[6] * std::pow(x, 3)),
            -(std::pow(x, 3)) / (1.0 + b[4] * x + b[5] * x * x + b[6] * std::pow(x, 3)),
            (b[0] * x + b[1] * x * x + b[2] * std::pow(x, 3) + b[3] * std::pow(x, 4)) / std::pow((1.0 + b[4] * x + b[5] * x * x + b[6] * std::pow(x, 3)), 2),
            (b[0] * x * x + b[1] * std::pow(x, 3) + b[2] * std::pow(x, 4) + b[3] * std::pow(x, 5)) / std::pow((1.0 + b[4] * x + b[5] * x * x + b[6] * std::pow(x, 3)), 2),
            (b[0] * std::pow(x, 3) + b[1] * std::pow(x, 4) + b[2] * std::pow(x, 5) + b[3] * std::pow(x, 6)) / std::pow((1.0 + b[4] * x + b[5] * x * x + b[6] * std::pow(x, 3)), 2)
    };
    return result;
}

int main() {
    dir_hessian_type b[7] = {
            dir_hessian_type(d_scalar(1.2881396800E+03, 1), 0),
            dir_hessian_type(d_scalar(1.4910792535E+03, 1), 1),
            dir_hessian_type(d_scalar(5.8323836877E+02, 1), 2),
            dir_hessian_type(d_scalar(7.5416644291E+01, 1), 3),
            dir_hessian_type(d_scalar(9.6629502864E-01, 1), 4),
            dir_hessian_type(d_scalar(3.9797285797E-01, 1), 5),
            dir_hessian_type(d_scalar(4.9727297349E-02, 1), 6)
    };
    long double b_scal[7] = {
            1.2881396800E+03,
            1.4910792535E+03,
            5.8323836877E+02,
            7.5416644291E+01,
            9.6629502864E-01,
            3.9797285797E-01,
            4.9727297349E-02
    };
    double x[37] = {
            -3.067E0,
            -2.981E0,
            -2.921E0,
            -2.912E0,
            -2.840E0,
            -2.797E0,
            -2.702E0,
            -2.699E0,
            -2.633E0,
            -2.481E0,
            -2.363E0,
            -2.322E0,
            -1.501E0,
            -1.460E0,
            -1.274E0,
            -1.212E0,
            -1.100E0,
            -1.046E0,
            -0.915E0,
            -0.714E0,
            -0.566E0,
            -0.545E0,
            -0.400E0,
            -0.309E0,
            -0.109E0,
            -0.103E0,
            0.010E0,
            0.119E0,
            0.377E0,
            0.790E0,
            0.963E0,
            1.006E0,
            1.115E0,
            1.572E0,
            1.841E0,
            2.047E0,
            2.200E0
    }, y[37] = {
            80.574E0,
            84.248E0,
            87.264E0,
            87.195E0,
            89.076E0,
            89.608E0,
            89.868E0,
            90.101E0,
            92.405E0,
            95.854E0,
            100.696E0,
            101.060E0,
            401.672E0,
            390.724E0,
            567.534E0,
            635.316E0,
            733.054E0,
            759.087E0,
            894.206E0,
            990.785E0,
            1090.109E0,
            1080.914E0,
            1122.643E0,
            1178.351E0,
            1260.531E0,
            1273.514E0,
            1288.339E0,
            1327.543E0,
            1353.863E0,
            1414.509E0,
            1425.208E0,
            1421.384E0,
            1442.962E0,
            1464.350E0,
            1468.705E0,
            1447.894E0,
            1457.628E0
    };
    bool equiv = true;
    for(int i = 0; i < 37; i++){
        dir_hessian_type eps = thurber(x[i], y[i], b);
        dual::getGradient(eps);
        long double *grad = thurber_grad(x[i], y[i], b_scal);
        for(int j = 0; j < 7; j++){
            equiv *= std::abs(eps.y[j].x - grad[j]) / std::max(1.0L, grad[j]) < 1e-16;
        }
        delete[] grad;
    }
    std::cout << (equiv ? "The gradients match! Passed! " : "The gradients don't match. Failed. ");
    return 0;
}
