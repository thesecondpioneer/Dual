#include <iostream>
#include "Clock.h"
#include <eigen3/Eigen/Core>
#include "Dual.h"
#include "DualScalar.h"

int main() {
    dual::Dual<dual::DualScalar<double>, 2> dir_hessian, a(dual::DualScalar<double>(1.0, 1.0), 0),
    b(dual::DualScalar<double>(2.0, 2.0), 1);
    dir_hessian = cos(a) * sin(b) * log(b) * exp(a) * sinh(a);
    std::cout << getDirectionalHessian(dir_hessian).transpose() << std::endl;
    return 0;
}