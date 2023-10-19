#include <iostream>
#include "Dual.h"
#include "Clock.h"

using namespace dual;

double f(double x, double y, double b1, double b2, double b3, double b4) {
    return b1 * pow(1.0 + exp(b2 - b3 * x), -1.0 / b4) - y;
}

Eigen::Matrix<double, 4, 1> der(double x, double y, double b1, double b2, double b3, double b4) {
    return Eigen::Matrix<double, 4, 1>{
            -1.0 / pow(1.0 + exp(b2 - b3 * x), 1.0 / b4),
            b1 * exp(b2 - b3 * x) / (b4 * pow(1.0 + exp(b2 - b3 * x), (1.0 + b4) / b4)),
            -b1 * exp(b2 - b3 * x) * x / (b4 * pow(1.0 + exp(b2 - b3 * x), (1.0 + b4) / b4)),
            -b1 * log(1.0 + exp(b2 - b3 * x)) / (b4 * b4 * pow(1.0 + exp(b2 - b3 * x), 1.0 / b4))
    };
}

void opt(double x, double y, double b1, double b2, double b3, double b4, double &f,
         Eigen::Matrix<double, 4, 1> &der) {
    double ex = exp(b2 - b3 * x), exp1 = ex + 1.0, pw = pow(exp1, 1.0 / b4), db2 = b1 * ex / (b4 * pw * exp1);
    f = b1 / pw - y;
    der = Eigen::Matrix<double, 4, 1>{
            -1.0 / pw,
            db2,
            -x * db2,
            -b1 * log(exp1) / (b4 * b4 * pw)
    };
}

Dual<double, 4> autod(double x, double y, Dual<double, 4> b1, Dual<double, 4> b2, Dual<double, 4> b3, Dual<double, 4> b4) {
    return b1 * pow(1.0 + exp(b2 - b3 * x), -1.0 / b4) - y;
}

int main() {
    double opt_func, analyt_func;
    Eigen::Matrix<double, 4, 1> opt_deriv, analyt_deriv;
    Dual<double, 4> dual_deriv;
    std::cout.precision(16);
    stopwatch opt_time, dual_time, analyt_time;
    opt_time.start();
    for (double x = 1; x <= 1; x += 0.001) {
        opt(x, 3.0, 1.0, 1.0, 1.0, 1.0, opt_func, opt_deriv);
    }
    opt_time.stop();
    dual_time.start();
    for (double x = 1; x <= 1; x += 0.001) {
        dual_deriv = autod(x, 3.0, Dual<double, 4>(1.0, 0), Dual<double, 4>(1.0, 1), Dual<double, 4>(1.0, 2), Dual<double, 4>(1.0, 3));
    }
    dual_time.stop();
    analyt_time.start();
    for (double x = 1; x <= 1; x += 0.001) {
        analyt_func = f(x, 3.0, 1.0, 1.0, 1.0, 1.0);
        analyt_deriv = der(x, 3.0, 1.0, 1.0, 1.0, 1.0);
    }
    analyt_time.stop();
    std::cout << std::string(33, ' ') << "Last function value  " << "  Last derivative value" << std::string(63, ' ') <<"Time" << std::endl
              << std::fixed << "Analytical derivative:" << std::string(11, ' ') << analyt_func << "    " << analyt_deriv.transpose()
              << "     " << analyt_time.total_time() << std::endl << "Optimized analytical derivative: " << opt_func
              << "    " << opt_deriv.transpose() << "     " << opt_time.total_time() << std::endl << "Automatic differentiation:"
              << std::string(7, ' ') << dual_deriv.x << "    " << dual_deriv.y.transpose() << "     " << dual_time.total_time()
              << std::endl;
    return 0;
}