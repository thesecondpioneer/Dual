#include <iostream>
#include "Clock.h"
#include <vector>
#include "Codual.h"
#include "Eigen/Core"

#include "Dual.h"
dual::Codual<double> rosenbrock_codual(const std::vector<dual::Codual<double>> &xval) {
    dual::Codual<double> result(0, [=](const double &dx) {});
    for (int i = 0; i < xval.size() - 1; i++) {
        result += dual::pow(1.0 - xval[i], 2.0) + 100.0 * (xval[i + 1] - dual::pow(xval[i], 2.0));
    }
    return result;
}

//template<int N>
//dual::Dual<double, N> rosenbrock_dual(const std::vector<dual::Dual<double, N>> &xval){
//    dual::Dual<double, N> result(0);
//    for (int i = 0; i < N - 1; i++){
//        result += dual::pow(1.0 - xval[i], 2.0) + 100.0 * (xval[i + 1] - dual::pow(xval[i], 2.0));
//    }
//    return result;
//}
//
//template<int N>
//std::vector<dual::Dual<double, N>> dual_arg(const std::vector<double> &xval){
//    std::vector<dual::Dual<double, N>> result(N);
//    for(int i = 0; i < N; i++){
//        result[i].x = xval[i];
//        result[i].y[i] = 1.0;
//    }
//    return result;
//}

std::vector<double>
get_derivative(const std::vector<double> &xval, dual::Codual<double> (*f)(const std::vector<dual::Codual<double>> &)) {
    std::vector<double> dx(xval.size(), 0);
    std::vector<dual::Codual<double>> args(xval.size());
    for (int i = 0; i < xval.size(); i++) {
        args[i] = dual::Codual(xval[i],
                               std::function<void(const double &)>([=, &dx](const double &dy) { dx[i] += dy; }));
    }
    dual::Codual<double> y = f(args);
    y.derivative(1.0);
    return dx;
}

//dual::Codual<double> scalarfunction(const dual::Codual<double> &xval){
//    dual::Codual<double> result(0, [&](const double &dx){});
//    result += dual::pow(xval, 2.0) + 100.0 * (xval - dual::pow(xval, 2.0));
//    return result;
//}
//
//double get_derivative_scalar(const double &xval, dual::Codual<double> (*f)(const dual::Codual<double>&)){
//    double dx;
//    dual::Codual<double> arg;
//    arg = dual::Codual(xval, std::function<void(const double &)>([&dx](const double &dy){dx += dy;}));
//    dual::Codual<double> y = f(arg);
//    y.derivative(1.0);
//    return dx;
//}

int main(int argc, char *argv[]) {
    std::vector<double> xvals(1000, 2.0);
    stopwatch time;
    //double xval = 2.0;
    //double deriv = get_derivative_scalar(xval, scalarfunction);
    time.start();
    std::vector<double> derivs = get_derivative(xvals, rosenbrock_codual);
    time.stop();
    for (auto i: derivs) {
        std::cout << i << ' ';
    }
    std::cout << std::endl;
    printf("%.2f\n", time.total_time()/1e9);
    return 0;
}
