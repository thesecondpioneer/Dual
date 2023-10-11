#include <iostream>
#include "Dual.h"
#include "Clock.h"

using namespace dual;

float f(float x) {
    return std::pow(std::cos(x), std::sin(x));
}

float y(float x) {
    return -pow(std::cos(x), std::sin(x) - 1) * std::pow(std::sin(x), 2) +
           std::pow(std::cos(x), std::sin(x) + 1) * std::log(std::cos(x));
}

void opt(float x, float &f, float &y) {
    float sx = std::sin(x), cx = std::cos(x), ctsx = std::pow(cx, sx);
    f = ctsx;
    y = -ctsx / cx * std::pow(sx, 2) + ctsx * cx * std::log(cx);
}

Dual<float> autod(Dual<float> x) {
    return pow(cos(x), sin(x));
}

int main() {
    float opt_func, opt_deriv, analyt_func, analyt_deriv;
    Dual<float> dual_deriv;
    std::cout.precision(16);
    stopwatch opt_time, dual_time, analyt_time;
    opt_time.start();
    for (float x = 1; x < 3; x += 0.001f) {
        opt(1.5f, opt_func, opt_deriv);
    }
    opt_time.stop();
    dual_time.start();
    for (float x = 1; x < 3; x += 0.001f) {
        dual_deriv = autod(Dual<float>(1.5f, 1.0f));
    }
    dual_time.stop();
    analyt_time.start();
    for (float x = 1; x < 3; x += 0.001f) {
        analyt_func = f(1.5);
        analyt_deriv = y(1.5);
    }
    analyt_time.stop();
    std::cout << std::string(33, ' ') << "Last function value  " << " Last derivative value  Time" << std::endl
              << std::fixed << "Analytical derivative:" << std::string(11, ' ') << analyt_func << "    " << analyt_deriv
              << "     " << analyt_time.total_time() << std::endl << "Optimized analytical derivative: " << opt_func
              << "    " << opt_deriv << "     " << opt_time.total_time() << std::endl << "Automatic differentiation:"
              << std::string(7, ' ') << dual_deriv.x << "    " << dual_deriv.y << "     " << dual_time.total_time()
              << std::endl;
    return 0;
}