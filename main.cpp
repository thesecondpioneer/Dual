#include <iostream>
#include "Dual.h"

using namespace dual;

float f(float x) {
    return std::pow(std::cos(x), std::sin(x));
}

float y(float x) {
    return -pow(std::cos(x), std::sin(x) - 1) * std::pow(std::sin(x), 2) + std::pow(std::cos(x), std::sin(x) + 1) * std::log(std::cos(x));
}

void opt(float x, float &f, float &y) {
    float sx = std::sin(x), cx = std::cos(x), ctsx = std::pow(cx, sx);
    f = ctsx;
    y = -(ctsx / cx) * std::pow(sx, 2) + ctsx * cx * std::log(cx);
}

Dual<float> autod(Dual<float> x){
    return pow(cos(x), sin(x));
}

int main() {
    float func, deriv;
    std::cout.precision(16);
    Dual<float> der = autod(Dual<float>(1.5f, 1.0f));
    opt(1.5f, func, deriv);
    std::cout << std::fixed << f(1.5) << ' ' << y(1.5) << std::endl << func << ' ' << deriv << std::endl << der.x << ' ' << der.y << std::endl;
    return 0;
}