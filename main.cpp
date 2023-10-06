#include <iostream>
#include "Dual.h"

using namespace dual;

int main() {
    dual::Dual<float> a(3.0, 1.0), b(7.0, 1.0);
    std::cout << (1.0f + a).x << std::endl;
    std::cout << (a + b).x << ' ' << (a + b).y << std::endl;
    std::cout << log(a).x << ' ' << log(a).y << std::endl;
    std::cout << (a ^ dual::Dual<float>(4.0, 0)).x << ' ' << (a ^ dual::Dual<float>(4.0, 0)).y << std::endl;
    return 0;
}