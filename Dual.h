#ifndef DUAL_DUAL_H
#define DUAL_DUAL_H

#include <cmath>
#include "Dual.h"

namespace dual {
    template<typename F>
    struct Dual {
        F x;
        F y;

        Dual();

        Dual(F x_, F y_);
    };

    template<typename F>
    Dual<F> operator+(const Dual<F> &a, const Dual<F> &b);

    template<typename F>
    Dual<F> operator-(const Dual<F> &a, const Dual<F> &b);

    template<typename F>
    Dual<F> operator*(const Dual<F> &a, const Dual<F> &b);

    template<typename F>
    Dual<F> operator/(const Dual<F> &a, const Dual<F> &b);

    template<typename F>
    Dual<F> operator^(const Dual<F> &a, const Dual<F> &b);

    template<typename F>
    Dual<F> log(const Dual<F> &a);

    template<typename F>
    Dual<F> sin(const Dual<F> &a);

    template<typename F>
    Dual<F> cos(const Dual<F> &a);

    template<typename F>
    Dual<F> exp(const Dual<F> &a);

    template<typename F>
    Dual<F> asin(const Dual<F> &a);

    template<typename F>
    Dual<F> acos(const Dual<F> &a);

    template<typename F>
    Dual<F> sinh(const Dual<F> &a);

    template<typename F>
    Dual<F> cosh(const Dual<F> &a);

    template<typename F>
    Dual<F> atan(const Dual<F> &a);

    template<typename F>
    Dual<F> tan(const Dual<F> &a);

    template<typename F>
    Dual<F> asinh(const Dual<F> &a);

    template<typename F>
    Dual<F> acosh(const Dual<F> &a);

    template<typename F>
    Dual<F> atanh(const Dual<F> &a);

    template<typename F>
    Dual<F>::Dual() = default;

    template<typename F>
    Dual<F>::Dual(F x_, F y_) {
        x = x_;
        y = y_;
    }

    template<typename F>
    Dual<F> operator+(const dual::Dual<F> &a, const dual::Dual<F> &b) {
        dual::Dual<F> result;
        result.x = a.x + b.x;
        result.y = a.y + b.y;
        return result;
    }

    template<typename F>
    Dual<F> operator-(const dual::Dual<F> &a, const dual::Dual<F> &b) {
        dual::Dual<F> result;
        result.x = a.x - b.x;
        result.y = a.y - b.y;
        return result;
    }

    template<typename F>
    Dual<F> operator*(const dual::Dual<F> &a, const dual::Dual<F> &b) {
        dual::Dual<F> result;
        result.x = a.x * b.x;
        result.y = a.x * b.y + b.x * a.y;
        return result;
    }

    template<typename F>
    Dual<F> operator/(const dual::Dual<F> &a, const dual::Dual<F> &b) {
        dual::Dual<F> result;
        result.x = a.x / b.x;
        result.y = (b.x * a.y - a.x * b.y) / (b.x * b.x);
        return result;
    }

    template<typename F>
    Dual<F> operator^(const dual::Dual<F> &a, const dual::Dual<F> &b) {
        dual::Dual<F> result;
        result.x = std::pow(a.x, b.x);
        result.y = result.x * b.y * std::log(a.x) + a.y * b.x * std::pow(a.x, b.x -1);
        return result;
    }

    template<typename F>
    Dual<F> log(const dual::Dual<F> &a) {
        dual::Dual<F> result;
        result.x = std::log(a.x);
        result.y = a.y / a.x;
        return result;
    }

    template<typename F>
    Dual<F> sin(const dual::Dual<F> &a) {
        dual::Dual<F> result;
        result.x = std::sin(a.x);
        result.y = a.y * std::cos(a.x);
        return result;
    }

    template<typename F>
    Dual<F> cos(const dual::Dual<F> &a) {
        dual::Dual<F> result;
        result.x = std::cos(a.x);
        result.y = -a.y * std::sin(a.x);
        return result;
    }

    template<typename F>
    Dual<F> exp(const dual::Dual<F> &a) {
        dual::Dual<F> result;
        result.x = std::exp(a.x);
        result.y = a.y * std::exp(a.x);
        return result;
    }

    template<typename F>
    Dual<F> asin(const dual::Dual<F> &a) {
        dual::Dual<F> result;
        result.x = std::asin(a.x);
        result.y = a.y / std::sqrt(1 - a.x * a.x);
        return result;
    }

    template<typename F>
    Dual<F> acos(const dual::Dual<F> &a) {
        dual::Dual<F> result;
        result.x = std::asin(a.x);
        result.y = -a.y / std::sqrt(1 + a.x * a.x);
        return result;
    }

    template<typename F>
    Dual<F> sinh(const dual::Dual<F> &a) {
        dual::Dual<F> result;
        result.x = std::sinh(a.x);
        result.y = a.y * std::cosh(a.x);
        return result;
    }

    template<typename F>
    Dual<F> cosh(const dual::Dual<F> &a) {
        dual::Dual<F> result;
        result.x = std::cosh(a.x);
        result.y = a.y * std::sinh(a.x);
        return result;
    }

    template<typename F>
    Dual<F> atan(const dual::Dual<F> &a) {
        dual::Dual<F> result;
        result.x = std::atan(a.x);
        result.y = a.y / (1 + a.x * a.x);
        return result;
    }

    template<typename F>
    Dual<F> tan(const dual::Dual<F> &a) {
        dual::Dual<F> result;
        result.x = std::tan(a.x);
        F c = std::cos(a.x);
        result.y = a.y / (c * c);
        return result;
    }

    template<typename F>
    Dual<F> asinh(const dual::Dual<F> &a) {
        dual::Dual<F> result;
        result.x = std::asinh(a.x);
        result.y = a.y / std::sqrt(1 + a.x * a.x);
        return result;
    }

    template<typename F>
    Dual<F> acosh(const dual::Dual<F> &a) {
        dual::Dual<F> result;
        result.x = std::acosh(a.x);
        result.y = a.y / std::sqrt(a.x * a.x - 1);
        return result;
    }

    template<typename F>
    Dual<F> atanh(const dual::Dual<F> &a) {
        dual::Dual<F> result;
        result.x = std::atanh(a.x);
        result.y = a.y / (1 - a.x * a.x);
        return result;
    }
}
#endif
