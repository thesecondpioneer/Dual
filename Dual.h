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

        Dual(const Dual<F> &a);
    };

    template<typename F>
    Dual<F>::Dual() = default;

    template<typename F>
    Dual<F>::Dual(F x_, F y_) {
        x = x_;
        y = y_;
    }

    template<typename F>
    Dual<F>::Dual(const Dual<F> &a) {
        x = a.x;
        y = a.y;
    }

    template<typename F>
    Dual<F> operator+=(dual::Dual<F> a, const dual::Dual<F> &b) {
        a.x += b.x;
        a.y += b.y;
        return a;
    }

    template<typename F>
    Dual<F> operator-=(dual::Dual<F> a, const dual::Dual<F> &b) {
        a.x -= b.x;
        a.y -= b.y;
        return a;
    }

    template<typename F>
    Dual<F> operator*=(dual::Dual<F> a, const dual::Dual<F> &b) {
        a.x *= b.x;
        a.y = a.x * b.y + b.x * a.y;
        return a;
    }

    template<typename F>
    Dual<F> operator/=(dual::Dual<F> a, const dual::Dual<F> &b) {
        a.x /= b.x;
        a.y = (b.x * a.y - a.x * b.y) / (b.x * b.x);
        return a;
    }

    template<typename F>
    Dual<F> operator+(const dual::Dual<F> &a, const dual::Dual<F> &b) {
        return Dual<F>(a) += b;
    }

    template<typename F>
    Dual<F> operator-(const dual::Dual<F> &a, const dual::Dual<F> &b) {
        return Dual<F>(a) -= b;
    }

    template<typename F>
    Dual<F> operator*(const dual::Dual<F> &a, const dual::Dual<F> &b) {
        return Dual<F>(a) *= b;
    }

    template<typename F>
    Dual<F> operator/(const dual::Dual<F> &a, const dual::Dual<F> &b) {
        return Dual<F>(a) /= b;
    }

    template<typename F>
    Dual<F> operator^(const dual::Dual<F> &a, const dual::Dual<F> &b) {
        dual::Dual<F> result;
        result.x = std::pow(a.x, b.x);
        result.y = result.x * b.y * std::log(a.x) + a.y * b.x * std::pow(a.x, b.x - 1);
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
    Dual<F> log10(const dual::Dual<F> &a) {
        dual::Dual<F> result;
        result.x = std::log10(a.x);
        result.y = a.y / (std::log(10) * a.x);
        return result;
    }

    template<typename F>
    Dual<F> log2(const dual::Dual<F> &a) {
        dual::Dual<F> result;
        result.x = std::log2(a.x);
        result.y = a.y / (std::log(2) * a.x);
        return result;
    }

    template<typename F>
    Dual<F> log1p(const dual::Dual<F> &a) {
        dual::Dual<F> result;
        result.x = std::log1p(a.x);
        result.y = a.y / a.x;
        return result;
    }

    template<typename F>
    Dual<F> pow(const dual::Dual<F> &a, const dual::Dual<F> &b) {
        return a ^ b;
    }

    template<typename F>
    Dual<F> sqrt(const dual::Dual<F> &a) {
        return a ^ 0.5;
    }

    template<typename F>
    Dual<F> cbrt(const dual::Dual<F> &a) {
        return a ^ (1.0 / 3);
    }

    template<typename F>
    Dual<F> hypot(const dual::Dual<F> &a, const dual::Dual<F> &b) {
        return sqrt(a ^ 2 + b ^ 2);
    }

    template<typename F>
    Dual<F> exp2(const dual::Dual<F> &a) {
        dual::Dual<F> result;
        result.x = std::exp2(a.x);
        result.y = a.y * (std::log(2) * pow(2, a.x));
        return result;
    }

    template<typename F>
    Dual<F> expm1(const dual::Dual<F> &a) {
        dual::Dual<F> result;
        result.x = std::expm1(a.x);
        result.y = a.y * std::exp(a.x);
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
    Dual<F> tanh(const dual::Dual<F> &a) {
        dual::Dual<F> result;
        double ch = std::cosh(a.x);
        result.x = std::tanh(a.x);
        result.y = a.y / (ch * ch);
        return result;
    }

    template<typename F>
    Dual<F> atanh(const dual::Dual<F> &a) {
        dual::Dual<F> result;
        result.x = std::atanh(a.x);
        result.y = a.y / (1 - a.x * a.x);
        return result;
    }

    template<typename F>
    Dual<F> atan2(const dual::Dual<F> &a, const dual::Dual<F> &b) {
        return atan(a / b);
    }
}
#endif
