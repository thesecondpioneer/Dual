#ifndef DUAL_DUAL_H
#define DUAL_DUAL_H

#include <cmath>

namespace dual {
    template<typename F>
    struct Dual {
        F x;
        F y;

        Dual();

        Dual(const F &x_, const F &y_);

        explicit Dual(const F &x_);

        Dual(const Dual<F> &a);

        //compound operators
        Dual<F> operator+=(const dual::Dual<F> &b) {
            x += b.x;
            y += b.y;
            return *this;
        };

        Dual<F> operator-=(const dual::Dual<F> &b) {
            x -= b.x;
            y -= b.y;
            return *this;
        };

        Dual<F> operator*=(const dual::Dual<F> &b) {
            x *= b.x;
            y = x * b.y + b.x * y;
            return *this;
        }

        Dual<F> operator/=(const dual::Dual<F> &b) {
            x /= b.x;
            y = (b.x * y - x * b.y) / (b.x * b.x);
            return *this;
        }

        // compound operators with a scalar
        Dual<F> operator+=(const F &b) {
            x = x + b;
            return *this;
        }

        Dual<F> operator-=(const F &b) {
            x = x - b;
            return *this;
        }

        Dual<F> operator*=(const F &b) {
            x = x * b;
            return *this;
        }

        Dual<F> operator/=(const F &b) {
            x = x / b;
            return *this;
        }
    };

    template<typename F>
    Dual<F>::Dual(const F &x_) {
        x = x_;
        y = F(1.0);
    }

    template<typename F>
    Dual<F>::Dual() = default;

    template<typename F>
    Dual<F>::Dual(const F &x_, const F &y_) {
        x = x_;
        y = y_;
    }

    template<typename F>
    Dual<F>::Dual(const Dual<F> &a) {
        x = a.x;
        y = a.y;
    }

    //unary -
    template<typename F>
    inline __attribute__((always_inline)) Dual<F> operator+(const dual::Dual<F> &a) {
        return a;
    }

    //unary +
    template<typename F>
    inline __attribute__((always_inline)) Dual<F> operator-(const dual::Dual<F> &a) {
        return Dual<F>(-a.x, -a.y);
    }

    //binary +
    template<typename F>
    inline __attribute__((always_inline)) Dual<F> operator+(const dual::Dual<F> &a, const dual::Dual<F> &b) {
        return Dual<F>(a) += b;
    }

    //binary + with a scalar
    template<typename F>
    inline __attribute__((always_inline)) Dual<F> operator+(const dual::Dual<F> &a, const F &b) {
        return Dual<F>(a) += b;
    }

    //binary -
    template<typename F>
    inline __attribute__((always_inline)) Dual<F> operator-(const dual::Dual<F> &a, const dual::Dual<F> &b) {
        return Dual<F>(a) -= b;
    }

    //binary - with a scalar
    template<typename F>
    inline __attribute__((always_inline)) Dual<F> operator-(const dual::Dual<F> &a, const F &b) {
        return Dual<F>(a) -= b;
    }


    //binary *
    template<typename F>
    inline __attribute__((always_inline)) Dual<F> operator*(const dual::Dual<F> &a, const dual::Dual<F> &b) {
        return Dual<F>(a) *= b;
    }

    //binary * with a scalar
    template<typename F>
    inline __attribute__((always_inline)) Dual<F> operator*(const dual::Dual<F> &a, F &b) {
        return Dual<F>(a) *= b;
    }

    //binary /
    template<typename F>
    inline __attribute__((always_inline)) Dual<F> operator/(const dual::Dual<F> &a, const dual::Dual<F> &b) {
        return Dual<F>(a) /= b;
    }

    //binary / with a scalar
    template<typename F>
    inline __attribute__((always_inline)) Dual<F> operator/(const dual::Dual<F> &a, const F &b) {
        return Dual<F>(a) /= b;
    }

    //power operator
    template<typename F>
    inline __attribute__((always_inline)) Dual<F> operator^(const dual::Dual<F> &a, const dual::Dual<F> &b) {
        const F a_to_b = std::pow(a.x, b.x);
        return Dual<F>(a_to_b, a_to_b * b.y * std::log(a.x) + a.y * b.x * std::pow(a.x, b.x - 1));
    }

    //power operator with dual scalar exponent
    template<typename F>
    inline __attribute__((always_inline)) Dual<F> operator^(const dual::Dual<F> &a, const F &b) {
        const F a_to_b = std::pow(a.x, b);
        return Dual<F>(a_to_b, a.y * b * std::pow(a.x, b - 1));
    }

    //power operator with scalar base
    template<typename F>
    inline __attribute__((always_inline)) Dual<F> operator^(const F &a, const dual::Dual<F> &b) {
        const F a_to_b = std::pow(a, b.x);
        return Dual<F>(a_to_b, a_to_b * b.y * std::log(a.x));
    }

    template<typename F>
    inline __attribute__((always_inline)) Dual<F> abs(const dual::Dual<F> &a) {
        return Dual<F>(abs(a.x), a.y * copysign(F(1.0), a.x));
    }

    template<typename F>
    inline __attribute__((always_inline)) Dual<F> log(const dual::Dual<F> &a) {
        return Dual<F>(std::log(a.x), a.y / a.x);
    }

    template<typename F>
    inline __attribute__((always_inline)) Dual<F> log10(const dual::Dual<F> &a) {
        return Dual<F>(std::log10(a.x), a.y / (std::log(F(10.0)) * a.x));
    }

    template<typename F>
    inline __attribute__((always_inline)) Dual<F> log2(const dual::Dual<F> &a) {
        return Dual<F>(std::log2(a.x), a.y / (std::log(F(2.0)) * a.x));
    }

    //natural base log of a+1
    template<typename F>
    inline __attribute__((always_inline)) Dual<F> log1p(const dual::Dual<F> &a) {
        return Dual<F>(std::log1p(a.x), a.y / a.x);
    }

    // power function
    template<typename F>
    inline __attribute__((always_inline)) Dual<F> pow(const dual::Dual<F> &a, const dual::Dual<F> &b) {
        return a ^ b;
    }

    //power function with scalar exponent
    template<typename F>
    inline __attribute__((always_inline)) Dual<F> pow(const dual::Dual<F> &a, const F &b) {
        return a ^ b;
    }

    //power function with scalar base
    template<typename F>
    inline __attribute__((always_inline)) Dual<F> pow(const F &a, const dual::Dual<F> &b) {
        return a ^ b;
    }

    //square root
    template<typename F>
    inline __attribute__((always_inline)) Dual<F> sqrt(const dual::Dual<F> &a) {
        const F sqrtx = std::sqrt(a.x);
        return Dual<F>(sqrtx, a.y / F(2.0) * sqrtx);
    }

    //cubic root
    template<typename F>
    inline __attribute__((always_inline)) Dual<F> cbrt(const dual::Dual<F> &a) {
        const F cbrtx = std::cbrt(a.x);
        return Dual<F>(cbrtx, a.y / F(3.0) * cbrtx * cbrtx);
    }

    template<typename F>
    inline __attribute__((always_inline)) Dual<F> hypot(const dual::Dual<F> &a, const dual::Dual<F> &b) {
        const F hpab = std::hypot(a.x, b.y);
        return Dual<F>(hpab, (a.x * a.y + b.x * b.y) / hpab);
    }

    template<typename F>
    inline __attribute__((always_inline)) Dual<F> exp2(const dual::Dual<F> &a) {
        const F exp2x = std::exp2(a.x);
        return Dual<F>(exp2x, a.y * exp2x * std::log(F(2.0)));
    }

    template<typename F>
    inline __attribute__((always_inline)) Dual<F> expm1(const dual::Dual<F> &a) {
        const F expm1x = std::expm1(a.x);
        return Dual<F>(expm1x, a.y * (expm1x + F(1.0)));
    }

    template<typename F>
    inline __attribute__((always_inline)) Dual<F> sin(const dual::Dual<F> &a) {
        return Dual<F>(std::sin(a.x), a.y * std::cos(a.x));
    }

    template<typename F>
    inline __attribute__((always_inline)) Dual<F> cos(const dual::Dual<F> &a) {
        return Dual<F>(std::cos(a.x), -a.y * std::sin(a.x));
    }

    template<typename F>
    inline __attribute__((always_inline)) Dual<F> exp(const dual::Dual<F> &a) {
        const F expx = std::exp(a.x);
        return Dual<F>(expx, a.y * expx);
    }

    template<typename F>
    inline __attribute__((always_inline)) Dual<F> asin(const dual::Dual<F> &a) {
        return Dual<F>(std::asin(a.x), a.y / std::sqrt(F(1.0) - a.x * a.x));
    }

    template<typename F>
    inline __attribute__((always_inline)) Dual<F> acos(const dual::Dual<F> &a) {
        return Dual<F>(std::acos(a.x), -a.y / std::sqrt(F(1.0) - a.x * a.x));
    }

    template<typename F>
    inline __attribute__((always_inline)) Dual<F> sinh(const dual::Dual<F> &a) {
        return Dual<F>(std::sinh(a.x), a.y * std::cosh(a.x));
    }

    template<typename F>
    inline __attribute__((always_inline)) Dual<F> cosh(const dual::Dual<F> &a) {
        return Dual<F>(std::cosh(a.x), a.y * std::sinh(a.x));
    }

    template<typename F>
    inline __attribute__((always_inline)) Dual<F> atan(const dual::Dual<F> &a) {
        return Dual<F>(std::atan(a.x), a.y / (1 + a.x * a.x));
    }

    template<typename F>
    inline __attribute__((always_inline)) Dual<F> tan(const dual::Dual<F> &a) {
        const F tanx = std::tan(a.x);
        return Dual<F>(tanx, a.y * (F(1.0) + tanx * tanx));
    }

    template<typename F>
    inline __attribute__((always_inline)) Dual<F> asinh(const dual::Dual<F> &a) {
        return Dual<F>(std::asinh(a.x), a.y / std::sqrt(a.x * a.x + F(1.0)));;
    }

    template<typename F>
    inline __attribute__((always_inline)) Dual<F> acosh(const dual::Dual<F> &a) {
        return Dual<F>(std::acosh(a.x), a.y / std::sqrt(a.x * a.x - F(1.0)));
    }

    template<typename F>
    inline __attribute__((always_inline)) Dual<F> tanh(const dual::Dual<F> &a) {
        dual::Dual<F> result;
        const F tanhx = std::tanh(a.x);
        return Dual<F>(tanhx, a.y * (F(1.0) - tanhx * tanhx));
    }

    template<typename F>
    inline __attribute__((always_inline)) Dual<F> atanh(const dual::Dual<F> &a) {
        return Dual<F>(std::atanh(a.x), a.y / (F(1.0) - a.x * a.x));
    }

    template<typename F>
    inline __attribute__((always_inline)) Dual<F> atan2(const dual::Dual<F> &a, const dual::Dual<F> &b) {
        return Dual<F>(std::atan2(a.x, b.y), (-b.x * a.y + a.x * b.y) / (a.x * a.x + b.x * b.x));
    }
}
#endif
