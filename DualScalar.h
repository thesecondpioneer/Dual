#ifndef DUAL_DUALSCALAR_H
#define DUAL_DUALSCALAR_H
#include <cmath>
namespace dual {
    template <typename F>
    struct DualScalar {
        static_assert(std::is_arithmetic_v<F>, "k");
        F x;
        F y;

        DualScalar() : x(), y(){};

        DualScalar(F x_, F y_) {
            x = x_;
            y = y_;
        }

        DualScalar(F x_) {
            x = x_;
            y = F(0);
        }

        DualScalar(const DualScalar<F> &a) {
            x = a.x;
            y = a.y;
        }

        // compound operators
        inline DualScalar<F> operator+=(const DualScalar<F> &b) {
            x += b.x;
            y += b.y;
            return *this;
        };

        inline DualScalar<F> operator-=(const DualScalar<F> &b) {
            x -= b.x;
            y -= b.y;
            return *this;
        };

        inline DualScalar<F> operator*=(const DualScalar<F> &b) {
            y = x * b.y + b.x * y;
            x *= b.x;
            return *this;
        }

        inline DualScalar<F> operator/=(const DualScalar<F> &b) {
            const F inv = F(1.0) / b.x;
            x *= inv;
            // x at this point is already divided by b.x which turns (b.x * y - x * b.y) / (b.x * b.x)
            // into (y - x * b.y) / b.x
            y = (y - x * b.y) * inv;
            return *this;
        }

        // scalar assignation
        inline DualScalar<F> &operator=(F b) {
            x = b;
            y = F(0);
            return *this;
        }

        // compound operators with a scalar
        inline DualScalar<F> operator+=(F b) {
            x += b;
            return *this;
        }

        inline DualScalar<F> operator-=(F b) {
            x = x - b;
            return *this;
        }

        inline DualScalar<F> operator*=(F b) {
            x *= b;
            y *= b;
            return *this;
        }

        inline DualScalar<F> operator/=(F b) {
            x /= b;
            y /= b;
            return *this;
        }
    };

    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline bool isless(const DualScalar<F> &a, const DualScalar<F> &b) {
        return std::isless(a.x, b.x);
    }

    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline bool isgreater(const DualScalar<F> &a, const DualScalar<F> &b) {
        return std::isgreater(a.x, b.x);
    }

    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline bool islessequal(const DualScalar<F> &a, const DualScalar<F> &b) {
        return std::islessequal(a.x, b.x);
    }

    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline bool islessgreater(const DualScalar<F> &a, const DualScalar<F> &b) {
        return std::islessgreater(a.x, b.x);
    }

    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline bool isgreaterequal(const DualScalar<F> &a,
                                                              const DualScalar<F> &b) {
        return std::isgreaterequal(a.x, b.x);
    }

    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline bool isunordered(const DualScalar<F> &a, const DualScalar<F> &b) {
        return std::isunordered(a.x, b.x);
    }

    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline bool operator<(const DualScalar<F> &a, const DualScalar<F> &b) {
        return isless(a, b);
    }

    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline bool operator>(const DualScalar<F> &a, const DualScalar<F> &b) {
        return isgreater(a, b);
    }

    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline bool operator<=(const DualScalar<F> &a, const DualScalar<F> &b) {
        return islessequal(a, b);
    }

    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline bool operator>=(const DualScalar<F> &a, const DualScalar<F> &b) {
        return isgreaterequal(a, b);
    }

    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline bool operator==(const DualScalar<F> &a, const DualScalar<F> &b) {
        return (a.x == b.x && a.y == b.y);
    }

    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline bool operator!=(const DualScalar<F> &a, const DualScalar<F> &b) {
        return !(a.x == b.x && a.y == b.y);
    }

// unary +
    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualScalar<F> operator+(const DualScalar<F> &a) {
        return a;
    }

// unary -
    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualScalar<F> operator-(const DualScalar<F> &a) {
        return DualScalar<F>(-a.x, -a.y);
    }

// binary +
    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualScalar<F> operator+(const DualScalar<F> &a,
                                                           const DualScalar<F> &b) {
        return DualScalar<F>(a.x + b.x, a.y + b.y);
    }

// binary + with a scalar
    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualScalar<F> operator+(const DualScalar<F> &a, F b) {
        return DualScalar<F>(a.x + b, a.y);
    }

    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualScalar<F> operator+(F a, const DualScalar<F> &b) {
        return DualScalar<F>(a + b.x, b.y);
    }

// binary -
    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualScalar<F> operator-(const DualScalar<F> &a,
                                                               const DualScalar<F> &b) {
        return DualScalar<F>(a.x - b.x, a.y - b.y);
    }

// binary - with a scalar
    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualScalar<F> operator-(const DualScalar<F> &a, F b) {
        return DualScalar<F>(a.x - b, a.y);
    }

    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualScalar<F> operator-(F a, const DualScalar<F> &b) {
        return DualScalar<F>(a - b.x, -b.y);
    }

// binary *
    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualScalar<F> operator*(const DualScalar<F> &a, const DualScalar<F> &b) {
        return DualScalar<F>(a.x * b.x, a.x * b.y + b.x * a.y);
    }

// binary * with a scalar
    template <typename F, typename Scalar, typename std::enable_if<std::is_arithmetic_v<Scalar>, bool>::type = true>
    inline DualScalar<F> operator*(const DualScalar<F> &a, Scalar &b) {
        return DualScalar<F>(a.x * b, b * a.y);
    }

    template <typename F, typename Scalar, typename std::enable_if<std::is_arithmetic_v<Scalar>, bool>::type = true>
    inline DualScalar<F> operator*(Scalar a, const DualScalar<F> &b) {
        return DualScalar<F>(a * b.x, b.y * a);
    }

// binary /
    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualScalar<F> operator/(const DualScalar<F> &a,
                                                               const DualScalar<F> &b) {
        const F bxinv = F(1.0) / b.x, axbybx = a.x * bxinv;
        return DualScalar<F>(axbybx, (a.y - axbybx * b.y) * bxinv);
    }

// binary / with a scalar
    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualScalar<F> operator/(const DualScalar<F> &a, F b) {
        const F binv = F(1.0) / b;
        return DualScalar<F>(a.x * binv, a.y * binv);
    }

    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualScalar<F> operator/(F a, const DualScalar<F> &b) {
        const F abybx = a / b.x;
        return DualScalar<F>(abybx, b.y * (-abybx / b.x));
    }

    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualScalar<F> abs(const DualScalar<F> &a) {
        return DualScalar<F>(std::abs(a.x), a.y * copysign(F(1.0), a.x));
    }

    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualScalar<F> ceil(const dual::DualScalar<F> &a) {
        return DualScalar<F>(std::ceil(a.x));
    }

    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualScalar<F> floor(const dual::DualScalar<F> &a) {
        return DualScalar<F>(std::floor(a.x));
    }

    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualScalar<F> log(const DualScalar<F> &a) {
        const F inv = F(1.0) / a.x;
        return DualScalar<F>(std::log(a.x), a.y * inv);
    }

    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualScalar<F> log10(const DualScalar<F> &a)
// log(10) is precalced
    {
        const F inv = F(1.0) / (F(2.30258509299404568) * a.x);
        return DualScalar<F>(std::log10(a.x), a.y * inv);
    }

    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualScalar<F> log2(const DualScalar<F> &a)
// log(2) is precalced
    {
        const F inv = F(1.0) / (F(0.6931471805599453) * a.x);
        return DualScalar<F>(std::log2(a.x), a.y * inv);
    }

// natural base log of a+1
    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualScalar<F> log1p(const DualScalar<F> &a) {
        F inv = F(1.0) / (a.x + F(1.0));
        return DualScalar<F>(std::log1p(a.x), a.y * inv);
    }

// power function
    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualScalar<F> pow(const DualScalar<F> &a, const DualScalar<F> &b) {
        if (a.x == 0 && b.x >= 1) {
            if (b.x > 1) {
                return DualScalar<F>(F());
            }
            return a;
        }
        const F a_to_b = std::pow(a.x, b.x), bx_by_a_to_bm1 = b.x * a_to_b / a.x;
        if (a.x < 0 && b.x == std::floor(b.x)) {
            DualScalar<F> result(a_to_b, bx_by_a_to_bm1 * a.y);
            if (std::fpclassify(b.y) != FP_ZERO) {
                result.y = F(std::numeric_limits<F>::quiet_NaN());
            }
            return result;
        }
        return DualScalar<F>(a_to_b, bx_by_a_to_bm1 * a.y + a_to_b * std::log(a.x) * b.y);
    }

// power function with scalar exponent
    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualScalar<F> pow(const DualScalar<F> &a, F b) {
        if (a.x == 0 && b >= 1) {
            if (b > 1) {
                return DualScalar<F>(F());
            }
            return a;
        }
        return DualScalar<F>(std::pow(a.x, b), b * std::pow(a.x, b - F(1.0)) * a.y);
    }

// power function with scalar base
    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualScalar<F> pow(F a, const DualScalar<F> &b) {
        if (a == 0 && b.x >= 1) {
            if (b.x > 1) {
                return DualScalar<F>(F());
            }
            return DualScalar<F>(a);
        }
        if (a < 0 && b.x == std::floor(b.x)) {
            DualScalar<F> result(std::pow(a, b.x));
            if (std::fpclassify(b.y) != FP_ZERO) {
                result.y = std::numeric_limits<F>::quiet_NaN();
            }
            return result;
        }
        const F a_to_b = std::pow(a, b.x);
        return DualScalar<F>(a_to_b, a_to_b * std::log(a.x) * b.y);
    }

// square root
    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualScalar<F> sqrt(const DualScalar<F> &a) {
        const F sqrtx = std::sqrt(a.x), inv = F(1.0) / (F(2.0) * sqrtx);
        return DualScalar<F>(sqrtx, a.y * inv);
    }

// cubic root
    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualScalar<F> cbrt(const DualScalar<F> &a) {
        const F cbrtx = std::cbrt(a.x), inv = F(1.0) / (F(3.0) * std::cbrt(a.x * a.x));
        return DualScalar<F>(cbrtx, a.y * inv);
    }

    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualScalar<F> hypot(const DualScalar<F> &a, const DualScalar<F> &b) {
        const F hpab = std::hypot(a.x, b.x);
        return DualScalar<F>(hpab, (a.x / hpab * a.y + b.x / hpab * b.y));
    }

    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualScalar<F> hypot(F a, const DualScalar<F> &b) {
        const F hpab = std::hypot(a, b.x);
        return DualScalar<F>(hpab, b.x / hpab * b.y);
    }

    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualScalar<F> hypot(const DualScalar<F> &a, F b) {
        return hypot(b, a);
    }

    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualScalar<F> hypot(const DualScalar<F> &a, const DualScalar<F> &b, const DualScalar<F> &c) {
        const F hpabc = std::hypot(a.x, b.x, c.x);
        return DualScalar<F>(hpabc, a.x / hpabc * a.y + b.x / hpabc * b.y + c.x / hpabc * c.y);
    }

    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualScalar<F> hypot(F a, const DualScalar<F> &b, const DualScalar<F> &c) {
        const F hpabc = std::hypot(a, b.x, c.x);
        return DualScalar<F>(hpabc, b.x / hpabc * b.y + c.x / hpabc * c.y);
    }

    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualScalar<F> hypot(F a, F b, const DualScalar<F> &c) {
        const F hpabc = std::hypot(a, b, c.x);
        return DualScalar<F>(hpabc, c.x / hpabc * c.y);
    }

    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualScalar<F> exp2(const DualScalar<F> &a) {
        const F exp2x = std::exp2(a.x);
        return DualScalar<F>(exp2x, exp2x * F(0.6931471805599453) * a.y);  // precalced log(2)
    }

    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualScalar<F> expm1(const DualScalar<F> &a) {
        const F expm1x = std::expm1(a.x);
        return DualScalar<F>(expm1x, a.y * (expm1x + F(1.0)));
    }

    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualScalar<F> sin(const DualScalar<F> &a) {
        return DualScalar<F>(std::sin(a.x), a.y * std::cos(a.x));
    }

    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualScalar<F> cos(const DualScalar<F> &a) {
        return DualScalar<F>(std::cos(a.x), -a.y * std::sin(a.x));
    }

    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualScalar<F> exp(const DualScalar<F> &a) {
        const F expx = std::exp(a.x);
        return DualScalar<F>(expx, expx * a.y);
    }

    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualScalar<F> asin(const DualScalar<F> &a) {
        const F inv = F(1.0) / std::sqrt(F(1.0) - a.x * a.x);
        return DualScalar<F>(std::asin(a.x), inv * a.y);
    }

    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualScalar<F> acos(const DualScalar<F> &a) {
        const F inv = F(-1.0) / std::sqrt(F(1.0) - a.x * a.x);
        return DualScalar<F>(std::acos(a.x), inv * a.y);
    }

    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualScalar<F> sinh(const DualScalar<F> &a) {
        return DualScalar<F>(std::sinh(a.x), std::cosh(a.x) * a.y);
    }

    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualScalar<F> cosh(const DualScalar<F> &a) {
        return DualScalar<F>(std::cosh(a.x), std::sinh(a.x) * a.y);
    }

    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualScalar<F> atan(const DualScalar<F> &a) {
        const F inv = F(1.0) / (F(1.0) + a.x * a.x);
        return DualScalar<F>(std::atan(a.x), inv * a.y);
    }

    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualScalar<F> tan(const DualScalar<F> &a) {
        const F tanx = std::tan(a.x), tmp = (F(1.0) + tanx * tanx);
        return DualScalar<F>(tanx, tmp * a.y);
    }

    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualScalar<F> asinh(const DualScalar<F> &a) {
        const F inv = F(1.0) / std::sqrt(a.x * a.x + F(1.0));
        return DualScalar<F>(std::asinh(a.x), inv * a.y);
    }

    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualScalar<F> acosh(const DualScalar<F> &a) {
        const F inv = F(1.0) / std::sqrt(a.x * a.x - F(1.0));
        return DualScalar<F>(std::acosh(a.x), inv * a.y);
    }

    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualScalar<F> tanh(const DualScalar<F> &a) {
        const F tanhx = std::tanh(a.x), tmp = (F(1.0) - tanhx * tanhx);
        return DualScalar<F>(tanhx, tmp * a.y);
    }

    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualScalar<F> atanh(const DualScalar<F> &a) {
        const F inv = F(1.0) / (F(1.0) - a.x * a.x);
        return DualScalar<F>(std::atanh(a.x), inv * a.y);
    }

    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualScalar<F> atan2(const DualScalar<F> &a, const DualScalar<F> &b) {
        const F inv = F(1.0) / (a.x * a.x + b.x * b.x);
        return DualScalar<F>(std::atan2(a.x, b.y), (inv * b.x) * a.y - (inv * a.x) * b.y);
    }

    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualScalar<F> atan2(F a, const DualScalar<F> &b) {
        return DualScalar<F>(std::atan2(a, b.y), -(a / (a * a + b.x * b.x)) * b.y);
    }

    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualScalar<F> atan2(const DualScalar<F> &a, F b) {
        return atan2(b, a);
    }

#ifndef __clang__  // bessel functions
    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualScalar<F> bessel_j0(const DualScalar<F> &a) {
        return DualScalar<F>(std::cyl_bessel_j(0, a.x), -std::cyl_bessel_j(1, a.x) * a.y);
    }

    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualScalar<F> bessel_j1(const DualScalar<F> &a) {
        return DualScalar<F>(std::cyl_bessel_j(1, a.x),
                          F(0.5) * (std::cyl_bessel_j(0, a.x) - std::cyl_bessel_j(2, a.x)) * a.y);
    }

    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualScalar<F> bessel_jn(int n, const DualScalar<F> &a) {
        return DualScalar<F>(
                std::cyl_bessel_j(n, a.x),
                F(0.5) * (std::cyl_bessel_j(n - 1, a.x) - std::cyl_bessel_j(n + 1, a.x)) * a.y);
    }

    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualScalar<F> bessel_i0(const DualScalar<F> &a) {
        return DualScalar<F>(std::cyl_bessel_i(0, a.x), std::cyl_bessel_i(1, a.x) * a.y);
    }

    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualScalar<F> bessel_i1(const DualScalar<F> &a) {
        return DualScalar<F>(std::cyl_bessel_i(1, a.x),
                          F(0.5) * (std::cyl_bessel_i(0, a.x) + std::cyl_bessel_i(2, a.x)) * a.y);
    }

    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualScalar<F> bessel_in(int n, const DualScalar<F> &a) {
        return DualScalar<F>(
                std::cyl_bessel_i(n, a.x),
                F(0.5) * (std::cyl_bessel_i(n - 1, a.x) + std::cyl_bessel_i(n + 1, a.x)) * a.y);
    }

    template <typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualScalar<F> bessel_sph_jn(int n, const DualScalar<F> &a) {
        const F tmp = std::cyl_bessel_i(n, a.x);
        return DualScalar<F>(tmp, (-std::sph_bessel(n + 1, a.x) + F(n / a.x) * tmp) * a.y);
    }
#endif  // bessel functions
}  // namespace dual
namespace std {
    template <typename F>
    struct is_arithmetic<dual::DualScalar<F>> : true_type {};
}
#endif
