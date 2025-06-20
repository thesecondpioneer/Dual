#ifndef DUAL_DUAL_H
#define DUAL_DUAL_H

#include <cmath>
#include "DualScalar.h"

#include <eigen3/Eigen/Core>


namespace dual {
    template <typename F, int N>
    struct Dual {
        static_assert(std::is_arithmetic_v<F>, "k");
        F x;
        Eigen::Vector<F, N> y;

        Dual() : x() { y.setZero(); }

        template <typename der>
        EIGEN_STRONG_INLINE Dual(const F &x_, const Eigen::DenseBase<der> &y_) {
            x = x_;
            y = y_;
        }

        Dual(const F &x_, int k) {
            x = x_;
            y.setZero();
            y[k] = F(1.0);
        }

        explicit Dual(const F &x_) {
            x = x_;
            y.setZero();
        }

        Dual(const Dual<F, N> &a) {
            x = a.x;
            y = a.y;
        }

        // compound operators
        inline Dual<F, N> operator+=(const Dual<F, N> &b) {
            x += b.x;
            y += b.y;
            return *this;
        };

        inline Dual<F, N> operator-=(const Dual<F, N> &b) {
            x -= b.x;
            y -= b.y;
            return *this;
        };

        inline Dual<F, N> operator*=(const Dual<F, N> &b) {
            y = x * b.y + b.x * y;
            x *= b.x;
            return *this;
        }

        inline Dual<F, N> operator/=(const Dual<F, N> &b) {
            const F inv = F(1.0) / b.x;
            x *= inv;
            // x at this point is already divided by b.x which turns (b.x * y - x * b.y) / (b.x * b.x)
            // into (y - x * b.y) / b.x
            y = (y - x * b.y) * inv;
            return *this;
        }

        // scalar assignation
        template<typename Scalar, typename std::enable_if<std::is_arithmetic_v<Scalar>, bool>::type = true>
        inline Dual<F, N> &operator=(const Scalar &b) {
            x = b;
            y.setZero();
            return *this;
        }

        // compound operators with a scalar
        template<typename Scalar, typename std::enable_if<std::is_arithmetic_v<Scalar>, bool>::type = true>
        inline Dual<F, N> operator+=(const Scalar &b) {
            x += b;
            return *this;
        }

        template<typename Scalar, typename std::enable_if<std::is_arithmetic_v<Scalar>, bool>::type = true>
        inline Dual<F, N> operator-=(const Scalar &b) {
            x = x - b;
            return *this;
        }

        template<typename Scalar, typename std::enable_if<std::is_arithmetic_v<Scalar>, bool>::type = true>
        inline Dual<F, N> operator*=(const Scalar &b) {
            x *= b;
            y *= b;
            return *this;
        }

        template<typename Scalar, typename std::enable_if<std::is_arithmetic_v<Scalar>, bool>::type = true>
        inline Dual<F, N> operator/=(const Scalar &b) {
            x /= b;
            y /= b;
            return *this;
        }
    };
    using std::abs;
    using std::acos;
    using std::asin;
    using std::atan;
    using std::atan2;
    using std::cbrt;
    using std::ceil;
    using std::copysign;
    using std::cos;
    using std::cosh;
#ifndef __clang__
    using std::cyl_bessel_j;
    using std::cyl_bessel_i;
    using std::sph_bessel;
#endif
    using std::erf;
    using std::erfc;
    using std::exp;
    using std::exp2;
    using std::expm1;
    using std::fdim;
    using std::floor;
    using std::fma;
    using std::fmax;
    using std::fmin;
    using std::fpclassify;
    using std::hypot;
    using std::isfinite;
    using std::isinf;
    using std::isnan;
    using std::isnormal;
    using std::log;
    using std::log10;
    using std::log1p;
    using std::log2;
    using std::norm;
    using std::pow;
    using std::signbit;
    using std::sin;
    using std::sinh;
    using std::sqrt;
    using std::tan;
    using std::tanh;

    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline bool isless(const Dual<F, N> &a, const Dual<F, N> &b) {
        return isless(a.x, b.x);
    }

    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline bool isgreater(const Dual<F, N> &a, const Dual<F, N> &b) {
        return isgreater(a.x, b.x);
    }

    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline bool islessequal(const Dual<F, N> &a, const Dual<F, N> &b) {
        return islessequal(a.x, b.x);
    }

    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline bool islessgreater(const Dual<F, N> &a, const Dual<F, N> &b) {
        return islessgreater(a.x, b.x);
    }

    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline bool isgreaterequal(const Dual<F, N> &a,
                               const Dual<F, N> &b) {
        return isgreaterequal(a.x, b.x);
    }

    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline bool isunordered(const Dual<F, N> &a, const Dual<F, N> &b) {
        return isunordered(a.x, b.x);
    }

    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline bool fpclassify(const Dual<F, N> &a) {
        return fpclassify(a.x);
    }

    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline bool operator<(const Dual<F, N> &a, const Dual<F, N> &b) {
        return isless(a, b);
    }

    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline bool operator>(const Dual<F, N> &a, const Dual<F, N> &b) {
        return isgreater(a, b);
    }

    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline bool operator<=(const Dual<F, N> &a, const Dual<F, N> &b) {
        return islessequal(a, b);
    }

    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline bool operator>=(const Dual<F, N> &a, const Dual<F, N> &b) {
        return isgreaterequal(a, b);
    }

    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline bool operator==(const Dual<F, N> &a, const Dual<F, N> &b) {
        return (a.x == b.x && a.y == b.y);
    }

    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline bool operator!=(const Dual<F, N> &a, const Dual<F, N> &b) {
        return !(a.x == b.x && a.y == b.y);
    }

// unary +
    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline Dual<F, N> operator+(const Dual<F, N> &a) {
        return a;
    }

// unary -
    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline Dual<F, N> operator-(const Dual<F, N> &a) {
        return Dual<F, N>(-a.x, -a.y);
    }

// binary +
    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline Dual<F, N> operator+(const Dual<F, N> &a,
                                const Dual<F, N> &b) {
        return Dual<F, N>(a.x + b.x, a.y + b.y);
    }

// binary + with a scalar
    template <typename F, typename Scalar, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true,
            typename std::enable_if<std::is_arithmetic_v<Scalar>, bool>::type = true>
    inline Dual<F, N> operator+(const Dual<F, N> &a, const Scalar &b) {
        return Dual<F, N>(a.x + b, a.y);
    }

    template <typename F, typename Scalar, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true,
            typename std::enable_if<std::is_arithmetic_v<Scalar>, bool>::type = true>
    inline Dual<F, N> operator+(const Scalar &a, const Dual<F, N> &b) {
        return Dual<F, N>(a + b.x, b.y);
    }

// binary -
    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline Dual<F, N> operator-(const Dual<F, N> &a, const Dual<F, N> &b) {
        return Dual<F, N>(a.x - b.x, a.y - b.y);
    }

// binary - with a scalar
    template <typename F, typename Scalar, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true,
            typename std::enable_if<std::is_arithmetic_v<Scalar>, bool>::type = true>
    inline Dual<F, N> operator-(const Dual<F, N> &a, const Scalar &b) {
        return Dual<F, N>(a.x - b, a.y);
    }

    template <typename F, typename Scalar, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true,
            typename std::enable_if<std::is_arithmetic_v<Scalar>, bool>::type = true>
    inline Dual<F, N> operator-(const Scalar &a, const Dual<F, N> &b) {
        return Dual<F, N>(a - b.x, -b.y);
    }

// binary *
    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline Dual<F, N> operator*(const Dual<F, N> &a, const Dual<F, N> &b) {
        return Dual<F, N>(a.x * b.x, a.x * b.y + b.x * a.y);
    }

// binary * with a scalar
    template <typename F, typename Scalar, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true,
            typename std::enable_if<std::is_arithmetic_v<Scalar>, bool>::type = true>
    inline Dual<F, N> operator*(const Dual<F, N> &a, const Scalar &b) {
        return Dual<F, N>(a.x * b, b * a.y);
    }

    template <typename F, typename Scalar, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true,
            typename std::enable_if<std::is_arithmetic_v<Scalar>, bool>::type = true>
    inline Dual<F, N> operator*(const Scalar &a, const Dual<F, N> &b) {
        return Dual<F, N>(a * b.x, b.y * a);
    }

// binary /
    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline Dual<F, N> operator/(const Dual<F, N> &a, const Dual<F, N> &b) {
        const F bxinv = F(1.0) / b.x, axbybx = a.x * bxinv;
        return Dual<F, N>(axbybx, (a.y - axbybx * b.y) * bxinv);
    }

// binary / with a scalar
    template <typename F, typename Scalar, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true,
            typename std::enable_if<std::is_arithmetic_v<Scalar>, bool>::type = true>
    inline Dual<F, N> operator/(const Dual<F, N> &a, const Scalar &b) {
        const F binv = F(1.0) / b;
        return Dual<F, N>(a.x * binv, a.y * binv);
    }

    template <typename F, typename Scalar, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true,
            typename std::enable_if<std::is_arithmetic_v<Scalar>, bool>::type = true>
    inline Dual<F, N> operator/(const Scalar &a, const Dual<F, N> &b) {
        const F abybx = a / b.x;
        return Dual<F, N>(abybx, b.y * (-abybx / b.x));
    }

    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline Dual<F, N> abs(const Dual<F, N> &a) {
        return Dual<F, N>(abs(a.x), a.y * copysign(F(1.0), a.x));
    }

    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline Dual<F, N> ceil(const dual::Dual<F, N> &a) {
        return Dual<F, N>(ceil(a.x));
    }

    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline Dual<F, N> floor(const dual::Dual<F, N> &a) {
        return Dual<F, N>(floor(a.x));
    }

    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline Dual<F, N> log(const Dual<F, N> &a) {
        const F inv = F(1.0) / a.x;
        return Dual<F, N>(log(a.x), a.y * inv);
    }

    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline Dual<F, N> log10(const Dual<F, N> &a)
// log(10) is precalced
    {
        const F inv = F(1.0) / (F(2.30258509299404568) * a.x);
        return Dual<F, N>(log10(a.x), a.y * inv);
    }

    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline Dual<F, N> log2(const Dual<F, N> &a)
// log(2) is precalced
    {
        const F inv = F(1.0) / (F(0.6931471805599453) * a.x);
        return Dual<F, N>(log2(a.x), a.y * inv);
    }

// natural base log of a+1
    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline Dual<F, N> log1p(const Dual<F, N> &a) {
        F inv = F(1.0) / (a.x + F(1.0));
        return Dual<F, N>(log1p(a.x), a.y * inv);
    }

// power function
    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline Dual<F, N> pow(const Dual<F, N> &a, const Dual<F, N> &b) {
        if (a.x == F(0) && b.x >= F(1)) {
            if (b.x > F(1)) {
                return Dual<F, N>(F());
            }
            return a;
        }
        const F a_to_b = pow(a.x, b.x), bx_by_a_to_bm1 = b.x * a_to_b / a.x;
        if (a.x < F(0) && b.x == floor(b.x)) {
            Dual<F, N> result(a_to_b, bx_by_a_to_bm1 * a.y);
            for (int i = 0; i < N; i++) {
                if (fpclassify(b.y[i]) != FP_ZERO) {
                    result.y[i] = F(std::numeric_limits<F>::quiet_NaN());
                }
            }
            return result;
        }
        return Dual<F, N>(a_to_b, bx_by_a_to_bm1 * a.y + a_to_b * log(a.x) * b.y);
    }

// power function with scalar exponent
    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline Dual<F, N> pow(const Dual<F, N> &a, const F &b) {
        if (a.x == F(0) && b >= F(1)) {
            if (b > 1) {
                return Dual<F, N>(F());
            }
            return a;
        }
        return Dual<F, N>(pow(a.x, b), b * pow(a.x, b - F(1.0)) * a.y);
    }

// power function with scalar base
    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline Dual<F, N> pow(const F &a, const Dual<F, N> &b) {
        if (a == F(0) && b.x >= F(1)) {
            if (b.x > 1) {
                return Dual<F, N>(F());
            }
            return Dual<F, N>(a);
        }
        if (a < F(0) && b.x == floor(b.x)) {
            Dual<F, N> result(pow(a, b.x));
            for (int i = 0; i < N; i++) {
                if (fpclassify(b.y[i]) != FP_ZERO) {
                    result.y[i] = std::numeric_limits<F>::quiet_NaN();
                }
            }
            return result;
        }
        const F a_to_b = pow(a, b.x);
        return Dual<F, N>(a_to_b, a_to_b * log(a.x) * b.y);
    }

// square root
    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline Dual<F, N> sqrt(const Dual<F, N> &a) {
        const F sqrtx = sqrt(a.x), inv = F(1.0) / (F(2.0) * sqrtx);
        return Dual<F, N>(sqrtx, a.y * inv);
    }

// cubic root
    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline Dual<F, N> cbrt(const Dual<F, N> &a) {
        const F cbrtx = cbrt(a.x), inv = F(1.0) / (F(3.0) * cbrt(a.x * a.x));
        return Dual<F, N>(cbrtx, a.y * inv);
    }

    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline Dual<F, N> hypot(const Dual<F, N> &a, const Dual<F, N> &b) {
        const F hpab = hypot(a.x, b.x);
        return Dual<F, N>(hpab, (a.x / hpab * a.y + b.x / hpab * b.y));
    }

    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline Dual<F, N> hypot(const F &a, const Dual<F, N> &b) {
        const F hpab = hypot(a, b.x);
        return Dual<F, N>(hpab, b.x / hpab * b.y);
    }

    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline Dual<F, N> hypot(const Dual<F, N> &a, const F &b) {
        return hypot(b, a);
    }

    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline Dual<F, N> hypot(const Dual<F, N> &a, const Dual<F, N> &b,
                            const Dual<F, N> &c) {
        const F hpabc = hypot(a.x, b.x, c.x);
        return Dual<F, N>(hpabc, a.x / hpabc * a.y + b.x / hpabc * b.y + c.x / hpabc * c.y);
    }

    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline Dual<F, N> hypot(const F &a, const Dual<F, N> &b,
                            const Dual<F, N> &c) {
        const F hpabc = hypot(a.x, b.x, c.x);
        return Dual<F, N>(hpabc, b.x / hpabc * b.y + c.x / hpabc * c.y);
    }

    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline Dual<F, N> hypot(const F &a, const F &b,
                            const Dual<F, N> &c) {
        const F hpabc = hypot(a.x, b.x, c.x);
        return Dual<F, N>(hpabc, c.x / hpabc * c.y);
    }

    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline Dual<F, N> exp2(const Dual<F, N> &a) {
        const F exp2x = exp2(a.x);
        return Dual<F, N>(exp2x, exp2x * F(0.6931471805599453) * a.y);  // precalced log(2)
    }

    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline Dual<F, N> expm1(const Dual<F, N> &a) {
        const F expm1x = expm1(a.x);
        return Dual<F, N>(expm1x, a.y * (expm1x + F(1.0)));
    }

    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline Dual<F, N> sin(const Dual<F, N> &a) {
        return Dual<F, N>(sin(a.x), a.y * cos(a.x));
    }

    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline Dual<F, N> cos(const Dual<F, N> &a) {
        return Dual<F, N>(cos(a.x), a.y * -sin(a.x));
    }

    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline Dual<F, N> exp(const Dual<F, N> &a) {
        const F expx = exp(a.x);
        return Dual<F, N>(expx, expx * a.y);
    }

    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline Dual<F, N> asin(const Dual<F, N> &a) {
        const F inv = F(1.0) / sqrt(F(1.0) - a.x * a.x);
        return Dual<F, N>(asin(a.x), inv * a.y);
    }

    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline Dual<F, N> acos(const Dual<F, N> &a) {
        const F inv = F(-1.0) / sqrt(F(1.0) - a.x * a.x);
        return Dual<F, N>(acos(a.x), inv * a.y);
    }

    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline Dual<F, N> sinh(const Dual<F, N> &a) {
        return Dual<F, N>(sinh(a.x), cosh(a.x) * a.y);
    }

    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline Dual<F, N> cosh(const Dual<F, N> &a) {
        return Dual<F, N>(cosh(a.x), sinh(a.x) * a.y);
    }

    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline Dual<F, N> atan(const Dual<F, N> &a) {
        const F inv = F(1.0) / (F(1.0) + a.x * a.x);
        return Dual<F, N>(atan(a.x), inv * a.y);
    }

    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline Dual<F, N> tan(const Dual<F, N> &a) {
        const F tanx = tan(a.x), tmp = (F(1.0) + tanx * tanx);
        return Dual<F, N>(tanx, tmp * a.y);
    }

    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline Dual<F, N> asinh(const Dual<F, N> &a) {
        const F inv = F(1.0) / sqrt(a.x * a.x + F(1.0));
        return Dual<F, N>(asinh(a.x), inv * a.y);
    }

    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline Dual<F, N> acosh(const Dual<F, N> &a) {
        const F inv = F(1.0) / sqrt(a.x * a.x - F(1.0));
        return Dual<F, N>(acosh(a.x), inv * a.y);
    }

    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline Dual<F, N> tanh(const Dual<F, N> &a) {
        const F tanhx = tanh(a.x), tmp = (F(1.0) - tanhx * tanhx);
        return Dual<F, N>(tanhx, tmp * a.y);
    }

    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline Dual<F, N> atanh(const Dual<F, N> &a) {
        const F inv = F(1.0) / (F(1.0) - a.x * a.x);
        return Dual<F, N>(atanh(a.x), inv * a.y);
    }

    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline Dual<F, N> atan2(const Dual<F, N> &a, const Dual<F, N> &b) {
        const F inv = F(1.0) / (a.x * a.x + b.x * b.x);
        return Dual<F, N>(atan2(a.x, b.y), (inv * b.x) * a.y - (inv * a.x) * b.y);
    }

    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline Dual<F, N> atan2(const F &a, const Dual<F, N> &b) {
        return Dual<F, N>(atan2(a, b.y), -(a / (a * a + b.x * b.x)) * b.y);
    }

    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline Dual<F, N> atan2(const Dual<F, N> &a, const F &b) {
        return atan2(b, a);
    }

#ifndef __clang__  // bessel functions
    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline Dual<F, N> bessel_j0(const Dual<F, N> &a) {
        return Dual<F, N>(std::cyl_bessel_j(0, a.x), -std::cyl_bessel_j(1, a.x) * a.y);
    }

    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline Dual<F, N> bessel_j1(const Dual<F, N> &a) {
        return Dual<F, N>(std::cyl_bessel_j(1, a.x),
                          F(0.5) * (std::cyl_bessel_j(0, a.x) - std::cyl_bessel_j(2, a.x)) * a.y);
    }

    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline Dual<F, N> bessel_jn(int n, const Dual<F, N> &a) {
        return Dual<F, N>(
                std::cyl_bessel_j(n, a.x),
                F(0.5) * (std::cyl_bessel_j(n - 1, a.x) - std::cyl_bessel_j(n + 1, a.x)) * a.y);
    }

    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline Dual<F, N> bessel_i0(const Dual<F, N> &a) {
        return Dual<F, N>(std::cyl_bessel_i(0, a.x), std::cyl_bessel_i(1, a.x) * a.y);
    }

    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline Dual<F, N> bessel_i1(const Dual<F, N> &a) {
        return Dual<F, N>(std::cyl_bessel_i(1, a.x),
                          F(0.5) * (std::cyl_bessel_i(0, a.x) + std::cyl_bessel_i(2, a.x)) * a.y);
    }

    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline Dual<F, N> bessel_in(int n, const Dual<F, N> &a) {
        return Dual<F, N>(
                std::cyl_bessel_i(n, a.x),
                F(0.5) * (std::cyl_bessel_i(n - 1, a.x) + std::cyl_bessel_i(n + 1, a.x)) * a.y);
    }

    template <typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline Dual<F, N> bessel_sph_jn(int n, const Dual<F, N> &a) {
        const F tmp = std::cyl_bessel_i(n, a.x);
        return Dual<F, N>(tmp, (-std::sph_bessel(n + 1, a.x) + F(n / a.x) * tmp) * a.y);
    }

    template <typename F, int N>
    Eigen::Vector<F, N> getGradient(const dual::Dual<dual::DualScalar<F>, N> &a){
        Eigen::Vector<F, N> result;
        for(int i = 0; i < N; i++){
            result(i) = a.y(i).x;
        }
        return result;
    }
    template <typename F, int N>
    Eigen::Vector<F, N> getDirectionalHessian(const dual::Dual<dual::DualScalar<F>, N> &a){
        Eigen::Vector<F, N> result;
        for(int i = 0; i < N; i++){
            result(i) = a.y(i).y;
        }
        return result;
    }
#endif  // bessel functions
}  // namespace dual
namespace std{
    template <typename F, int N>
    static constexpr dual::Dual<F, N> quiet_NaN() noexcept {
        return dual::Dual<F, N>(std::numeric_limits<F>::quiet_NaN());
    }
}
#endif
