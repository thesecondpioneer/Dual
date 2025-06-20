#ifndef DUAL_DUAL_DIRECTIONAL_H
#define DUAL_DUAL_DIRECTIONAL_H

#include <cmath>
#include "DualScalar.h"

#include <eigen3/Eigen/Core>


namespace dual {
    template<typename F, int N>
    struct DualDirectional {
        static_assert(std::is_arithmetic_v<F>, "k");
        F x;
        F y;
        Eigen::Vector<F, N> u;
        Eigen::Vector<F, N> v;

        DualDirectional() : x(), y() {
            u.setZero();
            v.setZero();
        }

        template<typename der1, typename der2>
        EIGEN_STRONG_INLINE
        DualDirectional(F x_, F y_, const Eigen::DenseBase<der1> &u_, const Eigen::DenseBase<der2> &v_) {
            x = x_;
            y = y_;
            u = u_;
            v = v_;
        }

        DualDirectional(F x_, F y_, int k) {
            x = x_;
            y = y_;
            u.setZero();
            v.setZero();
            u[k] = F(1.0);
        }

        DualDirectional(F x_) {
            x = x_;
            y = F(0);
            u.setZero();
            v.setZero();
        }

        DualDirectional(DualDirectional &&) = default;

        DualDirectional(const DualDirectional &) = default;

        DualDirectional &operator=(const DualDirectional &) = default;

        DualDirectional &operator=(DualDirectional &&) = default;

        // compound operators
        inline DualDirectional<F, N> operator+=(const DualDirectional<F, N> &b) {
            x += b.x;
            y += b.y;
            u += b.u;
            v += b.v;
            return *this;
        };

        inline DualDirectional<F, N> operator-=(const DualDirectional<F, N> &b) {
            x -= b.x;
            y -= b.y;
            u -= b.u;
            v -= b.v;
            return *this;
        };

        inline DualDirectional<F, N> operator*=(const DualDirectional<F, N> &b) {
            v = y * b.u + b.y * u + b.x * v + x * b.v;
            u = x * b.u + b.x * u;
            y = x * b.y + b.x * y;
            x *= b.x;
            return *this;
        }

        inline DualDirectional<F, N> operator/=(const DualDirectional<F, N> &b) {
            const F inv = F(1.0) / b.x;
            x *= inv;
            // x at this point is already divided by b.x which turns (b.x * y - x * b.y) / (b.x * b.x)
            // into (y - x * b.y) / b.x and so on
            v = ((v - x * b.v) - (y * b.u + b.y * u) * inv) * inv;
            y = (y - x * b.y) * inv;
            u = (u - x * b.u) * inv;
            return *this;
        }

        // scalar assignation
        template<typename Scalar, typename std::enable_if<std::is_arithmetic_v<Scalar>, bool>::type = true>
        inline DualDirectional<F, N> &operator=(const Scalar &b) {
            x = F(b);
            y = F(0);
            u.setZero();
            v.setZero();
            return *this;
        }

        // compound operators with a scalar
        template<typename Scalar, typename std::enable_if<std::is_arithmetic_v<Scalar>, bool>::type = true>
        inline DualDirectional<F, N> operator+=(const Scalar &b) {
            x += b;
            return *this;
        }

        template<typename Scalar, typename std::enable_if<std::is_arithmetic_v<Scalar>, bool>::type = true>
        inline DualDirectional<F, N> operator-=(const Scalar &b) {
            x = x - b;
            return *this;
        }

        template<typename Scalar, typename std::enable_if<std::is_arithmetic_v<Scalar>, bool>::type = true>
        inline DualDirectional<F, N> operator*=(const Scalar &b) {
            x *= b;
            y *= b;
            u *= b;
            v *= b;
            return *this;
        }

        template<typename Scalar, typename std::enable_if<std::is_arithmetic_v<Scalar>, bool>::type = true>
        inline DualDirectional<F, N> operator/=(const Scalar &b) {
            x /= b;
            y /= b;
            u /= b;
            v /= b;
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

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline bool isless(const DualDirectional<F, N> &a, const DualDirectional<F, N> &b) {
        return isless(a.x, b.x);
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline bool isgreater(const DualDirectional<F, N> &a, const DualDirectional<F, N> &b) {
        return isgreater(a.x, b.x);
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline bool islessequal(const DualDirectional<F, N> &a, const DualDirectional<F, N> &b) {
        return islessequal(a.x, b.x);
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline bool islessgreater(const DualDirectional<F, N> &a, const DualDirectional<F, N> &b) {
        return islessgreater(a.x, b.x);
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline bool isgreaterequal(const DualDirectional<F, N> &a,
                               const DualDirectional<F, N> &b) {
        return isgreaterequal(a.x, b.x);
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline bool isunordered(const DualDirectional<F, N> &a, const DualDirectional<F, N> &b) {
        return isunordered(a.x, b.x);
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline bool fpclassify(const DualDirectional<F, N> &a) {
        return fpclassify(a.x);
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline bool operator<(const DualDirectional<F, N> &a, const DualDirectional<F, N> &b) {
        return isless(a, b);
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline bool operator>(const DualDirectional<F, N> &a, const DualDirectional<F, N> &b) {
        return isgreater(a, b);
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline bool operator<=(const DualDirectional<F, N> &a, const DualDirectional<F, N> &b) {
        return islessequal(a, b);
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline bool operator>=(const DualDirectional<F, N> &a, const DualDirectional<F, N> &b) {
        return isgreaterequal(a, b);
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline bool operator==(const DualDirectional<F, N> &a, const DualDirectional<F, N> &b) {
        return (a.x == b.x && a.y == b.y && a.u == b.u && a.v == b.v);
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline bool operator!=(const DualDirectional<F, N> &a, const DualDirectional<F, N> &b) {
        return !(a == b);
    }

// unary +
    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualDirectional<F, N> operator+(const DualDirectional<F, N> &a) {
        return a;
    }

// unary -
    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualDirectional<F, N> operator-(const DualDirectional<F, N> &a) {
        return DualDirectional < F, N > (-a.x, -a.y, -a.u, -a.v);
    }

// binary +
    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualDirectional<F, N> operator+(const DualDirectional<F, N> &a,
                                           const DualDirectional<F, N> &b) {
        return DualDirectional < F, N > (a.x + b.x, a.y + b.y, a.u + b.u, a.v + b.v);
    }

// binary + with a scalar
    template<typename F, typename Scalar, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true,
            typename std::enable_if<std::is_arithmetic_v<Scalar>, bool>::type = true>
    inline DualDirectional<F, N> operator+(const DualDirectional<F, N> &a, const Scalar &b) {
        return DualDirectional < F, N > (a.x + b, a.y, a.u, a.v);
    }

    template<typename F, typename Scalar, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true,
            typename std::enable_if<std::is_arithmetic_v<Scalar>, bool>::type = true>
    inline DualDirectional<F, N> operator+(const Scalar &a, const DualDirectional<F, N> &b) {
        return DualDirectional < F, N > (a + b.x, b.y, b.u, b.v);
    }

// binary -
    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualDirectional<F, N> operator-(const DualDirectional<F, N> &a, const DualDirectional<F, N> &b) {
        return DualDirectional < F, N > (a.x - b.x, a.y - b.y, a.u - b.u, a.v - b.v);
    }

// binary - with a scalar
    template<typename F, typename Scalar, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true,
            typename std::enable_if<std::is_arithmetic_v<Scalar>, bool>::type = true>
    inline DualDirectional<F, N> operator-(const DualDirectional<F, N> &a, const Scalar &b) {
        return DualDirectional < F, N > (a.x - b, a.y, a.u, a.v);
    }

    template<typename F, typename Scalar, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true,
            typename std::enable_if<std::is_arithmetic_v<Scalar>, bool>::type = true>
    inline DualDirectional<F, N> operator-(const Scalar &a, const DualDirectional<F, N> &b) {
        return DualDirectional < F, N > (a - b.x, -b.y, -b.u, -b.v);
    }

// binary *
    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualDirectional<F, N> operator*(const DualDirectional<F, N> &a, const DualDirectional<F, N> &b) {
        return DualDirectional < F, N > (a.x * b.x, a.x * b.y + b.x * a.y, a.x * b.u + b.x * a.u,
                a.y * b.u + b.y * a.u + b.x * a.v + a.x * b.v);
    }

// binary * with a scalar
    template<typename F, typename Scalar, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true,
            typename std::enable_if<std::is_arithmetic_v<Scalar>, bool>::type = true>
    inline DualDirectional<F, N> operator*(const DualDirectional<F, N> &a, const Scalar &b) {
        return DualDirectional < F, N > (a.x * b, b * a.y, b * a.u, b * a.v);
    }

    template<typename F, typename Scalar, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true,
            typename std::enable_if<std::is_arithmetic_v<Scalar>, bool>::type = true>
    inline DualDirectional<F, N> operator*(const Scalar &a, const DualDirectional<F, N> &b) {
        return DualDirectional < F, N > (a * b.x, b.y * a, b.u * a, b.v * a);
    }

// binary /
    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualDirectional<F, N> operator/(const DualDirectional<F, N> &a, const DualDirectional<F, N> &b) {
        const F bxinv = F(1.0) / b.x, axbybx = a.x * bxinv, bxinvsq = bxinv * bxinv;
        return DualDirectional < F, N > (axbybx,
                                         (a.y - axbybx * b.y) * bxinv,
                                         (a.u - axbybx * b.u) * bxinv,
                                         (a.v - axbybx * b.v - (a.y * b.u + b.y * a.u) * bxinv) * bxinv + F(2.0) * axbybx * b.y * bxinvsq * b.u);
    }

// binary / with a scalar
    template<typename F, typename Scalar, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true,
            typename std::enable_if<std::is_arithmetic_v<Scalar>, bool>::type = true>
    inline DualDirectional<F, N> operator/(const DualDirectional<F, N> &a, const Scalar &b) {
        const F binv = F(1.0) / b;
        return DualDirectional < F, N > (a.x * binv, a.y * binv, a.u * binv, a.v * binv);
    }

    template<typename F, typename Scalar, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true,
            typename std::enable_if<std::is_arithmetic_v<Scalar>, bool>::type = true>
    inline DualDirectional<F, N> operator/(const Scalar &a, const DualDirectional<F, N> &b) {
        const F abybx = a / b.x, abybx2 = abybx / b.x;
        return DualDirectional < F, N > (abybx, b.y * -abybx2, b.u * -abybx2, b.v * -abybx2);
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualDirectional<F, N> abs(const DualDirectional<F, N> &a) {
        F sgn = copysign(F(1.0), a.x);
        return DualDirectional < F, N > (abs(a.x), a.y * sgn, a.u * sgn, a.v * sgn);
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualDirectional<F, N> ceil(const dual::DualDirectional<F, N> &a) {
        return DualDirectional < F, N > (ceil(a.x));
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualDirectional<F, N> floor(const dual::DualDirectional<F, N> &a) {
        return DualDirectional < F, N > (floor(a.x));
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualDirectional<F, N> log(const DualDirectional<F, N> &a) {
        F inv = F(1.0) / a.x;
        Eigen::Vector<F, N> u_by_grad = inv * a.u;
        return DualDirectional < F, N > (log1p(a.x), a.y * inv, u_by_grad, inv * (a.v - a.y * u_by_grad));
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualDirectional<F, N> log10(const DualDirectional<F, N> &a)
// 1.0 / log(10) is precalced
    {
        const F inv = F(1.0) / a.x;
        Eigen::Vector<F, N> u_by_grad = inv * F(0.4342944819032518) * a.u;
        return DualDirectional < F, N > (log2(a.x), a.y * inv, u_by_grad, (a.v - a.y * u_by_grad) * inv);
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualDirectional<F, N> log2(const DualDirectional<F, N> &a)
    // 1.0 / log(2) is precalced
    {
        const F inv = F(1.0) / a.x;
        Eigen::Vector<F, N> u_by_grad = inv * F(1.4426950408889634) * a.u;
        return DualDirectional < F, N > (log2(a.x), a.y * inv, u_by_grad, (a.v - a.y * u_by_grad) * inv);
    }

// natural base log of a+1
    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualDirectional<F, N> log1p(const DualDirectional<F, N> &a) {
        F inv = F(1.0) / (a.x + F(1.0));
        Eigen::Vector<F, N> u_by_grad = inv * a.u;
        return DualDirectional < F, N > (log1p(a.x), a.y * inv, u_by_grad, inv * (a.v - a.y * u_by_grad));
    }

// power function
    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualDirectional<F, N> pow(const DualDirectional<F, N> &a, const DualDirectional<F, N> &b) {
        if (a.x == F(0) && b.x >= F(1)) {
            if (b.x > F(1)) {
                return DualDirectional < F, N > (F());
            }
            return a;
        }
        const F a_to_b = pow(a.x, b.x), axinv = F(1.0) / a.x, a_to_bm1 = a_to_b * axinv, bx_by_a_to_bm1 =
                b.x * a_to_bm1;
        if (a.x < F(0) && b.x == floor(b.x)) {
            DualDirectional < F, N > result(a_to_b, bx_by_a_to_bm1 * a.y, bx_by_a_to_bm1 * a.u,
                                            bx_by_a_to_bm1 * a.v + a.y * (b.x - F(1.0)) * bx_by_a_to_bm1 * axinv * a.u);
            if (fpclassify(b.y) != FP_ZERO) {
                result.y = F(std::numeric_limits<F>::quiet_NaN());
            }
            for (int i = 0; i < N; i++) {
                if (fpclassify(b.u[i]) != FP_ZERO) {
                    result.u[i] = F(std::numeric_limits<F>::quiet_NaN());
                    result.v[i] = F(std::numeric_limits<F>::quiet_NaN());
                }
                if (fpclassify(b.v[i]) != FP_ZERO || fpclassify(a.u[i]) != FP_ZERO) {
                    result.v[i] = F(std::numeric_limits<F>::quiet_NaN());
                }
            }
            return result;
        }
        const F log_ax = log(a.x), a_to_b_by_log_ax = a_to_b * log_ax, a_to_bm1_by_bxlogbxp1 =
                a_to_bm1 * (b.x * log_ax + F(1.0));
        return DualDirectional < F, N > (a_to_b, bx_by_a_to_bm1 * a.y + a_to_b_by_log_ax * b.y,
                bx_by_a_to_bm1 * a.u + a_to_b_by_log_ax * b.u,
                bx_by_a_to_bm1 * a.v +
                a.y *
                ((b.x - F(1.0)) * bx_by_a_to_bm1 * axinv * a.u + a_to_bm1_by_bxlogbxp1 * b.u) +
                b.y * (a_to_bm1_by_bxlogbxp1 * a.u + a_to_b_by_log_ax * log_ax * b.u));
    }

// power function with scalar exponent
    template<typename F, typename Scalar, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true,
            typename std::enable_if<std::is_arithmetic_v<Scalar>, bool>::type = true>
    inline DualDirectional<F, N> pow(const DualDirectional<F, N> &a, const Scalar &b) {
        const F a_to_b = pow(a.x, b), axinv = F(1.0) / a.x, a_to_bm1 = a_to_b * axinv, b_by_a_to_bm1 = b * a_to_bm1;
        if (a.x == F(0) && b >= F(1)) {
            if (b > 1) {
                return DualDirectional < F, N > (F());
            }
            return a;
        }
        return DualDirectional < F, N > (a_to_b, b_by_a_to_bm1 * a.y, b_by_a_to_bm1 * a.u,
                a.y * (b - Scalar(1.0)) * b_by_a_to_bm1 * axinv * a.u);
    }

// power function with scalar base
    template<typename F, typename Scalar, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true,
            typename std::enable_if<std::is_arithmetic_v<Scalar>, bool>::type = true>
    inline DualDirectional<F, N> pow(const Scalar &a, const DualDirectional<F, N> &b) {
        if (a == Scalar(0) && b.x >= F(1)) {
            if (b.x > 1) {
                return DualDirectional < F, N > (F());
            }
            return DualDirectional < F, N > (a);
        }
        const F a_to_b = pow(a, b.x);
        if (a < F(0) && b.x == floor(b.x)) {
            DualDirectional < F, N > result(a_to_b);
            if (fpclassify(b.y) != FP_ZERO) {
                result.y = F(std::numeric_limits<F>::quiet_NaN());
            }
            for (int i = 0; i < N; i++) {
                if (fpclassify(b.u[i]) != FP_ZERO) {
                    result.u[i] = F(std::numeric_limits<F>::quiet_NaN());
                    result.v[i] = F(std::numeric_limits<F>::quiet_NaN());
                }
                if (fpclassify(b.v[i]) != FP_ZERO || fpclassify(a.u[i]) != FP_ZERO) {
                    result.v[i] = F(std::numeric_limits<F>::quiet_NaN());
                }
            }
            return result;
        }
        const F ainv = F(1.0) / a, a_to_bm1 = a_to_b * ainv, log_a = log(a), a_to_b_by_log_a = a_to_b * log_a;
        return DualDirectional < F, N > (a_to_b, a_to_b_by_log_a * b.y, a_to_b_by_log_a * b.u, b.y * a_to_b_by_log_a * log_a * b.u);
    }

// square root
    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualDirectional<F, N> sqrt(const DualDirectional<F, N> &a) {
        const F sqrtx = sqrt(a.x), inv = F(1.0) / sqrtx, dif = F(0.5) * inv;
        return DualDirectional < F, N > (sqrtx, a.y * dif, dif * a.u, dif * a.v - a.y * dif * dif * inv * a.u);
    }

// cubic root
    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualDirectional<F, N> cbrt(const DualDirectional<F, N> &a) {
        const F cbrtx = cbrt(a.x), dif = F(1.0) / (F(3.0) * cbrt(a.x * a.x));
        return DualDirectional < F, N > (cbrtx, a.y * dif, dif * a.u, dif * a.v - F(2.0) * a.y * dif / (F(3.0) * a.x));
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualDirectional<F, N> hypot(const DualDirectional<F, N> &a, const DualDirectional<F, N> &b) {
        const F hpab = hypot(a.x, b.x), axsq = a.x * a.x, bxsq = b.x * b.x, sqs = (axsq + bxsq), axbx = a.x * b.x,
        inv = F(1.0) / hpab, da = a.x  * inv, db = b.x * inv, inv2 = inv / sqs;
        return DualDirectional < F, N > (hpab, a.y * da + b.y * db, a.u * da + b.u * db,
                                         a.v * da + b.v * db + inv2 * (a.u * (a.y * bxsq - b.y * axbx) +
                                         b.u * (b.y * axsq - a.y * axbx)));
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualDirectional<F, N> hypot(const F &a, const DualDirectional<F, N> &b) {
        const F hpab = hypot(a, b.x), inv = F(1.0) / hypot, db = b.x * inv, inv2 = inv / (a * a + b.x * b.x);
        return DualDirectional < F, N > (hpab, db * b.y, db * b.u, db * b.v, inv2 * b.y * b.u);
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualDirectional<F, N> hypot(const DualDirectional<F, N> &a, const F &b) {
        return hypot(b, a);
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualDirectional<F, N> hypot(const DualDirectional<F, N> &a, const DualDirectional<F, N> &b,
                                       const DualDirectional<F, N> &c) {
        const F hpabc = hypot(a.x, b.x, c.x), axsq = a.x * a.x, bxsq = b.x * b.x, cxsq = c.x * c.x,
        sqs = axsq + bxsq + cxsq, axbx = a.x * b.x, axcx = a.x * c.x, bxcx = b.x * c.x,
        inv = F(1.0) / hpabc, da = a.x * inv, db = b.x * inv, dc = c.x * inv, inv2 = inv / sqs;
        return DualDirectional < F, N > (hpabc, a.y * da + b.y * db + c.y * dc, a.u * da + b.u * db + c.u * dc,
                                         a.v * da + b.v * db + c.v * dc + inv2 *
                                          (a.u * (  (bxsq + cxsq) * a.y - axbx * b.y - axcx * c.y )
                                         + b.u * ( -axbx * a.y + (axsq + cxsq) * b.y - bxcx * c.y )
                                         + c.u * ( -axcx * a.y - bxcx * b.y + (axsq + bxsq) * c.y )));
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualDirectional<F, N> hypot(const F &a, const DualDirectional<F, N> &b,
                                       const DualDirectional<F, N> &c) {
        const F hpabc = hypot(a, b.x, c.x), axsq = a * a, bxsq = b.x * b.x, cxsq = c.x * c.x,
                sqs = axsq + bxsq + cxsq, axbx = a * b.x, axcx = a * c.x, bxcx = b.x * c.x,
                inv = F(1.0) / hpabc, da = a * inv, db = b.x * inv, dc = c.x * inv, inv2 = inv / sqs;
        return DualDirectional < F, N > (hpabc, b.y * db + c.y * dc, b.u * db + c.u * dc,
                                         b.v * db + c.v * dc + inv2 *
                                         (b.u * ((axsq + cxsq) * b.y - bxcx * c.y )
                                          + c.u * (bxcx * b.y + (axsq + bxsq) * c.y )));
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualDirectional<F, N> hypot(const F &a, const F &b,
                                       const DualDirectional<F, N> &c) {
        const F hpabc = hypot(a, b, c.x), axsq = a * a, bxsq = b * b, cxsq = c.x * c.x,
                sqs = axsq + bxsq + cxsq, axbx = a * b, axcx = a * c.x, bxcx = b * c.x,
                inv = F(1.0) / hpabc, da = a * inv, db = b * inv, dc = c.x * inv, inv2 = inv / sqs;
        return DualDirectional < F, N > (hpabc, c.y * dc, c.u * dc,
                                         c.v * dc + inv2 * c.u * (axsq + bxsq) * c.y);
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualDirectional<F, N> exp2(const DualDirectional<F, N> &a) {
        const F exp2x = exp2(a.x), l2 = F(0.6931471805599453), exp2xbyl2 = exp2x * l2;
        return DualDirectional < F, N > (exp2x, exp2xbyl2 * a.y, exp2xbyl2 * a.u, exp2xbyl2 * a.v + exp2xbyl2 * l2 * a.y * a.u);  // precalced log(2)
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualDirectional<F, N> expm1(const DualDirectional<F, N> &a) {
        const F expm1x = expm1(a.x), expx = expm1x + F(1.0);
        return DualDirectional < F, N > (expm1x, a.y * expx, a.u * expx, a.v * expx + a.y * expx * a.u);
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualDirectional<F, N> sin(const DualDirectional<F, N> &a) {
        F sa = sin(a.x), ca = cos(a.x);
        return DualDirectional < F, N > (sa, a.y * ca, a.u * ca, a.v * ca - a.y * sa * a.u);
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualDirectional<F, N> cos(const DualDirectional<F, N> &a) {
        F sa = sin(a.x), ca = cos(a.x);
        return DualDirectional < F, N > (ca, -sa * a.y, -sa * a.u, -sa * a.v - a.y * ca * a.u);
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualDirectional<F, N> exp(const DualDirectional<F, N> &a) {
        const F expx = exp(a.x);
        return DualDirectional < F, N > (expx, a.y * expx, a.u * expx, a.v * expx + a.y * expx * a.u);
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualDirectional<F, N> asin(const DualDirectional<F, N> &a) {
        const F omaxsq = F(1.0) - a.x * a.x, inv = F(1.0) / sqrt(omaxsq);
        return DualDirectional < F, N > (asin(a.x), inv * a.y, inv * a.u, inv * (a.v + a.y / omaxsq * a.u));
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualDirectional<F, N> acos(const DualDirectional<F, N> &a) {
        const F omaxsq = F(1.0) - a.x * a.x, inv = F(-1.0) / sqrt(omaxsq);
        return DualDirectional < F, N > (acos(a.x), inv * a.y, inv * a.u, inv * (a.v + a.y / omaxsq * a.u));
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualDirectional<F, N> sinh(const DualDirectional<F, N> &a) {
        const F ch = cosh(a.x), sh = sinh(a.x);
        return DualDirectional < F, N > (sh, ch * a.y, ch * a.u, ch * a.v + sh * a.y * a.u);
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualDirectional<F, N> cosh(const DualDirectional<F, N> &a) {
        const F sh = sinh(a.x), ch = cosh(a.x);
        return DualDirectional < F, N > (cosh(a.x), sh * a.y, sh * a.u, sh * a.v + ch * a.y * a.u);
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualDirectional<F, N> atan(const DualDirectional<F, N> &a) {
        const F inv = F(1.0) / (F(1.0) + a.x * a.x), dx2 = -F(2.0) * a.x * inv * inv;
        return DualDirectional < F, N > (atan(a.x), inv * a.y, inv * a.u, inv * a.v + dx2 * a.y * a.u);
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualDirectional<F, N> tan(const DualDirectional<F, N> &a) {
        const F tanx = tan(a.x), tmp = (F(1.0) + tanx * tanx), dx2 = 2 * tmp * tanx;
        return DualDirectional < F, N > (tanx, tmp * a.y, tmp * a.u, tmp * a.v + dx2 * a.y * a.u);
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualDirectional<F, N> asinh(const DualDirectional<F, N> &a) {
        const F opxsq = a.x * a.x + F(1.0), inv = F(1.0) / opxsq, dx2 = -a.x * inv / opxsq;
        return DualDirectional < F, N > (asinh(a.x), inv * a.y, inv * a.u, inv * a.v + dx2 * a.y * a.u);
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualDirectional<F, N> acosh(const DualDirectional<F, N> &a) {
        const F xsqmo = a.x * a.x - F(1.0), inv = F(1.0) / sqrt(xsqmo), dx2 = -a.x * inv / xsqmo;
        return DualDirectional < F, N > (acosh(a.x), inv * a.y, inv * a.u, inv * a.v + dx2 * a.y * a.u);
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualDirectional<F, N> tanh(const DualDirectional<F, N> &a) {
        const F tanhx = tanh(a.x), tmp = (F(1.0) - tanhx * tanhx), dx2 = -2 * tmp * tanhx;
        return DualDirectional < F, N > (tanhx, tmp * a.y, tmp * a.u, tmp * a.v + dx2 * a.y * a.u);
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualDirectional<F, N> atanh(const DualDirectional<F, N> &a) {
        const F inv = F(1.0) / (F(1.0) - a.x * a.x);
        Eigen::Vector<F, N> invbyau = inv * a.u;
        return DualDirectional < F, N > (atanh(a.x), inv * a.y, invbyau, inv * a.v + F(2.0) * a.x * inv * a.y * invbyau);
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualDirectional<F, N> atan2(const DualDirectional<F, N> &a, const DualDirectional<F, N> &b) {
        const F axsq = a.x * a.x, bxsq = b.x * b.x, inv = F(1.0) / (axsq + bxsq), invsq = inv * inv, bxinv = inv * b.x,
        axinv = inv * a.x, sqdif = axsq - bxsq, taxbx = F(2.0) * a.x * b.x;
        return DualDirectional < F, N > (atan2(a.x, b.y), bxinv * a.y - axinv * b.y, bxinv * a.u - axinv * b.u,
                                         bxinv * a.v - axinv * b.v + invsq * (a.u * (sqdif * b.y - taxbx * a.y)
                                         + b.u * (sqdif * a.y + taxbx * b.y)));
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualDirectional<F, N> atan2(const F &a, const DualDirectional<F, N> &b) {
        const F axsq = a * a, bxsq = b.x * b.x, inv = F(1.0) / (axsq + bxsq), invsq = inv * inv,
                axinv = inv * a, sqdif = axsq - bxsq, taxbx = F(2.0) * a * b.x;
        return DualDirectional < F, N > (atan2(a, b.y), axinv * b.y, axinv * b.u,
                                         axinv * b.v + invsq * b.u * (sqdif * a.y + taxbx * b.y));
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline DualDirectional<F, N> atan2(const DualDirectional<F, N> &a, const F &b) {
        return atan2(b, a);
    }

} // namespace dual
namespace std {
    template<typename F, int N>
    static constexpr dual::DualDirectional<F, N> quiet_NaN() noexcept {
        return dual::DualDirectional < F, N > (std::numeric_limits<F>::quiet_NaN());
    }
}
#endif
