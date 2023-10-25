#ifndef DUAL_DUAL_H
#define DUAL_DUAL_H

#include <cmath>
#include "Eigen/Core"

namespace dual {
    template<typename F, int N>
    struct Dual {
        static_assert(std::is_arithmetic_v<F>, "k");
        F x;
        Eigen::Vector<F, N> y;

        Dual() : x() {
            y.setConstant(F());
        }

        template<typename der>
        EIGEN_STRONG_INLINE Dual(const F &x_, const Eigen::DenseBase <der> &y_) {
            x = x_;
            y = y_;
        }

        Dual(const F &x_, int k) {
            x = x_;
            y.setConstant(F());
            y[k] = F(1.0);
        }

        explicit Dual(const F &x_) {
            x = x_;
            y.setConstant(F());
        }

        Dual(const Dual<F, N> &a) {
            x = a.x;
            y = a.y;
        }

        //compound operators
        inline __attribute__((always_inline)) Dual<F, N> operator+=(const Dual<F, N> &b) {
            x += b.x;
            y += b.y;
            return *this;
        };

        inline __attribute__((always_inline)) Dual<F, N> operator-=(const Dual<F, N> &b) {
            x -= b.x;
            y -= b.y;
            return *this;
        };

        inline __attribute__((always_inline)) Dual<F, N> operator*=(const Dual<F, N> &b) {
            y = x * b.y + b.x * y;
            x *= b.x;
            return *this;
        }

        inline __attribute__((always_inline)) Dual<F, N> operator/=(const Dual<F, N> &b) {
            x /= b.x;
            //x at this point is already divided by b.x which turns (b.x * y - x * b.y) / (b.x * b.x) into (y - x * b.y) / b.x
            y = (y - x * b.y) / b.x;
            return *this;
        }

        //scalar assignation
        inline __attribute__((always_inline)) Dual<F, N> &operator=(const F &b) {
            x = b;
            y.setConstant(F());
            return *this;
        }

        // compound operators with a scalar
        inline __attribute__((always_inline)) Dual<F, N> operator+=(const F &b) {
            x += b;
            return *this;
        }

        inline __attribute__((always_inline)) Dual<F, N> operator-=(const F &b) {
            x = x - b;
            return *this;
        }

        inline __attribute__((always_inline)) Dual<F, N> operator*=(const F &b) {
            x *= b;
            y *= b;
            return *this;
        }

        inline __attribute__((always_inline)) Dual<F, N> operator/=(const F &b) {
            x /= b;
            y /= b;
            return *this;
        }
    };

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) bool isless(const Dual<F, N> &a, const Dual<F, N> &b) {
        return std::isless(a.x, b.x);
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) bool isgreater(const Dual<F, N> &a, const Dual<F, N> &b) {
        return std::isgreater(a.x, b.x);
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) bool islessequal(const Dual<F, N> &a, const Dual<F, N> &b) {
        return std::islessequal(a.x, b.x);
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) bool islessgreater(const Dual<F, N> &a, const Dual<F, N> &b) {
        return std::islessgreater(a.x, b.x);
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) bool isgreaterequal(const Dual<F, N> &a, const Dual<F, N> &b) {
        return std::isgreaterequal(a.x, b.x);
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) bool isunordered(const Dual<F, N> &a, const Dual<F, N> &b) {
        return std::isunordered(a.x, b.x);
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) bool operator<(const Dual<F, N> &a, const Dual<F, N> &b) {
        return isless(a, b);
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) bool operator>(const Dual<F, N> &a, const Dual<F, N> &b) {
        return isgreater(a, b);
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) bool operator<=(const Dual<F, N> &a, const Dual<F, N> &b) {
        return islessequal(a, b);
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) bool operator>=(const Dual<F, N> &a, const Dual<F, N> &b) {
        return isgreaterequal(a, b);
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) bool operator==(const Dual<F, N> &a, const Dual<F, N> &b) {
        return (a.x == b.x && a.y == b.y);
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) bool operator!=(const Dual<F, N> &a, const Dual<F, N> &b) {
        return !(a.x == b.x && a.y == b.y);
    }

    //unary +
    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Dual<F, N> operator+(const Dual<F, N> &a) {
        return a;
    }

    //unary -
    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Dual<F, N> operator-(const Dual<F, N> &a) {
        return Dual<F, N>(-a.x, -a.y);
    }

    //binary +
    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Dual<F, N> operator+(const Dual<F, N> &a, const Dual<F, N> &b) {
        return Dual<F, N>(a.x + b.x, a.y + b.y);
    }

    //binary + with a scalar
    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Dual<F, N> operator+(const Dual<F, N> &a, const F &b) {
        return Dual<F, N>(a.x + b, a.y);
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Dual<F, N> operator+(const F &a, const Dual<F, N> &b) {
        return Dual<F, N>(a + b.x, b.y);
    }

    //binary -
    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Dual<F, N> operator-(const Dual<F, N> &a, const Dual<F, N> &b) {
        return Dual<F, N>(a.x - b.x, a.y - b.y);
    }

    //binary - with a scalar
    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Dual<F, N> operator-(const Dual<F, N> &a, const F &b) {
        return Dual<F, N>(a.x - b, a.y);
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Dual<F, N> operator-(const F &a, const Dual<F, N> &b) {
        return Dual<F, N>(a - b.x, -b.y);
    }

    //binary *
    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Dual<F, N> operator*(const Dual<F, N> &a, const Dual<F, N> &b) {
        return Dual<F, N>(a.x * b.x, a.x * b.y + b.x * a.y);
    }

    //binary * with a scalar
    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Dual<F, N> operator*(const Dual<F, N> &a, F &b) {
        return Dual<F, N>(a.x * b, b * a.y);
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Dual<F, N> operator*(const F &a, const Dual<F, N> &b) {
        return Dual<F, N>(a * b.x, b.y * a);
    }

    //binary /
    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Dual<F, N> operator/(const Dual<F, N> &a, const Dual<F, N> &b) {
        const F bxinv = F(1.0) / b.x, axbybx = a.x * bxinv;
        return Dual<F, N>(a)(axbybx, (a.y - axbybx * b.y) * bxinv);
    }

    //binary / with a scalar
    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Dual<F, N> operator/(const Dual<F, N> &a, const F &b) {
        const F binv = F(1.0) / b;
        return Dual<F, N>(a.x * binv, a.y * binv);
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Dual<F, N> operator/(const F &a, const Dual<F, N> &b) {
        const F abybx = a / b.x;
        return Dual<F, N>(abybx, b.y * (-abybx / b.x));
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Dual<F, N> abs(const Dual<F, N> &a) {
        return Dual<F, N>(abs(a.x), a.y * copysign(F(1.0), a.x));
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Dual<F, N> ceil(const dual::Dual<F, N> &a) {
        return Dual<F, N>(std::ceil(a.x));
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Dual<F, N> floor(const dual::Dual<F, N> &a) {
        return Dual<F, N>(std::floor(a.x));
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Dual<F, N> log(const Dual<F, N> &a) {
        return Dual<F, N>(std::log(a.x), a.y / a.x);
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Dual<F, N> log10(const Dual<F, N> &a) {
        return Dual<F, N>(std::log10(a.x), a.y / (std::log(F(10.0)) * a.x));
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Dual<F, N> log2(const Dual<F, N> &a) {
        return Dual<F, N>(std::log2(a.x), a.y / (std::log(F(2.0)) * a.x));
    }

    //natural base log of a+1
    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Dual<F, N> log1p(const Dual<F, N> &a) {
        return Dual<F, N>(std::log1p(a.x), a.y / a.x);
    }

    // power function
    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Dual<F, N> pow(const Dual<F, N> &a, const Dual<F, N> &b) {
        if (a.x == 0 && b.x >= 1) {
            if (b.x > 1) {
                return Dual<F, N>(F());
            }
            return a;
        } else {
            if (a.x < 0 && b.x == std::floor(b.x)) {
                const F a_to_b = std::pow(a.x, b.x);
                if (b.y == -b.y) {
                    return Dual<F, N>(a_to_b, (b.x * a_to_b / a.x) * a.y);
                }
                return Dual<F, N>(a_to_b, F(std::numeric_limits<double>::quiet_NaN()));
            }
        }
        const F a_to_b = std::pow(a.x, b.x);
        return Dual<F, N>(a_to_b, b.y * (a_to_b * std::log(a.x)) + a.y * (b.x * a_to_b / a.x));
    }

    //power function with scalar exponent
    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Dual<F, N> pow(const Dual<F, N> &a, const F &b) {
        if (a.x == 0 && b.x >= 1) {
            if (b > 1) {
                return Dual<F, N>(F());
            }
            return a;
        }
        return Dual<F, N>(std::pow(a.x, b), a.y * b * std::pow(a.x, b - F(1.0)));
    }

    //power function with scalar base
    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Dual<F, N> pow(const F &a, const Dual<F, N> &b) {
        if (a == 0 && b.x >= 1) {
            if (b.x > 1) {
                return Dual<F, N>(F());
            }
            return Dual<F, N>(a);
        } else {
            if (a < 0 && b.x == std::floor(b.x)) {
                if (b.y == 0) {
                    return Dual<F, N>(std::pow(a.x, b.x));
                }
                return Dual<F, N>(std::pow(a.x, b.x), F(std::numeric_limits<double>::quiet_NaN()));
            }
        }
        const F a_to_b = std::pow(a, b.x);
        return Dual<F, N>(a_to_b, a_to_b * b.y * std::log(a.x));
    }

    //square root
    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Dual<F, N> sqrt(const Dual<F, N> &a) {
        const F sqrtx = std::sqrt(a.x);
        return Dual<F, N>(sqrtx, a.y / F(2.0) * sqrtx);
    }

    //cubic root
    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Dual<F, N> cbrt(const Dual<F, N> &a) {
        const F cbrtx = std::cbrt(a.x);
        return Dual<F, N>(cbrtx, a.y / F(3.0) * cbrtx * cbrtx);
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Dual<F, N> hypot(const Dual<F, N> &a, const Dual<F, N> &b) {
        const F hpab = std::hypot(a.x, b.x);
        return Dual<F, N>(hpab, (a.x * a.y + b.x * b.y) / hpab);
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Dual<F, N> hypot(const F &a, const Dual<F, N> &b) {
        const F hpab = std::hypot(a, b.x);
        return Dual<F, N>(hpab, (b.x * b.y) / hpab);
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Dual<F, N> hypot(const Dual<F, N> &a, const F &b) {
        return hypot(b, a);
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Dual<F, N>
    hypot(const Dual<F, N> &a, const Dual<F, N> &b, const Dual<F, N> &c) {
        const F hpabc = std::hypot(a.x, b.x, c.x);
        return Dual<F, N>(hpabc, (a.x * a.y + b.x * b.y + c.x * c.y) / hpabc);
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Dual<F, N> hypot(const F &a, const Dual<F, N> &b, const Dual<F, N> &c) {
        const F hpabc = std::hypot(a.x, b.x, c.x);
        return Dual<F, N>(hpabc, (b.x * b.y + c.x * c.y) / hpabc);
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Dual<F, N> hypot(const F &a, const F &b, const Dual<F, N> &c) {
        const F hpabc = std::hypot(a.x, b.x, c.x);
        return Dual<F, N>(hpabc, c.x * c.y / hpabc);
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Dual<F, N> exp2(const Dual<F, N> &a) {
        const F exp2x = std::exp2(a.x);
        return Dual<F, N>(exp2x, a.y * exp2x * std::log(F(2.0)));
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Dual<F, N> expm1(const Dual<F, N> &a) {
        const F expm1x = std::expm1(a.x);
        return Dual<F, N>(expm1x, a.y * (expm1x + F(1.0)));
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Dual<F, N> sin(const Dual<F, N> &a) {
        return Dual<F, N>(std::sin(a.x), a.y * std::cos(a.x));
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Dual<F, N> cos(const Dual<F, N> &a) {
        return Dual<F, N>(std::cos(a.x), -a.y * std::sin(a.x));
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Dual<F, N> exp(const Dual<F, N> &a) {
        const F expx = std::exp(a.x);
        return Dual<F, N>(expx, expx * a.y);
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Dual<F, N> asin(const Dual<F, N> &a) {
        return Dual<F, N>(std::asin(a.x), a.y / std::sqrt(F(1.0) - a.x * a.x));
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Dual<F, N> acos(const Dual<F, N> &a) {
        return Dual<F, N>(std::acos(a.x), -a.y / std::sqrt(F(1.0) - a.x * a.x));
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Dual<F, N> sinh(const Dual<F, N> &a) {
        return Dual<F, N>(std::sinh(a.x), a.y * std::cosh(a.x));
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Dual<F, N> cosh(const Dual<F, N> &a) {
        return Dual<F, N>(std::cosh(a.x), a.y * std::sinh(a.x));
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Dual<F, N> atan(const Dual<F, N> &a) {
        return Dual<F, N>(std::atan(a.x), a.y / (1 + a.x * a.x));
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Dual<F, N> tan(const Dual<F, N> &a) {
        const F tanx = std::tan(a.x);
        return Dual<F, N>(tanx, a.y * (F(1.0) + tanx * tanx));
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Dual<F, N> asinh(const Dual<F, N> &a) {
        return Dual<F, N>(std::asinh(a.x), a.y / std::sqrt(a.x * a.x + F(1.0)));
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Dual<F, N> acosh(const Dual<F, N> &a) {
        return Dual<F, N>(std::acosh(a.x), a.y / std::sqrt(a.x * a.x - F(1.0)));
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Dual<F, N> tanh(const Dual<F, N> &a) {
        Dual<F, N> result;
        const F tanhx = std::tanh(a.x);
        return Dual<F, N>(tanhx, a.y * (F(1.0) - tanhx * tanhx));
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Dual<F, N> atanh(const Dual<F, N> &a) {
        return Dual<F, N>(std::atanh(a.x), a.y / (F(1.0) - a.x * a.x));
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Dual<F, N> atan2(const Dual<F, N> &a, const Dual<F, N> &b) {
        return Dual<F, N>(std::atan2(a.x, b.y), (-b.x * a.y + a.x * b.y) / (a.x * a.x + b.x * b.x));
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Dual<F, N> atan2(const F &a, const Dual<F, N> &b) {
        return Dual<F, N>(std::atan2(a, b.y), (a * b.y) / (a * a + b.x * b.x));
    }

    template<typename F, int N, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Dual<F, N> atan2(const Dual<F, N> &a, const F &b) {
        return hypot(b, a);
    }
}
#endif
