#ifndef CODUAL_CODUAL_H
#define CODUAL_CODUAL_H

#include <cmath>
#include <functional>

namespace dual {
    template<typename F>
    struct Codual {
        static_assert(std::is_arithmetic_v<F>, "k");
        F x;
        std::function<void(const F&)> derivative;

        Codual() {
            x = F(0);
            derivative = [](const F &dx) {};
        }

        explicit Codual(F a) {
            x = a;
            derivative = [](const F &dx) -> void {};
        }

        Codual(F a, std::function<void(const F&)> b) {
            x = a;
            derivative = b;
        }

        //compound operators
        inline __attribute__((always_inline)) Codual<F> operator+=(const Codual<F> &b) {
            x += b.x;
            std::function<void(const F&)> old_derivative = this->derivative;
            derivative = [b, old_derivative](const F &dx) -> void {
                old_derivative(dx);
                b.derivative(dx);
            };
            return *this;
        }

        inline __attribute__((always_inline)) Codual<F> operator-=(const Codual<F> &b) {
            x -= b.x;
            std::function<void(const F&)> old_derivative = this->derivative;
            derivative = [b, old_derivative](const F &dx) -> void {
                old_derivative(dx);
                b.derivative(-dx);
            };
            return *this;
        }

        inline __attribute__((always_inline)) Codual<F> operator*=(const Codual<F> &b) {
            std::function<void(const F&)> old_derivative = this->derivative;
            derivative = [this, b, old_derivative](const F &dx) -> void {
                old_derivative(b.x * dx);
                b.derivative(this->x * dx);
            };
            x *= b.x;
            return *this;
        }

        inline __attribute__((always_inline)) Codual<F> operator/=(const Codual<F> &b) {
            const F inv = F(1.0) / b.x;
            x *= inv;
            std::function<void(const F&)> old_derivative = this->derivative;
            derivative = [this, b, inv, old_derivative](const F &dx) -> void {
                old_derivative(dx * inv);
                b.derivative(-(this->x) * dx * inv);
            };
            return *this;
        }

        inline __attribute__((always_inline)) Codual<F> &operator=(const F &b) {
            x = b;
            derivative = [](F &dx) -> void {};
        }
    };

    template<typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) bool isless(const Codual<F> &a, const Codual<F> &b) {
        return std::isless(a.x, b.x);
    }

    template<typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) bool isgreater(const Codual<F> &a, const Codual<F> &b) {
        return std::isgreater(a.x, b.x);
    }

    template<typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) bool islessequal(const Codual<F> &a, const Codual<F> &b) {
        return std::islessequal(a.x, b.x);
    }

    template<typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) bool islessgreater(const Codual<F> &a, const Codual<F> &b) {
        return std::islessgreater(a.x, b.x);
    }

    template<typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) bool isgreaterequal(const Codual<F> &a, const Codual<F> &b) {
        return std::isgreaterequal(a.x, b.x);
    }

    template<typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) bool isunordered(const Codual<F> &a, const Codual<F> &b) {
        return std::isunordered(a.x, b.x);
    }

    template<typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) bool operator<(const Codual<F> &a, const Codual<F> &b) {
        return isless(a, b);
    }

    template<typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) bool operator>(const Codual<F> &a, const Codual<F> &b) {
        return isgreater(a, b);
    }

    template<typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) bool operator<=(const Codual<F> &a, const Codual<F> &b) {
        return islessequal(a, b);
    }

    template<typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) bool operator>=(const Codual<F> &a, const Codual<F> &b) {
        return isgreaterequal(a, b);
    }

    template<typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) bool operator==(const Codual<F> &a, const Codual<F> &b) {
        return (a.x == b.x && a.derivative == b.derivative);
    }

    template<typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) bool operator!=(const Codual<F> &a, const Codual<F> &b) {
        return !(a.x == b.x && a.derivative == b.derivative);
    }

    template<typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Codual<F> operator+(const Codual<F> &a) {
        return a;
    }

    template<typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Codual<F> operator-(const Codual<F> &a) {
        return Codual<F>(-a.x, [a](const F &dx) -> void { a.derivative(-dx); });
    }

    template<typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Codual<F> operator+(const Codual<F> &a, const Codual<F> &b) {
        return Codual<F>(a.x + b.x, [a, b](const F &dx) -> void {
            a.derivative(dx);
            b.derivative(dx);
        });
    }

    template<typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Codual<F> operator-(const Codual<F> &a, const Codual<F> &b) {
        return Codual<F>(a.x - b.x, [a, b](const F &dx) -> void {
            a.derivative(dx);
            b.derivative(-dx);
        });
    }

    template<typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Codual<F> operator-(const Codual<F> &a, const F &b) {
        return Codual<F>(a.x - b, [a](const F &dx) -> void {
            a.derivative(dx);
        });
    }

    template<typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Codual<F> operator-(const F &a, const Codual<F> &b) {
        return Codual<F>(a - b.x, [b](const F &dx) -> void {
            b.derivative(-dx);
        });
    }

    template<typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Codual<F> operator*(const Codual<F> &a, const Codual<F> &b) {
        return Codual<F>(a.x * b.x, [a, b](const F &dx) -> void {
            a.derivative(dx * b.x);
            b.derivative(dx * a.x);
        });
    }

    template<typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Codual<F> operator*(const Codual<F> &a, const F &b) {
        return Codual<F>(a.x * b, [a, b](const F &dx) -> void {
            a.derivative(dx * b);
        });
    }

    template<typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Codual<F> operator*(const F &a, const Codual<F> &b) {
        return Codual<F>(a * b.x, [a, b](const F &dx) -> void {
            b.derivative(dx * a);
        });
    }

    template<typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Codual<F> operator/(const Codual<F> &a, const Codual<F> &b) {
        const F inv = F(1.0) / b.x;
        return Codual<F>(a.x * inv, [a, b, inv](const F &dx) -> void {
            const F dx_by_inv = dx * inv;
            a.derivative(dx_by_inv);
            b.derivative(-a.x * dx_by_inv);
        });
    }

    template<typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Codual<F> abs(const Codual<F> &a) {
        return Codual<F>(std::abs(a.x), [a](const F &dx) -> void {
            a.derivative(dx * copysign(F(1.0), a.x));
        });
    }

    template<typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Codual<F> log(const Codual<F> &a) {
        return Codual<F>(std::log(a.x), [a](const F &dx) -> void {
            a.derivative(dx / a.x);
        });
    }

    template<typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Codual<F> pow(const Codual<F> &a, const Codual<F> &b) {
        if (a.x == 0 && b.x >= 1) {
            if (b.x > 1) {
                return Codual<F>();
            }
            return a;
        }
        const F a_to_b = std::pow(a.x, b.x), bx_by_a_to_bm1 = b.x * a_to_b / a.x;
        if (a.x < 0 && b.x == std::floor(b.x)) {
            Codual<F> result(a_to_b, [a, b, bx_by_a_to_bm1, a_to_b](const F &dx) -> void {
                a.derivative(dx * bx_by_a_to_bm1);
            });
            if (b.y != [](F &dx) {}) {
                result.derivative = [a]() { a.derivative(F(std::numeric_limits<F>::quiet_NaN())); };
            }
            return result;
        }
        return Codual<F>(a_to_b, [a, b, bx_by_a_to_bm1, a_to_b](const F &dx) -> void {
            a.derivative(dx * bx_by_a_to_bm1);
            b.derivative(dx * a_to_b * std::log(a.x));
        });
    }

    template<typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Codual<F> pow(const Codual<F> &a, const F &b) {
        if (a.x == 0 && b >= 1) {
            if (b > 1) {
                return Codual<F>();
            }
            return a;
        }
        const F a_to_b = std::pow(a.x, b), bx_by_a_to_bm1 = b * a_to_b / a.x;
        return Codual<F>(a_to_b, [a, bx_by_a_to_bm1](const F &dx) -> void {
            a.derivative(dx * bx_by_a_to_bm1);
        });
    }

    template<typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Codual<F> exp(const Codual<F> &a) {
        const F expx = std::exp(a.x);
        return Codual<F>(expx, [a, expx](const F &dx) -> void {
            a.derivative(dx * expx);
        });
    }

    template<typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Codual<F> sin(const Codual<F> &a) {
        return Codual<F>(std::sin(a.x), [a](const F &dx) -> void {
            a.derivative(dx * std::cos(a.x));
        });
    }

    template<typename F, typename std::enable_if<std::is_arithmetic_v<F>, bool>::type = true>
    inline __attribute__((always_inline)) Codual<F> cos(const Codual<F> &a) {
        return Codual<F>(std::cos(a.x), [a](const F &dx) -> void {
            a.derivative(-dx * std::sin(a.x));
        });
    }
}
#endif //CODUAL_CODUAL_H