#include <iostream>
#include "Dual.h"
#include "Clock.h"
#include "ceres/ceres.h"
#include "glog/logging.h"
#include "Eigen/Core"

double f(double x, double y, double b1, double b2, double b3, double b4) {
    return b1 * pow(1.0 + exp(b2 - b3 * x), -1.0 / b4) - y;
}

Eigen::Matrix<double, 4, 1> der(double x, double y, double b1, double b2, double b3, double b4) {
    return Eigen::Matrix<double, 4, 1>{
            1.0 / pow(1.0 + exp(b2 - b3 * x), 1.0 / b4),
            -b1 * exp(b2 - b3 * x) / (b4 * pow(1.0 + exp(b2 - b3 * x), (1.0 + b4) / b4)),
            b1 * exp(b2 - b3 * x) * x / (b4 * pow(1.0 + exp(b2 - b3 * x), (1.0 + b4) / b4)),
            b1 * log(1.0 + exp(b2 - b3 * x)) / (b4 * b4 * pow(1.0 + exp(b2 - b3 * x), 1.0 / b4))
    };
}

void opt(double x, double y, double b1, double b2, double b3, double b4, double &f,
         Eigen::Matrix<double, 4, 1> &der) {
    double ex = exp(b2 - b3 * x), exp1 = ex + 1.0, pw = pow(exp1, 1.0 / b4), db2 = b1 * ex / (b4 * pw * exp1);
    der[0] = 1.0 / pw;
    f = b1 * der[0] - y;
    der[1] = -db2;
    der[2] = x * db2;
    der[3] = b1 * log(exp1) / (b4 * b4 * pw);
}

dual::Dual<double, 4>
autod(double x, double y, dual::Dual<double, 4> b1, dual::Dual<double, 4> b2, dual::Dual<double, 4> b3, dual::Dual<double, 4> b4) {
    return b1 * pow(1.0 + exp(b2 - b3 * x), -1.0 / b4) - y;
}

ceres::Jet<double, 4>
autodcr(double x, double y, ceres::Jet<double, 4> b1, ceres::Jet<double, 4> b2, ceres::Jet<double, 4> b3,
        ceres::Jet<double, 4> b4) {
    return b1 * pow(1.0 + exp(b2 - b3 * x), -1.0 / b4) - y;
}

int main() {
    std::string mode;
    std::cin >> mode;
    if (mode == "vs") { //mode for comparing the time performance
        double opt_func, analyt_func;
        int count1 = 0, count2 = 0, count3 = 0, count4 = 0;
        Eigen::Matrix<double, 4, 1> opt_deriv, analyt_deriv;
        dual::Dual<double, 4> dual_deriv;
        ceres::Jet<double, 4> ceres_deriv;
        std::cout.precision(16);
        stopwatch opt_time, dual_time, analyt_time, ceres_time;
        opt_time.start();
        for (double x = 1; x <= 1.1; x += 0.01) {
            for (double y = 1; y <= 1.1; y += 0.01) {
                for (double b1 = 1; b1 <= 1.1; b1 += 0.02) {
                    for (double b2 = 1; b2 <= 1.1; b2 += 0.02) {
                        for (double b3 = 1; b3 <= 1.1; b3 += 0.02) {
                            for (double b4 = 1; b4 <= 1.1; b4 += 0.02) {
                                count1++;
                                opt(x, y, b1, b2, b3, b4, opt_func, opt_deriv);
                            }
                        }
                    }
                }
            }
        }
        opt_time.stop();
        analyt_time.start();
        for (double x = 1; x <= 1.1; x += 0.01) {
            for (double y = 1; y <= 1.1; y += 0.01) {
                for (double b1 = 1; b1 <= 1.1; b1 += 0.02) {
                    for (double b2 = 1; b2 <= 1.1; b2 += 0.02) {
                        for (double b3 = 1; b3 <= 1.1; b3 += 0.02) {
                            for (double b4 = 1; b4 <= 1.1; b4 += 0.02) {
                                count2++;
                                analyt_func = f(x, y, b1, b2, b3, b4);
                                analyt_deriv = der(x, y, b1, b2, b3, b4);
                            }
                        }
                    }
                }
            }
        }
        analyt_time.stop();
        ceres_time.start();
        for (double x = 1; x <= 1.1; x += 0.01) {
            for (double y = 1; y <= 1.1; y += 0.01) {
                for (double b1 = 1; b1 <= 1.1; b1 += 0.02) {
                    for (double b2 = 1; b2 <= 1.1; b2 += 0.02) {
                        for (double b3 = 1; b3 <= 1.1; b3 += 0.02) {
                            for (double b4 = 1; b4 <= 1.1; b4 += 0.02) {
                                count3++;
                                ceres_deriv = autodcr(x, y, ceres::Jet<double, 4>(b1, 0), ceres::Jet<double, 4>(b2, 1),
                                                      ceres::Jet<double, 4>(b3, 2), ceres::Jet<double, 4>(b4, 3));
                            }
                        }
                    }
                }
            }
        }
        ceres_time.stop();
        dual_time.start();
        for (double x = 1; x <= 1.1; x += 0.01) {
            for (double y = 1; y <= 1.1; y += 0.01) {
                for (double b1 = 1; b1 <= 1.1; b1 += 0.02) {
                    for (double b2 = 1; b2 <= 1.1; b2 += 0.02) {
                        for (double b3 = 1; b3 <= 1.1; b3 += 0.02) {
                            for (double b4 = 1; b4 <= 1.1; b4 += 0.02) {
                                count4++;
                                dual_deriv = autod(x, y, dual::Dual<double, 4>(b1, 0), dual::Dual<double, 4>(b2, 1),
                                                   dual::Dual<double, 4>(b3, 2), dual::Dual<double, 4>(b4, 3));
                            }
                        }
                    }
                }
            }
        }
        dual_time.stop();
        std::cout << std::string(33, ' ') << "Last function value     Last derivative value"
                  << std::string(62, ' ') << "Average time/ns" << std::endl;

        std::cout << std::fixed << "Analytical derivative:" << std::string(11, ' ') << analyt_func
                  << std::string(4, ' ') << analyt_deriv.transpose() << "     " << analyt_time.total_time()/count1
                  << std::endl;

        std::cout << "Optimized analytical derivative: " << opt_func << std::string(4, ' ') << opt_deriv.transpose()
                  << std::string(5, ' ') << opt_time.total_time()/count2 << std::endl;

        std::cout << "Automatic differentiation:" << std::string(7, ' ') << dual_deriv.x
                  << std::string(4, ' ') << dual_deriv.y.transpose() << std::string(5, ' ')
                  << dual_time.total_time()/count3 << std::endl;

        std::cout << "Ceres automatic differentiation:" << std::string(1, ' ') << ceres_deriv.a << std::string(4, ' ')
                  << ceres_deriv.v.transpose() << std::string(5, ' ') << ceres_time.total_time()/count4 << std::endl;

        std::cout << "Time difference compared to Ceres: "
                  << ceres_time.total_time() - dual_time.total_time() << "ns, "
                  << 100 * (ceres_time.total_time() - dual_time.total_time()) / ceres_time.total_time() << "%"
                  << std::endl;

    } else if (mode == "cr") { //mode for comparing the precision of functions
        std::cout.precision(16);
        double eps, merr = -1;
        int c = 0;
        for(double x = 1; x <= 10; x+=0.01){
            dual::Dual<double, 1> t1(x, Eigen::Vector<double, 1>(pow(x, 2))), t1r;
            ceres::Jet<double, 1> t1c(x, Eigen::Vector<double, 1>(pow(x, 2))), t1cr;
            dual::Dual<double, 2> t2(x, Eigen::Vector<double, 2>(pow(x, 2), pow(x, 3))), t2r;
            ceres::Jet<double, 2> t2c(x, Eigen::Vector<double, 2>(pow(x, 2), pow(x, 3))), t2cr;
            dual::Dual<double, 3> t3(x, Eigen::Vector<double, 3>(pow(x, 2), pow(x, 3), pow(x, 4))), t3r;
            ceres::Jet<double, 3> t3c(x, Eigen::Vector<double, 3>(pow(x, 2), pow(x, 3), pow(x, 4))), t3cr;
            t1r = dual::abs(t1);
            t1cr = ceres::abs(t1c);
            if ((t1r.y != t1cr.v) or (t1r.x != t1cr.a)){
                std::cerr << "error in abs" << std::endl;
                std::cout << x << std::endl << std::fixed << t1r.x << ' ' << t1cr.a << std::endl << t1r.y << std::endl << t1cr.v;
                break;
            }
            t1r = dual::log(t1);
            t1cr = ceres::log(t1c);
            if ((t1r.y != t1cr.v) or (t1r.x != t1cr.a)){
                std::cerr << "error in log" << std::endl;
                std::cout << x << std::endl << std::fixed << t1r.x << ' ' << t1cr.a << std::endl << t1r.y << std::endl << t1cr.v;
                break;
            }
            t1r = dual::log10(t1);
            t1cr = ceres::log10(t1c);
            if ((t1r.y != t1cr.v) or (t1r.x != t1cr.a)){
                std::cerr << "error in log10" << std::endl;
                std::cout << x << std::endl << std::fixed << t1r.x << ' ' << t1cr.a << std::endl << t1r.y << std::endl << t1cr.v;
                break;
            }
            t1r = dual::log2(t1);
            t1cr = ceres::log2(t1c);
            if ((t1r.y != t1cr.v) or (t1r.x != t1cr.a)){
                std::cerr << "error in log2" << std::endl;
                std::cout << x << std::endl << std::fixed << t1r.x << ' ' << t1cr.a << std::endl << t1r.y << std::endl << t1cr.v;
                break;
            }
            t1r = dual::log1p(t1);
            t1cr = ceres::log1p(t1c);
            if ((t1r.y != t1cr.v) or (t1r.x != t1cr.a)){
                std::cerr << "error in log1p" << std::endl;
                std::cout << x << std::endl << std::fixed << t1r.x << ' ' << t1cr.a << std::endl << t1r.y << std::endl << t1cr.v;
                break;
            }
            t2r = dual::pow(t2, t2);
            t2cr = ceres::pow(t2c, t2c);
            Eigen::Vector<double, 2>t2rd = t2r.y - t2cr.v;
            if ((t2r.y != t2cr.v) or (t2r.x != t2cr.a)){
                std::cerr << "error in pow" << std::endl;
                std::cout << x << std::endl << std::fixed << t2r.x << ' ' << t2cr.a << std::endl << t2r.y << std::endl << t2cr.v << std::endl;
                c++;
                merr = std::max((t2r.y - t2cr.v).array().abs().sum(), merr);
                //break;
            }
        }
        std::cout << c << "/1000 errors with a maximal error of " << merr << std::endl;
    }
    return 0;
}