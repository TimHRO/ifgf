#include <Eigen/Dense>

#include "../helmholtz_ifgf.hpp"
#include "../ifgfoperator.hpp"
#include "../modified_helmholtz_ifgf.hpp"
#include "../octree.hpp"

#include "../grad_helmholtz_ifgf.hpp"

#include <cstdlib>
#include <fenv.h>
#include <tbb/global_control.h>
#include <tbb/task_arena.h>

#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using Complex = std::complex<double>;
const double PI = 3.14159265358979323846;

// --------------------------------------------------------------
// Exact kernel and exact weights
// --------------------------------------------------------------
double scaling_factor(double r) { return 1.0 / (4.0 * PI * r); }

Complex K_exact(const Complex& s, double r) { return scaling_factor(r) * std::exp(-s * r); }

std::vector<double> exact_weights(double dt, double r, int N)
{
    double C = scaling_factor(r);
    double factor = C * std::exp(-r / dt);
    double ratio = r / dt;
    double power = 1.0;
    double factorial = 1.0;
    std::vector<double> w(N + 1);
    for (int j = 0; j <= N; ++j)
    {
        w[j] = factor * power / factorial;
        power *= ratio;
        factorial *= (j + 1);
    }
    return w;
}

// --------------------------------------------------------------
// IFGF approximation simulation: adds a complex relative error of
// given magnitude 'eps' (deterministic random seed)
// --------------------------------------------------------------
Complex K_ifgf(const Complex& s, double r, double eps)
{
    static std::mt19937 rng(12345); // fixed seed
    static std::uniform_real_distribution<double> dist(-1.0, 1.0);
    Complex exact = K_exact(s, r);
    if (eps == 0.0)
        return exact;
    Complex error = eps * Complex(dist(rng), dist(rng)) * std::pow(2.0, s);
    // Complex error = eps * std::exp(-s.real());
    // Complex error = eps * s;
    // Complex error = eps * (std::sin(s.real()) + Complex(0.0, std::cos(s.imag()))/ (2.0 * (1 +
    // std::abs(s))));

    return exact * (1.0 + error);
}

// --------------------------------------------------------------
// Compute approximate weights via contour integral (3.4)
// --------------------------------------------------------------
std::vector<double> compute_weights_contour(Complex (*K_fun)(const Complex&, double, double),
                                            double dt, double r, int N, double lambda, double eps)
{
    int M = N + 1;
    Complex zeta = std::exp(Complex(0.0, 2.0 * PI / M));
    std::vector<double> w_hat(N + 1, 0.0);

    for (int l = 0; l < M; ++l)
    {
        Complex z = lambda * std::pow(zeta, -l);
        Complex s = (1.0 - z) / dt;
        Complex K_val = K_fun(s, r, eps);
        for (int j = 0; j <= N; ++j)
        {
            Complex factor = std::pow(lambda, -j) / double(M) * std::pow(zeta, l * j);
            w_hat[j] += (K_val * factor).real();
        }
    }
    return w_hat;
}

// --------------------------------------------------------------
// Debug mode: verify quadrature error using exact kernel
// --------------------------------------------------------------
void debug_mode()
{
    double r = 0.5;
    double dt = 0.01;
    int N = 10;
    double lambda_fixed = 0.99; // typical value for testing

    std::vector<double> w_exact = exact_weights(dt, r, N);
    std::vector<double> w_quad = compute_weights_contour(K_ifgf, dt, r, N, lambda_fixed, 0.0);
    double max_err = 0.0;
    for (int j = 0; j <= N; ++j)
    {
        double err = std::abs(w_exact[j] - w_quad[j]);
        if (err > max_err)
            max_err = err;
    }
    std::cout << "Quadrature error (exact kernel, λ=" << lambda_fixed << ", N=" << N
              << ") = " << max_err << std::endl;
}

// --------------------------------------------------------------
// Main: expects either "debug" or a numeric eps
// --------------------------------------------------------------
int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <eps>   (for convergence test)\n"
                  << "   or: " << argv[0] << " debug   (to check quadrature error)\n";
        return 1;
    }

    std::string arg = argv[1];
    if (arg == "debug")
    {
        debug_mode();
        return 0;
    }

    double eps = std::stod(arg);

    // Fixed parameters
    double r = 0.5;
    double dt = 0.01;
    int N = 100;

    // Optimal lambda for this eps
    // double lambda_opt = std::pow(eps, 1.0 / (2.0 * N + 1.0));
    double lambda_opt = std::pow(10.0, -8 / N);

    std::vector<double> w_exact = exact_weights(dt, r, N);
    std::vector<double> w_approx = compute_weights_contour(K_ifgf, dt, r, N, lambda_opt, eps);

    double max_err = 0.0;
    for (int j = 0; j <= N; ++j)
    {
        double err = std::abs(w_exact[j] - w_approx[j]);
        if (err > max_err)
            max_err = err;
    }

    // Print only the error (for Python parsing)
    std::cout << max_err << std::endl;
    return 0;
}