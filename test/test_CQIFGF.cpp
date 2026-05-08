#include <Eigen/Dense>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "../modified_helmholtz_ifgf.hpp"

using Complex = std::complex<double>;
const double PI = 3.14159265358979323846;
typedef Eigen::Matrix<double, 3, Eigen::Dynamic> PointArray;

Eigen::Vector3d randomPointOnSphere()
{
    static std::mt19937 gen(42);
    static std::uniform_real_distribution<> dis(0.0, 1.0);
    double theta = dis(gen) * 2.0 * PI;
    double phi = std::acos(2.0 * dis(gen) - 1.0);
    return 5 * Eigen::Vector3d(std::sin(phi) * std::cos(theta), std::sin(phi) * std::sin(theta),
                           std::cos(phi));
}

// Exact kernel K(s) = exp(-s*r) / (4π r)

Complex K_exact(const Complex& s, double r)
{
    if (r < 1e-12)
        return Complex(0.0, 0.0);
    return std::exp(-s * r) / (4.0 * PI * r);
}

// Exact kernel matrix (targets × sources), later summed via inner product with source weights
// (matrix vector multiplication)

Eigen::MatrixXcd exact_kernel_matrix(const Complex& s, const PointArray& sources,
                                     const PointArray& targets)
{
    int nsrc = sources.cols();
    int ntgt = targets.cols();
    Eigen::MatrixXcd Kmat(ntgt, nsrc);
    for (int j = 0; j < ntgt; ++j)
    {
        for (int i = 0; i < nsrc; ++i)
        {
            double r = (sources.col(i) - targets.col(j)).norm();
            if (r < 1e-12)
                r = 1e-12;
            Kmat(j, i) = K_exact(s, r);
        }
    }
    return Kmat;
}

// Generate fixed random source vector (size nsrc)

Eigen::VectorXcd random_source_vector(int nsrc)
{
    static std::mt19937 gen(12345);
    static std::uniform_real_distribution<double> dist(-1.0, 1.0);
    Eigen::VectorXcd w(nsrc);
    for (int i = 0; i < nsrc; ++i)
    {
        w(i) = Complex(dist(gen), dist(gen));
    }
    return w;
}

// Compute exact time‑domain solution for all targets
// For a fixed source vector w_src, compute the exact target vector
// at each quadrature frequency s
// Return matrix (ntgt × (N+1)) of time‑domain values

Eigen::MatrixXd exact_solution(const PointArray& sources, const PointArray& targets,
                               const Eigen::VectorXcd& w_src, double dt, int N, double lambda)
{
    int ntgt = targets.cols();
    int M = N + 1;
    Complex zeta = std::exp(Complex(0.0, 2.0 * PI / M));
    Eigen::MatrixXd sol(ntgt, N + 1);
    sol.setZero();

    for (int l = 0; l < M; ++l)
    {
        Complex z = lambda * std::pow(zeta, -l);
        Complex s = (1.0 - z) / dt; // BDF1 frequency
        Eigen::MatrixXcd Kmat = exact_kernel_matrix(s, sources, targets);
        Eigen::VectorXcd b_exact = Kmat * w_src; // exact target vector

        for (int j = 0; j < ntgt; ++j)
        {
            Complex Kval = b_exact(j);
            for (int jj = 0; jj <= N; ++jj)
            {
                Complex factor = std::pow(lambda, -jj) / double(M) * std::pow(zeta, l * jj);
                sol(j, jj) += (Kval * factor).real();
            }
        }
    }
    return sol;
}

// Compute approximate time‑domain solution using IFGF operator

Eigen::MatrixXd approx_solution(const PointArray& sources, const PointArray& targets,
                                const Eigen::VectorXcd& w_src, double dt, int N, double lambda,
                                int tarPerBox, int o, int base_n)
{
    int ntgt = targets.cols();
    int M = N + 1;
    Complex zeta = std::exp(Complex(0.0, 2.0 * PI / M));
    Eigen::MatrixXd sol(ntgt, N + 1);
    sol.setZero();

    for (int l = 0; l < M; ++l)
    {
        Complex z = lambda * std::pow(zeta, -l);
        Complex s = (1.0 - z) / dt;
        std::cout << "s: " << s << "\n";
        ModifiedHelmholtzIfgfOperator<3> op(s, tarPerBox, o, base_n, -1);
        op.init(sources, targets);
        Eigen::VectorXcd b_approx = op.mult(w_src); // approximate target vector

        for (int j = 0; j < ntgt; ++j)
        {
            Complex Kval = b_approx(j);
            for (int jj = 0; jj <= N; ++jj)
            {
                Complex factor = std::pow(lambda, -jj) / double(M) * std::pow(zeta, l * jj);
                sol(j, jj) += (Kval * factor).real();
            }
        }
    }
    return sol;
}

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cerr << "  Usage " << argv[0]
                  << " <tarPerBox> <base_n> <num_points> <dt> <o> <N> <lambda>\n";
        return 1;
    }

    int tarPerBox = std::stoi(argv[1]);
    int base_n = std::stoi(argv[2]);
    int num_points = std::stoi(argv[3]);
    double dt = std::stod(argv[4]);
    int o = std::stoi(argv[5]);
    int N = std::stoi(argv[6]);
    double lambda = std::stod(argv[7]);

    // Generate source and target points on unit sphere
    PointArray sources(3, num_points), targets(3, num_points);
    for (int i = 0; i < num_points; ++i)
    {
        sources.col(i) = randomPointOnSphere();
        targets.col(i) = randomPointOnSphere();
    }

    Eigen::VectorXcd w_src = random_source_vector(num_points);

    Eigen::MatrixXd exact = exact_solution(sources, targets, w_src, dt, N, lambda);

    Eigen::MatrixXd approx =
        approx_solution(sources, targets, w_src, dt, N, lambda, tarPerBox, o, base_n);

    double abs_error = (exact - approx).norm();
    double norm_exact = exact.norm();
    double rel_error = abs_error / (norm_exact + 1e-15);

    // Largest Absolute Error at j=N
    double N_error = (exact.col(N) - approx.col(N)).norm();

    std::cout << std::scientific << "Relative Error: " << rel_error << std::endl;
    std::cout << std::scientific << "N Absolute Error: " << N_error << std::endl;
    std::cout << std::scientific << "Absolute Error: " << abs_error << std::endl;
    return 0;
}