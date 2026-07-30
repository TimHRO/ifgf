#include <Eigen/Dense>
#include <iostream>

#include <cmath>

#include <chrono>
#include <cstdlib>
#include <fenv.h>
#include <oneapi/tbb/blocked_range.h>
#include <random>
#include <string>
#include <tbb/global_control.h>
#include <tbb/task_arena.h>

#include "../core/config.hpp"
#include "../core/ifgf_library.hpp"

#include <tbb/parallel_for.h>
#include <sycl/sycl.hpp>

const int dim = 3;

typedef std::complex<double> Complex;
typedef Eigen::Vector<double, dim> Point;


Complex kappa = Complex(0.001, 3.14 * 4.0);
std::string op = "SL";  // "SL", "DL", or "CF"

// Single layer:  exp(-k r) / (4 pi r)
std::complex<double> kernel_sl(const Point &x, const Point &y,
                               const Point &normal) {
  (void)normal;
  const double r = (x - y).norm();
  if (r < 1e-14)
    return 0;
  return std::exp(-((std::complex<double>)kappa) * r) / (4.0 * M_PI * r);
}

// Double layer: -1/(4 pi) * 1/r^2 * exp(-k r) * (-k - 1/r) * <x-y, n>
std::complex<double> kernel_dl(const Point &x, const Point &y,
                               const Point &normal) {
  const Point d = x - y;
  const double r = d.norm();
  if (r < 1e-14)
    return 0;

  const std::complex<double> k = (std::complex<double>)kappa;
  const double xn = d.dot(normal);
  const double id = 1.0 / r;

  return -(1.0 / (4.0 * M_PI)) * (id * id) * std::exp(-k * r) *
         (-k - std::complex<double>(id)) * xn;
}

// Combined field:
// G = exp(-k r)/(4 pi r), as dG/dn_y - eta*G with eta = -k:
//   exp(-k r)/(4 pi r^3) * ( <n_y, x-y>(1 + k r) + k r^2 )
// nxy = +<n_y, x-y>
std::complex<double> kernel_cf(const Point &x, const Point &y,
                               const Point &normal) {
  const Point d = x - y;
  const double r = d.norm();
  if (r < 1e-14)
    return 0;

  const std::complex<double> k = (std::complex<double>)kappa;
  const double nxy = normal.dot(d);

  return std::exp(-k * r) / (4.0 * M_PI * r * r * r) *
         (nxy * (1.0 + k * r) + k * r * r);
}

std::complex<double> my_kernel(const Point &x, const Point &y,
                               const Point &normal) {
  if (op == "DL") return kernel_dl(x, y, normal);
  if (op == "CF") return kernel_cf(x, y, normal);
  return kernel_sl(x, y, normal);
}

auto randomPointOnSphere(double r) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);

  PointScalar theta = dis(gen) * 2.0 * M_PI;      // Random angle theta
  PointScalar phi = acos(2.0 * dis(gen) - 1.0);   // Random angle phi

  PointScalar x = r * sin(phi) * cos(theta);
  PointScalar y = r * sin(phi) * sin(theta);
  PointScalar z = r * cos(phi);

  return Eigen::Vector<PointScalar, 3>(x, y, z);
}

int main(int argc, char **argv) {
  srand((unsigned int)1);
  typedef Eigen::Matrix<PointScalar, dim, Eigen::Dynamic> PointArray;

  // Command line: <N> <k.real> <k.imag> <operator>
  const int N = argc > 1 ? atoi(argv[1]) : 100000;
  const double kre = argc > 2 ? atof(argv[2]) : 0.001;
  const double kim = argc > 3 ? atof(argv[3]) : 3.14 * 4.0;
  if (argc > 4) op = argv[4];

  kappa = Complex((RealScalar)kre, (RealScalar)kim);
  const double rad = 1.0;

  std::cout << "N=" << N << "  kappa=(" << kre << ", " << kim << ")"
            << "  operator=" << op << "  radius=" << rad << std::endl;

  for (auto platform : sycl::platform::get_platforms()) {
    std::cout << "Platform: "
              << platform.get_info<sycl::info::platform::name>() << std::endl;

    for (auto device : platform.get_devices()) {
      std::cout << "\tDevice: " << device.get_info<sycl::info::device::name>()
                << std::endl;
    }
  }

  auto global_control =
      tbb::global_control(tbb::global_control::max_allowed_parallelism, 32);

  PointArray srcs(3, N);
  tbb::parallel_for(tbb::blocked_range<size_t>(0, srcs.cols()),
                    [&](tbb::blocked_range<size_t> r) {
                      for (size_t i = r.begin(); i < r.end(); i++) {
                        srcs.col(i) = randomPointOnSphere(rad);
                      }
                    });

  PointArray normals = srcs;
  PointArray targets = srcs;
  normals.colwise().normalize();

  const PointScalar *p_srcs = srcs.data();
  const PointScalar *p_targets = targets.data();
  const PointScalar *p_normals = normals.data();

  for (int j = 0; j < 1; j++) {
    using namespace std::chrono;
    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    std::unique_ptr<ifgf::HelmholtzSL3D> op_sl;
    std::unique_ptr<ifgf::HelmholtzDL3D> op_dl;
    std::unique_ptr<ifgf::HelmholtzCF3D> op_cf;

    if (op == "DL") {
      op_dl = std::make_unique<ifgf::HelmholtzDL3D>(kappa, 1000, 8, 1, -1., -1.,
                                                    -1.);
      op_dl->init(p_srcs, srcs.cols(), p_targets, targets.cols(), p_normals,
                  normals.cols());
    } else if (op == "CF") {
      op_cf = std::make_unique<ifgf::HelmholtzCF3D>(kappa, 1000, 8, 1, -1., -1.,
                                                    -1.);
      op_cf->init(p_srcs, srcs.cols(), p_targets, targets.cols(), p_normals,
                  normals.cols());
    } else {
      op_sl = std::make_unique<ifgf::HelmholtzSL3D>(kappa, 1000, 8, 1, -1., -1.,
                                                    -1.);
      op_sl->init(p_srcs, srcs.cols(), p_targets, targets.cols());
    }

    Eigen::Vector<std::complex<RealScalar>, Eigen::Dynamic> weights(
        srcs.cols());
    weights = Eigen::Vector<RealScalar, Eigen::Dynamic>::Random(srcs.cols());

    Eigen::Vector<std::complex<RealScalar>, Eigen::Dynamic> result(
        targets.cols());
    result.setZero();

    auto do_mult = [&]() {
      if (op == "DL")
        op_dl->mult(weights.data(), weights.size(), result.data(),
                    result.size());
      else if (op == "CF")
        op_cf->mult(weights.data(), weights.size(), result.data(),
                    result.size());
      else
        op_sl->mult(weights.data(), weights.size(), result.data(),
                    result.size());
    };

    high_resolution_clock::time_point t12 = high_resolution_clock::now();
    duration<PointScalar> time_span =
        duration_cast<duration<PointScalar>>(t12 - t1);
    std::cout << "init" << time_span.count() << " seconds" << std::endl;

    // first one is not timed!
    do_mult();

    high_resolution_clock::time_point t13 = high_resolution_clock::now();
    const int Nmult = 1;
    for (int i = 0; i < Nmult; i++) {
      std::cout << "mult" << std::endl;
      do_mult();
      std::cout << "done multiplying" << std::endl;
    }
    high_resolution_clock::time_point t2 = high_resolution_clock::now();

    time_span = duration_cast<duration<PointScalar>>(t2 - t13);
    std::cout << "mult time per iter=" << time_span.count() / Nmult
              << " seconds" << std::endl;
    std::cout << "qusi gmres total=" << time_span.count() << " seconds"
              << std::endl;

    fedisableexcept(FE_DIVBYZERO | FE_OVERFLOW | FE_UNDERFLOW | FE_INVALID);

    srand((unsigned)time(NULL));
    double e_n = 0;
    double e_d = 0;
    for (int s = 0; s < 200; s++) {
      std::complex<double> val = 0;
      int index = rand() % targets.cols();
      for (int i = 0; i < srcs.cols(); i++) {
        val += std::complex<double>(weights[i]) *
               my_kernel(srcs.col(i), targets.col(index), normals.col(i));
      }

      const std::complex<double> got = std::complex<double>(result[index]);
      e_n += std::abs(val - got) * std::abs(val - got);
      e_d += std::abs(val) * std::abs(val);
    }
    double e_m = std::sqrt(e_n / e_d);
    std::cout << "summary e_m: " << e_m << std::endl;
  }
}