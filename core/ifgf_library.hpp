#ifndef __IFGF_LIBRARY__
#define __IFGF_LIBRARY__

#include "config.hpp"
#include <complex>
#include <memory>
#include <Eigen/Dense>

namespace ifgf {

class HIfgfSLPrivate;

class HelmholtzSL3D {
public:
  typedef Eigen::Array<PointScalar, 3, Eigen::Dynamic> PointArray;

  HelmholtzSL3D(RealScalar waveNumber, size_t leafSize, size_t order,
                size_t n_elem = 1, PointScalar tol = -1);
  ~HelmholtzSL3D();

  void init(const PointScalar *srcs, size_t n_srcs, const PointScalar *targets,
            size_t n_targets);

  void mult(const std::complex<float> *weights, size_t n_weights,
            std::complex<float> *result, size_t n_targets);

  // for convenience also provide a double version that casts
  void mult(const std::complex<double> *weights, size_t n_weights,
            std::complex<double> *result, size_t n_targets);

private:
  std::unique_ptr<HIfgfSLPrivate> d;
};

class MHIfgfSLPrivate;

class ModHelmholtzSL3D {
public:
  typedef Eigen::Array<PointScalar, 3, Eigen::Dynamic> PointArray;

  ModHelmholtzSL3D(std::complex<RealScalar> waveNumber, size_t leafSize,
                   size_t order, size_t n_elem = 1, PointScalar tol = -1,
                   double maxk = -1, double minSigma = -1);
  ~ModHelmholtzSL3D();

  void init(const PointScalar *srcs, size_t n_srcs, const PointScalar *targets,
            size_t n_targets);

  void mult(const std::complex<float> *weights, size_t n_weights,
            std::complex<float> *result, size_t n_targets);

  // for convenience also provide a double version that casts
  void mult(const std::complex<double> *weights, size_t n_weights,
            std::complex<double> *result, size_t n_targets);

private:
  std::unique_ptr<MHIfgfSLPrivate> d;
};

class MHIfgfDLPrivate;

class ModHelmholtzDL3D {
public:
  typedef Eigen::Array<PointScalar, 3, Eigen::Dynamic> PointArray;

  ModHelmholtzDL3D(std::complex<RealScalar> waveNumber, size_t leafSize,
                   size_t order, size_t n_elem = 1, PointScalar tol = -1,
                   double maxk = -1, double minSigma = -1);
  ~ModHelmholtzDL3D();

  // normals are the per-source normals: DIM*n_normals, laid out like srcs,
  // and n_normals must equal n_srcs
  void init(const PointScalar *srcs, size_t n_srcs, const PointScalar *targets,
            size_t n_targets, const PointScalar *normals, size_t n_normals);

  void mult(const std::complex<float> *weights, size_t n_weights,
            std::complex<float> *result, size_t n_targets);

  // for convenience also provide a double version that casts
  void mult(const std::complex<double> *weights, size_t n_weights,
            std::complex<double> *result, size_t n_targets);

private:
  std::unique_ptr<MHIfgfDLPrivate> d;
};

class MHIfgfCFPrivate;

class ModHelmholtzCF3D {
public:
  typedef Eigen::Array<PointScalar, 3, Eigen::Dynamic> PointArray;

  ModHelmholtzCF3D(std::complex<RealScalar> waveNumber, size_t leafSize,
                   size_t order, size_t n_elem = 1, PointScalar tol = -1,
                   double maxk = -1, double minSigma = -1);
  ~ModHelmholtzCF3D();

  // normals are the per-source normals: DIM*n_normals, laid out like srcs,
  // and n_normals must equal n_srcs
  void init(const PointScalar *srcs, size_t n_srcs, const PointScalar *targets,
            size_t n_targets, const PointScalar *normals, size_t n_normals);

  void mult(const std::complex<float> *weights, size_t n_weights,
            std::complex<float> *result, size_t n_targets);

  // for convenience also provide a double version that casts
  void mult(const std::complex<double> *weights, size_t n_weights,
            std::complex<double> *result, size_t n_targets);

private:
  std::unique_ptr<MHIfgfCFPrivate> d;
};

} // namespace ifgf

#endif