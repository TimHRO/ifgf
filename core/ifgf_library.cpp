#include "ifgf_library.hpp"

#include <Eigen/Dense>
#include <iostream>

#include <cmath>

#include <cassert>
#include <chrono>
#include <cstdlib>
#include <fenv.h>
#include <oneapi/tbb/blocked_range.h>
#include <random>
#include <tbb/global_control.h>
#include <tbb/task_arena.h>

#include "combined_field_helmholtz_ifgf.hpp"
#include "config.hpp"
#include "double_layer_helmholtz_ifgf.hpp"
#include "helmholtz_ifgf.hpp"
#include "ifgfoperator.hpp"
#include "modified_helmholtz_ifgf.hpp"
#include "octree.hpp"

namespace ifgf {

class HIfgfSLPrivate {
public:
  std::unique_ptr<HelmholtzIfgfOperator<3>> ptr;
};

HelmholtzSL3D::HelmholtzSL3D(RealScalar waveNumber, size_t leafSize,
                             size_t order, size_t n_elem, PointScalar tol) {
  d = std::make_unique<HIfgfSLPrivate>();
  d->ptr = std::make_unique<HelmholtzIfgfOperator<3>>(waveNumber, leafSize,
                                                      order, n_elem, tol);
}

HelmholtzSL3D::~HelmholtzSL3D() {}

void HelmholtzSL3D::init(const PointScalar *srcs, size_t n_srcs,
                         const PointScalar *targets, size_t n_targets) {
  std::cout << "init!" << std::endl;
  Eigen::Map<const PointArray> e_srcs(srcs, 3, n_srcs);
  Eigen::Map<const PointArray> e_targets(targets, 3, n_targets);

  d->ptr->init(e_srcs, e_targets);
}

void HelmholtzSL3D::mult(const std::complex<float> *weights, size_t n_weights,
                         std::complex<float> *result, size_t n_targets) {
  Eigen::Map<const Eigen::Array<std::complex<float>, Eigen::Dynamic, 1>>
      e_weights(weights, n_weights);

  Eigen::Map<Eigen::Array<std::complex<float>, Eigen::Dynamic, 1>> e_res(
      result, n_targets);

  e_res = d->ptr->mult(e_weights.template cast<std::complex<RealScalar>>())
              .template cast<std::complex<float>>();
}

// for convenience also provide a double version that casts
void HelmholtzSL3D::mult(const std::complex<double> *weights, size_t n_weights,
                         std::complex<double> *result, size_t n_targets) {
  Eigen::Map<const Eigen::Array<std::complex<double>, Eigen::Dynamic, 1>>
      e_weights(weights, n_weights);

  Eigen::Map<Eigen::Array<std::complex<double>, Eigen::Dynamic, 1>> e_res(
      result, n_targets);

  e_res = d->ptr->mult(e_weights.template cast<std::complex<RealScalar>>())
              .template cast<std::complex<double>>();
}

class MHIfgfSLPrivate {
public:
  std::unique_ptr<ModifiedHelmholtzIfgfOperator<3>> ptr;
};

ModHelmholtzSL3D::ModHelmholtzSL3D(std::complex<RealScalar> waveNumber,
                                   size_t leafSize, size_t order, size_t n_elem,
                                   PointScalar tol, double maxk,
                                   double minSigma) {
  d = std::make_unique<MHIfgfSLPrivate>();
  d->ptr = std::make_unique<ModifiedHelmholtzIfgfOperator<3>>(
      waveNumber, leafSize, order, n_elem, tol, maxk, minSigma);
}

ModHelmholtzSL3D::~ModHelmholtzSL3D() {}

void ModHelmholtzSL3D::init(const PointScalar *srcs, size_t n_srcs,
                            const PointScalar *targets, size_t n_targets) {
  std::cout << "init!" << std::endl;
  Eigen::Map<const PointArray> e_srcs(srcs, 3, n_srcs);
  Eigen::Map<const PointArray> e_targets(targets, 3, n_targets);

  d->ptr->init(e_srcs, e_targets);
}

void ModHelmholtzSL3D::mult(const std::complex<float> *weights,
                            size_t n_weights, std::complex<float> *result,
                            size_t n_targets) {
  Eigen::Map<const Eigen::Vector<std::complex<float>, Eigen::Dynamic>>
      e_weights(weights, n_weights);

  Eigen::Map<Eigen::Vector<std::complex<float>, Eigen::Dynamic>> e_res(
      result, n_targets);

  e_res = d->ptr->mult(e_weights.template cast<std::complex<RealScalar>>())
              .template cast<std::complex<float>>();
}

// for convenience also provide a double version that casts
void ModHelmholtzSL3D::mult(const std::complex<double> *weights,
                            size_t n_weights, std::complex<double> *result,
                            size_t n_targets) {
  Eigen::Map<const Eigen::Array<std::complex<double>, Eigen::Dynamic, 1>>
      e_weights(weights, n_weights);

  Eigen::Map<Eigen::Array<std::complex<double>, Eigen::Dynamic, 1>> e_res(
      result, n_targets);

  e_res = d->ptr->mult(e_weights.template cast<std::complex<RealScalar>>())
              .template cast<std::complex<double>>();
}

class MHIfgfDLPrivate {
public:
  std::unique_ptr<DoubleLayerHelmholtzIfgfOperator<3>> ptr;
};

ModHelmholtzDL3D::ModHelmholtzDL3D(std::complex<RealScalar> waveNumber,
                                   size_t leafSize, size_t order, size_t n_elem,
                                   PointScalar tol, double maxk,
                                   double minSigma) {
  d = std::make_unique<MHIfgfDLPrivate>();
  d->ptr = std::make_unique<DoubleLayerHelmholtzIfgfOperator<3>>(
      waveNumber, leafSize, order, n_elem, tol, maxk, minSigma);
}

ModHelmholtzDL3D::~ModHelmholtzDL3D() {}

void ModHelmholtzDL3D::init(const PointScalar *srcs, size_t n_srcs,
                            const PointScalar *targets, size_t n_targets,
                            const PointScalar *normals, size_t n_normals) {
  std::cout << "init!" << std::endl;
  // the kernel needs one normal per source
  assert(n_normals == n_srcs && "double layer: need one normal per source");

  Eigen::Map<const PointArray> e_srcs(srcs, 3, n_srcs);
  Eigen::Map<const PointArray> e_targets(targets, 3, n_targets);
  Eigen::Map<const PointArray> e_normals(normals, 3, n_normals);

  d->ptr->init(e_srcs, e_targets, e_normals);
}

void ModHelmholtzDL3D::mult(const std::complex<float> *weights,
                            size_t n_weights, std::complex<float> *result,
                            size_t n_targets) {
  Eigen::Map<const Eigen::Vector<std::complex<float>, Eigen::Dynamic>>
      e_weights(weights, n_weights);

  Eigen::Map<Eigen::Vector<std::complex<float>, Eigen::Dynamic>> e_res(
      result, n_targets);

  e_res = d->ptr->mult(e_weights.template cast<std::complex<RealScalar>>())
              .template cast<std::complex<float>>();
}

// for convenience also provide a double version that casts
void ModHelmholtzDL3D::mult(const std::complex<double> *weights,
                            size_t n_weights, std::complex<double> *result,
                            size_t n_targets) {
  Eigen::Map<const Eigen::Array<std::complex<double>, Eigen::Dynamic, 1>>
      e_weights(weights, n_weights);

  Eigen::Map<Eigen::Array<std::complex<double>, Eigen::Dynamic, 1>> e_res(
      result, n_targets);

  e_res = d->ptr->mult(e_weights.template cast<std::complex<RealScalar>>())
              .template cast<std::complex<double>>();
}

class MHIfgfCFPrivate {
public:
  std::unique_ptr<CombinedFieldHelmholtzIfgfOperator<3>> ptr;
};

ModHelmholtzCF3D::ModHelmholtzCF3D(std::complex<RealScalar> waveNumber,
                                   size_t leafSize, size_t order, size_t n_elem,
                                   PointScalar tol, double maxk,
                                   double minSigma) {
  d = std::make_unique<MHIfgfCFPrivate>();
  d->ptr = std::make_unique<CombinedFieldHelmholtzIfgfOperator<3>>(
      waveNumber, leafSize, order, n_elem, tol, maxk, minSigma);
}

ModHelmholtzCF3D::~ModHelmholtzCF3D() {}

void ModHelmholtzCF3D::init(const PointScalar *srcs, size_t n_srcs,
                            const PointScalar *targets, size_t n_targets,
                            const PointScalar *normals, size_t n_normals) {
  std::cout << "init!" << std::endl;
  // the kernel needs one normal per source
  assert(n_normals == n_srcs && "combined field: need one normal per source");

  Eigen::Map<const PointArray> e_srcs(srcs, 3, n_srcs);
  Eigen::Map<const PointArray> e_targets(targets, 3, n_targets);
  Eigen::Map<const PointArray> e_normals(normals, 3, n_normals);

  d->ptr->init(e_srcs, e_targets, e_normals);
}

void ModHelmholtzCF3D::mult(const std::complex<float> *weights,
                            size_t n_weights, std::complex<float> *result,
                            size_t n_targets) {
  Eigen::Map<const Eigen::Vector<std::complex<float>, Eigen::Dynamic>>
      e_weights(weights, n_weights);

  Eigen::Map<Eigen::Vector<std::complex<float>, Eigen::Dynamic>> e_res(
      result, n_targets);

  e_res = d->ptr->mult(e_weights.template cast<std::complex<RealScalar>>())
              .template cast<std::complex<float>>();
}

// for convenience also provide a double version that casts
void ModHelmholtzCF3D::mult(const std::complex<double> *weights,
                            size_t n_weights, std::complex<double> *result,
                            size_t n_targets) {
  Eigen::Map<const Eigen::Array<std::complex<double>, Eigen::Dynamic, 1>>
      e_weights(weights, n_weights);

  Eigen::Map<Eigen::Array<std::complex<double>, Eigen::Dynamic, 1>> e_res(
      result, n_targets);

  e_res = d->ptr->mult(e_weights.template cast<std::complex<RealScalar>>())
              .template cast<std::complex<double>>();
}

} // namespace ifgf
