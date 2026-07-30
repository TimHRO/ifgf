#ifndef __IFGF_LIBRARY__
#define __IFGF_LIBRARY__

#include <complex>
#include <cstddef>
#include <memory>

// ---------------------------------------------------------------------------
// The API speaks only in double: wavenumbers, tol, and
// point coordinates are all double.
// The only exception are float weights that do not get promoted to double.
// The internal compute precision is not exposed here. It can be set in the
// config and the kernel file.
// It is possible to set RealScalar=float and still call every function
// with double arguments, the conversion happens inside.
// ---------------------------------------------------------------------------

namespace ifgf {

// provide float path for weights that does not promote to double
#define IFGF_DECLARE_MULT()                                                   \
    void mult(const std::complex<float>* weights, size_t n_weights,           \
              std::complex<float>* result, size_t n_targets);                 \
    void mult(const std::complex<double>* weights, size_t n_weights,          \
              std::complex<double>* result, size_t n_targets)


class HelmholtzSLPrivate;

class HelmholtzSL3D {
public:
    HelmholtzSL3D(std::complex<double> waveNumber, size_t leafSize,
                  size_t order, size_t n_elem = 1, double tol = -1,
                  double maxk = -1, double minSigma = -1);

    HelmholtzSL3D(double waveNumber, size_t leafSize,
                  size_t order, size_t n_elem = 1, double tol = -1,
                  double maxk = -1, double minSigma = -1);

    ~HelmholtzSL3D();

    // Coordinates are assumed to be always double and DIM-major (x0,y0,z0, x1,y1,z1, ...)!!!
    void init(const double* srcs, size_t n_srcs,
              const double* targets, size_t n_targets);

    IFGF_DECLARE_MULT();

private:
    std::unique_ptr<HelmholtzSLPrivate> d;
};


class HelmholtzDLPrivate;

class HelmholtzDL3D {
public:
    HelmholtzDL3D(std::complex<double> waveNumber, size_t leafSize,
                  size_t order, size_t n_elem = 1, double tol = -1,
                  double maxk = -1, double minSigma = -1);

    HelmholtzDL3D(double waveNumber, size_t leafSize,
                  size_t order, size_t n_elem = 1, double tol = -1,
                  double maxk = -1, double minSigma = -1);

    ~HelmholtzDL3D();

    void init(const double* srcs, size_t n_srcs,
              const double* targets, size_t n_targets,
              const double* normals, size_t n_normals);

    IFGF_DECLARE_MULT();

private:
    std::unique_ptr<HelmholtzDLPrivate> d;
};


class HelmholtzCFPrivate;

class HelmholtzCF3D {
public:
    HelmholtzCF3D(std::complex<double> waveNumber, size_t leafSize,
                  size_t order, size_t n_elem = 1, double tol = -1,
                  double maxk = -1, double minSigma = -1);

    HelmholtzCF3D(double waveNumber, size_t leafSize,
                  size_t order, size_t n_elem = 1, double tol = -1,
                  double maxk = -1, double minSigma = -1);

    ~HelmholtzCF3D();

    void init(const double* srcs, size_t n_srcs,
              const double* targets, size_t n_targets,
              const double* normals, size_t n_normals);

    IFGF_DECLARE_MULT();

private:
    std::unique_ptr<HelmholtzCFPrivate> d;
};

#undef IFGF_DECLARE_MULT

} // namespace ifgf

#endif