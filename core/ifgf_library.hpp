#ifndef __IFGF_LIBRARY__
#define __IFGF_LIBRARY__

#include <complex>
#include <cstddef>
#include <memory>

// ---------------------------------------------------------------------------
// Point coordinates may be passed as either double or float: each init()
// has both overloads, and the coordinates are converted to the internal
// PointScalar inside. Weights likewise have float and double overloads.
// Wavenumbers and tol are still double.
// The internal compute precision is not exposed here. It can be set in the
// config and the kernel file.
// It is possible to set RealScalar=float and still call every function
// with double arguments (or vice versa), the conversion happens inside.
// ---------------------------------------------------------------------------

namespace ifgf {

// provide float path for weights that does not promote to double
#define IFGF_DECLARE_MULT()                                                   \
    void mult(const std::complex<float>* weights, size_t n_weights,           \
              std::complex<float>* result, size_t n_targets);                 \
    void mult(const std::complex<double>* weights, size_t n_weights,          \
              std::complex<double>* result, size_t n_targets)

// provide float and double coordinate paths
#define IFGF_DECLARE_INIT()                                                   \
    void init(const double* srcs, size_t n_srcs,                             \
              const double* targets, size_t n_targets);                       \
    void init(const float* srcs, size_t n_srcs,                              \
              const float* targets, size_t n_targets)

#define IFGF_DECLARE_INIT_NORMALS()                                          \
    void init(const double* srcs, size_t n_srcs,                             \
              const double* targets, size_t n_targets,                        \
              const double* normals, size_t n_normals);                       \
    void init(const float* srcs, size_t n_srcs,                             \
              const float* targets, size_t n_targets,                         \
              const float* normals, size_t n_normals)


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

    // Coordinates are DIM-major (x0,y0,z0, x1,y1,z1, ...) float or double
    IFGF_DECLARE_INIT();

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

    IFGF_DECLARE_INIT_NORMALS();

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

    IFGF_DECLARE_INIT_NORMALS();

    IFGF_DECLARE_MULT();

private:
    std::unique_ptr<HelmholtzCFPrivate> d;
};

#undef IFGF_DECLARE_MULT
#undef IFGF_DECLARE_INIT
#undef IFGF_DECLARE_INIT_NORMALS

} // namespace ifgf

#endif
