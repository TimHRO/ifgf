#include "ifgf_library.hpp"

#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <iostream>

#include "config.hpp"
#include "helmholtz_operators.hpp"
#include "ifgfoperator.hpp"
#include "octree.hpp"

namespace ifgf {

namespace {

typedef Eigen::Array<PointScalar, 3, Eigen::Dynamic> PointArray;

template <typename Coord>
static inline PointArray make_points(const Coord* p, size_t n)
{
    Eigen::Map<const Eigen::Array<Coord, 3, Eigen::Dynamic>> m(p, 3, n);
    return m.template cast<PointScalar>();  // no-op copy if Coord == PointScalar
}

// Public wavenumber is std::complex<double>; internal is std::complex<RealScalar>
static inline std::complex<RealScalar> to_internal_k(std::complex<double> k)
{
    return std::complex<RealScalar>(RealScalar(k.real()), RealScalar(k.imag()));
}




// operators without normals
struct OpNoNormals {
    virtual ~OpNoNormals() = default;
    virtual void init(const Eigen::Ref<const PointArray>& srcs,
                      const Eigen::Ref<const PointArray>& targets) = 0;
    virtual void mult(const std::complex<RealScalar>* w, size_t nw,
                      std::complex<RealScalar>* out, size_t nt) = 0;
};

template <typename Op>
struct OpNoNormalsImpl final : OpNoNormals {
    Op op;

    template <typename... Args>
    explicit OpNoNormalsImpl(Args&&... args) : op(std::forward<Args>(args)...) {}

    void init(const Eigen::Ref<const PointArray>& srcs,
              const Eigen::Ref<const PointArray>& targets) override
    {
        op.init(PointArray(srcs), PointArray(targets));
    }

    void mult(const std::complex<RealScalar>* w, size_t nw,
              std::complex<RealScalar>* out, size_t nt) override
    {
        Eigen::Map<const Eigen::Vector<std::complex<RealScalar>, Eigen::Dynamic>>
            e_w(w, nw);
        Eigen::Map<Eigen::Array<std::complex<RealScalar>, Eigen::Dynamic, 1>>
            e_out(out, nt);
        e_out = op.mult(e_w);
    }
};

// operators with normals
struct OpWithNormals {
    virtual ~OpWithNormals() = default;
    virtual void init(const Eigen::Ref<const PointArray>& srcs,
                      const Eigen::Ref<const PointArray>& targets,
                      const Eigen::Ref<const PointArray>& normals) = 0;
    virtual void mult(const std::complex<RealScalar>* w, size_t nw,
                      std::complex<RealScalar>* out, size_t nt) = 0;
};

template <typename Op>
struct OpWithNormalsImpl final : OpWithNormals {
    Op op;

    template <typename... Args>
    explicit OpWithNormalsImpl(Args&&... args) : op(std::forward<Args>(args)...) {}

    void init(const Eigen::Ref<const PointArray>& srcs,
              const Eigen::Ref<const PointArray>& targets,
              const Eigen::Ref<const PointArray>& normals) override
    {
        op.init(PointArray(srcs), PointArray(targets), PointArray(normals));
    }

    void mult(const std::complex<RealScalar>* w, size_t nw,
              std::complex<RealScalar>* out, size_t nt) override
    {
        Eigen::Map<const Eigen::Vector<std::complex<RealScalar>, Eigen::Dynamic>>
            e_w(w, nw);
        Eigen::Map<Eigen::Array<std::complex<RealScalar>, Eigen::Dynamic, 1>>
            e_out(out, nt);
        e_out = op.mult(e_w);
    }
};

// A real wavenumber means no decay, reduce instantiation
// TODO keep this or remove decayisImagPart since it is always
inline bool isPurelyOscillatory(const std::complex<double>& k,
                                bool decayIsImagPart)
{
    return decayIsImagPart ? (k.imag() == 0.0)
                           : (k.real() == 0.0);
}

// what if ngsolve uses double weights and results but ifgf should use RealScalar float
// mult_cast does internal cast
template <typename Holder, typename Scalar>
inline void mult_cast(Holder& h, const std::complex<Scalar>* weights,
                      size_t n_weights, std::complex<Scalar>* result,
                      size_t n_targets)
{
    Eigen::Map<const Eigen::Array<std::complex<Scalar>, Eigen::Dynamic, 1>>
        e_weights(weights, n_weights);

    Eigen::Array<std::complex<RealScalar>, Eigen::Dynamic, 1> in =
        e_weights.template cast<std::complex<RealScalar>>();
    Eigen::Array<std::complex<RealScalar>, Eigen::Dynamic, 1> out(n_targets);

    h->mult(in.data(), n_weights, out.data(), n_targets);

    Eigen::Map<Eigen::Array<std::complex<Scalar>, Eigen::Dynamic, 1>>
        e_res(result, n_targets);
    e_res = out.template cast<std::complex<Scalar>>();
}

} // namespace


// define mult here, its the same for all operators
// provide double and float API mult
#define IFGF_DEFINE_MULT(Class)                                               \
    void Class::mult(const std::complex<float>* weights, size_t n_weights,    \
                     std::complex<float>* result, size_t n_targets)           \
    {                                                                         \
        mult_cast(d->ptr, weights, n_weights, result, n_targets);             \
    }                                                                         \
    void Class::mult(const std::complex<double>* weights, size_t n_weights,   \
                     std::complex<double>* result, size_t n_targets)          \
    {                                                                         \
        mult_cast(d->ptr, weights, n_weights, result, n_targets);             \
    }

// define init (no normals) for both double and float coordinates
// make_points<Coord> casts to PointScalar
#define IFGF_DEFINE_INIT(Class)                                               \
    void Class::init(const double* srcs, size_t n_srcs,                       \
                     const double* targets, size_t n_targets)                 \
    {                                                                         \
        d->ptr->init(make_points(srcs, n_srcs),                              \
                     make_points(targets, n_targets));                        \
    }                                                                         \
    void Class::init(const float* srcs, size_t n_srcs,                        \
                     const float* targets, size_t n_targets)                  \
    {                                                                         \
        d->ptr->init(make_points(srcs, n_srcs),                              \
                     make_points(targets, n_targets));                        \
    }

// define init (with normals) for both double and float coordinates
#define IFGF_DEFINE_INIT_NORMALS(Class, what)                                 \
    void Class::init(const double* srcs, size_t n_srcs,                       \
                     const double* targets, size_t n_targets,                 \
                     const double* normals, size_t n_normals)                 \
    {                                                                         \
        assert(n_normals == n_srcs && what ": need one normal per source");   \
        d->ptr->init(make_points(srcs, n_srcs),                              \
                     make_points(targets, n_targets),                         \
                     make_points(normals, n_normals));                        \
    }                                                                         \
    void Class::init(const float* srcs, size_t n_srcs,                        \
                     const float* targets, size_t n_targets,                  \
                     const float* normals, size_t n_normals)                  \
    {                                                                         \
        assert(n_normals == n_srcs && what ": need one normal per source");   \
        d->ptr->init(make_points(srcs, n_srcs),                              \
                     make_points(targets, n_targets),                         \
                     make_points(normals, n_normals));                        \
    }



class HelmholtzSLPrivate {
public:
    std::unique_ptr<OpNoNormals> ptr;
};

HelmholtzSL3D::HelmholtzSL3D(std::complex<double> waveNumber,
                             size_t leafSize, size_t order, size_t n_elem,
                             double tol, double maxk, double minSigma)
{
    using ifgf_operators::ModifiedHelmholtz;

    d = std::make_unique<HelmholtzSLPrivate>();

    const std::complex<RealScalar> k = to_internal_k(waveNumber);

    if (isPurelyOscillatory(waveNumber, /*decayIsImagPart=*/true)) {
        d->ptr = std::make_unique<
            OpNoNormalsImpl<ModifiedHelmholtz<3, false>>>(
                k, leafSize, order, n_elem, PointScalar(tol), maxk, minSigma);
    } else {
        d->ptr = std::make_unique<
            OpNoNormalsImpl<ModifiedHelmholtz<3, true>>>(
                k, leafSize, order, n_elem, PointScalar(tol), maxk, minSigma);
    }
}

HelmholtzSL3D::HelmholtzSL3D(double waveNumber, size_t leafSize,
                             size_t order, size_t n_elem, double tol,
                             double maxk, double minSigma)
    : HelmholtzSL3D(std::complex<double>(0.0, waveNumber),
                    leafSize, order, n_elem, tol, maxk, minSigma)
{

}

HelmholtzSL3D::~HelmholtzSL3D() {}

IFGF_DEFINE_INIT(HelmholtzSL3D)

IFGF_DEFINE_MULT(HelmholtzSL3D)



class HelmholtzDLPrivate {
public:
    std::unique_ptr<OpWithNormals> ptr;
};

HelmholtzDL3D::HelmholtzDL3D(std::complex<double> waveNumber,
                             size_t leafSize, size_t order, size_t n_elem,
                             double tol, double maxk, double minSigma)
{
    using ifgf_operators::DoubleLayerHelmholtz;

    d = std::make_unique<HelmholtzDLPrivate>();

    const std::complex<RealScalar> k = to_internal_k(waveNumber);

    if (isPurelyOscillatory(waveNumber, /*decayIsImagPart=*/true)) {
        d->ptr = std::make_unique<
            OpWithNormalsImpl<DoubleLayerHelmholtz<3, false>>>(
                k, leafSize, order, n_elem, PointScalar(tol), maxk, minSigma);
    } else {
        d->ptr = std::make_unique<
            OpWithNormalsImpl<DoubleLayerHelmholtz<3, true>>>(
                k, leafSize, order, n_elem, PointScalar(tol), maxk, minSigma);
    }
}

HelmholtzDL3D::HelmholtzDL3D(double waveNumber, size_t leafSize,
                             size_t order, size_t n_elem, double tol,
                             double maxk, double minSigma)
    : HelmholtzDL3D(std::complex<double>(0.0, waveNumber),
                    leafSize, order, n_elem, tol, maxk, minSigma)
{
}

HelmholtzDL3D::~HelmholtzDL3D() {}

IFGF_DEFINE_INIT_NORMALS(HelmholtzDL3D, "double layer")

IFGF_DEFINE_MULT(HelmholtzDL3D)



class HelmholtzCFPrivate {
public:
    std::unique_ptr<OpWithNormals> ptr;
};

HelmholtzCF3D::HelmholtzCF3D(std::complex<double> waveNumber,
                             size_t leafSize, size_t order, size_t n_elem,
                             double tol, double maxk, double minSigma)
{
    using ifgf_operators::CombinedFieldHelmholtz;

    d = std::make_unique<HelmholtzCFPrivate>();

    const std::complex<RealScalar> k = to_internal_k(waveNumber);

    if (isPurelyOscillatory(waveNumber, /*decayIsImagPart=*/true)) {
        d->ptr = std::make_unique<
            OpWithNormalsImpl<CombinedFieldHelmholtz<3, false>>>(
                k, leafSize, order, n_elem, PointScalar(tol), maxk, minSigma);
    } else {
        d->ptr = std::make_unique<
            OpWithNormalsImpl<CombinedFieldHelmholtz<3, true>>>(
                k, leafSize, order, n_elem, PointScalar(tol), maxk, minSigma);
    }
}

HelmholtzCF3D::HelmholtzCF3D(double waveNumber, size_t leafSize,
                             size_t order, size_t n_elem, double tol,
                             double maxk, double minSigma)
    : HelmholtzCF3D(std::complex<double>(waveNumber, 0.0),
                    leafSize, order, n_elem, tol, maxk, minSigma)
{

}

HelmholtzCF3D::~HelmholtzCF3D() {}

IFGF_DEFINE_INIT_NORMALS(HelmholtzCF3D, "combined field")

IFGF_DEFINE_MULT(HelmholtzCF3D)

#undef IFGF_DEFINE_MULT
#undef IFGF_DEFINE_INIT
#undef IFGF_DEFINE_INIT_NORMALS

} // namespace ifgf
