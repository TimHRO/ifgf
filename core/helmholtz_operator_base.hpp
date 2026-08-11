#ifndef __HELMHOLTZ_OPERATOR_BASE_HPP__
#define __HELMHOLTZ_OPERATOR_BASE_HPP__

#include <cmath>
#include <functional>
#include "helmholtz_kernels.hpp"
#include "ifgfoperator.hpp"
#include "util.hpp"

namespace ifgf_kernels {

template <size_t dim, bool HasNormals>
struct NormalsStorage;

template <size_t dim>
struct NormalsStorage<dim, false>
{

};

template <size_t dim>
struct NormalsStorage<dim, true>
{
    typedef Eigen::Array<PointScalar, dim, Eigen::Dynamic> PointArray;
    PointArray m_normals;
};


template <typename Derived, size_t dim, bool HasNormals>
class HelmholtzOperatorBase
    : public IfgfOperator<std::complex<RealScalar>, dim, 1, Derived>,
      protected NormalsStorage<dim, HasNormals>
{
protected:
    struct OctreeKeyType {
        RealScalar maxk;
        RealScalar minSigma;
        size_t     Ndof;
        size_t     Ndof2;

        auto operator==(const OctreeKeyType& other) const
        {
            return std::abs(maxk - other.maxk) < 1e-12
                && Ndof  == other.Ndof
                && Ndof2 == other.Ndof2
                && std::abs(minSigma - other.minSigma) < 1e-12;
        }
    };

public:
    static constexpr bool HAS_NORMALS = HasNormals;

    typedef std::complex<RealScalar>                       T;
    typedef Eigen::Array<PointScalar, dim, Eigen::Dynamic> PointArray;
    typedef Eigen::Vector<PointScalar, dim>                Point;

    typedef IfgfOperator<std::complex<RealScalar>, dim, 1, Derived> BaseOp;

    HelmholtzOperatorBase(std::complex<RealScalar> waveNumber,
                          size_t leafSize,
                          size_t order,
                          size_t n_elem,
                          PointScalar tol,
                          RealScalar p_maxk,
                          RealScalar p_minSigma)
        : BaseOp(leafSize, order, n_elem, tol),
          k(waveNumber),
          maxk(p_maxk),
          minSigma(p_minSigma)
    {

    }

    template <bool B = HasNormals,
              typename = std::enable_if_t<!B>>
    void init(const PointArray& srcs, const PointArray targets)
    {
        initCommon(srcs, targets);
    }

    template <bool B = HasNormals,
              typename = std::enable_if_t<B>>
    void init(const PointArray& srcs, const PointArray targets,
              const PointArray& normals)
    {
        this->m_normals = normals;
        initCommon(srcs, targets);
    }

    // Once the octree is ready, reorder normals so Morton order is observed (same as for src)
    void onOctreeReady()
    {
        if constexpr (HasNormals) {
            PointArray sorted(dim, this->m_normals.cols());
            Util::copy_with_permutation_colwise<PointScalar, dim>(
                this->m_normals, this->m_octree->srcPermutation(), sorted);
            this->m_normals = sorted;
        }
    }

    // The base uses this (only when HAS_NORMALS) to build the device buffer
    const PointScalar* sourceNormalsData() const
    {
        if constexpr (HasNormals) {
            return this->m_normals.data();
        } else {
            return nullptr;
        }
    }

    // interpolation parameters
    inline Eigen::Vector<int, dim>
    orderForBox(PointScalar H, Eigen::Vector<int, dim> baseOrder,
                int step = 0) const
    {
        (void)H;
        Eigen::Vector<int, dim> order = baseOrder;

        if (step == 0) {
            order = (baseOrder.array() - 2).cwiseMax(2);
        }
        return order;
    }

    double cutoff_limit(double H)
    {
        (void)H;
        double smin = 1e-4;
        return std::min(smin, sqrt((double)dim) / dim);
    }

    inline Eigen::Vector<size_t, dim>
    elementsForBox(PointScalar H, Eigen::Vector<int, dim> baseOrder,
                   Eigen::Vector<size_t, dim> base, int step = 0) const
    {
        (void)baseOrder;
        Eigen::Vector<size_t, dim> els;

        if (step == 0) {
            base *= 2;
        }

        for (int i = 0; i < (int)dim; i++) {
            PointScalar delta = std::max((PointScalar)(maxk * H), (PointScalar)1.);
            els[i] = std::max(base[i] * ((int)ceil(delta)), (size_t)1);
        }
        return els;
    }

    bool farfieldCanBeSkipped(PointScalar H)
    {
    	return (H * this->k.imag()) > 40.0;
    }

protected:
    void initDefaults(RealScalar defaultMinSigma, RealScalar defaultMaxk)
    {
        if (minSigma < 0) {
            minSigma = defaultMinSigma;
        }
        std::cout << "minSigma=" << minSigma << std::endl;

        if (maxk < 0) {
            maxk = defaultMaxk;
        }
        std::cout << "maxk=" << maxk << std::endl;
    }

    void initCommon(const PointArray& srcs, const PointArray targets)
    {
        OctreeKeyType key;
        key.maxk     = maxk;
        key.Ndof     = srcs.cols();
        key.Ndof2    = targets.cols();
        key.minSigma = minSigma;

        auto oct = OctreeCache<T, dim, OctreeKeyType>::getInstance().find(key);
        if (oct) {
            std::cout << "using cached octree=" << oct << std::endl;
            this->m_octree = oct;
        }

	// CutOff in build interaction list
        std::function<bool(double)> cutOff;
        const double kim = static_cast<double>(this->k.imag());
        const double tol = static_cast<double>(this->tolerance());
        if (tol > 0.0 && kim > 0.0) {
            cutOff = [kim, tol](double dist) {
                return std::exp(-dist * kim) < tol;
            };
        } else {
            cutOff = [](double) { return false; };
        }

        BaseOp::init(srcs, targets, cutOff);

	#ifdef CACHE_OCTREE
        OctreeCache<T, dim, OctreeKeyType>::getInstance().add(key, this->m_octree);
	#endif
    }


    std::complex<RealScalar> k;
    RealScalar               maxk;
    RealScalar               minSigma;
};

} // namespace ifgf_kernels

#endif
