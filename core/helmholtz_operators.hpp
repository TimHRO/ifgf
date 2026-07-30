#ifndef __HELMHOLTZ_OPERATORS_HPP__
#define __HELMHOLTZ_OPERATORS_HPP__

#include "helmholtz_kernels.hpp"
#include "helmholtz_operator_base.hpp"

namespace ifgf_operators {

using ifgf_kernels::HelmholtzOperatorBase;
using ifgf_kernels::ModifiedHelmholtzKernelFunctions;
using ifgf_kernels::DoubleLayerHelmholtzKernelFunctions;
using ifgf_kernels::CombinedFieldHelmholtzKernelFunctions;

template <size_t dim, bool WithDecay>
class ModifiedHelmholtz
    : public HelmholtzOperatorBase<
                 ModifiedHelmholtz<dim, WithDecay>, dim, false>
{
public:
    typedef HelmholtzOperatorBase<
                ModifiedHelmholtz<dim, WithDecay>, dim, false> Base;
    typedef typename Base::PointArray PointArray;
    typedef typename Base::Point      Point;
    typedef typename Base::T          T;

    ModifiedHelmholtz(std::complex<RealScalar> waveNumber,
                                  size_t leafSize,
                                  size_t order,
                                  size_t n_elem = 1,
                                  PointScalar tol = -1,
                                  RealScalar p_maxk = -1,
                                  RealScalar p_minSigma = -1)
        : Base(waveNumber, leafSize, order, n_elem, tol, p_maxk, p_minSigma)
    {
        this->initDefaults(
            /*MinSigma=*/ this->k.real(),
            /*Maxk=*/     RealScalar(0.3) * std::abs(this->k.imag())
                                 / std::max((RealScalar)1.0, this->k.real()));
    }

    ~ModifiedHelmholtz()
    {
        std::cout << "deleting modified helmholtz ifgf" << std::endl;
    }

    inline ModifiedHelmholtzKernelFunctions<WithDecay> kernelFunctions() const
    {
        return ModifiedHelmholtzKernelFunctions<WithDecay>(this->k);
    }
};

template <size_t dim, bool WithDecay>
class DoubleLayerHelmholtz
    : public HelmholtzOperatorBase<
                 DoubleLayerHelmholtz<dim, WithDecay>, dim, true>
{
public:
    typedef HelmholtzOperatorBase<
                DoubleLayerHelmholtz<dim, WithDecay>, dim, true> Base;
    typedef typename Base::PointArray PointArray;
    typedef typename Base::Point      Point;
    typedef typename Base::T          T;

    DoubleLayerHelmholtz(std::complex<RealScalar> waveNumber,
                                     size_t leafSize,
                                     size_t order,
                                     size_t n_elem = 1,
                                     PointScalar tol = -1,
                                     RealScalar p_maxk = -1,
                                     RealScalar p_minSigma = -1)
        : Base(waveNumber, leafSize, order, n_elem, tol, p_maxk, p_minSigma)
    {
        this->initDefaults(
            /*MinSigma=*/ this->k.real(),
            /*Maxk=*/     RealScalar(0.3) * std::abs(this->k.imag())
                                 / std::max((RealScalar)1.0, this->k.real()));
    }

    ~DoubleLayerHelmholtz()
    {
        std::cout << "deleting DL helmholtz ifgf" << std::endl;
    }

    inline DoubleLayerHelmholtzKernelFunctions<WithDecay> kernelFunctions() const
    {
        return DoubleLayerHelmholtzKernelFunctions<WithDecay>(this->k);
    }
};

template <size_t dim, bool WithDecay>
class CombinedFieldHelmholtz
    : public HelmholtzOperatorBase<
                 CombinedFieldHelmholtz<dim, WithDecay>, dim, true>
{
public:
    typedef HelmholtzOperatorBase<
                CombinedFieldHelmholtz<dim, WithDecay>, dim, true> Base;
    typedef typename Base::PointArray PointArray;
    typedef typename Base::Point      Point;
    typedef typename Base::T          T;

    CombinedFieldHelmholtz(std::complex<RealScalar> waveNumber,
                                       size_t leafSize,
                                       size_t order,
                                       size_t n_elem = 1,
                                       PointScalar tol = -1,
                                       RealScalar p_maxk = -1,
                                       RealScalar p_minSigma = -1)
        : Base(waveNumber, leafSize, order, n_elem, tol, p_maxk, p_minSigma)
    {
        this->initDefaults(
            /*MinSigma=*/ std::abs(this->k.real()),
            /*Maxk=*/     RealScalar(0.3) * std::abs(this->k.imag())
                                 / std::max((RealScalar)1.0, this->k.real()));
    }

    ~CombinedFieldHelmholtz()
    {
        std::cout << "deleting CF helmholtz ifgf" << std::endl;
    }

    inline CombinedFieldHelmholtzKernelFunctions<WithDecay> kernelFunctions() const
    {
        return CombinedFieldHelmholtzKernelFunctions<WithDecay>(this->k);
    }
};

} // namespace ifgf_operators

#endif
