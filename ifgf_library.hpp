
#ifndef __IFGF_LIBRARY__
#define __IFGF_LIBRARY__


#include "config.hpp"
#include <complex>




class HIfgfPrivate;

class HelmholtzIfgfOperator3d 
{
public:
    HelmholtzIfgfOperator3d(RealScalar waveNumber,
			    size_t leafSize,
			    size_t order,
			    size_t n_elem=1,PointScalar tol=-1);
    ~HelmholtzIfgfOperator3d();


    void init(const PointScalar* srcs, size_t n_srcs ,const PointScalar* targets, size_t n_targets);

    void mult(const std::complex<float>* weights, size_t n_weights,std::complex<float>* result, size_t n_targets);


    //for convenience also provide a double version that casts
    void mult(const std::complex<double>* weights, size_t n_weights,std::complex<double>* result, size_t n_targets);


private:
    HIfgfPrivate* d;

};


class MHIfgfPrivate;

class ModifiedHelmholtzIfgfOperator3d 
{
public:
    ModifiedHelmholtzIfgfOperator3d(std::complex<RealScalar> waveNumber,
			    size_t leafSize,
			    size_t order,
			    size_t n_elem=1,PointScalar tol=-1);
    ~ModifiedHelmholtzIfgfOperator3d();


    void init(const PointScalar* srcs, size_t n_srcs ,const PointScalar* targets, size_t n_targets);

    void mult(const std::complex<float>* weights, size_t n_weights,std::complex<float>* result, size_t n_targets);


    //for convenience also provide a double version that casts
    void mult(const std::complex<double>* weights, size_t n_weights,std::complex<double>* result, size_t n_targets);


private:
    MHIfgfPrivate* d;

};


#endif
