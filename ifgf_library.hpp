
#ifndef __IFGF_LIBRARY__
#define __IFGF_LIBRARY__


#include <Eigen/Dense>

#include "config.hpp"
#include "helmholtz_ifgf.hpp"
#include "ifgfoperator.hpp"
#include "octree.hpp"



typedef std::complex<RealScalar> Complex;

class HelmholtzIfgfOperator3d :  public HelmholtzIfgfOperator<3>
{
public:
    HelmholtzIfgfOperator3d(RealScalar waveNumber,
			    size_t leafSize,
			    size_t order,
			    size_t n_elem=1,PointScalar tol=-1);
    ~HelmholtzIfgfOperator3d();


    void init(const PointScalar* srcs, size_t n_srcs ,const PointScalar* targets, size_t n_targets);

    void mult(const Complex* weights, size_t n_weights,Complex* result, size_t n_targets);


    //for convenience also provide a double version that casts
    void mult(const std::complex<double>* weights, size_t n_weights,std::complex<double>* result, size_t n_targets);

};


#endif
