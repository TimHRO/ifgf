#include "ifgf_library.hpp"

#include <Eigen/Dense>
#include <iostream>

#include <cmath>

#include <oneapi/tbb/blocked_range.h>
#include <random>
#include <cstdlib>
#include <tbb/task_arena.h>
#include <tbb/global_control.h>
#include <fenv.h>
#include <chrono>

#include "config.hpp"
#include "helmholtz_ifgf.hpp"
#include "ifgfoperator.hpp"
#include "octree.hpp"



HelmholtzIfgfOperator3d::HelmholtzIfgfOperator3d(RealScalar waveNumber,
						 size_t leafSize,
						 size_t order,
						 size_t n_elem,PointScalar tol) :
    HelmholtzIfgfOperator<3>(waveNumber,leafSize,order,n_elem,tol)
{
    
}

HelmholtzIfgfOperator3d::~HelmholtzIfgfOperator3d()
{

}


void HelmholtzIfgfOperator3d::init(const PointScalar* srcs, size_t n_srcs ,const PointScalar* targets, size_t n_targets)
{
    std::cout<<"init!"<<std::endl;
    Eigen::Map<const Eigen::Array<PointScalar, 3, Eigen::Dynamic> > e_srcs(srcs, 3,n_srcs);
    Eigen::Map<const Eigen::Array<PointScalar, 3, Eigen::Dynamic> > e_targets(targets, 3,n_targets);
    
    HelmholtzIfgfOperator<3>::init(e_srcs,e_targets);	
}

void HelmholtzIfgfOperator3d::mult(const Complex* weights, size_t n_weights,Complex* result, size_t n_targets)
{
    Eigen::Map<const Eigen::Array<Complex, Eigen::Dynamic, 1> > e_weights(weights,n_weights);
    
    Eigen::Map<Eigen::Array<Complex, Eigen::Dynamic, 1>  > e_res(result,n_targets);
	
    e_res=HelmholtzIfgfOperator<3>::mult(e_weights);	
}


//for convenience also provide a double version that casts
void HelmholtzIfgfOperator3d::mult(const std::complex<double>* weights, size_t n_weights,std::complex<double>* result, size_t n_targets)
{
    Eigen::Map<const Eigen::Array<std::complex<double>, Eigen::Dynamic, 1> > e_weights(weights,n_weights);
    
    Eigen::Map<Eigen::Array<std::complex<double>, Eigen::Dynamic, 1>  > e_res(result,n_targets);

    e_res=HelmholtzIfgfOperator<3>::mult(e_weights.template cast<Complex>()).template cast<std::complex<double> >();
	
}   
