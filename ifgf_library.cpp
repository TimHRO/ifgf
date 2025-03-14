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
#include "modified_helmholtz_ifgf.hpp"
#include "ifgfoperator.hpp"
#include "octree.hpp"


class HIfgfPrivate {
public:
    std::unique_ptr<HelmholtzIfgfOperator<3> >  ptr;
};


HelmholtzIfgfOperator3d::HelmholtzIfgfOperator3d(RealScalar waveNumber,
						 size_t leafSize,
						 size_t order,
						 size_t n_elem,PointScalar tol)
{
    d=std::make_unique<HIfgfPrivate>();
    d->ptr=std::make_unique<HelmholtzIfgfOperator<3> >(waveNumber,leafSize,order,n_elem,tol);
    
}

HelmholtzIfgfOperator3d::~HelmholtzIfgfOperator3d()
{

}


void HelmholtzIfgfOperator3d::init(const PointScalar* srcs, size_t n_srcs ,const PointScalar* targets, size_t n_targets)
{
    std::cout<<"init!"<<std::endl;
    Eigen::Map<const Eigen::Array<PointScalar, 3, Eigen::Dynamic> > e_srcs(srcs, 3,n_srcs);
    Eigen::Map<const Eigen::Array<PointScalar, 3, Eigen::Dynamic> > e_targets(targets, 3,n_targets);
    
    d->ptr->init(e_srcs,e_targets);	
}

void HelmholtzIfgfOperator3d::mult(const std::complex<float>* weights, size_t n_weights,std::complex<float>* result, size_t n_targets)
{
    Eigen::Map<const Eigen::Array<std::complex<float>, Eigen::Dynamic, 1> > e_weights(weights,n_weights);
    
    Eigen::Map<Eigen::Array<std::complex<float>, Eigen::Dynamic, 1>  > e_res(result,n_targets);
	
    e_res=d->ptr->mult(e_weights.template cast<std::complex<RealScalar> >()).template cast<std::complex<float> >();	
}


//for convenience also provide a double version that casts
void HelmholtzIfgfOperator3d::mult(const std::complex<double>* weights, size_t n_weights,std::complex<double>* result, size_t n_targets)
{
    Eigen::Map<const Eigen::Array<std::complex<double>, Eigen::Dynamic, 1> > e_weights(weights,n_weights);
    
    Eigen::Map<Eigen::Array<std::complex<double>, Eigen::Dynamic, 1>  > e_res(result,n_targets);

    e_res=d->ptr->mult(e_weights.template cast<std::complex<RealScalar>>()).template cast<std::complex<double> >();
	
}



class MHIfgfPrivate {
public:	
    std::unique_ptr<ModifiedHelmholtzIfgfOperator<3> > ptr;
};


ModifiedHelmholtzIfgfOperator3d::ModifiedHelmholtzIfgfOperator3d(std::complex<RealScalar> waveNumber,
						 size_t leafSize,
						 size_t order,
								 size_t n_elem,PointScalar tol,double maxk)
{
    d=std::make_unique<MHIfgfPrivate>();
    d->ptr=std::make_unique<ModifiedHelmholtzIfgfOperator<3> >(waveNumber,leafSize,order,n_elem,tol,maxk);
    
}

ModifiedHelmholtzIfgfOperator3d::~ModifiedHelmholtzIfgfOperator3d()
{
}


void ModifiedHelmholtzIfgfOperator3d::init(const PointScalar* srcs, size_t n_srcs ,const PointScalar* targets, size_t n_targets)
{
    std::cout<<"init!"<<std::endl;
    Eigen::Map<const Eigen::Array<PointScalar, 3, Eigen::Dynamic> > e_srcs(srcs, 3,n_srcs);
    Eigen::Map<const Eigen::Array<PointScalar, 3, Eigen::Dynamic> > e_targets(targets, 3,n_targets);
    
    d->ptr->init(e_srcs,e_targets);	
}

void ModifiedHelmholtzIfgfOperator3d::mult(const std::complex<float>* weights, size_t n_weights,std::complex<float>* result, size_t n_targets)
{
    Eigen::Map<const Eigen::Vector<std::complex<float>, Eigen::Dynamic> > e_weights(weights,n_weights);
    
    Eigen::Map<Eigen::Vector<std::complex<float>, Eigen::Dynamic>  > e_res(result,n_targets);
	
    e_res=d->ptr->mult(e_weights.template cast<std::complex<RealScalar> >()).template cast<std::complex<float> >();	
}


//for convenience also provide a double version that casts
void ModifiedHelmholtzIfgfOperator3d::mult(const std::complex<double>* weights, size_t n_weights,std::complex<double>* result, size_t n_targets)
{
    Eigen::Map<const Eigen::Array<std::complex<double>, Eigen::Dynamic, 1> > e_weights(weights,n_weights);
    
    Eigen::Map<Eigen::Array<std::complex<double>, Eigen::Dynamic, 1>  > e_res(result,n_targets);

    e_res=d->ptr->mult(e_weights.template cast<std::complex<RealScalar> >()).template cast<std::complex<double> >();
	
}   


