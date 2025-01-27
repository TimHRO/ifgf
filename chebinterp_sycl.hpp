#ifndef _SYCL_CHEBINTERP_HPP_
#define _SYCL_CHEBINTERP_HPP_

#include "Eigen/src/Core/util/Constants.h"
#include "boundingbox.hpp"
#include <cstddef>
#include <tuple>

#include <fenv.h>
#include <Eigen/Dense>
#include <tbb/parallel_for.h>
#include <exception>
#include <iostream>

#include <tbb/spin_mutex.h>

#include "cone_domain.hpp"

#include <sycl/sycl.hpp>
#include "sycl_helpers.hpp"
//using namespace sycl;

#ifdef USE_NGSOLVE
    #include </home/arieder/devel/install/include/core/ngcore.hpp>
#endif

namespace SyclChebychevInterpolation
{

    template<typename TV, typename  TP>
    auto inline  transformNodesToInterval(TV& nodes, TP min, TP max) 
    {
	const TP a=(max-min)/2.0;
	const TP b=(max+min)/2.0;
	nodes=a*nodes+b;
    }


    template<int DIM>
    inline constexpr size_t max_buffer_size() 
    {
	size_t s=1;
	for(int i=0;i<DIM;i++){
	    s*=12;
	}

	return s;
    }




    template <typename T, int DIM, int DIMX>
    void chebtransform_impl(const sycl::accessor<T> &src,
                       sycl::marray<T, max_buffer_size<DIM>() > & dest,
                       const std::array<int,DIMX>& ns,
		       const sycl::accessor< double,1, sycl::access_mode::read>& cv,
		       size_t offset,
		       size_t dest_offset,
		       size_t cv_offset
		       )
    {
	//evaluate all line-functions
	int nsigma=1;
	for(int i=0;i<DIM-1;i++)
	    nsigma*=ns[i];
	if constexpr(DIM==1) {
	    nsigma=1;
	}
	const int stride=nsigma;


	dest=0;
	assert(ns[DIM-1]<12); //We have a max buffer size since we are working on the stack
	
	const size_t Nd=ns[DIM-1];

	if constexpr(DIM==1) {
	    //do a straight forwad summation of the innermost dimension	    
	    for(size_t idx=0;idx<ns[0];idx++) {
		for(size_t sigma=0;sigma<ns[0];sigma++)  {
		    const double Td=cv[cv_offset+idx*Nd+sigma];
		    //const double Td=cos(idx*M_PI*(2.*sigma+1.)/(2.*ns[DIM-1]));
		    assert(nsigma==1);
		    dest[dest_offset+idx]+=src[offset+sigma]*Td;
		    //dest.segment(idx*stride,nsigma)+=src.segment(sigma*stride,nsigma)*Td;
		}
		dest[dest_offset+idx]*=(idx ==0 ? 1.:2. )*1./ns[0];
	    }
	    //std::cout<<"done inner"<<std::endl;
	}else {
	    std::array<sycl::marray<T, max_buffer_size<DIM-1>() >, 12 > M;
	    
	    for(size_t idx=0;idx<ns[DIM-1];idx++) {
		M[idx]=0;

		//std::cout<<"idx"<<idx<<" "<<idx*stride<<" "<<M.size()<<" "<<src.size()<<" "<<nsigma<<std::endl;
		//chebtransform<T,DIM-1>(src.segment(idx*stride,nsigma),M.segment(idx*stride,nsigma),ns.template head<DIM-1>());
		chebtransform_impl<T,DIM-1,DIMX>(src,
				       M[idx],
				       ns,
				       cv,
				       offset+idx*stride,
				       0,				       
				       cv_offset+Nd*Nd
				       );
	    }

	    for(size_t idx=0;idx<ns[DIM-1];idx++) {
		for(size_t sigma=0;sigma<ns[DIM-1];sigma++)  {
		    const double Td=cv[cv_offset+idx*Nd+sigma];
		    const double Td2=cos(idx*M_PI*(2.*sigma+1.)/(2.*ns[DIM-1]));

		    assert(abs(Td-Td2)<1e-12);
		    /*const auto M_it=M[sigma].begin();
		    const auto dest_it=dest.begin()+dest_offset+idx*stride;
		    //std::copy(M_it,M_it+nsigma, dest_it);
		    (*dest_it)*=Td;*/

		    for(size_t l=0;l<nsigma;l++) {
			dest[dest_offset+idx*stride+l]+=M[sigma][l]*Td;
		    }
		    
		    
		    //dest.segment(idx*stride,nsigma)+=M.segment(sigma*stride,nsigma)*Td;
		}

		for(size_t l=0;l<nsigma;l++)  {
		    dest[dest_offset+idx*stride+l]*=(idx ==0 ? 1.:2. )*1./ns[DIM-1];
		}
		//dest.segment(idx*stride,nsigma)*=(idx ==0 ? 1:2 )*1./ns[DIM-1];
	    }

	    //std::cout<<"err sum_factor="<<(D-dest).matrix().norm()<<std::endl;
	}
    }

        
    template <typename T, int DIM, typename BufType>
    void chebtransform_inplace(BufType &buf,
			       const std::array<int,DIM>& ns,
			       const sycl::accessor<double,1, sycl::access_mode::read>& cv,
			       size_t offset
			       )
    {
	sycl::marray<T, max_buffer_size<DIM>()> tmp;
	tmp=0;
	chebtransform_impl<T,DIM,DIM>(buf,tmp,ns,cv,offset,0,0);

	size_t size=1;
	for(int i=0;i<DIM;i++) {
	    size*=ns[i];
	}
	std::copy(tmp.begin(),tmp.begin()+size,buf.begin()+offset);
    }


    template <typename T, int POINTS_AT_COMPILE_TIME, int DIM, int DIM_X,unsigned int DIMOUT, int ND=-1, int... Ns>    
    class ClenshawEvaluator
    {
    public:
	inline  sycl::marray<T, POINTS_AT_COMPILE_TIME>
	operator()(const SyclRowMatrix<double,DIM_X, POINTS_AT_COMPILE_TIME>  &x,
		   const sycl::accessor<const T,1,sycl::access_mode::read> &vals,
		   const std::array<int,DIM_X>& ns, size_t offset=0 )
    {
	static_assert(DIMOUT==1); //For now only 1d output works.
	static_assert(DIM>0);
	static_assert(DIM<=DIM_X);

	sycl::marray<T, POINTS_AT_COMPILE_TIME> b1;
	sycl::marray<T, POINTS_AT_COMPILE_TIME> b2;
	sycl::marray<T, POINTS_AT_COMPILE_TIME> tmp;

	sycl::marray<T, POINTS_AT_COMPILE_TIME> result;


	const int Nd= ND >0 ? ND : ns[DIM-1];
	
	if constexpr (DIM<=1)	    
	{
	    const int Nd= ND >0 ? ND : ns[0];
	    if(Nd<=2) {
		if(Nd==1) {
		    result=vals[0+offset];
		    return result;
		}else {
		    return vals[1+offset]*x.row(0)+vals[0+offset];			
		}
	    }

	    b1=2*x.row(0);
	    b1*=vals[Nd-1+offset];
	    b1+=vals[Nd-2+offset];
	    
	    b2=(vals[Nd-1+offset]);
	

	    for(size_t j=Nd-3;j>0;j--) {
		tmp=(2.*((b1)*x.row(0))-(b2))+vals[j+offset];
	    
		b2=b1;
		b1=tmp;
	    
	    }
	    
	    return (b1*x.row(0)-b2)+vals[0+offset];
	}else //recurse down
	{	   	    
	    size_t stride = 1;
	    for(int i=0;i<DIM-1;i++)
		stride*=ns[i];
		

	    ClenshawEvaluator<T, POINTS_AT_COMPILE_TIME, std::max(DIM-1,1), DIM_X,DIMOUT,Ns...> clenshaw;
	    if(Nd<=2) {
		const sycl::marray<T,POINTS_AT_COMPILE_TIME>& c0=clenshaw(x,
					vals,
					ns,0+offset); //offset=0

		if(Nd==1) {
		    return c0;
		}else {
		    b1=clenshaw(x,
				vals,
				ns,offset+stride); //offset=stride i.e., shifted by 1 package

		    result=b1*(x.row(DIM-1));
		    result+=c0;
		    return result;
		}
	    }


	    
	    b2=clenshaw(x,
			vals,
			ns,(Nd-1) * stride+offset); //offset = (Nd-1)*stride, i.e., last package
	    const auto& cn2=clenshaw(x,
				    vals,
                                    ns,(Nd-2)*stride+offset); //second to last package


	    b1=2.*b2*x.row(DIM-1)+cn2;

	    const auto& c0=clenshaw(x,
			     vals,
                             ns,0+offset);
	    for(size_t j=Nd-3;j>0;j--) {
		tmp= clenshaw(x,
			      vals,
			      ns,j*stride+offset); //offset=j*stride
		tmp+=(2.*(b1*x.row(DIM-1))-b2);
		b2=b1;
		b1=tmp;
		
	    }

	    return (b1*x.row(DIM-1)-b2) + c0;

	}
    }
    };


    template <typename T, unsigned int DIM,  char package,int... Ns>
    inline int __eval(const sycl::accessor<const double,1,sycl::access_mode::read>& points,
		      const sycl::accessor<const T,1,sycl::access_mode::read> &interp_values,
		      const std::array<int,DIM>& ns,
		      sycl::accessor<T,1,sycl::access_mode::write> dest,
		      size_t i, size_t n_points)
    {

	const int DIMOUT=1;
	const unsigned int packageSize = 1 << package;
	const size_t np = n_points / packageSize;
	n_points = n_points % packageSize;
	
	SyclRowMatrix<double,DIM,packageSize> tmp;

	
	SyclChebychevInterpolation::ClenshawEvaluator<T, packageSize,  DIM,DIM, DIMOUT,Ns...> clenshaw;
	for (int j = 0; j < np; j++) {
	    //std::cout<<"copying package"<<std::endl;
	    for(int l=0;l<packageSize;l++) { //TODO:more efficient way?
		for(int k=0;k<DIM;k++) {
		    //std::cout<<"i="<<i<<" j="<<j<<" k="<<k<<std::endl;
		    //std::cout<<"pn"<<points[0]<<std::endl;
		    tmp(k,l)=points[(i+l)*DIM+k];
		}
	    }
	    //tmp=points.middleCols(i, packageSize);	    

	    //std::cout<<"123"<<std::endl;
	    //dest.segment(i, packageSize) = 
	    const sycl::marray<T,packageSize>& result=clenshaw(tmp, interp_values,ns);
	    for(int l=0;l<packageSize;l++)
	    {
		dest[l+i]=result[l];
	    }

	    
	    i += packageSize;
	}
	if constexpr(package > 0) {
	    if (n_points > 0) {
		i = __eval < T,  DIM,  package - 1,Ns... > (points, interp_values, ns, dest, i, n_points);
	    }
	}

	return i;

    }


    template <typename T, unsigned int DIM, unsigned int DIMOUT>
    void parallel_evaluate(
			   const Eigen::Ref<const Eigen::Array<double, DIM, Eigen::Dynamic> >
			   &points,
			   const Eigen::Ref<const Eigen::Array<T, Eigen::Dynamic, DIMOUT> >
			   &interp_values,
			   Eigen::Ref<Eigen::Array<T, Eigen::Dynamic, DIMOUT> > dest,
			   const Eigen::Vector<int, DIM>& ns,
			   BoundingBox<DIM> box);


    //#include "chebinterp_sycl.cpp"
};
#endif
