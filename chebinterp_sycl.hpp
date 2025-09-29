#ifndef _SYCL_CHEBINTERP_HPP_
#define _SYCL_CHEBINTERP_HPP_

#include "Eigen/src/Core/util/Constants.h"
#include "boundingbox.hpp"
#include <cstddef>
#include <sycl/access/access.hpp>
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
    inline constexpr size_t max_buffer_size(int order, int els=1) 
    {
        if constexpr(DIM==1) {
            return std::max((order-2),2)*els;
        }

        if constexpr(DIM==2){
            return std::max((order-2),2)*order*els*els;
         }
        
        return std::max((order-2),2)*order*order*els*els*els;

    }

    


    template <typename T, int DIM, int DIMX, int MAX_ORDER>
    void chebtransform_impl(const sycl::accessor<T> &src,
                            sycl::marray<T, max_buffer_size<DIM>(MAX_ORDER) > & dest,
                            const std::array<int,DIMX>& ns,
                            const sycl::accessor< PointScalar,1, sycl::access_mode::read>& cv,
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
	assert(ns[DIM-1]<=MAX_ORDER); //We have a max buffer size since we are working on the stack
	
	const size_t Nd=ns[DIM-1];

	if constexpr(DIM==1) {
	    //do a straight forwad summation of the innermost dimension	    
	    for(size_t idx=0;idx<ns[0];idx++) {
		for(size_t sigma=0;sigma<ns[0];sigma++)  {
		    const PointScalar Td=cv[cv_offset+idx*Nd+sigma];
		    //const PointScalar Td=cos(idx*M_PI*(2.*sigma+1.)/(2.*ns[DIM-1]));
		    assert(nsigma==1);
		    dest[dest_offset+idx]+=src[offset+sigma]*T(Td);
		    //dest.segment(idx*stride,nsigma)+=src.segment(sigma*stride,nsigma)*Td;
		}
		dest[dest_offset+idx]*=(idx ==0 ? 1.:2. )*1./ns[0];
	    }
	    //std::cout<<"done inner"<<std::endl;
	}else {
	    std::array<sycl::marray<T, max_buffer_size<DIM-1>(MAX_ORDER) >, MAX_ORDER > M;	    
	    size_t idx=0;
	    for(idx=0;idx<ns[DIM-1];idx++) {
		//M[idx]=0; //just to be safe

		//std::cout<<"idx"<<idx<<" "<<idx*stride<<" "<<M.size()<<" "<<src.size()<<" "<<nsigma<<std::endl;
		//chebtransform<T,DIM-1>(src.segment(idx*stride,nsigma),M.segment(idx*stride,nsigma),ns.template head<DIM-1>());
		chebtransform_impl<T,DIM-1,DIMX,MAX_ORDER>(src,
				       M[idx],
				       ns,
				       cv,
				       offset+idx*stride,
				       0,				       
				       cv_offset+Nd*Nd
				       );
	    }
	    /*for(;idx<MAX_ORDER;idx++) {
		M[idx]=0; //make sure we are not using uninitialized memory
		}*/

	    for(size_t idx=0;idx<ns[DIM-1];idx++) {
		for(size_t sigma=0;sigma<ns[DIM-1];sigma++)  {
		    const PointScalar Td=cv[cv_offset+idx*Nd+sigma];
		    //const PointScalar Td2=cos(idx*M_PI*(2.*sigma+1.)/(2.*ns[DIM-1]));

		    //assert(abs(Td-Td2)<1e-12);
		    /*const auto M_it=M[sigma].begin();
		    const auto dest_it=dest.begin()+dest_offset+idx*stride;
		    //std::copy(M_it,M_it+nsigma, dest_it);
		    (*dest_it)*=Td;*/

		    for(size_t l=0;l<nsigma;l++) {
			dest[dest_offset+idx*stride+l]+=M[sigma][l]*T(Td);
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

        
    template <typename T, int DIM, int MAX_ORDER,typename BufType>
    void chebtransform_inplace(BufType &buf,
			       const std::array<int,DIM>& ns,
			       const sycl::accessor<PointScalar,1, sycl::access_mode::read>& cv,
			       size_t offset
			       )
    {
	sycl::marray<T, max_buffer_size<DIM>(MAX_ORDER)> tmp;
	tmp=0;
	chebtransform_impl<T,DIM,DIM,MAX_ORDER>(buf,tmp,ns,cv,offset,0,0);

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
	template <typename AccessorType>
	inline  sycl::marray<T, POINTS_AT_COMPILE_TIME>
	operator()(const SyclRowMatrix<PointScalar,DIM_X, POINTS_AT_COMPILE_TIME>  &x,
		   const AccessorType &vals,
		   const std::array<int,DIM_X>& ns, size_t offset=0)
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
		    //result = vals[1+offset]*x.row(0)+vals[0+offset];
		    for(int j=0;j<POINTS_AT_COMPILE_TIME;j++) {
			result[j]=vals[1+offset]*T(x(0,j))+vals[0+offset];
		    }
		    return result;
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
    inline int __eval(const sycl::accessor<const PointScalar,1,sycl::access_mode::read>& points,
		      const sycl::accessor<const T,1,sycl::access_mode::read> &interp_values,
		      const std::array<int,DIM>& ns,
		      sycl::accessor<T,1,sycl::access_mode::write> dest,
		      size_t i, size_t n_points)
    {

	const int DIMOUT=1;
	const unsigned int packageSize = 1 << package;
	const size_t np = n_points / packageSize;
	n_points = n_points -np*packageSize;
	
	SyclRowMatrix<PointScalar,DIM,packageSize> tmp;

	
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


    template <typename T, unsigned int DIM,  char package, typename PointAccessorType, typename ValueAccessorType, typename DestAccessorType>
    inline int eval(const PointAccessorType& points,
		    size_t pnt_offset,
		    const ValueAccessorType &interp_values,
		    size_t v_offset,
		    const std::array<int,DIM>& ns,
		    DestAccessorType dest,
		    size_t dest_offset,
		    size_t i, size_t n_points)
    {

	const int DIMOUT=1;
	const unsigned int packageSize = 1 << package;
	const size_t np = n_points / packageSize;
	n_points = n_points - np*packageSize;
	
	SyclRowMatrix<PointScalar,DIM,packageSize> tmp;

	
	SyclChebychevInterpolation::ClenshawEvaluator<T, packageSize,  DIM,DIM, DIMOUT> clenshaw;
	for (int j = 0; j < np; j++) {
	    //std::cout<<"copying package"<<std::endl;
	    for(int l=0;l<packageSize;l++) { //TODO:more efficient way?
		for(int k=0;k<DIM;k++) {
		    //std::cout<<"i="<<i<<" j="<<j<<" k="<<k<<std::endl;
		    //std::cout<<"pn"<<points[0]<<std::endl;
		    tmp(k,l)=points[pnt_offset+(i+l)*DIM+k];
		}
	    }
	    //tmp=points.middleCols(i, packageSize);	    

	    //std::cout<<"123"<<std::endl;
	    //dest.segment(i, packageSize) = 
	    const sycl::marray<T,packageSize>& result=clenshaw(tmp, interp_values,ns,v_offset);
	    for(int l=0;l<packageSize;l++)
	    {
		dest[dest_offset+l+i]=result[l];
	    }

	    
	    i += packageSize;
	}
	if constexpr(package > 0) {
	    if (n_points > 0) {
		i = eval < T,  DIM,  package - 1> (points,pnt_offset, interp_values,v_offset, ns, dest,dest_offset, i, n_points);
	    }
	}

	return i;

    }



    template <typename T, unsigned int DIM, unsigned int DIMOUT>
    void parallel_evaluate(
			   const Eigen::Ref<const Eigen::Array<PointScalar, DIM, Eigen::Dynamic> >
			   &points,
			   const Eigen::Ref<const Eigen::Array<T, Eigen::Dynamic, DIMOUT> >
			   &interp_values,
			   Eigen::Ref<Eigen::Array<T, Eigen::Dynamic, DIMOUT> > dest,
			   const Eigen::Vector<int, DIM>& ns,
			   BoundingBox<DIM> box);

    template <typename T, unsigned int DIM, unsigned int DIMX, size_t POINTS_AT_CTIME, typename DestType, typename TmpType>
    void tp_evaluate_int(
			 const sycl::marray<PointScalar, POINTS_AT_CTIME> &points,
			 int pnt_offset,
			 const sycl::accessor<const T,1,sycl::access_mode::read> &interp_values,
			 size_t offset,
			 DestType& dest,				
			 const std::array<int, DIMX> &ns,
			 const std::array<int, DIMX> &nps,
			 TmpType& tmp,
			 size_t dest_offset,
			 size_t tmp_offset
			 )
    {

	assert(pnt_offset>=0);
        if constexpr (DIM == 1) {
            // std::cout<<"a"<<points[0].size()<<" "<<dest.rows()<<std::endl;
            // assert(points[0].size()==dest.rows());
            //ChebychevInterpolation::parallel_evaluate<T, 1, DIMOUT>(
            //    points[0].transpose(), interp_values, dest, ns);

	    assert(pnt_offset==0);
            ClenshawEvaluator<T, 1, 1, 1, 1> eval;
	    SyclRowMatrix<PointScalar, 1, 1>  mp;
	    std::array<int,1> mn;
	    mn[0]=ns[0];
	    for(int i=0;i<nps[0];i++) {
		mp[0][0]=points[pnt_offset+i];

		 // T val=0;
		 // for(int sigma=0;sigma<ns[0];sigma++) {
		 //     const PointScalar Td=cos(sigma*acos(points[pnt_offset+i]));
		 //     val+=interp_values[offset+sigma]*Td;
		 // }
		 // dest[dest_offset+i]=val;
		auto res=eval(mp,interp_values, mn,offset);
		dest[dest_offset+i]=res[0];
            }
            //auto mp= reinterpret_cast<const SyclRowMatrix<PointScalar, 1, MAX_LOW_ORDER> & >(points[pnt_offset]);
        } else {
	    assert(DIM>1);
	    size_t Np=1;
	    size_t n_values = 1;
	    for(int i=0;i<DIM-1;i++) {
		Np*=nps[i];
		n_values*=ns[i];
	    }
	
	    
	    const size_t Ny = nps[DIM-1];

	    assert(Ny< POINTS_AT_CTIME);


            //Eigen::Array<T, Eigen::Dynamic, 1> M(ns[DIM - 1] * Np);
	    


            for (size_t idx = 0; idx < ns[DIM - 1]; idx++) {
		size_t new_pnt_offset= DIM >= 2 ?   pnt_offset-nps[DIM-2] : 0;
		assert(new_pnt_offset>=0);
		tp_evaluate_int<T, DIM - 1, DIMX, POINTS_AT_CTIME>(
		 						   points,
								   new_pnt_offset,
								   interp_values,
								   offset+idx * n_values,
		 						   tmp,
		 						   ns,
								   nps,
								   tmp,
								   tmp_offset+idx*Np,
								   tmp_offset+ns[DIM-1]*Np
								   );
            }


	    for(size_t p=0;p<Np;p++)
	    {
		T b1=0;
		T b2=0;
		T tmp2=0;	    

		for (size_t sigma = 0; sigma < Ny; sigma++) {
		    if (ns[DIM - 1] <= 2) {		    		    
			if (ns[DIM - 1] == 0) {
			    b1=tmp[tmp_offset+p];
			} else {
			    b1= T(points[pnt_offset+sigma]) * tmp[tmp_offset+1*Np+p]
				+ tmp[tmp_offset+0*Np+p];			    
			}

			dest[dest_offset+p+sigma*Np]=b1;
		    } else {
			b1 = RealScalar(2) * T(points[pnt_offset+sigma]) *
			    tmp[tmp_offset+(ns[DIM-1]-1)*Np+p]+
			    tmp[tmp_offset+(ns[DIM-1]-2)*Np+p];
			b2 = tmp[tmp_offset+(ns[DIM-1]-1)*Np+p];

			for (size_t j = ns[DIM - 1] - 3; j > 0; j--) {
			    tmp2 = (RealScalar(2) * (T(points[pnt_offset+sigma])*(b1)) - (b2)) +
				tmp[tmp_offset+(j)*Np+p];

			    b2 = b1;
			    b1 = tmp2;
			}

			const T result=(T(points[pnt_offset+sigma])*b1 - b2) + tmp[tmp_offset+(0)*Np+p];

			dest[dest_offset+sigma*Np+p]=result;
		    }
		}
	    }

        }

    }

    template <typename T, unsigned int DIM, size_t PointsAtCompileTime ,typename AccessorType1,typename AccessorType2,typename AccessorType3>
    void tp_evaluate_t(	
               const sycl::marray<PointScalar, PointsAtCompileTime>& points,
	       const AccessorType1& interp_values,
	       size_t offset,		       
	       const std::array<int, DIM>& ns,
	       const std::array<int,DIM>& np,
	       AccessorType2& dest,
	       AccessorType3& tmp,
	       size_t dest_offset,
	       size_t tmp_offset
	       )
    {
	int pnt_offset=0;
	for(int i=0;i<DIM-1;i++) {
	    pnt_offset+=np[i];
	}	

	tp_evaluate_int<T, DIM, DIM>(points,pnt_offset,  interp_values, offset, dest, ns,np,tmp, dest_offset, tmp_offset);
	

    }


    //#include "chebinterp_sycl.cpp"
};
#endif
