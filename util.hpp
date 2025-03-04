#ifndef _UTIL_HPP_
#define _UTIL_HPP_

#include <Eigen/Dense>
#include <numeric>

#include <algorithm>
#include <oneapi/tbb/parallel_for.h>
#include <tbb/parallel_sort.h>

namespace Util
{
    template <typename RandomIt, class Compare>
    auto sort_with_permutation( RandomIt cbegin, RandomIt cend, Compare comp)
    {
	auto len = std::distance(cbegin, cend);
	std::vector<size_t> perm(len);
	std::iota(perm.begin(), perm.end(), 0U);
	std::sort (perm.begin(), perm.end(),
			    [&](const size_t &a, const size_t &b) {
				return comp(*(cbegin + a), *(cbegin + b));
			    });
	return perm;
    }

    template <typename T, int DIMOUT>
    void copy_with_permutation_rowwise(const Eigen::Ref<const Eigen::Array<T, Eigen::Dynamic, DIMOUT> > &v, const std::vector<size_t> &permutation,
				       Eigen::Ref<Eigen::Array<T, Eigen::Dynamic, DIMOUT> > target)
    {
	for (size_t i = 0; i < v.rows(); i++) {	
	    target.row(i) = v.row(permutation[i]);
	}
    }

    template <typename T, int DIMOUT>
    void copy_with_permutation_colwise(const Eigen::Ref<const Eigen::Array<T, DIMOUT, Eigen::Dynamic> > &v, const std::vector<size_t> &permutation,
				       Eigen::Ref<Eigen::Array<T, DIMOUT, Eigen::Dynamic> > target)
    {
	for (size_t i = 0; i < v.cols(); i++) {	
	    target.col(i) = v.col(permutation[i]);
	}    
    }


    template <typename T, int DIMOUT>
    void copy_with_inverse_permutation_colwise(const Eigen::Ref<const Eigen::Array<T, DIMOUT, Eigen::Dynamic> > &v, const std::vector<size_t> &permutation,
					       Eigen::Ref<Eigen::Array<T, DIMOUT,Eigen::Dynamic> > target)
    {
	for (size_t i = 0; i < v.cols(); i++) {	
	    target.col(permutation[i]) = v.col(i);	    
	}
    }

    template <typename T, int DIMOUT>
    void copy_with_inverse_permutation_rowwise(const Eigen::Ref<const Eigen::Array<T, Eigen::Dynamic, DIMOUT> > &v, const std::vector<size_t> &permutation,
					       Eigen::Ref<Eigen::Array<T, Eigen::Dynamic, DIMOUT> > target)
    {
	for (size_t i = 0; i < v.rows(); i++) {
	    //std::cout<<"permutation"<<permutation[i]<<" "<<target.rows()<<" "<<std::endl;
	    target.row(permutation[i]) = v.row(i);
	}       
    }

    //assumes: p[0] in (0,1) and p[1] in (-pi,pi)
    template<size_t DIM,long int POINTS>
    inline Eigen::Vector<PointScalar, DIM> sphericalToCart(const Eigen::Ref<const Eigen::Array<PointScalar, DIM, POINTS> >& p) 
    {
        Eigen::Array<PointScalar, DIM,POINTS> res(POINTS,p.cols());

        if constexpr(DIM==2) {
	    res.row(0)= p.row(0)*Eigen::cos(p.row(1));
	    res.row(1)= p.row(0)*Eigen::sin(p.row(1));
        }else {
            static_assert(DIM==3);
	    res.row(0)=p.row(0)*Eigen::cos(p.row(2))*Eigen::sin(p.row(1));
	    res.row(1)=p.row(0)*Eigen::sin(p.row(2))*Eigen::sin(p.row(1));
	    res.row(2)=p.row(0)*Eigen::cos(p.row(1));	    
        }

        return  res;
    }

    

    template<size_t DIM,typename PointVector>
    inline typename PointVector::PlainObject interpToCart(const Eigen::ArrayBase<PointVector>& p, const Eigen::Vector<PointScalar, DIM> &xc, PointScalar H)
    {
	typename PointVector::PlainObject res(p.rows(),p.cols());

        if constexpr(DIM==2) {
	    res.row(0)= xc[0]+(H/p.row(0))*Eigen::cos(p.row(1));
	    res.row(1)= xc[1]+(H/p.row(0))*Eigen::sin(p.row(1));
        }else {
            static_assert(DIM==3);
	    res.row(0)=xc[0]+(H/p.row(0))*Eigen::cos(p.row(2))*Eigen::sin(p.row(1));
	    res.row(1)=xc[1]+(H/p.row(0))*Eigen::sin(p.row(2))*Eigen::sin(p.row(1));
	    res.row(2)=xc[2]+(H/p.row(0))*Eigen::cos(p.row(1));	    
        }

        return  res;
    }



    template<size_t DIM>
    inline void interpToCart(const sycl::marray<PointScalar,DIM>& p, sycl::marray<PointScalar,DIM>& res, const sycl::marray<PointScalar,DIM>& xc, PointScalar H)
    {
        if constexpr(DIM==2) {
	    res[0]= xc[0]+(H/p[0])*cos(p[1]);
	    res[1]= xc[1]+(H/p[0])*sin(p[1]);
        }else {
            static_assert(DIM==3);
	    res[0]=xc[0]+(H/p[0])*cos(p[2])*sin(p[1]);
	    res[1]=xc[1]+(H/p[0])*sin(p[2])*sin(p[1]);
	    res[2]=xc[2]+(H/p[0])*cos(p[1]);	    
        }
    }




    template<size_t DIM>
    inline Eigen::Vector<PointScalar, DIM> cartToSpherical(const Eigen::Ref<const Eigen::Vector<PointScalar, DIM> >& p) 
    {
	Eigen::Vector<PointScalar,DIM> xp = p ;


        if constexpr (DIM==2) {
	    const PointScalar r = xp.norm();

	    const long double theta = std::atan2( (long double) xp[1], (long double) xp[0]);

            Eigen::Vector<PointScalar, DIM> res;
            res[0] = r;
            res[1] = theta ;

            //assert(-1.0001 <= res[0] && res[0] <= 1.0001);
            //assert(-1 <= res[1] && res[1] <= 1);

            return  res;

        }else{
            static_assert(DIM==3);
            const PointScalar phi = std::atan2(xp[1], xp[0]);
            const PointScalar a=(xp[0]*xp[0]+xp[1]*xp[1]);
            const PointScalar theta= std::atan2(sqrt(a),xp[2]);
            const PointScalar r= sqrt(a+xp[2]*xp[2]);

            Eigen::Vector<PointScalar, DIM> res;
            res[0] = r;
            res[1] = theta;
            res[2] = phi;

            return  res;

        }
    }


    template<size_t DIM,typename PointVector, typename PointVector2>
    inline typename PointVector::PlainObject cartToInterp2(const Eigen::ArrayBase<PointVector>& x, const Eigen::Vector<PointScalar, DIM> &xc, PointScalar H, PointVector2& rs)
    {

	auto p=x.colwise()-xc.array();
	
	const auto a = p.row(0)*p.row(0)+p.row(1)*p.row(1);
	rs.row(2)= p.row(0).binaryExpr(p.row(1), [](PointScalar a,PointScalar b) {return  std::atan2(b,a);});
	rs.row(1)= p.row(2).binaryExpr(a, [](PointScalar b,PointScalar aj){return std::atan2(std::sqrt(aj),b);});
	rs.row(0)=H/((a+p.row(2)*p.row(2)).sqrt());


	return rs;
    }
    
        template<size_t DIM>
    inline Eigen::Vector<PointScalar, DIM> cartToInterp(Eigen::Vector<PointScalar, DIM> p, const Eigen::Vector<PointScalar, DIM> &xc, PointScalar H) 
    {
	auto ps=cartToSpherical<DIM>(p-xc);
	ps[0]=H/ps[0];


	return ps;
    }



    template<size_t DIM>
    inline void cartToInterp(const sycl::marray<PointScalar,DIM>& p, sycl::marray<PointScalar,DIM>& res, const sycl::marray<PointScalar,DIM>& xc, PointScalar H)
    {
	auto xp=p-xc;
        if constexpr (DIM==2) {
	    const PointScalar r = sqrt(xp[0]*xp[0]+xp[1]*xp[1]);

	    const long double theta = atan2( (long double) xp[1], (long double) xp[0]);
            
            res[0] = H/r;
            res[1] = theta ;

        }else{
            static_assert(DIM==3);
            const PointScalar phi = atan2(xp[1], xp[0]);
            const PointScalar a=(xp[0]*xp[0]+xp[1]*xp[1]);
            const PointScalar theta= atan2(sqrt(a),xp[2]);
            const PointScalar r= sqrt(a+xp[2]*xp[2]);


            res[0] = H/r;
            res[1] = theta;
            res[2] = phi;
        }

    }


    template<int DIM>
    inline Eigen::Vector<size_t,DIM> indicesFromId(size_t j, const Eigen::Ref<const Eigen::Vector<size_t,DIM> > &ns)  {
	Eigen::Vector<size_t,DIM> indices;	
	for(int i=0;i<DIM;i++) {
	    const size_t idx=j % ns[i];
	    j=j / ns[i];
	    
	    indices[i]=idx;
	}

	return indices;
    }


    template<int DIM>
    inline size_t indicesToId(const Eigen::Ref<const Eigen::Vector<size_t,DIM> >& idcs, const Eigen::Ref<const Eigen::Vector<size_t,DIM> > &ns)  {
	size_t id=0;
	size_t stride=1;
	for(int i=0;i<DIM;i++) {
	    id+=idcs[i]*stride;
	    stride*=ns[i];
	}

	return id;
    }



    template <typename T,int DIM,int DIMOUT>
    PointScalar compute_slice_norm(const Eigen::Ref<const Eigen::Array<T,Eigen::Dynamic, DIMOUT> >& data, const Eigen::Vector<size_t, DIM>& ns,int axis, int layers=1)
    {
	PointScalar v1=0;
	PointScalar v2=0;
	
	for(size_t idx=0;idx<data.rows();idx++) {
	    Eigen::Vector<size_t,DIM> split=indicesFromId<DIM>(idx,ns);
	    PointScalar n=data.row(idx).matrix().squaredNorm();
	    
	    v1+=n;
	    if(split[axis]==ns[axis]-layers) {
		v2+=n;
	    }
	}

	return sqrt(v2)/std::max(1.,sqrt(v1));
    }




    }; // namespace Util


template<typename T, size_t DIM>
std::ostream & operator<<(std::ostream &os, const std::array<T,DIM>& p)
{
    for(size_t i=0;i<DIM;i++) {
	os << p[i]<<" ";
    }
    return os;
}
    

#endif
