#ifndef _CONE_DOMAIN_HPP_
#define _CONE_DOMAIN_HPP_

#include <Eigen/Dense>
#include <memory>
#include "config.hpp"


#include <iostream>

#include "boundingbox.hpp"
#include "util.hpp"
#include "sycl_helpers.hpp"

class ConeRef
{
public:
    ConeRef(size_t level, size_t id,size_t memId, size_t boxId,size_t globalId):
	m_level(level),
	m_id(id),
	m_memId(memId),
	m_boxId(boxId),
	m_globalId(globalId)
    {

    }

    size_t level() const {
	return m_level;
    }


    //index in the full NxNxN grid
    size_t id() const
    {
	return m_id;
    }


    //index in memory (i.e. when skipping al non-active cones
    size_t memId() const
    {
	return m_memId;
    }

    


    size_t boxId() const
    {
	return m_boxId;
    }

    size_t globalId() const
    {
	return m_globalId;
    }

private:
    size_t m_level;
    size_t m_id;
    size_t m_boxId;
    size_t m_memId;
    size_t m_globalId;

};
    


template<size_t DIM>
class ConeDomain
{
    typedef Eigen::Matrix<double, DIM, Eigen::Dynamic> PointArray;
    
public:
    ConeDomain()
    {
	m_h.fill(1);
    }

    ConeDomain(Eigen::Vector<size_t, DIM> numEls, const BoundingBox<DIM>& domain) :
	m_numEls(numEls),
	m_domain(domain)
    {
	if(!domain.isNull()) {
	    m_h=m_domain.diagonal().array()/numEls.template cast<double>().array();
	}else{
	    m_h.fill(1);
	}
    }

    ~ConeDomain()
    {
	//std::cout<<"CD dying"<<std::endl;
    }

    static inline std::pair<size_t,size_t> keyForCone(int step,size_t box,size_t cone)
    {
	assert(step<=1);
	size_t key1=(box<<1)+step; //store the step in the lowest bit. there should be plenty of space in a size_t
	return std::make_pair(key1,cone);
    }

    inline void setNElements(Eigen::Vector<size_t, DIM> numEls) {
	m_numEls=numEls;	
    }

    inline constexpr size_t n_elements() const
    {
	size_t n=1;
	for(size_t i=0;i<DIM;i++)
	    n*=m_numEls[i];
	return n;
    }

    inline constexpr size_t n_elements(size_t d) const {
	return m_numEls[d];
    }

    inline constexpr Eigen::Vector<size_t,DIM> num_elements() const {
	return m_numEls;
    }

    

    inline const std::vector<size_t>& activeCones() const
    {
	return m_activeCones;
    }

    inline void setActiveCones( std::vector<size_t>& cones)
    {
	m_activeCones=cones;

    }

    double h() const
    {
	return m_h[2];
    }




    inline void setConeMap( IndexMap& cone_map)
    {
	m_coneMap=cone_map;
    }


    inline size_t memId(size_t el) const
    {
	/*if(!isActive(el)) {
	    std::cout<<"isactive"<<el<<std::endl;
	    }*/
	assert(isActive(el));
	return m_coneMap.at(el);
	
    }


    inline bool isActive(size_t el) const
    {
	return m_coneMap.count(el)>0;
    }
    
    inline BoundingBox<DIM> domain() const {
	return m_domain;
    }

    inline IndexMap coneMap() const {
	return m_coneMap;
    }

    //transforms from (-1,1) to K_el
    Eigen::Matrix<double,DIM,Eigen::Dynamic> transform(size_t el,const Eigen::Ref<const PointArray >& pnts) const 
    {
	const BoundingBox bbox=region(el);
	Eigen::Matrix<double,DIM,Eigen::Dynamic> tmp(DIM,pnts.cols());
	const auto a=0.5*m_h.array();
	const Eigen::Array<double, DIM,1> b=m_domain.min().array()+(indicesFromId(el).template cast<double>().array()+0.5)*m_h.array();	

	for(int i=0;i<pnts.cols();i++) {
	    tmp.col(i)=(pnts.array().col(i)*a)+b;
	}


	
	return tmp;
	    
    }


    //transforms from K_el to (-1,1)
    void transformBackwardsInplace(size_t el,Eigen::Ref<const  PointArray > pnts) const 
    {
	const BoundingBox bbox=region(el);
	Eigen::Matrix<double,DIM,Eigen::Dynamic> tmp(DIM,pnts.cols());

	const auto a=0.5*(bbox.max()-bbox.min()).array();
	const auto b=0.5*(bbox.max()+bbox.min()).array();

	pnts.array()=(pnts.array().colwise()-b).colwise()/a;
    }


    //transforms from K_el to (-1,1)
    Eigen::Matrix<double,DIM,Eigen::Dynamic> transformBackwards(size_t el,const Eigen::Ref<const  PointArray >& pnts) const 
    {
	const BoundingBox bbox=region(el);
	Eigen::Matrix<double,DIM,Eigen::Dynamic> tmp(DIM,pnts.cols());

	const auto a=0.5*(m_h).array();
	const Eigen::Array<double, DIM,1> b=m_domain.min().array()+(indicesFromId(el).template cast<double>().array()+0.5)*m_h.array();

	tmp.array()=(pnts.array().colwise()-b).colwise()/a;
	
	return tmp;	    
    }

    inline Eigen::Vector<size_t,DIM> indicesFromId(size_t j) const {
	Eigen::Vector<size_t,DIM> indices;	
	for(int i=0;i<DIM;i++) {
	    const size_t idx=j % m_numEls[i];
	    j=j / m_numEls[i];
	    
	    indices[i]=idx;
	}

	return indices;
    }

    inline size_t idFromIndices(const Eigen::Vector<size_t,DIM>& indices) const {
	size_t id;
	size_t stride=1;
	for(int i=0;i<DIM;i++) {
	    id+=indices[i]*stride;
	    stride*=m_numEls[i];

	}

	return id;
    }



    BoundingBox<DIM> region(size_t j) const
    {
	size_t idx=0;
	auto j0=j;

	assert(j<n_elements());
	Eigen::Vector<double, DIM> min,max;
	Eigen::Vector<double, DIM> h=m_domain.diagonal();
	for(int i=0;i<DIM;i++) {
	    const size_t idx=j % m_numEls[i];
	    j=j / m_numEls[i];
	    
	    
	    
	    min[i]=m_domain.min()[i]+(idx*(h[i]/((double) m_numEls[i])));
	    max[i]=(min[i]+(h[i]/((double) m_numEls[i])));
	}
	return  BoundingBox<DIM>(min,max);
    }

    size_t elementForPoint(const Eigen::Ref<const Eigen::Vector<double,DIM> > & pnt) const
    {
	size_t idx=0;
	int stride=1;

	if(m_domain.squaredExteriorDistance(pnt)>0) {
	    return SIZE_MAX;
	}
	   
	for(int j=0;j<DIM;j++) {	    
	    const int q=std::floor( (pnt[j]-m_domain.min()[j])/m_h[j]);
	    

	    const size_t ij=( std::clamp(q,0, (int) ( m_numEls[j]-1)));

	    idx+=ij*stride;
	    stride*=m_numEls[j];
	}


	//std::cout<<"pnt: "<<pnt.transpose()<<" "<<idx<<" "<<m_domain<<std::endl;
	assert(idx<n_elements());
	return idx;

    }

    bool isEmpty() const
    {
	return m_domain.isEmpty();
    }


    inline Eigen::Matrix<double,DIM,Eigen::Dynamic> rotated_points(size_t el,const Eigen::Ref<const  PointArray >& pnts, Eigen::Vector3d direction,bool backward) const
    {
	Eigen::Vector3d xc=Eigen::Vector3d::Zero();
	const double H=1;
       
	auto targets=Util::interpToCart<DIM>(transform(el,pnts).array(), xc, H);

	auto rotation=Eigen::Quaternion<double>::FromTwoVectors(direction,Eigen::Vector3d({0,0,1}));
	Eigen::Array<double, DIM,1> pnt;
	if(backward) {
	    rotation=rotation.inverse();
	}
	
	for(int j=0;j<targets.cols();j++) {
	    pnt=rotation*(targets.col(j));
	    targets.col(j)=pnt;
	}	       

	//transform to the un-rotated interpolation domain
	PointArray transformed(3,targets.cols());
	Util::cartToInterp2<DIM>(targets.array(),  xc, H,transformed);

	return transformed;
    }


    inline Eigen::Matrix<double,DIM,Eigen::Dynamic> translated_points(size_t el,const Eigen::Ref<const  PointArray >& pnts, Eigen::Vector3d xc, double H, Eigen::Vector3d pxc, double pH) const
    {	
	auto targets=Util::interpToCart<DIM>(transform(el,pnts).array(), xc-Eigen::Vector3d(0,0,(pxc-xc).norm()), pH);
	    
	//transform to the child interpolation domain
	PointArray transformed(3,targets.cols());
	Util::cartToInterp2<DIM>(targets.array(), xc, H,transformed);
	return transformed;

    }

private:
    BoundingBox<DIM> m_domain;
    Eigen::Vector<size_t, DIM> m_numEls;
    Eigen::Vector<double, DIM> m_h;
    std::vector<size_t> m_activeCones;
    IndexMap m_coneMap;
};



template<size_t DIM>
class SyclConeDomain
{

public:
    SyclConeDomain()
    {
	m_h.fill(1);
    }

    SyclConeDomain(const ConeDomain<DIM>& cd)
    {
	//std::cout<<"cloning cone domain to sycl"<<cd.domain()<<std::endl;
	const Eigen::Vector<double,DIM> h=cd.domain().diagonal();
	const Eigen::Vector<double,DIM> min=(cd.domain().min());

	for(int i=0;i<DIM;i++) {
	    m_min[i]=min[i];
	    m_numEls[i]=cd.n_elements(i);
	    m_h[i]=h[i]/m_numEls[i];
	}
    }


    /*    SyclConeDomain(sycl::marray<size_t,DIM> numEls, sycl::marray<double,DIM> h,sycl::marray<double,DIM> center) :
	m_numEls(numEls),
	m_h(h),
	m_min(mi)
    {
    }
    */

    ~SyclConeDomain()
    {

    }

    inline void setNElements(Eigen::Vector<size_t, DIM> numEls) {
	for(int i=0;i<DIM;i++) {	    
	    m_numEls[i]=numEls[i];
	}
    }

    inline constexpr size_t n_elements() const
    {
	size_t n=1;
	for(size_t i=0;i<DIM;i++)
	    n*=m_numEls[i];
	return n;
    }

    inline constexpr size_t n_elements(size_t d) const {
	return m_numEls[d];
    }

    inline constexpr sycl::marray<size_t,DIM> num_elements() const {
	return m_numEls;
    }

    double h() const
    {
	return m_h[2];
    }
    
    //transforms from (-1,1) to K_el
    void transform(size_t el,const sycl::accessor<const double,1,sycl::access_mode::read>& pnts, sycl::marray<double,DIM>& dest, size_t i) const 
    {	
	for(int j=0;j<DIM;j++) {
	    const size_t idx=el % m_numEls[j];
	    el=el / m_numEls[j];


	    
	    double b=m_min[j]+(idx+0.5)*m_h[j];	
	    
	    dest[j]=(0.5*pnts[i*DIM+j]*m_h[j])+b;	    
	}

    }

    

    
    inline sycl::marray<size_t, DIM> indicesFromId(size_t j) const {
	sycl::marray<size_t, DIM> indices;
	for(int i=0;i<DIM;i++) {
	    const size_t idx=j % m_numEls[i];
	    j=j / m_numEls[i];
	    
	    indices[i]=idx;
	}

	return indices;
    }


    
    
    #if 0
    inline void setConeMap( IndexMap& cone_map)
    {
	m_coneMap=cone_map;
    }


    inline size_t memId(size_t el) const
    {
	assert(isActive(el));
	return m_coneMap.at(el);
	
    }


    inline bool isActive(size_t el) const
    {
	return m_coneMap.count(el)>0;
    }





    // //transforms from (-1,1) to K_el
    // void transform(size_t el,const sycl::accessor<const double>& pnts, sycl::accessor<double>& dest, size_t n) const 
    // {
    // 	const auto a=0.5*m_h;
    // 	const Eigen::Array<double, DIM,1> b=m_center+(indicesFromId(el))*m_h;	

    // 	for(int i=0;i<n;i++) {
    // 	    for(int j=0;j<DIM;j++) {
    // 		dest[i*DIM+j]=pnts[i*DIM+j]*a+b;
    // 	    }
    // 	}

	
	    
    // }


    //transforms from K_el to (-1,1)
    void transformBackwardsInplace(size_t el,Eigen::Ref<const  PointArray > pnts) const 
    {
	const BoundingBox bbox=region(el);
	Eigen::Matrix<double,DIM,Eigen::Dynamic> tmp(DIM,pnts.cols());

	const auto a=0.5*(bbox.max()-bbox.min()).array();
	const auto b=0.5*(bbox.max()+bbox.min()).array();

	pnts.array()=(pnts.array().colwise()-b).colwise()/a;
    }


    //transforms from K_el to (-1,1)
    Eigen::Matrix<double,DIM,Eigen::Dynamic> transformBackwards(size_t el,const Eigen::Ref<const  PointArray >& pnts) const 
    {
	const BoundingBox bbox=region(el);
	Eigen::Matrix<double,DIM,Eigen::Dynamic> tmp(DIM,pnts.cols());

	const auto a=0.5*(m_h).array();
	const Eigen::Array<double, DIM,1> b=m_domain.min().array()+(indicesFromId(el).template cast<double>().array()+0.5)*m_h.array();

	tmp.array()=(pnts.array().colwise()-b).colwise()/a;
	
	return tmp;	    
    }

    inline size_t idFromIndices(const Eigen::Vector<size_t,DIM>& indices) const {
	size_t id;
	size_t stride=1;
	for(int i=0;i<DIM;i++) {
	    id+=indices[i]*stride;
	    stride*=m_numEls[i];

	}

	return id;
    }



    BoundingBox<DIM> region(size_t j) const
    {
	size_t idx=0;
	auto j0=j;

	assert(j<n_elements());
	Eigen::Vector<double, DIM> min,max;
	Eigen::Vector<double, DIM> h=m_domain.diagonal();
	for(int i=0;i<DIM;i++) {
	    const size_t idx=j % m_numEls[i];
	    j=j / m_numEls[i];
	    
	    
	    
	    min[i]=m_domain.min()[i]+(idx*(h[i]/((double) m_numEls[i])));
	    max[i]=(min[i]+(h[i]/((double) m_numEls[i])));
	}
	return  BoundingBox<DIM>(min,max);
    }

    size_t elementForPoint(const Eigen::Ref<const Eigen::Vector<double,DIM> > & pnt) const
    {
	size_t idx=0;
	int stride=1;

	if(m_domain.squaredExteriorDistance(pnt)>0) {
	    return SIZE_MAX;
	}
	   
	for(int j=0;j<DIM;j++) {	    
	    const int q=std::floor( (pnt[j]-m_domain.min()[j])/m_h[j]);
	    

	    const size_t ij=( std::clamp(q,0, (int) ( m_numEls[j]-1)));

	    idx+=ij*stride;
	    stride*=m_numEls[j];
	}

	//std::cout<<"pnt: "<<pnt.transpose()<<" "<<idx<<" "<<m_domain<<std::endl;
	assert(idx<n_elements());
	return idx;

    }
#endif

private:
    sycl::marray<size_t,DIM>  m_numEls;
    sycl::marray<double,DIM> m_h;
    //sycl::marray<double,DIM> m_center;
    sycl::marray<double,DIM> m_min;  
    //SyclHelpers::SyclIndexMap m_coneMap;
};



#endif
