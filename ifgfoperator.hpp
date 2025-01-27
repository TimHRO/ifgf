#ifndef __IFGFOPERATOR_HPP_
#define __IFGFOPERATOR_HPP_

#include "Eigen/src/Core/util/Constants.h"
#include "config.hpp"

#include <Eigen/Dense>
#include <tbb/queuing_mutex.h>
#include <tbb/spin_mutex.h>
#include <tbb/queuing_mutex.h>
#include <tbb/parallel_for.h>
#include <tbb/global_control.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_reduce.h>


#include "boundingbox.hpp"
#include "cone_domain.hpp"
#include "helmholtz_ifgf.hpp"
#include "octree.hpp"
#include "chebinterp.hpp"
#include "chebinterp_sycl.hpp"

//#include <fstream>
#include <iostream>

#include <memory>

template<typename T, unsigned int DIM, unsigned int DIMOUT, typename Derived>
class IfgfOperator
{
public:
    typedef Eigen::Array<double, DIM, Eigen::Dynamic> PointArray;     //, Eigen::RowMajor?

    enum RefinementType { RefineH, RefineP};

    IfgfOperator(long int maxLeafSize = -1, size_t order=5, size_t n_elements=1, double tolerance = -1)
    {
	assert(n_elements>0);
	if constexpr (DIM==3) {
	    m_base_n_elements[0]=1;
	    m_base_n_elements[1]=2;
	    m_base_n_elements[2]=4;
	}else {
	    m_base_n_elements[0]=1;
	    m_base_n_elements[1]=2;
	}
	m_base_n_elements*=n_elements;	

	std::cout<<"creating new ifgf operator. n_leaf="<<maxLeafSize<<" order= "<<order<<" n_elements="<<n_elements<<std::endl;
        m_src_octree = std::make_unique<Octree<T, DIM> >(maxLeafSize);
	m_target_octree = std::make_unique<Octree<T, DIM> >(maxLeafSize);
	m_baseOrder.fill(order);
	m_tolerance=tolerance;


    }

    ~IfgfOperator()
    {
	std::cout<<"freeing ifgf"<<std::endl;
    }

    const Octree<T,DIM>& src_octree() const {
	return *m_src_octree;
    }



    void init(const PointArray &srcs, const PointArray targets)
    {
        m_src_octree->build(srcs);
	m_target_octree->build(targets);
	
	m_src_octree->buildInteractionList(*m_target_octree);


	static_cast<Derived *>(this)->onOctreeReady();
        //m_src_octree->sanitize();

        m_numTargets = targets.cols();
        m_numSrcs = srcs.cols();

	if(m_tolerance>0) {
	
	    //m_base_n_elements*=estimateRefinement(m_tolerance,RefineH);
	    m_baseOrder=estimateRefinement(m_tolerance,RefineP).template cast<int>();

	    std::cout<<"settled on order="<<m_baseOrder.transpose()<<std::endl;
	}else {
	    m_baseOrder[0]=std::max(m_baseOrder[0]-3,2); 
	}


	std::cout<<"calculating interp range"<<std::endl;
	m_src_octree->calculateInterpolationRange([this](double H,int step){return static_cast<Derived *>(this)->orderForBox(H, m_baseOrder,step);},
						  [this](double H, int step){return static_cast<Derived *>(this)->elementsForBox(H, this->m_baseOrder,this->m_base_n_elements,step);},
						  [this](double H){return static_cast<Derived *>(this)->cutoff_limit(H);},
						  *m_target_octree);

	std::cout<<"done initializing"<<std::endl;
    }


    
    Eigen::Vector<size_t, DIM>  estimateRefinement(double tol,RefinementType refine)
    { 
	std::cout<<"estimating the order needed to achieve "<<tol<< "using "<< (refine==RefineH ? "h":"p")<<"-refinement"<<m_baseOrder<<" "<<m_base_n_elements.transpose()<<std::endl;
	//use n boxes randomly to estimate the interpolation error
	const size_t level=m_src_octree->levels()-1;
	const size_t Nboxes= m_src_octree->numBoxes(level);
	const size_t sampleBoxes=10;
	const size_t stride= Nboxes/sampleBoxes;

	std::cout<<"working on level"<<level<<" "<<m_src_octree->numBoxes(level)<<std::endl;

	auto ref=tbb::parallel_reduce<tbb::blocked_range<size_t>,Eigen::Vector<size_t,DIM> >(
					tbb::blocked_range<size_t>(0,sampleBoxes),
					Eigen::Vector<size_t,DIM>::Zero(),					
					[&](const tbb::blocked_range<size_t>& r, const Eigen::Vector<size_t, DIM>& refinements) {
					    Eigen::Vector<size_t,DIM> tmp=refinements;
					    for(size_t i=r.begin();i<r.end();i++)
					    {
						const size_t boxId=i*stride;
						tmp=refinements.cwiseMax(estimateRefinementOnBox(tol,level,boxId,refine));
					    }
					    return tmp;
					},[](const Eigen::Vector<size_t, DIM>&  a, const Eigen::Vector<size_t, DIM>& b) -> Eigen::Vector<size_t,DIM> {return  a.cwiseMax(b);});

	std::cout<<"using refinemnt="<<ref.transpose()<<std::endl;;
	return ref;
    }

    Eigen::Vector<size_t,DIM> estimateRefinementOnBox(double tol,size_t level,size_t id, RefinementType refine)
    {
	const int maxR=refine== RefineH ? 4 : 10;
	
	BoundingBox bbox = m_src_octree->bbox(level, id);
        auto center = bbox.center();
        double H = bbox.sideLength();
        IndexRange srcs = m_src_octree->points(level, id);
        const size_t nS = srcs.second - srcs.first;
	
	double smax=sqrt(DIM)/DIM;
	//if the sources and targets are well-separated we don't have to cover the near field 
	const double smin=static_cast<Derived *>(this)->cutoff_limit(H);
	std::cout<<"working on smin="<<smin<<" H="<<H<<std::endl;
	
	BoundingBox<DIM> int_box;
	int_box.min()(0)=smin;
	int_box.max()(0)=smax;
	
	//now scale to (smin,smax) x (0,PI) x (-M_PI,M_PI) (in 3d)
	if constexpr(DIM==2) {	    
	    int_box.min()(1)=-M_PI;
	    int_box.max()(1)=M_PI;
	}else{
	    int_box.min()(1)=0;
	    int_box.max()(1)=M_PI;
	    
	    int_box.min()(2)=-M_PI;
	    int_box.max()(2)=M_PI;
	}

	Eigen::Vector<size_t,DIM> base;
	base.fill(0);;
	Eigen::Vector<double, DIM> error;
	error.fill(std::numeric_limits<double>::max());


        Eigen::Vector<T,Eigen::Dynamic> weights=Eigen::Vector<T,Eigen::Dynamic>::Ones(nS);

	
	Eigen::Vector<size_t,DIM> new_n_els;
	Eigen::Vector<int,DIM> new_p;
	
	
	
	while(error.maxCoeff() > tol && base.maxCoeff()<maxR) {	    
	    new_p=refine==RefineP ? m_baseOrder+base.template cast<int>(): m_baseOrder;
	    new_n_els=refine==RefineH ?  m_base_n_elements.array()*Eigen::pow(2*Eigen::Vector<size_t, DIM>::Ones().array(),base.array()) : m_base_n_elements;
	    

	    //we compute a higher order interpolation and then sum over the additional chebychev coefficients to get
	    //a better understanding over the error
	    const int layers=3;
	    const auto order = static_cast<Derived *>(this)->orderForBox(H, new_p.array()+layers+1,1 );

	    //std::cout<<"trying order"<<order<<std::endl;
	    Eigen::Vector<size_t, DIM> n_els= static_cast<Derived *>(this)->elementsForBox(H, new_p.array()+layers+1,  new_n_els,1 );

	    //std::cout<<"trying orer"<<order.transpose()<<" and "<<n_els.transpose()<<"  elements"<<std::endl;

	    ConeDomain<DIM> grid(n_els,int_box);
	    PointArray chebNodes=ChebychevInterpolation::chebnodesNdd<double,DIM>(order);
	    PointArray transformedNodes(DIM,chebNodes.cols());
	    Eigen::Array<T, Eigen::Dynamic, DIMOUT> data(chebNodes.cols(),DIMOUT);
	    Eigen::Array<T, Eigen::Dynamic, DIMOUT> trafo_data(chebNodes.cols(),DIMOUT);
	    
	    error.fill(0);
	    size_t el=(grid.n_elements()*Eigen::Vector<size_t,1>::Random().value())/SIZE_MAX;
	    transformInterpToCart(grid.transform(el,chebNodes), transformedNodes, center, H);

	    data= static_cast<const Derived *>(this)->evaluateFactoredKernel
		(m_src_octree->points(srcs), transformedNodes, weights, center, H,srcs);

	    ChebychevInterpolation::chebtransform<T,DIM>(data,trafo_data,order);
	    for(int d=0;d<DIM;d++) {
		const double e=Util::compute_slice_norm<T,DIM,DIMOUT>(trafo_data,order.template cast<size_t>(), d,layers);
		error[d]=e;
	    }
		
	    double maxe=error.maxCoeff();
	    for(int d=0;d<DIM;d++){
		if(error[d] > m_tolerance ) {
		    base[d]+=1;
		}
	    }
	    //std::cout<<"error="<<error<<std::endl;
	}
	
	return refine == RefineH ? new_n_els : new_p.template cast<size_t>();
    }
    
    


    Eigen::Array<T, Eigen::Dynamic,DIMOUT> mult(const Eigen::Ref<const Eigen::Vector<T, Eigen::Dynamic> > &weights)
    {
	std::cout<<"mult "<<m_baseOrder.transpose()<<std::endl;

	std::cout<<"permutation"<<std::endl;
	
        Eigen::Array<T, Eigen::Dynamic, DIMOUT> result(m_numTargets,DIMOUT);
        result.fill(0);
        int level = m_src_octree->levels() - 1;

	std::cout<<"boxes="<<m_src_octree->numBoxes(level)<<std::endl;
	const double hmin=m_src_octree->diameter()*std::pow(0.5,m_src_octree->levels());
	std::cout<<"base size"<<static_cast<Derived *>(this)->elementsForBox(hmin, m_baseOrder,this->m_base_n_elements).transpose()<<std::endl;
	std::cout<<"now go"<<std::endl;

	//std::vector<tbb::queuing_mutex> resultMutex(m_numTargets);


	sycl::queue Q;
	const auto &device = Q.get_device();
	std::cout << "Running on: "
		  << Q.get_device().get_info<sycl::info::device::name>()
		  <<device.get_info<sycl::info::device::max_compute_units>()
		  << std::endl;

	//push some global data to the GPU
	Eigen::Vector<T, Eigen::Dynamic> new_weights(weights.size());
        Util::copy_with_permutation_rowwise<T,1> (weights.array(), m_src_octree->permutation(),new_weights.array());
	sycl::buffer<const T, 1> b_weights(new_weights.data(),weights.size());
	sycl::buffer<const double, 1> b_srcs(m_src_octree->points().data(),m_src_octree->numPoints()*DIM);
	sycl::buffer<const double, 1> b_targets(m_target_octree->points().data(),m_target_octree->numPoints()*DIM);

	sycl::buffer<T, 1> b_result(result.data(),result.size());


	//TODO remove?
	tbb::enumerable_thread_specific<Eigen::Array<T, Eigen::Dynamic, DIMOUT> > local_result(result);
	tbb::enumerable_thread_specific<Eigen::Array<T, Eigen::Dynamic, DIMOUT> > tmp_result;
	tbb::enumerable_thread_specific<Eigen::Array<T, Eigen::Dynamic, 1 > > tmp_chebt;
	tbb::enumerable_thread_specific<Eigen::Array<T, Eigen::Dynamic, 1 > > tmp_coordTrafo;
	tbb::enumerable_thread_specific<PointArray > transformedNodes;



	    
	
	std::vector<ChebychevInterpolation::InterpolationData<T,DIM,DIMOUT> > interpolationData(m_src_octree->numBoxes(level));
        std::vector<ChebychevInterpolation::InterpolationData<T,DIM,DIMOUT> > parentInterpolationData;

	std::unique_ptr<sycl::buffer<T,1> > interpolationDataBuffer;
	std::unique_ptr<sycl::buffer<T,1> > parentInterpolationDataBuffer;
		
        for (; level >= 0; --level) {
	    	    
            std::cout << "level=" << level << " "<< m_src_octree->numBoxes(level)<< std::endl;
	    
	    std::cout << "near field" <<std::endl;
	    OctreeLevelData<T,DIM> srcData(*m_src_octree,level);

	    

	    
#define SYCL_NF	    
#ifdef SYCL_NF
            Q.submit([&](sycl::handler &h) {
              // start by pushing  some data to the GPU (octree stuff)
              sycl::accessor a_srcs(b_srcs, h, sycl::read_only);
              sycl::accessor a_targets(b_targets, h, sycl::read_only);
              sycl::accessor a_weights(b_weights, h, sycl::read_only);

              sycl::accessor a_result(b_result, h, sycl::read_write);

              const auto &srcDataAcc = srcData.accessor(h);
              const auto functions =
                  static_cast<Derived *>(this)->kernelFunctions();


              auto out = sycl::stream(1024, 768, h);
	      const size_t num_targets=m_target_octree->numPoints();
	     
              h.parallel_for(
                  sycl::range(num_targets),
                  [=](sycl::id<1> i) {
		      //out<<"pnt"<<i<<"\n";
		      for( size_t boxId : srcDataAcc.nearFieldBoxes(i)) {
			  
			  IndexRange srcs = srcDataAcc.points(boxId);	
			  const size_t nS = srcs.second - srcs.first;
			  if (nS == 0) { //skip empty boxes
			      continue;
			  }
			  
			  a_result[i]+=functions.evaluateKernel(a_srcs, srcs.first, srcs.second,
								a_targets, i, a_weights);
		      }
		  });
            });
#else
	    tbb::parallel_for(tbb::blocked_range<size_t>(0, m_target_octree->numPoints()),
	      [&](tbb::blocked_range<size_t> r) {
              for (size_t i = r.begin(); i < r.end(); i++) {
		  for( size_t boxId : m_src_octree->nearFieldBoxes(level,i) ){			  
		      IndexRange srcs = m_src_octree->points(level, boxId);
		      const size_t nS = srcs.second - srcs.first;
		      if(nS==0) { //skip empty boxes
			  continue;
		      }


		      static_cast<Derived *>(this)->template evaluateKernel<1>(
								   m_src_octree->points(srcs),
								   m_target_octree->point(i),
								   new_weights.segment(srcs.first, nS),
								   result.row(i),
								   srcs);
		  }
	      }});
#endif
            Q.wait();

            //Get an exemplary bbox to determine the interpolation order
	    BoundingBox bbox = m_src_octree->bbox(level, 0);
	    double H0 = bbox.sideLength();
	    const auto order = static_cast<Derived *>(this)->orderForBox(H0, m_baseOrder,0);
	    const auto& chebNodes=ChebychevInterpolation::chebnodesNdd<double,DIM>(order);
	    const auto high_order = static_cast<Derived *>(this)->orderForBox(H0, m_baseOrder,1);
	    const auto& ho_chebNodes=ChebychevInterpolation::chebnodesNdd<double,DIM>(high_order);

	    //Cache chebychev nodes on the GPU

	    sycl::buffer<const double,1> b_chebNodes(chebNodes.data(),chebNodes.cols()*DIM);
	    sycl::buffer<const double,1> b_hoChebNodes(ho_chebNodes.data(),ho_chebNodes.cols()*DIM);
	    



            const size_t stride=chebNodes.cols();



	    //there is no more far field or interpolation happening
	    /*if(level<1) {
		break;
		}*/




	    //prepare the interpolation data for all leaves
	    if(level==m_src_octree->levels()-1) {
		std::cout<<"init"<<std::endl;
		initInterpolationData(level,1, interpolationData, interpolationDataBuffer);
	    }


	    //TODO TEMPORARY!!!
	    {
		std::cout<<"making sure intData and intDataBuffer are in sync"<<std::endl;

		sycl::host_accessor a_intData(*interpolationDataBuffer,  sycl::read_write);
		
		tbb::parallel_for(tbb::blocked_range<size_t>(0, m_src_octree->numActiveCones(level,1)),
				  [&](tbb::blocked_range<size_t> r) {
				      for (size_t i = r.begin(); i < r.end(); i++) {
					  ConeRef cone=m_src_octree->activeCone(level,i,1);
					  size_t boxId=cone.boxId();
					  const size_t stride=ho_chebNodes.cols();

					  
					  
					  for(size_t l=0;l<stride;l++) {			    
						  a_intData[cone.globalId()*stride+l]=interpolationData[boxId].values[cone.memId()*stride+l];
					      }
					  


				      }
				  });
	    }

	    //END TODO


	    

	    std::cout<<"interpolate leaves"<<m_src_octree->numLeafCones(level)<<std::endl;

	    if(m_src_octree->numLeafCones(level) > 0)
	    {

#define SYCL_INTERP_LEAVES
#ifdef SYCL_INTERP_LEAVES
		Q.submit([&](sycl::handler &h) {
		    // start by pushing  some data to the GPU (octree stuff)
		    sycl::accessor a_srcs(b_srcs, h, sycl::read_only);
		    sycl::accessor a_targets(b_targets, h, sycl::read_only);
		    sycl::accessor a_weights(b_weights, h, sycl::read_only);

		    sycl::accessor a_intData(*interpolationDataBuffer, h, sycl::read_write);

		    sycl::accessor a_hoChebNodes(b_hoChebNodes, h, sycl::read_only);

		    const auto &srcDataAcc = srcData.accessor(h);
		    const auto functions =
			static_cast<Derived *>(this)->kernelFunctions();


		    const auto& coneMap=srcData.coneMap(0);
		    const size_t stride=ho_chebNodes.cols();


		    const size_t sizeB=interpolationDataBuffer->size();

		    const size_t  numLeafCones=m_src_octree->numLeafCones(level);
		    std::cout<<"numm="<<numLeafCones<<std::endl;
		    auto out = sycl::stream(1024, 1024, h);
		    h.parallel_for(sycl::range<1>( numLeafCones),
				   [=](sycl::id<1> i)
				   {

				       const ConeRef ref=srcDataAcc.leafCone(i);
				       const size_t boxId=ref.boxId();				       
				       if( srcDataAcc.hasFarTargetsIncludingAncestors(boxId)){ // we dont need the interpolation info for those levels.
					   sycl::marray<double, DIM>  center=srcDataAcc.boxCenter(boxId);
					   double H=srcDataAcc.boxSize(boxId);

					   auto grid=srcDataAcc.coneDomain(boxId,1);
					   IndexRange srcs=srcDataAcc.points(boxId);
					   const size_t nS=srcs.second-srcs.first;

					   const size_t offset=ref.globalId()*stride;


					   sycl::marray<double,DIM> transformed;
					   sycl::marray<double,DIM> transformed2;
					   for(size_t j=0;j<stride;j++) {
					       grid.transform(ref.id(),a_hoChebNodes,transformed,j);



					       
					       Util::interpToCart(transformed,transformed2,center,H);


					       //out<<"t"<<i<<" in "<<j<<" at "<<transformed2[0]<<"/"<<transformed2[1]<<"/"<<transformed2[2]<<"\n";
					       a_intData[j+offset]=functions.evaluateFactoredKernel(a_srcs, srcs.first, srcs.second,
					       							    transformed2, a_weights,center, H);

					   }
					     
					     // interpolationData[boxId].values.middleRows(ref.memId()*stride,stride) =
					     // static_cast<const Derived *>(this)
					     // ->evaluateFactoredKernel(m_src_octree->points(srcs),
					     // transformedNodes.local(),
					     // new_weights.segment(srcs.first, nS), center, H,srcs);
				       }

				   });
		 });
#else
                tbb::parallel_for(tbb::blocked_range<size_t>(0, m_src_octree->numLeafCones(level)),
		[&](tbb::blocked_range<size_t> r) {
		for (size_t i = r.begin(); i < r.end(); i++) {
		    const ConeRef ref=m_src_octree->leafCone(level,i);
		    const size_t boxId=ref.boxId();

		    if( ! m_src_octree->hasFarTargetsIncludingAncestors(level, boxId)){ //we dont need the interpolation info for those levels.
			continue;
		    }

		    
		    assert(m_src_octree->isLeaf(level,boxId)==true);
		    BoundingBox bbox = m_src_octree->bbox(level, boxId);
		    auto center = bbox.center();
		    double H = bbox.sideLength();


		    

		    //std::cout<<"H2="<<H<<std::endl;

		    auto grid=m_src_octree->coneDomain(level,boxId, 1);

		    IndexRange srcs=m_src_octree->points(level,boxId);
		    const size_t nS=srcs.second-srcs.first;

		    transformedNodes.local().resize(3,ho_chebNodes.cols());

		    const size_t stride=ho_chebNodes.cols();			
		    transformInterpToCart(grid.transform(ref.id(),ho_chebNodes), transformedNodes.local(), center, H);


		    interpolationData[boxId].values.middleRows(ref.memId()*stride,stride) =
			static_cast<const Derived *>(this)
			->evaluateFactoredKernel(m_src_octree->points(srcs),
						 transformedNodes.local(),
						 new_weights.segment(srcs.first, nS), center, H,srcs);			

		}});
	

#endif	    
	    }

	    //chebtrafo everything
	    std::cout<<"chebtrafo"<<std::endl;
	    const size_t cv_size=high_order.unaryExpr([&](int v){ return v*v; }).sum();



	    

#define SYCL_CHEBTRAFO
#ifdef SYCL_CHEBTRAFO
	    sycl::buffer<double> b_chebvals(cv_size);
	    {
		sycl::host_accessor a_cv(b_chebvals);
		size_t idx=0;
		//make sure the factors for the chebtrafo are precomputed...
		for(int d=DIM-1;d>=0;d--) {
		    std::cout<<"idx="<<idx<<" vs "<<cv_size<<" "<<d<<std::endl;
		    const auto& cv=ChebychevInterpolation::chebvals<double>(high_order[d]);
		    std::copy(cv.reshaped().begin(),cv.reshaped().end(),a_cv.begin()+idx);
		    idx+=high_order[d]*high_order[d];

		}

		assert(idx==cv_size); //check that we initialized correctly (TODO remove)
	    }



	    
	    Q.submit([&](sycl::handler &h) {
		// start by pushing  some data to the GPU (octree stuff)

		sycl::accessor a_intData(*interpolationDataBuffer, h, sycl::read_write);
				


		//TODO unify order type
		std::array<int,DIM> ns_ho;
		std::copy(high_order.begin(),high_order.end(),ns_ho.begin());
		
		
		const sycl::accessor a_chebvals(b_chebvals,h,sycl::read_only);


		const size_t sizeB=interpolationDataBuffer->size();
		const size_t numActiveCones= m_src_octree->numActiveCones(level,1);

		const size_t stride=ho_chebNodes.cols();
		std::cout<<"survived setup"<<std::endl;

		h.parallel_for(sycl::range<1>( numActiveCones),
			       [=](sycl::id<1> i)
		{
		    //before we can use the interpolation data, we habe to run a chebychev transform on it
		    SyclChebychevInterpolation::chebtransform_inplace<T,DIM>( a_intData,  ns_ho, a_chebvals,i*stride);

		});
	    });
#else

	    {
	    sycl::host_accessor a_intData(*interpolationDataBuffer,  sycl::read_only);
	    std::cout<<"copying back to cpu 1"<<std::endl;
	    tbb::parallel_for(tbb::blocked_range<size_t>(0, m_src_octree->numActiveCones(level,1)),
            [&](tbb::blocked_range<size_t> r) {
            for (size_t i = r.begin(); i < r.end(); i++) {
		ConeRef cone=m_src_octree->activeCone(level,i,1);
		size_t boxId=cone.boxId();
		const size_t stride=ho_chebNodes.cols();

		
		if(m_src_octree->isLeaf(level,boxId)) {
		    for(size_t l=0;l<stride;l++) {	       
			interpolationData[boxId].values[cone.memId()*stride+l]=a_intData[cone.globalId()*stride+l];		    
		    }
		}


	    }
	    });
	    }



	    tbb::parallel_for(tbb::blocked_range<size_t>(0, m_src_octree->numActiveCones(level,1)),
            [&](tbb::blocked_range<size_t> r) {
            for (size_t i = r.begin(); i < r.end(); i++) {
		ConeRef cone=m_src_octree->activeCone(level,i,1);
		size_t boxId=cone.boxId();
		const size_t stride=ho_chebNodes.cols();			




		
		if(!m_src_octree->hasPoints(level,boxId))
		    continue;

		
		if( ! m_src_octree->hasFarTargetsIncludingAncestors(level, boxId)){ //we dont need the interpolation info for those levels.
		    continue;
		}


		//before we can use the interpolation data, we habe to run a chebychev transform on it

		tmp_chebt.local().resize(stride);

		ChebychevInterpolation::chebtransform<T,DIM>(interpolationData[boxId].values.middleRows(cone.memId()*stride,stride),tmp_chebt.local(),high_order);
		interpolationData[boxId].values.middleRows(cone.memId()*stride,stride)=tmp_chebt.local();

	    }});
	    
#endif	    
#ifdef SYCL_CHEBTRAFO
	    ///TODO temporary copy back the interpolation data from the GPU to the old style CPU array. That way we can
	    //rely on the old codepath for the rest of the steps 
	    //TODO: TEMP
	    
	    sycl::host_accessor a_intData(*interpolationDataBuffer,  sycl::read_only);
	    std::cout<<"copying back to cpu"<<std::endl;
	    tbb::parallel_for(tbb::blocked_range<size_t>(0, m_src_octree->numActiveCones(level,1)),
            [&](tbb::blocked_range<size_t> r) {
            for (size_t i = r.begin(); i < r.end(); i++) {
		ConeRef cone=m_src_octree->activeCone(level,i,1);
		size_t boxId=cone.boxId();
		const size_t stride=ho_chebNodes.cols();

		
		for(size_t l=0;l<stride;l++) {			
		    interpolationData[boxId].values[cone.memId()*stride+l]=a_intData[cone.globalId()*stride+l];		    
		    
		}


	    }
	    });

	    //END TODO
#endif//SYCL_CHEBTRAFO


	    
	    //interpolation Data contains the values using the coarse- high-order interpolation scheme. Project to the low-order fine grid
	    //which is faster for point-evaluations. We use parentInteprolationData as a termpoary buffer.
	    initInterpolationData(level,0, parentInterpolationData,parentInterpolationDataBuffer);
	    std::cout<<"reinterpolating"<<std::endl;
	    
	    tbb::parallel_for(tbb::blocked_range<size_t>(0, m_src_octree->numActiveCones(level,1)), //iterative over the few-high order cones to take full advantage of the TP structure
		[&](tbb::blocked_range<size_t> r) {
#ifdef USE_NGSOLVE
	    static ngcore::Timer t("ngbem reinterpolate");
	    ngcore::RegionTimer reg(t);
#endif
	    for (size_t i = r.begin(); i < r.end(); i++) {
		    //parent node
		    ConeRef hoCone=m_src_octree->activeCone(level,i,1);
		    size_t boxId=hoCone.boxId();
		    const auto& hoGrid= m_src_octree->coneDomain(level,boxId,1);
		    
		    if (!m_src_octree->hasPoints(level, boxId)) {
			continue;
		    }

		    //BoundingBox region=//hoGrid.region(hoCone.id());


		    if( ! m_src_octree->hasFarTargetsIncludingAncestors(level, boxId)){ //we dont need the interpolation info for those levels.
			continue;
		    }

		    BoundingBox bbox = m_src_octree->bbox(level, boxId);
		    auto center = bbox.center();
		    double H = bbox.sideLength();


		    coarseToFine(interpolationData[boxId], level, hoCone, tmp_result.local(), order, high_order,H,parentInterpolationData[boxId]);
	    }});

	    

	    // chebtrafo everything again now on the finer grid
	    std::cout<<"fine chebtrafo"<<std::endl;
	    tbb::parallel_for(tbb::blocked_range<size_t>(0, m_src_octree->numActiveCones(level,0)),
            [&](tbb::blocked_range<size_t> r) {
            for (size_t i = r.begin(); i < r.end(); i++) {
		ConeRef cone=m_src_octree->activeCone(level,i,0);
		size_t boxId=cone.boxId();
		if(!m_src_octree->hasPoints(level,boxId))
		    continue;
		
		//const size_t stride=chebNodes.cols();			
		//before we can use the interpolation data, we habe to run a chebychev transform on it

		tmp_chebt.local().resize(stride);

		ChebychevInterpolation::chebtransform<T,DIM>(parentInterpolationData[boxId].values.middleRows(cone.memId()*stride,stride),tmp_chebt.local(),order);
		parentInterpolationData[boxId].values.middleRows(cone.memId()*stride,stride)=tmp_chebt.local();
	    }});


	    std::cout<<"swapping"<<std::endl;
	    std::swap(interpolationData,parentInterpolationData);
	    parentInterpolationData.resize(0);

	    std::cout<<"evaluate far field"<<std::endl;
	    tbb::parallel_for(tbb::blocked_range<size_t>(0, m_target_octree->numPoints()),
	        [&](tbb::blocked_range<size_t> r) {
		    tmp_result.local().resize(1,DIMOUT);
		for (size_t i = r.begin(); i < r.end(); i++) {
		    for( size_t boxId : m_src_octree->farfieldBoxes(level,i) ){
			BoundingBox bbox = m_src_octree->bbox(level, boxId);
			auto center = bbox.center();
			double H = bbox.sideLength();

			//evaluate for the cousin targets using the interpolated data

			evaluateSingleFromInterp(interpolationData[boxId], m_target_octree->point(i), center, H,
					   tmp_result.local());

			result.row(i) += tmp_result.local();
		    }
		}});
	
	    /*#ifdef CHECK_CONNECTIVITY
            std::cout<<"connectivity"<<std::endl<<m_connectivity<<std::endl;
	    #endif*/


	    std::cout<<"propagating"<<std::endl;

	    if(level<1) //there is no parent
	    {		
		break;
	    }
	    
            //Now transform the interpolation data to the parents
	    std::cout<<"propagating upward"<<std::endl;
	    initInterpolationData(level-1,1, parentInterpolationData,parentInterpolationDataBuffer);
	    tbb::parallel_for(tbb::blocked_range<size_t>(0, m_src_octree->numActiveCones(level-1,1)),
		[&](tbb::blocked_range<size_t> r) {
		    tmp_result.local().resize(ho_chebNodes.cols(),DIMOUT);
		    
		    transformedNodes.local().resize(3,ho_chebNodes.cols());
#ifdef USE_NGSOLVE
	    static ngcore::Timer t("ngbem propagate up");
	    ngcore::RegionTimer reg(t);
#endif

		for (size_t i = r.begin(); i < r.end(); i++) {
		    //parent node
		    ConeRef parentCone=m_src_octree->activeCone(level-1,i,1);
                    size_t parentId = parentCone.boxId();
		    auto pGrid= m_src_octree->coneDomain(level-1,parentId,1);				    
                    BoundingBox parent_bbox = m_src_octree->bbox(level - 1, parentId);
                    auto parent_center = parent_bbox.center();
                    double pH = parent_bbox.sideLength();

		    if (!m_src_octree->hasPoints(level-1, parentId)) {
			   continue;
		   }


		    if( ! m_src_octree->hasFarTargetsIncludingAncestors(level-1, parentId)){ //we dont need the interpolation info for those levels.
			continue;
		    } 

		    transformInterpToCart(pGrid.transform(parentCone.id(),ho_chebNodes), transformedNodes.local(), parent_center, pH);
		    const size_t stride=ho_chebNodes.cols();

		    //std::cout<<"pc"<<parentCone.id()<<" "<<parentCone.memId()<<" "<<nterpolationData[parentId].values.size()<<std::endl;
		    parentInterpolationData[parentId].values.middleRows(parentCone.memId()*stride,stride).fill(0);
		    for(size_t childBox : m_src_octree->childBoxes(level-1,parentId) ) {
			//current node
			BoundingBox bbox = m_src_octree->bbox(level, childBox);
			auto center = bbox.center();
			double H = bbox.sideLength();
			
			

			transferInterp(interpolationData[childBox], transformedNodes.local(), center, H, parent_center, pH, tmp_result.local());
						
			parentInterpolationData[parentId].values.middleRows(parentCone.memId()*stride,stride)+=tmp_result.local();
		    }
	    }});




	    
            std::swap(interpolationData, parentInterpolationData);

	    interpolationDataBuffer.reset();
	    std::swap(interpolationDataBuffer,parentInterpolationDataBuffer);	    
	    parentInterpolationDataBuffer.reset();
            parentInterpolationData.resize(0);

        }


	Eigen::Array<T, Eigen::Dynamic, DIMOUT> true_result(result.rows(),result.cols());
        Util::copy_with_inverse_permutation_rowwise<T,DIMOUT>(result, m_target_octree->permutation(),true_result);

	return true_result;
    }


    void initInterpolationData(size_t level, size_t step, std::vector<ChebychevInterpolation::InterpolationData<T,DIM,DIMOUT> >& i_data,std::unique_ptr<sycl::buffer<T,1>> & buf )
    {
	assert(level<m_src_octree->levels());
	i_data.resize(m_src_octree->numBoxes(level));


	//get an exemplary box to query the polynomial order
	BoundingBox bbox = m_src_octree->bbox(level , 0);
	//std::cout<<"bbox="<<bbox.min().transpose()<<" "<<bbox.max().transpose()<<std::endl;
	auto center = bbox.center();
	double H = bbox.sideLength();
	auto order = static_cast<Derived *>(this)->orderForBox(H, m_baseOrder,step);

	//make sure no old buffer is around	
	buf=std::make_unique<sycl::buffer<T,1> > (m_src_octree->numActiveCones(level,step)*order.prod());
	// {
	//     auto bufA=buf->get_host_access();
	//     std::fill(bufA.begin(),bufA.end(),0);
	// }

        tbb::parallel_for(
                tbb::blocked_range<size_t>(0, i_data.size()),
                [&](tbb::blocked_range<size_t> r) {
                for (int id = r.begin(); id < r.end(); id++) {

		    
		    auto grid= m_src_octree->coneDomain(level,id,step);		


		    i_data[id].values.resize(grid.activeCones().size()*order.prod());
		    i_data[id].s_values=std::make_unique<sycl::buffer<T,DIMOUT> >(grid.activeCones().size()*order.prod());
		    //i_data[id].values.fill(0);
		    i_data[id].order=order;
		    i_data[id].grid=grid;
	       }
	    });

    }

    void evaluateNearField(size_t level,long int id,
			  const Eigen::Ref<const Eigen::Vector<T,Eigen::Dynamic> >& weights,
			  Eigen::Ref<Eigen::Array<T, Eigen::Dynamic,DIMOUT> > result,
			   Eigen::Array<T, Eigen::Dynamic,DIMOUT> &tmp_result,
			   tbb::queuing_mutex* result_mutex=0
			   )
    {
#ifdef USE_NGSOLVE
      static ngcore::Timer t("ngbem eval Near Field");
      ngcore::RegionTimer reg(t);
#endif
      
	IndexRange srcs = m_src_octree->points(level, id);
	const size_t nS = srcs.second - srcs.first;

	if(nS==0) //skip empty boxes
	    return;

	std::vector<IndexRange> targetList = m_src_octree->nearTargets(level, id);

	for (const auto &targets : targetList) {
	    const size_t nT = targets.second - targets.first;

	    //std::cout<<"srcs="<<srcs.first<<" "<<srcs.second<<" "<<new_weights.size()<<std::endl;
	    //std::cout<<"targets="<<targets.first<<" "<<targets.second<<" "<<std::endl;

	    tmp_result.resize(nT,DIMOUT);
	    tmp_result.fill(0);

	    static_cast<Derived *>(this)->template evaluateKernel<-1>(
							 m_src_octree->points(srcs),
							 m_target_octree->points(targets),
							 weights.segment(srcs.first, nS),
							 tmp_result,srcs);
		    
	    {
		if(result_mutex) {
		    tbb::queuing_mutex::scoped_lock lock(*result_mutex);
		    result.middleRows(targets.first, nT) += tmp_result;
		}else{
		    result.middleRows(targets.first, nT) += tmp_result;
		}
	    }
	    
	}	

    }

    void evaluateFarField(size_t level,long int id,
			  const Eigen::Ref<const Eigen::Vector<T,Eigen::Dynamic> >& weights,
			  Eigen::Ref<Eigen::Array<T, Eigen::Dynamic,DIMOUT> > result,
			  ChebychevInterpolation::InterpolationData<T,DIM,DIMOUT>& interpolationData,
			  Eigen::Array<T, Eigen::Dynamic,DIMOUT> &tmp_result			  
			 )
    {
#ifdef USE_NGSOLVE
      static ngcore::Timer t("ngbem eval Far Field");
      ngcore::RegionTimer reg(t);


#endif
      
	BoundingBox bbox = m_src_octree->bbox(level, id);
        auto center = bbox.center();
        double H = bbox.sideLength();
        
	//evaluate for the cousin targets using the interpolated data
	const std::vector<IndexRange> cousinTargets = m_src_octree->farTargets(level, id);
	for (unsigned int l = 0; l < cousinTargets.size(); l++) {
	    const size_t nT = cousinTargets[l].second - cousinTargets[l].first;
	    
	    tmp_result.resize(nT,DIMOUT);
			
	    evaluateFromInterp(interpolationData, m_target_octree->points(cousinTargets[l]), center, H,
			       tmp_result);
	    
	    {
		//tbb::queuing_mutex::scoped_lock lock(resultMutex[cousinTargets[l].first]);
		result.middleRows(cousinTargets[l].first, nT) += tmp_result;
	    }
	}  
    }



    
    
    void coarseToFine(const ChebychevInterpolation::InterpolationData<T,DIM,DIMOUT>& interpolationData, size_t level, const ConeRef& hoCone, 
		      Eigen::Array<T, Eigen::Dynamic,1>& tmp_result, const Eigen::Vector<int, DIM>& order, const Eigen::Vector<int, DIM>& high_order,double H0,
		      ChebychevInterpolation::InterpolationData<T,DIM,DIMOUT>& result)
    {
#ifdef USE_NGSOLVE
	static ngcore::Timer t("ngbem coarse to fine");
	ngcore::RegionTimer reg(t);
#endif

	size_t boxId=hoCone.boxId();
	const auto& hoGrid= m_src_octree->coneDomain(level,boxId,1);
	
	if (!m_src_octree->hasPoints(level, boxId)) {
	    return;
	}

	//BoundingBox region=//hoGrid.region(hoCone.id());
	if(! m_src_octree->hasFarTargetsIncludingAncestors(level, boxId)){ //we dont need the interpolation info for those levels.
	    return;
	}


	auto fine_N=static_cast<Derived *>(this)->elementsForBox(H0, this->m_baseOrder,this->m_base_n_elements,0);
	Eigen::Vector<size_t,DIM> factor=fine_N.array()/hoGrid.num_elements().array();

	
	std::array<Eigen::Array<double,Eigen::Dynamic,1>, DIM > points;
	size_t Np=1;
	for(int d=0;d<DIM;d++) {
	    auto chebNodes1d=ChebychevInterpolation::chebnodesNdd<double,1>(Eigen::Vector<int,1>(order[d]));
	    points[d].resize(chebNodes1d.size()*factor[d]);

	    Np*=points[d].size();

	    for(int j=0;j<factor[d];j++){
		const auto h=2;
		auto min=-1+(j*(h/((double) factor[d])));
		auto max=(min+(h/((double) factor[d])));
		const double a=0.5*(max-min);
		const double b=0.5*(max+min);


		points[d].segment(j*chebNodes1d.size(),chebNodes1d.size())=(chebNodes1d.array()*a)+b;
	    }
	}

	tmp_result.resize(Np,DIMOUT);
	const size_t ho_stride=high_order.prod();
	ChebychevInterpolation::tp_evaluate<T,DIM,DIMOUT>(points, interpolationData.values.middleRows(hoCone.memId()*ho_stride,ho_stride),tmp_result, high_order);
		    
	
	
	Eigen::Vector<double,DIM> pnt;
	size_t fine_stride=order.prod();
	size_t idx_coarse=0;
	//now distribute the results to the right places. Do it in a slow but safe way		    
	for(int l=0;l<factor[2]*order[2];l++) {
	    for(int j=0;j<factor[1]*order[1];j++) {
		for(int k=0;k<factor[0];k++) //the innermost dimension is continuous in memory so we do things blocked
		{
				
		    auto ho_id=hoGrid.indicesFromId(hoCone.id());

		    const size_t fine_el=
			(ho_id[2]*factor[2]+(l/order[2]))*result.grid.n_elements(1)*result.grid.n_elements(0)+
			(ho_id[1]*factor[1]+(j/order[1]))*result.grid.n_elements(0)+
			(ho_id[0]*factor[0]+(k));
				
		    //auto fine_el2=result[boxId].grid.elementForPoint(pnt);

		    //std::cout<<"fine_el"<<fine_el<<" "<<fine_el2<<" "<<hoCone.id()<<" "<<ho_id.transpose()<<" ljk:"<<l<<" "<<j<<" "<<k<<std::endl;
		    //assert(fine_el==fine_el2);
				
		    if(result.grid.isActive(fine_el)) { //if the cone is not active, discard it
			size_t fine_memId=result.grid.memId(fine_el);
			size_t idx_fine=(l % order[2])*order[0]*order[1]+(j % order[1])*order[0];// + (k % order[0]);

			result.values.middleRows(fine_memId*fine_stride+idx_fine, order[0])=tmp_result.middleRows(idx_coarse,order[0]);
		    }
		    idx_coarse+=order[0];
				
			    
		}

	    }
	}

    }

  
    void transferInterp(const ChebychevInterpolation::InterpolationData<T,DIM,DIMOUT>& data, const Eigen::Ref<const PointArray> &targets,
                        const Eigen::Ref<const Eigen::Vector<double, DIM> > &xc, double H,
                        const Eigen::Ref<const Eigen::Vector<double, DIM> > &p_xc, double pH,
                        Eigen::Ref<Eigen::Array<T, Eigen::Dynamic, DIMOUT> > result) const
    {
#ifdef USE_NGSOLVE
	static ngcore::Timer t("ngbem transfer interp");
	ngcore::RegionTimer reg(t);

#endif

	assert(result.rows()==targets.cols());

        //transform to the child interpolation domain
        PointArray transformed(DIM, targets.cols());
        transformCartToInterp(targets, transformed, xc, H);

	//result.fill(0);
	const size_t stride=data.computeStride();

	//std::cout<<"stride"<<stride<<std::endl;
	const auto& grid=data.grid;

	const int N=targets.cols();
	std::vector<size_t> elIds(N);	    
	for(size_t idx=0;idx<N;idx++) {
	    elIds[idx]=data.grid.elementForPoint(transformed.col(idx));
	}
	std::vector<size_t> perm=Util::sort_with_permutation(elIds.begin(),elIds.end(), [](auto x, auto y){ return x<y;});
	PointArray tmp(DIM,transformed.cols());
	Util::copy_with_permutation_colwise<double,DIM>(transformed,perm,tmp);
	Eigen::Array<T, Eigen::Dynamic, DIMOUT> tmp_result(transformed.cols(),DIMOUT);
	size_t idx=0;
	while (idx<N)
	{
	    size_t nb=1;
	    const size_t el=elIds[perm[idx]];

	    //look if any of the following points are also in this element. that way we can process them together
	    while(idx+nb<transformed.cols() && elIds[perm[idx+nb]]==el) {
		nb++;
	    }

	    if(el==SIZE_MAX) { //cutoff values like that
		tmp_result.middleRows(idx,nb).fill(0);
	    }else{
		const size_t memId=data.grid.memId(el);

		tmp.middleCols(idx,nb)=data.grid.transformBackwards(el,tmp.middleCols(idx,nb));
		ChebychevInterpolation::parallel_evaluate<T, DIM,DIMOUT>(tmp.array().middleCols(idx,nb), data.values.middleRows(memId*stride,stride), tmp_result.middleRows(idx,nb), data.order);
	    }
	    idx+=nb;
	}
	Util::copy_with_inverse_permutation_rowwise<T,DIMOUT>(tmp_result,perm,result);
        
	static_cast<const Derived *>(this)->transfer_factor(targets, xc, H, p_xc, pH, result);	
    }


    //evaluateFromIntepr simplified codepath if we now we are dealing with a single target
    void inline evaluateSingleFromInterp(const ChebychevInterpolation::InterpolationData<T,DIM,DIMOUT>& data,
				  const Eigen::Ref<const Eigen::Array<double,DIM,1> > &target,
				  const Eigen::Ref<const Eigen::Vector<double, DIM> > &xc, double H,
				  Eigen::Ref<Eigen::Array<T, 1, DIMOUT> > result) const
    {

	Eigen::Array<double,DIM, 1> transformed(DIM);
	transformCartToInterp(target, transformed, xc, H);
	const size_t stride=data.computeStride();
	const size_t el=data.grid.elementForPoint(transformed);
	
	if(el==SIZE_MAX) { //cutoff values like that
	    result.fill(0);
	    return;
	}

	const size_t memId=data.grid.memId(el);

	transformed=data.grid.transformBackwards(el,transformed);
	//ChebychevInterpolation::ClenshawEvaluator<T,1, DIM,DIM, DIMOUT, -1,-1,-1> clenshaw;
	//result=clenshaw(transformed, data.values.middleRows(memId*stride,stride),  data.order);
	//TODO port to sycl
	ChebychevInterpolation::parallel_evaluate<T,DIM,DIMOUT>(transformed,data.values.middleRows(memId*stride,stride),result,data.order);
	
	
	const auto cf = static_cast<const Derived *>(this)->CF(target.matrix() - xc);
	result*= cf;

    }


    
    void evaluateFromInterp(const ChebychevInterpolation::InterpolationData<T,DIM,DIMOUT>& data,
                            const Eigen::Ref<const PointArray> &targets,
                            const Eigen::Ref<const Eigen::Vector<double, DIM> > &xc, double H,
                            Eigen::Ref<Eigen::Array<T, Eigen::Dynamic, DIMOUT> > result) const
    {
	if(targets.cols()==0) {
	    return;
	}
	
	//std::cout<<"efromInt"<<data.order<<std::endl;
	//sort the points into the corresponding cones
	const int N=targets.cols();
	PointArray transformed(DIM, targets.cols());
	transformCartToInterp(targets, transformed, xc, H);
	const size_t stride=data.computeStride();
	//std::cout<<"stride"<<stride<<std::endl;
	/*	assert(data.order==7);
		const static Eigen::Vector<double,7> nodes = ChebychevInterpolation::chebnodes1d<double, 7>();
	
	for(size_t idx=0;idx<transformed.cols();idx++) {
	    const size_t el=data.grid.elementForPoint(transformed.col(idx));	    
	    const size_t memId=data.grid.memId(el);

	    transformed.col(idx)=data.grid.transformBackwards(el,transformed.col(idx));
	    result(idx)=ChebychevInterpolation::evaluate_slow<T, 1, 7, DIM>(transformed.array().col(idx), data.values.segment(memId*stride,stride),nodes)[0];
	    const auto cf = static_cast<const Derived *>(this)->CF(targets.col(idx) - xc);
            result[idx] *= cf;
	}*/


	if(true) {
	    size_t idx=0;
	    size_t nextElement=data.grid.elementForPoint(transformed.col(0));
	    while (idx<N)
	    {
		size_t nb=0;
		const size_t el=nextElement;

		//transformed.col(idx)=data.grid.transformBackwards(el,transformed.col(idx));
		//look if any of the following points are also in this element. that way we can process them together
		while(nextElement==el) {
		    nb++;
		    if(idx+nb<transformed.cols())  {
			nextElement=data.grid.elementForPoint(transformed.col(idx+nb));
		    }else{
			break;
		    }
		    //transformed.col(idx+nb)=data.grid.transformBackwards(el,transformed.col(idx+nb));		
		}


		if(el==SIZE_MAX) { //cutoff values like that
		    result.middleRows(idx,nb).fill(0);
		}else{
		    const size_t memId=data.grid.memId(el);

		    transformed.middleCols(idx,nb)=data.grid.transformBackwards(el,transformed.middleCols(idx,nb));
		    //result.row(idx)=ChebychevInterpolation::evaluate_clenshaw<T, 1,DIM,DIMOUT>(transformed.array().middleCols(idx,nb), data.values.middleRows(memId*stride,stride),  data.order);
		    ChebychevInterpolation::parallel_evaluate<T, DIM,DIMOUT>(transformed.array().middleCols(idx,nb), data.values.middleRows(memId*stride,stride), result.middleRows(idx,nb), data.order);
		}
		idx+=nb;
	    }
	    for (unsigned int j = 0; j < targets.cols(); j++) {
		const auto cf = static_cast<const Derived *>(this)->CF(targets.col(j).matrix() - xc);
		result.row(j) *= cf;
	    }

	}else{
	    std::vector<int> elIds(N);
	    for(size_t idx=0;idx<N;idx++) {
		elIds[idx]=data.grid.elementForPoint(transformed.col(idx));
	    }
	    std::vector<size_t> perm=Util::sort_with_permutation(elIds.begin(),elIds.end(), [](auto x, auto y){ return x<y;});
	    PointArray tmp(DIM,transformed.cols());
	    Util::copy_with_permutation_colwise<double,DIM>(transformed,perm,tmp);
	    Eigen::Array<T, Eigen::Dynamic, DIMOUT> tmp_result(transformed.cols(),DIMOUT);
	    size_t idx=0;
	    while (idx<N)
	    {
		size_t nb=1;
		const size_t el=elIds[perm[idx]];

		//look if any of the following points are also in this element. that way we can process them together
		while(idx+nb<transformed.cols() && elIds[perm[idx+nb]]==el) {
		    nb++;
		}
		if(el==SIZE_MAX) { //cutoff values like that
		    tmp_result.middleRows(idx,nb).fill(0);
		}else{
		    const size_t memId=data.grid.memId(el);

		    //tmp.middleCols(idx,nb)=data.grid.transformBackwards(el,tmp.middleCols(idx,nb));
		    ChebychevInterpolation::parallel_evaluate<T, DIM,DIMOUT>(tmp.array().middleCols(idx,nb), data.values.middleRows(memId*stride,stride), tmp_result.middleRows(idx,nb), data.order,
									     data.grid.region(el));
		}
		idx+=nb;
	    }

	    Util::copy_with_inverse_permutation_rowwise<T,DIMOUT>(tmp_result,perm,result);
	    for (unsigned int j = 0; j < targets.cols(); j++) {
		const auto cf = static_cast<const Derived *>(this)->CF(targets.col(j).matrix() - xc);
		result.row(j) *= cf;
	    }

	}

	
	// size_t idx=0;
	// while (idx<transformed.cols())
	// {
	//     //std::cout<<"idx"<<idx<<std::endl;
	//     size_t nb=1;	    
	//     const size_t el=data.grid.elementForPoint(transformed.col(idx));	    
	//     const size_t memId=data.grid.memId(el);
	   

	//     //std::cout<<"el="<<el<<" "<<transformed.col(idx)<<std::endl;
	//     transformed.col(idx)=data.grid.transformBackwards(el,transformed.col(idx));
	//     //look if any of the following points are also in this elemnt. that way we can process them together
	//     while(idx+nb<transformed.cols() && data.grid.elementForPoint(transformed.col(idx+nb))==el) {
	// 	transformed.col(idx+nb)=data.grid.transformBackwards(el,transformed.col(idx+nb));
	// 	nb++;
	//     }
	//     ChebychevInterpolation::parallel_evaluate<T, DIM>(transformed.array().middleCols(idx,nb), data.values.segment(memId*stride,stride), result.segment(idx,nb), data.order);
	//     idx+=nb;
	// }
	
		
	//std::cout<<"done"<<data.order<<std::endl;
    }


    inline void transformCartToInterp(const Eigen::Ref<const PointArray > &nodes,
				      Eigen::Ref<PointArray > transformed, const Eigen::Vector<double, DIM> &xc, double H) const
    {
	Util::cartToInterp2<DIM>(nodes.array(), xc, H,transformed);
        /*for (int i = 0; i < nodes.cols(); i++) {
            transformed.col(i) = Util::cartToInterp<DIM>(nodes.col(i), xc, H);
	    }*/
    }

    inline void transformInterpToCart(const Eigen::Ref<const PointArray > &nodes,
                               Eigen::Ref<PointArray > transformed, const Eigen::Vector<double, DIM> &xc, double H) const
    { 

	transformed = Util::interpToCart<DIM>(nodes.array(), xc, H);
       /*for (int i = 0; i < nodes.cols(); i++) {
	 transformed.col(i) = Util::interpToCart<DIM>(nodes.col(i), xc, H);
	 }*/
    }


    inline  double  cutoff_limit(double H) const
    {
	return 1e-4;
    }


    inline double tolerance() const {
	return m_tolerance;
    }

protected:
    void onOctreeReady()
    {
	//do nothing. but give subclasses the opportunity to initialize some things
    }

private:
    std::unique_ptr<Octree<T, DIM> > m_src_octree;
    std::unique_ptr<Octree<T, DIM> > m_target_octree;
    unsigned int m_numTargets;
    unsigned int m_numSrcs;
    Eigen::Vector<size_t, DIM> m_base_n_elements;
    Eigen::Vector<int, DIM> m_baseOrder;
    double m_tolerance;
};

#endif
