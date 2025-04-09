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
#include "sycl_helpers.hpp"

//#include <fstream>
#include <iostream>

#include <memory>

template<typename T, unsigned int DIM, unsigned int DIMOUT, typename Derived>
class IfgfOperator
{
public:
    typedef Eigen::Array<PointScalar, DIM, Eigen::Dynamic> PointArray;     //, Eigen::RowMajor?

    enum RefinementType { RefineH, RefineP};

    IfgfOperator(long int maxLeafSize = -1, size_t order=5, size_t n_elements=1, PointScalar tolerance = -1)
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
	    m_baseOrder[0]=std::max(m_baseOrder[0]-2,2); 
	}


	std::cout<<"calculating interp range"<<std::endl;
	m_src_octree->calculateInterpolationRange([this](PointScalar H,int step){return static_cast<Derived *>(this)->orderForBox(H, m_baseOrder,step);},
						  [this](PointScalar H, int step){return static_cast<Derived *>(this)->elementsForBox(H, this->m_baseOrder,this->m_base_n_elements,step);},
						  [this](PointScalar H){return static_cast<Derived *>(this)->cutoff_limit(H);},
						  *m_target_octree);


#ifdef KEEP_LEVEL_DATA
	//copy to level data	
	m_octreeData.resize(m_src_octree->levels());
	for(int level=0;level<m_src_octree->levels();level++) {
	    m_octreeData[level]=std::make_unique<OctreeLevelData<T,DIM> >(*m_src_octree,level);
	}
#endif

	
	std::cout<<"done initializing"<<std::endl;
    }


    
    Eigen::Vector<size_t, DIM>  estimateRefinement(PointScalar tol,RefinementType refine)
    {
	#if 0
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
	#endif
	
    }



    Eigen::Array<T, Eigen::Dynamic,DIMOUT> mult(const Eigen::Ref<const Eigen::Vector<T, Eigen::Dynamic> > &weights)
    {
	
        Eigen::Array<T, Eigen::Dynamic, DIMOUT> result(m_numTargets,DIMOUT);
        result.fill(0);
        int level = m_src_octree->levels() - 1;

	//std::cout<<"boxes="<<m_src_octree->numBoxes(level)<<std::endl;
	const PointScalar hmin=m_src_octree->diameter()*std::pow(0.5,m_src_octree->levels());
	//std::vector<tbb::queuing_mutex> resultMutex(m_numTargets);

        { //scope to contain all the sycl stuff. that way we make sure that all the data is copied to the host before proceeding.

	sycl::queue& Q=SyclHelpers::QueueSingleton::getInstance().queue();//(sycl::default_selector_v);

        //push some global data to the GPU
	Eigen::Vector<T, Eigen::Dynamic> new_weights(weights.size());
        Util::copy_with_permutation_rowwise<T,1> (weights.array(), m_src_octree->permutation(),new_weights.array());
	sycl::buffer<const T, 1> b_weights(new_weights.data(),weights.size());
	sycl::buffer<const PointScalar, 1> b_srcs(m_src_octree->points().data(),m_src_octree->numPoints()*DIM);
	sycl::buffer<const PointScalar, 1> b_targets(m_target_octree->points().data(),m_target_octree->numPoints()*DIM);

	sycl::buffer<T, 1> b_result(result.data(),result.size());       

	std::unique_ptr<sycl::buffer<T,1> > interpolationDataBuffer;
	std::unique_ptr<sycl::buffer<T,1> > parentInterpolationDataBuffer;

	std::unique_ptr<OctreeLevelData<T,DIM> > parentData;
	std::unique_ptr<OctreeLevelData<T,DIM> > srcData;

        for (; level >= 0; --level) {
	    if(parentData==0) {
#ifdef KEEP_LEVEL_DATA	   
		srcData = m_octreeData[level];//std::make_unique< OctreeLevelData<T,DIM> >(*m_src_octree,level);
#else
		srcData = std::make_unique< OctreeLevelData<T,DIM> >(*m_src_octree,level);
#endif
	    }else {
		std::swap(parentData,srcData);
		parentData.reset();
	    }

	    //std::cout<<"created ocdata"<<std::endl;
	    
	    {
		//std::cout<<"nearfield"<<std::endl;
		Q.submit([&](sycl::handler &h) {
		    // start by pushing  some data to the GPU (octree stuff)
		    sycl::accessor a_srcs(b_srcs, h, sycl::read_only);
		    sycl::accessor a_targets(b_targets, h, sycl::read_only);
		    sycl::accessor a_weights(b_weights, h, sycl::read_only);

		    sycl::accessor a_result(b_result, h, sycl::read_write);

		    const auto &srcDataAcc = srcData->accessor(h);
		    const auto functions =
			static_cast<Derived *>(this)->kernelFunctions();


		    auto out = sycl::stream(1024, 768, h);
		    const size_t num_targets=m_target_octree->numPoints();

		    //std::cout<<"setup complete"<<num_targets<<std::endl;

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

	    }
            Q.wait();

            //Get an exemplary bbox to determine the interpolation order
	    BoundingBox bbox = m_src_octree->bbox(level, 0);
	    PointScalar H0 = bbox.sideLength();
	    const auto order = static_cast<Derived *>(this)->orderForBox(H0, m_baseOrder,0);
	    const auto& chebNodes=ChebychevInterpolation::chebnodesNdd<PointScalar,DIM>(order);
	    const auto high_order = static_cast<Derived *>(this)->orderForBox(H0, m_baseOrder,1);
	    const auto& ho_chebNodes=ChebychevInterpolation::chebnodesNdd<PointScalar,DIM>(high_order);

	    //Cache chebychev nodes on the GPU

	    sycl::buffer<const PointScalar,1> b_chebNodes(chebNodes.data(),chebNodes.cols()*DIM);
	    sycl::buffer<const PointScalar,1> b_hoChebNodes(ho_chebNodes.data(),ho_chebNodes.cols()*DIM);


            const size_t stride=chebNodes.cols();




	    //prepare the interpolation data for all leaves
	    if(level==m_src_octree->levels()-1) {
		//std::cout<<"init"<<std::endl;
		initInterpolationData(level,1, interpolationDataBuffer);
	    }

    

	    //std::cout<<"interpolate leaves"<<m_src_octree->numLeafCones(level)<<std::endl;

	    if(m_src_octree->numLeafCones(level) > 0)
	    {

		{
		Q.submit([&](sycl::handler &h) {
		    // start by pushing  some data to the GPU (octree stuff)
		    sycl::accessor a_srcs(b_srcs, h, sycl::read_only);
		    sycl::accessor a_targets(b_targets, h, sycl::read_only);
		    sycl::accessor a_weights(b_weights, h, sycl::read_only);

		    sycl::accessor a_intData(*interpolationDataBuffer, h, sycl::read_write);

		    sycl::accessor a_hoChebNodes(b_hoChebNodes, h, sycl::read_only);

		    const auto &srcDataAcc = srcData->accessor(h);
		    const auto functions =
			static_cast<Derived *>(this)->kernelFunctions();



		    const size_t stride=ho_chebNodes.cols();


		    const size_t sizeB=interpolationDataBuffer->size();

		    const size_t  numLeafCones=m_src_octree->numLeafCones(level);
		    //std::cout<<"numm="<<numLeafCones<<std::endl;
		    auto out = sycl::stream(1024, 1024, h);
		    h.parallel_for(sycl::range<1>( numLeafCones),
				   [=](sycl::id<1> i)
				   {

				       const ConeRef ref=srcDataAcc.leafCone(i);
				       const size_t boxId=ref.boxId();				       
				       if( srcDataAcc.hasFarTargetsIncludingAncestors(boxId)){ // we dont need the interpolation info for those levels.
					   sycl::marray<PointScalar, DIM>  center=srcDataAcc.boxCenter(boxId);
					   PointScalar H=srcDataAcc.boxSize(boxId);

					   auto grid=srcDataAcc.coneDomain(boxId,1);
					   IndexRange srcs=srcDataAcc.points(boxId);
					   const size_t nS=srcs.second-srcs.first;

					   const size_t offset=ref.globalId()*stride;
					   

					   sycl::marray<PointScalar,DIM> transformed;
					   sycl::marray<PointScalar,DIM> transformed2;
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
		Q.wait();
		}
	    }

	    //chebtrafo everything
	    //std::cout<<"chebtrafo"<<std::endl;


	    {
		const size_t cv_size=high_order.unaryExpr([&](int v){ return v*v; }).sum();
		sycl::buffer<PointScalar> b_chebvals(cv_size);
		{
		    sycl::host_accessor a_cv(b_chebvals);
		    size_t idx=0;
		    //make sure the factors for the chebtrafo are precomputed...
		    for(int d=DIM-1;d>=0;d--) {
			//std::cout<<"idx="<<idx<<" vs "<<cv_size<<" "<<d<<std::endl;
			const auto& cv=ChebychevInterpolation::chebvals<PointScalar>(high_order[d]);
			std::copy(cv.reshaped().begin(),cv.reshaped().end(),a_cv.begin()+idx);
			idx+=high_order[d]*high_order[d];

		    }

		    assert(idx==cv_size); //check that we initialized correctly (TODO remove)
		}



		Q.wait();
		Q.submit([&](sycl::handler &h) {
		    // start by pushing  some data to the GPU (octree stuff)

		    sycl::accessor a_intData(*interpolationDataBuffer, h, sycl::read_write);
				

		    //TODO unify order type
		    std::array<int,DIM> ns_ho;
		    std::copy(high_order.begin(),high_order.end(),ns_ho.begin());
		
		
		    const sycl::accessor a_chebvals(b_chebvals,h,sycl::read_only);
		    const auto &srcDataAcc = srcData->accessor(h);

		    const size_t sizeB=interpolationDataBuffer->size();
		    const size_t numActiveCones= m_src_octree->numActiveCones(level,1);

		    const size_t stride=ho_chebNodes.cols();
		    //std::cout<<"survived setup"<<std::endl;
		    auto out = sycl::stream(1024, 1024, h);


		    h.parallel_for(sycl::range<1>( numActiveCones),
				   [=](sycl::id<1> i)
				   {
				       
				       const ConeRef ref=srcDataAcc.activeCone(i);
				       const size_t boxId=ref.boxId();				       

				       if( srcDataAcc.hasFarTargetsIncludingAncestors(boxId)){ // we dont need the interpolation info for those levels.
					   //out<<"i="<<i<<"\n";
					   //before we can use the interpolation data, we habe to run a chebychev transform on it
					   SyclChebychevInterpolation::chebtransform_inplace<T,DIM>( a_intData,  ns_ho, a_chebvals,i*stride);
					   //out<<"i2="<<a_intData[i*stride].real()<<"\n";
				       }
				   });
		});
		Q.wait();
	    }
	    initInterpolationData(level,0, parentInterpolationDataBuffer);
	    {
		//interpolation Data contains the values using the coarse- high-order interpolation scheme. Project to the low-order fine grid
		//which is faster for point-evaluations. We use parentInteprolationData as a termpoary buffer.

		//std::cout<<"reinterpolating"<<std::endl;

		//first set up some common data structures
		{
		    auto fine_N=static_cast<Derived *>(this)->elementsForBox(H0, this->m_baseOrder,this->m_base_n_elements,0);
		    const auto& hoGrid= m_src_octree->coneDomain(level,0,1);
		    Eigen::Vector<size_t,DIM> factor=fine_N.array()/hoGrid.num_elements().array();


		    //std::cout<<"fine:"<<fine_N<<std::endl;

		    size_t Ntp=order.sum();

		    Eigen::Array<PointScalar,Eigen::Dynamic,1>  points(Ntp);

		    std::array<int, DIM> ns=SyclHelpers::EigenVectorToCPPArray<int, DIM>(high_order);
		    std::array<int, DIM> lo_ns=SyclHelpers::EigenVectorToCPPArray<int, DIM>(order);
		    std::array<int, DIM> factors=SyclHelpers::EigenVectorToCPPArray<int,DIM>(factor.template cast<int>());
		    std::array<int, DIM> n_elements=SyclHelpers::EigenVectorToCPPArray<int,DIM>(fine_N.template cast<int>());
		

		    //std::cout<<"stuff; "<<ns<<" "<<lo_ns<<" "<<factors<<" "<<n_elements<<std::endl;
		    size_t Np=1;
		    size_t offset=0;
		    size_t buffer_size=1;
		    //store the points for the inner most dimension separately from the others
		    for(int d=0;d<DIM;d++) {
			const auto& chebNodes1d=ChebychevInterpolation::chebnodesNdd<PointScalar,1>(Eigen::Vector<int,1>(order[d]));

			points.segment(offset,chebNodes1d.size())=chebNodes1d.array();
			offset+=chebNodes1d.size();		    

		    }

		    for(int d=DIM-1;d>0;d--) {
			int Np=1;
			for(int j=0;j<d;j++)
			{
			    Np*=order[j];
			}
			buffer_size+=high_order[d]*Np;
		    }
		
		    const size_t ho_stride=high_order.prod();
		    sycl::buffer<PointScalar> b_points(points.data(),points.size());

		    // Make sure the size is not too large
		    //auto has_local_mem = (device.get_info<sycl::info::device::local_mem_type>() != sycl::info::local_mem_type::none);
		    //auto local_mem_size = device.get_info<sycl::info::device::local_mem_size>();

		    
		    const size_t cv_size=order.unaryExpr([&](int v){ return v*v; }).sum();
                    sycl::buffer<PointScalar> b_chebvals(cv_size);
		    {
			sycl::host_accessor a_cv(b_chebvals);
			size_t idx=0;
			//make sure the factors for the chebtrafo are precomputed...
			for(int d=DIM-1;d>=0;d--) {
			    //std::cout<<"idx="<<idx<<" vs "<<cv_size<<" "<<d<<std::endl;
			    const auto& cv=ChebychevInterpolation::chebvals<PointScalar>(order[d]);
			    std::copy(cv.reshaped().begin(),cv.reshaped().end(),a_cv.begin()+idx);
			    idx+=order[d]*order[d];

			}

			assert(idx==cv_size); //check that we initialized correctly (TODO remove)
		    }





		    //std::cout<<"buffer="<<buffer_size<<std::endl;

	    
		    const size_t numActiveCones= m_src_octree->numActiveCones(level,1);
		    if(numActiveCones == 0)		    
		        continue;

		    Q.submit([&](sycl::handler &h) {
			sycl::accessor a_intData(*interpolationDataBuffer, h, sycl::read_only);
			sycl::accessor a_parentIntData(*parentInterpolationDataBuffer, h, sycl::read_write);
		    
			sycl::accessor a_points(b_points,h, sycl::read_only);

				    
			
			const sycl::accessor a_chebvals(b_chebvals,h,sycl::read_only);


			size_t fine_stride=order.prod();

			const auto &srcDataAcc = srcData->accessor(h);

			std::array<size_t, DIM> n_el=SyclHelpers::EigenVectorToCPPArray<size_t,DIM>(hoGrid.num_elements());
			auto out = sycl::stream(100, 100, h);


			sycl::accessor a_chebNodes(b_chebNodes,h,sycl::read_only);



			const int nF=factor.prod();
			//std::cout<<"doing it"<<std::endl;
			h.parallel_for(sycl::range( {numActiveCones*nF}), [=](auto it)			
			{		
			    
			    const size_t coneId=it/nF;		
			    
			    
			    ConeRef hoCone=srcDataAcc.activeCone(coneId);
			    if( srcDataAcc.hasFarTargetsIncludingAncestors(hoCone.boxId())){ // we dont need the interpolation info for those levels.
				auto ho_id=SyclConeDomain<DIM>::indicesFromId(hoCone.id(),n_el);

				auto lid=SyclConeDomain<DIM>::indicesFromId(it%nF,factors);
				const size_t fine_el=
				    (ho_id[2]*factors[2]+(lid[2]))*n_elements[1]*n_elements[0]+
				    (ho_id[1]*factors[1]+(lid[1]))*n_elements[0]+
				    (ho_id[0]*factors[0]+(lid[0]));
		       
				const size_t fineMemId=srcDataAcc.memId(hoCone.boxId(),fine_el);
						
				if(fineMemId<SIZE_MAX-1) { ///the target cone is active!
				    for(int i=0;i<fine_stride;i++) {
					a_parentIntData[fineMemId*fine_stride+i]=0;
				    }
				
					
				    //out<<it<<" "<<coneId<<"\n";
				    size_t offset=0;
				    const int MAX_LOW_ORDER=8;
				    sycl::marray<PointScalar,MAX_LOW_ORDER*DIM> t_pnts;
				    sycl::marray<T,541> tmp; //Temporary storage for the sum-factorization. This should be enough up to orders (8,10,10) (5,7,7)*3
			


				    tmp=0;
				    t_pnts=0;		//Fill up the remaining points. otherwise the compiler optimization breaks the code
				    for(int d=0;d<DIM;d++) {
					const PointScalar h=2;
					assert(lo_ns[d]<=MAX_LOW_ORDER);
				    
					const PointScalar mmin=-1+(lid[d]*(h/((PointScalar) factors[d])));
					const PointScalar mmax=(mmin+(h/((PointScalar) factors[d])));
					const PointScalar a=0.5*(mmax-mmin);
					const PointScalar b=0.5*(mmax+mmin);

					
					for(size_t l=0;l<lo_ns[d];l++) {
					    t_pnts[offset]=a*a_points[offset]+b;
					    offset++;
					}
				    }
		    								    

				    const size_t int_data_offset=coneId*ho_stride;
				    
				    SyclChebychevInterpolation::tp_evaluate_t<T,DIM>(t_pnts, a_intData, int_data_offset,
										     ns,lo_ns, a_parentIntData,
										     tmp,
										     fineMemId*fine_stride, 0);


				    //and chebtrafo all in one go
				    SyclChebychevInterpolation::chebtransform_inplace<T,DIM>( a_parentIntData,  lo_ns, a_chebvals,fineMemId*fine_stride);
				    
				
				}

			    }
		    
			});		  
		    
		    });


		}
	    }


	    Q.wait();

	    std::swap(interpolationDataBuffer,parentInterpolationDataBuffer);
	    parentInterpolationDataBuffer.reset();


	    Q.wait();
	    Q.submit([&](sycl::handler &h) {
		sycl::accessor a_intData(*interpolationDataBuffer, h, sycl::read_only);
		sycl::accessor a_targets(b_targets,h, sycl::read_only);

		sycl::accessor a_result(b_result, h, sycl::read_write);

		std::array<int, DIM> ns=SyclHelpers::EigenVectorToCPPArray<int, DIM>(order);


		const auto &srcDataAcc = srcData->accessor(h);
		const auto functions =
		    static_cast<Derived *>(this)->kernelFunctions();
		
		const size_t stride=order.prod();
		auto out = sycl::stream(100, 100, h);

		h.parallel_for(sycl::range{m_target_octree->numPoints()}, [=](auto it)
		{
		    for( size_t boxId : srcDataAcc.farfieldBoxes(it) ){
			auto grid=srcDataAcc.coneDomain(boxId,0);
			auto center = srcDataAcc.boxCenter(boxId);

			PointScalar H = srcDataAcc.boxSize(boxId);

			//evaluate for the cousin targets using the interpolated data

			sycl::marray<PointScalar,DIM> target_pnt;
			for(int j=0;j<DIM;j++)
			    target_pnt[j]=a_targets[it*DIM+j];
			sycl::marray<PointScalar,DIM> transformed;

			const auto cf = functions.CF(target_pnt - center);
						

			transformed=0;

			Util::cartToInterp(target_pnt,transformed,center,H);
			
			//out<<transformed[0]<<" "<<transformed[1]<<" "<<transformed[2]<<"\n";
                        const size_t el=grid.elementForPoint(transformed);
			
			//assert(el<SIZE_MAX); //we used to cutoff targets like that
			if(el== SIZE_MAX) {
			    return;
			}

			
			const size_t memId=srcDataAcc.memId(boxId,el);
			if(memId==SIZE_MAX){
			    return;
			}
			assert(memId<SIZE_MAX); //the requested element should always be active since it contains a target point

			target_pnt=0;

			grid.transformBackwards(el,transformed,target_pnt);
						
			
			SyclChebychevInterpolation::ClenshawEvaluator<T,1, DIM,DIM, DIMOUT> clenshaw;
			const size_t offset=stride*memId;
			T res=clenshaw(SyclRowMatrix<PointScalar, DIM,1>(target_pnt), a_intData, ns, offset);
			
			
			
			res*= cf;

			a_result[it]+=res;
		    }

		});
	    });


	    if(level<1) //there is no parent
	    {		
		break;
	    }
	    
            //Now transform the interpolation data to the parents
	    //std::cout<<"propagating upward"<<std::endl;

	    initInterpolationData(level-1,1, parentInterpolationDataBuffer);

	    const size_t numActiveParentCones= m_src_octree->numActiveCones(level-1,1);
	    if(numActiveParentCones==0) {
		continue;
	    }

	    Q.wait();
#ifdef KEEP_LEVEL_DATA
	    parentData = m_octreeData[level-1];//std::make_unique< OctreeLevelData<T,DIM> >(*m_src_octree,level-1);
#else
	    parentData = std::make_unique< OctreeLevelData<T,DIM> >(*m_src_octree,level-1);
#endif
	    Q.submit([&](sycl::handler &h) {
		// start by pushing  some data to the GPU (octree stuff)

		sycl::accessor a_intData(*interpolationDataBuffer, h, sycl::read_only);
		sycl::accessor a_parentIntData(*parentInterpolationDataBuffer, h, sycl::read_write);
		

		std::array<int,DIM> ns=SyclHelpers::EigenVectorToCPPArray<int,DIM>(order);
		std::array<int,DIM> ns_ho=SyclHelpers::EigenVectorToCPPArray<int,DIM>(high_order);

		

		sycl::accessor a_hoChebNodes(b_hoChebNodes,h,sycl::read_only);
		
		const auto &srcDataAcc = srcData->accessor(h);
		const auto &parentDataAcc = parentData->accessor(h);
	
		
		const size_t stride=ho_chebNodes.cols();
		const size_t lo_stride=chebNodes.cols();
		auto out = sycl::stream(100, 100, h);
		//std::cout<<"survived setup1234"<<std::endl;

		const auto functions =
		    static_cast<Derived *>(this)->kernelFunctions();


		    
		h.parallel_for(sycl::range<1>( numActiveParentCones), [=](sycl::id<1> i)
		{
		    ConeRef parentCone=parentDataAcc.activeCone(i); //parent!!
		    size_t parentBoxId = parentCone.boxId();
		    auto pGrid= parentDataAcc.coneDomain(parentBoxId,1);//m_src_octree->coneDomain(level-1,parentId,1);				    
                    auto parent_center = parentDataAcc.boxCenter(parentBoxId);
                    PointScalar pH = parentDataAcc.boxSize(parentBoxId);


		    if( ! parentDataAcc.hasFarTargetsIncludingAncestors(parentBoxId)){ //we dont need the interpolation info for those levels.
			return;
		    }

		    for(size_t j=0;j<stride;j++) {
			sycl::marray<PointScalar,DIM> pnt;
			sycl::marray<PointScalar,DIM> cart_pnt;
			sycl::marray<PointScalar,DIM> pnt2;
			
			//std::copy(a_hoChebNodes.begin()+j*DIM,a_hoChebNodes.begin()+(j+1)*DIM,pnt.begin());
			pGrid.transform(parentCone.id(),a_hoChebNodes,pnt,j);
			Util::interpToCart(pnt,cart_pnt,parent_center,pH);


			a_parentIntData[i*stride+j]=0;
			for(size_t childBox : parentDataAcc.children(parentBoxId)) {
			    if(childBox==SIZE_MAX) {
				continue;
			    }

			    auto center = srcDataAcc.boxCenter(childBox);
			    PointScalar H = srcDataAcc.boxSize(childBox);
			    const auto grid=srcDataAcc.coneDomain(childBox,0);
			    
			    //Transfer to the interpolation domain relative to the child box
			    Util::cartToInterp(cart_pnt,pnt2,center,H);
			
			    const size_t el=grid.elementForPoint(pnt2);
			
			    assert(el<SIZE_MAX); //we used to cutoff targets like that

			    const size_t memId=srcDataAcc.memId(childBox,el);
			    if(memId < SIZE_MAX) {				
				grid.transformBackwards(el,pnt2,pnt);						
			
				SyclChebychevInterpolation::ClenshawEvaluator<T,1, DIM,DIM, DIMOUT> clenshaw;
				const size_t offset=lo_stride*memId;
				T res=clenshaw(SyclRowMatrix<PointScalar, DIM,1>(pnt), a_intData, ns, offset);

				T TF=functions.transfer_factor(cart_pnt,center,H,parent_center,pH);
			    
				a_parentIntData[i*stride+j]+=res*TF;
			    }
			}
		    }


		});});

	//std::cout<<"done"<<std::endl;

	    Q.wait();

	    
            //std::swap(interpolationData, parentInterpolationData);

	    interpolationDataBuffer.reset();
	    std::swap(interpolationDataBuffer,parentInterpolationDataBuffer);	    
	    parentInterpolationDataBuffer.reset();
            //parentInterpolationData.resize(0);

	    //std::cout<<"done with this level"<<std::endl;
        }
        } //end of sycl scope
	//std::cout<<"mult over\n";

	Eigen::Array<T, Eigen::Dynamic, DIMOUT> true_result(result.rows(),result.cols());
        Util::copy_with_inverse_permutation_rowwise<T,DIMOUT>(result, m_target_octree->permutation(),true_result);

	return true_result;
    }


    void initInterpolationData(size_t level, size_t step, std::unique_ptr<sycl::buffer<T,1>> & buf )
    {
	assert(level<m_src_octree->levels());


	//get an exemplary box to query the polynomial order
	BoundingBox bbox = m_src_octree->bbox(level , 0);
	//std::cout<<"bbox="<<bbox.min().transpose()<<" "<<bbox.max().transpose()<<std::endl;
	auto center = bbox.center();
	PointScalar H = bbox.sideLength();
	auto order = static_cast<Derived *>(this)->orderForBox(H, m_baseOrder,step);

	//make sure no old buffer is around	
	buf=std::make_unique<sycl::buffer<T,1> > (m_src_octree->numActiveCones(level,step)*order.prod());
    }



    inline void transformCartToInterp(const Eigen::Ref<const PointArray > &nodes,
				      Eigen::Ref<PointArray > transformed, const Eigen::Vector<PointScalar, DIM> &xc, PointScalar H) const
    {
	Util::cartToInterp2<DIM>(nodes.array(), xc, H,transformed);
        /*for (int i = 0; i < nodes.cols(); i++) {
            transformed.col(i) = Util::cartToInterp<DIM>(nodes.col(i), xc, H);
	    }*/
    }

    inline void transformInterpToCart(const Eigen::Ref<const PointArray > &nodes,
                               Eigen::Ref<PointArray > transformed, const Eigen::Vector<PointScalar, DIM> &xc, PointScalar H) const
    { 

	transformed = Util::interpToCart<DIM>(nodes.array(), xc, H);
       /*for (int i = 0; i < nodes.cols(); i++) {
	 transformed.col(i) = Util::interpToCart<DIM>(nodes.col(i), xc, H);
	 }*/
    }


    inline  PointScalar  cutoff_limit(PointScalar H) const
    {
	return 0;
    }


    inline PointScalar tolerance() const {
	return m_tolerance;
    }

protected:
    void onOctreeReady()
    {
	//do nothing. but give subclasses the opportunity to initialize some things
    }

private:
#ifdef KEEP_LEVEL_DATA
    //std::vector<std::shared_ptr<OctreeLevelData<T,DIM> > > m_octreeData;
#endif
    std::unique_ptr<Octree<T, DIM> > m_src_octree;
    std::unique_ptr<Octree<T, DIM> > m_target_octree;
    unsigned int m_numTargets;
    unsigned int m_numSrcs;
    Eigen::Vector<size_t, DIM> m_base_n_elements;
    Eigen::Vector<int, DIM> m_baseOrder;
    PointScalar m_tolerance;
};

#endif
