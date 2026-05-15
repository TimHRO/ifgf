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
#include "octree.hpp"
#include "chebinterp.hpp"
#include "chebinterp_sycl.hpp"
#include "sycl_helpers.hpp"
#include "util.hpp"

//#include <fstream>
#include <iostream>

#include <memory>


template<int DIM>
constexpr int _CtFBufferSize(int order,int high_order)
{
    size_t buffer_size=0;	
    for(int d=DIM-1;d>0;d--) {
	int Np=1;
	for(int j=0;j<d;j++)
	{
	    Np*= (j== 0 ? std::max(order-2,2) : order);
	}
	buffer_size+=(d== 0 ? std::max(high_order-2,2) : high_order) *Np;
    }
	
    return buffer_size;
}




template<typename T, unsigned int DIM, unsigned int DIMOUT, typename Derived>
class IfgfOperator
{
public:
    typedef Eigen::Array<PointScalar, DIM, Eigen::Dynamic> PointArray;     //, Eigen::RowMajor?

    enum RefinementType { RefineH, RefineP};

    IfgfOperator(long int maxLeafSize = -1, size_t order=5, size_t n_elements=1, PointScalar tolerance = -1):
	m_maxLeafSize(maxLeafSize)
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
	m_baseOrder.fill(order);
	m_tolerance=tolerance;


    }

    ~IfgfOperator()
    {
	std::cout<<"freeing ifgf"<<std::endl;
    }

    const FlatOctree<T,DIM>& src_octree() const {
	return *m_octree;
    }



	void init(const PointArray &srcs, const PointArray targets)
	{
		std::cout << "init" << std::endl;

		auto tmp_src_octree = std::make_unique<Octree<T, DIM>>(m_maxLeafSize);
		auto tmp_target_octree = std::make_unique<Octree<T, DIM>>(m_maxLeafSize);

		bool is_ready = (m_octree != nullptr);
		if (!is_ready) {
			std::cout << "not ready" << std::endl;
			tmp_src_octree->build(srcs);
			tmp_target_octree->build(targets);
			tmp_src_octree->buildInteractionList(*tmp_target_octree);
		}

		static_cast<Derived *>(this)->onOctreeReady();

		m_numTargets = targets.cols();
		m_numSrcs = srcs.cols();

		if (m_tolerance > 0) {
			std::cout << "ADAPTIVITY NOT IMPLEMENTED YET. IGNORING TOL" << std::endl;
		}

		{
			m_baseOrder[0] = std::max(m_baseOrder[0] - 2, 2);
		}

		if (!is_ready) {
			std::cout << "calculating interp range" << std::endl;
			tmp_src_octree->calculateInterpolationRange(
				[this](PointScalar H, int step) {
					return static_cast<Derived *>(this)->orderForBox(H, m_baseOrder, step);
				},
				[this](PointScalar H) {
					return static_cast<Derived *>(this)->elementsForBox(H, this->m_baseOrder, this->m_base_n_elements);
				},
				[this](PointScalar H) {
					return static_cast<Derived *>(this)->cutoff_limit(H, this->m_baseOrder);
				},
				*tmp_target_octree);

			m_octree = std::make_shared<FlatOctree<T, DIM>>(*tmp_src_octree, *tmp_target_octree);
			m_src_octree = std::move(tmp_src_octree);
		}

		
		size_t nLevels = m_src_octree->levels();
		// ---------------------------------------------------
		// Build target-centric data for near-field GPU kernel
		// ---------------------------------------------------
		m_allNearFieldMetaData.resize(nLevels);

		for(int level=0;level<nLevels;level++){
			auto& nearFieldMeta = m_allNearFieldMetaData[level];
			size_t numBoxes = m_src_octree->numBoxes(level);

			for(size_t boxIdx=0; boxIdx<numBoxes; boxIdx++){
				IndexRange srcRange = m_src_octree->points(level,boxIdx);

				//auto center = m_src_octree->boxCenter(level,boxIdx);
				//double H = m_src_octree->boxSize(level,boxIdx);
				//auto boxMin = m_src_octree->boxMin(level,boxIdx);

				NearFieldMetaData nf;
				nf.sourceStart = static_cast<uint32_t>(srcRange.first);
				nf.sourceEnd = static_cast<uint32_t>(srcRange.second);

				BoundingBox<DIM> bbox = m_src_octree->bbox(level,boxIdx);
				auto center = bbox.center();
				double H = bbox.sideLength();


				const auto& nearTargetRange = m_src_octree->nearTargets(level, boxIdx);
				const uint32_t MAX_WORK_SIZE = 64;
				for (const auto& range : nearTargetRange){
					uint32_t tStart = (uint32_t)range.first;
					uint32_t tEnd = (uint32_t)range.second;
					uint32_t totalTargets = tEnd - tStart;

					for (uint32_t chunkStart = tStart; chunkStart < tEnd; chunkStart += MAX_WORK_SIZE) {
						uint32_t chunkEnd = std::min(chunkStart + MAX_WORK_SIZE, tEnd);

						nf.targetStart = static_cast<uint32_t>(chunkStart);
						nf.targetEnd = static_cast<uint32_t>(chunkEnd);
						nf.H = H;
						for(int d=0; d<DIM; ++d) {
							nf.Center[d] = center[d];
						}
						nearFieldMeta.push_back(nf);
					}
				}

			}
			// Add this at the end of your init() function
				std::sort(nearFieldMeta.begin(), nearFieldMeta.end(), [](const auto& a, const auto& b) {
					if (a.targetStart != b.targetStart)
						return a.targetStart < b.targetStart;
					return a.sourceStart < b.sourceStart;
				});
		}


		// ------------------------------------------------------------
		// Build cone‑centric data for far‑field GPU kernel
		// ------------------------------------------------------------


		// Determine the (constant) low‑order stride used in mult_impl
		PointScalar H0 = m_octree->sideLength();
		auto order0 = static_cast<Derived *>(this)->orderForBox(
			H0 * std::pow(0.5, m_octree->levels() + 5), m_baseOrder, 0);
		const size_t stride = order0.prod();   // this is the number of Chebyshev coefficients per cone

		m_allLevelConeInfo.resize(nLevels);

		for (size_t level = 0; level < nLevels; ++level) {
			InfoPerLevel &info = m_allLevelConeInfo[level];
			info.metaData.clear();
			info.targetIds.clear();
			info.normPoints.clear();

			size_t numBoxes = m_src_octree->numBoxes(level);
			for (size_t boxIdx = 0; boxIdx < numBoxes; ++boxIdx) {
				// Only boxes that have far‑field targets (including ancestors) contribute
				if (!m_src_octree->hasFarTargetsIncludingAncestors(level, boxIdx))
					continue;

				auto bbox = m_src_octree->bbox(level, boxIdx);
				const auto center = bbox.center();
				const PointScalar H = bbox.sideLength();

				// Low‑order cone domain (step = 0)
				const ConeDomain<DIM> &coneDomain = m_src_octree->coneDomain(level, boxIdx, 0);
				const size_t nConesInBox = coneDomain.n_elements();

				// Temporary storage for this box: for each cone index (local) collect target indices and normalised points
				std::vector<std::vector<uint32_t>> coneTargets(nConesInBox);
				std::vector<std::vector<PointScalar>> coneNormPoints(nConesInBox); // DIM values per target

				// Get all far‑field target ranges for this source box
				const auto &farRanges = m_src_octree->farTargets(level, boxIdx);
				for (const auto &range : farRanges) {
					for (size_t tIdx = range.first; tIdx < range.second; ++tIdx) {
						// 1. Transform target point to box‑local coordinates (interpolation domain)
						//const auto pnt = targets.col(tIdx).matrix();
						const auto pnt = m_octree->targetPoints().col(tIdx).matrix();
						const auto transformed = Util::cartToInterp<DIM>(pnt, center, H);

						// 2. Find which cone element (local index) contains this transformed point
						size_t el = coneDomain.elementForPoint(transformed);
						if (el == SIZE_MAX)
							continue; // outside the valid cone domain (should not happen for far field)

						// 3. Map the point from the cone sub‑cell to the reference [-1,1] Chebyshev domain
						const auto normPointMatrix = coneDomain.transformBackwards(el, transformed);

						// Store target index (global target point number) and the normalised point
						coneTargets[el].push_back(static_cast<uint32_t>(tIdx));
						for (int d = 0; d < DIM; ++d)
							coneNormPoints[el].push_back(static_cast<PointScalar>(normPointMatrix(d, 0)));
					}
				}

				// For each cone that actually has targets, create a ConeMetaData entry
				for (size_t el = 0; el < nConesInBox; ++el) {
					if (coneTargets[el].empty())
						continue;

					// Get the global ID of this cone (used for coefficient offset)
					// From the raw Octree: active cones for step 0 are stored in m_activeCones[level][0]
					// We need to find the ConeRef that has boxId == boxIdx and id == el.
					// For convenience, we can use the coneMap built in the Octree:
					const auto &coneMap = m_src_octree->coneMaps(level)[boxIdx];
					auto it = coneMap.find(el);
					if (it == coneMap.end())
						continue; // cone not active (should not happen if there are targets, but safe)
					size_t globalId = it->second; // this is the index in the activeCones list = memory index

					// Coefficient offset = globalId * stride
					uint32_t coeffOffset = static_cast<uint32_t>(globalId * stride);

					// Record metadata
					ConeMetaData cmd;
					for (int d = 0; d < DIM; ++d)
						cmd.center[d] = static_cast<double>(center[d]);
					cmd.H = static_cast<double>(H);
					cmd.coeffOffset = coeffOffset;
					cmd.targetOffset = static_cast<uint32_t>(info.targetIds.size());   // start index in flat arrays
					cmd.numTargets = static_cast<uint32_t>(coneTargets[el].size());
					cmd.stride = static_cast<uint32_t>(stride);
					cmd.localConeIdx = static_cast<uint32_t>(el);
					info.metaData.push_back(cmd);

					// Append target indices and normalised points for this cone
					info.targetIds.insert(info.targetIds.end(),
										coneTargets[el].begin(), coneTargets[el].end());
					info.normPoints.insert(info.normPoints.end(),
										coneNormPoints[el].begin(), coneNormPoints[el].end());
				}
			}

			// Optional: shrink vectors to remove unused capacity
			info.metaData.shrink_to_fit();
			info.targetIds.shrink_to_fit();
			info.normPoints.shrink_to_fit();

			std::cout << "Level " << level << ": " << info.metaData.size()
					<< " cones, " << info.targetIds.size() << " target interactions" << std::endl;
		}

		std::cout << "done initializing" << std::endl;
	}

    Eigen::Array<T, Eigen::Dynamic,DIMOUT> mult(const Eigen::Ref<const Eigen::Vector<T, Eigen::Dynamic> > &weights)
    {
	switch(m_baseOrder.maxCoeff()) {
	case 1: 
	case 2: 
	case 3: 
	    //case 4:  return mult_impl<4>(weights);
	case 5:  
 	case 6:  return mult_impl<6>(weights);
	case 7:  
	case 8:  return mult_impl<8>(weights);
	    //	case 16:  return mult_impl<16>(weights);
	    //case 32:  return mult_impl<32>(weights);
	    //case 10:  return mult_impl<10>(weights);
	default: std::cout<<"not implemented"<<m_baseOrder.transpose()<<std::endl; return mult_impl<8>(weights);
	}
	
    }





    template <int MAX_ORDER>
    Eigen::Array<T, Eigen::Dynamic,DIMOUT> mult_impl(const Eigen::Ref<const Eigen::Vector<T, Eigen::Dynamic> > &weights)
    {
        std::cout<<"multimpl"<<std::endl;
        Eigen::Array<T, Eigen::Dynamic, DIMOUT> result(m_numTargets,DIMOUT);
        result.fill(0);
        int level = levels() - 1;

	//std::cout<<"boxes="<<m_octree->numBoxes(level)<<std::endl;
	const PointScalar hmin=m_octree->diameter()*std::pow(0.5,m_octree->levels());
	//std::vector<tbb::queuing_mutex> resultMutex(m_numTargets);

        { //scope to contain all the sycl stuff. that way we make sure that all the data is copied to the host before proceeding.

	sycl::queue& Q=SyclHelpers::QueueSingleton::getInstance().queue();//(sycl::default_selector_v);

        //push some global data to the GPU
	Eigen::Vector<T, Eigen::Dynamic> new_weights(weights.size());
        Util::copy_with_permutation_rowwise<T,1> (weights.array(), m_octree->srcPermutation(),new_weights.array());
	sycl::buffer<const T, 1> b_weights(new_weights.data(),weights.size());
	sycl::buffer<const PointScalar, 1> b_srcs(m_octree->srcPoints().data(),m_octree->srcPoints().cols()*DIM);
	sycl::buffer<const PointScalar, 1> b_targets(m_octree->targetPoints().data(),m_octree->targetPoints().cols()*DIM);

	sycl::buffer<T, 1> b_result(result.data(),result.size());       

	std::unique_ptr<sycl::buffer<T,1> > interpolationDataBuffer;
	std::unique_ptr<sycl::buffer<T,1> > parentInterpolationDataBuffer;

        std::shared_ptr<OctreeLevelData<T,DIM> > parentData;
	std::shared_ptr<OctreeLevelData<T,DIM> > srcData;


	//Get an exemplary bbox to determine the interpolation order
	PointScalar H0 = m_octree->sideLength();
	const auto order = static_cast<Derived *>(this)->orderForBox(H0*std::pow(0.5,m_octree->levels()+5), m_baseOrder,0);
	const auto& chebNodes=ChebychevInterpolation::chebnodesNdd<PointScalar,DIM>(order);
	const auto high_order = static_cast<Derived *>(this)->orderForBox(H0*std::pow(0.5,m_octree->levels()+5), m_baseOrder,1);
	const auto& ho_chebNodes=ChebychevInterpolation::chebnodesNdd<PointScalar,DIM>(high_order);

	//Cache chebychev nodes on the GPU
	sycl::buffer<const PointScalar,1> b_chebNodes(chebNodes.data(),chebNodes.cols()*DIM);
	sycl::buffer<const PointScalar,1> b_hoChebNodes(ho_chebNodes.data(),ho_chebNodes.cols()*DIM);


	size_t Ntp=order.sum();
	Eigen::Array<PointScalar,Eigen::Dynamic,1>  points(Ntp);

	std::array<int, DIM> ns=SyclHelpers::EigenVectorToCPPArray<int, DIM>(high_order);
	std::array<int, DIM> lo_ns=SyclHelpers::EigenVectorToCPPArray<int, DIM>(order);
	std::array<int, DIM> ho_ns = SyclHelpers::EigenVectorToCPPArray<int, DIM>(high_order);
		

	//std::cout<<"stuff; "<<ns<<" "<<lo_ns<<" "<<factors<<" "<<n_elements<<std::endl;
	size_t Np=1;
	size_t offset=0;

	//store the points for the inner most dimension separately from the others
	for(int d=0;d<DIM;d++) {
	    const auto& chebNodes1d=ChebychevInterpolation::chebnodesNdd<PointScalar,1>(Eigen::Vector<int,1>(order[d]));

	    points.segment(offset,chebNodes1d.size())=chebNodes1d.array();
	    offset+=chebNodes1d.size();		    

	}

		
	const size_t ho_stride=high_order.prod();
	sycl::buffer<PointScalar> b_points(points.data(),points.size());

		    
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


	const size_t hoCv_size=high_order.unaryExpr([&](int v){ return v*v; }).sum();
	sycl::buffer<PointScalar> b_hoChebvals(hoCv_size);
	{
	    sycl::host_accessor a_cv(b_hoChebvals);
	    size_t idx=0;
	    //make sure the factors for the chebtrafo are precomputed...
	    for(int d=DIM-1;d>=0;d--) {
		//std::cout<<"idx="<<idx<<" vs "<<cv_size<<" "<<d<<std::endl;
		const auto& cv=ChebychevInterpolation::chebvals<PointScalar>(high_order[d]);
		std::copy(cv.reshaped().begin(),cv.reshaped().end(),a_cv.begin()+idx);
		idx+=high_order[d]*high_order[d];
		
	    }
	    
	    assert(idx==hoCv_size); //check that we initialized correctly (TODO remove)
	}


		
		//std::vector<InfoPerLevel> allLevelConeInfo= prepareConeMetaData(m_src_octree, m_octree->targetPoints(), order.prod());


        for (; level >= 0; --level) {
	    if(parentData==0) {
		srcData = m_octree->data(level);//std::make_unique< OctreeLevelData<T,DIM> >(*m_octree,level);
	    }else {
		std::swap(parentData,srcData);
		parentData.reset();
	    }

	    std::cout<<"level="<<level<<std::endl;
	    //std::cout<<"created ocdata"<<std::endl;
	    
	    {
		Q.wait();
		std::cout<<"nearfield"<<std::endl;
		auto& nfMeta = m_allNearFieldMetaData[level];
		sycl::buffer<NearFieldMetaData,1> b_meta(nfMeta.data(), nfMeta.size());

		auto e=Q.submit([&](sycl::handler &h) {
		    // start by pushing  some data to the GPU (octree stuff)
		    sycl::accessor a_srcs(b_srcs, h, sycl::read_only);
		    sycl::accessor a_targets(b_targets, h, sycl::read_only);
		    sycl::accessor a_weights(b_weights, h, sycl::read_only);

		    sycl::accessor a_result(b_result, h, sycl::read_write);

			sycl::accessor a_meta(b_meta, h, sycl::read_only);

		    //const auto &srcDataAcc = srcData->accessor(h);
		    const auto functions =
			static_cast<Derived *>(this)->kernelFunctions();


		    //auto out = sycl::stream(1024, 768, h);
		    const size_t num_targets=m_octree->targetPoints().cols();

			//const size_t groupSize = 258;
			//const size_t nGroups =  (num_targets + groupSize-1) / groupSize;

		    //std::cout<<"setup complete"<<num_targets<<std::endl;

		    h.parallel_for(
				   sycl::range<1>(nfMeta.size()),
				   [=](sycl::id<1> idx) {
				       //out<<"pnt"<<i<<"\n";
					   const auto& task = a_meta[idx];

						// This loop is now very tight and linear in memory
						for (uint32_t t = task.targetStart; t < task.targetEnd; ++t) {
							T res = functions.evaluateKernel(
								a_srcs, task.sourceStart, task.sourceEnd,
								a_targets, t, a_weights
							);


							using ScalarT = typename T::value_type; // Usually double or float

							ScalarT* base_ptr = reinterpret_cast<ScalarT*>(&a_result[t]);

							sycl::atomic_ref<ScalarT, sycl::memory_order::relaxed, 
											sycl::memory_scope::device, 
											sycl::access::address_space::global_space> atm_real(base_ptr[0]);

							sycl::atomic_ref<ScalarT, sycl::memory_order::relaxed, 
											sycl::memory_scope::device, 
											sycl::access::address_space::global_space> atm_imag(base_ptr[1]);

							atm_real.fetch_add(res.real());
							atm_imag.fetch_add(res.imag());
							}
				   });
		});

		std::cout << "escpaed near field" << std::endl;
		/*Q.wait();
		std::cout<<"done nf"<<(e.template get_profiling_info<sycl::info::event_profiling::command_end>() -
		e.template get_profiling_info<sycl::info::event_profiling::command_start>())/(1.0e9)<<std::endl;*/
	    }

            Q.wait();

            {
                const double H=m_octree->sideLength()*pow(2,-level);
                if(static_cast<Derived *>(this)->farfieldCanBeSkipped(H)) {
                    std::cout<<"skipping farfield computation"<<std::endl;
                    continue;
                }
            }

            //const size_t stride=chebNodes.cols();

	    //prepare the interpolation data for all leaves
	    if(level==m_octree->levels()-1) {
                std::cout<<"init"<<level<<" "<<H0*std::pow(0.5,m_octree->levels())<<std::endl;
		initInterpolationData(level,1, interpolationDataBuffer);
	    }

		//prepare interpolation data that is computed by interpolation and projection
		initInterpolationData(level,0,parentInterpolationDataBuffer);
		Q.wait();

		// prepare chebtransoformed data for each coarse cone, if its leaf cone evaluate kernel first
		// when finished project coarse to fine but still looping over coarse cones - one coarse -> 8 fine cones sequentially
		{

			const size_t stride = ho_chebNodes.cols();
			const size_t fine_stride = order.prod();
			const size_t numActive = m_octree->numActiveCones(level,1);

			if(numActive==0) continue;

			//std::array<int, DIM> ho_ns;
			//std::copy(high_order.begin(), high_order.end(), ho_ns.begin());
			std::cout << "Launching work group with " << stride << " threads" << "\n";

			auto e = Q.submit([&](sycl::handler &h){
				//sycl::stream out(1024, 256, h);
				sycl::accessor a_srcs(b_srcs, h, sycl::read_only);
				sycl::accessor a_points(b_points, h, sycl::read_only);		    
				sycl::accessor a_weights(b_weights, h, sycl::read_only);
				sycl::accessor a_intData(*interpolationDataBuffer, h, sycl::read_write);
				sycl::accessor a_parentIntData(*parentInterpolationDataBuffer, h, sycl::read_write);
				sycl::accessor a_hoChebNodes(b_hoChebNodes, h, sycl::read_only);
				sycl::accessor a_hoChebVals(b_hoChebvals, h, sycl::read_only);
				sycl::accessor a_chebVals(b_chebvals, h, sycl::read_only);
				const auto &srcDataAcc = srcData->accessor(h);
				const auto functions =
				static_cast<Derived *>(this)->kernelFunctions();
				sycl::local_accessor<T,1> rawData(sycl::range<1>(stride), h);

				// precompute parameters for coarse to fine refinement

				const double H=H0*pow(2,-level);
				Eigen::Vector<size_t,DIM> coarse_N=static_cast<Derived *>(this)->elementsForBox(H, this->m_baseOrder,this->m_base_n_elements);
				Eigen::Vector<size_t,DIM> fine_N=(size_t) (std::pow((unsigned int) REFINEMENT_FACTOR, (unsigned int) ( REFINEMENT_LEVELS)))*coarse_N;//
				std::array<int, DIM> n_elements=SyclHelpers::EigenVectorToCPPArray<int,DIM>(fine_N.template cast<int>());
				std::array<size_t, DIM> n_el=SyclHelpers::EigenVectorToCPPArray<size_t,DIM>(coarse_N);
				sycl::accessor a_chebNodes(b_chebNodes,h,sycl::read_only);
				const int nF=std::pow(REFINEMENT_FACTOR,DIM); 
				constexpr int MAX_LOW_ORDER=std::max(MAX_ORDER-3,1);
				constexpr int BUF_SIZE=_CtFBufferSize<DIM>(MAX_LOW_ORDER,MAX_ORDER);
				sycl::local_accessor<T,1> ctfScratch(sycl::range<1>(BUF_SIZE),h);

				std::cout << "BUF_SIZE = " << BUF_SIZE
				<< " sizeof(tmp)=" << sizeof(T)*BUF_SIZE
				<< std::endl;

				h.parallel_for(sycl::nd_range<1>(numActive * stride, stride), [=](sycl::nd_item<1> item){
					const size_t coneIdx = item.get_group_linear_id();
					const size_t localId = item.get_local_linear_id();
					const ConeRef ref = srcDataAcc.activeCone(coneIdx);
					const size_t boxId = ref.boxId();
					const bool isLeaf = srcDataAcc.isLeaf(boxId);
					const bool hasFar = srcDataAcc.hasFarTargetsIncludingAncestors(boxId);

					if(!hasFar) return;

					//sycl::local_ptr<T> rawData = sycl::local_alloc<T>(stride, item.get_group());

					const size_t globalOffset = ref.globalId() * stride;

					if(isLeaf){
						sycl::marray<PointScalar,DIM> center = srcDataAcc.boxCenter(boxId);
						PointScalar H = srcDataAcc.boxSize(boxId);
						auto grid = srcDataAcc.coneDomain(boxId,1);
						IndexRange srcs = srcDataAcc.points(boxId);
						sycl::marray<PointScalar,DIM> transformed, cartesian;
						grid.transform(ref.id(), a_hoChebNodes, transformed, localId);
						Util::interpToCart(transformed, cartesian, center, H);

						rawData[localId] = functions.evaluateFactoredKernel(a_srcs, srcs.first, srcs.second, cartesian, a_weights, center, H);
					}
					// GOAL nothing todo here, but for now grab from global
					else {
						rawData[localId] = a_intData[globalOffset + localId];
					}

					// barrier to wait for all local threads before doing chebtransform
					item.barrier(sycl::access::fence_space::global_and_local);
					if(localId==0){
						SyclChebychevInterpolation::chebtransform_inplace<T,DIM,MAX_ORDER>(
							rawData, ho_ns, a_hoChebVals, 0
						);
					}
					item.barrier();
					a_intData[globalOffset + localId] = rawData[localId];
					item.barrier();

					// --------------- finished computation of interpolation data on coarse cone -----------

					// --------------- projecting them to fine cones ---------------------------
					

					// we dont need interpolation info for hoCones without farfield - already skipped them earlier :)
					auto ho_id=SyclConeDomain<DIM>::indicesFromId(ref.id(),n_el);
					

					std::array<size_t, DIM> factors;
					factors.fill(REFINEMENT_FACTOR);


					// sequentially do CTF
					if(localId==0){
						//out << "LocalID 0 is working on fine grid projection" << "\n";
					for(size_t sub=0; sub < nF; sub++){
						//auto ho_id = SyclConeDomain<DIM>::indicesFromId(sub, n_el);
						auto lid=SyclConeDomain<DIM>::indicesFromId(sub,factors);
						const size_t fine_el=
						(ho_id[2]*REFINEMENT_FACTOR+(lid[2]))*n_elements[1]*n_elements[0]+
						(ho_id[1]*REFINEMENT_FACTOR+(lid[1]))*n_elements[0]+
						(ho_id[0]*REFINEMENT_FACTOR+(lid[0]));

						const size_t fineMemId = srcDataAcc.memId(ref.boxId(), fine_el);

						//out << "Now setting parent int data 0" << "\n";

						if(fineMemId<SIZE_MAX-1){
							for(int i=0; i<fine_stride; i++){
								a_parentIntData[fineMemId*fine_stride+i]=T(0);
							}
						
							size_t offset=0;
							//constexpr int MAX_LOW_ORDER=std::max(MAX_ORDER-3,1);
							//constexpr int BUF_SIZE=_CtFBufferSize<DIM>(MAX_LOW_ORDER,MAX_ORDER);

							
							sycl::marray<PointScalar,MAX_LOW_ORDER*DIM> t_pnts;
							
							sycl::marray<T,BUF_SIZE> tmp; //Temporary storage for the sum-factorization. 
							//T* tmp = &ctfScratch[0];


							tmp=0;
							//for(int i = 0; i < BUF_SIZE; i++) tmp[i] = T(0);
							//for(int i = 0; i < BUF_SIZE; i++) t_pnts[i] = PointsScalar(0);
							t_pnts=0;		//Fill up the remaining points. otherwise the compiler optimization breaks the code
							for(int d=0;d<DIM;d++) {
								const PointScalar h=2;
								assert(lo_ns[d]<=MAX_LOW_ORDER);
								
								const PointScalar mmin=-1+(lid[d]*(h/((PointScalar) REFINEMENT_FACTOR)));
								const PointScalar mmax=(mmin+(h/((PointScalar) REFINEMENT_FACTOR)));
								const PointScalar a=0.5*(mmax-mmin);
								const PointScalar b=0.5*(mmax+mmin);

								
								for(size_t l=0;l<lo_ns[d];l++) {
									t_pnts[offset]=a*a_points[offset]+b;
									offset++;
								}
							}

							//out << "arrived at tp evaluate" << "\n";

							const size_t int_data_offset = coneIdx * stride;
							//const size_t int_data_offset = globalOffset;
							SyclChebychevInterpolation::tp_evaluate_t<T,DIM>(t_pnts, a_intData, int_data_offset,
												ho_ns,lo_ns, a_parentIntData,
												tmp,
												fineMemId*fine_stride, 0);

							//out << "arrived at cheptransform " << "\n";
												
							SyclChebychevInterpolation::chebtransform_inplace<T,DIM,MAX_ORDER>( a_parentIntData,  lo_ns, a_chebVals,fineMemId*fine_stride);
					}
					}
					}
				});
			});
		}


		/*

	    if(m_octree->numLeafCones(level) > 0)
	    {
                std::cout<<"interp "<< m_octree->numLeafCones(level)<<b_hoChebNodes.size()<<std::endl;
		{
		    auto e=Q.submit([&](sycl::handler &h) {
		    // start by pushing  some data to the GPU (octree stuff)
		    sycl::accessor a_srcs(b_srcs, h, sycl::read_only);		    
		    sycl::accessor a_weights(b_weights, h, sycl::read_only);

		    sycl::accessor a_intData(*interpolationDataBuffer, h, sycl::read_write);

		    sycl::accessor a_hoChebNodes(b_hoChebNodes, h, sycl::read_only);

		    const auto &srcDataAcc = srcData->accessor(h);
		    const auto functions =
			static_cast<Derived *>(this)->kernelFunctions();



		    const size_t stride=ho_chebNodes.cols();


		    const size_t sizeB=interpolationDataBuffer->size();

		    const size_t  numLeafCones=m_octree->numLeafCones(level);
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

 
					       a_intData[j+offset]=functions.evaluateFactoredKernel(a_srcs, srcs.first, srcs.second,
					       							    transformed2, a_weights,center, H);

					   }
				       }

				   });
		 });
		    //std::cout<<"intrp="<<(e.template get_profiling_info<sycl::info::event_profiling::command_end>() -
		    //  e.template get_profiling_info<sycl::info::event_profiling::command_start>())/(1.0e9)<<std::endl;
		//Q.wait();
		}
	    }

	    //chebtrafo everything

	    {
                std::cout<<"chebtrafo"<<std::endl;
		//Q.wait();
		auto e=Q.submit([&](sycl::handler &h) {
		    // start by pushing  some data to the GPU (octree stuff)

		    sycl::accessor a_intData(*interpolationDataBuffer, h, sycl::read_write);
				

		    //TODO unify order type
		    std::array<int,DIM> ns_ho;
		    std::copy(high_order.begin(),high_order.end(),ns_ho.begin());
		
		
		    const sycl::accessor a_chebvals(b_hoChebvals,h,sycl::read_only);
		    const auto &srcDataAcc = srcData->accessor(h);

		    const size_t sizeB=interpolationDataBuffer->size();
		    const size_t numActiveCones= m_octree->numActiveCones(level,1);

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
					   SyclChebychevInterpolation::chebtransform_inplace<T,DIM, MAX_ORDER>( a_intData,  ns_ho, a_chebvals,i*stride);
					   //out<<"i2="<<a_intData[i*stride].real()<<"\n";
				       }
				   });
		});

		*/
		
		/*Q.wait();
		std::cout<<"done trafo"<<(e.template get_profiling_info<sycl::info::event_profiling::command_end>() -
		e.template get_profiling_info<sycl::info::event_profiling::command_start>())/(1.0e9)<<std::endl;*/
	    
		/*
	    initInterpolationData(level,0, parentInterpolationDataBuffer);
	    {
                std::cout<<"CTF"<<std::endl;
		//interpolation Data contains the values using the coarse- high-order interpolation scheme. Project to the low-order fine grid
		//which is faster for point-evaluations. We use parentInteprolationData as a termpoary buffer.


		//first set up some common data structures
		{
		    const size_t numActiveCones= m_octree->numActiveCones(level,1);
		    if(numActiveCones == 0)		    
		        continue;

		    auto e=Q.submit([&](sycl::handler &h) {
			sycl::accessor a_intData(*interpolationDataBuffer, h, sycl::read_only);
			sycl::accessor a_parentIntData(*parentInterpolationDataBuffer, h, sycl::read_write);
		    
			sycl::accessor a_points(b_points,h, sycl::read_only);

				    
			
			const sycl::accessor a_chebvals(b_chebvals,h,sycl::read_only);



			size_t fine_stride=order.prod();

			const auto &srcDataAcc = srcData->accessor(h);
			

			const double H=H0*pow(2,-level);//m_octree->bbox(level,0).sideLength();//H0*pow(2,-level);
			Eigen::Vector<size_t,DIM> coarse_N=static_cast<Derived *>(this)->elementsForBox(H, this->m_baseOrder,this->m_base_n_elements);
			Eigen::Vector<size_t,DIM> fine_N=(size_t) (std::pow((unsigned int) REFINEMENT_FACTOR, (unsigned int) ( REFINEMENT_LEVELS)))*coarse_N;//
		    					
			std::array<int, DIM> n_elements=SyclHelpers::EigenVectorToCPPArray<int,DIM>(fine_N.template cast<int>());


			std::array<size_t, DIM> n_el=SyclHelpers::EigenVectorToCPPArray<size_t,DIM>(coarse_N);


			sycl::accessor a_chebNodes(b_chebNodes,h,sycl::read_only);

			const int nF=std::pow(REFINEMENT_FACTOR,DIM);
			//std::cout<<"doing it"<<std::endl;
			h.parallel_for(sycl::range( {numActiveCones*nF}), [=](auto it)			
			{		
			    
			    const size_t coneId=it/nF;		
			    
			    
			    ConeRef hoCone=srcDataAcc.activeCone(coneId);
			    if( srcDataAcc.hasFarTargetsIncludingAncestors(hoCone.boxId())){ // we dont need the interpolation info for those levels.
				auto ho_id=SyclConeDomain<DIM>::indicesFromId(hoCone.id(),n_el);

				std::array<size_t, DIM> factors;
				factors.fill(REFINEMENT_FACTOR);

				auto lid=SyclConeDomain<DIM>::indicesFromId(it%nF,factors);
				const size_t fine_el=
				    (ho_id[2]*REFINEMENT_FACTOR+(lid[2]))*n_elements[1]*n_elements[0]+
				    (ho_id[1]*REFINEMENT_FACTOR+(lid[1]))*n_elements[0]+
				    (ho_id[0]*REFINEMENT_FACTOR+(lid[0]));
		       
				const size_t fineMemId=srcDataAcc.memId(hoCone.boxId(),fine_el);
						
				if(fineMemId<SIZE_MAX-1) { ///the target cone is active!
				    for(int i=0;i<fine_stride;i++) {
					a_parentIntData[fineMemId*fine_stride+i]=0;
				    }
				
					
				    //out<<it<<" "<<coneId<<"\n";
				    size_t offset=0;
				    constexpr int MAX_LOW_ORDER=std::max(MAX_ORDER-3,1);
				    constexpr int BUF_SIZE=_CtFBufferSize<DIM>(MAX_LOW_ORDER,MAX_ORDER);

				    sycl::marray<PointScalar,MAX_LOW_ORDER*DIM> t_pnts;
				    
				    sycl::marray<T,BUF_SIZE> tmp; //Temporary storage for the sum-factorization. 
			


				    tmp=0;
				    t_pnts=0;		//Fill up the remaining points. otherwise the compiler optimization breaks the code
				    for(int d=0;d<DIM;d++) {
					const PointScalar h=2;
					assert(lo_ns[d]<=MAX_LOW_ORDER);
				    
					const PointScalar mmin=-1+(lid[d]*(h/((PointScalar) REFINEMENT_FACTOR)));
					const PointScalar mmax=(mmin+(h/((PointScalar) REFINEMENT_FACTOR)));
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


				    SyclChebychevInterpolation::chebtransform_inplace<T,DIM,MAX_ORDER>( a_parentIntData,  lo_ns, a_chebvals,fineMemId*fine_stride);
				    
				
				}

			    }
		    
			});		  
		    
		    });

		    //Q.wait();
		    //std::cout<<"done CTF"<<(e.template get_profiling_info<sycl::info::event_profiling::command_end>() -
		    //e.template get_profiling_info<sycl::info::event_profiling::command_start>())/(1.0e9)<<std::endl;

		}
	    }
		*/
	    
	    Q.wait();
		std::cout << "escaped interpolation hell" << "\n";

	    std::swap(interpolationDataBuffer,parentInterpolationDataBuffer);
	    parentInterpolationDataBuffer.reset();

            {
                const double H=m_octree->sideLength()*pow(2,-level);
                std::cout<<"far field"<<H<<std::endl;
            }
	    //Q.wait();
		{

		auto& thisLevelConeInfo = m_allLevelConeInfo[level];

		const size_t numCones = thisLevelConeInfo.metaData.size();
		if (numCones == 0) continue;

		sycl::buffer<ConeMetaData,1> buf_meta(thisLevelConeInfo.metaData);
		sycl::buffer<uint32_t,1> buf_targetIds(thisLevelConeInfo.targetIds);
		sycl::buffer<PointScalar,1> buf_normPoints(thisLevelConeInfo.normPoints);

		Q.submit([&](sycl::handler &h) {
			sycl::accessor a_intData(*interpolationDataBuffer, h, sycl::read_only);
			sycl::accessor a_result(b_result, h, sycl::read_write);
			sycl::accessor a_meta(buf_meta, h, sycl::read_only);
			sycl::accessor a_targets(b_targets, h, sycl::read_only);
			sycl::accessor a_targetIds(buf_targetIds, h, sycl::read_only);
			sycl::accessor a_normPoints(buf_normPoints, h, sycl::read_only);

			//std::array<int, DIM> ns = ...; // order (low order)
			std::array<int, DIM> ns = SyclHelpers::EigenVectorToCPPArray<int, DIM>(order);
			const size_t stride = order.prod();
			const auto functions = static_cast<Derived *>(this)->kernelFunctions();

			// Use a workgroup to load coefficients into local memory
			const size_t groupSize = 256; // tune
			sycl::local_accessor<T, 1> localCoeffs(sycl::range<1>(stride), h);
			h.parallel_for(sycl::nd_range<1>{numCones * groupSize, groupSize},
				[=](sycl::nd_item<1> it) {
					size_t coneIdx = it.get_group_linear_id();
					size_t localId = it.get_local_id(0);
					const ConeMetaData cmd = a_meta[coneIdx];

					// Load coefficients into local memory
					
					for (size_t i = localId; i < stride; i += groupSize) {
						localCoeffs[i] = a_intData[cmd.coeffOffset + i];
					}
					it.barrier();

					// Process targets assigned to this cone
					const size_t numTargets = cmd.numTargets;
					const size_t targetOffset = cmd.targetOffset;
					//const sycl::marray<PointScalar, DIM> center = ...; // from cmd
					const PointScalar H = cmd.H;

					sycl::marray<PointScalar, DIM> center;
                	for (int d = 0; d < DIM; ++d){
                    	center[d] = static_cast<PointScalar>(cmd.center[d]);
					}

					for (size_t t = localId; t < numTargets; t += groupSize) {
						uint32_t targetId = a_targetIds[targetOffset + t];
						// Load the normalised point (DIM scalars)
						sycl::marray<PointScalar, DIM> norm;
						for (int d = 0; d < DIM; ++d) {
							norm[d] = a_normPoints[(targetOffset + t) * DIM + d];
						}

						SyclChebychevInterpolation::ClenshawEvaluator<T, 1, DIM, DIM, DIMOUT> clenshaw;

						// Evaluate Chebyshev series
						T res = clenshaw(SyclRowMatrix<PointScalar, DIM, 1>(norm),
										localCoeffs, ns, 0);

						// Phase factor
						sycl::marray<PointScalar, DIM> xyz; // we need the physical target point
						// Actually we also need the physical target point for CF.
						// You can either store it in a separate buffer or recompute from norm?
						// Better: store physical target point as well, or compute on the fly from targetId.
						// Let's assume we have a buffer b_targets (already there).
						// We can access a_targets from outside capture.
						// For brevity, assume we can get target_pnt from a_targets.
						sycl::marray<PointScalar, DIM> target_pnt;
						for (int d = 0; d < DIM; ++d)
							target_pnt[d] = a_targets[targetId * DIM + d];
						auto cf = functions.CF(target_pnt - center, H);
						res *= cf;



						using ScalarT = typename T::value_type; // Usually double or float

						ScalarT* base_ptr = reinterpret_cast<ScalarT*>(&a_result[targetId]);

						sycl::atomic_ref<ScalarT, sycl::memory_order::relaxed, 
										sycl::memory_scope::device, 
										sycl::access::address_space::global_space> atm_real(base_ptr[0]);

						sycl::atomic_ref<ScalarT, sycl::memory_order::relaxed, 
										sycl::memory_scope::device, 
										sycl::access::address_space::global_space> atm_imag(base_ptr[1]);

						atm_real.fetch_add(res.real());
						atm_imag.fetch_add(res.imag());
					}
			
		    // h.parallel_for(sycl::range<1>{nT}, [=](auto it)
		    // {

			// IndexRange rng= srcDataAcc.farfieldRange(it);
			// for( size_t l=rng.first;l<rng.second;l++){
			//     size_t memId=srcDataAcc.ffBIndex(l);

			//     if(memId==SIZE_MAX) {
			// 	continue;
			//     }

			//     size_t boxId=srcDataAcc.farfieldBoxId(l);
			//     auto center = srcDataAcc.boxCenter(boxId);
			//     sycl::marray<PointScalar,DIM> target_pnt;
			//     for(int j=0;j<DIM;j++)
			// 	target_pnt[j]=a_targets[it*DIM+j];
			//     sycl::marray<PointScalar,DIM> transformed;
			//     const auto cf = functions.CF(target_pnt - center,H);

			//     transformed=srcDataAcc.farfieldPoint(l);
						
			
			//     SyclChebychevInterpolation::ClenshawEvaluator<T,1, DIM,DIM, DIMOUT> clenshaw;
			//     const size_t offset=stride*memId;
			//     T res=clenshaw(SyclRowMatrix<PointScalar, DIM,1>(transformed), a_intData, ns, offset);						
			
			//     res*= cf;

			//     a_result[it]+=res;
			// }

		    });
			
		});

		/*std::cout<<"done FF"<<(e.template get_profiling_info<sycl::info::event_profiling::command_end>() -
		  e.template get_profiling_info<sycl::info::event_profiling::command_start>())/(1.0e9)<<std::endl;*/
	    }


	    if(level<1) //there is no parent
	    {		
		break;
	    }

	    Q.wait();
            //Now transform the interpolation data to the parents
	    //std::cout<<"propagating upward"<<std::endl;

	    initInterpolationData(level-1,1, parentInterpolationDataBuffer);

	    const size_t numActiveParentCones= m_octree->numActiveCones(level-1,1);
	    if(numActiveParentCones==0) {
		continue;
	    }


	    std::cout<<"Child To Parent "<<level<<std::endl;
	    

	    parentData = m_octree->data(level-1);//std::make_unique< OctreeLevelData<T,DIM> >(*m_octree,level-1);

#ifdef  FAST_CTP
	    const bool use_fast_ctp=true;//level>3;//level>3;
#else
	    const bool use_fast_ctp=false;
#endif
	    

	    

	    if(use_fast_ctp) {

		Q.fill(parentInterpolationDataBuffer->get_access(), T(0.0));
		Q.wait();

		std::cout << "Local Memory Size: "
			  << Q.get_device().get_info<sycl::info::device::local_mem_size>()
			  << std::endl;

		size_t local_mem_size=Q.get_device().get_info<sycl::info::device::local_mem_size>();

		size_t storage_per_cone=ho_chebNodes.cols()*sizeof(T)+_CtFBufferSize<DIM>(order.maxCoeff(),high_order.maxCoeff());


		size_t conesPerGroup=local_mem_size/storage_per_cone;

		std::cout<<"using "<<conesPerGroup<<" as at the same time"<<std::endl;


		auto e=Q.submit([&](sycl::handler &h) {
		    // start by pushing  some data to the GPU (octree stuff)
		    sycl::accessor a_intData(*interpolationDataBuffer, h, sycl::read_only);
		    sycl::accessor a_parentIntData(*parentInterpolationDataBuffer, h, sycl::read_write);

		    std::array<int,DIM> ns=SyclHelpers::EigenVectorToCPPArray<int,DIM>(order);
		    std::array<int,DIM> ns_ho=SyclHelpers::EigenVectorToCPPArray<int,DIM>(high_order);


		    //sycl::local_accessor<T> coarseData(high_order.prod());

		    sycl::accessor a_hoChebNodes(b_hoChebNodes,h,sycl::read_only);
		
		    const auto &srcDataAcc = srcData->accessor(h);
		    const auto &parentDataAcc = parentData->accessor(h);
	
		
		    const size_t stride=ho_chebNodes.cols();
		    const size_t lo_stride=chebNodes.cols();
		    auto out = sycl::stream(100, 100, h);
		    //std::cout<<"survived setup1234"<<std::endl;

		    const auto functions =
			static_cast<Derived *>(this)->kernelFunctions();

		    const size_t numActiveCones= m_octree->numActiveCones(level,0);

		    
		    
		    h.parallel_for(sycl::range<1>( numActiveCones ), [=](sycl::id<1> i)
		    {
			ConeRef cone=srcDataAcc.fineActiveCone(i);

			size_t childBox=cone.boxId();
			auto center = srcDataAcc.boxCenter(childBox);
			PointScalar H = srcDataAcc.boxSize(childBox);
			const auto grid=srcDataAcc.coneDomain(childBox,0);
			    

			for(size_t chunkIdx=srcDataAcc.ctpData().fineConeShift(i);chunkIdx<srcDataAcc.ctpData().fineConeShift(i+1);chunkIdx++)
			{
			    	sycl::marray<PointScalar,DIM> pnt;
				sycl::marray<PointScalar,DIM> cart_pnt;
				sycl::marray<PointScalar,DIM> pnt2;

				ConeRef parentCone=srcDataAcc.ctpData().parentConeId(chunkIdx);
				size_t parentBoxId = parentCone.boxId();
				auto pGrid= parentDataAcc.coneDomain(parentBoxId,1);
				auto parent_center = parentDataAcc.boxCenter(parentBoxId);
				PointScalar pH = parentDataAcc.boxSize(parentBoxId);


				
				if( ! parentDataAcc.hasFarTargetsIncludingAncestors(parentBoxId)){ //we dont need the interpolation info for those levels.
				    return;
				}
				for(size_t pntOfChunk=srcDataAcc.ctpData().chunkShift(chunkIdx);pntOfChunk<srcDataAcc.ctpData().chunkShift(chunkIdx+1);++pntOfChunk) {
				    size_t pntIdx=srcDataAcc.ctpData().pntId(pntOfChunk);
				    pGrid.transform(parentCone.id(),a_hoChebNodes,pnt,pntIdx);
				    Util::interpToCart(pnt,cart_pnt,parent_center,pH);
								
				    //Transfer to the interpolation domain relative to the child box
				    Util::cartToInterp(cart_pnt,pnt2,center,H);
				    const size_t el2=grid.elementForPoint(pnt2);
				    assert(el2==cone.id());

			
				    const size_t memId=cone.globalId();
				    size_t el=cone.id();
				    grid.transformBackwards(el,pnt2,pnt);

				
				    SyclChebychevInterpolation::ClenshawEvaluator<T,1, DIM,DIM, DIMOUT> clenshaw;
				    const size_t offset=lo_stride*memId;
				    T res=clenshaw(SyclRowMatrix<PointScalar, DIM,1>(pnt), a_intData, ns, offset);
				
				    T TF=functions.transfer_factor(cart_pnt,center,H,parent_center,pH);


				    //
				    double & dest_real=*reinterpret_cast<double*>(& a_parentIntData[parentCone.globalId()*stride+pntIdx]);
				    double & dest_imag=*(reinterpret_cast<double*>(& a_parentIntData[parentCone.globalId()*stride+pntIdx])+1);
				    //double & dest_imag=reinterpret_cast<double&>a_parentIntData[parentCone.globalId()*stride+pntIdx].imag();
				    
					
				    sycl::atomic_ref<double,sycl::memory_order_relaxed,
						     sycl::memory_scope_device>    atomic_op(dest_real);

				    sycl::atomic_ref<double,sycl::memory_order_relaxed,
				    		     sycl::memory_scope_device>    atomic_op2(dest_imag);						


				    atomic_op+=(res*TF).real();
				    atomic_op2+=(res*TF).imag();
				    ///
				    //a_parentIntData[parentCone.globalId()*stride+pntIdx]+=res*TF;			    
				}
			
			}
		   
		    
		    });
		});
			
		Q.wait();
		/*std::cout<<"done old CTP"<<(e.template get_profiling_info<sycl::info::event_profiling::command_end>() -
		  e.template get_profiling_info<sycl::info::event_profiling::command_start>())/(1.0e9)<<std::endl;*/


		//Q.wait();
	    }else{ //"regular" CTP
		auto e=Q.submit([&](sycl::handler &h) {
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
			auto pGrid= parentDataAcc.coneDomain(parentBoxId,1);//m_octree->coneDomain(level-1,parentId,1);				    
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
		//Q.wait();
		/*std::cout<<"done old CTP"<<(e.template get_profiling_info<sycl::info::event_profiling::command_end>() -
		  e.template get_profiling_info<sycl::info::event_profiling::command_start>())/(1.0e9)<<std::endl;*/

	    }
	    Q.wait();

	    
            //std::swap(interpolationData, parentInterpolationData);

	    interpolationDataBuffer.reset();
	    std::swap(interpolationDataBuffer,parentInterpolationDataBuffer);	    
	    parentInterpolationDataBuffer.reset();
            //parentInterpolationData.resize(0);

	    std::cout<<"done with this level"<<std::endl;
        }
            std::cout<<"copying back"<<std::endl;
        } //end of sycl scope
	//std::cout<<"mult over\n";

	Eigen::Array<T, Eigen::Dynamic, DIMOUT> true_result(result.rows(),result.cols());
        Util::copy_with_inverse_permutation_rowwise<T,DIMOUT>(result, m_octree->targetPermutation(),true_result);

	return true_result;
    }


    void initInterpolationData(size_t level, size_t step, std::unique_ptr<sycl::buffer<T,1>> & buf )
    {
	assert(level<m_octree->levels());


	PointScalar H = m_octree->sideLength()*std::pow(0.5,level+5);
 	std::cout<<"h in init?"<<H<<std::endl;
        auto order = static_cast<Derived *>(this)->orderForBox(H, m_baseOrder,step);

	//make sure no old buffer is around	
	buf=std::make_unique<sycl::buffer<T,1> > (m_octree->numActiveCones(level,step)*order.prod());
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


    inline  PointScalar  cutoff_limit(PointScalar H, Eigen::Vector<int,DIM> baseOrder) const
    {
	return 2e-3;
    }


    inline PointScalar tolerance() const {
	return m_tolerance;
    }

    int levels() const {
	return m_octree->levels();
    }

    bool farfieldCanBeSkipped(PointScalar H) const {    
         return false;
    }

protected:
    void onOctreeReady()
    {
	//do nothing. but give subclasses the opportunity to initialize some things
    }

    std::shared_ptr<FlatOctree<T, DIM> > m_octree;


private:
#if 0
    template<int package,typename A1, typename A2, typename A3,typename A4>
    static inline  void __eval_mult_add(const typename SyclChildToParentData<T,DIM>::ConeData& data, A1& a_parentIntData, size_t shift, const A2& val_cache,
					const A3& ctpData, const A4 a_transfer_factors, size_t tf_shift, const std::array<int,DIM>& ns,size_t pnt_shift, size_t n_points)  {
	
	const int packageSize=1 << package;
	
	SyclChebychevInterpolation::ClenshawEvaluator<T, packageSize,  DIM,DIM, DIMOUT> clenshaw;
	
	SyclRowMatrix<PointScalar,DIM,packageSize> tmp;
	const size_t np = n_points / packageSize;

	n_points = n_points - np*packageSize;
	size_t pnt=pnt_shift;
	
	for(size_t pkg=0;pkg<np;pkg++){
	    for(int l=0;l<packageSize;l++) { //TODO:more efficient way?
		for(int k=0;k<DIM;k++) {
		    tmp(k,l)=ctpData.points()[(pnt+l)*DIM+k];
		}
	    }
	    const auto result=clenshaw(tmp,val_cache,ns,0); //a_intData, offset
	    
	    for(size_t l=0;l<packageSize && pnt+l<data.pnts.second;l++) {
		size_t pntId=pnt+l;
		//TF shifts
		T TF=a_transfer_factors[tf_shift+pntId];//functions.transfer_factor(cart_pnt,center,H,parent_center,pH);				    
		a_parentIntData[shift+ctpData.realPointId(pntId)]+=result[l]*TF;
		
	    }
	    
	    pnt+=packageSize;
	}

	if constexpr(package>0) {
	    if(pnt<data.pnts.second) {
		return __eval_mult_add<std::max(package-2,0)>(data,a_parentIntData,shift,val_cache,ctpData,a_transfer_factors, tf_shift,ns,pnt,n_points);
	    }
	}
    }
#endif

private:
    unsigned int m_maxLeafSize;
    size_t m_numTargets;
    size_t m_numSrcs;
	std::shared_ptr<Octree<T,DIM>> m_src_octree;
    Eigen::Vector<size_t, DIM> m_base_n_elements;
    Eigen::Vector<int, DIM> m_baseOrder;
    PointScalar m_tolerance;

	struct NearFieldMetaData{
		uint32_t targetStart;
		uint32_t targetEnd;
		uint32_t sourceStart;
		uint32_t sourceEnd;

		double Center[DIM];
		double H;
	};

	std::vector<std::vector<NearFieldMetaData>> m_allNearFieldMetaData;


	struct ConeMetaData {
		double center[DIM];
		double H;
		uint32_t coeffOffset;   // globalId * stride
		uint32_t targetOffset;  // start index in targetIds / normPoints
		uint32_t numTargets;
		uint32_t stride;        // = order.prod() – stored for safety
		uint32_t localConeIdx;  // optional, for debugging
	};

	struct ConeTargetData {
		uint32_t targetId;
		// we will store normPoints separately, not in this struct
	};

	struct InfoPerLevel {
		std::vector<ConeMetaData> metaData;
		std::vector<uint32_t> targetIds;
		std::vector<PointScalar> normPoints; // DIM values per target, flat
	};
	std::vector<InfoPerLevel> m_allLevelConeInfo;
};

#endif
