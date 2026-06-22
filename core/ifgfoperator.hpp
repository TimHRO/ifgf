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

#include <chrono>
#include <thread>

//#include <fstream>
#include <iostream>
#include <algorithm>
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
// template<int DIM>
// constexpr int _CtFBufferSize(int order, int high_order)
// {
//     size_t buffer_size = 0;
//     
//     int Np_xy = 1;
//     for(int j = 0; j < DIM-1; j++) {
//         Np_xy *= (j == 0 ? std::max(order-2, 2) : order);
//     }
//     buffer_size += high_order * Np_xy;  // high_order here, not order

//     // level DIM-2 down to 1: all use lo_order
//     for(int d = DIM-2; d > 0; d--) {
//         int Np = 1;
//         for(int j = 0; j < d; j++) {
//             Np *= (j == 0 ? std::max(order-2, 2) : order);
//         }
//         buffer_size += high_order * Np;
//     }

//     return buffer_size;
// }




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

		// ------------------------------------------------
		// Build cone‑centric data for far‑field GPU kernel
		// ------------------------------------------------


		PointScalar H0 = m_octree->sideLength();
		auto order0 = static_cast<Derived *>(this)->orderForBox(
			H0 * std::pow(0.5, m_octree->levels() + 5), m_baseOrder, 0);
		const size_t stride = order0.prod();   // number of Chebyshev coefficients per cone

		m_allLevelConeInfo.resize(nLevels);

		// Build target-centric far-field data


		const size_t numTargets = m_octree->targetPoints().cols();
		const auto kf = static_cast<Derived*>(this)->kernelFunctions();

		struct FarFieldEntry {
			uint32_t targetId;
			uint32_t fineMemId;
			float    normPnt[DIM];
			float    cf_real;
			float    cf_imag;
		};

		for (size_t level = 0; level < nLevels; ++level) {
			InfoPerLevel& info = m_allLevelConeInfo[level];

			const size_t numBoxes = m_src_octree->numBoxes(level);

			// CPU thread-local entry lists, no locking during parallel box processing
			tbb::enumerable_thread_specific<std::vector<FarFieldEntry>> tls_entries;

			tbb::parallel_for(tbb::blocked_range<size_t>(0, numBoxes),
				[&](const tbb::blocked_range<size_t>& r) {
				auto& local = tls_entries.local();
				for (size_t boxIdx = r.begin(); boxIdx < r.end(); ++boxIdx) {
					if (!m_src_octree->hasFarTargetsIncludingAncestors(level, boxIdx))
						continue;

					const auto bbox   = m_src_octree->bbox(level, boxIdx);
					const auto center = bbox.center();
					const PointScalar H = bbox.sideLength();
					const auto& coneDomain = m_src_octree->coneDomain(level, boxIdx, 0);
					const auto& coneMap    = m_src_octree->coneMaps(level)[boxIdx];

					const auto& farRanges = m_src_octree->farTargets(level, boxIdx);
					for (const auto& range : farRanges) {
						for (size_t tIdx = range.first; tIdx < range.second; ++tIdx) {
							const auto pnt        = m_octree->targetPoints().col(tIdx).matrix();
							const auto transformed = Util::cartToInterp<DIM>(pnt, center, H);

							const size_t el = coneDomain.elementForPoint(transformed);
							if (el == SIZE_MAX) continue;

							auto it = coneMap.find(el);
							if (it == coneMap.end()) continue;
							const uint32_t fineMemId = static_cast<uint32_t>(it->second);

							const auto normMat = coneDomain.transformBackwards(el, transformed);

							sycl::marray<PointScalar,DIM> diff;
							for (int d = 0; d < DIM; ++d)
								diff[d] = static_cast<PointScalar>(pnt[d] - center[d]);
							const auto cf = kf.CF(diff, H);

							FarFieldEntry e;
							e.targetId  = static_cast<uint32_t>(tIdx);
							e.fineMemId = fineMemId;
							for (int d = 0; d < DIM; ++d)
								e.normPnt[d] = static_cast<float>(normMat(d, 0));
							e.cf_real = static_cast<float>(cf.real());
							e.cf_imag = static_cast<float>(cf.imag());
							local.push_back(e);
						}
					}
				}
			});

            // Build CSR

			// count entries per target across all thread-local vectors
			size_t totalEntries = 0;
			for (auto& v : tls_entries) totalEntries += v.size();

			info.tgtConeShift.assign(numTargets + 1, 0);
			for (auto& v : tls_entries)
				for (auto& e : v)
					++info.tgtConeShift[e.targetId + 1];

			// Compute prefix sum for shift
			for (size_t t = 0; t < numTargets; ++t)
				info.tgtConeShift[t+1] += info.tgtConeShift[t];

			info.tgtConeIds.resize(totalEntries);
			info.tgtNormPnts.resize(totalEntries * DIM);
			info.tgtCF_real.resize(totalEntries);
			info.tgtCF_imag.resize(totalEntries);
			std::vector<size_t> cursor(info.tgtConeShift.begin(),
			                           info.tgtConeShift.begin() + numTargets);
			for (auto& v : tls_entries) {
				for (auto& e : v) {
					const size_t pos       = cursor[e.targetId]++;
					info.tgtConeIds[pos]   = e.fineMemId;
					for (int d = 0; d < DIM; ++d)
						info.tgtNormPnts[pos*DIM+d] = e.normPnt[d];
					info.tgtCF_real[pos]   = e.cf_real;
					info.tgtCF_imag[pos]   = e.cf_imag;
				}
			}

			info.tgtConeShift.shrink_to_fit();
			info.tgtConeIds.shrink_to_fit();
			info.tgtNormPnts.shrink_to_fit();
			info.tgtCF_real.shrink_to_fit();
			info.tgtCF_imag.shrink_to_fit();

			std::cout << "Level " << level << ": "
			          << totalEntries << " far-field target interactions" << std::endl;
		}

		std::cout << "done initializing" << std::endl;
		precomputeCtpData();
	}

	void precomputeCtpData()
	{
		const auto functions  = static_cast<Derived*>(this)->kernelFunctions();
		const size_t nLevels  = m_src_octree->levels();
		const PointScalar H0  = m_octree->sideLength();
		const auto high_order = static_cast<Derived*>(this)->orderForBox(
			H0 * std::pow(0.5, nLevels + 5), m_baseOrder, 1);
		const auto& ho_chebNodes =
			ChebychevInterpolation::chebnodesNdd<PointScalar,DIM>(high_order);

		const size_t hw = std::max(1u, std::thread::hardware_concurrency());
		tbb::global_control tbb_ctl(tbb::global_control::max_allowed_parallelism, hw);

		m_ctpGeom.resize(nLevels);

		for (size_t level = 1; level < nLevels; ++level) {
			auto levelData = m_octree->data(level);

			const auto& fineConeShifts = levelData->h_fineConeShifts();
			const auto& chunkShifts    = levelData->h_chunkShifts();
			const auto& parentConeIds  = levelData->h_parentConeIds();
			const auto& pntIds         = levelData->h_pntIds();
			const size_t totalPnts     = pntIds.size();
			if (totalPnts == 0) continue;

			CtpGeomLevel& geom = m_ctpGeom[level];
			geom.localPnts.resize(totalPnts * DIM);
			geom.TF_real.resize(totalPnts);
			geom.TF_imag.resize(totalPnts);

			const size_t numFineCones = m_octree->numActiveCones(level, 0);

			tbb::parallel_for(tbb::blocked_range<size_t>(0, numFineCones),
				[&](const tbb::blocked_range<size_t>& r) {
				for (size_t fineMemId = r.begin(); fineMemId < r.end(); ++fineMemId) {

					ConeRef fineCone          = levelData->fineActiveCone(fineMemId);
					const size_t childBox     = fineCone.boxId();
					const auto childBbox      = m_src_octree->bbox(level, childBox);
					const auto child_center   = childBbox.center();
					const PointScalar child_H = childBbox.sideLength();

					const ConeDomain<DIM>& fineGrid =
						m_src_octree->coneDomain(level, childBox, 0);
					const auto fineIndices = fineGrid.indicesFromId(fineCone.id());
					const auto& fdom = fineGrid.domain();
					Eigen::Vector<PointScalar,DIM> fine_a, fine_b;
					for (int d = 0; d < DIM; ++d) {
						const PointScalar fh =
							fdom.diagonal()[d] / (PointScalar)fineGrid.n_elements(d);
						fine_a[d] = PointScalar(0.5) * fh;
						fine_b[d] = fdom.min()[d]
							+ ((PointScalar)fineIndices[d] + PointScalar(0.5)) * fh;
					}

					for (size_t chunkIdx = fineConeShifts[fineMemId];
						     chunkIdx < fineConeShifts[fineMemId + 1]; ++chunkIdx) {

						ConeRef parentCone       = parentConeIds[chunkIdx];
						const size_t parentBox   = parentCone.boxId();
						const auto parentBbox    = m_src_octree->bbox(level-1, parentBox);
						const auto parent_center = parentBbox.center();
						const PointScalar pH     = parentBbox.sideLength();
						const ConeDomain<DIM>& parentGrid =
							m_src_octree->coneDomain(level-1, parentBox, 1);

						const auto interp_pnts =
							parentGrid.transform(parentCone.id(), ho_chebNodes);
						const auto cart_pnts =
							Util::interpToCart<DIM>(interp_pnts.array(), parent_center, pH);

						for (size_t k = chunkShifts[chunkIdx];
							     k < chunkShifts[chunkIdx + 1]; ++k) {

							const size_t nodeIdx = pntIds[k];
							const auto cart_pnt  = cart_pnts.col(nodeIdx).matrix().eval();
							const auto pnt2      =
								Util::cartToInterp<DIM>(cart_pnt, child_center, child_H);

							for (int d = 0; d < DIM; ++d)
								geom.localPnts[k * DIM + d] =
									float((pnt2[d] - fine_b[d]) / fine_a[d]);

							const T tf = functions.transfer_factor(
								cart_pnt, child_center, child_H, parent_center, pH);
							geom.TF_real[k] = float(tf.real());
							geom.TF_imag[k] = float(tf.imag());
						}
					}
				}
			});

			// Upload geometry to GPU as IfgfOperator-owned SYCL buffers
			geom.buf_localPnts = std::make_unique<sycl::buffer<float,1>>(geom.localPnts.data(),totalPnts * DIM);
			geom.buf_TF_real   = std::make_unique<sycl::buffer<float,1>>(geom.TF_real.data(),totalPnts);
			geom.buf_TF_imag   = std::make_unique<sycl::buffer<float,1>>(geom.TF_imag.data(),totalPnts);


			std::cout << "Level " << level << " CTP geometry precomputed: "
			          << totalPnts << " entries" << std::endl;
		}
	}

    Eigen::Array<T, Eigen::Dynamic,DIMOUT> mult(const Eigen::Ref<const Eigen::Vector<T, Eigen::Dynamic> > &weights)
    {
	switch(m_baseOrder.maxCoeff()) {
	case 1: 
	case 2: 
	case 3: 
	case 4:  return mult_impl<4>(weights);
	case 5:  
 	case 6:  return mult_impl<6>(weights);
	case 7:  
	case 8:  return mult_impl<8>(weights);
	    //	case 16:  return mult_impl<16>(weights);
	    //case 32:  return mult_impl<32>(weights);
	case 10:  return mult_impl<10>(weights);
	default: std::cout<<"not implemented"<<m_baseOrder.transpose()<<std::endl; return mult_impl<8>(weights);
	}
	
    }





    template <int MAX_ORDER>
    Eigen::Array<T, Eigen::Dynamic,DIMOUT> mult_impl(const Eigen::Ref<const Eigen::Vector<T, Eigen::Dynamic> > &weights)
    {
	using namespace std::chrono;
    	high_resolution_clock::time_point t1 = high_resolution_clock::now();
        std::cout<<"multimpl"<<std::endl;
        Eigen::Array<T, Eigen::Dynamic, DIMOUT> result(m_numTargets,DIMOUT);
        result.fill(0);
        int level = levels() - 1;

	//std::cout<<"boxes="<<m_octree->numBoxes(level)<<std::endl;
	const PointScalar hmin=m_octree->diameter()*std::pow(0.5,m_octree->levels());
	//std::vector<tbb::queuing_mutex> resultMutex(m_numTargets);

        { //scope to contain all the sycl stuff. that way we make sure that all the data is copied to the host before proceeding.

	sycl::queue& Q=SyclHelpers::QueueSingleton::getInstance().queue();//(sycl::default_selector_v);
	auto dev = Q.get_device();
	std::cout << "Device: "
          << dev.get_info<sycl::info::device::name>()
          << "\n";

	std::cout << "Max work-group size: "
			<< dev.get_info<
					sycl::info::device::max_work_group_size>()
			<< "\n";

	std::cout << "Local memory size (bytes): "
			<< dev.get_info<
					sycl::info::device::local_mem_size>()
			<< "\n";

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

		//-----------------------
		// near Field computation
		//-----------------------

		{
		Q.wait();
		std::cout<<"nearfield"<<std::endl;
		auto e=Q.submit([&](sycl::handler &h) {
		    // start by pushing  some data to the GPU (octree stuff)
		    sycl::accessor a_srcs(b_srcs, h, sycl::read_only);
		    sycl::accessor a_targets(b_targets, h, sycl::read_only);
		    sycl::accessor a_weights(b_weights, h, sycl::read_only);

		    sycl::accessor a_result(b_result, h, sycl::read_write);

		    const auto &srcDataAcc = srcData->accessor(h);
		    const auto functions =
			static_cast<Derived *>(this)->kernelFunctions();


		    //auto out = sycl::stream(1024, 768, h);
		    const size_t num_targets=m_octree->targetPoints().cols();

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

		std::unique_ptr<sycl::buffer<T,1>> parentCTPBuffer;
		bool hasParentData = false;

		if (level > 0) {
			size_t numParentCones = m_octree->numActiveCones(level-1, 1);
			if (numParentCones > 0) {
				initInterpolationData(level-1, 1, parentCTPBuffer);
				hasParentData = true;
				Q.submit([&](sycl::handler& h){
					sycl::accessor a(*parentCTPBuffer, h, sycl::write_only, sycl::no_init);
					h.fill(a, T(0));
				}).wait();
			} else {
				// No parent cones – create dummy buffer to avoid null dereference
				parentCTPBuffer = std::make_unique<sycl::buffer<T,1>>(sycl::range<1>(1));
			}
		} else {
			// level == 0 – dummy buffer (unused)
			parentCTPBuffer = std::make_unique<sycl::buffer<T,1>>(sycl::range<1>(1));
		}
		Q.wait();

		// set parentData before the kernel so we can access level-1 geometry inside it
		if(level > 0)
			parentData = m_octree->data(level-1);

		
		Q.wait();

		//-----------------------------------------------
		// Interpolation Data + Far Fiel Evaluation + CTF
		//-----------------------------------------------

		{

			const size_t stride = ho_chebNodes.cols();
			const size_t fine_stride = order.prod();
			const size_t numActive = m_octree->numActiveCones(level,1);

			if(numActive==0) continue;

			//std::array<int, DIM> ho_ns;
			//std::copy(high_order.begin(), high_order.end(), ho_ns.begin());
			std::cout << "Launching work group with (high_order_stride)" << stride << " threads" << "\n";
			std::cout << "Fine stride is " << fine_stride << "\n"; 

			auto& thisLevelConeInfo = m_allLevelConeInfo[level];

			// sycl buffers to far-field info including centered factor and normed points
			const bool hasTgt = !thisLevelConeInfo.tgtConeShift.empty();
			const size_t numTgtEntries = thisLevelConeInfo.tgtConeIds.size();
			sycl::buffer<size_t,1>   buf_tgtShift(
				hasTgt ? thisLevelConeInfo.tgtConeShift.data() : nullptr,
				sycl::range<1>(hasTgt ? thisLevelConeInfo.tgtConeShift.size() : 2));
			sycl::buffer<uint32_t,1> buf_tgtConeIds(
				hasTgt && numTgtEntries > 0 ? thisLevelConeInfo.tgtConeIds.data() : nullptr,
				sycl::range<1>(std::max(numTgtEntries, (size_t)1)));
			sycl::buffer<float,1>    buf_tgtNormPnts(
				hasTgt && numTgtEntries > 0 ? thisLevelConeInfo.tgtNormPnts.data() : nullptr,
				sycl::range<1>(std::max(numTgtEntries * DIM, (size_t)1)));
			sycl::buffer<float,1>    buf_tgtCFr(
				hasTgt && numTgtEntries > 0 ? thisLevelConeInfo.tgtCF_real.data() : nullptr,
				sycl::range<1>(std::max(numTgtEntries, (size_t)1)));
			sycl::buffer<float,1>    buf_tgtCFi(
				hasTgt && numTgtEntries > 0 ? thisLevelConeInfo.tgtCF_imag.data() : nullptr,
				sycl::range<1>(std::max(numTgtEntries, (size_t)1)));




			// CTP geometry buffers, GPU kernel does not need to access geometrie info
			const bool hasCTPGeom = (level > 2) && hasParentData
			                       && (level < m_ctpGeom.size())
			                       && (m_ctpGeom[level].buf_localPnts != nullptr);
			sycl::buffer<float,1>* ctpBufLP  = hasCTPGeom
			    ? m_ctpGeom[level].buf_localPnts.get() : nullptr;
			sycl::buffer<float,1>* ctpBufTFr = hasCTPGeom
			    ? m_ctpGeom[level].buf_TF_real.get()   : nullptr;
			sycl::buffer<float,1>* ctpBufTFi = hasCTPGeom
			    ? m_ctpGeom[level].buf_TF_imag.get()   : nullptr;
			// Dummy buffers for levels where CTP doesn't run
			sycl::buffer<float,1> ctpDummy(sycl::range<1>(1));

			const size_t numFineCones_ff = m_octree->numActiveCones(level, 0);
			sycl::buffer<T,1> fineInterpBuffer(
			    sycl::range<1>(numFineCones_ff * fine_stride));


			// Need to be known at compile time
			// TODO make MAX_LEAF_SRCS generic
			constexpr size_t MAX_STRIDE      = (size_t)MAX_ORDER * MAX_ORDER * MAX_ORDER;
			constexpr int    MAX_LOW_ORDER_K  = std::max(MAX_ORDER-3, 1);
			constexpr size_t MAX_FINE_STRIDE  = (size_t)MAX_LOW_ORDER_K * MAX_LOW_ORDER_K * MAX_LOW_ORDER_K;
			constexpr size_t MAX_LEAF_SRCS    = 300; 

			// Build compact list of leaf cone indices on CPU
			std::vector<uint32_t> h_leafIds;
			h_leafIds.reserve(numActive / 4);
			for (size_t ci = 0; ci < numActive; ++ci) {
				const ConeRef ref = srcData->activeCone(ci);
				if (m_src_octree->isLeaf(level, ref.boxId()))
					h_leafIds.push_back(static_cast<uint32_t>(ci));
			}
			const size_t numLeafCones = h_leafIds.size();

			if (numLeafCones > 0) {
				sycl::buffer<uint32_t,1> buf_leafIds(
					h_leafIds.data(), sycl::range<1>(numLeafCones));

				Q.submit([&](sycl::handler& h_leaf) {
					sycl::accessor a_srcs_l    (b_srcs,    h_leaf, sycl::read_only);
					sycl::accessor a_weights_l (b_weights, h_leaf, sycl::read_only);
					sycl::accessor a_intData_l (*interpolationDataBuffer, h_leaf, sycl::read_write);
					sycl::accessor a_hoChebNodes_l(b_hoChebNodes, h_leaf, sycl::read_only);
					sycl::accessor<uint32_t,1,sycl::access_mode::read>
					    a_leafIds(buf_leafIds, h_leaf);
					const auto& srcDataAcc_l = srcData->accessor(h_leaf);
					const auto  functions_l  = static_cast<Derived*>(this)->kernelFunctions();

					sycl::local_accessor<PointScalar,1>
					    sh_srcs(sycl::range<1>(MAX_LEAF_SRCS * DIM), h_leaf);
					sycl::local_accessor<T,1>
					    sh_ws  (sycl::range<1>(MAX_LEAF_SRCS),       h_leaf);

					h_leaf.parallel_for(
						sycl::nd_range<1>(
							sycl::range<1>(numLeafCones * MAX_STRIDE),
							sycl::range<1>(MAX_STRIDE)),
						[=](sycl::nd_item<1> item) {
						const size_t groupId = item.get_group(0);
						const size_t nodeId  = item.get_local_id(0);

						const uint32_t coneIdx = a_leafIds[groupId];
						const ConeRef  ref     = srcDataAcc_l.activeCone(coneIdx);
						const size_t   boxId   = ref.boxId();

						const IndexRange srcs  = srcDataAcc_l.points(boxId);
						const size_t     nSrcs = srcs.second - srcs.first;

						for (size_t s = nodeId; s < nSrcs; s += MAX_STRIDE) {
							const size_t si = srcs.first + s;
							for (int d = 0; d < DIM; d++)
								sh_srcs[s*DIM+d] = a_srcs_l[si*DIM+d];
							sh_ws[s] = a_weights_l[si];
						}
						item.barrier(sycl::access::fence_space::local_space);

						// All MAX_STRIDE threads active, no divergence
						// All threads in a workgroup reuse geomtrie info from cache since boxId is equal
						if (nodeId >= stride) return;

						const sycl::marray<PointScalar,DIM> center = srcDataAcc_l.boxCenter(boxId);
						const PointScalar H_box = srcDataAcc_l.boxSize(boxId);
						const auto grid = srcDataAcc_l.coneDomain(boxId, 1);

						sycl::marray<PointScalar,DIM> transformed, cartesian;
						grid.transform(ref.id(), a_hoChebNodes_l, transformed, nodeId);
						Util::interpToCart(transformed, cartesian, center, H_box);

						a_intData_l[ref.globalId() * stride + nodeId] =
							functions_l.evaluateFactoredKernel(
								sh_srcs, (size_t)0, nSrcs,
								cartesian, sh_ws, center, H_box);
					});
				});
				Q.wait();
			}

			auto e = Q.submit([&](sycl::handler &h){
				//sycl::stream out(1024, 256, h);
				sycl::accessor a_srcs(b_srcs, h, sycl::read_only);
				sycl::accessor a_points(b_points, h, sycl::read_only);		    
				sycl::accessor a_weights(b_weights, h, sycl::read_only);
				sycl::accessor a_intData(*interpolationDataBuffer, h, sycl::read_write);
				//sycl::accessor a_parentIntData(*parentInterpolationDataBuffer, h, sycl::read_write);
				sycl::accessor a_hoChebNodes(b_hoChebNodes, h, sycl::read_only);
				sycl::accessor a_hoChebVals(b_hoChebvals, h, sycl::read_only);
				sycl::accessor a_chebVals(b_chebvals, h, sycl::read_only);



				sycl::accessor a_targets(b_targets, h, sycl::read_only);
				sycl::accessor a_result(b_result, h, sycl::read_write);

				// fine coefficient output — read by CTP kernel and far-field kernel
				sycl::accessor<T,1,sycl::access_mode::write> a_fineInterpData(
				    fineInterpBuffer, h, sycl::no_init);

				const auto &srcDataAcc = srcData->accessor(h);
				const auto functions =
				static_cast<Derived *>(this)->kernelFunctions();
				//sycl::local_accessor<T,1> rawData(sycl::range<1>(stride), h);

				// precompute parameters for coarse to fine refinement

				const double H=H0*pow(2,-level);
				Eigen::Vector<size_t,DIM> coarse_N=static_cast<Derived *>(this)->elementsForBox(H, this->m_baseOrder,this->m_base_n_elements);
				Eigen::Vector<size_t,DIM> fine_N=(size_t) (std::pow((unsigned int) REFINEMENT_FACTOR, (unsigned int) ( REFINEMENT_LEVELS)))*coarse_N;//
				std::array<int, DIM> n_elements=SyclHelpers::EigenVectorToCPPArray<int,DIM>(fine_N.template cast<int>());
				std::array<size_t, DIM> n_el=SyclHelpers::EigenVectorToCPPArray<size_t,DIM>(coarse_N);
				sycl::accessor a_chebNodes(b_chebNodes,h,sycl::read_only);
				const int nF=std::pow(REFINEMENT_FACTOR,DIM); 
				constexpr int MAX_LOW_ORDER=std::max(MAX_ORDER-3,1);
				//constexpr int BUF_SIZE=_CtFBufferSize<DIM>(MAX_LOW_ORDER,MAX_ORDER);
				constexpr int BUF_SIZE = MAX_ORDER * (MAX_ORDER-3) * (MAX_ORDER-3) + MAX_ORDER * (MAX_ORDER-3);

				std::cout << "rawData local mem = " << nF * stride * sizeof(T) << " bytes\n";
				std::cout << "local mem limit = " 
						<< Q.get_device().get_info<sycl::info::device::local_mem_size>() 
						<< " bytes\n";

				//constexpr int MAX_LOW_ORDER = std::max(MAX_ORDER-3, 1); // = 5
				size_t true_buf = (size_t)ho_ns[2] * lo_ns[0] * lo_ns[1]
								+ (size_t)ho_ns[1] * lo_ns[0];
				std::cout << "MAX_ORDER=" << MAX_ORDER 
						<< " MAX_LOW_ORDER=" << MAX_LOW_ORDER
						<< " ho_ns=" << ho_ns[0] << "," << ho_ns[1] << "," << ho_ns[2]
						<< " lo_ns=" << lo_ns[0] << "," << lo_ns[1] << "," << lo_ns[2]
						<< " true_buf=" << true_buf 
						<< " BUF_SIZE=" << BUF_SIZE << "\n";

				h.parallel_for(sycl::range<1>(numActive), [=](sycl::id<1> i){
					const ConeRef ref = srcDataAcc.activeCone(i);
					const size_t boxId = ref.boxId();
					//if(!srcDataAcc.hasFarTargetsIncludingAncestors(boxId)) return;

					const size_t globalOffset = ref.globalId() * stride;

					T coarseCoeffs[MAX_STRIDE];

					for(size_t node = 0; node < stride; node++)
						coarseCoeffs[node] = a_intData[globalOffset + node];

					SyclChebychevInterpolation::chebtransform_inplace<T,DIM,MAX_ORDER>(
						coarseCoeffs, ho_ns, a_hoChebVals, 0);

					auto ho_id = SyclConeDomain<DIM>::indicesFromId(ref.id(), n_el);
					std::array<size_t,DIM> factors;
					factors.fill(REFINEMENT_FACTOR);

					for(size_t sub = 0; sub < nF; sub++){
						auto lid = SyclConeDomain<DIM>::indicesFromId(sub, factors);
						const size_t fine_el =
							(ho_id[2]*REFINEMENT_FACTOR+lid[2])*n_elements[1]*n_elements[0]+
							(ho_id[1]*REFINEMENT_FACTOR+lid[1])*n_elements[0]+
							(ho_id[0]*REFINEMENT_FACTOR+lid[0]);
						const size_t fineMemId = srcDataAcc.memId(ref.boxId(), fine_el);
						if(fineMemId >= SIZE_MAX-1) continue;

						T fineCoeffs[MAX_FINE_STRIDE];
						for(size_t i=0; i<fine_stride; i++){
							fineCoeffs[i]=T(0);
						}
						sycl::marray<PointScalar, MAX_ORDER*DIM> t_pnts(PointScalar(0));
						sycl::marray<T, BUF_SIZE> tmp(T(0));

						size_t offset = 0;
						for(int d = 0; d < DIM; d++){
							const PointScalar h = 2;
							const PointScalar mmin = -1 + (lid[d]*(h/(PointScalar)REFINEMENT_FACTOR));
							const PointScalar mmax = mmin + (h/(PointScalar)REFINEMENT_FACTOR);
							const PointScalar a = 0.5*(mmax-mmin);
							const PointScalar b = 0.5*(mmax+mmin);
							for(size_t l = 0; l < lo_ns[d]; l++){
								t_pnts[offset] = a*a_points[offset]+b;
								offset++;
							}
						}

						SyclChebychevInterpolation::tp_evaluate_t<T,DIM>(
							t_pnts, coarseCoeffs, 0,
							ho_ns, lo_ns, fineCoeffs, tmp, 0, 0);

						SyclChebychevInterpolation::chebtransform_inplace<T,DIM,MAX_ORDER>(
							fineCoeffs, lo_ns, a_chebVals, 0);

						const size_t fineOffset = fineMemId * fine_stride;
						for(size_t j = 0; j < fine_stride; ++j)
							a_fineInterpData[fineOffset + j] = fineCoeffs[j];
					} // end for(sub)
				});
			});

	    Q.wait();


		if (hasParentData && level > 2) {
			const size_t numFineCones_ctp = m_octree->numActiveCones(level, 0);
			Q.submit([&](sycl::handler& h_ctp) {
				sycl::accessor<T,1,sycl::access_mode::read>
				    a_fineCoeffs_ctp(fineInterpBuffer, h_ctp);
				sycl::accessor<T,1,sycl::access_mode::read_write>
				    a_parentCTP(*parentCTPBuffer, h_ctp);
				sycl::accessor<float,1,sycl::access_mode::read> a_ctpLP(
				    hasCTPGeom ? *ctpBufLP  : ctpDummy, h_ctp);
				sycl::accessor<float,1,sycl::access_mode::read> a_ctpTFr(
				    hasCTPGeom ? *ctpBufTFr : ctpDummy, h_ctp);
				sycl::accessor<float,1,sycl::access_mode::read> a_ctpTFi(
				    hasCTPGeom ? *ctpBufTFi : ctpDummy, h_ctp);
				const auto& srcDataAcc_ctp = srcData->accessor(h_ctp);
				const std::array<int,DIM> lo_ns_ctp  = lo_ns;
				const std::array<int,DIM> ho_ns_ctp  = ho_ns;
				const size_t fine_stride_ctp = fine_stride;
				const size_t stride_ctp      = stride;

				h_ctp.parallel_for(
				    sycl::range<1>(numFineCones_ctp),
				    [=](sycl::id<1> fineMemId) {
				    if (!hasCTPGeom) return;

				    const auto& ctp = srcDataAcc_ctp.ctpData();
				    const size_t chunkStart = ctp.fineConeShift(fineMemId);
				    const size_t chunkEnd   = ctp.fineConeShift(fineMemId + 1);
				    if (chunkStart == chunkEnd) return;

				    // Read fine coefficients from fineInterpBuffer
				    const size_t fineOffset = fineMemId * fine_stride_ctp;

				    SyclChebychevInterpolation::ClenshawEvaluator<T,1,DIM,DIM,DIMOUT> clenshaw;

				    for (size_t chunkIdx = chunkStart; chunkIdx < chunkEnd; ++chunkIdx) {
				        const ConeRef parentCone  = ctp.parentConeId(chunkIdx);
				        const size_t  parentMemId = parentCone.globalId();
				        for (size_t k = ctp.chunkShift(chunkIdx);
				                    k < ctp.chunkShift(chunkIdx + 1); ++k) {
				            sycl::marray<PointScalar,DIM> localPnt;
				            for (int d = 0; d < DIM; ++d)
				                localPnt[d] = static_cast<PointScalar>(
				                    a_ctpLP[k * DIM + d]);
				            const T res = clenshaw(
				                SyclRowMatrix<PointScalar,DIM,1>(localPnt),
				                a_fineCoeffs_ctp, lo_ns_ctp, fineOffset);
				            const T TF(
				                static_cast<typename T::value_type>(a_ctpTFr[k]),
				                static_cast<typename T::value_type>(a_ctpTFi[k]));
				            const T contrib = res * TF;
				            const size_t pntIdx = ctp.pntId(k);
				            using ScalarT = typename T::value_type;
				            ScalarT* bp = reinterpret_cast<ScalarT*>(
				                &a_parentCTP[parentMemId * stride_ctp + pntIdx]);
				            sycl::atomic_ref<ScalarT,
				                sycl::memory_order::relaxed,
				                sycl::memory_scope::device,
				                sycl::access::address_space::global_space>
				                ar(bp[0]), ai(bp[1]);
				            ar.fetch_add(contrib.real());
				            ai.fetch_add(contrib.imag());
				        }
				    }
				});
			});
			Q.wait();
		}

		// Target-centric far-field kernel
		// Each target thread gathers from its fine cones and accumulates locally
		// No atomic adds needed
		// Quasi atomic add between levels needed but we have implicit synchronization between levels
		if (hasTgt && numTgtEntries > 0) {
			const size_t numTargets_ff = m_octree->targetPoints().cols();
			Q.submit([&](sycl::handler& h){
				sycl::accessor<T,1,sycl::access_mode::read>      a_fineCoeffs(fineInterpBuffer, h);
				sycl::accessor<size_t,1,sycl::access_mode::read>  a_tgtShift(buf_tgtShift, h);
				sycl::accessor<uint32_t,1,sycl::access_mode::read> a_tgtConeIds(buf_tgtConeIds, h);
				sycl::accessor<float,1,sycl::access_mode::read>   a_tgtNorm(buf_tgtNormPnts, h);
				sycl::accessor<float,1,sycl::access_mode::read>   a_tgtCFr(buf_tgtCFr, h);
				sycl::accessor<float,1,sycl::access_mode::read>   a_tgtCFi(buf_tgtCFi, h);
				sycl::accessor<T,1,sycl::access_mode::read_write> a_result2(b_result, h);
				const std::array<int,DIM> lo_ns_cap = lo_ns;
				h.parallel_for(sycl::range<1>(numTargets_ff), [=](sycl::id<1> tgtId){
					const size_t tStart = a_tgtShift[tgtId];
					const size_t tEnd   = a_tgtShift[tgtId + 1];
					if (tStart == tEnd) return;
					T result(0);
					SyclChebychevInterpolation::ClenshawEvaluator<T,1,DIM,DIM,DIMOUT> clenshaw;
					for (size_t ci = tStart; ci < tEnd; ++ci) {
						const uint32_t fineMemId = a_tgtConeIds[ci];
						const size_t fineOffset  = fineMemId * fine_stride;
						sycl::marray<PointScalar,DIM> norm;
						for (int d = 0; d < DIM; d++)
							norm[d] = static_cast<PointScalar>(a_tgtNorm[ci * DIM + d]);
						T val = clenshaw(SyclRowMatrix<PointScalar,DIM,1>(norm),
							a_fineCoeffs, lo_ns_cap, fineOffset);
						const T cf(
							static_cast<typename T::value_type>(a_tgtCFr[ci]),
							static_cast<typename T::value_type>(a_tgtCFi[ci]));
						result += val * cf;
					}
					//using ScalarT = typename T::value_type;
					//ScalarT* bp = reinterpret_cast<ScalarT*>(&a_result2[tgtId]);
					//sycl::atomic_ref<ScalarT,
					//	sycl::memory_order::relaxed,
					//	sycl::memory_scope::device,
					//	sycl::access::address_space::global_space>
					//	ar(bp[0]), ai(bp[1]);
					//ar.fetch_add(result.real());
					//ai.fetch_add(result.imag());
					a_result2[tgtId] += result;
				});
			});
			Q.wait();
		}

		} // end interpolation data block

		interpolationDataBuffer.reset();
		if(level > 2 && hasParentData) {
			std::swap(interpolationDataBuffer, parentCTPBuffer);
			parentCTPBuffer.reset();
		} else {
			parentCTPBuffer.reset();
		}

		if(level < 1) break;

	    std::cout<<"done with this level"<<std::endl;
        }
        std::cout<<"copying back"<<std::endl;
    	} //end of sycl scope
	//std::cout<<"mult over\n";

		Eigen::Array<T, Eigen::Dynamic, DIMOUT> true_result(result.rows(),result.cols());
        Util::copy_with_inverse_permutation_rowwise<T,DIMOUT>(result, m_octree->targetPermutation(),true_result);
	high_resolution_clock::time_point t12 = high_resolution_clock::now();
    	duration<PointScalar> time_span = duration_cast<duration<PointScalar>>(t12 - t1);
    	std::cout <<"----- mult time ------ "<< time_span.count() << " seconds" << std::endl;
		return true_result;
    }


    void initInterpolationData(size_t level, size_t step, std::unique_ptr<sycl::buffer<T,1>> & buf )
    {
	assert(level<m_octree->levels());


	PointScalar H = m_octree->sideLength()*std::pow(0.5,level+5);
 	std::cout<<"h in init?"<<H<<std::endl;
        auto order = static_cast<Derived *>(this)->orderForBox(H, m_baseOrder,step);

	//make sure no old buffer is around	
	//buf=std::make_unique<sycl::buffer<T,1> > (m_octree->numActiveCones(level,step)*order.prod());
	size_t numCones = m_octree->numActiveCones(level, step);
	if (numCones == 0) {
		buf = std::make_unique<sycl::buffer<T,1>>(sycl::range<1>(1));
		return;
	}
	buf = std::make_unique<sycl::buffer<T,1>>(numCones * order.prod());
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


	struct InfoPerLevel {
		std::vector<size_t>   tgtConeShift; // size: numTargets+1
		std::vector<uint32_t> tgtConeIds;   // fineMemId per entry
		std::vector<float>    tgtNormPnts;  // DIM floats per entry
		std::vector<float>    tgtCF_real;   // precomputed CF per entry
		std::vector<float>    tgtCF_imag;
	};
	std::vector<InfoPerLevel> m_allLevelConeInfo;

	struct CtpGeomLevel {
		std::vector<float> localPnts;
		std::vector<float> TF_real;
		std::vector<float> TF_imag;
		std::unique_ptr<sycl::buffer<float,1>> buf_localPnts;
		std::unique_ptr<sycl::buffer<float,1>> buf_TF_real;
		std::unique_ptr<sycl::buffer<float,1>> buf_TF_imag;
	};
	std::vector<CtpGeomLevel> m_ctpGeom; // one per level
};

#endif