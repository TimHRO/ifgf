#ifndef _OCTREE_HPP_
#define _OCTREE_HPP_

#include "Eigen/src/Core/VectorBlock.h"
#include "Eigen/src/Core/util/XprHelper.h"
#include "config.hpp"


#include <Eigen/Dense>
#include <iterator>
#include <memory>
#include <map>
#include <oneapi/tbb/parallel_for.h>
#include <sys/types.h>
#include <vector>
#include <execution>
#include <iostream>

#include "util.hpp"

#include "boundingbox.hpp"
#include "zorder_less.hpp"
#include "chebinterp.hpp"
#include "cone_domain.hpp"

#include <tbb/queuing_mutex.h>
#include <tbb/spin_mutex.h>
typedef std::pair<size_t, size_t> IndexRange;

// #define  EXACT_INTERP_RANGE

#ifdef BE_FAST
const int N_STEPS=4;
#else
const int N_STEPS=2;
#endif


#include <sycl/sycl.hpp>
#include "sycl_helpers.hpp"

template<typename T, size_t DIM>
class Octree
{

public:
    enum {N_Children = 1 << DIM };
    enum TransformationMode { Decomposition, TwoGrid , Regular};
    typedef Eigen::Array<PointScalar, DIM, Eigen::Dynamic> PointArray;
    typedef Eigen::Vector<PointScalar, DIM> Point;

    struct FieldInfo {
	std::vector<size_t> indices;
	std::vector<size_t> starts;
    };

    class OctreeNode
    { 
    private:
        std::weak_ptr<OctreeNode > m_parent;
        long int m_id;
        std::weak_ptr<OctreeNode> m_children[N_Children];
        //std::vector<std::shared_ptr<const OctreeNode> > m_neighbors;

	std::vector<IndexRange> m_farTargets;
	std::vector<IndexRange> m_nearTargets;

        IndexRange m_pntRange;

	std::array<ConeDomain<DIM>,N_STEPS> m_coneDomain;

        BoundingBox<DIM> m_bbox;


        bool m_isLeaf;
        unsigned int m_level;
    public:

        OctreeNode(std::shared_ptr<OctreeNode> parent, unsigned int level) :
            m_parent(parent),
            m_isLeaf(true),
            m_level(level)
        {

        }

        ~OctreeNode()
        {	    
        }

        void setId(size_t id)
        {
            m_id = id;
        }

        long int id() const
        {
            return m_id;
        }

        IndexRange pntRange() const
        {
            return m_pntRange;
        }


        void setPntRange(const IndexRange &range)
        {
            m_pntRange = range;

        }

        void setChild(size_t idx, std::shared_ptr<OctreeNode> node)
        {
	    if(node!=0) {
		m_children[idx] = node;
		m_isLeaf = false;
	    }
        }

	void setConeDomain(const ConeDomain<DIM>& domain, int substep=0)
	{
	    m_coneDomain[substep]=domain;
	}

	const ConeDomain<DIM>& coneDomain(int substep=0) const
	{
	    return m_coneDomain[substep];
	}


	BoundingBox<DIM> interpolationRange(int substep=0) const
	{
	    return m_coneDomain[substep].domain();
	}

        const std::weak_ptr<const OctreeNode> child(size_t idx) const
        {
            return m_children[idx];
        }

        std::weak_ptr<OctreeNode> child(size_t idx)
        {
            return m_children[idx];
        }

        BoundingBox<DIM> boundingBox() const
        {
            return m_bbox;
        }

        void setBoundingBox(const BoundingBox<DIM> &bbox)
        {
            m_bbox = bbox;
        }

        bool isLeaf() const
        {
            return m_isLeaf;
        }

        bool hasPoints() const
        {            
            return m_pntRange.first != m_pntRange.second;
        }

        unsigned int level() const
        {
            return m_level;
        }


        const std::weak_ptr<const OctreeNode> parent() const
        {
            return m_parent;
        }

	void addNearInteraction(const OctreeNode& target)
	{
	    if(target.hasPoints())
		m_nearTargets.push_back(target.pntRange());
	}
	
	void addFarInteraction(const OctreeNode& target)
	{
	    if(target.hasPoints())
		m_farTargets.push_back(target.pntRange());
	}

	

	const std::vector<IndexRange>& nearTargets() const
	{
	    return m_nearTargets;
	}

	const std::vector<IndexRange>& farTargets() const
	{
	    return m_farTargets;
	}

	
	
        void print(std::string prefix = "") const
        {
            std::cout << prefix;
            std::cout << "----";

            std::cout << m_id << " ";
            std::cout << "(" << m_pntRange.first << " " << m_pntRange.second << ")"; //<<std::endl;
            //std::cout << m_bbox.min().transpose() << "   " << m_bbox.max().transpose() << std::endl;
	    std::cout<< m_isLeaf <<std::endl;

            //std::cout<<"("<<m_pntRange.first<<" "<<m_pntRange.second<<")"<<std::endl;
            //std::cout<<" "<<m_bbox.min().transpose()<<" to "<<m_bbox.max().transpose()<<std::endl;
            /*std::cout<<m_id<<" ";
            if(m_parent)
            std::cout<<m_parent->id();
            std::cout<<std::endl;*/
            if (!m_isLeaf) {
                for (int i = 0; i < N_Children; i++) {
                    const auto& c = m_children[i].lock();
                    if (c) {
                        c->print(prefix + "    ");
                    } else {
                        std::cout << prefix + "    ----x" << std::endl;
                    }
                }
            }
        }

    };

    typedef std::pair<size_t, size_t> BoxIndex; //level + index in level

    Octree(int maxLeafSize):
        m_maxLeafSize(maxLeafSize),
	m_levels(0)
    {

	PointScalar eta=(PointScalar) sqrt((PointScalar) DIM);
	m_isAdmissible= [eta] (const BoundingBox<DIM>& src,const BoundingBox<DIM>& target) { return target.exteriorDistance(src.center()) >= eta* src.sideLength();};

    }

    ~Octree()
    {

    }

    void build(const PointArray &pnts)
    {
        //std::cout << "building a new octree" << pnts.cols()<<std::endl;

        //std::cout << "finding bbox" << std::endl;
        Point min = pnts.col(0);
        Point max = pnts.col(0);

	min=pnts.rowwise().minCoeff();
	max=pnts.rowwise().maxCoeff();
	


        //make the bbox slightly larger to not have to deal with boundary issues
        min.array() -= 0.1 * min.norm();
        max.array() += 0.1 * max.norm();

        BoundingBox<DIM> bbox(min, max);
        //std::cout << "bbox=" << min.transpose() << "\t" << max.transpose() << std::endl;

        //sort the points by their morton order for better locality later on
        //std::cout << "sorting..." << std::endl;
        m_permutation = Util::sort_with_permutation( pnts.colwise().begin(), pnts.colwise().end(), zorder_knn::Less<Point, DIM>(bbox));
	m_pnts.resize(DIM, pnts.cols());
        Util::copy_with_permutation_colwise<PointScalar,DIM>(pnts, m_permutation,m_pnts);

	m_diameter=bbox.diagonal().norm();
	m_sideLength=bbox.sideLength();

	m_levels = 0;
	m_depth = -1;
        m_root = buildOctreeNode(0, std::make_pair(0, m_pnts.cols()), bbox);
	m_depth=m_levels;


        //std::cout << "building the nodes up to level"<<m_depth << std::endl;
  

	
    }

    void printInteractionList(std::shared_ptr<OctreeNode>  src) {
	std::cout<<"interactions for "<<src->id()<<" "<<src->level()<<std::endl;
	for( auto near : src->nearTargets()) {
	    std::cout<<near.first<<" "<<near.second<<std::endl;
	}
	std::cout<<"far"<<std::endl;

	for( auto far : src->farTargets()) {
	    std::cout<<far.first<<" "<<far.second<<std::endl;
	}	
		    
    }

    void buildInteractionList(const Octree&  target_tree)
    {
	buildInteractionList(m_root,target_tree.m_root);

	/*printInteractionList(m_root);
	for(int i=0;i<N_Children;i++){
	    printInteractionList(m_root->child(i));
	    }*/

    }

    void buildInteractionList(std::shared_ptr<OctreeNode>  src,std::shared_ptr<const OctreeNode>  target)
    {
	if(!src || !target || !src->hasPoints() || ! target->hasPoints()) {
	    return;
	}


	//Ideally, this interaction is admissible, so we can interpolate it at this level
	if(m_isAdmissible(src->boundingBox(),target->boundingBox())) {
	    src->addFarInteraction(*target);
	    return;
	}

	
	//If both of them are leaves, we can't proceed by recursion. So let's just stop and
	//do it the hard way	
	if(src->isLeaf() && target->isLeaf()) {	    
	    src->addNearInteraction(*target);
	    return;
	}

	//If either src or target has children recurse down
	if(src->isLeaf() && !target->isLeaf()) {
	    for (int j = 0; j < N_Children; j++) {
		buildInteractionList(src,target->child(j).lock());
	    }
	    return;
	}

	if(target->isLeaf() && !src->isLeaf()) {
	    for (int j = 0; j < N_Children; j++) {
		buildInteractionList(src->child(j).lock(),target);
	    }
	    return;
	}

	//if both src and target have children, we recurse down the one with the larger bbox
	if(src->boundingBox().sideLength() > target->boundingBox().sideLength()) {
	    for (int j = 0; j < N_Children; j++) {
		buildInteractionList(src->child(j).lock(),target);
	    }
	    return;
	}else {
	    for (int j = 0; j < N_Children; j++) {
		buildInteractionList(src,target->child(j).lock());
	    }
	    return;
	}

	std::cout<<"this does not happen!"<<std::endl;
	src->print();
	target->print();
    }

    void calculateInterpolationRange(  std::function<Eigen::Vector<int,DIM>(PointScalar,int)> order_for_H,
				       std::function<Eigen::Vector<size_t,DIM>(PointScalar,int )> N_for_H,
				       std::function<PointScalar(PointScalar)> smin_for_H,
				       const Octree& target)
    {

	m_farFieldBoxes.resize(levels());
	m_nearFieldBoxes.resize(levels());

	m_coneMaps.resize(levels());

	m_numLeafCones.resize(levels());
	
	BoundingBox<DIM> global_box;
	//global_box.min().fill(10);
	//global_box.max().fill(0);

	TransformationMode mode=TwoGrid;

	const auto & target_points=target.points();
	tbb::spin_mutex activeConeMutex;
	tbb::spin_mutex ptCMutex;

	Eigen::Vector<size_t,DIM> oldN;
	oldN.fill(0);

	Eigen::Vector<size_t,DIM> oldPN;
	oldN.fill(0);
	double est_H=m_root->boundingBox().sideLength(); //size of the whole domain.


	m_activeHoCones.resize(levels());

	
	for (size_t level=0;level<levels();level++) {
	    typedef ankerl::unordered_dense::map<size_t,std::vector<size_t> > ConeToActivityMap;
	    std::array<ConeToActivityMap, 1 << DIM > parentToChildActiveSets;

	    std::cout<<"l= "<<level<<" "<<est_H<<std::endl;
	    //update all nodes in this level

	    std::array<std::vector<ConeRef>,N_STEPS> activeCones;


            //check if we are in the cutoff regime (only relevant for modified Helmholtz with large real part)
            {
                auto order=order_for_H(est_H,0);
                if(smin_for_H(est_H)>=(sqrt(DIM)/DIM-1e-10)) {
                    std::cout<<"cutting of at level"<<level<<" "<<est_H<<" order="<<order<<std::endl;
                    tbb::parallel_for(tbb::blocked_range<size_t>(0,numBoxes(level)), [&](tbb::blocked_range<size_t> r) {
                    for(size_t n=r.begin();n<r.end();++n) {
                        std::shared_ptr<OctreeNode> node=m_nodes[level][n];

                        BoundingBox<DIM> domain;
                        domain.min()=Eigen::Vector<PointScalar,DIM>::Zero();
                        domain.max()=Eigen::Vector<PointScalar,DIM>::Zero();
                        ConeDomain<DIM> d0(Eigen::Vector<size_t,DIM>::Ones(),domain);
                        node->setConeDomain(d0,0);
                        node->setConeDomain(d0,1);
                     }});
                     m_activeCones.push_back(activeCones);
                     m_leafCones.push_back( std::vector<ConeRef>());
                     m_numLeafCones[level]=0;


                     //compute the farFieldBoxes
                     FieldInfo info;
                     info.indices.resize(0);
                     info.starts.resize(target.points().size()+1);
                     std::fill(info.starts.begin(),info.starts.end(),0);
                     m_farFieldBoxes[level]=info;//computeFieldInfo(level,target, true);
                     m_nearFieldBoxes[level]=computeFieldInfo(level,target, false); //we still do the nearfield regularly


 
                     est_H/=2.;
                     continue;
                }
            } 

	    //make some heuristic about the number of cones
	    for(int step=0;step<N_STEPS;step++) {
		auto N=N_for_H(est_H,step);
		const size_t na=numBoxes(level)*std::pow((double) N.prod(),2.0/3.0);
		//std::cout<<"reserving "<<na<<std::endl;	
		activeCones[step].reserve(na);
	    }

	    //We use a coarse grid with high order and a fine grid of lower order
	    BoundingBox<DIM> interp_box;
	    PointScalar smax=sqrt(DIM)/DIM;	    
	    PointScalar smin= smin_for_H(est_H);

	    interp_box.min()(0)=smin;
	    interp_box.max()(0)=smax;
	    
			
	    if constexpr(DIM==2) {	    
		interp_box.min()(1)=-M_PI;
		interp_box.max()(1)=M_PI;
	    }else{
		interp_box.min()(1)=0;
		interp_box.max()(1)=M_PI;
			    
		interp_box.min()(2)=-M_PI;
		interp_box.max()(2)=M_PI;
	    }

            BoundingBox<DIM> p_interp_box=interp_box;
            p_interp_box.min()(0)=smin_for_H(2*est_H);


	    const PointScalar pH=bbox(level > 0 ? (level-1): 0,0).sideLength();
	    const ConeDomain<DIM> p_hoGrid(N_for_H(pH,1), p_interp_box );

	    const size_t numBoxesOnLevel=numBoxes(level);
	    const size_t numParentCones=p_hoGrid.n_elements();
	    
	    const bool should_be_cached= false; //level > 2 && ((1 << DIM) * numParentCones < 1024 ); //keep the cache small, for larger ones it wont really pay off

	    std::cout<<"caching level "<<level<<" "<<should_be_cached<<" "<<N_for_H(pH,1)<<std::endl;


	    tbb::enumerable_thread_specific<Eigen::Array<PointScalar, DIM, Eigen::Dynamic> > local_pnts;
	    tbb::enumerable_thread_specific<Eigen::Array<PointScalar, DIM, Eigen::Dynamic> > local_interp_pnts;	    
	    const auto calculateParentToChildActiveSet=[&](int cube_corner, size_t el) {   
		const PointScalar pH=bbox(level-1,0).sideLength();
		const PointScalar HH=bbox(level,0).sideLength();

		assert(abs(HH-est_H)<1e-5);


		auto HoChebNodes = ChebychevInterpolation::chebnodesNdd<PointScalar, DIM>(order_for_H(pH,1));
		const ConeDomain<DIM> p_hoGrid(N_for_H(pH,1), interp_box );
		const ConeDomain<DIM> loGrid(N_for_H(HH,0), interp_box );
		

		local_pnts.local().resize(DIM, HoChebNodes.cols());
		local_interp_pnts.local().resize(DIM, HoChebNodes.cols());
		
			      
		Eigen::Vector<double,DIM> d;
		for(int j=0;j<DIM;j++) {
		    bool flag=cube_corner & (1<<j);
		    d[j]= flag ? 0.5: -0.5;		    
		}
		
				
		IndexSet is_active;
		is_active.reserve(1 << DIM);
		local_pnts.local()=Util::interpToCart<DIM>(p_hoGrid.transform(el,HoChebNodes).array(),Eigen::Vector3d::Zero(),pH);
		Util::cartToInterp2<DIM>(local_pnts.local().array(),d*HH,HH,local_interp_pnts.local().array());  //xc-pxc
		for (size_t i=0;i<HoChebNodes.cols();i++) {
		    auto coneId=loGrid.elementForPoint(local_interp_pnts.local().col(i));
		    if(coneId<SIZE_MAX) {
			auto p=is_active.emplace(coneId);
			//numCones+=(p.second ? 1 : 0);
			
		    }
		}
		    
		return is_active.values();
		//std::sort(parentToChildActiveSets[cube_corner][el].begin(),parentToChildActiveSets[cube_corner][el].begin());
		
	    };
		
 

	    

	    
	    std::vector<ConeRef> leafCones;
	    m_coneMaps[level].resize(numBoxes(level));
	    tbb::parallel_for(tbb::blocked_range<size_t>(0,numBoxes(level)), [&](tbb::blocked_range<size_t> r) {
            for(size_t n=r.begin();n<r.end();++n) {
		std::shared_ptr<OctreeNode> node=m_nodes[level][n];
		BoundingBox<DIM> box;

		const Point xc=node->boundingBox().center();
		
		const PointScalar H=node->boundingBox().sideLength();

		//std::cout<<"est_H"<<est_H<<" vs "<<H<<std::endl;
		assert(abs(est_H -H)<1e-5);
		const std::vector<IndexRange> farTargets=node->farTargets();

		//just use a default value for the boxes

		PointScalar smax=sqrt(DIM)/DIM;
		
                
		std::shared_ptr<const OctreeNode> parent=node->parent().lock();
		BoundingBox<DIM> pBox;
		if(parent && parentHasFarTargets(node))
		{	    
		    pBox=parent->interpolationRange();
		}

		
		box=interp_box;
		ConeDomain<DIM> domain(N_for_H(H,0),box);

		auto hoN=N_for_H(H,1);

		ConeDomain<DIM> coarseDomain(hoN,box);

		    
		assert(H>0);

		if(!box.isNull())
		    global_box.extend(box);
		
		//now we need to do the whole thing again to figure out which cones are active...
		// 0 = fine grid low order (used for evaluating FF and propagating upwards
		// 1 = coarse grid high order (used as target for interplating leaves and for propagation from below)
		std::array<IndexSet,N_STEPS> is_cone_active;
		auto N=N_for_H(est_H,0);
		//is_cone_active[0].reserve(std::pow((double) N.prod(),(DIM-1.0)/DIM));
		
		std::array<size_t,N_STEPS> numActiveCones;
		std::fill(numActiveCones.begin(),numActiveCones.end(),0);



		//now add all the parents targets
		if(!pBox.isNull())
		{
		    const Point pxc=parent->boundingBox().center();
                    const PointScalar pH=parent->boundingBox().sideLength();

                    const ConeDomain<DIM>& p_grid=parent->coneDomain(1);

		    if(parentHasFarTargets(level,n))
                    {
			int fingerprint=Util::calculateFingerprint<DIM>(xc,pxc,H);
			
		       
			//We use a coarse grid with high order and a fine grid of lower order
			auto HoChebNodes = ChebychevInterpolation::chebnodesNdd<PointScalar, DIM>(order_for_H(pH,1));
			const ConeDomain<DIM>& p_hoGrid=parent->coneDomain(1);

			PointArray pnts(DIM,HoChebNodes.cols());
			PointArray interp_pnts(DIM,HoChebNodes.cols());
			IndexSet activity;
			for(size_t el : p_hoGrid.activeCones() ) {
			    bool is_cached=should_be_cached;
			    {
				tbb::spin_mutex::scoped_lock lock(ptCMutex);
				m_activeHoCones[level].emplace(el); //mark that this is a type of HO element that exists.
			    }


			    if(should_be_cached)
			    {				
				is_cached=(parentToChildActiveSets[fingerprint].count(el)>0);
			    }

			    const std::vector<size_t>& pElSet= (!is_cached ? calculateParentToChildActiveSet(fingerprint, el) : parentToChildActiveSets[fingerprint][el]);
			    is_cone_active[0].reserve(pElSet.size());
			    for ( size_t coneId : pElSet )
			    {
				
				auto p=is_cone_active[0].emplace(coneId);
				numActiveCones[0]+=(p.second ? 1 : 0);
			    }

			    
			    if(!is_cached && should_be_cached) {
				tbb::spin_mutex::scoped_lock lock(ptCMutex);
				if(parentToChildActiveSets[fingerprint].count(el)>0) {
				    parentToChildActiveSets[fingerprint].emplace(el,std::move(pElSet));
				}
			    }

						     
			
			}		    

			
                    }

		}

		//now add the actual targets
		PointArray s(DIM,32);
		for(const IndexRange& iR : farTargets)
		{
		    s.resize(DIM,std::max((size_t) s.cols(),iR.second-iR.first));
		    Util::cartToInterp2<DIM>(target_points.middleCols(iR.first,iR.second-iR.first),xc,H,s.leftCols(iR.second-iR.first).array());			  
		    for(int i=0;i<iR.second-iR.first;i++)
		    {
			//const auto s=Util::cartToInterp<DIM>(target_points.col(i),xc,H);			  
			auto coneId=domain.elementForPoint(s.col(i));
			if(coneId<SIZE_MAX) {
			    auto p=is_cone_active[0].emplace(coneId);
                            numActiveCones[0]+=(p.second ? 1 : 0);
			}
		    }
		}






		//We now activate all of the elements in the coarse(high order grid) that are needed to
		//create the fine grid later on.
		{
		    auto factor=9;//(N_for_H(est_H,0).array()/N_for_H(est_H,1).array()).prod();
		    is_cone_active[1].reserve(is_cone_active.size()/factor);
		    for( size_t el : is_cone_active[0])
		    {
			const auto pnt=domain.transform(el, Eigen::Vector<PointScalar,DIM>::Zero());
			auto coneId=coarseDomain.elementForPoint(pnt);
			if(coneId<SIZE_MAX) {
                            auto p=is_cone_active[1].emplace(coneId);
                            numActiveCones[1]+=(p.second ? 1 : 0);
			}
		    }
		}
		

		

		for (int step=0;step<N_STEPS;step++ ) {
		    std::vector<size_t> local_active_cones;
		    local_active_cones.reserve(numActiveCones[step]);
		    IndexMap cone_map;
		    cone_map.reserve(numActiveCones[step]);
		    /*{
			tbb::spin_mutex::scoped_lock lock(activeConeMutex);
			const size_t  na=activeCones[step].size()+numActiveCones[step];
			activeCones[step].reserve(na);
			}*/
		    
		    //std::cout<<"NA="<<na<<std::endl;

		    //local_active_cones.reserve(domain.n_elements());
		     for( size_t i : is_cone_active[step])
		     {			 
			 tbb::spin_mutex::scoped_lock lock(activeConeMutex);
			 
			 ConeRef cone(level, i, local_active_cones.size(), n,activeCones[step].size());
			 activeCones[step].push_back(cone);
			 

			 int leafStep=0;
			 if(mode!=Regular) {
			     leafStep=1;
			 }
			     
			 if(node->isLeaf() && step==leafStep) {
			     leafCones.push_back(cone);
			 }

		     

			 local_active_cones.push_back(i);
			 cone_map[i]=local_active_cones.size()-1;
			     

			 {
			     if(step==0) {
				 m_coneMaps[level][cone.boxId()][i]=cone.globalId();
			     }
			 }
			     
			 
		     }
		     

		     //local_active_cones.trim();
		     ConeDomain d0=coarseDomain;
		     if(step==0 && mode!=Regular) {
			 d0=domain;
		     }
		     d0.setActiveCones(local_active_cones);		
		     d0.setConeMap(cone_map);		     
		     
		     node->setConeDomain(d0,step);		     
		}
	    }});

	    //Free any unused memory
	    for(int step=0;step<N_STEPS;step++) {
		activeCones[step].shrink_to_fit();
		leafCones.shrink_to_fit();
	    }
	    {
		//tbb::queuing_mutex::scoped_lock lock(activeConeMutex);
		std::cout<<"level= "<<level<<" active cones"<<activeCones[0].size()<<" full: "<<numBoxes(level)*coneDomain(level,0,0).n_elements()<<std::endl;
		std::cout<<"level= "<<level<<" active cones"<<activeCones[1].size()<<" full: "<<numBoxes(level)*coneDomain(level,0,1).n_elements()<<std::endl;		
		std::cout<<"level= "<<level<<" leafCones "<<leafCones.size()<<std::endl;
	    }
	    m_activeCones.push_back(activeCones);
	    m_leafCones.push_back(leafCones);
	    m_numLeafCones[level]=leafCones.size();


	    //compute the farFieldBoxes
	    m_farFieldBoxes[level]=computeFieldInfo(level,target, true);
	    m_nearFieldBoxes[level]=computeFieldInfo(level,target, false);			

	    est_H/=2;
	}


	m_activeCones.shrink_to_fit();


	buildChildToParentData(order_for_H,N_for_H,smin_for_H);
	//std::cout<<"interp_domain:" <<global_box<<std::endl;

    }


    FieldInfo computeFieldInfo(int level, const Octree& target, bool isFarField=true) const
    {
	FieldInfo info;

	//we start out by computing how much storage we might need
	size_t cnt=0;
	Eigen::Vector<size_t, Eigen::Dynamic> nBoxPerTarget(target.points().size());
	nBoxPerTarget.fill(0);

	for(size_t n=0;n<numBoxes(level);++n) {
	    std::shared_ptr<OctreeNode> node=m_nodes[level][n];
	    const std::vector<IndexRange>& targets=isFarField ? node->farTargets() : node->nearTargets();		    
	    for( const auto tRange : targets) {
		cnt+=tRange.second-tRange.first;
		for(size_t trg=tRange.first;trg<tRange.second;++trg) {
		    nBoxPerTarget[trg]++;
		}
	    }
	}
	if(cnt>0)
	{
	    //now we populate	
	    info.indices.resize(cnt);
	    info.starts.resize(target.points().size()+1);

	    //now fill the starts vector
	    info.starts[0]=0;
	    for(size_t trg=1;trg<info.starts.size();trg++) {
		info.starts[trg]=info.starts[trg-1]+nBoxPerTarget[trg-1];

	    }

	    assert(info.starts[target.points().size()]==cnt);

	    nBoxPerTarget.fill(0);
			    
	    for(size_t n=0;n<numBoxes(level);++n) {		
		std::shared_ptr<OctreeNode> node=m_nodes[level][n];
		const std::vector<IndexRange>& targets=isFarField ? node->farTargets() : node->nearTargets();		    
		for( const auto tRange : targets) {		
		    for(size_t trg=tRange.first;trg<tRange.second;++trg) {
			info.indices[info.starts[trg]+nBoxPerTarget[trg]]=n;
			nBoxPerTarget[trg]++;
		    }
		}
	    }


	}else{
	    info.indices.resize(0);
	    info.starts.resize(target.points().size()+1);
	    std::fill(info.starts.begin(),info.starts.end(),0);
		    
	}
	return info;
		
    }


	
    
    inline PointScalar diameter () const
    {
	return m_diameter;
    }

    inline PointScalar sideLength() const
    {
	return m_sideLength;
    }

    unsigned int levels() const
    {
        return m_levels;
    }

    long int child(size_t level, size_t id, size_t childIndex) const
    {
	const auto  child=m_nodes[level][id]->child(childIndex);
	if(child) {
	    return m_nodes[level][id]->child(childIndex)->id();
	}else{
	    return -1;
	}
    }

    const std::vector<size_t> activeChildren( size_t level,size_t id) const
    {
        const auto node=m_nodes[level][id];
        std::vector<size_t> aC;
        aC.reserve(N_Children);
        for (int i=0;i<N_Children;i++) {
            const auto child=node->child(i);
            if(child->hasPoints()) {
                aC.push_back(child->id());
            }
        }
        return aC;
    }

    unsigned int numBoxes(unsigned int level) const
    {
        return m_numBoxes[level];
    }

    const auto points(IndexRange index) const
    {
        return m_pnts.middleCols(index.first, index.second - index.first);
    }

    const auto point(size_t id) const
    {
        return m_pnts.col(id);
    }


    const std::vector<size_t> permutation() const
    {
        return m_permutation;
    }




    const IndexRange points(unsigned int level, size_t i) const
    {
	assert(level< m_nodes.size());
	assert(i < m_nodes[level].size());
	std::shared_ptr<OctreeNode> node = m_nodes[level][i];

        return node->pntRange();
    }

    inline const std::vector<IndexRange> nearTargets(unsigned int level, size_t i) const
    {
        std::shared_ptr<OctreeNode> node = m_nodes[level][i];
        return node->nearTargets();
    }

    inline const std::vector<IndexRange> farTargets(unsigned int level, size_t i) const
    {
        std::shared_ptr<OctreeNode> node = m_nodes[level][i];
        return node->farTargets();
    }


    bool hasFarTargetsIncludingAncestors(unsigned int level, size_t i) const
    {
	const std::shared_ptr<const OctreeNode>& node = m_nodes[level][i];
	if(node) {
	    return node->farTargets().size()>0 || parentHasFarTargets(node);
	}
	return false;
    }

    bool parentHasFarTargets(const std::shared_ptr<const OctreeNode>& node) const
    {
	const std::shared_ptr<const OctreeNode>& parent = node->parent().lock();	
	if(parent) {
	    return parent->farTargets().size()>0 || parentHasFarTargets(parent);
	}else{
	    return false;
	}
    }

    bool parentHasFarTargets(unsigned int level, size_t i) const
    {
	const std::shared_ptr<const OctreeNode>& node = m_nodes[level][i];
	if(node) {
	    return parentHasFarTargets(node);
	}
	return false;
    }

    const BoundingBox<DIM> bbox(unsigned int level, size_t i) const
    {
        return m_nodes[level][i]->boundingBox();
    }


    const BoundingBox<DIM> interpolationRange(unsigned int level, size_t i) const
    {
        return m_nodes[level][i]->interpolationRange();
    }

    const ConeDomain<DIM> coneDomain(unsigned int level, size_t i) const
    {
        return m_nodes[level][i]->coneDomain();
    }

    
    const ConeDomain<DIM> coneDomain(unsigned int level, size_t i,size_t substep) const
    {
        return m_nodes[level][i]->coneDomain(substep);
    }



    const std::vector<size_t> childBoxes(unsigned int level, size_t i) const
    {
	auto parent=m_nodes[level][i];
	std::vector<size_t> children;
	for(int i=0;i<N_Children;i++) {
	    const auto child=parent->child(i).lock();
	    if(child) {
		children.push_back(child->id());
	    }	    
	}
	return children;
    }
    
    const size_t parentId(unsigned int level, size_t i) const
    {
	auto parent=m_nodes[level][i]->parent().lock();
	size_t id = parent->id();
	assert(m_nodes[level-1][id]==parent);
        return id;
    }

    const bool hasPoints(unsigned int level, size_t i) const
    {
        const auto range = points(level, i);
        return range.first != range.second;
    }

    const bool isLeaf(unsigned int level, size_t i) const
    {
	return m_nodes[level][i]->isLeaf();        
    }

    size_t numActiveCones(size_t level,size_t step=0) const {
	return m_activeCones[level][step].size();
    }

    ConeRef activeCone(size_t level,size_t id,size_t step=0) const
    {
	return m_activeCones[level][step][id];
    }


    size_t numLeafCones(size_t level) const
    {
	return m_numLeafCones[level];
    }

    ConeRef leafCone(size_t level, size_t num) const
    {
	return m_leafCones[level][num];
    }
   


    const auto farfieldBoxes(size_t level, size_t targetPoint) const
    {
	const auto& ffB=m_farFieldBoxes[level];

	//	assert(targetPoint+1<ffB.starts.size());
	const size_t start=ffB.starts[targetPoint];
	const size_t end=ffB.starts[targetPoint+1];


	
	return ffB.indices.segment(start, end-start);
    }

    const auto nearFieldBoxes(size_t level, size_t targetPoint) const
    {
	const auto& nfB=m_nearFieldBoxes[level];

	//	assert(targetPoint+1<ffB.starts.size());
	const size_t start=nfB.starts[targetPoint];
	const size_t end=nfB.starts[targetPoint+1];


	
	return nfB.indices.segment(start, end-start);
    }


    size_t numPoints() const {
	return m_pnts.cols();
    }

    void sanitize()
    {
	Eigen::VectorXi leaf_indices(m_pnts.cols());	
	leaf_indices.fill(0);
        for (int level = 0; level < m_levels; level++) {
            Eigen::VectorXi indices(m_pnts.cols());
	    indices=leaf_indices;            
            for (int i = 0; i < m_nodes[level].size(); i++) {
                size_t a = m_nodes[level][i]->pntRange().first;
                size_t b = m_nodes[level][i]->pntRange().second;
		
                for (int l = a; l < b; l++) {
                    indices[l] += 1;
		    if(m_nodes[level][i]->isLeaf()) {
			leaf_indices[l]+=1;
		    }
		    
                }
            }

            for (int i = 0; i < indices.size(); i++) {
                if (indices[i] != 1) {
                    std::cout << "wrong" << indices[i] << " " << i << " level" << level << std::endl;
                    std::cout << " at " << m_pnts.col(i) << std::endl;
                }
                assert(indices[i] == 1);
            }
        }
    }


    void freeData() {
	m_root.reset();
	m_nodes.clear();
	m_nodes.shrink_to_fit();

	/*for(int i=0;i<levels();i++){
	    m_activeCones[i][0].clear();
	    m_activeCones[i][0].shrink_to_fit();
	    }*/
	//m_activeCones.clear();
	//m_activeCones.shrink_to_fit();
	//m_leafCones.clear();
	//m_leafCones.shrink_to_fit();
	//m_farFieldBoxes.clear();
	//m_farFieldBoxes.shrink_to_fit();
	//m_nearFieldBoxes.clear();
	//m_nearFieldBoxes.shrink_to_fit();

	//m_coneMaps.clear();
	//m_coneMaps.shrink_to_fit();
	
    }

    const auto& points() const {
	return m_pnts;
    }


    const std::vector<IndexMap>& coneMaps(size_t level) const
    {
	return m_coneMaps[level];
    }


private:
    std::shared_ptr<OctreeNode> buildOctreeNode(std::shared_ptr<OctreeNode > parent, const IndexRange &pnt_range, const BoundingBox<DIM> &bbox, unsigned int level = 0)
    {
	
	if(pnt_range.first==pnt_range.second) //we only keep non-empty nodes around
	{
	    return 0;
	}
	
        if (level >= m_levels) {
            m_levels++;
            m_nodes.push_back(std::vector<std::shared_ptr<OctreeNode> >());
            m_numBoxes.push_back(0);
        }

	auto node = std::make_shared<OctreeNode >(parent, level);

	
	node->setPntRange(pnt_range);
	    
	node->setBoundingBox(bbox);

	m_numBoxes[level] += 1;
	node->setId(m_nodes[level].size());
	
	m_nodes[level].push_back(node);
	

	//check how big we are. If the number of points
	//is small enough we create a new leaf.
	const size_t N= pnt_range.second-pnt_range.first;

	
	if( N<= m_maxLeafSize ) {
	    return node;
	}

        size_t pnt_idx = pnt_range.first;

        for (int j = 0; j < N_Children; j++) {
            Point min;
            Point max;

            //std::cout<<"finding bbox from parent"<<std::endl;
            min = bbox.min();
            max = bbox.max();

            Eigen::Vector<PointScalar, DIM> size = 0.5 * bbox.diagonal();

            //find the quadrant that the next src idx belongs to

            auto tuple_idx = compute_tuple_idx(j);
            min.array() += size.array() * tuple_idx.array();
            max = min + size;

            BoundingBox<DIM> child_bbox(min, max);

            //std::cout<<"building child "<<j<<" at"<<child_bbox.min().transpose()<<" "<<child_bbox.max().transpose()<<std::endl;

            size_t end_pnt = pnt_idx;

            while (end_pnt < pnt_range.second && child_bbox.contains(m_pnts.col(end_pnt).matrix())) {
                ++end_pnt;
            }


            //assert(src_idx>=src_range.first && end_src <=src_range.second);
            //std::cout<<"src:"<<src_idx<<end_src<<std::endl;
            const IndexRange pnts(pnt_idx, end_pnt);
            node->setChild(j, buildOctreeNode(node, pnts, child_bbox, level + 1));

            pnt_idx = end_pnt;
        }

        return node;

    }

    inline Eigen::Vector<PointScalar, DIM> compute_tuple_idx(size_t idx) const
    {
        Eigen::Vector<PointScalar, DIM> tuple;
        tuple.fill(0);

        for (size_t j = 0; j < DIM; j++) {
            tuple[j] = idx & 1;

            idx = idx >> 1;

        }

        return tuple;

    }

    inline size_t numChildBoxes(size_t level) const
    {
	return level+1 < m_levels ? numBoxes(level)*pow(2,DIM) : 0 ; //The finest level does not have children
    }




    void buildChildToParentData(std::function<Eigen::Vector<int,DIM>(PointScalar,int)> order_for_H,
				std::function<Eigen::Vector<size_t,DIM>(PointScalar,int )> N_for_H,
				std::function<PointScalar(PointScalar)> smin_for_H)
    {
	std::cout<<"building CtP data"<<std::endl;
	auto global_control = tbb::global_control( tbb::global_control::max_allowed_parallelism,      1);
	double H=m_sideLength;

	m_childToParent.resize(levels());


	
	for(int level=1;level<levels();level++) {
	    H/=2.0;
	    std::cout<<"CtP level"<<level<<std::endl;

	    PointScalar smax=sqrt(DIM)/DIM;	    
	    PointScalar smin= smin_for_H(H);

	    BoundingBox<DIM> interp_box;
	    interp_box.min()(0)=smin;
	    interp_box.max()(0)=smax;
	    
	    if constexpr(DIM==2) {	    
		interp_box.min()(1)=-M_PI;
		interp_box.max()(1)=M_PI;
	    }else{
		interp_box.min()(1)=0;
		interp_box.max()(1)=M_PI;
		
		interp_box.min()(2)=-M_PI;
		interp_box.max()(2)=M_PI;
	    }
	    


	    



	    const auto pH=2.0*H;
	    Eigen::Vector<size_t, DIM> numParentEls=N_for_H(pH,1); //parent is high order
	    Eigen::Vector<size_t, DIM> numChildEls=N_for_H(H,0); //child is low order

	    auto hoChebNodes = ChebychevInterpolation::chebnodesNdd<PointScalar, DIM>(order_for_H(pH,1));
	    size_t parentShift=0;

	    ConeDomain<DIM> parentDomain(numParentEls, interp_box);
	    const Point parent_center=Point::Zero();

	    
	    size_t  totalNumParentEls=numParentEls.prod();

	    //std::vector<ChildToParentData> dataList;
	    
	    //For each parent cone, we precompute possible CtF transformations

	    tbb::spin_mutex ptCMutex;


	    ChildToParentData ex;
	    ex.parentElementShifts.resize(2*totalNumParentEls);
	    ex.directionShifts.resize(1);
	    ex.directionShifts[0]=0;		
	    std::fill(ex.parentElementShifts.begin(),ex.parentElementShifts.end(),SIZE_MAX);

	    tbb::enumerable_thread_specific<ChildToParentData> partial_data(ex);


	    std::cout<<"found "<<m_activeHoCones[level].values().size()<<" vs "<<totalNumParentEls<<" pH= "<<pH<<" "<<N_for_H(pH,1)<<std::endl;
	    
			
	    tbb::parallel_for(tbb::blocked_range<size_t>(0,m_activeHoCones[level].values().size()), [&](tbb::blocked_range<size_t> r) {
	
		PointArray cart_pnts(DIM,hoChebNodes.cols());
		PointArray tmp(DIM,hoChebNodes.cols());
		PointArray transformed(DIM,hoChebNodes.cols());
	    
		std::vector<size_t> coneIds(hoChebNodes.cols());
		PointArray transformedPnts(DIM,hoChebNodes.cols());
		Point tmp2;

		auto& data=partial_data.local();


		size_t dirIdx=data.directionShifts.size()-1;
		size_t globalPntIdx=data.points.cols();
		size_t totalPntCount=data.points.cols();


		for(size_t iter=r.begin();iter<r.end();iter++) {
		    size_t pEl=m_activeHoCones[level].values()[iter];

		    data.directionShifts.resize(data.directionShifts.size()+N_Children);

		    data.parentElementShifts[2*pEl]=dirIdx; //store where the direction for the current element begins
		    data.parentElementShifts[2*pEl+1]=0;//cartPntIdx;

		    data.points.conservativeResize(DIM,std::max((size_t) data.points.cols(),globalPntIdx+N_Children*hoChebNodes.cols()));
		    data.cart_pnts.conservativeResize(DIM,std::max((size_t) data.cart_pnts.cols(),globalPntIdx+N_Children*hoChebNodes.cols()));
		    data.point_ids.reserve(data.points.cols());
		    data.directionForPoint.reserve(data.points.cols());

		    tmp=parentDomain.transform(pEl,hoChebNodes);
		    cart_pnts=Util::interpToCart<DIM>(tmp.array(),parent_center,pH);
		
		    size_t parentElShift=0;
		    
		    //Go through the different cases of parent/child boxes
		    for(int parentId=0;parentId<N_Children;parentId++) { 
			const int cube_corner=parentId;
			Eigen::Vector<double,DIM> d;
			for(int j=0;j<DIM;j++) {
			    bool flag=cube_corner & (1<<j);
			    d[j]= flag ? 0.5: -0.5;		    
			}

	       		
			ConeDomain<DIM> childDomain(numChildEls, interp_box);

			Point center=parent_center+d*H;

			assert(Util::calculateFingerprint<DIM>(center,parent_center,H)==parentId);

			size_t pnt_id=0;
			coneIds.resize(hoChebNodes.cols());
       
			//Transfer to the interpolation domain relative to the child box
			Util::cartToInterp2<DIM>(cart_pnts.array(),center,H,transformed);

			for(size_t l=0;l<transformed.cols();l++)  {
			    const size_t el=childDomain.elementForPoint(transformed.col(l));			    
			    //std::cout<<"el="<<el<<std::endl;
			    assert(el!=SIZE_MAX); //shoudln't happen for Helmholtz
			    if(el<SIZE_MAX) {
				coneIds[pnt_id]=el;
				transformedPnts.col(pnt_id)=childDomain.transformBackwards(el,transformed.col(l));
				pnt_id++;
			    }	       
				
			}
			size_t pointsInDirection=pnt_id;
			size_t basePoint=totalPntCount;
			totalPntCount+=pointsInDirection;
			//std::cout<<"sizes="<<pointsInDirection<<" vs "<<coneIds.size()<<" vs "<<hoChebNodes.cols()<<std::endl;
			
			coneIds.resize(pointsInDirection);


		
			auto permutation=Util::sort_with_permutation(coneIds.begin(),coneIds.end(), std::less<size_t>());
			
			Util::copy_with_permutation_colwise<PointScalar,DIM>(transformedPnts.array().leftCols(pointsInDirection),
									     permutation,tmp.leftCols(pointsInDirection));
			data.points.middleCols(basePoint,pointsInDirection)=tmp;
			Util::copy_with_permutation_colwise<PointScalar,DIM>(cart_pnts.array().leftCols(pointsInDirection),
									     permutation,data.cart_pnts.middleCols(basePoint,pointsInDirection));


			
		
			size_t idx=0;
			size_t cone=0;
			while(idx<pointsInDirection) {
			    size_t real_id=permutation[idx];
			    size_t oldCone=coneIds[real_id];			    

			    //seek until the child cone changes
			    for(;idx<pointsInDirection;idx++) {
				if(coneIds[permutation[idx]]!=oldCone) {
				    break;
				}
				data.point_ids.push_back(permutation[idx]);
				data.directionForPoint.push_back(parentId);
			    }

			    //std::cout<<"pusing"<<oldCone<<" "<<basePoint<<" "<<idx<<std::endl;
			    data.fineElementInfo.push_back(oldCone);
			    data.fineElementInfo.push_back(basePoint+idx);



			    cone++;
			}
			assert(data.point_ids.size()==totalPntCount);
		    
			    
			globalPntIdx+=pointsInDirection;
			
			data.directionShifts[dirIdx+1]=data.directionShifts[dirIdx]+cone;
			dirIdx++;

		    }
		    //dirIdx+=N_Children;
		    data.points.resize(DIM,totalPntCount); 
		    data.fineElementInfo.shrink_to_fit();


		    data.is_valid=31415;
		    //dataList.push_back(data);		    		    
		}
	    });


	    std::cout<<"adding partial data ";
	    for(auto& data : partial_data) {
		std::cout<<m_childToParent[level].size()<<"..";
		m_childToParent[level].push_back(data);
	    }
	    std::cout<<"\n";
	    
	}
		

    }




private:

    template<typename T2,int DIM2>
    friend class OctreeLevelData;
    
    std::shared_ptr<OctreeNode > m_root;
    std::vector<std::vector<std::shared_ptr<OctreeNode> > > m_nodes;
    std::vector<std::array<std::vector<ConeRef>,N_STEPS> > m_activeCones;
    std::vector<std::vector<ConeRef> > m_leafCones;
    std::vector< FieldInfo > m_farFieldBoxes;  // on each level: for each target point y store the source boxes such that y is in the farfield
    std::vector< FieldInfo > m_nearFieldBoxes;  // on each level: for each target point y store the source boxes such that y is in the farfield 
    std::vector<unsigned int> m_numBoxes;
    std::vector<unsigned int> m_numLeafCones;


    struct ChildToParentData {
	ChildToParentData() {

	    is_valid=0;
	}
	std::vector<size_t > parentElementShifts; // even numbers: direction, odd numbers: cartesion pnts
	std::vector<size_t> directionShifts;
	std::vector<size_t > fineElementInfo;

	std::vector<size_t> directionForPoint;
	std::vector<size_t > point_ids;
	PointArray points;

	PointArray cart_pnts;
	
	int is_valid;
    };
    

    std::vector< std::vector<ChildToParentData> > m_childToParent;



    std::vector< IndexSet > m_activeHoCones;


    unsigned int m_levels;

    std::vector<std::vector<ConeMap> > m_coneMaps;

    unsigned int m_depth;
    size_t m_maxLeafSize;
    PointArray m_pnts;
    std::vector<size_t> m_permutation;
    PointScalar m_diameter;
    PointScalar m_sideLength;

    std::function<bool(const BoundingBox<DIM>&, const BoundingBox<DIM>&) > m_isAdmissible;
};


template<typename T,int DIM>
class SyclChildToParentData {
public:
    SyclChildToParentData(const Octree<T,DIM>::ChildToParentData& data):
	points(data.points),
	parentElementShifts(data.parentElementShifts),
	directionShifts((data.directionShifts)),
	fineElementInfo((data.fineElementInfo)),
	point_ids((data.point_ids)),
	directionForPoint(data.directionForPoint),
	cart_pnts(data.cart_pnts),
	is_valid(data.is_valid)
    {

    }


    struct ConeData {
	size_t id;
	IndexRange pnts;	    
    };

    class Accessor;
    class CtpChildConeIterator
    {
	using iterator_category = std::forward_iterator_tag;
	using difference_type   = std::ptrdiff_t;
	using value_type        = ConeData;
	using pointer           = ConeData*;  // or also value_type*
	using reference         = ConeData&;  // or also value_type&
    public:
	CtpChildConeIterator( const SyclChildToParentData<T,DIM>::Accessor* acc,size_t cur) :
	    m_acc(acc),
	    m_cur(cur)
	{

	}


	
	value_type operator*() const {
	    ConeData data;
	    data.id=m_acc->fineElementInfo(m_cur,0);
	    if(m_cur==0) {
		data.pnts.first=0;
	    }else {
		data.pnts.first=m_acc->fineElementInfo(m_cur-1,1);
	    }
		
	    data.pnts.second=m_acc->fineElementInfo(m_cur,1);
		
	    return data;
	}

	size_t cur() {
	    return m_cur;
	}

	// Prefix increment
	CtpChildConeIterator& operator++() { m_cur++; return *this; }  

	// Postfix increment
	CtpChildConeIterator operator++(int) { CtpChildConeIterator tmp = *this; ++(*this); return tmp; }

	friend bool operator== (const CtpChildConeIterator& a, const CtpChildConeIterator& b) { return a.m_cur == b.m_cur; };
	friend bool operator!= (const CtpChildConeIterator& a, const CtpChildConeIterator& b) { return a.m_cur != b.m_cur; };     



    private:
	size_t m_cur;
	const SyclChildToParentData<T,DIM>::Accessor* m_acc;
    };
    
    


    class Accessor {
    public:
	Accessor(SyclChildToParentData& data,sycl::handler& h):
	    m_points(data.points,h),
	    m_parentElementShifts(data.parentElementShifts,h),
	    m_directionShifts(data.directionShifts,h),
	    m_fineElementInfo(data.fineElementInfo,h),
	    m_point_ids(data.point_ids,h),
	    m_directionForPoint(data.directionForPoint,h),
	    m_cart_pnts(data.cart_pnts,h),
	    m_is_valid(data.is_valid)
	{

	}

	Accessor()
	{

	}

	class ChildConeRange {
	public:
	    ChildConeRange(size_t elId,int direction,const Accessor* acc) :
		m_elId(elId),
		m_dir(direction),
		m_acc(acc)
	    {

	    }

	    auto begin() const {
		size_t idx=m_acc->parentElementShift(m_elId,0)+m_dir;
		return CtpChildConeIterator(m_acc,m_acc->directionShifts(idx));
	    }

	    auto end() const {
		size_t idx=m_acc->parentElementShift(m_elId,0)+m_dir;
		return CtpChildConeIterator(m_acc,m_acc->directionShifts(idx+1));
	    }


	private:
	    size_t m_elId;
	    int m_dir;
	    const Accessor* m_acc;
	};

	ChildConeRange childCones(size_t elId,size_t direction) const{
	    return ChildConeRange(elId, direction,this);
	}

	size_t directionShifts(size_t direction) const {
	    return m_directionShifts[direction];
	}


	size_t parentElementShift(size_t pEl,int type) const {
	    return m_parentElementShifts[2*pEl+type];
	}

	size_t fineElementInfo(size_t id, int type) const {
	    return m_fineElementInfo[2*id+type];
	}


	inline sycl::marray<PointScalar, DIM> point(size_t idx) const {
	    sycl::marray<PointScalar, DIM> pnt;
	    for(int i=0;i<DIM;i++) {
		pnt[i]=m_points[DIM*idx+i];
	    }
	    
	    return pnt;
	}

	inline sycl::marray<PointScalar, DIM> cart_point(size_t idx) const {
	    sycl::marray<PointScalar, DIM> pnt;
	    for(int i=0;i<DIM;i++) {
		pnt[i]=m_cart_pnts[DIM*idx+i];
	    }
	    
	    return pnt;
	}

		inline sycl::marray<PointScalar, DIM> cart_point(size_t pEl,size_t idx) const {
	    sycl::marray<PointScalar, DIM> pnt;
	    for(int i=0;i<DIM;i++) {
		pnt[i]=m_cart_pnts[DIM*parentElementShift(pEl,1)+DIM*idx+i];
	    }
	    
	    return pnt;
	}
	
	inline size_t directionForPoint(size_t pntId) const {
	    return m_directionForPoint[pntId];
	}



	inline 	size_t realPointId (size_t pointId)  const {
	    return m_point_ids[pointId];
	}

	inline 	size_t numPoints() const {
	    return m_points.size()/DIM;
	}

	inline 	size_t numDirections() const {
	    return  m_directionShifts.size();
	}


	inline const sycl::accessor< PointScalar,1,sycl::access_mode::read>& points() const {
	    return m_points;
	};

    private:

	sycl::accessor< PointScalar,1,sycl::access_mode::read> m_points;
	sycl::accessor< size_t,1,sycl::access_mode::read> m_parentElementShifts;
	sycl::accessor< size_t,1,sycl::access_mode::read> m_directionShifts;
	sycl::accessor< size_t,1,sycl::access_mode::read> m_fineElementInfo;
	sycl::accessor< size_t,1,sycl::access_mode::read> m_point_ids;
	sycl::accessor< size_t,1,sycl::access_mode::read> m_directionForPoint;
	sycl::accessor< PointScalar,1,sycl::access_mode::read> m_cart_pnts;
	int m_is_valid;

    };

    




    Accessor accessor(sycl::handler& h)
    {
	return Accessor(*this,h);	
    }


    size_t numPoints() const {
	return points.size()/DIM;
    }
    
    size_t numDirections() const {
	return  directionShifts.size();
    }


    
private:
    sycl::buffer<size_t,1> parentElementShifts;
    sycl::buffer<size_t,1> directionShifts;
    sycl::buffer<size_t,1 > fineElementInfo;
    

    sycl::buffer<size_t,1 > point_ids;
    sycl::buffer<size_t,1 > directionForPoint;
    sycl::buffer<PointScalar,1> points;
    sycl::buffer<PointScalar,1> cart_pnts;
    int is_valid;
};


template <typename T, size_t DIM>
struct sycl::is_device_copyable<SyclChildToParentData<T,DIM> > : std::true_type {};


template<typename T,int DIM>
class OctreeLevelData
{
public:
    OctreeLevelData<T,DIM>(const Octree<T,DIM>& octree,size_t level):
    points_start(octree.numBoxes(level)),
    points_end(octree.numBoxes(level)),
    ffBi_vec(std::move(octree.m_farFieldBoxes[level].indices)),
    ffB_indices(ffBi_vec),
    ffBs_vec(std::move(octree.m_farFieldBoxes[level].starts)),
    ffB_starts(ffBs_vec),
    nfBi_vec(std::move(octree.m_nearFieldBoxes[level].indices)),
    nfB_indices(nfBi_vec),
    nfBs_vec(std::move(octree.m_nearFieldBoxes[level].starts)),
    nfB_starts(nfBs_vec),
    leafCones_vec((octree.m_leafCones[level])),
    leafCones(leafCones_vec),
    ftAFlags(octree.numBoxes(level)),
    boxCenters(octree.numBoxes(level)*DIM),
    boxSizes(octree.numBoxes(level)),
    coneDomains0(octree.numBoxes(level)),
    coneDomains1(octree.numBoxes(level)),
    activeCones_vec(std::move(octree.m_activeCones[level][1])),
    activeCones(activeCones_vec),
    coneMap(SyclHelpers::SyclIndexMap<size_t>::fromList(octree.coneMaps(level))),
    childBoxes(octree.numChildBoxes(level)),
    childrenPerBox(octree.numChildBoxes(level)/octree.numBoxes(level)),
    m_ctpData_vec(std::move(octree.m_childToParent[level]))
    {
	std::cout<<"creating ocdata"<<level<<std::endl;
	sycl::host_accessor starts(points_start,sycl::write_only);
	sycl::host_accessor ends(points_end,sycl::write_only);

	sycl::host_accessor flags(ftAFlags,sycl::write_only);

	sycl::host_accessor bs(boxSizes,sycl::write_only);
	sycl::host_accessor bc(boxCenters,sycl::write_only);

	sycl::host_accessor cds0(coneDomains0,sycl::write_only);
	sycl::host_accessor cds1(coneDomains1,sycl::write_only);


	sycl::host_accessor cB(childBoxes,sycl::write_only);
	
	//serialize the points
	for(size_t box=0;box<octree.numBoxes(level);box++)
	{
	    
	    auto range =octree.points(level,box);
	    starts[box]=range.first;	    
	    ends[box]=range.second;

	    flags[box]=octree.hasFarTargetsIncludingAncestors(level,box);

	    auto bbox=octree.bbox(level,box);
	    bs[box]=bbox.sideLength();
	    for(int j=0;j<DIM;j++) {
		bc[box*DIM+j]=bbox.center()[j];
	    }

	    cds0[box]=octree.coneDomain(level,box,0);
	    cds1[box]=octree.coneDomain(level,box,1);
	    
	    
	    
	    if(octree.numChildBoxes(level)>0) {
		size_t id=0;

		for( const auto & cb : octree.childBoxes(level,box)) {
		    cB[box*childrenPerBox+id]=cb;
		    id++;
		}
		for(;id<childrenPerBox;id++) {
		    cB[box*childrenPerBox+id]=SIZE_MAX;
		}
	    }
	    

	}

	for(int i=0;i<N_STEPS;i++){
	    m_numActiveCones[i]=octree.numActiveCones(level,i);
	}


	//std::cout<<"done"<<std::endl;

	m_ctpData.reserve(octree.m_childToParent[level].size());
	for(const auto& data : m_ctpData_vec) {
	    m_ctpData.push_back(std::make_unique<SyclChildToParentData<T,DIM> >(data));
	}

    }
  
    


    class Accessor
    {
    public:
	Accessor(OctreeLevelData& data,sycl::handler& h) :
	    ffB_indices(data.ffB_indices,h),
	    ffB_starts(data.ffB_starts,h),
	    nfB_indices(data.nfB_indices,h),
	    nfB_starts(data.nfB_starts,h),
	    points_start(data.points_start,h),
	    points_end(data.points_end,h),
	    leafCones(data.leafCones,h),
	    ftAFlags(data.ftAFlags,h),
	    boxCenters(data.boxCenters,h),
	    boxSizes(data.boxSizes,h),
	    activeCones(data.activeCones,h),
	    coneMap(data.coneMap.accessor(h)),
	    childBoxes(data.childBoxes,h),
	    childrenPerBox(data.childrenPerBox)
	{
	    coneDomains0=sycl::accessor(data.coneDomains0,h);
	    coneDomains1=sycl::accessor(data.coneDomains1,h);
	    

	}

    

	inline IndexRange  points(size_t boxId) const
	{
	    return IndexRange({points_start[boxId],points_end[boxId]});
	}



	const inline  auto farfieldBoxes(size_t targetPoint) const
	{
	    const size_t start=ffB_starts[targetPoint];
	    const size_t end=ffB_starts[targetPoint+1];


	    
	    return SyclHelpers::SubRange<sycl::accessor<const size_t,1,sycl::access_mode::read> >(ffB_indices.cbegin()+start,ffB_indices.cbegin()+end);
	}

	const inline  auto nearFieldBoxes(size_t targetPoint) const
	{	    
	    const size_t start=nfB_starts[targetPoint];
	    const size_t end=nfB_starts[targetPoint+1];


	    return SyclHelpers::SubRange<sycl::accessor<const size_t,1,sycl::access_mode::read> >(nfB_indices.cbegin()+start,nfB_indices.cbegin()+end);
	}

    

	const ConeRef& leafCone(size_t id) const
	{
	    return leafCones[id];
	}



	bool hasFarTargetsIncludingAncestors(size_t boxId) const
	{
	    return ftAFlags[boxId];
	}

	PointScalar boxSize(size_t boxId) const
	{
	    return boxSizes[boxId];
	}

	sycl::marray<PointScalar,DIM> boxCenter(size_t boxId) const
	{
	    sycl::marray<PointScalar,DIM> c;
	    for(int i=0;i<DIM;i++) {
		c[i]=boxCenters[boxId*DIM+i];
	    }

	    return c;
	}
	
	const SyclConeDomain<DIM>& coneDomain(size_t boxId, int step) const
	{
	    if(step==0)
		return coneDomains0[boxId];
	    else
		return coneDomains1[boxId];
	}

	ConeRef activeCone(size_t index) const
	{
	    return activeCones[index];
	}

	

	size_t memId(size_t box, size_t el) const {
	    return coneMap.find(box,el);
	}

	const inline  auto children(size_t boxId) const
	{
	    const size_t start=boxId*childrenPerBox;
	    const size_t end=(boxId+1)*childrenPerBox;
	    
	    return SyclHelpers::SubRange<sycl::accessor<const size_t,1,sycl::access_mode::read> >(childBoxes.cbegin()+start,childBoxes.cbegin()+end);
	}

    private:
	//far field boxes
	sycl::accessor< size_t,1,sycl::access_mode::read> ffB_indices;
	sycl::accessor< size_t,1,sycl::access_mode::read> ffB_starts;

	//near field boxes
	sycl::accessor< size_t,1,sycl::access_mode::read> nfB_indices;
	sycl::accessor< size_t,1,sycl::access_mode::read> nfB_starts;

	//points
	sycl::accessor< size_t,1,sycl::access_mode::read> points_start;
	sycl::accessor< size_t,1,sycl::access_mode::read> points_end;



	sycl::accessor< ConeRef,1,sycl::access_mode::read> leafCones;

	sycl::accessor< ConeRef,1,sycl::access_mode::read> activeCones;

	sycl::accessor< char,1,sycl::access_mode::read> ftAFlags;



	sycl::accessor< PointScalar ,1,sycl::access_mode::read> boxSizes;
	sycl::accessor< PointScalar ,1,sycl::access_mode::read> boxCenters;
      

	//interpolation data
	sycl::accessor< T ,1,sycl::access_mode::read> interpolationData;
	
	sycl::accessor< SyclConeDomain<DIM> ,1,sycl::access_mode::read> coneDomains0;
	sycl::accessor< SyclConeDomain<DIM> ,1,sycl::access_mode::read> coneDomains1;

	
	SyclHelpers::SyclIndexMap<size_t >::Accessor  coneMap;

	sycl::accessor< size_t ,1,sycl::access_mode::read> childBoxes;
	size_t childrenPerBox;

	
    };


    ConeRef activeCone(size_t index) {
	return activeCones_vec[index];
    }


    Accessor accessor(sycl::handler& h)
    {
	return Accessor(*this,h);	
    }


    typename SyclChildToParentData<T,DIM>::Accessor  childToParent(size_t idx,sycl::handler& h) {
	return m_ctpData[idx]->accessor(h);
    }

    size_t numCtpSlices() {
	return m_ctpData_vec.size();
    }

    size_t numLeafCones() {
	return leafCones.size();
    }

    size_t numPointsInSlice(size_t idx) {
	return m_ctpData[idx]->numPoints();
    }

    size_t numActiveCones(int step) {
	
	return m_numActiveCones[step];
    }


    
private:
    //far field boxes
    std::vector<size_t> ffBs_vec;
    std::vector<size_t> ffBi_vec;

    sycl::buffer<size_t,1> ffB_indices;
    sycl::buffer<size_t,1> ffB_starts;

    

    //near field boxes
    
    std::vector<size_t> nfBs_vec;
    std::vector<size_t> nfBi_vec;

    sycl::buffer<size_t,1> nfB_indices;
    sycl::buffer<size_t,1> nfB_starts;



    sycl::buffer<size_t,1> points_start;
    sycl::buffer<size_t,1> points_end;

    std::vector<ConeRef> leafCones_vec;
    sycl::buffer<ConeRef,1> leafCones;
    sycl::buffer<char,1> ftAFlags;


    sycl::buffer<PointScalar,1> boxSizes;
    sycl::buffer<PointScalar,1> boxCenters;

    std::vector<ConeRef> activeCones_vec;
    sycl::buffer<ConeRef> activeCones;

    sycl::buffer<SyclConeDomain<DIM>,1> coneDomains0;
    sycl::buffer<SyclConeDomain<DIM>,1> coneDomains1;
    

    SyclHelpers::SyclIndexMap<size_t> coneMap;


    sycl::buffer<size_t,1> childBoxes;
    size_t childrenPerBox;

    std::array<size_t, N_STEPS> m_numActiveCones;


    std::vector<typename Octree<T,DIM>::ChildToParentData > m_ctpData_vec;
    std::vector<std::unique_ptr<SyclChildToParentData<T,DIM> > > m_ctpData;

};



template <typename T, int DIM>
class FlatOctree
{
    typedef Eigen::Array<PointScalar, DIM, Eigen::Dynamic> PointArray;
    typedef Eigen::Vector<PointScalar, DIM> Point;

public:
    FlatOctree()
    {

    }
    
    FlatOctree(const Octree<T,DIM> src_octree,const Octree<T,DIM> target_octree):
	m_srcPermutation(src_octree.permutation()),
	m_srcPoints(src_octree.points()),
	m_targetPermutation(target_octree.permutation()),
	m_targetPoints(target_octree.points()),
	m_diameter(src_octree.diameter()),
	m_sideLength(src_octree.sideLength())
    {
	m_data.reserve(src_octree.levels());
	for(int level=0;level<src_octree.levels();level++) {
	    m_data.push_back(std::make_shared<OctreeLevelData<T,DIM> > (src_octree,level));
	}
		
    }

    std::shared_ptr<OctreeLevelData<T,DIM> >  data(int level) 
    {
	std::cout<<"accessing "<<level<<" vs "<<m_data.size()<<std::endl;
	return m_data[level];
    }

    int levels() const
    {
	return m_data.size();
    }


    const std::vector<size_t>& targetPermutation() const
    {
	return m_targetPermutation;
    }

    const std::vector<size_t>& srcPermutation() const
    {
	return m_srcPermutation;
    }

    const PointArray& srcPoints() const
    {
	return m_srcPoints;
    }    


    const PointArray& targetPoints() const
    {
	return m_targetPoints;
    }

    

    double diameter() const
    {
	return m_diameter;
    }

    double sideLength() const
    {
	return m_sideLength;
    }


    size_t numLeafCones(size_t level) {
	return m_data[level]->numLeafCones();
    }

    size_t numActiveCones(size_t level,size_t step) {
	return m_data[level]->numActiveCones(step);
    }



    


private:
    std::vector<std::shared_ptr<OctreeLevelData<T,DIM> > > m_data;
    std::vector<size_t> m_srcPermutation;
    PointArray m_srcPoints;

    //instead of storing the whole target octree with complicated data we just store points and permutation as that is all we really need
    std::vector<size_t> m_targetPermutation;
    PointArray m_targetPoints;

    double m_diameter;
    double m_sideLength;

    
};



template <typename T, int DIM,typename KeyType>
class OctreeCache
{
public:
    static OctreeCache& getInstance()
    {
	static OctreeCache    instance; // Guaranteed to be destroyed.
	// Instantiated on first use.
	return instance;
    }
    public:
    OctreeCache(OctreeCache const&)               = delete;
    void operator=(OctreeCache const&)  = delete;



    std::shared_ptr<FlatOctree<T,DIM> > find(KeyType key) {
	for(int i=0;i<m_cache.size();i++) {
	    if(m_cache[i].first==key){
		return m_cache[i].second;
	    }
	}
	return 0;
    }

    void add(KeyType key,std::shared_ptr<FlatOctree<T,DIM> > octree) {
	m_cache[m_idx]=std::make_pair(key, octree);
	m_idx=(m_idx+1) % m_cache.size();
        std::cout<<"cache size="<<m_cache.size()<<std::endl;
    }


private:
    OctreeCache()
    {
	m_cache.resize(1);
	m_idx=0;
    }
    ~OctreeCache()
    {

    }

    
    

private:
    int m_idx;
    std::vector<std::pair<KeyType, std::shared_ptr<FlatOctree<T,DIM> > > > m_cache;
    //ankerl::unordered_dense::map<KeyType, std::shared_ptr<Octree<T,DIM> > > m_cache;
};







    

    



#endif
