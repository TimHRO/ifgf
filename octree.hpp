#ifndef _OCTREE_HPP_
#define _OCTREE_HPP_

#include "Eigen/src/Core/VectorBlock.h"
#include "config.hpp"

#include <Eigen/Dense>
#include <iterator>
#include <memory>
#include <map>
#include <vector>
#include <execution>
#include <iostream>
#include <unordered_set>

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
	Eigen::Vector<size_t, Eigen::Dynamic> indices;
	Eigen::Vector<size_t, Eigen::Dynamic> starts;
    };

    class OctreeNode
    {
    private:
        std::weak_ptr<OctreeNode > m_parent;
        long int m_id;
        std::shared_ptr<OctreeNode> m_children[N_Children];
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

        const std::shared_ptr<const OctreeNode> child(size_t idx) const
        {
            return m_children[idx];
        }

        std::shared_ptr<OctreeNode> child(size_t idx)
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
                    if (m_children[i] != 0) {
                        m_children[i]->print(prefix + "    ");
                    } else {
                        std::cout << prefix + "    ----x" << std::endl;
                    }
                }
            }
        }

    };

    typedef std::pair<size_t, size_t> BoxIndex; //level + index in level

    Octree(int maxLeafSize):
        m_maxLeafSize(maxLeafSize)
    {

	PointScalar eta=(PointScalar) sqrt((PointScalar) DIM);
	m_isAdmissible= [eta] (const BoundingBox<DIM>& src,const BoundingBox<DIM>& target) { return target.exteriorDistance(src.center()) >= eta* src.sideLength();};

    }

    ~Octree()
    {

    }

    void build(const PointArray &pnts)
    {
        std::cout << "building a new octree" << pnts.cols()<<std::endl;

        std::cout << "finding bbox" << std::endl;
        Point min = pnts.col(0);
        Point max = pnts.col(0);

        for (size_t i = 0; i < pnts.cols(); i++) {
            min = min.array().min(pnts.col(i).array());
            max = max.array().max(pnts.col(i).array());
        }


        //make the bbox slightly larger to not have to deal with boundary issues
        min.array() -= 0.1 * min.norm();
        max.array() += 0.1 * max.norm();

        BoundingBox<DIM> bbox(min, max);
        std::cout << "bbox=" << min.transpose() << "\t" << max.transpose() << std::endl;

        //sort the points by their morton order for better locality later on
        std::cout << "sorting..." << std::endl;
        m_permutation = Util::sort_with_permutation( pnts.colwise().begin(), pnts.colwise().end(), zorder_knn::Less<Point, DIM>(bbox));
	m_pnts.resize(DIM, pnts.cols());
        Util::copy_with_permutation_colwise<PointScalar,DIM>(pnts, m_permutation,m_pnts);

	m_diameter=bbox.diagonal().norm();

	m_levels = 0;
	m_depth = -1;
        m_root = buildOctreeNode(0, std::make_pair(0, m_pnts.cols()), bbox);
	m_depth=m_levels;


        std::cout << "building the nodes up to level"<<m_depth << std::endl;
  

	
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
		buildInteractionList(src,target->child(j));
	    }
	    return;
	}

	if(target->isLeaf() && !src->isLeaf()) {
	    for (int j = 0; j < N_Children; j++) {
		buildInteractionList(src->child(j),target);
	    }
	    return;
	}

	//if both src and target have children, we recurse down the one with the larger bbox
	if(src->boundingBox().sideLength() > target->boundingBox().sideLength()) {
	    for (int j = 0; j < N_Children; j++) {
		buildInteractionList(src->child(j),target);
	    }
	    return;
	}else {
	    for (int j = 0; j < N_Children; j++) {
		buildInteractionList(src,target->child(j));
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
	
	BoundingBox<DIM> global_box;
	//global_box.min().fill(10);
	//global_box.max().fill(0);

	TransformationMode mode=TwoGrid;

	const auto & target_points=target.points();
	tbb::queuing_mutex activeConeMutex;

	std::vector<tbb::spin_mutex> target_mutexes(target.numPoints());
	Eigen::Vector<size_t,DIM> oldN;
	oldN.fill(0);
	//No interpolation at the two highest levels 
	for (size_t level=0;level<levels();level++) {
	    std::cout<<"l= "<<level<<std::endl;
	    //update all nodes in this level

	    std::array<std::vector<ConeRef>,N_STEPS> activeCones;
	    std::vector<ConeRef> leafCones;
	    m_coneMaps[level].resize(numBoxes(level));
	    tbb::parallel_for(tbb::blocked_range<size_t>(0,numBoxes(level)), [&](tbb::blocked_range<size_t> r) {
            for(size_t n=r.begin();n<r.end();++n) {
		std::shared_ptr<OctreeNode> node=m_nodes[level][n];
		BoundingBox<DIM> box;

		const Point xc=node->boundingBox().center();
		
		const PointScalar H=node->boundingBox().sideLength();		
		const std::vector<IndexRange> farTargets=node->farTargets();

		//just use a default value for the boxes

		PointScalar smax=sqrt(DIM)/DIM;
		//if the sources and targets are well-separated we don't have to cover the near field 
		const PointScalar dist=0;//bbox(0,0).exteriorDistance(target.bbox(0,0));
		if(dist >0) {
		    smax=std::min(smax, H/dist);
		}

                
		std::shared_ptr<const OctreeNode> parent=node->parent().lock();
		BoundingBox<DIM> pBox;
		//now also add all the parents targets	       
		if(parent && parentHasFarTargets(node))
		{	    
		    pBox=parent->interpolationRange();
		}

		
                const PointScalar dist_t=0;//target.bbox(0,0).exteriorDistance(xc);
		PointScalar smin= 1e-3;//smin_for_H(H);

		/*
                if(!pBox.isNull()) {
		    //std::cout<<"diam:"<<m_diameter<<" "<<pBox.min()[0]<<std::endl;
                    smin=std::min(smin,0.75/(sqrt(DIM)/3.0+2.0/pBox.min()[0]));
		    //smin=std::min(smin, pBox.min()[0]/2.0);
		}
		smin=1e-3;
		*/
                //1e-3;//H/(m_diameter+dist+target.m_diameter);


		if(smin >=smax) {
		    smin=smax-0.05;
		}
		
		box.min()(0)=smin;
		box.max()(0)=smax;

			
		if constexpr(DIM==2) {	    
		    box.min()(1)=-M_PI;
		    box.max()(1)=M_PI;
		}else{
		    box.min()(1)=0;
		    box.max()(1)=M_PI;
			    
		    box.min()(2)=-M_PI;
		    box.max()(2)=M_PI;
		}
	
		    

		BoundingBox<DIM> tbox(pBox);

		
		ConeDomain<DIM> domain(N_for_H(H,0),box);
		ConeDomain<DIM> trans_domain(N_for_H(2*H,1),tbox);

		auto hoN=N_for_H(H,1);

		ConeDomain<DIM> coarseDomain(hoN,box);

		    
		assert(H>0);
		if(N_for_H(H,0)!=oldN) {
		    oldN=N_for_H(H,0);
		    std::cout<<"n="<<N_for_H(H,0).transpose()<<"/"<<N_for_H(H,1).transpose()<<std::endl;
		    std::cout<<"box="<<box<<" "<<box.isNull()<<"H="<<H<<std::endl;
		}

		if(!box.isNull())
		    global_box.extend(box);

		//now we need to do the whole thing again to figure out which cones are active...
		// 0 = fine grid low order (used for evaluating FF and propagating upwards
		// 1 = coarse grid high order (used as target for interplating leaves and for propagation from below)
		std::array<IndexSet,N_STEPS> is_cone_active; 
		

		PointArray s(DIM,32);
		for(const IndexRange& iR : farTargets)
		{
		    s.resize(DIM,std::max((size_t) s.cols(),iR.second-iR.first));
		    Util::cartToInterp2<DIM>(target_points.middleCols(iR.first,iR.second-iR.first),xc,H,s.leftCols(iR.second-iR.first).array());			  
		    for(int i=0;i<iR.second-iR.first;i++)
		    {
			//const auto s=Util::cartToInterp<DIM>(target_points.col(i),xc,H);			  
			auto coneId=domain.elementForPoint(s.col(i));
			auto coneId2=coarseDomain.elementForPoint(s.col(i));
			if(coneId2<SIZE_MAX)
			    is_cone_active[1].insert(coneId2);
			if(coneId<SIZE_MAX)
			    is_cone_active[0].insert(coneId);
		    }
		}

		
		//now also add all the parents targets
		if(!pBox.isNull())
		{
		    const Point pxc=parent->boundingBox().center();
		    const PointScalar pH=parent->boundingBox().sideLength();

		    const ConeDomain<DIM>& p_grid=parent->coneDomain(1);

		    if(parentHasFarTargets(level,n))
		    {		    
			if(mode==TwoGrid) {
			    //We use a coarse grid with high order and a fine grid of lower order
			    auto HoChebNodes = ChebychevInterpolation::chebnodesNdd<PointScalar, DIM>(order_for_H(pH,1));			
			    const ConeDomain<DIM>& p_hoGrid=parent->coneDomain(1);
			    //

			    PointArray pnts(DIM,HoChebNodes.cols());
			    PointArray interp_pnts(DIM,HoChebNodes.cols());
			    for(size_t el : p_hoGrid.activeCones() ) {
				pnts=Util::interpToCart<DIM>(p_hoGrid.transform(el,HoChebNodes).array(),pxc,pH);
				Util::cartToInterp2<DIM>(pnts.array(),xc,H,interp_pnts.array());
				for (size_t i=0;i<HoChebNodes.cols();i++) {
				    auto coneId=domain.elementForPoint(interp_pnts.col(i));
				    auto coneId2=coarseDomain.elementForPoint(interp_pnts.col(i));
				    if(coneId2<SIZE_MAX)
					is_cone_active[1].insert(coneId2);
				    if(coneId<SIZE_MAX)
					is_cone_active[0].insert(coneId);
				}
			    }			

			}
			else   {
			    //now we add all the points used in the regular method
			    auto chebNodes = ChebychevInterpolation::chebnodesNdd<PointScalar, DIM>(order_for_H(pH,0));
			    for(size_t el : p_grid.activeCones() ) {	    
				for (size_t i=0;i<chebNodes.cols();i++) {
				    Point cart_pnt=Util::interpToCart<DIM>(p_grid.transform(el,chebNodes.col(i)).array(),pxc,pH);
				    Point interp_pnt=Util::cartToInterp<DIM>(cart_pnt,xc,H);

				    auto coneId=domain.elementForPoint(interp_pnt);
				    if(coneId<SIZE_MAX)
					is_cone_active[0].insert(coneId);
				}
			    }
			}
		    }
		}


		/*
		if(mode==TwoGrid) {
		    //mark all children of active elements in the coarse grid also as active.
		    //reinterpolation step from HO to regular
		    const int n_children=domain.n_elements()/coarseDomain.n_elements();
		    const int factor=domain.n_elements(0)/coarseDomain.n_elements(0);
		    for(size_t cone : is_cone_active[1]) {
			Eigen::Vector<size_t, DIM> indices=coarseDomain.indicesFromId(cone);
			indices*=factor;
			for(int i=0;i<n_children;i++) {
			    //compute the different indices given the id
			    Eigen::Vector<size_t, DIM> d;
			    size_t tmp=i;
			    for(int l=0;l<DIM;l++) {
				const size_t idx=tmp % factor;
				tmp=tmp / factor;
	    
				d[l]=idx;
			    }
			    
			    const size_t childId=domain.idFromIndices(indices+d);
			    is_cone_active[0].insert(childId);
			}
			}
			}*/

		for (int step=0;step<N_STEPS;step++ ) {
		     std::vector<size_t> local_active_cones;
		     IndexMap cone_map;
		     //local_active_cones.reserve(domain.n_elements());
		     for( size_t i : is_cone_active[step])
		     {			 
			 
			 
			 tbb::queuing_mutex::scoped_lock lock(activeConeMutex);
			 
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
	    {
		tbb::queuing_mutex::scoped_lock lock(activeConeMutex);
		std::cout<<"level= "<<level<<" active cones"<<activeCones[0].size()<<" full: "<<numBoxes(level)*coneDomain(level,0,0).n_elements()<<std::endl;
		std::cout<<"level= "<<level<<" active cones"<<activeCones[1].size()<<" full: "<<numBoxes(level)*coneDomain(level,0,1).n_elements()<<std::endl;		
		std::cout<<"level= "<<level<<" leafCones "<<leafCones.size()<<std::endl;
	    }
	    m_activeCones.push_back(activeCones);
	    m_leafCones.push_back(leafCones);


	    //compute the farFieldBoxes
	    m_farFieldBoxes[level]=computeFieldInfo(level,target, true);
	    m_nearFieldBoxes[level]=computeFieldInfo(level,target, false);
			

	}

	std::cout<<"interp_domain:" <<global_box<<std::endl;

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
	    info.starts.fill(0);		
		    
	}
	return info;
		
    }


	
    
    inline PointScalar diameter () const
    {
	return m_diameter;
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
	    const auto child=parent->child(i);
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
	return m_leafCones[level].size();
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
    unsigned int m_levels;

    std::vector<std::vector<ConeMap> > m_coneMaps;

    unsigned int m_depth;
    size_t m_maxLeafSize;
    PointArray m_pnts;
    std::vector<size_t> m_permutation;
    PointScalar m_diameter;

    std::function<bool(const BoundingBox<DIM>&, const BoundingBox<DIM>&) > m_isAdmissible;
};





template<typename T,int DIM>
class OctreeLevelData
{

public:
    OctreeLevelData<T,DIM>(const Octree<T,DIM>& octree,size_t level):
    points_start(octree.numBoxes(level)),
    points_end(octree.numBoxes(level)),
    ffB_indices(octree.m_farFieldBoxes[level].indices),	
    ffB_starts(octree.m_farFieldBoxes[level].starts),
    nfB_indices(octree.m_nearFieldBoxes[level].indices),
    nfB_starts(octree.m_nearFieldBoxes[level].starts),
    leafCones(octree.m_leafCones[level]),
    ftAFlags(octree.numBoxes(level)),
    boxCenters(octree.numBoxes(level)*DIM),
    boxSizes(octree.numBoxes(level)),
    coneDomains0(octree.numBoxes(level)),
    coneDomains1(octree.numBoxes(level)),
    activeCones(octree.m_activeCones[level][1]),
    coneMap(SyclHelpers::SyclIndexMap<size_t>::fromList(octree.coneMaps(level))),
    childBoxes(octree.numChildBoxes(level)),
    childrenPerBox(octree.numChildBoxes(level)/octree.numBoxes(level))
    {
	std::cout<<"creating ocdata"<<std::endl;
	sycl::host_accessor starts(points_start,sycl::write_only);
	sycl::host_accessor ends(points_end,sycl::write_only);

	sycl::host_accessor flags(ftAFlags,sycl::write_only);

	sycl::host_accessor bs(boxSizes,sycl::write_only);
	sycl::host_accessor bc(boxCenters,sycl::write_only);

	sycl::host_accessor cds0(coneDomains0,sycl::write_only);
	sycl::host_accessor cds1(coneDomains1,sycl::write_only);




	std::cout<<"boxcenters"<<boxCenters.size()<<"\n";


	

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

	std::cout<<"done"<<std::endl;

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


    Accessor accessor(sycl::handler& h)
    {
	return Accessor(*this,h);	
    }
        

    
private:
    //far field boxes
    sycl::buffer<size_t,1> ffB_indices;
    sycl::buffer<size_t,1> ffB_starts;

    //near field boxes
    sycl::buffer<size_t,1> nfB_indices;
    sycl::buffer<size_t,1> nfB_starts;


    sycl::buffer<size_t,1> points_start;
    sycl::buffer<size_t,1> points_end;

    sycl::buffer<ConeRef,1> leafCones;
    sycl::buffer<char,1> ftAFlags;


    sycl::buffer<PointScalar,1> boxSizes;
    sycl::buffer<PointScalar,1> boxCenters;

    sycl::buffer<ConeRef> activeCones;

    sycl::buffer<SyclConeDomain<DIM>,1> coneDomains0;
    sycl::buffer<SyclConeDomain<DIM>,1> coneDomains1;
    

    SyclHelpers::SyclIndexMap<size_t> coneMap;


    sycl::buffer<size_t,1> childBoxes;
    size_t childrenPerBox;
};









    

    



#endif
