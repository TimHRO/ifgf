#include <Eigen/Dense>

#include "chebinterp_sycl.hpp"



#if 0

template <typename T, unsigned int DIM, unsigned int DIMOUT>
void SyclChebychevInterpolation::parallel_evaluate(
					       const Eigen::Ref<const Eigen::Array<PointScalar, DIM, Eigen::Dynamic> >
					       &points,
					       const Eigen::Ref<const Eigen::Array<T, Eigen::Dynamic, DIMOUT> >
					       &interp_values,
					       Eigen::Ref<Eigen::Array<T, Eigen::Dynamic, DIMOUT> > dest,
					       const Eigen::Vector<int, DIM>& ns,
					       BoundingBox<DIM> box)
    {
	
	Eigen::Array<PointScalar,DIM,Eigen::Dynamic> points0(DIM,points.cols());

	const auto a=0.5*(box.max()-box.min()).array();
	const auto b=0.5*(box.max()+box.min()).array();

	if(!box.isNull()) {
	    points0.array()=(points.array().colwise()-b).colwise()/a;
	}else {
	    points0=points;
	}

	//std::cout<<"ev"<<nodes[DIM-1]<<std::endl;

	//dest.resize(DIMOUT,points.cols());
        
	//template <typename T, int N_POINTS_AT_COMPILE_TIME, unsigned int DIM, unsigned int DIMOUT, typename Derived1, typename Derived2, int N_AT_COMPILE_TIME, int... OTHER_NS>



	//for(int i=0;i<points.cols();)
	//{

	//std::cout<<"ize="<<interp_values.size()<<std::endl;
	sycl::buffer<T,1>  b_interp_values(interp_values.data(),sycl::range{(size_t) interp_values.size()});
	sycl::buffer<T,1>  b_dest(dest.data(),sycl::range{(size_t)  dest.size()});

	sycl::buffer<PointScalar>  b_pnts(points0.data(),sycl::range{DIM*((size_t)  points0.cols())});

	//sycl::marray<int,DIM> m_ns=SyclHelpers::EigenVectorToMArray(ns);

	std::array<int, DIM> m_ns;
	for(int i=0;i<DIM;i++)
	    m_ns[i]=ns[i];

	sycl::queue Q;
	const auto &device = Q.get_device();

	Q.submit([&] (sycl::handler& h) {
	    sycl::accessor a_interp_values(b_interp_values,h,sycl::read_only);
	    sycl::accessor a_dest(b_dest,h,sycl::write_only,sycl::no_init);
	    sycl::accessor a_pnts(b_pnts,h,sycl::read_only);
	    
	    size_t n_points = points.cols();
	    h.single_task( [a_pnts,a_interp_values,a_dest,n_points,m_ns]() {

		if constexpr(DIM==3) {

		    if(m_ns[0]==2 && m_ns[1]==4 && m_ns[2]==4)  {
			__eval<T, DIM, 5, 4, 4, 2>(a_pnts, a_interp_values, m_ns, a_dest, 0, n_points);
		    }
		    else if(m_ns[0]==4 && m_ns[1]==6 && m_ns[2]==6)  {
			__eval<T, DIM, 5, 6, 6, 4>(a_pnts, a_interp_values, m_ns, a_dest, 0, n_points);
		    }else if(m_ns[0]==6 && m_ns[1]==8 && m_ns[2]==8)  {
			__eval<T, DIM, 5, 8, 8, 6>(a_pnts, a_interp_values, m_ns, a_dest, 0, n_points);
			return;
		    }else if(m_ns[0]==8 && m_ns[1]==10 && m_ns[2]==10)  {
			__eval<T, DIM, 5, 10, 10, 8>(a_pnts, a_interp_values, m_ns, a_dest, 0, n_points);
			return;
		    }else{		
			__eval<T, DIM, 5,-1,-1,-1>(a_pnts, a_interp_values, m_ns, a_dest, 0, n_points);
		    }
		}else{
		    __eval<T, DIM, 5,-1,-1,-1>(a_pnts, a_interp_values, m_ns, a_dest, 0, n_points);
	    
		}
		
	    });
	}


      );
	Q.wait();

	//sycl::host_accessor result{b_dest, sycl::read_only};
	

	    //std::cout<<"i"<<i<<" vs "<<r.end()<<std::endl;
	    //assert(i == r.end());
	 //}
	    

    }

#endif
