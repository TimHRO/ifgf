#ifndef __HELMHOLTZ_IFGF_HPP__
#define __HELMHOLTZ_IFGF_HPP__

#include "config.hpp"
#include "ifgfoperator.hpp"



class HelmholtzKernelFunctions
{
    typedef std::complex<RealScalar> T;
    const static  int dim=3;
    typedef Eigen::Array<PointScalar, dim, Eigen::Dynamic> PointArray;
    typedef Eigen::Vector<PointScalar,dim> Point;

public:
    HelmholtzKernelFunctions(RealScalar waveNr):
	k(waveNr)
    {
    }


    inline T kernelFunction(const sycl::marray<PointScalar,3>& x) const
    {
        RealScalar d = sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
	return std::abs(d)==0.0 ? RealScalar(0) : RealScalar((1. / (4. * M_PI))) * T(cos(k*d),sin(k*d)) / (d);
    }


    
    template <typename AT1,typename AT2,typename AT3>
    T evaluateKernel(const AT1& xs, size_t x0, size_t xend, const AT2& ys, size_t y0,
			     const AT3& ws)  const
    {
	T result=0;

	sycl::marray<PointScalar,3> pnt;
	for (size_t i = x0; i < xend; i++) {
	    for(int l=0;l<dim;l++)  {
		pnt[l]=xs[i*dim+l]-ys[y0*dim+l];
	    }
	    result += ws[i] * kernelFunction(pnt);	
        }
	return result;
    }



    template <typename AT1,typename AT2>
    T  evaluateFactoredKernel(
			      const AT1& xs, size_t x0, size_t xend, const sycl::marray<PointScalar,dim>& y,
			      const AT2& ws, const sycl::marray<PointScalar,dim>& xc, PointScalar H) const
    {

	T result=0;

	sycl::marray<PointScalar,3> pnt{y[0]-xc[0],y[1]-xc[1],y[2]-xc[2]};
	RealScalar dc = sqrt(pnt[0]*pnt[0]+pnt[1]*pnt[1]+pnt[2]*pnt[2]);

	for(size_t i=x0;i<xend;i++) {
	    sycl::marray<PointScalar,3> pnt{xs[i*dim]-y[0],xs[i*dim+1]-y[1],xs[i*dim+2]-y[2]};
	    RealScalar d = sqrt(pnt[0]*pnt[0]+pnt[1]*pnt[1]+pnt[2]*pnt[2]);

	    //result +=  ws[i] *  sycl::exp(T(0,k)* (d - dc)) * (dc) / d;

	    result += (abs(d)<1e-12) ? RealScalar(0) :  ws[i] *  T(sycl::cos(k*(d-dc)),sycl::sin(k*(d-dc))) * (dc) / d;
	    //result+=(d<1e-12) ? 0 :   (ws[i] * (sycl::cos(k*(d-dc))+T(0,1)*sycl::sin(k*(d-dc)))*(dc/(d)));
	}
	return result;
    }




    template<typename TX>
    inline T CF(TX x) const
    {
	const RealScalar d2 = x[0]*x[0]+x[1]*x[1]+x[2]*x[2];

	if(abs(d2)<1e-12) {
	    return 0;
	}
	const RealScalar id=(sycl::rsqrt(d2));
	const RealScalar d=d2*id;

	
	return T(sycl::cos(k*d),sycl::sin(k*d))*id  *RealScalar(1./(4.0 * M_PI));	    

    }

    
    template<typename TX , typename TY>
    inline T transfer_factor(TX x, TY xc, PointScalar H, TY pxc, PointScalar pH) const
    {
	auto z=x-xc;
	auto zp=x-pxc;
	const RealScalar d = sqrt(z[0]*z[0]+z[1]*z[1]+z[2]*z[2]);
	const RealScalar dp = sqrt(zp[0]*zp[0]+zp[1]*zp[1]+zp[2]*zp[2]);

	if(abs(d)<1e-12) {
	    return 0;
	}

	return T(sycl::cos(k*(d-dp)),sycl::sin(k*(d-dp)))*dp/d;
	
    }





private:
    PointScalar k;
};


template<size_t dim >
class HelmholtzIfgfOperator : public IfgfOperator<std::complex<RealScalar>, dim,
                                                  1, HelmholtzIfgfOperator<dim> >
{
public:
    typedef Eigen::Array<PointScalar, dim, Eigen::Dynamic> PointArray;
    typedef Eigen::Vector<PointScalar,dim> Point;


    
    
    HelmholtzIfgfOperator(RealScalar waveNumber,
                          size_t leafSize,
                          size_t order,
                          size_t n_elem=1,PointScalar tol=-1):
        IfgfOperator<std::complex<RealScalar>, dim, 1, HelmholtzIfgfOperator<dim> >(leafSize,order, n_elem,tol),
        k(waveNumber)
    {
	//try to find an octree in the cache!
    }

    ~HelmholtzIfgfOperator()
    {
	std::cout<<"deleting helmholtz ifgf"<<std::endl;
    }

    typedef std::complex<PointScalar > T ;


    inline HelmholtzKernelFunctions kernelFunctions() const {
	HelmholtzKernelFunctions f(k);
	return f; 
    }
    /*
    template<typename TX>
    inline Eigen::Vector<T, TX::ColsAtCompileTime>  kernelFunction(TX x) const
    {
	Eigen::Array<typename TX::Scalar, 1, TX::ColsAtCompileTime> d2 = x.colwise().squaredNorm();

	auto invd=(d2 < std::numeric_limits<typename TX::Scalar>::min()).select(0,Eigen::rsqrt(d2));
	
	const auto d=d2*invd;

	const PointScalar factor=1.0/ (4.0 * M_PI);        
        
        return (factor*Eigen::exp(-k * d) * invd) ;
	}*/




    inline T kernelFunction(const Eigen::Ref< const Point >&  x) const
    {
        PointScalar d = x.norm();
        return (d == 0) ? 0 : T((1 / (4 * M_PI))) * exp(T(0,k) * d) / d;
    }

    template<typename TX>
    inline T CF(TX x) const
    {
	if constexpr(x.ColsAtCompileTime>1) {
	    const auto d2 = x.squaredNorm();

	    const auto invd=Eigen::rsqrt(d2.array());

	    const auto d=d2.array()*invd.array();
	    const PointScalar factor= (1.0/ (4.0 * M_PI));
	    return (Eigen::cos(k*d)+T(0,1)*Eigen::cos(k*d)) * invd *factor;
	}else
	{
	    
	    const auto d2 = x.squaredNorm();

	    const auto id=rsqrt(d2);
	    const auto d=d2*id;


	    return exp(T(0,k) * d)*id  * (1/(4.0 * M_PI));	
	}
    }

    

    template<typename TX, typename TY, typename TZ>
    inline void transfer_factor(TX x, TY xc, PointScalar H, TY pxc, PointScalar pH, TZ& result) const
    {
	const Eigen::Array<typename TX::Scalar, TX::ColsAtCompileTime, 1> d2=(x.matrix().colwise()-xc).colwise().squaredNorm().array();
	const Eigen::Array<typename TX::Scalar, TX::ColsAtCompileTime, 1> dp2=(x.matrix().colwise()-pxc).colwise().squaredNorm().array();

	/*const auto invd=Eigen::rsqrt(d2);

	const auto dp=Eigen::sqrt(dp2);
	const auto d=d2*invd;*/
	
	//PointScalar d = (x - xc).norm();
        ///Eigen::Array<typename TX::Scalar, TX::ColsAtCompileTime, 1> dp = (x.colwise() - pxc).norm();
	
        //result*= Eigen::exp( -k*(d-dp) )*(dp*invd);
	for (size_t i=0;i<x.cols();i++) {
	    const PointScalar d=(x.col(i).matrix()-xc).norm();
	    const PointScalar dp=(x.col(i).matrix()-pxc).norm();
	    
	    result(i)*=std::exp(T(0,k)*(d-dp))*dp/d;
	}
    }

    template<int TARGETS_AT_COMPILE_TIME>
    void evaluateKernel(const Eigen::Ref<const PointArray> &x, const Eigen::Ref<const PointArray> &y, const Eigen::Ref<const Eigen::Vector<T, Eigen::Dynamic> > &w,
                        Eigen::Ref<Eigen::Vector<T, TARGETS_AT_COMPILE_TIME> >  result,IndexRange srcsIds) const
    {
        assert(result.size() == y.cols());
        assert(w.size() == x.cols());

	
	for (int j = 0; j < y.cols(); j++) {
	    for (int i = 0; i < x.cols(); i++) {
	    //result+= w[i]* kernelFunction((- y).colwise()+x.col(i)).matrix();        
                result[j] += w[i] * kernelFunction(x.col(i) - y.col(j));
	    }
        }
    }



#if 0
    Eigen::Vector<T, Eigen::Dynamic>  evaluateFactoredKernel(const Eigen::Ref<const PointArray> &x, const Eigen::Ref<const PointArray> &y,
            const Eigen::Ref<const Eigen::Vector<T, Eigen::Dynamic> > &weights,
							     const Point& xc, PointScalar H, IndexRange srcsIds) const
    {

        Eigen::Vector<T, Eigen::Dynamic> result(y.cols());

	const int pkg_size=4;
	Eigen::Array<T, pkg_size,1> tmp;
	Eigen::Array<RealScalar, pkg_size,1> d;
        result.fill(0);        
	for (int j = 0; j < y.cols(); j++) {
            const RealScalar dc = (y.matrix().col(j) - xc).norm();

	    size_t i=0;
	    for (i = 0; i < x.cols()/ pkg_size; i++) {
		d=(x.middleCols( pkg_size*i, pkg_size).colwise()-y.col(j)).matrix().colwise().norm().array();
		tmp.real()=(Eigen::cos(k*(d-dc))*dc/d);
		tmp.imag()=Eigen::sin(k*(d-dc))*(dc/d);
		
                //const float d = (x.col(i) - y.col(j)).matrix().norm();
                result[j] +=  (weights.segment( pkg_size*i, pkg_size).matrix().transpose() * tmp.matrix()).value(); // *  std::complex<PointScalar>(  exp(std::complex<float>(0,(float) k) * (d - dc)) * (dc) / d);
	    }

	    for(size_t l=i* pkg_size;l<x.cols();l++)
	    {
		const RealScalar d = (x.col(l) - y.col(j)).matrix().norm();
                result[j] +=  weights[l] *  exp(T(0,k)* (d - dc)) * (dc) / d;
		
	    }
	}
        return result;
    }
#endif


        
    inline Eigen::Vector<int,dim> orderForBox(PointScalar H, Eigen::Vector<int,dim> baseOrder,int step=0) const
    {
	
	Eigen::Vector<int,dim> order=baseOrder;

	if(step==0) {
	    //order=(order.array()-3).cwiseMax(1);//baseOrder;
	    //order[1]/=2.0;
	    //order[2]/=2.0;
	    order=(order.array()-3).cwiseMax(1);//rder.fill(4);
	    //order[0]=4;
	    // order[1]=4;
	    //order[2]=4;
	    
	    
	}
	
        return order;
    }

    inline  Eigen::Vector<size_t,dim>  elementsForBox(PointScalar H, Eigen::Vector<int,dim> baseOrder,Eigen::Vector<size_t,dim> base, int step=0) const
    {
	const auto orders=orderForBox(H,baseOrder,step);
	Eigen::Vector<size_t,dim> els;

	const double sizes[]={1., 2,4};
	if(step==0){
	    base*=4;
	    //base[1]*=2;
	    //base[2]*=4;
	    
	    
	    //base[2]*=2;
	}
	    
	for(int i=0;i<dim;i++) {
	    //int delta=std::ceil(std::max( std::abs(k.imag())*H/(2*(2+k.real())) , 1.0)); //make sure that k H is bounded
	    PointScalar delta=std::max( k*H/2.,1.);
	    

	    
	    els[i]=base[i]*delta;//std::max((size_t) ceil((k*H*sizes[i])*(2.0/baseOrder[i])),(size_t ) round(2*H));//std::max(base[i]*((int) ceil(delta)),(size_t) 1);

	    //std::cout<<"els"<<i<<" "<<els[i]<<" "<<orders[i]<<std::endl;
	}
	    
	return els;	    
    }



private:
    PointScalar k;

};

#endif


