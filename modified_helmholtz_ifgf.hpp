#ifndef __MOD_HELMHOLTZ_IFGF_HPP__
#define __MOD_HELMHOLTZ_IFGF_HPP__

#include "ifgfoperator.hpp"



class ModifiedHelmholtzKernelFunctions
{
    typedef std::complex<RealScalar> T;
    const static  int dim=3;
    typedef Eigen::Array<PointScalar, dim, Eigen::Dynamic> PointArray;
    typedef Eigen::Vector<PointScalar,dim> Point;

public:
    ModifiedHelmholtzKernelFunctions(std::complex<RealScalar> waveNr):
	k(waveNr)
    {
    }


    inline T kernelFunction(const sycl::marray<PointScalar,3>& x) const
    {
        RealScalar d = sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
	return std::abs(d)==0.0 ? RealScalar(0) : RealScalar((1. / (4. * M_PI))) * T(exp(-k.real()*d))* T(cos(k.imag()*d),-sin(k.imag()*d)) / (d);
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

	    result += (abs(d)<1e-12) ? RealScalar(0) :  ws[i] *  T(exp(-k.real()*(d-dc)))*T(sycl::cos(k.imag()*(d-dc)),-sycl::sin(k.imag()*(d-dc))) * (dc) / d;
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
	const RealScalar id=1./(sqrt(d2));
	const RealScalar d=d2*id;

	
	return T(sycl::exp(-k.real()*d))*T(sycl::cos(k.imag()*d),-sycl::sin(k.imag()*d))*id  *RealScalar(1./(4.0 * M_PI));	    

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

	return T(exp(-k.real()*(d-dp)))*T(sycl::cos(k.imag()*(d-dp)),-sycl::sin(k.imag()*(d-dp)))*dp/d;
	
    }





private:
    T k;
};



template<size_t dim >
class ModifiedHelmholtzIfgfOperator : public IfgfOperator<std::complex<RealScalar>, dim,
							  1, ModifiedHelmholtzIfgfOperator<dim> >
{
public:
    typedef Eigen::Array<PointScalar, dim, Eigen::Dynamic> PointArray;
    typedef Eigen::Vector<PointScalar,dim> Point;
    ModifiedHelmholtzIfgfOperator(std::complex<RealScalar> waveNumber,
				  size_t leafSize,
				  size_t order,
				  size_t n_elem=1,PointScalar tol=-1,double p_maxk=-1):
        IfgfOperator<std::complex<RealScalar>, dim, 1, ModifiedHelmholtzIfgfOperator<dim> >(leafSize,order, n_elem,tol),
        k(waveNumber),
	maxk(p_maxk)
    {
	if(maxk<0) {
	    maxk=std::abs(k.imag())/(2*(2+k.real()));
	}
    }

    typedef std::complex<PointScalar > T ;


    inline ModifiedHelmholtzKernelFunctions kernelFunctions() const {
	ModifiedHelmholtzKernelFunctions f(k);
	return f; 
    }


        
    inline Eigen::Vector<int,dim> orderForBox(PointScalar H, Eigen::Vector<int,dim> baseOrder,int step=0) const
    {
	
	Eigen::Vector<int,dim> order=baseOrder;

	if(step==0) {
	    order=(baseOrder.array()-3).cwiseMax(2);//(baseOrder.array().template cast<PointScalar>()*Eigen::log(4./baseOrder.array().template cast<PointScalar>())).template cast<int>();
	}
	
        return order;
    }

    inline  Eigen::Vector<size_t,dim>  elementsForBox(PointScalar H, Eigen::Vector<int,dim> baseOrder,Eigen::Vector<size_t,dim> base, int step=0) const
    {
	const auto orders=orderForBox(H,baseOrder,step);
	Eigen::Vector<size_t,dim> els;

	if(step==0){
	    base*=3;
	    //base[2]*=2;
	}
	    
	for(int i=0;i<dim;i++) {
	    //int delta=std::ceil(std::max( std::abs(k.imag())*H/(2*(2+k.real())) , 1.0)); //make sure that k H is bounded	    
	    PointScalar delta=std::max( maxk *H/4 ,1.0);
	    

	    els[i]=std::max(base[i]*((int) ceil(delta)),(size_t) 1);	    
	}
	    
	return els;	    
    }



private:
    std::complex<RealScalar> k;
    double maxk;

};

#endif


