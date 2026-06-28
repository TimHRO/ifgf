#ifndef __MOD_HELMHOLTZ_IFGF_HPP__
#define __MOD_HELMHOLTZ_IFGF_HPP__

#include "ifgfoperator.hpp"

#define HIGH_EXP_CUTOFF 50  //constant where exp(-x) is considered zero  to avoid NaNs/denormalized numbers

class ModifiedHelmholtzKernelFunctions
{
    typedef std::complex<RealScalar> T;
    const static  int dim=3;
    typedef Eigen::Array<PointScalar, dim, Eigen::Dynamic> PointArray;
    typedef Eigen::Vector<PointScalar,dim> Point;

    // 1/(4*pi), hardcoded as RealScalar, avoids frequent recomputation on GPU
    static constexpr RealScalar INV_4PI = RealScalar(0.07957747154594766788444188168625718L);

public:
    ModifiedHelmholtzKernelFunctions(std::complex<RealScalar> waveNr):
	k(waveNr)
    {
    }


    inline T kernelFunction(const sycl::marray<PointScalar,3>& x) const
    {
        RealScalar d = sycl::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
        const RealScalar kr = k.real();
        const RealScalar ki = k.imag();
       if(std::abs(d)<=RealScalar(1e-15)  || d*kr > RealScalar(HIGH_EXP_CUTOFF)) {
            return RealScalar(0.0);
        }
	return INV_4PI * T(sycl::exp(-kr*d))* T(sycl::cos(ki*d),-sycl::sin(ki*d)) / (d);
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
	RealScalar dc = sycl::sqrt(pnt[0]*pnt[0]+pnt[1]*pnt[1]+pnt[2]*pnt[2]);

	const RealScalar kr = k.real();
	const RealScalar ki = k.imag();

	for(size_t i=x0;i<xend;i++) {
	    sycl::marray<PointScalar,3> p{xs[i*dim]-y[0],xs[i*dim+1]-y[1],xs[i*dim+2]-y[2]};
	    RealScalar d = sycl::sqrt(p[0]*p[0]+p[1]*p[1]+p[2]*p[2]);

	    result += (abs(d)<RealScalar(1e-15) ) ? T(RealScalar(0)) :  ws[i] *  T(sycl::exp(-kr*(d-dc)))*T(sycl::cos(ki*(d-dc)),-sycl::sin(ki*(d-dc))) * (dc) / d;
	}
	return result;
    }


    template<typename TX>
    inline T CF(TX x) const
    {
	const RealScalar d2 = x[0]*x[0]+x[1]*x[1]+x[2]*x[2];

	if(abs(d2)<RealScalar(1e-14)) {
	    return 0;
	}


	const RealScalar id=RealScalar(1)/(sycl::sqrt(d2));
	const RealScalar d=d2*id;
	const RealScalar kr = k.real();
	const RealScalar ki = k.imag();

        
        /*if(d*k.real()>HIGH_EXP_CUTOFF)
        {
            return 0.0;
        }*/

	return T(sycl::exp(-kr*d))*T(sycl::cos(ki*d),-sycl::sin(ki*d))*id  *INV_4PI;

    }

    
    template<typename TX , typename TY>
    inline T transfer_factor(TX x, TY xc, PointScalar H, TY pxc, PointScalar pH) const
    {
	auto z=x-xc;
	auto zp=x-pxc;
	const RealScalar d = sycl::sqrt(z[0]*z[0]+z[1]*z[1]+z[2]*z[2]);
	const RealScalar dp = sycl::sqrt(zp[0]*zp[0]+zp[1]*zp[1]+zp[2]*zp[2]);

	if(abs(d)<RealScalar(1e-15) ) {
	    return 0;
	}
        /*if((d-dp)*k.imag() <- HIGH_EXP_CUTOFF) { //truncate the transfer factor at around 10^16
    	        return T(exp(HIGH_EXP_CUTOFF))*T(sycl::cos(k.imag()*(d-dp)),-sycl::sin(k.imag()*(d-dp)))*dp/d;
        }*/

	const RealScalar kr = k.real();
	const RealScalar ki = k.imag();
	return T(sycl::exp(-kr*(d-dp)))*T(sycl::cos(ki*(d-dp)),-sycl::sin(ki*(d-dp)))*dp/d;
	
    }





private:
    T k;
};



template<size_t dim >
class ModifiedHelmholtzIfgfOperator : public IfgfOperator<std::complex<RealScalar>, dim,
							  1, ModifiedHelmholtzIfgfOperator<dim> >
{
private:
    struct OctreeKeyType {
	RealScalar maxk;
        RealScalar minSigma;
	size_t Ndof;
	size_t Ndof2;
	auto operator==(const OctreeKeyType& other) const
	{
	    return std::abs(maxk-other.maxk)<1e-12 && Ndof==other.Ndof && Ndof2==other.Ndof2 && std::abs(minSigma-other.minSigma)<1e-12; 
	}
    };

public:
    typedef Eigen::Array<PointScalar, dim, Eigen::Dynamic> PointArray;
    typedef Eigen::Vector<PointScalar,dim> Point;
    ModifiedHelmholtzIfgfOperator(std::complex<RealScalar> waveNumber,
				  size_t leafSize,
				  size_t order,
				  size_t n_elem=1,PointScalar tol=-1,RealScalar p_maxk=-1,RealScalar p_minSigma=-1):
        IfgfOperator<std::complex<RealScalar>, dim, 1, ModifiedHelmholtzIfgfOperator<dim> >(leafSize,order, n_elem,tol),
        k(waveNumber),
	maxk(p_maxk),
        minSigma(p_minSigma)
    {
	if(minSigma<0) {
            minSigma=k.real();
        }

        std::cout<<"minSigma="<<minSigma<<std::endl;
        if(maxk<0) {
	    maxk=0.5 * std::abs(k.imag())/std::max((RealScalar) 1.0,k.real());
            std::cout<<"maxk="<<maxk<<std::endl;
	}


    }

    typedef std::complex<RealScalar > T ;
    
    void init(const PointArray &srcs, const PointArray targets)
    {
	std::cout<<"modinit"<<std::endl;
	OctreeKeyType key;
	key.maxk=maxk;
	key.Ndof=srcs.cols();
	key.Ndof2=targets.cols();
        key.minSigma=minSigma;

	auto oct=OctreeCache<T,dim, OctreeKeyType>::getInstance().find(key);
	if(oct) {
	    std::cout<<"using cached octree="<<oct<<std::endl;
	    this->m_octree=oct;
	}
	    


	IfgfOperator<T,dim,1, ModifiedHelmholtzIfgfOperator<dim> >::init(srcs,targets);

#ifdef CACHE_OCTREE
	OctreeCache<T,dim, OctreeKeyType>::getInstance().add(key,this->m_octree);
#endif
	
    }




    inline ModifiedHelmholtzKernelFunctions kernelFunctions() const {
	ModifiedHelmholtzKernelFunctions f(k);
	return f; 
    }


        
    inline Eigen::Vector<int,dim> orderForBox(PointScalar H, Eigen::Vector<int,dim> baseOrder,int step=0) const
    {
	Eigen::Vector<int,dim> order=baseOrder;

        if(false){//0.75*(sqrt(dim)/dim)*H*minSigma> HIGH_EXP_CUTOFF) {
            std::cout<<"cutoff"<<H<<" "<<k.real()<<" "<<H*k.real()<<"\n";
            order.fill(0);

            return order;
        }



	if(step==0) {
	    order=(baseOrder.array()-3).cwiseMax(2);//(baseOrder.array().template cast<PointScalar>()*Eigen::log(4./baseOrder.array().template cast<PointScalar>())).template cast<int>();
	}
	
        return order;
    }

    double cutoff_limit(double H) {
        double rmax=3*HIGH_EXP_CUTOFF/std::max((RealScalar )1.,minSigma);
        double smin=1e-4;//0.5*std::max(H/rmax,1e-4);
        std::cout<<"smin="<<smin<<std::endl;
        return std::min(smin,sqrt(dim)/dim);
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
            PointScalar delta=std::max((PointScalar) maxk*H, (PointScalar)1.);
	    els[i]=std::max(base[i]*((int) ceil(delta)),(size_t) 1);	    
	}
	    
	return els;	    
    }


    bool farfieldCanBeSkipped(PointScalar H) {
        return false;//(sqrt(dim)/dim)*H*k.real()> HIGH_EXP_CUTOFF;
    }



private:
    std::complex<RealScalar> k;
    RealScalar maxk;
    RealScalar minSigma;

};

#endif
