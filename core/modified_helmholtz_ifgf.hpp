#ifndef __MOD_HELMHOLTZ_IFGF_HPP__
#define __MOD_HELMHOLTZ_IFGF_HPP__

#include "config.hpp"
#include "ifgfoperator.hpp"

#define HIGH_EXP_CUTOFF 50  //constant where exp(-x) is considered zero  to avoid NaNs/denormalized numbers

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
       if(std::abs(d)<=1e-15  || d*k.real() > HIGH_EXP_CUTOFF) {
            return RealScalar(0.0);
        }
	return RealScalar((1. / (4. * M_PI))) * T(exp(-k.real()*d))* T(cos(k.imag()*d),-sin(k.imag()*d)) / (d);
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
	    //-dc

	    assert((d-dc+sqrt(dim)*H)>=0);
	    result += (abs(d)<1e-15 ) ? RealScalar(0) :  ws[i] *  T(exp(-k.real()*(d-dc+sqrt(dim)*H)))*T(sycl::cos(k.imag()*(d-dc)),-sycl::sin(k.imag()*(d-dc))) * (dc) / d;
	    //result+=(d<1e-12) ? 0 :   (ws[i] * (sycl::cos(k*(d-dc))+T(0,1)*sycl::sin(k*(d-dc)))*(dc/(d)));
	}
	return result;
    }


    template<typename TX>
    inline T CF(TX x,PointScalar H) const
    {
	const RealScalar d2 = x[0]*x[0]+x[1]*x[1]+x[2]*x[2];

	if(abs(d2)<1e-14) {
	    return 0;
	}


	const RealScalar id=1./(sqrt(d2));
	const RealScalar d=d2*id;

        
        /*if(d*k.real()>HIGH_EXP_CUTOFF)
        {
            return 0.0;
        }*/

	//T(sycl::exp(-k.real()*d))*
	assert((d-sqrt(dim)*H)>=0);
	return T(sycl::exp(-k.real()*(d-sqrt(dim)*H)))*T(sycl::cos(k.imag()*d),-sycl::sin(k.imag()*d))*id  *RealScalar(1./(4.0 * M_PI));	    

    }

    
    template<typename TX , typename TY>
    inline T transfer_factor(TX x, TY xc, PointScalar H, TY pxc, PointScalar pH) const
    {
	auto z=x-xc;
	auto zp=x-pxc;
	const RealScalar d = sqrt(z[0]*z[0]+z[1]*z[1]+z[2]*z[2]);
	const RealScalar dp = sqrt(zp[0]*zp[0]+zp[1]*zp[1]+zp[2]*zp[2]);

	if(abs(d)<1e-15 ) {
	    return 0;
	}
        /*if((d-dp)*k.imag() <- HIGH_EXP_CUTOFF) { //truncate the transfer factor at around 10^16
    	        return T(exp(HIGH_EXP_CUTOFF))*T(sycl::cos(k.imag()*(d-dp)),-sycl::sin(k.imag()*(d-dp)))*dp/d;
        }*/

	assert((d-dp-sqrt(dim)*(H-pH))>=0);
	
	//T(exp(-k.real()*(d-dp)))*	
	return T(exp(-k.real()*(d-dp-sqrt(dim)*(H-pH))))*T(sycl::cos(k.imag()*(d-dp)),-sycl::sin(k.imag()*(d-dp)))*dp/d;
	
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
	    maxk=std::abs(k.imag()/std::max(1.0,0.75*k.real()));///(1+k.real())///std::max((RealScalar) 1.0,k.real());
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

    inline PointScalar cutoff_limit(PointScalar H,Eigen::Vector<int,dim> baseOrder) const {
        double rmax=6*baseOrder.minCoeff()/minSigma;
        double smin=std::max(H/rmax,1e-4);
        //std::cout<<"smin="<<smin<<std::endl;
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

	PointScalar delta=std::max( maxk*H, (PointScalar) 1.0);
	PointScalar delta0=delta*(sqrt(dim)/dim-cutoff_limit(H,baseOrder));
	els[0]=std::max(base[0]*((int) ceil(delta0)),(size_t) 1);	    //the first element might already be small
	for(int i=1;i<dim;i++) {
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


