#ifndef __DL_HELMHOLTZ_IFGF_HPP__
#define __DL_HELMHOLTZ_IFGF_HPP__

#include "ifgfoperator.hpp"
#include "util.hpp"

#define HIGH_EXP_CUTOFF 50  //constant where exp(-x) is considered zero  to avoid NaNs/denormalized numbers

// Switch to either compute kernels in RealScalar or double precision
// Allows to keep PointScalar double while computing expensive Kernels in FP32
// Frequent sqrt computation is very expensive on A40 in FP64

#ifdef KERNEL_HIGH_PRECISION
    typedef double KernelComputeScalar;
#else
    typedef RealScalar KernelComputeScalar;
#endif

class DoubleLayerHelmholtzKernelFunctions
{
    typedef std::complex<RealScalar> T;
    typedef KernelComputeScalar Kp;
    typedef std::complex<Kp>    Tk;
    const static  int dim=3;
    typedef Eigen::Array<PointScalar, dim, Eigen::Dynamic> PointArray;
    typedef Eigen::Vector<PointScalar,dim> Point;

    // 1/(4*pi), hardcoded as RealScalar, avoids frequent recomputation on GPU
    static constexpr Kp INV_4PI = Kp(0.07957747154594766788444188168625718L);

public:
    DoubleLayerHelmholtzKernelFunctions(std::complex<RealScalar> waveNr):
	k(waveNr)
    {
    }


    // Double layer kernel:
    //  -1/(4pi) * 1/d^2 * exp(-k d) * (-k - 1/d) * (x.n),  d = |x|
    inline T kernelFunction(const sycl::marray<Kp,3>& x, const sycl::marray<Kp,3>& n) const
    {
        const Kp d2 = x[0]*x[0]+x[1]*x[1]+x[2]*x[2];
        const Kp kr = Kp(k.real());
        const Kp ki = Kp(k.imag());
        if(sycl::fabs(d2)<=Kp(1e-24) || sycl::sqrt(d2)*kr > Kp(HIGH_EXP_CUTOFF)) {
            return RealScalar(0.0);
        }
        const Kp id = Kp(1.0)/sycl::sqrt(d2);
        const Kp d  = d2*id;

        const Kp xn = x[0]*n[0]+x[1]*n[1]+x[2]*n[2];

        const Tk emkd = Tk(sycl::exp(-kr*d)) * Tk(sycl::cos(ki*d),-sycl::sin(ki*d));
        const Tk val = -Kp(INV_4PI) * (id*id) * emkd * (Tk(-kr,-ki) - Tk(id)) * xn;
        return T(RealScalar(val.real()), RealScalar(val.imag()));
    }


    
    // ns is indexed like xs (global source index i), matching a_srcs / a_normals
    template <typename AT1,typename AT2,typename AT3,typename AT4>
    T evaluateKernel(const AT1& xs, size_t x0, size_t xend, const AT2& ys, size_t y0,
			     const AT3& ws, const AT4& ns)  const
    {
	T result=0;

	sycl::marray<Kp,3> pnt;
	sycl::marray<Kp,3> nrm;
	for (size_t i = x0; i < xend; i++) {
	    for(int l=0;l<dim;l++)  {
		pnt[l]=Kp(xs[i*dim+l])-Kp(ys[y0*dim+l]);
		nrm[l]=Kp(ns[i*dim+l]);
	    }
	    result += ws[i] * kernelFunction(pnt,nrm);	
        }
	return result;
    }



    // ns here is the SLM-staged normals array, indexed by the same local index as xs
    template <typename AT1,typename AT2,typename AT3>
    T  evaluateFactoredKernel(
			      const AT1& xs, size_t x0, size_t xend, const sycl::marray<PointScalar,dim>& y,
			      const AT2& ws, const AT3& ns, const sycl::marray<PointScalar,dim>& xc, PointScalar H) const
    {

	T result=0;

	const Kp dcx=Kp(y[0])-Kp(xc[0]);
	const Kp dcy=Kp(y[1])-Kp(xc[1]);
	const Kp dcz=Kp(y[2])-Kp(xc[2]);
	const Kp dc = sycl::sqrt(dcx*dcx+dcy*dcy+dcz*dcz);

	const Kp kr = Kp(k.real());
	const Kp ki = Kp(k.imag());

	for(size_t i=x0;i<xend;i++) {
	    const Kp px=Kp(xs[i*dim])   - Kp(y[0]);
	    const Kp py=Kp(xs[i*dim+1]) - Kp(y[1]);
	    const Kp pz=Kp(xs[i*dim+2]) - Kp(y[2]);
	    const Kp d2 = px*px+py*py+pz*pz;

	    const Kp id = (d2>Kp(1e-24)) ? Kp(1.0)/sycl::sqrt(d2) : Kp(0);
	    const Kp d  = d2*id;

	    const Kp xn = px*Kp(ns[i*dim]) + py*Kp(ns[i*dim+1]) + pz*Kp(ns[i*dim+2]);
	    const Kp w  = -(id*id) * dc * xn;

	    const Kp ddc = d-dc;
	    const Tk emkddc = Tk(sycl::exp(-kr*ddc)) * Tk(sycl::cos(ki*ddc),-sycl::sin(ki*ddc));
	    const Tk val = emkddc * (Tk(-kr,-ki) - Tk(id)) * w;
	    result += ws[i] * T(RealScalar(val.real()), RealScalar(val.imag()));
	}
	return result;
    }


    template<typename TX>
    inline T CF(TX x) const
    {
	const Kp d2 = Kp(x[0])*Kp(x[0])+Kp(x[1])*Kp(x[1])+Kp(x[2])*Kp(x[2]);

	if(sycl::fabs(d2)<Kp(1e-14)) {
	    return 0;
	}

	const Kp id=Kp(1)/(sycl::sqrt(d2));
	const Kp d=d2*id;
	const Kp kr = Kp(k.real());
	const Kp ki = Kp(k.imag());

	const Tk val = Tk(sycl::exp(-kr*d))*Tk(sycl::cos(ki*d),-sycl::sin(ki*d))*id*Kp(INV_4PI);
	return T(RealScalar(val.real()), RealScalar(val.imag()));
    }

    
    template<typename TX , typename TY>
    inline T transfer_factor(TX x, TY xc, PointScalar H, TY pxc, PointScalar pH) const
    {
	const Kp zx=Kp(x[0])-Kp(xc[0]);
	const Kp zy=Kp(x[1])-Kp(xc[1]);
	const Kp zz=Kp(x[2])-Kp(xc[2]);
	const Kp zpx=Kp(x[0])-Kp(pxc[0]);
	const Kp zpy=Kp(x[1])-Kp(pxc[1]);
	const Kp zpz=Kp(x[2])-Kp(pxc[2]);

	const Kp d  = sycl::sqrt(zx*zx+zy*zy+zz*zz);
	const Kp dp = sycl::sqrt(zpx*zpx+zpy*zpy+zpz*zpz);

	if(sycl::fabs(d)<Kp(1e-15) ) {
	    return 0;
	}

	const Kp kr = Kp(k.real());
	const Kp ki = Kp(k.imag());
	const Kp ddp = d-dp;

	const Tk val = Tk(sycl::exp(-kr*ddp))*Tk(sycl::cos(ki*ddp),-sycl::sin(ki*ddp))*(dp/d);
	return T(RealScalar(val.real()), RealScalar(val.imag()));
    }





private:
    std::complex<Kp> k;
};



template<size_t dim >
class DoubleLayerHelmholtzIfgfOperator : public IfgfOperator<std::complex<RealScalar>, dim,
							  1, DoubleLayerHelmholtzIfgfOperator<dim> >
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
    //this kernel needs the per-source normals, so the base binds a normals buffer
    static constexpr bool HAS_NORMALS = true;

    typedef Eigen::Array<PointScalar, dim, Eigen::Dynamic> PointArray;
    typedef Eigen::Vector<PointScalar,dim> Point;
    DoubleLayerHelmholtzIfgfOperator(std::complex<RealScalar> waveNumber,
				  size_t leafSize,
				  size_t order,
				  size_t n_elem=1,PointScalar tol=-1,RealScalar p_maxk=-1,RealScalar p_minSigma=-1):
        IfgfOperator<std::complex<RealScalar>, dim, 1, DoubleLayerHelmholtzIfgfOperator<dim> >(leafSize,order, n_elem,tol),
        k(waveNumber),
	maxk(p_maxk),
        minSigma(p_minSigma)
    {
	if(minSigma<0) {
            minSigma=k.real();
        }

        std::cout<<"minSigma="<<minSigma<<std::endl;
        if(maxk<0) {
	    maxk=0.136 * std::abs(k.imag())/std::max((RealScalar) 1.0,k.real());
            std::cout<<"maxk="<<maxk<<std::endl;
	}


    }

    ~DoubleLayerHelmholtzIfgfOperator()
    {
	std::cout<<"deleting  DL helmholtz ifgf"<<std::endl;
    }

    typedef std::complex<RealScalar > T ;
    
    void init(const PointArray &srcs, const PointArray targets, const PointArray& normals)
    {
	std::cout<<"dlinit"<<std::endl;
	m_normals=normals;

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
	    


	IfgfOperator<T,dim,1, DoubleLayerHelmholtzIfgfOperator<dim> >::init(srcs,targets);

#ifdef CACHE_OCTREE
	OctreeCache<T,dim, OctreeKeyType>::getInstance().add(key,this->m_octree);
#endif
	
    }

    //once the octree is ready, we can reorder the normals such that the morton-order is observed
    void onOctreeReady()
    {
	PointArray sorted(dim, m_normals.cols());
	Util::copy_with_permutation_colwise<PointScalar,dim>(m_normals, this->m_octree->srcPermutation(), sorted);
	m_normals=sorted;
    }

    //the base uses this (only when HAS_NORMALS) to build the device normals buffer
    const PointScalar* sourceNormalsData() const { return m_normals.data(); }




    inline DoubleLayerHelmholtzKernelFunctions kernelFunctions() const {
	DoubleLayerHelmholtzKernelFunctions f(k);
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
    PointArray m_normals;

};

#endif