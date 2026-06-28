#ifndef __CF_HELMHOLTZ_IFGF_HPP__
#define __CF_HELMHOLTZ_IFGF_HPP__

#include "ifgfoperator.hpp"
#include "util.hpp"


template<size_t dim >
class CombinedFieldHelmholtzIfgfOperator : public IfgfOperator<std::complex<PointScalar>, dim,
							     1, CombinedFieldHelmholtzIfgfOperator<dim> >
{
public:
    typedef Eigen::Array<PointScalar, dim, Eigen::Dynamic> PointArray;
    typedef Eigen::Vector<PointScalar,dim> Point;
    CombinedFieldHelmholtzIfgfOperator(PointScalar waveNumber,
				     size_t leafSize,
				     size_t order,
				     size_t n_elem=1,PointScalar tolerance=-1):
        IfgfOperator<std::complex<PointScalar>, dim, 1, CombinedFieldHelmholtzIfgfOperator<dim> >(leafSize,order, n_elem,tolerance),
        kappa(waveNumber)
    {
    }


    void init(const PointArray &srcs, const PointArray targets, const PointArray& normals)
    {
	m_normals=normals;
	IfgfOperator<T,dim,1,CombinedFieldHelmholtzIfgfOperator<dim> >::init(srcs,targets);
	
    }

    //once the octree is ready, we can reorder it such that the morton-order is observed
    void onOctreeReady()
    {
	m_normals=Util::copy_with_permutation(m_normals, this->src_octree().permutation());
    }
   
    
    typedef std::complex<PointScalar > T ;

  /** CombinedFieldKernel in 3D reads
      $$ G(x-y) = \frac{1}{4\,\pi} \, \frac{e^{i\,\kappa\,|x-y|}}{|x-y|^3} \, 
          \left( \langle n_y, x-y\rangle (1- i\,\kappa\, | x-y|) - i\,\kappa\,|x-y|^2 \right), 
          \quad x, y \in \mathbb R^3, \; x\not=y\,. $$ */
    inline T  kernelFunction(const Eigen::Ref< const Point >&  x,const Eigen::Ref< const Point >&  n) const
    {
	const PointScalar d2 = x.squaredNorm();
        const PointScalar invd = d2<1e-14 ?  0.0 : (1.0/sqrt(d2));
        const PointScalar d=d2*invd;
        

        PointScalar nxy = -n.dot(x);
        
        const PointScalar s=sin(kappa*(d));
        const PointScalar c=cos(kappa*(d));


        
        const PointScalar f=(1.0/(4.*M_PI))*invd*invd*invd;//*d2*d);
                

        PointScalar real=f*((c)*nxy + (s)*kappa*(d*nxy+d2));
        PointScalar imag=f*(-(c)*kappa*(d*nxy+d2) + (s)*nxy);

        return T(real,imag);

	/*PointScalar nxy = -n.dot(x);
	auto kern = exp(T(0,kappa)*d) / (4 * M_PI * d2*d)
	    * ( nxy * (std::complex<PointScalar>(1.,0)*T(1.) - std::complex<PointScalar>(0,kappa)*d)  - T(0,kappa)*d2);
	
            return d<1e-14 ? 0.0 : kern;*/
	    
    }

    template<typename TX>
    inline T CF(TX x) const
    {
	if constexpr(x.ColsAtCompileTime>1) {
	    const auto d2 = x.squaredNorm();

	    const auto invd=Eigen::rsqrt(d2.array());

	    const auto d=d2.array()*invd.array();
	    const PointScalar factor= (1.0/ (4.0 * M_PI));
	    return Eigen::exp(std::complex(0.,kappa) * d) * invd *factor;
	}else
	{
	    
	    const auto d2 = x.squaredNorm();

	    const auto id=1.0/(sqrt(d2));
	    const auto d=d2*id;

	    return exp(std::complex(0.,kappa) * d)*id  * (1/(4.0 * M_PI));	    
	}
    }


    

    template<typename TX, typename TY, typename TZ>
    inline void transfer_factor(TX x, TY xc, PointScalar H, TY pxc, PointScalar pH, TZ& result) const
    {
	const Eigen::Array<typename TX::Scalar, TX::ColsAtCompileTime, 1> d2=(x.matrix().colwise()-xc).colwise().squaredNorm().array();
	const Eigen::Array<typename TX::Scalar, TX::ColsAtCompileTime, 1> dp2=(x.matrix().colwise()-pxc).colwise().squaredNorm().array();

	const auto invd=Eigen::rsqrt(d2);

	const auto dp=Eigen::sqrt(dp2);
	const auto d=d2*invd;
	      
	result*=(Eigen::exp( std::complex(0.,kappa)*(d-dp) )*(dp*invd));
	
    }

    
    template<int TARGETS_AT_COMPILE_TIME>
    void evaluateKernel(const Eigen::Ref<const PointArray> &x, const Eigen::Ref<const PointArray> &y, const Eigen::Ref<const Eigen::Vector<T, Eigen::Dynamic> > &w,
                        Eigen::Ref<Eigen::Array<T, TARGETS_AT_COMPILE_TIME,1> >  result,IndexRange srcIds) const
    {
        assert(result.size() == y.cols());
        assert(w.size() == x.cols());

        for (int i = 0; i < x.cols(); i++) {
            //result+= w[i]* kernelFunction((- y).colwise()+x.col(i)).matrix();
            for (int j = 0; j < y.cols(); j++) {
                result[j] += w[i] * kernelFunction(x.col(i) - y.col(j),m_normals.col(srcIds.first+i));
            }
        }
            
		
    }


    

    Eigen::Array<T, Eigen::Dynamic,1>  evaluateFactoredKernel(const Eigen::Ref<const PointArray> &x, const Eigen::Ref<const PointArray> &y,
							      const Eigen::Ref<const Eigen::Vector<T, Eigen::Dynamic> > &weights,
							      const Point& xc, PointScalar H,IndexRange srcIds) const
    {
	Eigen::Array<T, Eigen::Dynamic,1> result(y.cols());


        const auto w_r=weights.real().eval();
        const auto w_i=weights.imag().eval();
            
        result.fill(0);
	for (int j = 0; j < y.cols(); j++) {
            const PointScalar dc = (y.matrix().col(j) - xc).norm();


            for (int i = 0; i < x.cols(); i++) {
                
                const Point& z=y.matrix().col(j)-x.matrix().col(i);

                const PointScalar d2 = z.squaredNorm();
		if(d2>1e-14 ){

		    const PointScalar nxy=z.dot(m_normals.col(srcIds.first+i).matrix());
                
		    const PointScalar id= 1.0/sqrt(d2);
		    const PointScalar d=d2*id;
                
		    const PointScalar s=sin(kappa*(d-dc));
		    const PointScalar c=cos(kappa*(d-dc));


		    const PointScalar f=dc/(d2*d);


		    result.row(j).real()+=f*((w_r[i]*c-w_i[i]*s)*nxy + (w_r[i]*s+w_i[i]*c)*kappa*(d*nxy+d2));
		    result.row(j).imag()+=f*(-(w_r[i]*c-w_i[i]*s)*kappa*(d*nxy+d2) + (w_r[i]*s+w_i[i]*c)*nxy);
		}

                
                
                //result.row(j) += weights[i]* ( exp(T(0, kappa) * (d - dc)) * dc / (d2*d)
                //                              * ( nxy * (std::complex<PointScalar>(1.,0)*T(1.)
                //                              - std::complex<PointScalar>(0,kappa)*d)  - T(0,kappa)*d*d));
            }
        
        }
        
        return result;        
    }
 



    
        
    inline Eigen::Vector<int,dim> orderForBox(PointScalar H, Eigen::Vector<int,dim> baseOrder,int step=0) const
    {
	
	Eigen::Vector<int,dim> order=baseOrder;

	if(step==0) {
	    order=baseOrder.array()-3;//(baseOrder.array().template cast<PointScalar>()*Eigen::log(4./baseOrder.array().template cast<PointScalar>())).template cast<int>();
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
	    PointScalar delta=std::max( kappa*H/4.,1.);
	    

	    els[i]=std::max(base[i]*((int) ceil(delta)),(size_t) 1);	    
	}
	    
	return els;	    
    }



private:
    PointScalar kappa;
    PointArray m_normals;
};

#endif
