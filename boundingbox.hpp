#ifndef __BBOX_HPP_
#define __BBOX_HPP_

#include <Eigen/Dense>

#include <Eigen/Geometry>

template <size_t DIM>
class BoundingBox : public Eigen::AlignedBox<PointScalar, DIM>
{
public:
    BoundingBox()
    {

    }

    BoundingBox(Eigen::Vector<PointScalar, DIM> min, Eigen::Vector<PointScalar, DIM> max):
        Eigen::AlignedBox<PointScalar, DIM>(min, max)
    {

	m_center=(max + min)/2.0;

	auto diag = this->diagonal();
        PointScalar m = 0;
        for (unsigned int i = 0; i < DIM; i++) {
            m = std::max(m, std::abs(diag[i]));
        }
        m_sideLength=m;
	
    }

    ~BoundingBox()
    {

    }

    inline const Eigen::Vector<PointScalar, DIM>& center() const
    {
        return  m_center;
    }

    inline PointScalar sideLength() const
    {
	return m_sideLength;
    }

    inline PointScalar distanceToBoundary( Eigen::Vector<PointScalar,DIM> p) const
    {
	PointScalar mD=0;
	unsigned int n_corners=1 << DIM;
	for (int j=0;j<n_corners;j++)
	{
	    Eigen::Vector<PointScalar,DIM> vertex;
	    for(int l=0;l<DIM;l++){
		vertex[l]= (j & 1 << l) == 0 ?  this->min()[l] : this->max()[l];
	    }
	    PointScalar d=(p-vertex).norm();
	    mD=std::max(mD,d);
	}
	return mD;
    }


private:
    Eigen::Vector<PointScalar,DIM> m_center;
    PointScalar m_sideLength;
    
    /*

    inline void absorb(const BoundingBox& other)
    {
    m_min=m_min.min(other.min());
    m_max=m_max.max(other.max());
    }

    Eigen::Vector<PointScalar,DIM> min() const
    {
    return m_min;
    }

    Eigen::Vector<PointScalar,DIM> max() const
    {
    return m_max;
    }


    inline PointScalar dist(const BoundingBox& other) const
    {
    //TODO

    return 0;
    }

    inline PointScalar dist(const Eigen::Vector<PointScalar,DIM>& x) const
    {
    //TODO
    return 0;
    }


    bool contains(const Eigen::Vector<PointScalar,DIM>& x)  const
    {
    return x>=m_min && x<=m_max;
    }




    private:
    Eigen::Vector<PointScalar,DIM> m_min;
    Eigen::Vector<PointScalar,DIM> m_max;
    */
};

template<size_t DIM>
std::ostream & operator<<(std::ostream &os, const BoundingBox<DIM>& p)
{
    return os << p.min().transpose()<<" "<<p.max().transpose();
}

#endif
