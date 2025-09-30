#ifndef __SYCL_HELPERS__
#define __SYCL_HELPERS__


#include "config.hpp"

#include <sycl/sycl.hpp>


// template <typename T, int ROWS, int COLS>
// class SyclRowMatrix : public std::array<sycl::marray<T, COLS>, ROWS>
// {
// public:
//     SyclRowMatrix()
//     {
// #ifdef INITIALIZE_BY_ZERO	
// 	for(int i=0;i<ROWS;i++) {
// 	    for(int j=0;j<COLS;j++){
// 		row(i)[j]=0;
// 	    }
// 	}
// #endif

//     }


//     SyclRowMatrix(const sycl::marray<T,ROWS>& data)
//     {
// 	static_assert(COLS==1);
// 	for(int i=0;i<ROWS;i++) {
// 	    (*this)[i][0]=data[i];
// 	}
//     }

//     SyclRowMatrix(std::array<sycl::marray<T,COLS>,ROWS> & data)
//     {
// 	for(int i=0;i<ROWS;i++) {
// 	    row(i)=data[i];
// 	}
//     }

//     const T& operator()(int i,int j)  const {
// 	return (*this)[i][j];
//     }

//     T& operator()(int i,int j) {
// 	return (*this)[i][j];
//     }


//     const sycl::marray<T,COLS>& row(int i) const {
// 	return (*this)[i];
//     }

//     sycl::marray<T,COLS>& row(int i) {
// 	return (*this)[i];
//     }
// };



template <typename T, int ROWS, int COLS>
class SyclRowMatrix : public sycl::marray<T, ROWS*COLS>
{
public:
    SyclRowMatrix()
    {
#ifdef INITIALIZE_BY_ZERO
	for(int j=0;j<COLS;j++){
	    for(int i=0;i<ROWS;i++) {		
		(*this)[j*ROWS+i]=0;
	    }
	}
#endif

    }


    SyclRowMatrix(const sycl::marray<T,ROWS>& data)
    {
	static_assert(COLS==1);
	for(int i=0;i<ROWS;i++) {
	    (*this)[i]=data[i];
	}
    }

    SyclRowMatrix(std::array<sycl::marray<T,COLS>,ROWS> & data)
    {
	for(int j=0;j<COLS;j++) {
	    for(int i=0;i<ROWS;i++) {
		(*this)[j*ROWS+i]=data[i][j];
	    }
	}
    }

    inline const T& operator()(int i,int j)  const {
	return (*this)[j*ROWS+i];
    }

    inline T& operator()(int i,int j) {
	return (*this)[j*ROWS+i];
    }


    inline const sycl::marray<T,COLS> row(int i) const {
	if constexpr(COLS==1) {
	    return (*this)[i];
	}
	
	sycl::marray<T,COLS> row;
	for(int j=0;j<COLS;j++) {
	    row[j]=(*this)(i,j);
	}
	return row;
    }

    
    sycl::marray<T,COLS>& row(int i) {
	static_assert(false);
    }

};



namespace SyclHelpers {

class QueueSingleton
{
    public:
        static QueueSingleton& getInstance()
        {
            static QueueSingleton    instance; // Guaranteed to be destroyed.
                                      // Instantiated on first use.
            return instance;
        }
    private:
        QueueSingleton() {
            std::cout<<"Building a new Queue"<<std::endl;
            Q=std::make_unique<sycl::queue>(sycl::default_selector_v);
            //const auto &device = Q.get_device();
            std::cout << "Running on: "
              << Q->get_device().get_info<sycl::info::device::name>()
              << std::endl;

        }                    // Constructor? (the {} brackets) are needed here.

        std::unique_ptr<sycl::queue> Q;
    public:
        QueueSingleton(QueueSingleton const&)               = delete;
        void operator=(QueueSingleton const&)  = delete;

        sycl::queue& queue() 
        {
           return *Q;
        }

        // Note: Scott Meyers mentions in his Effective Modern
        //       C++ book, that deleted functions should generally
        //       be public as it results in better error messages
        //       due to the compilers behavior to check accessibility
        //       before deleted status
};

    template<typename AccT>
    class SubRange
    {
    public:
	SubRange(AccT::const_iterator begin, AccT::const_iterator end):
	    m_begin(begin),
	    m_end(end)
	{
	}
	auto begin() const  {
	    return m_begin;
	}

	auto end() const {
	    return m_end;
	}




    private:
	AccT::const_iterator m_begin;
	AccT::const_iterator m_end;	    
    };


    template <typename KeyType>
    class SyclIndexMap
    {
    public:
	SyclIndexMap() :
	    m_values(0),
	    m_buckets(0),
	    m_shifts(0),
	    m_boxShifts(1),
	    m_num_buckets(0)
	{

	}
	
	SyclIndexMap(const std::vector<ankerl::unordered_dense::map<KeyType, size_t> >& maps, size_t nv, size_t nbuckets, size_t nboxes):
	    m_values(nv),
	    m_buckets(nbuckets),
	    m_shifts(nboxes),
	    m_boxShifts(nboxes+1),
	    m_num_buckets(nboxes)
	{

	    
	    sycl::host_accessor valA(m_values);
	    sycl::host_accessor buckA(m_buckets);

	    sycl::host_accessor shiftA(m_shifts);
	    sycl::host_accessor nbA(m_num_buckets);
	    sycl::host_accessor bsA(m_boxShifts);
	    



	    auto valIterator=valA.begin();
	    
	    size_t box=0;
	    size_t val_idx=0;
	    size_t bucket_idx=0;
	    
	    for( const ankerl::unordered_dense::map<KeyType,size_t>& map:  maps ) {
		bsA[box].first=val_idx;
		bsA[box].second=bucket_idx;

		for(size_t v=0;v<map.values().size();v++) {
		    valA[val_idx+v]=map.values()[v];
		}
		//valIterator=std::copy(map.values().begin(),map.values().end(), valIterator);

		for(size_t bucket=0;bucket<map.bucket_count();bucket++) {
		    buckA[bucket_idx+bucket]=map.bucket(bucket);
		}
		

		val_idx+=map.values().size();
		bucket_idx+=map.bucket_count();

		shiftA[box]=map.shifts();
		nbA[box]=map.bucket_count();
		
		
		box++;
	    }
	    bsA[box].first=val_idx;
	    bsA[box].second=bucket_idx;

	}

	static auto fromList(const std::vector<ankerl::unordered_dense::map<KeyType, size_t> >& maps)
	{
	    size_t nv=0;
	    size_t nbuckets=0;
	    size_t nboxes=maps.size();

	    for( const auto m : maps) {
		nbuckets+=m.bucket_count();
		nv+=m.values().size();
	    }

	    return SyclIndexMap<KeyType>(maps,nv,nbuckets,nboxes);
	}

	class Accessor {
	public:	    
	    Accessor( SyclIndexMap& map, sycl::handler & h):
		m_values(map.m_values,h),
		m_buckets(map.m_buckets,h),
		m_shifts(map.m_shifts,h),
		m_num_buckets(map.m_num_buckets,h),
		m_boxShifts(map.m_boxShifts,h)
	    {

	    }
	    

	    // The goal of mixed_hash is to always produce a high quality 64bit hash.
	    template <typename K>
	    [[nodiscard]] constexpr auto mixed_hash(K const& key) const -> uint64_t {
		if constexpr (ankerl::unordered_dense::detail::is_detected_v<ankerl::unordered_dense::detail::detect_avalanching, typeof m_hash>) {
		    // we know that the hash is good because is_avalanching.
		    if constexpr (sizeof(decltype(m_hash(key))) < sizeof(uint64_t)) {
			// 32bit hash and is_avalanching => multiply with a constant to avalanche bits upwards
			return m_hash(key) * UINT64_C(0x9ddfea08eb382d69);
		    } else {
			// 64bit and is_avalanching => only use the hash itself.
			return m_hash(key);
		    }
		} else {
		    // not is_avalanching => apply wyhash
		    return ankerl::unordered_dense::detail::wyhash::hash(m_hash(key));
		}
	    }

	    using dist_and_fingerprint_type = decltype(ankerl::unordered_dense::bucket_type::standard::m_dist_and_fingerprint);
	    using value_idx_type = decltype(ankerl::unordered_dense::bucket_type::standard::m_value_idx);
	    using value_type= std::pair<KeyType,size_t>;
	    using key_type= KeyType;
	    using Bucket=ankerl::unordered_dense::bucket_type::standard;

	    using value_container_type=std::vector<value_type>;
	    using bucket_alloc =
		typename std::allocator_traits<typename value_container_type::allocator_type>::template rebind_alloc<Bucket>;
	    using bucket_alloc_traits = std::allocator_traits<bucket_alloc>;
	    using bucket_pointer = typename std::allocator_traits<bucket_alloc>::pointer;	    


	    [[nodiscard]] auto next(value_idx_type bucket_idx,size_t box) const -> value_idx_type {
		return ANKERL_UNORDERED_DENSE_UNLIKELY(bucket_idx + 1U == m_num_buckets[box])
		    ? 0
		    : static_cast<value_idx_type>(bucket_idx + 1U);
	    }

	    
	    // Helper to access bucket through pointer types
	    [[nodiscard]] static constexpr auto at(bucket_pointer bucket_ptr, size_t offset) -> Bucket& {
		return *(bucket_ptr + static_cast<typename std::allocator_traits<bucket_alloc>::difference_type>(offset));
	    }



	    // use the dist_inc and dist_dec functions so that uint16_t types work without warning
	    [[nodiscard]] static constexpr auto dist_inc(dist_and_fingerprint_type x) -> dist_and_fingerprint_type {
		return static_cast<dist_and_fingerprint_type>(x + Bucket::dist_inc);
	    }
	    
	    [[nodiscard]] static constexpr auto dist_dec(dist_and_fingerprint_type x) -> dist_and_fingerprint_type {
		return static_cast<dist_and_fingerprint_type>(x - Bucket::dist_inc);
	    }


	    
	    constexpr auto dist_and_fingerprint_from_hash(uint64_t hash) const -> dist_and_fingerprint_type {
		return ankerl::unordered_dense::bucket_type::standard::dist_inc | (static_cast<dist_and_fingerprint_type>(hash) & ankerl::unordered_dense::bucket_type::standard::fingerprint_mask);
	    }

	    [[nodiscard]] constexpr auto bucket_idx_from_hash(uint64_t hash,size_t box) const -> value_idx_type {
		return static_cast<value_idx_type>(hash >> m_shifts[box]);
	    }

	    [[nodiscard]] static constexpr auto get_key(value_type const& vt) -> key_type const& {
		return vt.first;		
	    }


	
	    
	template <typename K>
	auto find(size_t box, K const& key) const  -> size_t {
	    if (ANKERL_UNORDERED_DENSE_UNLIKELY(m_boxShifts[box]==m_boxShifts[box+1])) {
		return SIZE_MAX;
	    }


	    //shifts due to the specific box
	    const size_t vs=m_boxShifts[box].first;
            const size_t bs=m_boxShifts[box].second;

	    

	    auto mh = mixed_hash(key);
	    auto dist_and_fingerprint = dist_and_fingerprint_from_hash(mh);
	    auto bucket_idx = bucket_idx_from_hash(mh,box);
	    auto* bucket=&m_buckets[bs+bucket_idx];

	    // unrolled loop. *Always* check a few directly, then enter the loop. This is faster.
	    if (dist_and_fingerprint == bucket->m_dist_and_fingerprint && m_equal(key, get_key(m_values[vs+bucket->m_value_idx]))) {
		return m_values[vs+(bucket->m_value_idx)].second;
	    }
	    dist_and_fingerprint = dist_inc(dist_and_fingerprint);
	    bucket_idx = next(bucket_idx,box);
	    bucket= &m_buckets[bs+bucket_idx];
	    
	    if (dist_and_fingerprint == bucket->m_dist_and_fingerprint && m_equal(key, get_key(m_values[vs+bucket->m_value_idx]))) {
		return m_values[vs+(bucket->m_value_idx)].second;		    
	    }
	    dist_and_fingerprint = dist_inc(dist_and_fingerprint);
	    bucket_idx = next(bucket_idx,box);
	    bucket = &m_buckets[bs+bucket_idx];
	    
	    while (true) {
		if (dist_and_fingerprint == bucket->m_dist_and_fingerprint) {
		    if (m_equal(key, get_key(m_values[vs+bucket->m_value_idx]))) {
			return m_values[vs+(bucket->m_value_idx)].second;			    
		    }
		} else if (dist_and_fingerprint > bucket->m_dist_and_fingerprint) {
		    return SIZE_MAX;
		}
		dist_and_fingerprint = dist_inc(dist_and_fingerprint);
		bucket_idx = next(bucket_idx,box);
		bucket = &m_buckets[bs+bucket_idx];
	    }
	}

	
	private:
	    sycl::accessor<std::pair<KeyType, size_t> > m_values;
	    sycl::accessor<ankerl::unordered_dense::bucket_type::standard  > m_buckets;
	    ankerl::unordered_dense::hash<KeyType> m_hash;
	    sycl::accessor<uint8_t> m_shifts;
	    sycl::accessor<size_t> m_num_buckets;
	    sycl::accessor<std::pair<size_t,size_t> > m_boxShifts; 
	    std::equal_to<KeyType> m_equal;

	};

	
	Accessor accessor(sycl::handler& h) 
	{
	    return Accessor(*this,h);	
	}
        


    private:
	sycl::buffer<uint8_t> m_shifts;
	sycl::buffer<size_t> m_num_buckets;
	sycl::buffer<std::pair<size_t,size_t>> m_boxShifts;
        sycl::buffer<std::pair<KeyType, size_t>,1> m_values;
        sycl::buffer<ankerl::unordered_dense::bucket_type::standard,1  > m_buckets;
    
    };



template<typename T, int size>
sycl::marray<T,size>  EigenVectorToMArray( const Eigen::Vector<T,size>& vec)
{
    sycl::marray<T,size> marray;
    for(size_t j=0;j<size;j++)
    {
	marray[j]=vec[j];
    }

    return marray;
}
    
    template<typename T, int size>
    std::array<T,size>  EigenVectorToCPPArray( const Eigen::Ref<const Eigen::Vector<T,size> >& vec)
{
    std::array<T,size> marray;
    for(size_t j=0;j<size;j++)
    {
	marray[j]=vec[j];
    }

    return marray;
}


}

//#include "sycl_ext_complex.hpp"
#define complex_type std::complex//sycl::ext::cplx::complex

//multiply real and complex marray
template<typename T, size_t size>
inline sycl::marray<complex_type<T>,size> operator *(const sycl::marray<T,size>& m1, const sycl::marray<complex_type<T>,size>& m2)
{
    sycl::marray<complex_type<T>,size> result;
    for(int j=0;j<size;j++) {
	result[j]=m1[j]*m2[j];
    }
    return result;
}




//needed to multiply complex marray by real value for some reason
template<typename T, size_t size>
inline sycl::marray<complex_type<T> ,size> operator *(T scal, const sycl::marray<complex_type<T>,size>& m2)
{
    sycl::marray<complex_type<T>,size> result;
    for(int j=0;j<size;j++) {
	result[j]=m2[j]*scal;
    }

    return result;
}

template<typename T, size_t size>
inline sycl::marray<complex_type<T> ,size> operator *(complex_type<T> scal, const sycl::marray<T,size>& m2)
{
    sycl::marray<complex_type<T>,size> result;
    for(int j=0;j<size;j++) {
	result[j]=m2[j]*scal;
    }

    return result;
}

template<typename T, size_t size>
inline sycl::marray<complex_type<T> ,size> operator +(const sycl::marray<complex_type<T>,size>& m2,complex_type<T> scal)
{
    sycl::marray<complex_type<T>,size> result;
    for(int j=0;j<size;j++) {
	result[j]=m2[j]+scal;
    }

    return result;
}




#endif
