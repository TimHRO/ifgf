#ifndef __IFGF_CONFIG_HPP
#define __IFGF_CONFIG_HPP


typedef  double PointScalar ;
typedef double  RealScalar;
typedef double ExtendedScalar;

//#define FAST_CTP
//#define DEBUG_CTP
#define USE_FAST_HASH

// #define INITIALIZE_BY_ZERO

#define CACHE_OCTREE 

//#define KEEP_LEVEL_DATA

#ifdef USE_FAST_HASH
#include "unordered_dense.h"
typedef ankerl::unordered_dense::map<size_t, size_t> IndexMap;
typedef ankerl::unordered_dense::set<size_t> IndexSet;

typedef ankerl::unordered_dense::map<size_t, size_t> ConeMap;

#else
#include <map>
typedef std::unordered_map<size_t, size_t> IndexMap;
typedef std::unordered_set<size_t> IndexSet;

typedef std::unordered_map<std::pair<size_t,size_t>, size_t> ConeMap;


#endif



#endif
