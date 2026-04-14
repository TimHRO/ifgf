#ifndef __IFGF_CONFIG_HPP
#define __IFGF_CONFIG_HPP



//default float type used
typedef  float DefaultScalarType;

//change specific instances of float if needed
typedef DefaultScalarType PointScalar ;
typedef DefaultScalarType RealScalar;
typedef DefaultScalarType ExtendedScalar;

//#define FAST_CTP
#define USE_FAST_HASH

// #define INITIALIZE_BY_ZERO

#define CACHE_OCTREE

// #define KEEP_LEVEL_DATA

#define REFINEMENT_FACTOR (unsigned int) 2
#define REFINEMENT_LEVELS (unsigned int) 1



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
