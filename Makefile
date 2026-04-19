CPPFLAGS=-O2 -g    -DEIGEN_FAST_MATH=1  -DEIGEN_DONT_PARALLELIZE   -march=native -std=c++20   -I/usr/include/eigen3/ -flto=auto

all: 
	g++ $(CPPFLAGS) -o ./test/test_ifgf  ./test/test_ifgf.cpp  -ltbb
#	g++ -O2 -g -ldl -gdwarf-3  -DEIGEN_FAST_MATH=1  -DEIGEN_DONT_PARALLELIZE   -march=native -std=c++20 -o test_ifgf_laplace test_ifgf_laplace.cpp -I/usr/include/eigen3/ -ltbb

#chebinterp.o: chebinterp.cpp chebinterp.hpp  boundingbox.hpp
#	g++ $(CPPFLAGS) -c chebinterp.cpp 


test_cheb:
	g++  -O3 -g -march=native -std=c++20 -o ./test/test_cheb ./test/test_chebinterp.cpp -I. -I/usr/include/eigen3/ -ltbb

test_fact_std:
	g++  -O3 -g -march=native -std=c++20 -o ./test/test_fact_std ./test/test_fact_std.cpp -I. -I/usr/include/eigen3/ -ltbb

test_fact_osz:
	g++  -O3 -g -march=native -std=c++20 -o ./test/test_fact_osz ./test/test_fact_osz.cpp -I. -I/usr/include/eigen3/ -ltbb
