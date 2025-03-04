#include <iostream>

#include <Eigen/Dense>
#include <cmath>
#include "chebinterp.hpp"

#include "sycl_ext_complex.hpp"
#include <complex>

#include <tbb/parallel_for.h>

#include <sycl/sycl.hpp>

int main()
{
    const int N = 10000;
    const int dim = 3;

    typedef PointScalar TSCAL;
    typedef std::complex<TSCAL> T;
    //typedef TSCAL T;
    Eigen::Array<TSCAL, dim, Eigen::Dynamic> points = Eigen::Array<TSCAL, dim, Eigen::Dynamic, Eigen::ColMajor>::Random(dim, N)+0.01;
   

    const TSCAL H = 0.5;
    const TSCAL k = 1;
    typedef Eigen::Vector<TSCAL, dim> Point;
    const Point xc{0, H / 2.0f,0};
    auto kernel = [&](auto pnt) {
        auto d=pnt.matrix().squaredNorm();
	return exp(d);
    };

    const int p = 20;
    Eigen::Vector<int, dim> ns{p,p,p};
    Eigen::Array<TSCAL, dim, Eigen::Dynamic> interp_nodes = ChebychevInterpolation::chebnodesNd<TSCAL,-1,-1,-1>(ns);

    Eigen::Array<T, Eigen::Dynamic, 1> interp_values(interp_nodes.cols());
    interp_values.fill(0);
    for (int i = 0; i < interp_nodes.cols(); i++) {
        const Eigen::Vector<TSCAL, dim> &pnt = interp_nodes.col(i);

        interp_values[i] = kernel(pnt);
    }

    Eigen::Array<T, Eigen::Dynamic, 1> transformed_values(interp_nodes.cols());
    ChebychevInterpolation::chebtransform<T,dim>(interp_values,transformed_values, ns);
    std::cout<<"1"<<std::endl;

    Eigen::Array<T, Eigen::Dynamic, 1> values(N);
    for (int i = 0; i < values.size(); i++) {
        values[i] = kernel(points.col(i));
    }
    std::cout<<"2"<<std::endl;
    Eigen::Array<T, Eigen::Dynamic, 1> approx_values(N);
    //Eigen::internal::set_is_malloc_allowed(false);

    ChebychevInterpolation::parallel_evaluate<T, dim,1>(points, transformed_values, approx_values,ns);

    std::cout<<"done";
    std::cout<<"approx="<<approx_values[11]<<std::endl;
    std::cout<<"ex="<<values[11]<<std::endl;
    //Eigen::internal::set_is_malloc_allowed(true);

    approx_values -= values;

    std::cout << "error=" << approx_values.matrix().norm() / sqrt(N) << std::endl;

    return 0;

}
