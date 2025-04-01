#include <Eigen/Dense>
#include <iostream>

#include <cmath>

#include <oneapi/tbb/blocked_range.h>
#include <random>
#include <cstdlib>
#include <tbb/task_arena.h>
#include <tbb/global_control.h>
#include <fenv.h>
#include <chrono>

#include "config.hpp"
#include "helmholtz_ifgf.hpp"
#include "ifgfoperator.hpp"
#include "octree.hpp"

#include "grad_helmholtz_ifgf.hpp"

const int dim=3;

typedef std::complex<RealScalar> Complex;
const std::complex<double>  kappa = Complex(0,-10);
typedef Eigen::Vector<PointScalar,dim> Point;
std::complex<double> my_kernel(const Point& x, const Point& y, const Point& normal)
{    
    double norm = (x-y).norm();
    double nxy = -normal.dot(x-y);
    if(norm < 1e-12) return 0;
    /*auto kern = exp(Complex(0,kappa)*norm) / (4 * M_PI * norm*norm*norm)
	* ( nxy * (Complex(1,0)*1. - Complex(0,kappa)*norm)  - Complex(0,kappa)*norm*norm);
	// return kern;*/

    auto kern = exp(-kappa*norm) / ((4.0 * M_PI * norm));
    //x	* ( nxy * (Complex(1,0)*1. - Complex(0,kappa)*norm)  - Complex(0,kappa)*norm*norm);
	// return kern;*/
    

    return kern;
}



auto randomPointOnSphere() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    PointScalar theta = dis(gen) * 2.0 * M_PI; // Random angle theta
    PointScalar phi = acos(2.0 * dis(gen) - 1.0); // Random angle phi

    PointScalar x = sin(phi) * cos(theta);
    PointScalar y = sin(phi) * sin(theta);
    PointScalar z = cos(phi);

    return Eigen::Vector< PointScalar,3>(x, y, z);
}



int main()
{
    srand((unsigned int) 1);    
    typedef Eigen::Matrix<PointScalar, dim, Eigen::Dynamic> PointArray ;

    const int N = 1000;

    for (auto platform : sycl::platform::get_platforms())
    {
        std::cout << "Platform: "
                  << platform.get_info<sycl::info::platform::name>()
                  << std::endl;

        for (auto device : platform.get_devices())
        {
            std::cout << "\tDevice: "
                      << device.get_info<sycl::info::device::name>()
                      << std::endl;
        }
    }


    //Eigen::initParallel();
    //auto global_control = tbb::global_control( tbb::global_control::max_allowed_parallelism,      1);
    //oneapi::tbb::task_arena arena(1);

    HelmholtzIfgfOperator<dim> op(-kappa.imag(),100,8,1,-1); //3
    //GradHelmholtzIfgfOperator<dim> op(kappa,10,3,1,1e-5); //3
    //op.setDx(-1);

    PointArray srcs(3,N);
    //PointArray srcs=load_csv_arma<PointArray>("srcs.csv");

    //std::cout<<"s"<<srcs<<std::endl;
    //size_t  N=srcs.cols();
    //(dim,N);

    //srcs <<5*(PointArray::Random(dim,N).array());//,0.5+0.1*(PointArray::Random(dim,N).array()) ;
    tbb::parallel_for(tbb::blocked_range<size_t>(0,srcs.cols()), [&](tbb::blocked_range<size_t> r) {
	for(size_t i=r.begin();i<r.end();i++){
	    srcs.col(i)=randomPointOnSphere();
	}});
    PointArray normals = srcs;//(PointArray::Random(dim,srcs.cols()).array());
    PointArray targets = 0.9*srcs;//(PointArray::Random(dim, N).array());

    tbb::parallel_for(tbb::blocked_range<size_t>(0,targets.cols()), [&](tbb::blocked_range<size_t> r) {
	for(size_t i=r.begin();i<r.end();i++){
	    targets.col(i)=randomPointOnSphere();
	}});
 

    normals.colwise().normalize();


    //feenableexcept(FE_DIVBYZERO | FE_OVERFLOW | FE_UNDERFLOW | FE_INVALID);
    op.init(srcs, targets);//,normals);

    Eigen::Vector<std::complex<RealScalar>, Eigen::Dynamic> weights(srcs.cols());
    weights = Eigen::Vector<RealScalar, Eigen::Dynamic>::Random(srcs.cols());

    Eigen::Vector<std::complex<RealScalar>, Eigen::Dynamic> result;


    //first one is not timed!
    result = op.mult(weights);
    using namespace std::chrono;
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    const int Nmult=1;
    for(int i=0;i<Nmult;i++) {
	std::cout<<"mult"<<std::endl;
	result = op.mult(weights);
	std::cout << "done multiplying" << std::endl;
    }
    high_resolution_clock::time_point t2 = high_resolution_clock::now();

    duration<PointScalar> time_span = duration_cast<duration<PointScalar>>(t2 - t1);
    std::cout << time_span.count()/Nmult << " seconds" << std::endl;

    srand((unsigned) time(NULL));
    double maxE = 0;
    for (int j = 0; j < 100; j++) {
        std::complex<double> val = 0;
        int index = rand() % targets.cols();
        //std::cout<<"idx"<<index<<std::endl;
        for (int i = 0; i < srcs.cols(); i++) {
            val += std::complex<double>( weights[i]) * my_kernel(srcs.col(i), targets.col(index),normals.col(i));
        }

        double e = std::abs(val - std::complex<double>(result[index]))/std::abs(val);
        maxE = std::max(e, maxE);
        //std::cout<<"e="<<e<<" val="<<val<<" vs" <<result[index]<<std::endl;
    }

    std::cout << "summary: e=" << maxE << std::endl;

}
