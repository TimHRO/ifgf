#include <Eigen/Dense>
#include <iostream>

#include <cmath>

#include "../helmholtz_ifgf.hpp"
#include "../ifgfoperator.hpp"
#include "../modified_helmholtz_ifgf.hpp"
#include "../octree.hpp"

#include "../grad_helmholtz_ifgf.hpp"

const int dim = 3;

typedef std::complex<double> Complex;
// const Complex kappa = Complex(0.0, -10);
typedef Eigen::Vector<double, dim> Point;
std::complex<double> kernel(Complex kappa, const Point& x, const Point& y, const Point& normal)
{
    double norm = (x - y).norm();
    double nxy = -normal.dot(x - y);
    if (norm < 1e-14)
        return 0;
    /*auto kern = exp(Complex(0,kappa)*norm) / (4 * M_PI * norm*norm*norm)
    * ( nxy * (Complex(1,0)*1. - Complex(0,kappa)*norm)  - Complex(0,kappa)*norm*norm);
    // return kern;*/

    auto kern = exp(-kappa * norm) / (4 * M_PI * norm);
    // x	* ( nxy * (Complex(1,0)*1. - Complex(0,kappa)*norm)  - Complex(0,kappa)*norm*norm);
    //  return kern;*/

    return kern;
}

#include <cstdlib>
#include <fenv.h>
#include <random>
#include <tbb/global_control.h>
#include <tbb/task_arena.h>

Eigen::Vector3d randomPointOnSphere()
{
    std::random_device rd;
    // std::mt19937 gen(rd());
    unsigned int seed = 42;
    static std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(0.0, 1.0);

    double theta = dis(gen) * 2.0 * M_PI;    // Random angle theta
    double phi = acos(2.0 * dis(gen) - 1.0); // Random angle phi

    double x = sin(phi) * cos(theta);
    double y = sin(phi) * sin(theta);
    double z = cos(phi);

    return Eigen::Vector3d(x, y, z);
}

int main(int argc, char** argv)
{

    typedef Eigen::Matrix<double, dim, Eigen::Dynamic> PointArray;

    const int N = atoi(argv[5]);
    int tarPerBox = atoi(argv[6]);
    std::complex<double> kappa(atof(argv[1]), atof(argv[2]));
    int o = atoi(argv[3]);
    int base_n = atoi(argv[4]);
    double threshold = atof(argv[7]);

    // Eigen::initParallel();
    // auto global_control = tbb::global_control( tbb::global_control::max_allowed_parallelism, 1);
    // oneapi::tbb::task_arena arena(1);

    // HelmholtzIfgfOperator<dim> op(-kappa.real(), 10, 10, 1, -1); // 3
    ModifiedHelmholtzIfgfOperator<dim> op(kappa, tarPerBox, o, base_n, -1);
    // GradHelmholtzIfgfOperator<dim> op(kappa,10,3,1,1e-5); //3
    // op.setDx(-1);

    PointArray srcs(3, N);
    // PointArray srcs=load_csv_arma<PointArray>("srcs.csv");

    // std::cout<<"s"<<srcs<<std::endl;
    // size_t  N=srcs.cols();
    //(dim,N);
    // srcs <<(PointArray::Random(dim,N).array());//,0.5+0.1*(PointArray::Random(dim,N).array()) ;
    for (int i = 0; i < srcs.cols(); i++)
    {
        srcs.col(i) = randomPointOnSphere();
    }
    PointArray normals = srcs; //(PointArray::Random(dim,srcs.cols()).array());
    PointArray targets = srcs; //(PointArray::Random(dim, N).array());
    /*for(int i=0;i<targets.cols();i++){
    targets.col(i)=randomPointOnSphere();
    }*/

    normals.colwise().normalize();

    // feenableexcept(FE_DIVBYZERO | FE_OVERFLOW | FE_UNDERFLOW | FE_INVALID);
    // double threshold = 1e-10;
    std::function<bool(double dist)> cutOff = [kappa, threshold](const double dist)
    { return (exp(-kappa.real() * dist) / dist) < threshold; };

    op.init(srcs, targets, cutOff); //,normals);

    Eigen::Vector<std::complex<double>, Eigen::Dynamic> weights(srcs.cols());
    weights = Eigen::VectorXd::Random(srcs.cols());

    Eigen::Vector<std::complex<double>, Eigen::Dynamic> result;
    for (int i = 0; i < 1; i++)
    {
        std::cout << "mult" << std::endl;
        result = op.mult(weights);
        std::cout << "done multiplying" << std::endl;
    }

    srand((unsigned)time(NULL));
    double maxE_rel = 0;
    double maxE_abs = 0;
    for (int j = 0; j < N; j++)
    {
        std::complex<double> val = 0;
        // int index = rand() % targets.cols();
        int index = j;
        // std::cout<<"idx"<<index<<std::endl;
        for (int i = 0; i < srcs.cols(); i++)
        {
            val += weights[i] * kernel(kappa, srcs.col(i), targets.col(index), normals.col(i));
        }

        double e_abs = std::abs((val - result[index]));
        double e_rel = e_abs / std::abs(val);
        maxE_abs = std::max(e_abs, maxE_abs);
        maxE_rel = std::max(e_rel, maxE_rel);
        // std::cout<<"e="<<e<<" val="<<val<<" vs" <<result[index]<<std::endl;
    }

    std::cout << "summary: e abs= " << maxE_abs << std::endl;
    std::cout << "summary: e rel= " << maxE_rel << std::endl;
}
