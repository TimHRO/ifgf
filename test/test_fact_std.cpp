#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "../boundingbox.hpp"
#include "../chebinterp.hpp"
#include "../cone_domain.hpp"
#include "../util.hpp"

typedef std::complex<double> T;
const T I(0, 1);
const int DIM = 3;
typedef Eigen::Array<double, 3, Eigen::Dynamic> PointArray;

inline void transformInterpToCart(const Eigen::Ref<const PointArray>& nodes,
                                  Eigen::Ref<PointArray> transformed, const Eigen::Vector3d& xc,
                                  double H)
{
    transformed = Util::interpToCart<3>(nodes, xc, H);
}

void save_target_points(const Eigen::Array<double, 3, Eigen::Dynamic>& tgt_points,
                        const std::string& filename = "./test/target_points.csv")
{
    std::ofstream outfile(filename);
    outfile << "x,y,z\n";
    for (int i = 0; i < tgt_points.cols(); ++i)
    {
        outfile << std::setprecision(10) << tgt_points(0, i) << "," << tgt_points(1, i) << ","
                << tgt_points(2, i) << "\n";
    }
    outfile.close();
}

int main(const int argc, char** argv)
{
    if (argc < 8)
    {
        std::cerr << "Usage: <H> <ka> <kb> <segments> <order> <max cone length> <simga - goemetric "
                     "splitt> <eps - cut below> \n";
        return 1;
    }

    // --- Setup Parameters ---
    double H = atof(argv[1]);
    const double h = (std::sqrt(3.0) / 2.0) * H;
    std::complex<double> kappa(atof(argv[2]), atof(argv[3]));
    int num_splits = atoi(argv[4]);
    int o = atoi(argv[5]);
    double e = atof(argv[6]);
    double f = atof(argv[7]);
    double eps = atof(argv[8]);

    const int N_src = 1000;
    const int N_tgt = 2000;
    Eigen::Vector3d xc = Eigen::Vector3d::Zero();
    Eigen::Vector<int, 3> order;
    order << o, o + 2, o + 2;

    double s_max_range = std::sqrt(3.0) / 3.0;
    double delta_s_total = s_max_range; //* std::min(1.0, 1.0/ std::abs(kappa * H));
    double s_min_range = s_max_range - delta_s_total + e;

    // --- Boundaries Setup ---
    std::vector<double> s_boundaries(num_splits + 1);
    double r_min = 3 / 2 * H;
    double r_max = h / s_min_range;
    double sigma = f;

    double delta_r = (r_max - r_min) / num_splits;
    std::cout << "Delta r: " << delta_r << "\n";

    for (int i = 0; i <= num_splits; ++i)
    {
        double x_l = std::pow(sigma, num_splits - i);
        // double x_l_transformed = r_min + (r_max - r_min) * x_l;
        double x_l_transformed = s_min_range + (s_max_range - s_min_range) * x_l;
        s_boundaries[i] = x_l_transformed;
        // s_boundaries[i] = h / (r_max - i * delta_r);
        std::cout << "grading: " << x_l << "\n";
        // std::cout << "s Intervall length: " <<  <<"\n";
    }

    s_boundaries[0] = s_min_range;
    s_boundaries[num_splits] = s_max_range;

    double max_theta = (M_PI / 4.0); //* std::min(1.0, 1.0/ std::abs(kappa * H));
    double t_min = M_PI / 2.0 - max_theta / 2.0, t_max = M_PI / 2.0 + max_theta / 2.0;
    double p_min = -max_theta / 2.0, p_max = max_theta / 2.0;

    std::srand(42);
    PointArray src_points = PointArray::Random(DIM, N_src) * H / 2.0;
    Eigen::Array<T, Eigen::Dynamic, 1> src_strengths =
        Eigen::Array<T, Eigen::Dynamic, 1>::Constant(N_src, 1.0);

    auto kernel = [&](const Eigen::Vector3d& x, const Eigen::Vector3d& xp) -> T
    {
        double r = x.norm();
        double R_tilde = (x - xp).norm();
        T exp_term = std::exp(I * kappa * (R_tilde - r)); // * std::exp(-kappa.real() * R_tilde);
        return ((r / (4.0 * M_PI * R_tilde)) * exp_term);
    };

    double max_center = 0.0;

    auto centered_factor = [&](const Eigen::Vector3d& x) -> T
    {
        double r = x.norm();
        T result = std::exp(I * kappa * r) / r;
        max_center = std::max(std::abs(result), max_center);
        return result;
    };

    // --- Piecewise Precomputation of Node Data and Target Points ---
    std::vector<Eigen::Array<T, Eigen::Dynamic, 1>> split_coeffs(num_splits);
    std::cout << "\n--- Geometric Split Information ---\n";

    for (int s = 0; s < num_splits; ++s)
    {
        double s0 = s_boundaries[s];
        double s1 = s_boundaries[s + 1];
        BoundingBox<3> box(Eigen::Vector3d(s0, t_min, p_min), Eigen::Vector3d(s1, t_max, p_max));
        ConeDomain<DIM> grid({1, 1, 1}, box);

        PointArray nodes = ChebychevInterpolation::chebnodesNdd<double, DIM>(order);
        PointArray physNodes(DIM, nodes.cols());
        transformInterpToCart(grid.transform(0, nodes), physNodes, xc, H);

        Eigen::Array<T, Eigen::Dynamic, 1> node_data(nodes.cols());
        double min_abs_val = 1e18;
        for (int i = 0; i < physNodes.cols(); ++i)
        {
            node_data(i) = 0;
            for (int j = 0; j < N_src; ++j)
                node_data(i) += src_strengths[j] * kernel(physNodes.col(i), src_points.col(j));
            min_abs_val = std::min(min_abs_val, std::abs(node_data(i)));
        }

        save_target_points(physNodes, "./test/interpolation_nodes_" + std::to_string(s) + ".csv");
        std::cout << "Segment " << s << " | Range s: [" << s0 << ", " << s1 << "] | Range r: ["
                  << h / s1 << ", " << h / s0 << "] |Min |g_S|: " << std::scientific << min_abs_val
                  << "\n";

        split_coeffs[s].resize(nodes.cols());
        ChebychevInterpolation::chebtransform<T, DIM>(node_data, split_coeffs[s], order);
    }

    // --- Compute Interpolated and Exact Vals at Target Points ---

    double r_min_val = h / s_max_range;
    double r_max_val = h / s_min_range;
    Eigen::ArrayXd r_samples = Eigen::ArrayXd::LinSpaced(N_tgt, r_min_val, r_max_val);

    std::mt19937 gen(42);
    std::uniform_real_distribution<double> unit_dist(0.0, 1.0);

    PointArray eval_s(DIM, N_tgt), eval_phys(DIM, N_tgt);
    Eigen::Array<T, Eigen::Dynamic, 1> exact_vals(N_tgt), interp_vals(N_tgt);
    interp_vals.setZero();
    std::vector<std::vector<int>> bins(num_splits);

    for (int i = 0; i < N_tgt; ++i)
    {
        double s_val = h / r_samples(i);
        double theta_val = t_min + unit_dist(gen) * (t_max - t_min);
        double phi_val = p_min + unit_dist(gen) * (p_max - p_min);
        eval_s.col(i) << s_val, theta_val, phi_val;
        eval_phys.col(i) = Util::interpToCart<3>(eval_s.col(i), xc, H);

        // Compute exact kernel sum
        for (int j = 0; j < N_src; ++j)
        {
            exact_vals(i) += src_strengths[j] * centered_factor(eval_phys.col(i)) *
                             kernel(eval_phys.col(i), src_points.col(j));
        }

        // Assign Target to Segment Bin by Id
        int seg_id = -1;
        for (int k = 0; k < num_splits; ++k)
        {
            double lower = s_boundaries[k];
            double upper = s_boundaries[k + 1];

            // Ensure we handle intervals correctly regardless of boundary order (ascending or
            // descending)
            if (s_val >= std::min(lower, upper) - 1e-15 && s_val <= std::max(lower, upper) + 1e-15)
            {
                seg_id = k;
                break; // Found the interval
            }
        }

        // Assign to bin if found
        if (seg_id != -1)
        {
            bins[seg_id].push_back(i);
        }
        else
        {
            // Fallback for points exactly on the very outer edges due to precision
            if (s_val < std::min(s_boundaries[0], s_boundaries[num_splits]))
            {
                bins[0].push_back(i);
            }
            else
            {
                bins[num_splits - 1].push_back(i);
            }
        }
    }

    // find segments to cut, check if kappa.imag r is large enough to cut segment
    int s_cut = 0;
    for (int s = 0; s < num_splits; s++)
    {
        if (std::exp(-kappa.imag() * h / s_boundaries[s]) <= eps)
        {
            s_cut = s;
        }
    }

    std::cout << "Starting interpolation from segment " << s_cut << " : " << s_boundaries[s_cut]
              << " setting rest to zero" << "\n";

    // --- Evaluation ---
    for (int s = 0; s < num_splits; ++s)
    {
        // if (bins[s].empty()) continue;
        int n_pts = bins[s].size();
        PointArray ref(DIM, n_pts);
        double s0 = s_boundaries[s], s1 = s_boundaries[s + 1];

        for (int k = 0; k < n_pts; ++k)
        {
            int idx = bins[s][k];
            ref.col(k) << 2.0 * (eval_s(0, idx) - s0) / (s1 - s0) - 1.0,
                2.0 * (eval_s(1, idx) - t_min) / (t_max - t_min) - 1.0,
                2.0 * (eval_s(2, idx) - p_min) / (p_max - p_min) - 1.0;
        }

        Eigen::Array<T, Eigen::Dynamic, 1> res(n_pts);
        Eigen::Array<T, Eigen::Dynamic, 1> subdomain_error(n_pts);
        res.setZero();
        subdomain_error.setZero();
        if (s_cut <= s)
        {
            ChebychevInterpolation::parallel_evaluate<T, DIM, 1>(ref, split_coeffs[s], res, order);
        }

        for (int k = 0; k < n_pts; ++k)
        {
            interp_vals(bins[s][k]) = res(k) * centered_factor(eval_phys.col(bins[s][k]));
            subdomain_error(k) = (interp_vals(bins[s][k]) - exact_vals(bins[s][k]));
        }
        std::cout << "Segment " << s << " | Absolute Error: " << subdomain_error.matrix().norm()
                  << " | Targets: " << bins[s].size() << "\n";
    }

    save_target_points(eval_phys, "./test/target_points.csv");

    std::cout << "\nRelative L2 Error: "
              << (interp_vals - exact_vals).matrix().norm() / exact_vals.matrix().norm() << "\n";
    std::cout << "\nAbsolute L2 Error: " << (interp_vals - exact_vals).matrix().norm() << "\n";
    return 0;
}