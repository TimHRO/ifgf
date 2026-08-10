#ifndef __HELMHOLTZ_KERNELS_HPP__
#define __HELMHOLTZ_KERNELS_HPP__

#include "config.hpp"
#include "ifgfoperator.hpp"

#ifndef HIGH_EXP_CUTOFF
#define HIGH_EXP_CUTOFF 50
#endif

#ifndef IFGF_KERNEL_HP_SCALAR
#define IFGF_KERNEL_HP_SCALAR double
#endif
#ifndef IFGF_KERNEL_LP_SCALAR
#define IFGF_KERNEL_LP_SCALAR float
#endif


namespace ifgf_kernels {

struct KernelTypes {
    typedef std::complex<RealScalar> T;    // storage / return type

    typedef IFGF_KERNEL_HP_SCALAR    hp;   // high-precision geometry scalar
    typedef IFGF_KERNEL_LP_SCALAR    lp;   // low-precision transcendental scalar
    typedef std::complex<hp>         Thp;
    typedef std::complex<lp>         Tlp;

    static constexpr int dim = 3;

    // 1/(4*pi), in hp
    static constexpr hp INV_4PI = hp(0.07957747154594766788444188168625718L);

    static inline T narrow(const Thp& v)
    {
        return T(RealScalar(v.real()), RealScalar(v.imag()));
    }
    static inline T narrow(const Tlp& v)
    {
        return T(RealScalar(v.real()), RealScalar(v.imag()));
    }
};


// ===========================================================================
// Modified Helmholtz single layer
// ===========================================================================
template <bool WithDecay>
class ModifiedHelmholtzKernelFunctions : public KernelTypes
{
public:
    static constexpr bool USES_NORMALS = false;

    explicit ModifiedHelmholtzKernelFunctions(std::complex<RealScalar> waveNr)
        : k(waveNr) {}

    // exp(i*kappa*r)
    inline Tlp emkd(lp kr, lp ki, lp d) const
    {
        const lp o = kr * d;
        const lp c = sycl::native::cos(o);
        const lp sn = sycl::native::sin(o);
        if constexpr (WithDecay) {
            return Tlp(sycl::native::exp(-ki * d)) * Tlp(c, sn);
        } else {
            (void)ki;
            return Tlp(c, sn);
        }
    }

    inline T kernelFunction(const sycl::marray<hp, 3>& x) const
    {
        const hp d  = sycl::sqrt(sycl::fma(x[2], x[2], sycl::fma(x[1], x[1], x[0] * x[0])));
        const hp kr = hp(k.real());
        const hp ki = hp(k.imag());

        if constexpr (WithDecay) {
            if (sycl::fabs(d) <= hp(1e-15) || d * ki > hp(HIGH_EXP_CUTOFF)) {
                return RealScalar(0.0);
            }
        } else {
            (void)ki;
            if (sycl::fabs(d) <= hp(1e-15)) {
                return RealScalar(0.0);
            }
        }

        const Tlp osc = emkd(lp(kr), lp(ki), lp(d));
        const Thp val = (INV_4PI / d) * Thp(hp(osc.real()), hp(osc.imag()));
        return narrow(val);
    }

    template <typename AT1, typename AT2, typename AT3, typename AT4>
    T evaluateKernel(const AT1& xs, size_t x0, size_t xend, const AT2& ys,
                     size_t y0, const AT3& ws, const AT4& ns) const
    {
        (void)ns;
        T result = 0;

        sycl::marray<hp, 3> pnt;
        for (size_t i = x0; i < xend; i++) {
            for (int l = 0; l < dim; l++) {
                pnt[l] = hp(xs[i * dim + l]) - hp(ys[y0 * dim + l]);
            }
            result += ws[i] * kernelFunction(pnt);
        }
        return result;
    }

    template <typename AT1, typename AT2, typename AT3>
    T evaluateFactoredKernel(const AT1& xs, size_t x0, size_t xend,
                             const sycl::marray<PointScalar, dim>& y,
                             const AT2& ws, const AT3& ns,
                             const sycl::marray<PointScalar, dim>& xc,
                             PointScalar H) const
    {
        (void)ns;
        (void)H;

        T result = 0;

        const hp dcx = hp(y[0]) - hp(xc[0]);
        const hp dcy = hp(y[1]) - hp(xc[1]);
        const hp dcz = hp(y[2]) - hp(xc[2]);
        const hp dc  = sycl::sqrt(sycl::fma(dcz, dcz, sycl::fma(dcy, dcy, dcx * dcx)));

        const lp kr = lp(k.real());
        const lp ki = lp(k.imag());

        for (size_t i = x0; i < xend; i++) {
            const hp px = hp(xs[i * dim])     - hp(y[0]);
            const hp py = hp(xs[i * dim + 1]) - hp(y[1]);
            const hp pz = hp(xs[i * dim + 2]) - hp(y[2]);
            const hp d2 = sycl::fma(pz, pz, sycl::fma(py, py, px * px));

            if (d2 < hp(1e-30)) continue;

            const hp id   = sycl::rsqrt(d2);
            const hp d    = d2 * id;
            const hp ddc  = d - dc;                 // use hp to avoid cancellation
            const lp dcod = lp(dc * id);

            const Tlp osc = emkd(kr, ki, lp(ddc));
            const Tlp val = osc * dcod;
            result += ws[i] * narrow(val);
        }
        return result;
    }

    template <typename TX>
    inline T CF(TX x) const
    {
        const lp d2 = sycl::fma(lp(x[2]), lp(x[2]),
                        sycl::fma(lp(x[1]), lp(x[1]), lp(x[0]) * lp(x[0])));

        if (sycl::fabs(d2) < lp(1e-14)) return 0;

        const lp id = sycl::rsqrt(d2);
        const lp d  = d2 * id;
        const lp kr = lp(k.real());
        const lp ki = lp(k.imag());

        const Tlp osc = emkd(kr, ki, d);
        const Tlp val = osc * (id * lp(INV_4PI));
        return narrow(val);
    }

    // SL and DL are currently identical, keep separate for potential changes
    template <typename TX, typename TY>
    inline T transfer_factor(TX x, TY xc, PointScalar H,
                             TY pxc, PointScalar pH) const
    {
        (void)H; (void)pH;

        const hp zx  = hp(x[0]) - hp(xc[0]);
        const hp zy  = hp(x[1]) - hp(xc[1]);
        const hp zz  = hp(x[2]) - hp(xc[2]);
        const hp zpx = hp(x[0]) - hp(pxc[0]);
        const hp zpy = hp(x[1]) - hp(pxc[1]);
        const hp zpz = hp(x[2]) - hp(pxc[2]);

        const hp d  = sycl::sqrt(sycl::fma(zz, zz, sycl::fma(zy, zy, zx * zx)));
        const hp dp = sycl::sqrt(sycl::fma(zpz, zpz, sycl::fma(zpy, zpy, zpx * zpx)));

        if (sycl::fabs(d) < hp(1e-15)) return 0;

        const hp ddp  = d - dp;                 // use hp to avoid cancellation
        const lp dpod = lp(dp / d);

        const lp kr = lp(k.real());
        const lp ki = lp(k.imag());

        const Tlp val = emkd(kr, ki, lp(ddp)) * dpod;
        return narrow(val);
    }

private:
    std::complex<RealScalar> k;
};


// ===========================================================================
// Modified Helmholtz double layer
// ===========================================================================
template <bool WithDecay>
class DoubleLayerHelmholtzKernelFunctions : public KernelTypes
{
public:
    static constexpr bool USES_NORMALS = true;

    explicit DoubleLayerHelmholtzKernelFunctions(std::complex<RealScalar> waveNr)
        : k(waveNr) {}

    // exp(i*kappa*d) = exp(-ki*d)*(cos(kr*d) + i*sin(kr*d))
    inline Tlp emkd(lp kr, lp ki, lp d) const
    {
        const lp o = kr * d;
        const lp c = sycl::native::cos(o);
        const lp sn = sycl::native::sin(o);
        if constexpr (WithDecay) {
            return Tlp(sycl::native::exp(-ki * d)) * Tlp(c, sn);
        } else {
            (void)ki;
            return Tlp(c, sn);
        }
    }

    inline T kernelFunction(const sycl::marray<hp, 3>& x,
                            const sycl::marray<hp, 3>& n) const
    {
        const hp d2 = sycl::fma(x[2], x[2], sycl::fma(x[1], x[1], x[0] * x[0]));
        const hp kr = hp(k.real());
        const hp ki = hp(k.imag());

        if constexpr (WithDecay) {
            if (sycl::fabs(d2) <= hp(1e-24)
                || sycl::sqrt(d2) * ki > hp(HIGH_EXP_CUTOFF)) {
                return RealScalar(0.0);
            }
        } else {
            if (sycl::fabs(d2) <= hp(1e-24)) {
                return RealScalar(0.0);
            }
        }

        const hp id = sycl::rsqrt(d2);
        const hp d  = d2 * id;
        const hp xn = sycl::fma(x[2], n[2], sycl::fma(x[1], n[1], x[0] * n[0]));
        // (i*kappa - 1/r) = (-ki - id) + i*kr
        // MINUS to match NGSolve's dG/dn_y orientation
        const Thp mk_minus_id = -Thp(-ki - id, kr);
        const Thp geom = -INV_4PI * (id * id) * hp(xn) * mk_minus_id;
        const Tlp osc = emkd(lp(kr), lp(ki), lp(d));
        const Thp val = geom * Thp(hp(osc.real()), hp(osc.imag()));

        return narrow(val);
    }

    template <typename AT1, typename AT2, typename AT3, typename AT4>
    T evaluateKernel(const AT1& xs, size_t x0, size_t xend, const AT2& ys,
                     size_t y0, const AT3& ws, const AT4& ns) const
    {
        T result = 0;

        sycl::marray<hp, 3> pnt;
        sycl::marray<hp, 3> nrm;
        for (size_t i = x0; i < xend; i++) {
            for (int l = 0; l < dim; l++) {
                pnt[l] = hp(xs[i * dim + l]) - hp(ys[y0 * dim + l]);
                nrm[l] = hp(ns[i * dim + l]);
            }
            result += ws[i] * kernelFunction(pnt, nrm);
        }
        return result;
    }

    template <typename AT1, typename AT2, typename AT3>
    T evaluateFactoredKernel(const AT1& xs, size_t x0, size_t xend,
                             const sycl::marray<PointScalar, dim>& y,
                             const AT2& ws, const AT3& ns,
                             const sycl::marray<PointScalar, dim>& xc,
                             PointScalar H) const
    {
        (void)H;

        T result = 0;

        const hp dcx = hp(y[0]) - hp(xc[0]);
        const hp dcy = hp(y[1]) - hp(xc[1]);
        const hp dcz = hp(y[2]) - hp(xc[2]);
        const hp dc  = sycl::sqrt(sycl::fma(dcz, dcz, sycl::fma(dcy, dcy, dcx * dcx)));

        const lp kr = lp(k.real());
        const lp ki = lp(k.imag());

        for (size_t i = x0; i < xend; i++) {
            const hp px = hp(xs[i * dim])     - hp(y[0]);
            const hp py = hp(xs[i * dim + 1]) - hp(y[1]);
            const hp pz = hp(xs[i * dim + 2]) - hp(y[2]);
            const hp d2 = sycl::fma(pz, pz, sycl::fma(py, py, px * px));

            const hp id = (d2 > hp(1e-24)) ? sycl::rsqrt(d2) : hp(0);
            const hp d  = d2 * id;
            const hp ddc = d - dc;                  // use hp to avoid cancellation

            const lp xn = lp(sycl::fma(pz, hp(ns[i * dim + 2]),
                              sycl::fma(py, hp(ns[i * dim + 1]), px * hp(ns[i * dim]))));
            const lp w  = lp(-(id * id) * dc) * xn;

            // (i*kappa - 1/r) = (-ki - id) + i*kr
            // MINUS to match NGSolve's dG/dn_y orientation
            const Tlp mk_minus_id = -Tlp(-ki - lp(id), kr);
            const Tlp geom = mk_minus_id * w;
            const Tlp osc = emkd(kr, ki, lp(ddc));
            const Tlp val = geom * osc;

            result += ws[i] * narrow(val);
        }
        return result;
    }

    template <typename TX>
    inline T CF(TX x) const
    {
        const lp d2 = sycl::fma(lp(x[2]), lp(x[2]),
                        sycl::fma(lp(x[1]), lp(x[1]), lp(x[0]) * lp(x[0])));

        if (sycl::fabs(d2) < lp(1e-14)) return 0;

        const lp id = sycl::rsqrt(d2);
        const lp d  = d2 * id;
        const lp kr = lp(k.real());
        const lp ki = lp(k.imag());

        const Tlp osc = emkd(kr, ki, d);
        const Tlp val = osc * (id * lp(INV_4PI));
        return narrow(val);
    }

    template <typename TX, typename TY>
    inline T transfer_factor(TX x, TY xc, PointScalar H,
                             TY pxc, PointScalar pH) const
    {
        (void)H; (void)pH;

        const hp zx  = hp(x[0]) - hp(xc[0]);
        const hp zy  = hp(x[1]) - hp(xc[1]);
        const hp zz  = hp(x[2]) - hp(xc[2]);
        const hp zpx = hp(x[0]) - hp(pxc[0]);
        const hp zpy = hp(x[1]) - hp(pxc[1]);
        const hp zpz = hp(x[2]) - hp(pxc[2]);

        const hp d  = sycl::sqrt(sycl::fma(zz, zz, sycl::fma(zy, zy, zx * zx)));
        const hp dp = sycl::sqrt(sycl::fma(zpz, zpz, sycl::fma(zpy, zpy, zpx * zpx)));

        if (sycl::fabs(d) < hp(1e-15)) return 0;

        const hp ddp  = d - dp;                 // use hp to avoid cancellation
        const lp dpod = lp(dp / d);

        const lp kr = lp(k.real());
        const lp ki = lp(k.imag());

        const Tlp val = emkd(kr, ki, lp(ddp)) * dpod;
        return narrow(val);
    }

private:
    std::complex<RealScalar> k;
};


// ===========================================================================
// Combined Field
// ===========================================================================
template <bool WithDecay>
class CombinedFieldHelmholtzKernelFunctions : public KernelTypes
{
public:
    static constexpr bool USES_NORMALS = true;

    explicit CombinedFieldHelmholtzKernelFunctions(std::complex<RealScalar> waveNr)
        : k(waveNr) {}

    inline T kernelFunction(const sycl::marray<hp, 3>& x,
                            const sycl::marray<hp, 3>& n) const
    {
        const hp d2 = sycl::fma(x[2], x[2], sycl::fma(x[1], x[1], x[0] * x[0]));
        const hp kr = hp(k.real());
        const hp ki = hp(k.imag());

        if constexpr (WithDecay) {
            if (sycl::fabs(d2) <= hp(1e-24)
                || sycl::sqrt(d2) * ki > hp(HIGH_EXP_CUTOFF)) {
                return RealScalar(0.0);
            }
        } else {
            if (sycl::fabs(d2) <= hp(1e-24)) {
                return RealScalar(0.0);
            }
        }

        const hp invd = sycl::rsqrt(d2);
        const hp d    = d2 * invd;
        // normal points opposite NGSolve's n_y
        const hp nxy  = -sycl::fma(n[2], x[2], sycl::fma(n[1], x[1], n[0] * x[0]));
        const hp f    = INV_4PI * invd * invd * invd;
        const Thp kd    = kt_hp(kr, ki, d);
        const Thp kd2   = kt_hp(kr, ki, d2);
        const Thp inner = Thp(hp(nxy)) * (Thp(1.0) + kd) + kd2;
        const Thp geom = Thp(f) * inner;
        const Tlp osc = emkd(lp(kr), lp(ki), lp(d));
        const Thp val = geom * Thp(hp(osc.real()), hp(osc.imag()));

        return narrow(val);
    }

    template <typename AT1, typename AT2, typename AT3, typename AT4>
    T evaluateKernel(const AT1& xs, size_t x0, size_t xend, const AT2& ys,
                     size_t y0, const AT3& ws, const AT4& ns) const
    {
        T result = 0;

        sycl::marray<hp, 3> pnt;
        sycl::marray<hp, 3> nrm;
        for (size_t i = x0; i < xend; i++) {
            for (int l = 0; l < dim; l++) {
                pnt[l] = hp(xs[i * dim + l]) - hp(ys[y0 * dim + l]);
                nrm[l] = hp(ns[i * dim + l]);
            }
            result += ws[i] * kernelFunction(pnt, nrm);
        }
        return result;
    }

    template <typename AT1, typename AT2, typename AT3>
    T evaluateFactoredKernel(const AT1& xs, size_t x0, size_t xend,
                             const sycl::marray<PointScalar, dim>& y,
                             const AT2& ws, const AT3& ns,
                             const sycl::marray<PointScalar, dim>& xc,
                             PointScalar H) const
    {
        (void)H;

        T result = 0;

        const hp dcx = hp(y[0]) - hp(xc[0]);
        const hp dcy = hp(y[1]) - hp(xc[1]);
        const hp dcz = hp(y[2]) - hp(xc[2]);
        const hp dc  = sycl::sqrt(sycl::fma(dcz, dcz, sycl::fma(dcy, dcy, dcx * dcx)));

        const lp kr = lp(k.real());
        const lp ki = lp(k.imag());

        for (size_t i = x0; i < xend; i++) {
            // x - y
            const hp zx = hp(xs[i * dim])     - hp(y[0]);
            const hp zy = hp(xs[i * dim + 1]) - hp(y[1]);
            const hp zz = hp(xs[i * dim + 2]) - hp(y[2]);
            const hp d2 = sycl::fma(zz, zz, sycl::fma(zy, zy, zx * zx));

            if (sycl::fabs(d2) < hp(1e-14)) continue;

            // normal points opposite NGSolve's n_y
            const lp nxy = -lp(sycl::fma(zz, hp(ns[i * dim + 2]),
                               sycl::fma(zy, hp(ns[i * dim + 1]), zx * hp(ns[i * dim]))));

            const hp id  = sycl::rsqrt(d2);
            const hp d   = d2 * id;
            const hp ddc = d - dc;                  // use hp to avoid cancellation
            const lp f   = lp(dc * id * id * id);
            const Tlp kd    = kt_lp(kr, ki, lp(d));
            const Tlp kd2   = kt_lp(kr, ki, lp(d2));
            const Tlp inner = Tlp(nxy) * (Tlp(1.0) + kd) + kd2;
            const Tlp geom = Tlp(f) * inner;
            const Tlp osc = emkd(kr, ki, lp(ddc));

            const Tlp val = geom * osc;
            result += ws[i] * narrow(val);
        }
        return result;
    }

    template <typename TX>
    inline T CF(TX x) const
    {
        const lp d2 = sycl::fma(lp(x[2]), lp(x[2]),
                        sycl::fma(lp(x[1]), lp(x[1]), lp(x[0]) * lp(x[0])));

        if (sycl::fabs(d2) < lp(1e-14)) return 0;

        const lp id = sycl::rsqrt(d2);
        const lp d  = d2 * id;
        const lp kr = lp(k.real());
        const lp ki = lp(k.imag());

        const Tlp osc = emkd(kr, ki, d);
        const Tlp val = osc * (id * lp(INV_4PI));
        return narrow(val);
    }

    template <typename TX, typename TY>
    inline T transfer_factor(TX x, TY xc, PointScalar H,
                             TY pxc, PointScalar pH) const
    {
        (void)H; (void)pH;

        const hp zx  = hp(x[0]) - hp(xc[0]);
        const hp zy  = hp(x[1]) - hp(xc[1]);
        const hp zz  = hp(x[2]) - hp(xc[2]);
        const hp zpx = hp(x[0]) - hp(pxc[0]);
        const hp zpy = hp(x[1]) - hp(pxc[1]);
        const hp zpz = hp(x[2]) - hp(pxc[2]);

        const hp d  = sycl::sqrt(sycl::fma(zz, zz, sycl::fma(zy, zy, zx * zx)));
        const hp dp = sycl::sqrt(sycl::fma(zpz, zpz, sycl::fma(zpy, zpy, zpx * zpx)));

        if (sycl::fabs(d) < hp(1e-15)) return 0;

        const hp ddp  = d - dp;                 // use hp to avoid cancellation
        const lp dpod = lp(dp / d);

        const lp kr = lp(k.real());
        const lp ki = lp(k.imag());

        const Tlp val = emkd(kr, ki, lp(ddp)) * dpod;
        return narrow(val);
    }

private:
    // helpers
    // exp(i*kappa*d) = exp(-ki*d)*(cos(kr*d) + i*sin(kr*d)), decay = ki
    inline Tlp emkd(lp kr, lp ki, lp d) const
    {
        const lp o = kr * d;
        const lp c = sycl::native::cos(o);
        const lp sn = sycl::native::sin(o);
        if constexpr (WithDecay) {
            return Tlp(sycl::native::exp(-ki * d)) * Tlp(c, sn);
        } else {
            (void)ki;
            return Tlp(c, sn);
        }
    }

    // -i*kappa*t = -i*(kr + i*ki)*t = (ki*t) + i*(-kr*t)
    inline Tlp kt_lp(lp kr, lp ki, lp t) const
    {
        // k_old * t = -i*kappa*t = (ki*t) + i*(-kr*t)
        if constexpr (WithDecay) {
            return Tlp(ki * t, -kr * t);
        } else {
            (void)ki;
            return Tlp(lp(0), -kr * t);
        }
    }

    // hp version
    inline Thp kt_hp(hp kr, hp ki, hp t) const
    {
        if constexpr (WithDecay) {
            return Thp(ki * t, -kr * t);
        } else {
            (void)ki;
            return Thp(hp(0), -kr * t);
        }
    }

    std::complex<RealScalar> k;
};

} // namespace ifgf_kernels

#endif
