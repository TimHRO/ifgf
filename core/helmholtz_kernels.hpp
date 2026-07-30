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

    // 1/(4*pi), in hp (it multiplies the hp geometry prefactor).
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
// G(x-y) = 1/(4 pi) * exp(-k |x-y|) / |x-y|
// ===========================================================================
template <bool WithDecay>
class ModifiedHelmholtzKernelFunctions : public KernelTypes
{
public:
    static constexpr bool USES_NORMALS = false;

    explicit ModifiedHelmholtzKernelFunctions(std::complex<RealScalar> waveNr)
        : k(waveNr) {}

    inline Tlp emkd(lp kr, lp ki, lp d) const
    {
        if constexpr (WithDecay) {
            return Tlp(sycl::exp(-kr * d))
                 * Tlp(sycl::cos(ki * d), -sycl::sin(ki * d));
        } else {
            (void)kr;
            return Tlp(sycl::cos(ki * d), -sycl::sin(ki * d));
        }
    }

    inline T kernelFunction(const sycl::marray<hp, 3>& x) const
    {
        // geometry: hp
        const hp d  = sycl::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
        const hp kr = hp(k.real());
        const hp ki = hp(k.imag());

        if constexpr (WithDecay) {
            if (sycl::fabs(d) <= hp(1e-15) || d * kr > hp(HIGH_EXP_CUTOFF)) {
                return RealScalar(0.0);
            }
        } else {
            (void)kr;
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
        const hp dc  = sycl::sqrt(dcx * dcx + dcy * dcy + dcz * dcz);

        const hp kr = hp(k.real());
        const hp ki = hp(k.imag());

        for (size_t i = x0; i < xend; i++) {
            const hp px = hp(xs[i * dim])     - hp(y[0]);
            const hp py = hp(xs[i * dim + 1]) - hp(y[1]);
            const hp pz = hp(xs[i * dim + 2]) - hp(y[2]);
            const hp d  = sycl::sqrt(px * px + py * py + pz * pz);

            if (sycl::fabs(d) < hp(1e-15)) continue;

            const hp ddc  = d - dc;
            const hp dcod = dc / d;         

            const Tlp osc = emkd(lp(kr), lp(ki), lp(ddc));

            const Thp val = Thp(hp(osc.real()), hp(osc.imag())) * dcod;
            result += ws[i] * narrow(val);
        }
        return result;
    }

    template <typename TX>
    inline T CF(TX x) const
    {
        const hp d2 = hp(x[0]) * hp(x[0]) + hp(x[1]) * hp(x[1])
                    + hp(x[2]) * hp(x[2]);

        if (sycl::fabs(d2) < hp(1e-14)) return 0;

        const hp id = hp(1) / (sycl::sqrt(d2));
        const hp d  = d2 * id;
        const hp kr = hp(k.real());
        const hp ki = hp(k.imag());

        const Tlp osc = emkd(lp(kr), lp(ki), lp(d));
        const Thp val = Thp(hp(osc.real()), hp(osc.imag())) * (id * INV_4PI);
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

        const hp d  = sycl::sqrt(zx * zx + zy * zy + zz * zz);
        const hp dp = sycl::sqrt(zpx * zpx + zpy * zpy + zpz * zpz);

        if (sycl::fabs(d) < hp(1e-15)) return 0;

        const hp ddp  = d - dp;
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
// -1/(4 pi) * 1/d^2 * exp(-k d) * (-k - 1/d) * (x.n),   d = |x|
// ===========================================================================
template <bool WithDecay>
class DoubleLayerHelmholtzKernelFunctions : public KernelTypes
{
public:
    static constexpr bool USES_NORMALS = true;

    explicit DoubleLayerHelmholtzKernelFunctions(std::complex<RealScalar> waveNr)
        : k(waveNr) {}

    inline Tlp emkd(lp kr, lp ki, lp d) const
    {
        if constexpr (WithDecay) {
            return Tlp(sycl::exp(-kr * d))
                 * Tlp(sycl::cos(ki * d), -sycl::sin(ki * d));
        } else {
            (void)kr;
            return Tlp(sycl::cos(ki * d), -sycl::sin(ki * d));
        }
    }

    inline T kernelFunction(const sycl::marray<hp, 3>& x,
                            const sycl::marray<hp, 3>& n) const
    {
        const hp d2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
        const hp kr = hp(k.real());
        const hp ki = hp(k.imag());

        if constexpr (WithDecay) {
            if (sycl::fabs(d2) <= hp(1e-24)
                || sycl::sqrt(d2) * kr > hp(HIGH_EXP_CUTOFF)) {
                return RealScalar(0.0);
            }
        } else {
            if (sycl::fabs(d2) <= hp(1e-24)) {
                return RealScalar(0.0);
            }
        }

        const hp id = hp(1.0) / sycl::sqrt(d2);
        const hp d  = d2 * id;
        const hp xn = x[0] * n[0] + x[1] * n[1] + x[2] * n[2];
        const Thp mk_minus_id = Thp(-kr, -ki) - Thp(id);
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
        const hp dc  = sycl::sqrt(dcx * dcx + dcy * dcy + dcz * dcz);

        const hp kr = hp(k.real());
        const hp ki = hp(k.imag());

        for (size_t i = x0; i < xend; i++) {
            const hp px = hp(xs[i * dim])     - hp(y[0]);
            const hp py = hp(xs[i * dim + 1]) - hp(y[1]);
            const hp pz = hp(xs[i * dim + 2]) - hp(y[2]);
            const hp d2 = px * px + py * py + pz * pz;

            const hp id = (d2 > hp(1e-24)) ? hp(1.0) / sycl::sqrt(d2) : hp(0);
            const hp d  = d2 * id;

            const hp xn = px * hp(ns[i * dim])
                        + py * hp(ns[i * dim + 1])
                        + pz * hp(ns[i * dim + 2]);
            const hp w  = -(id * id) * dc * xn;
            const hp ddc = d - dc;

            const Thp mk_minus_id = Thp(-kr, -ki) - Thp(id);
            const Thp geom = mk_minus_id * hp(w);
            const Tlp osc = emkd(lp(kr), lp(ki), lp(ddc));
            const Thp val = geom * Thp(hp(osc.real()), hp(osc.imag()));

            result += ws[i] * narrow(val);
        }
        return result;
    }

    template <typename TX>
    inline T CF(TX x) const
    {
        const hp d2 = hp(x[0]) * hp(x[0]) + hp(x[1]) * hp(x[1])
                    + hp(x[2]) * hp(x[2]);

        if (sycl::fabs(d2) < hp(1e-14)) return 0;

        const hp id = hp(1) / (sycl::sqrt(d2));
        const hp d  = d2 * id;
        const hp kr = hp(k.real());
        const hp ki = hp(k.imag());

        const Tlp osc = emkd(lp(kr), lp(ki), lp(d));
        const Thp val = Thp(hp(osc.real()), hp(osc.imag())) * (id * INV_4PI);
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

        const hp d  = sycl::sqrt(zx * zx + zy * zy + zz * zz);
        const hp dp = sycl::sqrt(zpx * zpx + zpy * zpy + zpz * zpz);

        if (sycl::fabs(d) < hp(1e-15)) return 0;

        const hp ddp  = d - dp;
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
// Combined field, built on G = exp(-k r)/(4 pi r):
//   G_CF(x-y) = exp(-k r)/(4 pi r^3) * ( <n_y,x-y>(1 + k r) + k r^2 )
//   Coupling eta = -k (mirrors eta = i*kappa under i*kappa -> -k)
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
        const hp d2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
        const hp kr = hp(k.real());
        const hp ki = hp(k.imag());

        if constexpr (WithDecay) {
            if (sycl::fabs(d2) <= hp(1e-24)
                || sycl::sqrt(d2) * kr > hp(HIGH_EXP_CUTOFF)) {
                return RealScalar(0.0);
            }
        } else {
            if (sycl::fabs(d2) <= hp(1e-24)) {
                return RealScalar(0.0);
            }
        }

        const hp invd = hp(1.0) / sycl::sqrt(d2);
        const hp d    = d2 * invd;
        const hp nxy  = n[0] * x[0] + n[1] * x[1] + n[2] * x[2];
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
        const hp dc  = sycl::sqrt(dcx * dcx + dcy * dcy + dcz * dcz);

        const hp kr = hp(k.real());
        const hp ki = hp(k.imag());

        for (size_t i = x0; i < xend; i++) {
            // x - y
            const hp zx = hp(xs[i * dim])     - hp(y[0]);
            const hp zy = hp(xs[i * dim + 1]) - hp(y[1]);
            const hp zz = hp(xs[i * dim + 2]) - hp(y[2]);
            const hp d2 = zx * zx + zy * zy + zz * zz;

            if (sycl::fabs(d2) < hp(1e-14)) continue;

            const hp nxy = zx * hp(ns[i * dim])
                         + zy * hp(ns[i * dim + 1])
                         + zz * hp(ns[i * dim + 2]);

            const hp id = hp(1.0) / sycl::sqrt(d2);
            const hp d  = d2 * id;
            const hp ddc = d - dc;
            const hp f   = dc * id * id * id;
            const Thp kd    = kt_hp(kr, ki, d);
            const Thp kd2   = kt_hp(kr, ki, d2);
            const Thp inner = Thp(hp(nxy)) * (Thp(1.0) + kd) + kd2;
            const Thp geom = Thp(f) * inner;
            const Tlp osc = emkd(lp(kr), lp(ki), lp(ddc));

            const Thp val = geom * Thp(hp(osc.real()), hp(osc.imag()));
            result += ws[i] * narrow(val);
        }
        return result;
    }

    template <typename TX>
    inline T CF(TX x) const
    {
        const hp d2 = hp(x[0]) * hp(x[0]) + hp(x[1]) * hp(x[1])
                    + hp(x[2]) * hp(x[2]);

        if (sycl::fabs(d2) < hp(1e-14)) return 0;

        const hp id = hp(1) / (sycl::sqrt(d2));
        const hp d  = d2 * id;
        const hp kr = hp(k.real());
        const hp ki = hp(k.imag());

        const Tlp osc = emkd(lp(kr), lp(ki), lp(d));
        const Thp val = Thp(hp(osc.real()), hp(osc.imag())) * (id * INV_4PI);
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

        const hp d  = sycl::sqrt(zx * zx + zy * zy + zz * zz);
        const hp dp = sycl::sqrt(zpx * zpx + zpy * zpy + zpz * zpz);

        if (sycl::fabs(d) < hp(1e-15)) return 0;

        const hp ddp  = d - dp;
        const lp dpod = lp(dp / d);

        const lp kr = lp(k.real());
        const lp ki = lp(k.imag());

        const Tlp val = emkd(kr, ki, lp(ddp)) * dpod;
        return narrow(val);
    }

private:
    // helpers
    inline Tlp emkd(lp kr, lp ki, lp d) const
    {
        if constexpr (WithDecay) {
            return Tlp(sycl::exp(-kr * d))
                 * Tlp(sycl::cos(ki * d), -sycl::sin(ki * d));
        } else {
            (void)kr;
            return Tlp(sycl::cos(ki * d), -sycl::sin(ki * d));
        }
    }

    inline Thp kt_hp(hp kr, hp ki, hp t) const
    {
        if constexpr (WithDecay) {
            return Thp(kr * t, ki * t);
        } else {
            (void)kr;
            return Thp(hp(0), ki * t);
        }
    }

    std::complex<RealScalar> k;
};

} // namespace ifgf_kernels

#endif