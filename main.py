#!/usr/bin/env python3
"""Standalone BSM + Heston research calculations and CSV command-line workflow.

Education and research only; no investment advice or execution functionality.
Adapted from the supplied v6.2 research prototype. No broker SDK, credentials,
network requests, GUI, sample quotes or synthetic market-data fallback.
Inputs must be user-supplied observed quotes and explicit valuation assumptions.

Rates/volatilities are decimals; Theta is per day; Vega/Rho per percentage point.
Heston variance risk and initial-volatility risk have separate output labels.
The inherited FFT implementation remains a research numerical method; convergence
and independent pricing validation are necessary before reliance on its Greeks.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import norm as _norm
from scipy.optimize import brentq, minimize, differential_evolution

DISCLAIMER = "Education and research only. Not investment advice or production trading software."


def bs_price(S, K, T, sigma, r, q, call):
    """European BSM value, including exact expiry and zero-volatility limits."""
    if not all(math.isfinite(float(x)) for x in (S, K, T, sigma, r, q)):
        raise ValueError("Pricing inputs must be finite.")
    if S < 0 or K <= 0 or T < 0 or sigma < 0:
        raise ValueError("Require S>=0, K>0, T>=0 and sigma>=0.")
    if T == 0:
        return max(S-K, 0.0) if call else max(K-S, 0.0)
    if S == 0:
        return 0.0 if call else K*math.exp(-r*T)
    eq, er = math.exp(-q*T), math.exp(-r*T)
    if sigma < 1e-12:
        return max(S*eq-K*er, 0.0) if call else max(K*er-S*eq, 0.0)
    sq = math.sqrt(T)
    d1 = (math.log(S/K)+(r-q+0.5*sigma*sigma)*T)/(sigma*sq)
    d2 = d1-sigma*sq
    if call:
        return S*eq*_norm.cdf(d1)-K*er*_norm.cdf(d2)
    return K*er*_norm.cdf(-d2)-S*eq*_norm.cdf(-d1)


def price_bounds(S, K, T, r, q, call):
    """Discounted European no-arbitrage bounds."""
    a, b = S*math.exp(-q*T), K*math.exp(-r*T)
    return (max(a-b, 0.0), a) if call else (max(b-a, 0.0), b)


def iv_solve(mkt, S, K, T, r, q, call, tol=1e-7):
    """Observed/model price to BSM IV; invalid or degenerate inverses return NaN."""
    if not all(math.isfinite(float(x)) for x in (mkt,S,K,T,r,q)):
        return float("nan")
    if S<=0 or K<=0 or T<=0:
        return float("nan")
    low, high = price_bounds(S,K,T,r,q,call)
    if mkt<=low or mkt>=high:
        return float("nan")
    sig = 0.20
    for _ in range(80):
        px = bs_price(S,K,T,sig,r,q,call)
        diff = px-mkt
        if abs(diff)<tol:
            return sig
        d1=(math.log(S/K)+(r-q+0.5*sig*sig)*T)/(sig*math.sqrt(T))
        vg=S*math.exp(-q*T)*_norm.pdf(d1)*math.sqrt(T)
        if vg<1e-12:
            break
        candidate = sig-diff/vg
        if not (1e-6 < candidate < 20.0):
            break
        sig=candidate
    try:
        return float(brentq(lambda x:bs_price(S,K,T,x,r,q,call)-mkt,
                            1e-8,20.0,xtol=tol,maxiter=300))
    except (ValueError, RuntimeError):
        return float("nan")


def iv_otm(ce_mkt, pe_mkt, S, K, T, r, q):
    """Select OTM-side IV, averaging valid independent IVs within 0.5% of spot.

    No missing-side price is synthesized through parity.
    """
    if K<S*0.995:
        return iv_solve(pe_mkt,S,K,T,r,q,False)
    if K>S*1.005:
        return iv_solve(ce_mkt,S,K,T,r,q,True)
    vals=[iv_solve(ce_mkt,S,K,T,r,q,True),iv_solve(pe_mkt,S,K,T,r,q,False)]
    vals=[v for v in vals if math.isfinite(v)]
    return sum(vals)/len(vals) if vals else float("nan")

def bs_greeks(S, K, T, sigma, r, q, call):
    """Analytical BSM Greeks at sigma. Theta/day; Vega and Rho per pp.

    vanna is dDelta/dsigma (raw). volga is d2V/dsigma2 /100,
    inherited from the prototype: not curvature per squared volatility pp.
    """
    nan = {k: float("nan") for k in
           ("price","delta","gamma","theta","vega","rho","vanna","volga","d1","d2")}
    if T < 1e-8 or sigma < 1e-8 or S <= 0 or K <= 0:
        return nan
    sq  = math.sqrt(T)
    eq  = math.exp(-q*T);  er = math.exp(-r*T)
    d1  = (math.log(S/K) + (r - q + .5*sigma**2)*T) / (sigma*sq)
    d2  = d1 - sigma*sq
    pd  = _norm.pdf(d1)
    price = bs_price(S, K, T, sigma, r, q, call)
    gamma = eq * pd / (S * sigma * sq)
    vraw  = S * eq * pd * sq
    vega  = vraw / 100
    vanna = -eq * pd * d2 / sigma
    volga = vraw * d1 * d2 / sigma / 100
    if call:
        delta = eq * _norm.cdf(d1)
        theta = (-eq*S*pd*sigma/(2*sq) - r*K*er*_norm.cdf(d2)
                 + q*S*eq*_norm.cdf(d1)) / 365
        rho   = K * T * er * _norm.cdf(d2) / 100
    else:
        delta = eq * (_norm.cdf(d1) - 1)
        theta = (-eq*S*pd*sigma/(2*sq) + r*K*er*_norm.cdf(-d2)
                 - q*S*eq*_norm.cdf(-d1)) / 365
        rho   = -K * T * er * _norm.cdf(-d2) / 100
    return dict(price=price, delta=delta, gamma=gamma, theta=theta,
                vega=vega, rho=rho, vanna=vanna, volga=volga, d1=d1, d2=d2)

def model_iv_otm_from_prices(ce_prices, pe_prices, S, Ks, T, r, q):
    ce_prices = np.asarray(ce_prices, dtype=float)
    pe_prices = np.asarray(pe_prices, dtype=float)
    Ks = np.asarray(Ks, dtype=float)
    return np.asarray([iv_otm(float(c), float(p), S, float(K), T, r, q)
                       for c, p, K in zip(ce_prices, pe_prices, Ks)], dtype=float)

def clean_calibration_points(Ks, ivs, S, T, r, q):
    Ks = np.asarray(Ks, dtype=float)
    ivs = np.asarray(ivs, dtype=float)
    good = np.isfinite(ivs) & (ivs > 0.03) & (ivs < 1.50)
    good &= (Ks / S > 0.80) & (Ks / S < 1.20)
    vegas = []
    for K, iv in zip(Ks, ivs):
        if not np.isfinite(iv) or iv <= 0:
            vegas.append(0.0); continue
        sq = math.sqrt(max(T, 1e-8)); eq = math.exp(-q*T)
        d1 = (math.log(S/K)+(r-q+.5*iv**2)*T)/(iv*sq)
        vegas.append(S*eq*_norm.pdf(d1)*sq)
    good &= np.asarray(vegas) > 0.05
    return good

def smooth_iv_smile_for_heston(Ks, ivs, S, base_weights=None):
    Ks = np.asarray(Ks, dtype=float)
    ivs = np.asarray(ivs, dtype=float)
    x = np.log(Ks / S)

    if base_weights is None:
        w = np.ones_like(ivs, dtype=float)
    else:
        w = np.asarray(base_weights, dtype=float).copy()
        w = np.where(np.isfinite(w) & (w > 0), w, 1.0)

    deg = int(min(2, max(1, len(Ks) - 1)))
    fit_w = w.copy()
    for _ in range(4):
        coef = np.polyfit(x, ivs, deg=deg, w=np.sqrt(np.maximum(fit_w, 1e-12)))
        iv_fit = np.polyval(coef, x)
        resid = ivs - iv_fit
        med = np.median(resid)
        mad = np.median(np.abs(resid - med))
        scale = max(1.4826 * mad, 0.0025)
        robust = 1.0 / (1.0 + (resid / (2.0 * scale))**2)
        fit_w = w * np.clip(robust, 0.15, 1.0)

    iv_smooth = np.polyval(coef, x)
    resid = ivs - iv_smooth
    residual_bp = resid * 10000.0

    abs_bp = np.abs(residual_bp)
    blend_to_smooth = np.clip((abs_bp - 35.0) / 115.0, 0.0, 0.85)
    iv_target = (1.0 - blend_to_smooth) * ivs + blend_to_smooth * iv_smooth
    robust_weights = np.clip(1.0 - 0.70 * blend_to_smooth, 0.25, 1.0)
    return iv_target, robust_weights, iv_smooth, residual_bp

def heston_cf(u, S, T, r, q, kap, the, xi, rho, v0):
    """Log-domain Heston characteristic function (decaying-exponential form).

    Avoids separate exponentially large/small factors in the old hyperbolic
    implementation. Exact normalization and stock moment are handled explicitly.
    """
    uu = np.asarray(u, dtype=complex)
    zero = np.abs(uu) < 1e-14
    stock_moment = np.abs(uu+1j) < 1e-14
    special = zero | stock_moment
    z = np.where(special, 0.5+0j, uu)
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        beta = kap-rho*xi*1j*z
        d = np.sqrt(beta*beta+xi*xi*(z*z+1j*z))
        d = np.where(np.real(d)<0, -d, d)
        g = (beta-d)/(beta+d)
        decay = np.exp(-d*T)
        C = (1j*z*(np.log(S)+(r-q)*T)
             +kap*the/(xi*xi)*((beta-d)*T
             -2*(np.log1p(-g*decay)-np.log1p(-g))))
        D = (beta-d)/(xi*xi)*(1-decay)/(1-g*decay)
        result = np.exp(C+D*v0)
    result = np.where(zero, 1+0j, result)
    result = np.where(stock_moment, S*math.exp((r-q)*T)+0j, result)
    return result


def heston_fft(S, strikes, T, r, q, kap, the, xi, rho, v0,
               N=4096, eta=0.05, alpha=1.5):
    strikes = np.asarray(strikes, dtype=float)
    if (strikes <= 0).any() or not np.isfinite(strikes).all():
        raise ValueError("FFT strikes must be finite and positive.")
    if N < 16 or N & (N-1) or eta <= 0 or alpha <= 0:
        raise ValueError("FFT N must be a power of two >=16; eta and alpha positive.")
    if T <= 0:
        return np.maximum(S-strikes,0), np.maximum(strikes-S,0)
    lam = 2.*math.pi/(N*eta);  b = 0.5*N*lam;  ku = -b + lam*np.arange(N)
    vu  = np.arange(N, dtype=float) * eta
    d0  = np.zeros(N);  d0[0] = 1.
    # Carr-Madan/Simpson weights start 1,4,2,4,... (eta/3).
    w   = eta/3. * (3. - (-1.)**np.arange(N) - d0)
    cf  = heston_cf(vu-(alpha+1.)*1j, S, T, r, q, kap, the, xi, rho, v0)
    psi = math.exp(-r*T) * cf / (alpha**2 + alpha - vu**2 + 1j*(2.*alpha+1.)*vu)
    x   = w * psi * np.exp(1j*b*vu)
    cg  = np.exp(-alpha*ku)/math.pi * np.real(np.fft.fft(x))
    if not np.isfinite(cg).all():
        raise ValueError("Non-finite FFT prices; inspect parameters and numerical grid.")
    ce  = np.interp(np.log(strikes), ku, cg)
    fwd = S * math.exp((r-q)*T)
    ce  = np.maximum(ce, np.maximum((fwd - strikes)*math.exp(-r*T), 0.))
    pe  = ce - S*math.exp(-q*T) + strikes*math.exp(-r*T)
    pe  = np.maximum(pe, np.maximum((strikes*math.exp(-r*T) - S*math.exp(-q*T)), 0.))
    return ce, pe

def calibrate_heston(strikes, mkt_ivs_pct, S, T, r, q, log_cb=None):
    Ks    = np.asarray(strikes, dtype=float)
    iv_mk = np.asarray(mkt_ivs_pct, dtype=float) / 100.

    good = clean_calibration_points(Ks, iv_mk, S, T, r, q)
    if good.sum() < 5:
        if log_cb:
            log_cb(f"Heston calibration stopped: only {int(good.sum())} valid points. Need 5+.", "warn")
        return None
    Ks, iv_mk = Ks[good], iv_mk[good]

    mkt_ce = np.array([bs_price(S, K, T, iv, r, q, True)
                        for K, iv in zip(Ks, iv_mk)])

    def _vg(K, iv):
        sq = math.sqrt(T);  eq = math.exp(-q*T)
        d1 = (math.log(S/K)+(r-q+.5*iv**2)*T)/(iv*sq)
        return S*eq*_norm.pdf(d1)*sq
    vegas = np.array([_vg(K, iv) for K, iv in zip(Ks, iv_mk)])
    w_raw = np.maximum(vegas, 0.1)

    iv_target, robust_mult, iv_smooth, raw_minus_smooth_bp = \
        smooth_iv_smile_for_heston(Ks, iv_mk, S, base_weights=w_raw)

    outlier_bp = 150.0
    keep = np.abs(raw_minus_smooth_bp) <= outlier_bp
    if keep.sum() >= 5 and keep.sum() < len(Ks):
        Ks, iv_mk, w_raw = Ks[keep], iv_mk[keep], w_raw[keep]
        iv_target, robust_mult, iv_smooth, raw_minus_smooth_bp = \
            smooth_iv_smile_for_heston(Ks, iv_mk, S, base_weights=w_raw)

    w = w_raw * robust_mult
    w /= w.sum()

    atm_idx  = int(np.argmin(np.abs(Ks - S)))
    atm_iv   = iv_mk[atm_idx]
    v0_init  = float(atm_iv**2)
    logm     = np.log(Ks / S)
    slope    = float(np.polyfit(logm, iv_target, 1)[0])
    rho_init = float(np.clip(slope * 2.5, -0.95, 0.95))
    xi_init  = float(min(1.5, max(0.1, atm_iv * 3.0)))

    bounds = [
        (0.10, 15.00),
        (0.001, 0.60),
        (0.05,  2.00),
        (-0.90, 0.90),
        (0.001, 0.60),
    ]
    b_lo = np.array([b[0] for b in bounds])
    b_hi = np.array([b[1] for b in bounds])

    def objective(p, N=2048):
        kap, the, xi, rho, v0 = p
        if xi**2 > 2*kap*the*1.05: return 1e8
        if kap<=0 or the<=0 or xi<=0 or v0<=0: return 1e8
        try:
            ce, pe = heston_fft(S, Ks, T, r, q, kap, the, xi, rho, v0, N=N)
            iv_m  = model_iv_otm_from_prices(ce, pe, S, Ks, T, r, q)
            bad   = np.isnan(iv_m)|np.isinf(iv_m)|(iv_m<=0)
            if bad.any():
                return 1e6 + 1e5 * int(bad.sum())
            sse   = float(np.sum(w * (iv_m - iv_target)**2))
            fell      = max(0., xi**2 - 2*kap*the)**2 * 200.
            reg_rho   = 0.00020 * (rho / 0.75)**4
            reg_xi    = 0.00005 * (xi  / 0.80)**4
            reg_theta = 0.00005 * (math.log(max(the,1e-8) / max(v0_init,1e-8)))**2
            reg_v0    = 0.00010 * (math.log(max(v0, 1e-8) / max(v0_init,1e-8)))**2
            return sse + fell + reg_rho + reg_xi + reg_theta + reg_v0
        except (ValueError, FloatingPointError, OverflowError): return 1e8

    seeds = [
        [2.0, v0_init, xi_init,    rho_init, v0_init],
        [1.5, v0_init, 0.40,      -0.70,     v0_init],
        [3.0, v0_init, 0.60,      -0.50,     v0_init],
        [0.8, v0_init*2, 0.80,    -0.80,     v0_init],
    ]
    best_x = np.clip(seeds[0], b_lo, b_hi)
    best_v = 1e10
    for s0 in seeds:
        s0 = np.clip(s0, b_lo, b_hi)
        try:
            r1 = minimize(lambda p: objective(p, 2048), s0,
                          method="L-BFGS-B", bounds=bounds,
                          options={"maxiter":200,"ftol":1e-10,"gtol":1e-8})
            if r1.fun < best_v and r1.fun < 1e7:
                best_v = r1.fun;  best_x = r1.x.copy()
        except Exception: pass

    de = differential_evolution(
        lambda p: objective(p, 2048), bounds,
        maxiter=50, popsize=8, seed=42, tol=1e-7,
        mutation=(0.4, 1.0), recombination=0.85, polish=False)

    start_x = de.x if de.fun < best_v else best_x
    res = minimize(lambda p: objective(p, 4096), start_x,
                   method="L-BFGS-B", bounds=bounds,
                   options={"maxiter":500,"ftol":1e-14,"gtol":1e-11})
    # Compare all candidate parameter vectors on the SAME final grid.
    candidates = [best_x, de.x, res.x]
    scores = [objective(candidate, 4096) for candidate in candidates]
    selected = int(np.argmin(scores))
    best_x = np.asarray(candidates[selected], dtype=float)
    best_v = float(scores[selected])
    if not np.isfinite(best_v) or best_v >= 1e6:
        raise ValueError("Calibration failed to produce a finite valid solution.")

    kap, the, xi, rho, v0 = best_x
    ce_c, pe_c = heston_fft(S, Ks, T, r, q, kap, the, xi, rho, v0)
    iv_c    = model_iv_otm_from_prices(ce_c, pe_c, S, Ks, T, r, q)
    iv_bp   = np.abs(iv_c - iv_mk) * 10000.
    avg_bp  = float(np.nanmean(iv_bp))
    feller  = 2 * kap * the > xi**2
    grade   = ("EXCELLENT (<20bp)" if avg_bp < 20 else
               "GOOD (<50bp)"      if avg_bp < 50 else
               "ACCEPTABLE"        if avg_bp < 100 else "POOR")

    if log_cb:
        log_cb(f"κ={kap:.5f}  θ={the:.5f}  ξ={xi:.5f}  ρ={rho:.5f}  v₀={v0:.5f}", "ok")
        log_cb(f"avg IV err = {avg_bp:.1f} bp  →  {grade}", "ok")

    return dict(kappa=kap, theta=the, xi=xi, rho=rho, v0=v0,
               sse=best_v, avg_iv_bp=avg_bp, grade=grade,
               feller_ok=feller, ks=Ks, iv_mk=iv_mk, iv_c=iv_c, iv_bp=iv_bp)

def net_greeks(legs):
    keys = ("price","delta","gamma","theta","vega","rho","vanna","volga")
    net  = {k: 0. for k in keys}
    for lg in legs:
        for k in keys:
            net[k] += lg["qty"] * lg["greeks"].get(k, 0.)
    return net

def pnl_attr(delta, gamma, theta, vega, dS, dIV_pct, dt_days):
    d = delta * dS
    g = 0.5 * gamma * dS**2
    v = vega  * dIV_pct
    t = theta * dt_days
    return dict(delta=d, gamma=g, vega=v, theta=t, total=d+g+v+t)

PARAMETER_KEYS = ("kappa", "theta", "xi", "rho", "v0")


def validate_parameters(parameters):
    values = {key: float(parameters[key]) for key in PARAMETER_KEYS}
    if not all(math.isfinite(v) for v in values.values()):
        raise ValueError("Heston parameters must be finite.")
    if any(values[key] <= 0 for key in ("kappa", "theta", "xi", "v0")):
        raise ValueError("kappa, theta, xi and v0 must be positive.")
    if not -1 < values["rho"] < 1:
        raise ValueError("Heston rho must be strictly between -1 and 1.")
    return values


def heston_greeks(S, K, T, r, q, parameters, call=True, hS=None, hv=None):
    """Frozen-parameter FFT risk; initial variance and volatility are distinct.

    Retains the prototype FFT grid. Values require numerical convergence checks.
    The one-day Theta is a finite scenario, not an annual derivative/365.
    """
    p = validate_parameters(parameters)
    v0 = p["v0"]
    hS = min(1.0, S*0.01) if hS is None else float(hS)
    hv = min(0.001, v0*0.25) if hv is None else float(hv)
    if not 0 < hS < S or not 0 < hv < v0:
        raise ValueError("Greek bumps must preserve positive spot and variance.")
    def price(s=S, t=T, variance=v0, rate=r):
        if t <= 0:
            return bs_price(s, K, 0, 0, rate, q, call)
        c, put = heston_fft(s, [K], t, rate, q, p["kappa"],
                           p["theta"], p["xi"], p["rho"], variance)
        return float(c[0] if call else put[0])
    base, up, down = price(), price(s=S+hS), price(s=S-hS)
    vup, vdown = price(variance=v0+hv), price(variance=v0-hv)
    dv = (vup-vdown)/(2*hv)
    dv2 = (vup-2*base+vdown)/hv**2
    sigma0 = math.sqrt(v0)
    vanna_var = (
        price(s=S+hS, variance=v0+hv)-price(s=S-hS, variance=v0+hv)
        -price(s=S+hS, variance=v0-hv)+price(s=S-hS, variance=v0-hv)
    )/(4*hS*hv)
    hr = 0.0001
    return {
        "price": base, "delta": (up-down)/(2*hS),
        "gamma": (up-2*base+down)/hS**2,
        "theta_one_day": price(t=max(T-1/365, 0))-base,
        "variance_sensitivity_01": dv/100,
        "initial_vol_vega_01": 2*sigma0*dv/100,
        "rho_01": (price(rate=r+hr)-price(rate=r-hr))/(2*hr)/100,
        "vanna_variance_raw": vanna_var,
        "vanna_initial_vol_raw": 2*sigma0*vanna_var,
        "volga_variance_raw": dv2,
        "volga_initial_vol_raw": 2*dv+4*v0*dv2,
        "spot_bump": hS, "variance_bump": hv,
    }


STRATEGIES = (
    "Long Call", "Short Call", "Long Put", "Short Put",
    "Long Straddle", "Short Straddle", "Long Strangle", "Short Strangle",
    "Long Butterfly CE", "Short Butterfly CE",
    "Long Butterfly PE", "Short Butterfly PE",
    "Long Iron Condor", "Short Iron Condor",
    "Long Iron Butterfly", "Short Iron Butterfly",
    "Bull Call Spread", "Bear Call Spread", "Bull Put Spread", "Bear Put Spread",
    "Bull Risk Reversal", "Bear Risk Reversal",
    "Long Call Ratio 1x2", "Short Call Ratio 1x2",
    "Long Put Ratio 1x2", "Short Put Ratio 1x2",
    "Covered Call", "Protective Put", "Cash-Secured Put", "Collar",
    "Calendar Spread",
)


def strategy_spec(name, strikes):
    """Explicit legs for every supported label; never default to another payoff.

    Standard long/debit vs short/credit iron-condor naming is used here.
    Strikes must be ascending. Ratios: long means 1 short + 2 long backspread.
    Calendar is a long call calendar: short near call, long far call.
    Return (strike, CE/PE, signed quantity, near/far), stock_qty, reserved_cash.
    """
    name = name.replace("×", "x")
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {name}")
    single = {
        "Long Call": ("CE", 1), "Short Call": ("CE", -1),
        "Long Put": ("PE", 1), "Short Put": ("PE", -1),
        "Covered Call": ("CE", -1), "Protective Put": ("PE", 1),
        "Cash-Secured Put": ("PE", -1),
    }
    if name in single or name in ("Long Straddle", "Short Straddle", "Calendar Spread"):
        count = 1
    elif "Butterfly" in name:
        count = 3
    elif "Condor" in name:
        count = 4
    else:
        count = 2
    ks = tuple(float(k) for k in strikes)
    if len(ks) != count or any(not math.isfinite(k) or k <= 0 for k in ks):
        raise ValueError(f"{name} requires {count} positive strike(s).")
    if any(a >= b for a, b in zip(ks, ks[1:])):
        raise ValueError("Provide distinct strikes in ascending order.")
    if "Butterfly" in name and not math.isclose(ks[1]-ks[0], ks[2]-ks[1], rel_tol=1e-8):
        raise ValueError("Symmetric butterfly requires equally spaced strikes.")
    def leg(i, side, qty, maturity="near"):
        return (ks[i], side, qty, maturity)
    stock = 1.0 if name in ("Covered Call", "Protective Put", "Collar") else 0.0
    cash = ks[0] if name == "Cash-Secured Put" else 0.0
    if name in single:
        side, qty = single[name]
        spec = [leg(0, side, qty)]
    elif name == "Calendar Spread":
        spec = [leg(0,"CE",-1), leg(0,"CE",1,"far")]
    elif name in ("Long Straddle", "Short Straddle"):
        sign = 1 if name.startswith("Long") else -1
        spec = [leg(0,"CE",sign), leg(0,"PE",sign)]
    elif name in ("Long Strangle", "Short Strangle"):
        sign = 1 if name.startswith("Long") else -1
        spec = [leg(0,"PE",sign), leg(1,"CE",sign)]
    elif name in ("Long Butterfly CE","Short Butterfly CE","Long Butterfly PE","Short Butterfly PE"):
        sign = 1 if name.startswith("Long") else -1
        side = name[-2:]
        spec = [leg(0,side,sign),leg(1,side,-2*sign),leg(2,side,sign)]
    elif "Iron Butterfly" in name:
        sign = 1 if name.startswith("Long") else -1
        spec = [leg(0,"PE",-sign),leg(1,"PE",sign),
                leg(1,"CE",sign),leg(2,"CE",-sign)]
    elif "Iron Condor" in name:
        sign = 1 if name.startswith("Long") else -1
        spec = [leg(0,"PE",-sign),leg(1,"PE",sign),
                leg(2,"CE",sign),leg(3,"CE",-sign)]
    elif name == "Bull Call Spread":
        spec = [leg(0,"CE",1),leg(1,"CE",-1)]
    elif name == "Bear Call Spread":
        spec = [leg(0,"CE",-1),leg(1,"CE",1)]
    elif name == "Bull Put Spread":
        spec = [leg(0,"PE",1),leg(1,"PE",-1)]
    elif name == "Bear Put Spread":
        spec = [leg(0,"PE",-1),leg(1,"PE",1)]
    elif name == "Bull Risk Reversal":
        spec = [leg(0,"PE",-1),leg(1,"CE",1)]
    elif name == "Bear Risk Reversal":
        spec = [leg(0,"PE",1),leg(1,"CE",-1)]
    elif "Call Ratio" in name:
        sign = 1 if name.startswith("Long") else -1
        spec = [leg(0,"CE",-sign),leg(1,"CE",2*sign)]
    elif "Put Ratio" in name:
        sign = 1 if name.startswith("Long") else -1
        spec = [leg(0,"PE",2*sign),leg(1,"PE",-sign)]
    elif name == "Collar":
        spec = [leg(0,"PE",1),leg(1,"CE",-1)]
    else:
        raise ValueError(f"No explicit mapping for {name}")
    return spec, stock, cash


def build_strategy(name, strikes, observations, S, T, r, q,
                   far_observations=None, far_T=None):
    """Build from actual input premiums, with each leg's own raw-side BSM IV.

    A cash-secured put's cash requirement is disclosed but interest is not modeled.
    Quantities are per underlying unit; contract sizing is applied by the CLI.
    """
    spec, stock, cash = strategy_spec(name, strikes)
    near = {row["strike"]: row for row in observations}
    far = {row["strike"]: row for row in (far_observations or [])}
    if any(maturity == "far" for _,_,_,maturity in spec):
        if far_T is None or far_T <= T or not far:
            raise ValueError("Calendar requires actual far-expiry quotes and a later expiry.")
    legs = []
    for K, side, qty, maturity in spec:
        rows, maturity_T = (far, far_T) if maturity == "far" else (near, T)
        if K not in rows:
            raise ValueError(f"Missing observed {maturity} quote at strike {K}.")
        entry = rows[K]["ce" if side == "CE" else "pe"]
        sigma = iv_solve(entry, S, K, maturity_T, r, q, side == "CE")
        if not math.isfinite(sigma):
            raise ValueError(f"No valid IV for {maturity} {K} {side}.")
        g = bs_greeks(S,K,maturity_T,sigma,r,q,side == "CE")
        legs.append(dict(strike=K,opt_type=side,qty=qty,T=maturity_T,
                         sigma=sigma,r=r,q=q,entry_px=entry,greeks=g,
                         maturity=maturity))
    net = net_greeks(legs)
    net["price"] = sum(lg["qty"]*lg["entry_px"] for lg in legs)+stock*S
    net["delta"] += stock
    return dict(name=name, legs=legs, stock_qty=stock, cash_required=cash,
                entry_spot=S, net_greeks=net)


def strategy_pnl(legs, spots, dT=0, dIV=0.0):
    """BSM scenario P&L against observed entries; exact intrinsic at expiry.

    For calendars, paths beyond the first expiry are rejected by the caller.
    Earlier exercise, dividend reinvestment and physical settlement are not modeled.
    """
    if dT < 0:
        raise ValueError("Elapsed days must be non-negative.")
    out = np.zeros(len(spots))
    for lg in legs:
        t = max(lg["T"]-dT/365, 0.0)
        if t < 1e-12:
            t = 0.0
        sigma = lg["sigma"]+dIV
        if sigma <= 0:
            raise ValueError("Volatility shock produces non-positive volatility.")
        for i, spot in enumerate(spots):
            value = bs_price(float(spot),lg["strike"],t,sigma,
                             lg["r"],lg["q"],lg["opt_type"]=="CE")
            out[i] += lg["qty"]*(value-lg["entry_px"])
    return out


def read_quotes(path, S, T, r, q):
    """CSV header must contain strike,ce,pe; blank/invalid/missing data is rejected."""
    output, seen = [], set()
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"strike", "ce", "pe"}.issubset(reader.fieldnames or []):
            raise ValueError("Quote CSV requires columns strike,ce,pe.")
        for line, row in enumerate(reader, 2):
            try:
                values = {key: float(row[key]) for key in ("strike", "ce", "pe")}
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid or missing quote at CSV line {line}.") from error
            if not all(math.isfinite(x) for x in values.values()):
                raise ValueError(f"Non-finite input at CSV line {line}.")
            K=values["strike"]
            if K<=0 or K in seen:
                raise ValueError(f"Invalid or duplicate strike at CSV line {line}.")
            for key, call in (("ce", True),("pe", False)):
                low, high=price_bounds(S,K,T,r,q,call)
                if not low < values[key] < high:
                    raise ValueError(f"{key} quote at line {line} is outside strict BSM IV bounds ({low}, {high}).")
            seen.add(K)
            output.append(values)
    if not output:
        raise ValueError("Quote CSV has no observations.")
    return sorted(output,key=lambda x:x["strike"])


def analyse_chain(observations, S, T, r, q, parameters):
    p=validate_parameters(parameters)
    results=[]
    for row in observations:
        K=row["strike"]
        raw_ce_iv=iv_solve(row["ce"],S,K,T,r,q,True)
        raw_pe_iv=iv_solve(row["pe"],S,K,T,r,q,False)
        market_iv=iv_otm(row["ce"],row["pe"],S,K,T,r,q)
        hce,hpe=heston_fft(S,[K],T,r,q,*[p[key] for key in PARAMETER_KEYS])
        model_iv=iv_otm(float(hce[0]),float(hpe[0]),S,K,T,r,q)
        hc=heston_greeks(S,K,T,r,q,p,True)
        hp=heston_greeks(S,K,T,r,q,p,False)
        results.append(dict(
            strike=K,observed_ce=row["ce"],observed_pe=row["pe"],
            raw_ce_iv=raw_ce_iv,raw_pe_iv=raw_pe_iv,selected_market_iv=market_iv,
            parity_residual=row["ce"]-row["pe"]-S*math.exp(-q*T)+K*math.exp(-r*T),
            intrinsic_ce=max(S-K,0),intrinsic_pe=max(K-S,0),
            heston_ce=float(hce[0]),heston_pe=float(hpe[0]),heston_iv=model_iv,
            ce_error=float(hce[0])-row["ce"],pe_error=float(hpe[0])-row["pe"],
            iv_error_bp=abs(model_iv-market_iv)*10000,
            bsm_ce_at_raw_iv=bs_greeks(S,K,T,raw_ce_iv,r,q,True),
            bsm_pe_at_raw_iv=bs_greeks(S,K,T,raw_pe_iv,r,q,False),
            bsm_ce_at_selected_iv=bs_greeks(S,K,T,market_iv,r,q,True),
            bsm_pe_at_selected_iv=bs_greeks(S,K,T,market_iv,r,q,False),
            bsm_ce_at_heston_iv=bs_greeks(S,K,T,model_iv,r,q,True),
            bsm_pe_at_heston_iv=bs_greeks(S,K,T,model_iv,r,q,False),
            heston_ce_greeks=hc,heston_pe_greeks=hp))
    return results


def parse_instant(value):
    instant=datetime.fromisoformat(value.replace("Z","+00:00"))
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise ValueError("Dates require an explicit timezone offset, e.g. +05:30.")
    return instant.astimezone(timezone.utc)


def json_safe(value):
    if isinstance(value, np.ndarray):
        return [json_safe(x) for x in value.tolist()]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {str(key):json_safe(x) for key,x in value.items()}
    if isinstance(value,(list,tuple)):
        return [json_safe(x) for x in value]
    if isinstance(value,float) and not math.isfinite(value):
        return None
    return value


def flatten(row, prefix=""):
    output={}
    for key,value in row.items():
        name=f"{prefix}{key}"
        if isinstance(value,dict):
            output.update(flatten(value,name+"_"))
        else:
            output[name]=value
    return output


def write_csv(path, rows):
    if not rows:
        return
    columns=list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-strategies",action="store_true")
    parser.add_argument("--quotes",type=Path,help="Your observed option CSV: strike,ce,pe")
    parser.add_argument("--spot",type=float)
    parser.add_argument("--expiry",help="ISO expiry including exchange close time and offset")
    parser.add_argument("--as-of",help="ISO valuation time with offset; defaults to current UTC")
    parser.add_argument("--rate",type=float,help="Annual decimal continuously compounded rate")
    parser.add_argument("--dividend",type=float,default=0.0,help="Decimal yield; explicit default 0")
    parser.add_argument("--params",type=Path,help="Existing calibrated parameter JSON; otherwise fit your quotes")
    parser.add_argument("--strategy",help="Optional strategy; use --list-strategies")
    parser.add_argument("--strikes",type=float,nargs="+",help="Strategy strikes in ascending order")
    parser.add_argument("--far-quotes",type=Path,help="Actual far-expiry CSV for Calendar Spread")
    parser.add_argument("--far-expiry",help="Calendar far-expiry ISO timestamp")
    parser.add_argument("--lot-size",type=int,default=1,help="Underlying units per contract; default 1")
    parser.add_argument("--contracts",type=int,default=1)
    parser.add_argument("--spot-change",type=float,default=0.0)
    parser.add_argument("--iv-change-pp",type=float,default=0.0)
    parser.add_argument("--days",type=float,default=0.0)
    parser.add_argument("--output",type=Path,default=Path("results"))
    parser.add_argument("--overwrite",action="store_true")
    args=parser.parse_args(argv)
    if args.list_strategies:
        print("\n".join(STRATEGIES))
        return 0
    if args.quotes is None or args.spot is None or args.expiry is None or args.rate is None:
        parser.error("--quotes, --spot, --expiry and --rate are required.")
    try:
        S,r,q=args.spot,args.rate,args.dividend
        if S<=0 or not all(math.isfinite(x) for x in (S,r,q)):
            raise ValueError("Spot must be positive and valuation inputs finite.")
        if args.lot_size<=0 or args.contracts<=0:
            raise ValueError("Lot size and contracts must be positive integers.")
        if args.days<0 or not all(math.isfinite(x) for x in (args.days,args.spot_change,args.iv_change_pp)):
            raise ValueError("Scenario inputs must be finite; days must be non-negative.")
        asof=parse_instant(args.as_of) if args.as_of else datetime.now(timezone.utc)
        expiry=parse_instant(args.expiry)
        T=(expiry-asof).total_seconds()/(365*86400)
        if T<=0:
            raise ValueError("Expiry must be after the valuation timestamp.")
        if args.days>T*365:
            raise ValueError("Elapsed scenario days cannot exceed time to near expiry.")
        quotes=read_quotes(args.quotes,S,T,r,q)
        far_quotes,far_T=None,None
        if args.far_quotes or args.far_expiry:
            if not(args.far_quotes and args.far_expiry):
                raise ValueError("Provide both --far-quotes and --far-expiry.")
            far_T=(parse_instant(args.far_expiry)-asof).total_seconds()/(365*86400)
            if far_T<=T:
                raise ValueError("Calendar far expiry must follow near expiry.")
            far_quotes=read_quotes(args.far_quotes,S,far_T,r,q)
        portfolio=None
        if args.strategy:
            if not args.strikes:
                raise ValueError("A strategy requires --strikes.")
            portfolio=build_strategy(args.strategy,args.strikes,quotes,S,T,r,q,far_quotes,far_T)
            if args.days>T*365:
                raise ValueError("Strategy scenarios stop at the near expiry; settlement is not modeled.")
        targets=[args.output/"analysis.json",args.output/"chain_analysis.csv"]
        if portfolio:
            targets.append(args.output/"strategy_pnl.csv")
        if not args.overwrite and any(path.exists() for path in targets):
            raise ValueError("Output exists. Choose another --output folder or use --overwrite.")
        logs=[]
        def log(message, tag=""):
            logs.append(dict(message=message,tag=tag))
            print(message,file=sys.stderr)
        if args.params:
            with args.params.open(encoding="utf-8") as handle:
                p=validate_parameters(json.load(handle))
            calibration=dict(mode="user_supplied_parameters")
        else:
            selected=[iv_otm(row["ce"],row["pe"],S,row["strike"],T,r,q)*100 for row in quotes]
            log("Calibrating against your observed quote IVs...")
            calibration=calibrate_heston([row["strike"] for row in quotes],selected,S,T,r,q,log)
            if calibration is None:
                raise ValueError("Need at least five valid calibration strikes; alternatively supply --params.")
            p=validate_parameters(calibration)
        chain=analyse_chain(quotes,S,T,r,q,p)
        atm=min(quotes,key=lambda row:abs(row["strike"]-S))
        sigma=iv_solve(atm["ce"],S,atm["strike"],T,r,q,True)
        g=bs_greeks(S,atm["strike"],T,sigma,r,q,True)
        if S+args.spot_change<0 or sigma+args.iv_change_pp/100<=0:
            raise ValueError("Scenario produces invalid spot or volatility.")
        attr=pnl_attr(g["delta"],g["gamma"],g["theta"],g["vega"],
                      args.spot_change,args.iv_change_pp,args.days)
        full=bs_price(S+args.spot_change,atm["strike"],max(T-args.days/365,0),
                      sigma+args.iv_change_pp/100,r,q,True)-g["price"]
        scale=args.lot_size*args.contracts
        payload=dict(disclaimer=DISCLAIMER,
            inputs=dict(spot=S,rate=r,dividend=q,as_of=asof.isoformat(),
                        expiry=expiry.isoformat(),T=T,quote_file=args.quotes.name,
                        lot_size=args.lot_size,contracts=args.contracts),
            conventions=dict(volatility="decimal",vega="per volatility percentage point",
                heston_variance="separately labeled initial variance sensitivity",
                theta="BS derivative/day; Heston one-day repricing",
                bsm_volga="legacy raw volatility curvature divided by 100",
                pricing="European; frozen parameters for Heston Greeks"),
            numerical_warning="Inherited FFT interpolation requires grid and bump convergence checks.",
            parameters=p,calibration=calibration,
            feller_ok=2*p["kappa"]*p["theta"]>p["xi"]**2,
            chain=chain,calibration_log=logs,
            attribution=dict(strike=atm["strike"],scope="nearest-to-spot call, per underlying unit",
                shocks=dict(spot=args.spot_change,iv_percentage_points=args.iv_change_pp,days=args.days),
                approximate=attr,full_repricing=full,residual=full-attr["total"]))
        scenario_rows=[]
        if portfolio:
            spots=np.linspace(S*0.84,S*1.16,401)
            checkpoints=[("now",0.0),("seven_days",min(7.0,T*365)),("near_expiry",T*365)]
            curves={name:strategy_pnl(portfolio["legs"],spots,days,args.iv_change_pp/100)
                    +portfolio["stock_qty"]*(spots-S) for name,days in checkpoints}
            shocked=strategy_pnl(portfolio["legs"],[S+args.spot_change],args.days,args.iv_change_pp/100)[0]
            shocked+=portfolio["stock_qty"]*args.spot_change
            payload["strategy"]=dict(**portfolio,units=scale,
                position_greeks={key:val*scale for key,val in portfolio["net_greeks"].items()},
                cash_reserve_required=portfolio["cash_required"]*scale,
                scenario_pnl_per_unit=float(shocked),scenario_pnl_position=float(shocked*scale),
                note="Observed entry premiums; BSM per-leg IV; no fees, slippage, margin or cash interest.")
            for i,spot in enumerate(spots):
                scenario_rows.append(dict(spot=float(spot),**{
                    f"{name}_pnl_position":float(curve[i]*scale) for name,curve in curves.items()}))
        safe=json_safe(payload)
        args.output.mkdir(parents=True,exist_ok=True)
        (args.output/"analysis.json").write_text(json.dumps(safe,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        write_csv(args.output/"chain_analysis.csv",[flatten(row) for row in safe["chain"]])
        if portfolio:
            write_csv(args.output/"strategy_pnl.csv",scenario_rows)
        print(f"Processed {len(chain)} observed strikes. Results: {args.output.resolve()}")
        print(DISCLAIMER)
        return 0
    except (ValueError,KeyError,OSError,RuntimeError,FloatingPointError,OverflowError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
