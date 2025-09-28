"""
Variable infectiousness model (E/P/I) for SARS-CoV-2 [1].

Implements the mechanistic model in Hart et al. (2021) in which infectiousness
may differ between presymptomatic (P) and symptomatic (I) stages, with
Gamma-distributed durations for E, P, I, and constant infectiousness within P and I.

Provided:
- InfectiousnessParams: container for model parameters and derived constants.
- TOST: pdf/rvs for time from symptom onset to transmission (x).
- TOIT: pdf/rvs for time from start of presymptomatic stage to transmission (y*).
- Convenience samplers for stage durations and generation time (E + y*).

References
----------
.. [1] Hart WS, Maini PK, Thompson RN (2021).
       High infectiousness immediately before COVID-19 symptom onset highlights
       the importance of continued contact tracing. eLife, 10:e65534.
       https://doi.org/10.7554/eLife.65534
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from numpy.random import Generator, default_rng
from scipy import stats

# NumPy 2.0 deprecates np.trapz in favour of np.trapezoid.
# Provide a small compatibility shim.

_trapz = getattr(np, "trapezoid", None)
if _trapz is None:
    _trapz = getattr(np, "trapz", None)

ArrayLike = npt.ArrayLike


@dataclass(frozen=True)
class InfectiousnessParams:
    """
    Parameter set for the COVID-19 variable infectiousness model (E/P/I).

    Follows the parameterization in [1], with Gamma-distributed stage durations:
    - y_E ~ Gamma(k_E, scale_inc)
    - y_P ~ Gamma(k_P, scale_inc), where k_P = k_inc - k_E
    - y_I ~ Gamma(k_I, scale_I), with scale_I = 1/(k_I * mu)

    The incubation period t_inc = y_E + y_P ~ Gamma(k_inc, scale_inc).
    Infectiousness is constant within P and within I, with relative level alpha.

    Parameters
    ----------
    k_inc : float, default=5.807
        Shape parameter for the incubation period distribution.
    scale_inc : float, default=0.948
        Scale parameter for the incubation period distribution.
    k_E : float, default=3.38
        Shape parameter for the exposed (latent) period.
    mu : float, default=0.37
        Symptomatic removal rate (1 / mean symptomatic duration when k_I=1).
    k_I : float, default=1.0
        Shape for symptomatic infectious period (exponential if 1).
    alpha : float, default=2.29
        Relative infectiousness level of P vs I.

    Attributes
    ----------
    k_P : float
        Shape parameter for the presymptomatic period (k_inc - k_E).
    scale_I : float
        Scale parameter for the infectious period distribution (1 / (k_I * mu)).
    gamma_rate : float
        g in [1], defined as 1 / (k_inc * scale_inc). Note: avoids clashing with scipy.stats.gamma.
    C : float
        Normalization/combination constant used in pdf definitions:
        C = (k_inc * g * mu) / (alpha * k_P * mu + k_inc * g).
    """

    k_inc: float = 5.807
    scale_inc: float = 0.948
    k_E: float = 3.38
    mu: float = 0.37
    k_I: float = 1.0
    alpha: float = 2.29

    # Derived
    @property
    def k_P(self) -> float:
        return self.k_inc - self.k_E

    @property
    def scale_I(self) -> float:
        return 1.0 / (self.k_I * self.mu)

    @property
    def gamma_rate(self) -> float:
        # Paper’s “g” (rate), avoid name clash with scipy.stats.gamma
        return 1.0 / (self.k_inc * self.scale_inc)

    @property
    def C(self) -> float:
        # Normalization/combination constant used in pdf definitions
        return (self.k_inc * self.gamma_rate * self.mu) / (
            (self.alpha * self.k_P * self.mu) + (self.k_inc * self.gamma_rate)
        )


class InfectiousnessProfile:
    """
    Base class for infectiousness profile models.

    Provides frozen scipy distributions and a reproducible RNG.
    Subclasses must implement pdf() and rvs().
    """

    def __init__(
        self,
        a: float,
        b: float,
        params: InfectiousnessParams | None = None,
        rng: Generator | None = None,
        rng_seed: int | None = 12345,
    ):
        self.a = float(a)
        self.b = float(b)
        self.params = params or InfectiousnessParams()
        self.rng: Generator = rng if rng is not None else default_rng(rng_seed)

        # Frozen gamma distributions
        p = self.params
        self.dist_inc = stats.gamma(a=p.k_inc, scale=p.scale_inc)
        self.dist_E = stats.gamma(a=p.k_E, scale=p.scale_inc)
        self.dist_P = stats.gamma(a=p.k_P, scale=p.scale_inc)
        self.dist_I = stats.gamma(a=p.k_I, scale=p.scale_I)

    def pdf(self, x: ArrayLike) -> np.ndarray:
        raise NotImplementedError

    def rvs(self, size: int | tuple[int, ...] = (1,)) -> np.ndarray:
        raise NotImplementedError

    # Convenience samplers for components, using numpy RNG for speed
    def sample_incubation(self, size: int | tuple[int, ...] = 1) -> np.ndarray:
        # numpy’s gamma uses shape k and scale theta
        return self.rng.gamma(shape=self.params.k_inc, scale=self.params.scale_inc, size=size)

    def sample_E(self, size: int | tuple[int, ...] = 1) -> np.ndarray:
        return self.rng.gamma(shape=self.params.k_E, scale=self.params.scale_inc, size=size)

    def sample_P(self, size: int | tuple[int, ...] = 1) -> np.ndarray:
        return self.rng.gamma(shape=self.params.k_P, scale=self.params.scale_inc, size=size)

    def sample_I(self, size: int | tuple[int, ...] = 1) -> np.ndarray:
        return self.rng.gamma(shape=self.params.k_I, scale=self.params.scale_I, size=size)


class TOIT(InfectiousnessProfile):
    """
    Time from start of P (presymptomatic infectiousness) to Transmission (y*).

    Implements the integral expression in [1], Appendix (“Our mechanistic model”):
        f_*(y*) = C [ a (1 - F_P(y*)) + ∫_0^{y*} (1 - F_I(y* - y_P)) f_P(y_P) dy_P ], for y* >= 0,
                = 0 otherwise.

    Vectorized integral in pdf for speed. rvs samples from a discretized pdf on [a, b].
    """

    def __init__(
        self,
        a: float = 0.0,
        b: float = 30.0,
        params: InfectiousnessParams | None = None,
        rng: Generator | None = None,
        rng_seed: int | None = 12345,
        # Optional clock utilities (not used by pdf/rvs)
        subs_rate: float = 1e-3,  # per site per year (median)
        relax_rate: bool = False,  # use relaxed clock
        subs_rate_sigma: float = 0.33,  # lognormal sigma
        gen_len: int = 29901,  # genome length
        # Integration/sampling grid
        y_grid_points: int = 2048,  # grid for inner integral over yP
        x_grid_points: int = 1024,  # grid for discretized sampling over [a, b]
    ):
        super().__init__(a=a, b=b, params=params, rng=rng, rng_seed=rng_seed)
        self.subs_rate = float(subs_rate)
        self.relax_rate = bool(relax_rate)
        self.subs_rate_sigma = float(subs_rate_sigma)
        self.gen_len = int(gen_len)

        self.y_grid_points = int(y_grid_points)
        self.x_grid_points = int(x_grid_points)

        # Lognormal params for relaxed rate: mean=log(median) - 0.5*sigma^2
        self.subs_rate_mu = np.log(self.subs_rate) - 0.5 * (self.subs_rate_sigma**2)

        # Cached grids for rvs()
        self._x_grid: np.ndarray | None = None
        self._pdf_grid: np.ndarray | None = None

    # Molecular clock utilities
    def sample_clock_rate_per_day(self, size: int | tuple[int, ...] = 1) -> np.ndarray:
        """
        Returns substitution rate per day (sites/day).
        If relax_rate, draws from lognormal around subs_rate; else returns constant.
        """
        if self.relax_rate:
            per_site_per_year = self.rng.lognormal(
                mean=self.subs_rate_mu, sigma=self.subs_rate_sigma, size=size
            )
        else:
            per_site_per_year = np.full(size, self.subs_rate, dtype=float)
        return (per_site_per_year * self.gen_len) / 365.0

    def generation_time(self, size: int | tuple[int, ...] = 1) -> np.ndarray:
        """
        Generation time proxy: E stage duration + a draw from TOIT distribution.
        Note: TOIT.rvs is discrete; suitable for stochastic simulations.
        """
        return self.sample_E(size=size) + self.rvs(size=size)

    def pdf(self, x: ArrayLike) -> np.ndarray:
        """
        PDF of TOIT as in [1] (Appendix). Computed via vectorized trapezoidal rule.

        pdf(x) = C * [ alpha * (1 - F_P(x)) + ∫_0^x (1 - F_I(x - yP)) f_P(yP) dyP ], for x >= 0
               = 0 for x < 0
        """
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        out = np.zeros_like(x_arr)

        mask = x_arr >= 0
        if not np.any(mask):
            return out

        x_valid = x_arr[mask]
        xmax = float(np.max(x_valid))
        if xmax <= 0:
            return out  # all zero

        # Inner grid over yP in [0, xmax]
        yP = np.linspace(0.0, xmax, num=max(2, self.y_grid_points))
        f_P = self.dist_P.pdf(yP)  # shape (Ny,)

        # Build matrix F_I(x - yP) for all x in x_valid
        X = x_valid[:, None]  # (Nx, 1)
        Y = yP[None, :]  # (1, Ny)
        FI = self.dist_I.cdf(X - Y)  # (Nx, Ny)
        integrand = (1.0 - FI) * f_P[None, :]  # (Nx, Ny)

        # Zero out region where yP > x (optional; FI already handles negatives)
        integrand = np.where(Y <= X, integrand, 0.0)

        # Integrate over yP (axis=1)
        if _trapz is None:
            raise ImportError("Neither np.trapezoid nor np.trapz found in NumPy!")
        else:
            integral = _trapz(integrand, yP, axis=1)  # (Nx,)

        p = self.params
        out[mask] = p.C * (p.alpha * (1.0 - self.dist_P.cdf(x_valid)) + integral)
        return out

    def _ensure_grid(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Precompute and cache discretized pdf on a fixed x-grid over [a, b].
        """
        if self._x_grid is None or self._pdf_grid is None:
            x = np.linspace(self.a, self.b, num=max(2, self.x_grid_points))
            pdf_vals = self.pdf(x)
            # Guard against zero/negative or NaN pdf
            pdf_vals = np.clip(pdf_vals, a_min=0.0, a_max=np.inf)
            s = pdf_vals.sum()
            if not np.isfinite(s) or s <= 0.0:
                # Fallback to a near-uniform distribution if pdf degenerates
                pdf_vals = np.ones_like(x) / len(x)
            else:
                pdf_vals = pdf_vals / s
            self._x_grid = x
            self._pdf_grid = pdf_vals
        return self._x_grid, self._pdf_grid

    def rvs(self, size: int | tuple[int, ...] = (1,)) -> np.ndarray:
        if isinstance(size, int):
            size = (size,)
        x, probs = self._ensure_grid()
        return self.rng.choice(x, size=size, p=probs)


class TOST(InfectiousnessProfile):
    """
    Time from Symptom Onset to Transmission (x).

    Piecewise pdf from [1], Appendix (“Our mechanistic model”):
      - x < 0:  f_tost(x) = alpha * C * (1 - F_P(-x))
      - x >= 0: f_tost(x) = C * (1.0 - F_I(x))

    Sampling is done via discretized inverse transform on [a, b].
    """

    def __init__(
        self,
        a: float = -10.0,
        b: float = 10.0,
        params: InfectiousnessParams | None = None,
        rng: Generator | None = None,
        rng_seed: int | None = 12345,
        x_grid_points: int = 2048,
    ):
        super().__init__(a=a, b=b, params=params, rng=rng, rng_seed=rng_seed)
        self.x_grid_points = int(x_grid_points)
        self._x_grid: np.ndarray | None = None
        self._pdf_grid: np.ndarray | None = None

    def pdf(self, x: ArrayLike) -> np.ndarray:
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        p = self.params
        # Piecewise: x<0 uses F_P; x>=0 uses F_I
        pdf_vals = np.where(
            x_arr < 0.0,
            p.alpha * p.C * (1.0 - self.dist_P.cdf(-x_arr)),
            p.C * (1.0 - self.dist_I.cdf(x_arr)),
        )
        return np.clip(pdf_vals, a_min=0.0, a_max=np.inf)

    def _ensure_grid(self) -> tuple[np.ndarray, np.ndarray]:
        if self._x_grid is None or self._pdf_grid is None:
            x = np.linspace(self.a, self.b, num=max(2, self.x_grid_points))
            pdf_vals = self.pdf(x)
            pdf_vals = np.clip(pdf_vals, a_min=0.0, a_max=np.inf)
            s = pdf_vals.sum()
            if not np.isfinite(s) or s <= 0.0:
                pdf_vals = np.ones_like(x) / len(x)
            else:
                pdf_vals = pdf_vals / s
            self._x_grid = x
            self._pdf_grid = pdf_vals
        return self._x_grid, self._pdf_grid

    def rvs(self, size: int | tuple[int, ...] = (1,)) -> np.ndarray:
        if isinstance(size, int):
            size = (size,)
        x, probs = self._ensure_grid()
        return self.rng.choice(x, size=size, p=probs)


def presymptomatic_fraction(p: InfectiousnessParams) -> float:
    """
    Presymptomatic transmission fraction (for symptomatic cases).

    q_P = (alpha * k_P * mu) / (alpha * k_P * mu + k_inc * gamma_rate).
    See Hart et al. (2021), Appendix “Our mechanistic model”.
    """
    g = p.gamma_rate
    num = p.alpha * p.k_P * p.mu
    den = num + p.k_inc * g
    return num / den
