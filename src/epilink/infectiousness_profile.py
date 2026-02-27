"""
E/P/I variable infectiousness model for SARS-CoV-2.

Implements the infectiousness model of Hart et al. (2021) with Gamma-distributed
durations for the latent (``E``), presymptomatic infectious (``P``), and
symptomatic infectious (``I``) stages. Infectiousness is piecewise-constant
within ``P`` and ``I``, with presymptomatic infectiousness scaled relative to the
symptomatic infectiousness.

Classes
-------
InfectiousnessParams
    Parameter container with derived quantities used in closed-form expressions.
MolecularClock
    Molecular clock model for substitution rates and expected mutations.
TOIT
    Distribution for time from start of presymptomatic stage to transmission.
TOST
    Distribution for time from symptom onset to transmission.

Functions
---------
presymptomatic_fraction
    Fraction of transmission occurring presymptomatically (conditional on symptomatic infection).

Notes
-----
Gamma distributions use SciPy/NumPy's ``Gamma(shape, scale)`` parametrization.

References
----------
Hart, W. S., Maini, P. K., & Thompson, R. N. (2021).
High infectiousness immediately before COVID-19 symptom onset highlights
the importance of continued contact tracing. Elife, 10, e65534.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from numpy.random import Generator, default_rng
from scipy import stats

# NumPy 2.0 deprecates np.trapz in favour of np.trapezoid.
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)

ArrayLike = npt.ArrayLike


@dataclass(frozen=True)
class InfectiousnessParams:
    r"""
    Parameters for the E/P/I variable infectiousness model.

    The model assumes stage durations follow Gamma distributions using the
    shape-scale parameterization.

    Parameters
    ----------
    incubation_shape : float, default=5.807
        Shape parameter :math:`k_{inc}` of the incubation period distribution.
    incubation_scale : float, default=0.948
        Scale parameter :math:`\theta_{inc}` of the incubation period distribution.
    latent_shape : float, default=3.38
        Shape parameter :math:`k_E` of the latent (``E``) stage.
        Must satisfy :math:`0 < k_E < k_{inc}`.
    symptomatic_rate : float, default=0.37
        Symptomatic removal rate :math:`\mu`.
        For :math:`k_I = 1`, the mean symptomatic duration is :math:`1/\mu`.
    symptomatic_shape : float, default=1.0
        Shape parameter :math:`k_I` of the symptomatic infectious (``I``) stage.
    rel_presymptomatic_infectiousness : float, default=2.29
        Relative infectiousness :math:`\alpha` of ``P`` compared with ``I``.

    Attributes
    ----------
    presymptomatic_shape : float
        Shape parameter of the presymptomatic (``P``) stage:
        :math:`k_P = k_{inc} - k_E`.
    symptomatic_scale : float
        Scale parameter of the symptomatic stage: :math:`\theta_I = 1 / (k_I \cdot \mu)`.
    incubation_rate : float
        Rate parameter :math:`\gamma`. Defined as
        :math:`\gamma = 1 / (k_{inc} \cdot \theta_{inc})`.
    infectiousness_normalisation : float
        Normalisation constant :math:`C = \beta_I / \beta_0` used in Hart et al. (2021).

    Notes
    -----
    The durations of the stages are distributed as follows:

    .. math::

        y_E &\sim \text{Gamma}(k_E, \theta_{inc}) \\
        y_P &\sim \text{Gamma}(k_P, \theta_{inc}) \\
        y_I &\sim \text{Gamma}(k_I, \theta_I)

    The incubation period :math:`\tau_{inc}` is the sum of the latent and
    presymptomatic periods:

    .. math::

        \tau_{inc} = y_E + y_P \sim \text{Gamma}(k_{inc}, \theta_{inc})

    The infectiousness normalisation constant :math:`C` is defined as:

    .. math::

        C = \frac{k_{inc} \cdot \gamma \cdot \mu}{\alpha \cdot k_P \cdot \mu + k_{inc} \cdot \gamma}

    **Mapping to Hart et al. (2021) notation:**

    =================================  ====================
    This code                          Hart et al. (2021)
    =================================  ====================
    incubation_shape                   :math:`k_{inc}`
    incubation_scale                   :math:`\theta_{inc}`
    latent_shape                       :math:`k_E`
    presymptomatic_shape               :math:`k_P`
    incubation_rate                    :math:`\gamma`
    symptomatic_rate                   :math:`\mu`
    symptomatic_shape                  :math:`k_I`
    rel_presymptomatic_infectiousness  :math:`\alpha`
    infectiousness_normalisation       :math:`C = \beta_I / \beta_0`
    =================================  ====================
    """

    incubation_shape: float = 5.807
    incubation_scale: float = 0.948
    latent_shape: float = 3.38
    symptomatic_rate: float = 0.37
    symptomatic_shape: float = 1.0
    rel_presymptomatic_infectiousness: float = 2.29

    def __post_init__(self) -> None:
        if self.incubation_shape <= 0 or self.incubation_scale <= 0:
            raise ValueError("incubation_shape and incubation_scale must be positive.")
        if self.latent_shape <= 0:
            raise ValueError("latent_shape must be positive.")
        if self.latent_shape >= self.incubation_shape:
            raise ValueError("latent_shape must be < incubation_shape (so presymptomatic_shape is positive).")
        if self.symptomatic_rate <= 0 or self.symptomatic_shape <= 0:
            raise ValueError("symptomatic_rate and symptomatic_shape must be positive.")
        if self.rel_presymptomatic_infectiousness <= 0:
            raise ValueError("rel_presymptomatic_infectiousness must be positive.")

    @property
    def presymptomatic_shape(self) -> float:
        return self.incubation_shape - self.latent_shape

    @property
    def symptomatic_scale(self) -> float:
        return 1.0 / (self.symptomatic_shape * self.symptomatic_rate)

    @property
    def incubation_rate(self) -> float:
        return 1.0 / (self.incubation_shape * self.incubation_scale)

    @property
    def infectiousness_normalisation(self) -> float:
        numerator = self.incubation_shape * self.incubation_rate * self.symptomatic_rate
        denominator = (
            self.rel_presymptomatic_infectiousness * self.presymptomatic_shape * self.symptomatic_rate
            + self.incubation_shape * self.incubation_rate
        )
        return numerator / denominator

    def __repr__(self) -> str:
        return (
            f"InfectiousnessParams("
            f"incubation_shape={self.incubation_shape}, "
            f"incubation_scale={self.incubation_scale}, "
            f"latent_shape={self.latent_shape}, "
            f"symptomatic_rate={self.symptomatic_rate}, "
            f"symptomatic_shape={self.symptomatic_shape}, "
            f"rel_presymptomatic_infectiousness={self.rel_presymptomatic_infectiousness})"
        )

class MolecularClock:
    r"""
    Molecular clock model for substitution rates and expected mutations.

    Supports both strict and relaxed (lognormal) molecular clocks for modelling
    substitution rates along phylogenetic branches.

    Parameters
    ----------
    subs_rate : float, default=1e-3
        Median substitution rate per site per year.
    relax_rate : bool, default=True
        Whether to draw clock rates from a lognormal distribution (relaxed clock).
        If False, uses a strict clock with a constant rate.
    subs_rate_sigma : float, default=0.33
        Lognormal dispersion (:math:`\sigma`) for the relaxed clock.
        Only used if ``relax_rate`` is True.
    gen_len : int, default=29903
        Genome length (number of sites) used for rate conversion.
        Default is SARS-CoV-2 genome length.
    rng : numpy.random.Generator, optional
        Random number generator. If None, a new generator is created.
    rng_seed : int, default=12345
        Seed for the RNG (ignored if ``rng`` is provided).

    Attributes
    ----------
    subs_rate_mu : float
        Mean parameter for the lognormal distribution, adjusted so that
        the median equals ``subs_rate``.

    References
    ----------
    Drummond, A. J., Ho, S. Y. W., Phillips, M. J., & Rambaut, A. (2006).
    Relaxed phylogenetics and dating with confidence. PLoS biology, 4(5), e88.
    """

    def __init__(
        self,
        subs_rate: float = 1e-3,
        relax_rate: bool = True,
        subs_rate_sigma: float = 0.33,
        gen_len: int = 29903,
        rng: Generator | None = None,
        rng_seed: int = 12345,
    ):
        self.subs_rate = float(subs_rate)
        self.relax_rate = bool(relax_rate)
        self.subs_rate_sigma = float(subs_rate_sigma)
        self.gen_len = int(gen_len)
        self.rng: Generator = rng if rng is not None else default_rng(rng_seed)

        # Validate before calculations
        if self.subs_rate <= 0:
            raise ValueError("subs_rate must be positive.")
        if self.subs_rate_sigma < 0:
            raise ValueError("subs_rate_sigma must be non-negative.")
        if self.gen_len <= 0:
            raise ValueError("gen_len must be positive.")

        # Adjust mu so that the median of lognormal equals subs_rate
        self.subs_rate_mu = np.log(self.subs_rate) - 0.5 * (self.subs_rate_sigma ** 2)

    def sample_clock_rate_per_day(self, size: int | tuple[int, ...] = 1) -> np.ndarray:
        r"""
        Sample substitution rates per day (mutations/day).

        If ``relax_rate`` is True, rates are drawn from a lognormal distribution with
        median ``subs_rate`` (per site per year) and dispersion ``subs_rate_sigma``.
        Otherwise, a strict clock with a constant rate is used.

        Parameters
        ----------
        size : int or tuple of int, default=1
            Output shape.

        Returns
        -------
        rates : numpy.ndarray
            Substitution rates in mutations per day.
        """
        if self.relax_rate:
            per_site_per_year = self.rng.lognormal(self.subs_rate_mu, self.subs_rate_sigma, size=size)
        else:
            per_site_per_year = np.full(size, self.subs_rate, dtype=float)
        return (per_site_per_year * self.gen_len) / 365.0

    def expected_mutations(self, times_in_days: ArrayLike, rates_per_day: ArrayLike | None = None) -> np.ndarray:
        r"""
        Compute expected mutation counts for given times (days).

        Parameters
        ----------
        times_in_days : array_like
            Times in days (e.g., TMRCAs or branch lengths).
        rates_per_day : array_like, optional
            Per-day substitution rates (mutations/day). If None, uses the
            deterministic rate derived from ``subs_rate`` and ``gen_len``.

        Returns
        -------
        expected : numpy.ndarray
            Expected mutation counts with the same shape as ``times_in_days``.

        Notes
        -----
        Expected mutations are calculated as:

        .. math::

            E[\text{mutations}] = \text{rate} \times \text{time}

        where rate is in mutations per day and time is in days.
        """
        times = np.asarray(times_in_days, dtype=float)
        if rates_per_day is None:
            rate = (self.subs_rate * self.gen_len) / 365.0
            mut = rate * times
        else:
            mut = np.asarray(rates_per_day, dtype=float) * times
        return np.clip(mut, a_min=0.0, a_max=np.inf)

    def __repr__(self) -> str:
        return (
            f"MolecularClock("
            f"subs_rate={self.subs_rate}, "
            f"relax_rate={self.relax_rate}, "
            f"subs_rate_sigma={self.subs_rate_sigma}, "
            f"gen_len={self.gen_len})"
        )

class InfectiousnessProfile:
    r"""
    Base class for infectiousness profile distributions.

    Provides shared parameters, a reproducible RNG, and frozen Gamma distributions
    for stage durations. Subclasses implement ``pdf`` and ``rvs``.

    Parameters
    ----------
    a, b : float
        Lower and upper bounds of the discretized support used by ``rvs``.
    params : InfectiousnessParams, optional
        Model parameters. If None, defaults to :class:`InfectiousnessParams`.
    rng : numpy.random.Generator, optional
        Random number generator. If None, a new generator is created.
    rng_seed : int or None, default=12345
        Seed for the RNG (ignored if ``rng`` is provided).
    grid_points : int, default=1024
        Number of grid points for discretized sampling over :math:`[a, b]`.

    Attributes
    ----------
    incubation : scipy.stats.rv_frozen
        Frozen Gamma distribution for :math:`\tau_{inc} \sim \text{Gamma}(k_{inc}, \theta_{inc})`.
    latent : scipy.stats.rv_frozen
        Frozen Gamma distribution for :math:`y_E \sim \text{Gamma}(k_E, \theta_{inc})`.
    presymptomatic : scipy.stats.rv_frozen
        Frozen Gamma distribution for :math:`y_P \sim \text{Gamma}(k_P, \theta_{inc})`.
    symptomatic : scipy.stats.rv_frozen
        Frozen Gamma distribution for :math:`y_I \sim \text{Gamma}(k_I, \theta_{I})`.

    Notes
    -----
    Convenience samplers (``sample_latent``, ``sample_presymptomatic``, ``sample_incubation``, ``sample_symptomatic``).
    """

    def __init__(
            self,
            a: float,
            b: float,
            params: InfectiousnessParams | None = None,
            rng: Generator | None = None,
            rng_seed: int | None = 12345,
            grid_points: int = 1024,
    ):
        self.a = float(a)
        self.b = float(b)
        self.params = params or InfectiousnessParams()
        self.rng: Generator = rng if rng is not None else default_rng(rng_seed)
        self.grid_points = int(grid_points)

        # Grid caching for discretized sampling
        self._grid: np.ndarray | None = None
        self._pdf_grid: np.ndarray | None = None

        p = self.params
        # frozen gamma distributions
        self.incubation = stats.gamma(a=p.incubation_shape, scale=p.incubation_scale)
        self.latent = stats.gamma(a=p.latent_shape, scale=p.incubation_scale)
        self.presymptomatic = stats.gamma(a=p.presymptomatic_shape, scale=p.incubation_scale)
        self.symptomatic = stats.gamma(a=p.symptomatic_shape, scale=p.symptomatic_scale)

    def pdf(self, x: ArrayLike) -> np.ndarray:
        """
        Evaluate the probability density function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the PDF (days).

        Returns
        -------
        pdf : numpy.ndarray
            Probability density values.
        """
        raise NotImplementedError

    def cdf(self, x: ArrayLike) -> np.ndarray:
        """
        Evaluate the cumulative distribution function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the CDF (days).

        Returns
        -------
        cdf : numpy.ndarray
            Cumulative probability values.

        Notes
        -----
        Default implementation uses numerical integration of the PDF.
        Subclasses may override with analytical expressions if available.
        """
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        result = np.zeros_like(x_arr)

        for i, x_val in enumerate(x_arr):
            if x_val <= self.a:
                result[i] = 0.0
            elif x_val >= self.b:
                result[i] = 1.0
            else:
                # Numerical integration from a to x_val
                grid_points = max(100, self.grid_points // 10)
                t = np.linspace(self.a, x_val, num=grid_points)
                pdf_vals = self.pdf(t)
                if _trapz is not None:
                    result[i] = _trapz(pdf_vals, t)
                else:
                    raise ImportError("Neither np.trapezoid nor np.trapz found in NumPy.")

        return np.clip(result, 0.0, 1.0)

    def mean(self) -> float:
        """
        Calculate the mean of the distribution.

        Returns
        -------
        mean : float
            Expected value (mean) in days.

        Notes
        -----
        Uses numerical integration over the support :math:`[a, b]`.
        """
        if _trapz is None:
            raise ImportError("Neither np.trapezoid nor np.trapz found in NumPy.")

        x = np.linspace(self.a, self.b, num=max(100, self.grid_points))
        pdf_vals = self.pdf(x)
        pdf_vals = pdf_vals / (pdf_vals.sum() + 1e-10)  # Normalize
        return float(_trapz(x * pdf_vals, x))

    def rvs(self, size: int | tuple[int, ...] = 1) -> np.ndarray:
        """
        Draw random variates from the distribution.

        Parameters
        ----------
        size : int or tuple of int, default=1
            Output shape.

        Returns
        -------
        samples : numpy.ndarray
            Random samples from the infectiousness profile.
        """
        raise NotImplementedError

    def sample_latent(self, size: int | tuple[int, ...] = 1) -> np.ndarray:
        r"""
        Draw latent durations.

        Parameters
        ----------
        size : int or tuple of int, default=1
            Output shape.

        Returns
        -------
        samples : numpy.ndarray
            Samples of :math:`y_E \sim \text{Gamma}(k_E, \theta_{inc})`.
        """
        return self.rng.gamma(shape=self.params.latent_shape, scale=self.params.incubation_scale, size=size)

    def sample_presymptomatic(self, size: int | tuple[int, ...] = 1) -> np.ndarray:
        r"""
        Draw presymptomatic durations.

        Parameters
        ----------
        size : int or tuple of int, default=1
            Output shape.

        Returns
        -------
        samples : numpy.ndarray
            Samples of :math:`y_P \sim \text{Gamma}(k_P, \theta_{inc})`.
        """
        return self.rng.gamma(shape=self.params.presymptomatic_shape, scale=self.params.incubation_scale, size=size)

    def sample_incubation(self, size: int | tuple[int, ...] = 1) -> np.ndarray:
        r"""
        Draw incubation periods.

        Samples represent the total time from infection to symptom onset:
        :math:`\tau_{inc} = y_E + y_P`.

        Parameters
        ----------
        size : int or tuple of int, default=1
            Output shape.

        Returns
        -------
        samples : numpy.ndarray
            Samples of :math:`\tau_{inc} \sim \text{Gamma}(k_{inc}, \theta_{inc})`.
        """
        return self.sample_latent(size) + self.sample_presymptomatic(size)

    def sample_symptomatic(self, size: int | tuple[int, ...] = 1) -> np.ndarray:
        r"""
        Draw symptomatic durations.

        Parameters
        ----------
        size : int or tuple of int, default=1
            Output shape.

        Returns
        -------
        samples : numpy.ndarray
            Samples of :math:`y_I \sim \text{Gamma}(k_I, \theta_I)`.
        """
        return self.rng.gamma(shape=self.params.symptomatic_shape, scale=self.params.symptomatic_scale, size=size)

    def _ensure_grid(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Precompute and cache a discretized PDF on a fixed grid.

        Returns
        -------
        grid : numpy.ndarray
            Grid values over ``[a, b]``.
        pdf : numpy.ndarray
            Normalized PDF values evaluated on ``grid``.
        """
        if self._grid is None or self._pdf_grid is None:
            x = np.linspace(self.a, self.b, num=max(2, self.grid_points))
            pdf_vals = np.clip(self.pdf(x), a_min=0.0, a_max=np.inf)
            s = pdf_vals.sum()
            pdf_vals = (np.ones_like(x) / len(x)) if (not np.isfinite(s) or s <= 0.0) else (pdf_vals / s)
            self._grid, self._pdf_grid = x, pdf_vals
        return self._grid, self._pdf_grid

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(a={self.a}, b={self.b}, params={self.params})"

class TOIT(InfectiousnessProfile):
    r"""
    Time from start of the presymptomatic stage to transmission (:math:`y^*`).

    This distribution represents the duration from the onset of presymptomatic
    infectiousness to a transmission event, as defined in Hart et al. (2021).

    Parameters
    ----------
    a : float, default=0.0
        Lower bound of the discretized support used by ``rvs``.
    b : float, default=60.0
        Upper bound of the discretized support used by ``rvs``.
    params : InfectiousnessParams, optional
        Model parameters. If None, defaults to :class:`InfectiousnessParams`.
    rng : numpy.random.Generator, optional
        Random number generator. If None, a new generator is created.
    rng_seed : int or None, default=12345
        Seed for the RNG (ignored if ``rng`` is provided).
    y_grid_points : int, default=2048
        Number of grid points for the numerical integration over :math:`y_P`.
    x_grid_points : int, default=1024
        Number of grid points for discretized sampling over :math:`[a, b]`.

    Notes
    -----
    The probability density function :math:`f_*(y^*)` is defined for
    :math:`y^* \geq 0` by the integral expression from Hart et al. (2021)
    (Appendix: "Our mechanistic model"):

    .. math::

       f^*(y^*) = C \left[ a (1 - F_P(y^*)) + \int_0^{y^*} (1 - F_I(y^* - y_P)) f_P(y_P)dy_P \right]

    where:

    - :math:`C` is the normalisation constant.
    - :math:`a` is the relative presymptomatic infectiousness.
    - :math:`F_P, F_I` are the CDFs of the ``P`` and ``I`` stage durations.
    - :math:`f_P` is the PDF of the presymptomatic stage duration.

    The implementation evaluates this integral numerically using a
    vectorized trapezoidal rule.
    """

    def __init__(
            self,
            a: float = 0.0,
            b: float = 60.0,
            params: InfectiousnessParams | None = None,
            rng: Generator | None = None,
            rng_seed: int | None = 12345,
            y_grid_points: int = 2048,
            x_grid_points: int = 1024,
    ):
        super().__init__(a=a, b=b, params=params, rng=rng, rng_seed=rng_seed, grid_points=x_grid_points)
        self.y_grid_points = int(y_grid_points)

    def generation_time(self, size: int | tuple[int, ...] = 1) -> np.ndarray:
        r"""
        Generate generation time samples.

        Generation time is calculated as :math:`\tau_{gen} = y_E + y^*`, where
        :math:`y_E` is the latent period and :math:`y^*` is the time from
        onset of presymptomatic infectiousness to transmission.

        Parameters
        ----------
        size : int or tuple of int, default=1
            Output shape.

        Returns
        -------
        samples : numpy.ndarray
            Generation time samples in days.
        """
        return self.sample_latent(size=size) + self.rvs(size=size)

    def pdf(self, y: ArrayLike) -> np.ndarray:
        r"""
        Evaluate the TOIT PDF at ``y`` (days).

        Uses the vectorized trapezoidal rule to approximate the integral in the paper.
        Returns zero for ``y < 0``.

        Parameters
        ----------
        y : array_like
            Points at which to evaluate the PDF (days).

        Returns
        -------
        pdf : numpy.ndarray
            Probability density values.
        """
        y_arr = np.asarray(y, dtype=float)
        out = np.zeros_like(y_arr)

        mask = y_arr >= 0
        if not np.any(mask):
            return out

        y_valid = y_arr[mask]
        y_max = float(np.max(y_valid))
        if y_max <= 0:
            return out

        if _trapz is None:
            raise ImportError("Neither np.trapezoid nor np.trapz found in NumPy.")

        # Create grid over presymptomatic duration in [0, y_max]
        presymp_duration_grid = np.linspace(0.0, y_max, num=max(2, self.y_grid_points))
        presymp_pdf = self.presymptomatic.pdf(presymp_duration_grid)

        # Broadcasting: y_valid as column vector, presymp_duration_grid as row vector
        y_broadcast = y_valid[:, None]
        presymp_broadcast = presymp_duration_grid[None, :]

        # Build matrix of F_I(y - presymp_duration) for all y in y_valid
        symptomatic_cdf = self.symptomatic.cdf(y_broadcast - presymp_broadcast)
        integrand = np.where(
            presymp_broadcast <= y_broadcast,
            (1.0 - symptomatic_cdf) * presymp_pdf[None, :],
            0.0
        )

        integral = _trapz(integrand, presymp_duration_grid, axis=1)

        p = self.params
        out[mask] = p.infectiousness_normalisation * (
            p.rel_presymptomatic_infectiousness * (1.0 - self.presymptomatic.cdf(y_valid)) + integral
        )
        return out

    def rvs(self, size: int | tuple[int, ...] = 1) -> np.ndarray:
        """
        Draw random variates from a discretized approximation.

        Parameters
        ----------
        size : int or tuple of int, default=1
            Output shape.

        Returns
        -------
        samples : numpy.ndarray
            Random samples from the TOIT distribution.
        """
        if isinstance(size, int):
            size = (size,)
        x, probs = self._ensure_grid()
        return self.rng.choice(x, size=size, p=probs)

class TOST(InfectiousnessProfile):
    r"""
    Time from symptom onset to transmission (:math:`x`).

    This distribution represents the interval between the onset of symptoms
    in the infector and the timing of a transmission event. Negative values
    indicate presymptomatic transmission.

    Parameters
    ----------
    a : float, default=-30.0
        Lower bound of the discretized support used by ``rvs``.
    b : float, default=30.0
        Upper bound of the discretized support used by ``rvs``.
    params : InfectiousnessParams, optional
        Model parameters. If None, defaults to :class:`InfectiousnessParams`.
    rng : numpy.random.Generator, optional
        Random number generator. If None, a new generator is created.
    rng_seed : int or None, default=12345
        Seed for the RNG (ignored if ``rng`` is provided).
    grid_points : int, default=2048
        Number of grid points for discretized sampling over :math:`[a, b]`.

    Notes
    -----
    The probability density function :math:`f(x)` follows the piecewise
    form defined in Hart et al. (2021):

    .. math::

       f(x) = \begin{cases}
       \alpha C (1 - F_P(-x)) & \text{for } x < 0 \\
       C (1 - F_I(x)) & \text{for } x \geq 0
       \end{cases}

    where:

    - :math:`\alpha` is the relative presymptomatic infectiousness.
    - :math:`C` is the infectiousness normalisation constant.
    - :math:`F_P` is the CDF of the presymptomatic stage duration.
    - :math:`F_I` is the CDF of the symptomatic stage duration.

    Random variates are drawn from a discretized approximation on a fixed
    grid over the support :math:`[a, b]`.
    """

    def __init__(
        self,
        a: float = -30.0,
        b: float = 30.0,
        params: InfectiousnessParams | None = None,
        rng: Generator | None = None,
        rng_seed: int | None = 12345,
        grid_points: int = 2048,
    ):
        super().__init__(a=a, b=b, params=params, rng=rng, rng_seed=rng_seed, grid_points=grid_points)

    def pdf(self, x: ArrayLike) -> np.ndarray:
        """
        Evaluate the TOST PDF at ``x`` (days).

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the PDF (days).

        Returns
        -------
        pdf : numpy.ndarray
            Probability density values.
        """
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        p = self.params
        pdf_vals = np.where(
            x_arr < 0.0,
            p.rel_presymptomatic_infectiousness
            * p.infectiousness_normalisation
            * (1.0 - self.presymptomatic.cdf(-x_arr)),
            p.infectiousness_normalisation * (1.0 - self.symptomatic.cdf(x_arr)),
        )
        return np.clip(pdf_vals, a_min=0.0, a_max=np.inf)

    def rvs(self, size: int | tuple[int, ...] = 1) -> np.ndarray:
        """
        Draw random variates from a discretized approximation.

        Parameters
        ----------
        size : int or tuple of int, default=1
            Output shape.

        Returns
        -------
        samples : numpy.ndarray
            Random samples from the TOST distribution.
        """
        if isinstance(size, int):
            size = (size,)
        x, probs = self._ensure_grid()
        return self.rng.choice(x, size=size, p=probs)

def presymptomatic_fraction(p: InfectiousnessParams) -> float:
    r"""
    Calculate the presymptomatic transmission fraction.

    Computes the proportion of transmissions occurring before symptom onset,
    conditional on the individual eventually becoming symptomatic.

    Parameters
    ----------
    p : InfectiousnessParams
        Model parameters containing stage shapes, scales, and relative
        infectiousness.

    Returns
    -------
    fraction : float
        The presymptomatic transmission fraction :math:`q_P \in [0, 1]`.

    Notes
    -----
    This implements the analytical expression from the Appendix of
    Hart et al. (2021):

    .. math::

        q_P = \frac{\alpha k_P \mu}{\alpha k_P \mu + k_{inc} \gamma}

    where:

    - :math:`\alpha` is the relative presymptomatic infectiousness.
    - :math:`k_P` is the presymptomatic shape (:math:`k_{inc} - k_E`).
    - :math:`\mu` is the symptomatic removal rate.
    - :math:`k_{inc}` is the incubation shape.
    - :math:`\gamma` is the incubation rate (:math:`1 / (k_{inc} \theta_{inc})`).
    """
    # Calculation using the notation mapped in the docstring
    num = p.rel_presymptomatic_infectiousness * p.presymptomatic_shape * p.symptomatic_rate
    den = num + (p.incubation_shape * p.incubation_rate)
    return num / den
