from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Real


@dataclass(frozen=True, slots=True, kw_only=True)
class NaturalHistoryParameters:
    r"""
    Parameters for the E/P/I variable infectiousness model [1].

    The model assumes stage durations follow Gamma distributions using the
    shape-scale parameterization.

    Attributes
    ----------
    incubation_shape : float, default=5.807
        Shape parameter :math:`k_{inc}` of the incubation period distribution.
        Must be a positive real number (dimensionless).
    incubation_scale : float, default=0.948
        Scale parameter :math:`\theta_{inc}` of the incubation period distribution.
        Must be a positive real number (days).
    latent_shape : float, default=3.38
        Shape parameter :math:`k_E` of the latent (``E``) stage.
        Must satisfy :math:`0 < k_E < k_{inc}` (dimensionless).
    symptomatic_rate : float, default=0.37
        Symptomatic removal rate :math:`\mu`.
        For :math:`k_I = 1`, the mean symptomatic duration is :math:`1/\mu`.
        Must be a positive real number (1/day).
    symptomatic_shape : float, default=1.0
        Shape parameter :math:`k_I` of the symptomatic infectious (``I``) stage.
        Must be a positive real number (dimensionless).
    transmission_rate_ratio : float, default=2.29
        Ratio of transmission rates in presymptomatic (``P``) and symptomatic
        infectious (``I``) stages, :math:`\alpha` for ``P`` compared with ``I``.
        Must be a positive real number (dimensionless).
    testing_delay_shape : float, default=2.0
        Shape parameter for the Gamma-distributed testing delay.
        Must be a positive real number (dimensionless).
    testing_delay_scale : float, default=1.0
        Scale parameter for the Gamma-distributed testing delay.
        Must be a positive real number (days).
    substitution_rate : float, default=1e-3
        Median substitution rate in substitutions per site per year.
        Must be a positive real number (substitutions/site/year).
    relaxation: float, default=0.33
        Lognormal standard deviation for relaxed clock branch-specific rates.
        Must be a non-negative real number (dimensionless).
        Set to 0 to use strict clock.
    genome_length: int, default=29903
        Genome length in sites. Must be a positive integer.

    Notes
    -----
    Derived parameters are exposed as read-only properties.

    `presymptomatic_shape`
        :math:`k_P = k_{inc} - k_E` for the presymptomatic infectious (``P``) stage.

    `symptomatic_scale`
        :math:`\theta_I = 1 / (k_I \cdot \mu)` for symptomatic stage duration (days).

    `incubation_rate`
        :math:`\lambda = 1 / (k_{inc} \cdot \theta_{inc})` (1/day).

    `infectiousness_normalisation`
        Normalisation constant :math:`C` such that the infectiousness profile integrates
        to unity over the infectious period:

        .. math::

            C = \frac{k_{inc} \cdot \lambda \cdot \mu}
            {\alpha \cdot k_P \cdot \mu + k_{inc} \cdot \lambda}

    `presymptomatic_transmission_fraction`
        Presymptomatic transmission fraction :math:`q_P \in [0, 1]`:

        .. math::

            q_P = \frac{\alpha \cdot k_P \cdot \mu}
            {\alpha \cdot k_P \cdot \mu + k_{inc} \cdot \lambda}

    References
    ----------
    .. [1] William S HartPhilip K MainiRobin N Thompson (2021)
        High infectiousness immediately before COVID-19 symptom onset
        highlights the importance of continued contact tracing eLife 10:e65534.
    """

    incubation_shape: float = 5.807
    incubation_scale: float = 0.948
    latent_shape: float = 3.38
    symptomatic_rate: float = 0.37
    symptomatic_shape: float = 1.0
    transmission_rate_ratio: float = 2.29
    testing_delay_shape: float = 2.0
    testing_delay_scale: float = 1.0
    substitution_rate: float = 1e-3
    relaxation: float = 0.33
    genome_length: int = 29903

    def __post_init__(self) -> None:
        positive_real_fields = (
            "incubation_shape",
            "incubation_scale",
            "latent_shape",
            "symptomatic_rate",
            "symptomatic_shape",
            "transmission_rate_ratio",
            "testing_delay_shape",
            "testing_delay_scale",
            "substitution_rate",
            "genome_length",
        )
        for field_name in positive_real_fields:
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{field_name} must be a real number.")
            if value <= 0:
                raise ValueError(f"{field_name} must be positive.")

        if isinstance(self.relaxation, bool) or not isinstance(self.relaxation, Real):
            raise TypeError("relaxation must be a real number.")
        if self.relaxation < 0:
            raise ValueError("relaxation must be non-negative.")

        if self.latent_shape >= self.incubation_shape:
            raise ValueError(
                "latent_shape must be < incubation_shape (so presymptomatic_shape is positive)."
            )

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
        numerator = (
                self.incubation_shape
                * self.incubation_rate
                * self.symptomatic_rate
        )
        denominator = (
            self.transmission_rate_ratio
            * self.presymptomatic_shape
            * self.symptomatic_rate
            + self.incubation_shape * self.incubation_rate
        )
        return numerator / denominator

    @property
    def presymptomatic_transmission_fraction(self) -> float:
        num = self.transmission_rate_ratio * self.presymptomatic_shape * self.symptomatic_rate
        den = num + (self.incubation_shape * self.incubation_rate)
        return num / den

    def to_dict(self) -> dict[str, float]:
        """Return constructor parameters as a plain dictionary."""
        return asdict(self)


__all__ = ["NaturalHistoryParameters"]
