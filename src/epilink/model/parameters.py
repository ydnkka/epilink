"""
Parameter definitions for infectiousness profile models.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NaturalHistoryParameters:
    r"""
    Parameters for the E/P/I variable infectiousness model.

    The model assumes stage durations follow Gamma distributions using the
    shape-scale parameterization.

    Attributes
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

    Notes
    -----
    Derived parameters are available as properties.

    presymptomatic_shape : float
        Derived shape parameter :math:`k_P = k_{inc} - k_E` of the presymptomatic
        infectious (``P``) stage.
    symptomatic_scale : float
        Derived scale :math:`\theta_I = 1 / (k_I \cdot \mu)` of the symptomatic
        infectious (``I``) stage duration.
    incubation_rate : float
        Derived rate :math:`\lambda = 1 / (k_{inc} \cdot \theta_{inc})` of
        the incubation period distribution.
    infectiousness_normalisation : float
        Derived normalisation constant :math:`C` ensuring the infectiousness
        profile integrates to unity over the infectious period:

        .. math::

            C = \frac{k_{inc} \cdot \lambda \cdot \mu}
                     {\alpha \cdot k_P \cdot \mu + k_{inc} \cdot \lambda}
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
            raise ValueError(
                "latent_shape must be < incubation_shape (so presymptomatic_shape is positive)."
            )
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
            self.rel_presymptomatic_infectiousness
            * self.presymptomatic_shape
            * self.symptomatic_rate
            + self.incubation_shape * self.incubation_rate
        )
        return numerator / denominator

    def __repr__(self) -> str:
        return (
            f"NaturalHistoryParameters("
            f"incubation_shape={self.incubation_shape}, "
            f"incubation_scale={self.incubation_scale}, "
            f"latent_shape={self.latent_shape}, "
            f"symptomatic_rate={self.symptomatic_rate}, "
            f"symptomatic_shape={self.symptomatic_shape}, "
            f"rel_presymptomatic_infectiousness={self.rel_presymptomatic_infectiousness})"
        )


def estimate_presymptomatic_transmission_fraction(p: NaturalHistoryParameters) -> float:
    r"""
    Calculate the presymptomatic transmission fraction.

    Parameters
    ----------
    p : NaturalHistoryParameters
        Model parameters containing stage shapes, scales, and relative
        infectiousness.

    Returns
    -------
    float
        The presymptomatic transmission fraction :math:`q_P \in [0, 1]`:

        .. math::

            q_P = \frac{\alpha \cdot k_P \cdot \mu}
                       {\alpha \cdot k_P \cdot \mu + k_{inc} \cdot \lambda}
    """

    num = p.rel_presymptomatic_infectiousness * p.presymptomatic_shape * p.symptomatic_rate
    den = num + (p.incubation_shape * p.incubation_rate)
    return num / den


__all__ = ["NaturalHistoryParameters", "estimate_presymptomatic_transmission_fraction"]
