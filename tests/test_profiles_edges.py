# tests/test_profiles_edges.py
import numpy as np
from epilink import TOIT, TOST

def test_toit_pdf_negative_x():
    toit = TOIT()
    x = np.linspace(-5, -0.1, 10)  # all negative -> pdf 0
    pdf = toit.pdf(x)
    assert np.all(pdf == 0)

def test_toit_pdf_no_valid_mask():
    toit = TOIT()
    pdf = toit.pdf([-1.0])  # all < 0
    assert pdf.shape == (1,)
    assert pdf[0] == 0.0

def test_toit_grid_fallback_uniform():
    # Create parameters that zero the pdf over [a,b] to trigger uniform fallback
    # Make a=b so the computed pdf over grid is effectively zero area
    # Use tiny grid to avoid numerical noise
    t = TOIT(a=5.0, b=5.0, x_grid_points=2, y_grid_points=2)
    xs, ps = t._ensure_grid()
    # With a == b, grid collapses; fallback produces uniform probs
    assert xs.size == 2
    assert np.allclose(ps.sum(), 1.0)
    assert np.all(ps >= 0)

def test_tost_pdf_piecewise():
    tost = TOST()
    x = np.array([-1.0, 0.0, 1.0])
    pdf = tost.pdf(x)
    # left side uses P; right side uses I; both nonnegative
    assert pdf[0] >= 0 and pdf[1] >= 0 and pdf[2] >= 0

def test_clock_rate_relaxed_and_fixed():
    # Fixed
    t_fixed = TOIT(relax_rate=False)
    r_fixed = t_fixed.sample_clock_rate_per_day(size=1000)
    assert np.all(r_fixed == r_fixed[0])
    # Relaxed
    t_relax = TOIT(relax_rate=True)
    r_relax = t_relax.sample_clock_rate_per_day(size=1000)
    assert np.std(r_relax) > 0

def test_toit_pdf_no_valid_mask_single():
    t = TOIT()
    pdf = t.pdf([-1.0])
    assert pdf.shape == (1,)
    assert pdf[0] == 0.0

def test_tost_pdf_piecewise_nonnegative():
    tost = TOST()
    x = np.array([-1.0, 0.0, 1.0])
    pdf = tost.pdf(x)
    assert np.all(pdf >= 0.0)

def test_clock_rate_relaxed_and_fixed_distributions():
    t_fixed = TOIT(relax_rate=False)
    r_fixed = t_fixed.sample_clock_rate_per_day(size=1000)
    assert np.allclose(r_fixed, r_fixed[0])
    t_relax = TOIT(relax_rate=True)
    r_relax = t_relax.sample_clock_rate_per_day(size=2000)
    assert np.std(r_relax) > 0.0

def test_toit_sampling_grid_normalized():
    t = TOIT(a=0.0, b=5.0, x_grid_points=128, y_grid_points=128)
    xs, ps = t._ensure_grid()
    assert xs.shape == ps.shape
    assert np.all(ps >= 0.0)
    assert np.isclose(ps.sum(), 1.0, atol=1e-6)

