import numpy as np
from epilink import InfectiousnessParams, TOST, TOIT, presymptomatic_fraction

def test_tost_pdf_basic():
    p = InfectiousnessParams()
    tost = TOST(params=p)
    x = np.linspace(-10, 10, 256)
    pdf = tost.pdf(x)
    assert np.all(pdf >= 0)
    # Ensure there is some mass on both sides
    assert pdf[x < 0].max() > 0
    assert pdf[x >= 0].max() > 0

def test_toit_pdf_nonnegative_and_sampling():
    toit = TOIT()
    x = np.linspace(0, 30, 256)
    pdf = toit.pdf(x)
    assert np.all(pdf >= 0)
    s = toit.rvs(size=100).astype(float)
    assert s.shape == (100,)

def test_presymptomatic_fraction_in_0_1():
    p = InfectiousnessParams()
    q = presymptomatic_fraction(p)
    assert 0 <= q <= 1
