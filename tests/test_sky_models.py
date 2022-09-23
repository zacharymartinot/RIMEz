import numpy as np
from RIMEz import sky_models

def test_random_power_law():
    """Test random_power_law function"""
    # initialize RNG seed
    np.random.seed(1)

    # get a power law
    S_min = 1e1
    S_max = 1e2
    alpha = -2.7
    pl = sky_models.random_power_law(S_min, S_max, alpha)
    # for S_min=1e1, S_max=1e2, alpha=-2.7, we get 12.20579531
    assert np.isclose(pl[0], 12.20579531)
    assert pl.size == 1

    return

def test_point_source_harmonics():
    np.random.seed(53)
    Nsrc = 10
    RA = np.random.rand(Nsrc) * 2*np.pi
    dec = np.random.rand(Nsrc) * np.pi - np.pi/2.
    x = np.linspace(0.5, 1.5, 201)
    alpha = np.random.randn(Nsrc)*0.2 - 0.8
    flux = np.random.randn(Nsrc) + 10
    I = flux[None,:] * x[:,None]**alpha[None,:]
    L = 200

    Ilm = sky_models.point_sources_harmonics_with_gridding(I, RA, dec, L)
    Ilm2 = sky_models.point_sources_harmonics(I, RA, dec, L)

    assert np.allclose(Ilm, Ilm2, atol=5e-9)
