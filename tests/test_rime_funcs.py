import pytest

import numpy as np
from RIMEz import rime_funcs

def test_make_sigma_tensor():
    sigma = rime_funcs.make_sigma_tensor()

    assert sigma.shape == (2,2,4)
    assert sigma.dtype == np.complex128

def test_make_bool_sigma_tensor():
    bsigma = rime_funcs.make_bool_sigma_tensor()

    assert bsigma.shape == (2,2,4)
    assert bsigma.dtype == np.bool

def test_fast_approx_radec2altaz():

    ra = np.linspace(0., 2*np.pi, 5, endpoint=False)
    dec = -np.radians(30.)*np.ones_like(ra)

    R = np.eye(3)

    s, alt, az = rime_funcs.fast_approx_radec2altaz(ra, dec, R)

    assert s.shape == (ra.size, 3)
    assert alt.shape == (ra.size,)
    assert az.shape == (ra.size,)
    assert all(az >= 0.), all(az < 2*np.pi)
    assert np.allclose(ra, az)

def test_RIME_sum():
    N = 2
    Jn = np.array([
        [1.+1j*2., 2.+1j*3.],
        [3.+1j*4., 4.+1j*5.]
    ], dtype=np.complex128)

    J1 = np.array([Jn, 2*Jn])
    J2_conj = np.conj(J1)

    F1 = np.ones(N)
    F2 = np.ones(N)
    S = np.zeros((N,4), dtype=np.float64)
    S[:,0] = 1.

    sigma = rime_funcs.make_sigma_tensor()
    bsigma = rime_funcs.make_bool_sigma_tensor()

    V = rime_funcs.RIME_sum(J1, J2_conj, F1, F2, S, sigma, bsigma)

    V2 = 5 * np.matmul(Jn, np.conj(Jn.T))

    assert np.all(V == V2)
    assert np.allclose(V, V2)

def test_visibility_calculations(visibility_calculation_fixed_test_parameters, visibility_calculation_fixed_test_output):
    (array_latitude, array_longitude, array_height, jd_axis, jd0,
    rotations_axis, nu_hz, r_axis, ant_pairs, beam_func, ant_ind2beam_func,
    S, RA, dec,
    Slm, R_0, L,
    delta_era_axis, integration_time) = visibility_calculation_fixed_test_parameters

    V_1src = rime_funcs.parallel_point_source_visibilities(rotations_axis, nu_hz, r_axis,
                                                          ant_pairs, beam_func,
                                                          ant_ind2beam_func,
                                                          S, RA, dec)

    Vm_1src = rime_funcs.parallel_mmode_unpol_visibilities(
        nu_hz, r_axis,
        ant_pairs, beam_func, ant_ind2beam_func,
        Slm, R_0, L, L)

    Vhrm_1src = rime_funcs.parallel_visibility_dft_from_mmodes(delta_era_axis, Vm_1src, integration_time)

    V_1src_rec, Vm_1src_rec, Vhrm_1src_rec = visibility_calculation_fixed_test_output

    print(np.max(np.abs(V_1src - V_1src_rec)))
    print(np.max(np.abs(Vm_1src - Vm_1src_rec)))
    print(np.max(np.abs(Vhrm_1src - Vhrm_1src_rec)))

    assert np.allclose(V_1src, V_1src_rec, atol=5e-14)
    assert np.allclose(Vm_1src, Vm_1src_rec, atol=5e-14)
    assert np.allclose(Vhrm_1src, Vhrm_1src_rec, atol=5e-14)
