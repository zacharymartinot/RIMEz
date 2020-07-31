import numpy as np
from RIMEz import utils
import astropy

def test_coords_to_location():
    hlat = np.radians(-30.0)
    hlon = np.radians(21.0)
    hheight = 1000.0
    loc = utils.coords_to_location(hlat, hlon, hheight)

    assert(np.allclose(loc.lat.rad, hlat))
    assert(np.allclose(loc.lon.rad, hlon))

    assert(isinstance(loc, astropy.coordinates.earth.EarthLocation))

def test_kernel_cutoff_estimate():
    c_mps = 299792458.0

    max_len_m = 1000.
    max_freq_hz = 250e6
    width_estimate = 151

    ell_cutoff1 = utils.kernel_cutoff_estimate(max_len_m, max_freq_hz, width_estimate=width_estimate)

    ell_cutoff2 = 2 * np.pi * max_len_m * max_freq_hz / c_mps + width_estimate
    ell_cutoff2 = int(np.ceil(ell_cutoff2)) + 1

    assert(ell_cutoff1 == ell_cutoff2)

def test_b_arc():

    b = np.array([0.,0.,0.])
    assert(np.isnan(utils.b_arc(b)))

    b = np.array([0., 1., 0.])
    assert(np.allclose(utils.b_arc(b), np.pi/2.))

    b = np.array([1.,1.,0.])
    b_grp = np.around(np.linalg.norm(b), 3)
    arc = np.around(b_grp * np.arctan(b[1] / b[0]), 3)

    assert(np.allclose(arc, utils.b_arc(b)))

def test_B():
    b = np.array([1.,1.,0.])
    B = np.around(np.linalg.norm(b), 3)
    assert(np.allclose(B, utils.B(b)))

def test_simple_array_generation():
    """
    tests `get_minimal_antenna_set` and `generate_hex_positions`
    """

    # generate_hex_positions against output from when it was known to be working
    answer = np.array([
       [-10.      ,   0.      ,   0.      ],
       [ -5.      ,  -8.660254,   0.      ],
       [ -5.      ,   8.660254,   0.      ],
       [  0.      ,   0.      ,   0.      ]
       ])

    r_axis = utils.generate_hex_positions(lattice_scale=10.0, u_lim=1, v_lim=2, w_lim=2)

    assert(np.allclose(r_axis, answer))

    # get_minimial_antenna_set
    ant_pairs, u2a, a2u = utils.get_minimal_antenna_set(r_axis)

    test1 = []
    for B in u2a.keys():
        for arc in u2a[B].keys():
            for apair in u2a[B][arc]:
                test1.append( a2u[apair] == (B, arc))

    assert(all(test1))

def test_JD2era():
    test_val = utils.JD2era(2458800.0)
    assert(test_val < 2*np.pi)
    assert(test_val > 0.0)

def test_JD2era_tot():
    test_val = utils.JD2era_tot(2458800.0)
    assert(test_val > 0)
>>>>>>> 3c2d717 (combination of several things, in particular:harmonic transform with gridding, tests)
