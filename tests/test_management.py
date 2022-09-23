import pytest

import numpy as np
from RIMEz import management

def test_get_versions():
    """Test _get_versions function"""
    rimez_version = RIMEz.__version__
    ssht_numba_version = ssht_numba.__version__
    spin1_beam_model_version = spin1_beam_model.__version__
    repo_versions = management._get_versions()

    assert repo_versions["RIMEz"] == rimez_version
    assert repo_versions["ssht_numba"] == ssht_numba_version
    assert repo_versions["spin1_beam_model"] == spin1_beam_model_version

    return

@pytest.fixture
def harmnonics_calculation_parameters():
    (array_latitude, array_longitude, array_height, jd_axis, jd0,
    rotations_axis, nu_hz, r_axis, ant_pairs, beam_func, ant_ind2beam_func,
    S, RA, dec,
    Slm, R_0, L,
    delta_era_axis, integration_time) = visibility_calculation_fixed_test_parameters

    parameters = {
        "array_latitude" : array_latitude,
        "array_longitude" : array_longitude,
        "array_height" : array_height,
        "initial_time_sample_jd" : jd0,
        "integration_time" : integration_time,
        "frequency_samples_hz" : nu_hz,
        "antenna_positions_meters" : r_axis,
        "antenna_pairs_used" : ant_pairs,
        "antenna_beam_function_map" : ant_ind2beam_func,
        "integral_kernel_cutoff" : L,
    }

    return parameters, beam_func, Slm

class TestVisibilityCalculation:

    @pytest.fixture(autouse=True) # not sure why scope="class" doesn't work here
    def _setup(self, visibility_calculation_fixed_test_parameters, visibility_calculation_fixed_test_output):
        (array_latitude, array_longitude, array_height, jd_axis, jd0,
        rotations_axis, nu_hz, r_axis, ant_pairs, beam_func, ant_ind2beam_func,
        S, RA, dec,
        Slm, R_0, L,
        delta_era_axis, integration_time) = visibility_calculation_fixed_test_parameters

        parameters = {
            "array_latitude" : array_latitude,
            "array_longitude" : array_longitude,
            "array_height" : array_height,
            "initial_time_sample_jd" : jd0,
            "integration_time" : integration_time,
            "frequency_samples_hz" : nu_hz,
            "antenna_positions_meters" : r_axis,
            "antenna_pairs_used" : ant_pairs,
            "antenna_beam_function_map" : ant_ind2beam_func,
            "integral_kernel_cutoff" : L,
        }

        self.parameters = parameters
        self.VC = management.VisibilityCalculation(parameters, beam_func, Slm)
        self.jd_axis = jd_axis
        self.integration_time = integration_time
        self.V_1src_rec, self.Vm_1src_rec, self.Vhrm_1src_rec = visibility_calculation_fixed_test_output

    # def test_setup(self):
    #     calculation_method = 'harmonics'
    #     VCh = management.VisibilityCalculation(self.parameters)

    def test_compute_fourier_modes(self):
        VC = self.VC
        VC.compute_fourier_modes()

        assert np.allclose(VC.Vm, 0.5*self.Vm_1src_rec, atol=1e-14)

    def test_compute_time_series(self):
        VC = self.VC
        VC.Vm = self.Vm_1src_rec

        time_sample_jds = self.jd_axis
        integration_time = self.integration_time

        VC.compute_time_series(time_sample_jds=time_sample_jds, integration_time=integration_time)

<<<<<<< HEAD
        assert np.allclose(VC.V, self.Vhrm_1src_rec, atol=1e-8)
>>>>>>> 3c2d717 (combination of several things, in particular:harmonic transform with gridding, tests)
=======
        assert np.allclose(VC.V, self.Vhrm_1src_rec, atol=1e-14)
>>>>>>> 723fd30 (tweaks and fixes)
