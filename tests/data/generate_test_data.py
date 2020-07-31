import numpy as np
import matplotlib.pyplot as plt

from RIMEz import management, utils, beam_models, sky_models, rime_funcs
import ssht_numba as sshtn

import numba as nb
import h5py
import os
import argparse
import datetime

def visibility_calculation_fixed_test_parameters():
    N_freq = 11
    N_times = 11

    array_latitude = utils.HERA_LAT
    array_longitude = utils.HERA_LON
    array_height = utils.HERA_HEIGHT

    nu_hz = 1e6*np.linspace(100,200,N_freq,endpoint=True) # Hz

    r_axis = np.array([
        [0,0,0],
        [20.,0,0],
        [0,20.,0],
    ], dtype=np.float64) # meters

    # choose which antenna pairs we want to compute visiblities for
    ant_pairs = np.array([
        [0,0],
        [0,1],
        [0,2],
        [1,2],
    ], dtype=np.int64)

    @nb.njit
    def beam_func(i, nu, alt, az):
        J = np.zeros((alt.shape[0], 2, 2))
        J[:, 0, 0] = np.sin(alt)**3.
        J[:, 1, 1] = np.sin(alt)**3.

        return J

    ant_ind2beam_func = np.zeros(r_axis.shape[0], dtype=np.int64)

    era0 = array_longitude

    jd_init = 2458845.5
    era_tot_init = utils.JD2era_tot(jd_init)

    era_correction = (era_tot_init % (2*np.pi) - era0)

    era_tot_init2 = era_tot_init - era_correction

    jd0 = utils.era_tot2JD(era_tot_init2)

    delta_era = (2*np.pi/(N_times-1))
    delta_jd = utils.era_tot2JD(delta_era) - utils.era_tot2JD(0.)

    jd_axis = jd0 + delta_jd * np.arange(-(N_times-1)//2, (N_times-1)//2 + 1)
    delta_era_axis = delta_era * np.arange(-(N_times-1)//2, (N_times-1)//2 + 1)
    era_axis = utils.JD2era(jd_axis)

    array_location = utils.coords_to_location(array_latitude, array_longitude, array_height)

    rotations_axis = utils.get_rotations_realistic_from_JDs(jd_axis, array_location, reference_jd=jd0)

    ra0, dec0 = era0 + array_longitude, array_latitude

    # arrays that define the sky model
    RA = np.array([ra0])
    dec = np.array([dec0])
    S = np.zeros((nu_hz.size, RA.shape[0], 4), dtype=np.float64)
    S[:,:,0] = 1.

    L = int(2*np.pi*(2/3)*np.sqrt(2)*20 + 100) # the beam width is 6 so this is plenty

    I_1src = S[...,0]
    Ilm_1src = sky_models.point_sources_harmonics_with_gridding(I_1src, RA, dec, L)

    Slm = Ilm_1src.reshape(Ilm_1src.shape + (1,))

    R_0 = utils.get_rotations_realistic_from_JDs(jd0, array_location, reference_jd=jd0)

    integration_time = 0.

    parameters = (array_latitude, array_longitude, array_height, jd_axis, jd0,
        rotations_axis, nu_hz, r_axis, ant_pairs, beam_func, ant_ind2beam_func,
        S, RA, dec,
        Slm, R_0, L,
        delta_era_axis, integration_time)

    return parameters

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate and save a set of visibility caclulation results using'
        'the current code to use in automated testing'
    )
    parser.add_argument('--overwrite', dest='overwrite', action='store_true')
    parser.add_argument('--no-archive', dest='archive', action='store_false')
    parser.set_defaults(archive=True)
    parser.set_defaults(overwrite=False)

    args = parser.parse_args()

    current_date = str(datetime.datetime.now().date())

    test_data_file_name = 'visibility_calculation_test_output.h5'
    base_dir = os.path.dirname(os.path.realpath(__file__))
    test_data_file_path = os.path.join(base_dir, test_data_file_name)


    rimez_version = management._get_versions()["RIMEz"]

    # check if test data exists...
    if os.path.exists(test_data_file_path):

        # ... and make a back up if requested...
        if args.overwrite and args.archive:

            with h5py.File(test_data_file_path, 'r') as h5f:
                old_file_rimez_version = h5f['version'][()].decode()
                old_file_date = h5f['date_created'][()].decode()

            archive_file_name = 'test_output_' + old_file_date + '_' + old_file_rimez_version + '.h5'
            archive_file_path = os.path.join(base_dir, 'old_test_data', archive_file_name)

            os.rename(test_data_file_path, archive_file_path)

        # ... otherwise, delete old file
        elif args.overwrite and not args.archive:
            os.remove(test_data_file_name)

        else:
            raise UserWarning("Test data file already exists and overwrite is not set.")

    print('Getting parameter data...')
    (array_latitude, array_longitude, array_height,
    rotations_axis, nu_hz, r_axis, ant_pairs, beam_func, ant_ind2beam_func,
    S, RA, dec,
    Slm, R_0, L,
    delta_era_axis, integration_time) = visibility_calculation_fixed_test_parameters()

    print('Starting point source calculation...')
    V_1src = rime_funcs.parallel_point_source_visibilities(rotations_axis, nu_hz, r_axis,
                                                          ant_pairs, beam_func,
                                                          ant_ind2beam_func,
                                                          S, RA, dec)
    print('Done. Starting harmonic calculations...')
    Vm_1src = rime_funcs.parallel_mmode_unpol_visibilities(
        nu_hz, r_axis,
        ant_pairs, beam_func, ant_ind2beam_func,
        Slm, R_0, L, L)

    print('Done. Starting time series synthesis...')
    Vhrm_1src = rime_funcs.parallel_visibility_dft_from_mmodes(delta_era_axis, Vm_1src, integration_time)
    print('Done. Preparinng to write data...')

    # write visibility results to new test file
    with h5py.File(test_data_file_path, 'w') as h5f:
        h5f.create_dataset('V_1src', data=V_1src)
        h5f.create_dataset('Vm_1src', data=Vm_1src)
        h5f.create_dataset('Vhrm_1src', data=Vhrm_1src)
        h5f.create_dataset('version', data=np.string_(rimez_version))
        h5f.create_dataset('date_created', data=np.string_(current_date))

    print('New test data file created.')
