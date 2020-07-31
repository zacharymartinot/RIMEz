import pytest

import numpy as np
import matplotlib.pyplot as plt

from RIMEz import utils, sky_models

import numba as nb
import h5py
import os

from data.generate_test_data import visibility_calculation_fixed_test_parameters

visibility_calculation_fixed_test_parameters = pytest.fixture(
    visibility_calculation_fixed_test_parameters,
    scope='session',
)


@pytest.fixture(scope='session')
def visibility_calculation_fixed_test_output():
    file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'data/visibility_calculation_test_output.h5'
    )

    with h5py.File(file_path, 'r') as h5f:
        V_1src = h5f['V_1src'][()]
        Vm_1src = h5f['Vm_1src'][()]
        Vhrm_1src = h5f['Vhrm_1src'][()]

    return V_1src, Vm_1src, Vhrm_1src

if __name__ == "__main__":
    print('test')
    print(type(visibility_calculation_fixed_test_parameters))
    print(type(visibility_calculation_fixed_test_output))
    
