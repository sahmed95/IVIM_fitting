""" Testing IVIM

"""
from __future__ import division, print_function, absolute_import

import numpy as np
from nose.tools import (assert_true, assert_equal,
                        assert_almost_equal, assert_raises)
import numpy.testing as npt
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_)
import nibabel as nib

import scipy.optimize as opt
from ivim import IvimModel

def test_ivim_fit():
    """
    Tests the ivim fitting functions using RMSE

    Uses bvals = bvals= [0., 10., 20., 30., 40., 60., 80., 100., 120., 140., 160., 180., 200.
        , 300., 400., 500., 600., 700., 800., 900., 1000.]

    S0= 1.0
    f= 0.15
    D_star= 0.028
    D= 0.00018

    """
    # gtab = .....

    npt.assert_raises(ValueError, IvimModel, gtab, fit_method='one_stage_fit',
                      min_signal=-1)

    model = IvimModel(gtab, fit_method='two_stage_fit')

    # tensor_est = model.fit(Y)
    # assert_equal(tensor_est.shape, Y.shape[:-1])
    # assert_array_almost_equal(tensor_est.evals[0], evals)
    # assert_array_almost_equal(tensor_est.quadratic_form[0], tensor,
    #                           err_msg="Calculation of tensor from Y does not "
    #                                   "compare to analytical solution")
    # assert_almost_equal(tensor_est.md[0], md)

    # # Test that we can fit a single voxel's worth of data (a 1d array)
    # y = Y[0]
    # tensor_est = model.fit(y)
    # assert_equal(tensor_est.shape, tuple())
    # assert_array_almost_equal(tensor_est.evals, evals)
    # assert_array_almost_equal(tensor_est.quadratic_form, tensor)
    # assert_almost_equal(tensor_est.md, md)
    # assert_array_almost_equal(tensor_est.lower_triangular(b0), D)

    # # Test using fit_method='LS'
    # model = TensorModel(gtab, fit_method='LS')
    # tensor_est = model.fit(y)
    # assert_equal(tensor_est.shape, tuple())
    # assert_array_almost_equal(tensor_est.evals, evals)
    # assert_array_almost_equal(tensor_est.quadratic_form, tensor)
    # assert_almost_equal(tensor_est.md, md)
    # assert_array_almost_equal(tensor_est.lower_triangular(b0), D)
    # assert_array_almost_equal(tensor_est.linearity, linearity(evals))
    # assert_array_almost_equal(tensor_est.planarity, planarity(evals))
    # assert_array_almost_equal(tensor_est.sphericity, sphericity(evals))
