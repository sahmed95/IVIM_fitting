#!/usr/bin/python
""" Classes and functions for fitting ivim model """
from __future__ import division, print_function, absolute_import


# from .base import IvimModel
# from dipy.ivim import (IvimFit)
import fitting_diffusion as ic
import numpy as np
import matplotlib.pylab as pl


def ivim_prediction(bvals, S0, f, D_star, D):
    """
    Predict a signal given the parameters of the IVIM model.

    Parameters
    ----------
    bvals : ndarray
    S0 :
    f :
    D_star
    D

    Notes
    -----
    The predicted signal is given by:
    $ S0*(f*np.exp(-b*D_star)+(1-f)*np.exp(-b*D)) $, where

    References
    ----------
    .. [1]

    """

    return [S0 * (f * np.exp(-b * D_star) + (1 - f) * np.exp(-b * D)) for b in bvals]

# inheti ivim base class


class IvimModel(object):
    """ Ivim model
    """

    def __init__(self, image, bvals, *args, **kwargs):
        """ Ivim model
        Parameters
        ----------
        img : img object obtained using nibabel as
        import nibabel as nb

        img = nb.load(img_file_name)

        fit_method : str or callable
            str can be one of the following:
            'one_stage'
            'two_stage'



        References
        ----------
        .. [1]

        """
        # IvimModel.__init__(self, img, bvals)

        self.image = image
        self.img = image.get_data().astype(np.dtype('d'))
        self.hdr = image.get_header()
        self.bvals = bvals
        self.args = args
        self.kwargs = kwargs

    def predict(self, S0, f, D_star, D):
        """
        Predict a signal for this Ivim class instance with given parameters.

        Parameters
        ----------
        ivim_params : ndarray


        S0 : float or ndarray
            The non diffusion-weighted signal in every voxel, or across all
            voxels. Default: 1
        """
        return ivim_prediction(self.bvals, S0, f, D_star, D)

    def fit(self, fit_method='one_stage_fit', min_method=None):
        """ Fit method of the Ivim model
        Parameters
        ----------
        data : array
            The measured signal from one bvalue.

        """

        bvals = self.bvals
        img = self.img
        hdr = self.hdr

        shape = img.shape
        shape3d = shape[0:-1]
        mask = np.ones(shape3d)

        bnds = ((0.5, 1.5), (0, 1), (0, 0.5), (0, 0.5))  # (S0, f, D, D*)
        bnds_S0_D = (bnds[0], bnds[2])
        bnds_f_Dstar = (bnds[1], bnds[3])
        bnds_Dstar = (bnds[3],)
        # initial guesses are from the typical values in "Quantitative Measurement
        # of Brin Perfusion with Intravoxel Incoherent Motion MR Imaging"
        S0_guess = 1.0  # img.ravel().max()
        f_guess = 6e-2  # 6% or 0.06
        D_guess = 0.9e-3  # note that D<D* because the slope is lower in the diffusion section
        Dstar_guess = 7.0e-3  # note that D*>D because the slope is higher in the perfusion section
        # noise_std /= img.ravel().max()

        bvals_le_200 = bvals[bvals <= 200]
        bvals_gt_200 = bvals[bvals > 200]
        b_le_200_cutoff = len(bvals_le_200)

        fit = np.zeros(list(shape3d) + [4, ])
        nit = np.zeros(shape3d)
        success = np.zeros(shape3d)
        fun = np.zeros(shape3d)
        residual = np.zeros(shape)
        curve = np.zeros(shape)
        S0_prime = 0
        D = 0
        S0 = 0
        Dstar_prime = 0
        f = 0

        # normalize img but store the first value
        img0 = np.zeros(shape3d)
        np.copyto(img0, img[..., 0])
        for item in np.ndindex(shape3d):
            if img[item][0] > 0:  # so we dont' divide by 0 or anything funky
                img[item] /= img[item][0]

        if fit_method == 'two_stage_fit':
            dc2 = ic.Diffusion_Curve(S0_reg_in=0., f_reg_in=0.,
                                     D_reg_in=0., Dstar_reg_in=0.)
            fitfun_S0_D = ic.IVIM_Curve(
                fun_in=dc2.IVIM_fun_lsqr_S0_D_sumsq, method_in=min_method, bounds_in=bnds_S0_D)
            print('Start two stage fit\n')
            print('Fitting S0prime and D\n')
            fit[..., 0:3:2], nit, success, fun = ic.IVIM_array_fit(
                fitfun_S0_D, [S0_guess, D_guess], img[..., b_le_200_cutoff:], mask, [0, 0], bvals_gt_200)
            # save the intercept for the first values, let's call it S0'
            S0_prime = np.copy(fit[..., 0])
            D = np.copy(fit[..., 2])  # save the slope for plotting later
            print('Fitting S0 and D*prime\n')
            fit[..., 0:4:3], nit, success, fun = ic.IVIM_array_fit(fitfun_S0_D, [
                                                                   S0_guess, Dstar_guess], img[..., :b_le_200_cutoff], mask, [0, 0], bvals_le_200)
            S0 = np.copy(fit[..., 0])  # save the intercept for plotting later
            Dstar_prime = np.copy(fit[..., 3])  # save the linear D* only fit
            print('Estimating f\n')
            # arbitrary range, but we want to cap it to [0,1]
            fit[..., 1] = 1 - S0_prime / fit[..., 0]
            fit[fit[..., 1] < 0, 1] = 0  # make sure we don't have f<0
            fit[fit[..., 1] > 1, 1] = 1  # make sure we don't have f>1
            f = np.copy(fit[..., 1])  # save the fraction for plotting later
            print('Fitting D*')
            fitfun_Dstar = ic.IVIM_Curve(
                fun_in=dc2.IVIM_fun_lsqr_Dstar_sumsq, method_in=min_method, bounds_in=bnds_Dstar)
            fit[..., 3:], nit, success, fun = ic.IVIM_array_fit(
                fitfun_Dstar, [Dstar_guess], img[..., :], mask, fit[..., 0:3], bvals)

            print('End two stage fit\n')
            # Dstar = np.copy(fit[..., 3])  # save D* for plotting later
            # print('Fitting S0, f and D*')
            # fitfun_f_Dstar = ic.IVIM_Curve(fun_in=dc2.IVIM_fun_lsqr_f_Dstar_sumsq, method_in=min_method, bounds_in=bnds_f_Dstar)
            # fit[..., 1:4:2], nit, success, fun =
            # ic.IVIM_array_fit(fitfun_f_Dstar, [f_guess, Dstar_guess],
            # img[..., :], mask, fit[..., 0:3:2], bvals)

        if fit_method == 'one_stage_fit':
            dc1 = ic.Diffusion_Curve(
                S0_reg_in=0.01, f_reg_in=0.01, D_reg_in=0.01, Dstar_reg_in=0.01)
            fitfun_S0_f_D_Dstar = ic.IVIM_Curve(
                fun_in=dc1.IVIM_fun_lsqr_sumsq, method_in=min_method, bounds_in=bnds)
            print('Start one stage fit\n')
            print('Fitting S0, f, D, and D* \n')
            if fit_method == 'two_stage_fit':  # if we fit above, then use the fit as an inital guess
                fit, nit, success, fun = ic.IVIM_array_fit(
                    fitfun_S0_f_D_Dstar, fit[..., :], img[..., :], mask, fit[..., :], bvals)
            else:
                fit, nit, success, fun = ic.IVIM_array_fit(fitfun_S0_f_D_Dstar, [
                                                           S0_guess, f_guess, D_guess, Dstar_guess], img[..., :], mask, [0, 0, 0, 0], bvals)
            print('End one stage fit\n')
        # calculate the residual and the fitted curve
        return fitted_model(bvals, img, hdr, shape, mask,
                            bnds,
                            bnds_S0_D,
                            bnds_f_Dstar,
                            bnds_Dstar,
                            S0_guess,
                            f_guess,
                            D_guess,
                            Dstar_guess,
                            bvals_le_200,
                            bvals_gt_200,
                            b_le_200_cutoff,
                            fit, nit, success, fun, residual,
                            curve, img0, S0_prime, D, S0, Dstar_prime, f)


class fitted_model(object):

    def __init__(self, bvals, img, hdr, shape, mask,
                 bnds,
                 bnds_S0_D,
                 bnds_f_Dstar,
                 bnds_Dstar,
                 S0_guess,
                 f_guess,
                 D_guess,
                 Dstar_guess,
                 bvals_le_200,
                 bvals_gt_200,
                 b_le_200_cutoff,
                 fit, nit, success, fun, residual,
                 curve, img0, S0_prime, D, S0, Dstar_prime, f):
        self.bvals = bvals
        self.img = img
        self.hdr = hdr
        self.shape = shape
        self.shape3d = shape[0:-1]
        self.mask = mask
        self.bnds = bnds
        self.bnds_S0_D = bnds_S0_D
        self.bnds_f_Dstar = bnds_f_Dstar
        self.bnds_Dstar = bnds_Dstar
        self.S0_guess = S0_guess
        self.f_guess = f_guess
        self.D_guess = D_guess
        self.Dstar_guess = Dstar_guess
        self.bvals_le_200 = bvals_le_200
        self.bvals_gt_200 = bvals_gt_200
        self.b_le_200_cutoff = b_le_200_cutoff
        self.fit = fit
        self.nit = nit
        self.success = success
        self.fun = fun
        self.residual = residual
        self.curve = curve
        self.img0 = img0
        self.S0_prime = S0_prime
        self.D = D
        self.S0 = S0
        self.Dstar_prime = Dstar_prime
        self.f = f

    def plot(self, num_plots=1):
        img = self.img
        shape3d = self.shape3d
        curve = self.curve
        bvals = self.bvals
        fit = self.fit
        img0 = self.img0
        residual = self.residual
        S0 = self.S0
        S0_prime = self.S0_prime
        D = self.D
        bvals_le_200 = self.bvals_le_200
        Dstar_prime = self.Dstar_prime
        f = self.f

        dc = ic.Diffusion_Curve()

        plot_count = 0
        for item in np.ndindex(shape3d):
            if plot_count < num_plots:
                curve[item] = dc.IVIM_fun(bvals, fit[item][0], fit[item][1], fit[
                                          item][2], fit[item][3]) * img0[item]
                residual[item] = img[item] - curve[item]

                D_line = dc.IVIM_fun(bvals, S0_prime[item], 0., D[
                                     item], 0.) * img0[item]
                Dstar_line = dc.IVIM_fun(
                    bvals_le_200, S0[item], 1., 0., Dstar_prime[item]) * img0[item]
                pl.plot(bvals, D_line, 'b--', label='D curve')
                pl.plot(bvals_le_200, Dstar_line, 'r--', label='D* curve')
                pl.plot(bvals, img[item] * img0[item], '.', label='Image values')
                pl.plot(bvals, curve[item], label='Fitted curve')
                pl.yscale('symlog')  # to protect against 0 or negative values
                pl.xlabel(r'b-value $(s/mm^2)$')
                pl.ylabel(r'Signal intensity $(a.u.)$')
                pl.legend(loc='best')
                #pl.gca().text(0.25, 0.75, 'S0=%f f=%f D=%f D*=%f' %(fit[item][0], fit[item][1], fit[item][2], fit[item][3]))
                #ax = pl.axes()
                text_fit = 'S0={:.2e} f={:.2e}\nD={:.2e} D*={:.2e}'.format(
                    fit[item][0], fit[item][1], fit[item][2], fit[item][3])
                #'S0=%f f=%f D=%f D*=%f' %(fit[item][0], fit[item][1], fit[item][2], fit[item][3])
                pl.gca().text(0.25, 0.85, text_fit, horizontalalignment='center',
                              verticalalignment='center', transform=pl.gca().transAxes)

                # pl.cla()
                pl.show()
                plot_count += 1
            else:
                return 1
