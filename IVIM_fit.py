__author__ = 'eric'

import nibabel as nb
import sys
import csv
import numpy as np
import scipy.optimize as op
import scipy.stats as st
import getopt as go
import fitting_diffusion as ic
# import matplotlib.pylab as pl
import matplotlib as ml
import os.path as os



def print_help():
    print('This program processes an IVIM scan and calculates the 4 parameters')
    print('These parameters are S0, f, Dstar, D')
    print('')
    print('Example:')
    print('%s -i <input file> -b <b-balue file> -f <output fit file name> ' %(str(sys.argv[0])))
    print('Input arguments:')
    print('-i <input file name of the 4D stacked image file>')
    print('-b <b-value file name of the csv file>')
    print('-m <mask image file name> (optional)')
    print('-1 indicates a one stage fitting (either -1 or -2 must be selected)')
    print('-2 indicates a two stage fitting (either -1 or -2 must be selected)')
    print('-e <minimization method, can be levenberg-marquardt or anything minimize accepts')
    print('-t <text file to fit (rather than an image)>')
    print('-p <folder to save plot figures into>')
    print('-a <name file for the regions (for use with the -t option)>')
    print('-h prints this help dialog')
    print('Output arguments:')
    print('-f <output fit file name>')
    print('-n <output niterations file name> (optional)')
    print('-c <output success flag file name> (optional)')
    print('-u <output fitness file name> (optional)')
    print('-r <output residual file name> (optional)')
    print('-v <output curve file name> (optional)')



def save_nii_or_txt(name,data,hdr):
    if ".nii" in name:
        nb.save(nb.Nifti1Image(data, hdr.get_affine()), name)  # hdr.get_header()
    else:
        np.savetxt(name, data, delimiter=",")

# use getopt to parse the input arguments
img_file_name = ''
bvalue_file_name = ''
mask_file_name = ''
output_fit_name = ''
output_niterations_name = ''
output_success_name = ''
output_fitness_name = ''
output_residual_name = ''
output_curve_name = ''
one_stage_fit = False
two_stage_fit = False
min_method = None  # L-BFGS-B
text_file_name = ''
plot_folder_name = ''
curve_name_file = ''
# use_jacobian = False
try:
    opts, args = go.getopt(sys.argv[1:], "hi:b:f:n:c:u:m:r:v:l12e:t:p:a:")
except go.GetoptError:
    print('Input option error')
    print_help()
    sys.exit(2)
for opt, arg in opts:
    print(opt, arg)
    if opt == '-h':
        print_help()
        sys.exit()
    elif opt == '-i':
        img_file_name = arg
    elif opt == '-b':
        bvalue_file_name = arg
    elif opt == '-f':
        output_fit_name = arg
    elif opt == '-n':
        output_niterations_name = arg
    elif opt == '-c':
        output_success_name = arg
    elif opt == '-u':
        output_fitness_name = arg
    elif opt == '-m':
        mask_file_name = arg
    elif opt == '-r':
        output_residual_name = arg
    elif opt == '-v':
        output_curve_name = arg
    elif opt == '-1':
        one_stage_fit = True
    elif opt == '-2':
        two_stage_fit = True
    elif opt == '-e':
        min_method = arg
    elif opt == '-t':
        text_file_name = arg
    elif opt == '-p':
        plot_folder_name = arg
    elif opt == '-a':
        curve_name_file = arg
#    elif opt == '-j':
#        use_jacobian = True
    else:
        assert False, "Unhandled Option %s" % opt


if not (img_file_name or text_file_name) or not bvalue_file_name or not output_fit_name or not (one_stage_fit or two_stage_fit):  # we need at least 2 inputs and some reasonable outputs
    print('Missing one of the following:')
    print('Image file name: %s' % img_file_name)
    print('bvalue file name: %s' % bvalue_file_name)
    print('output fit file name: %s' % output_fit_name)
    print('number of fit stages -1 or -2')
    print_help()
    sys.exit(2)


print('minimization method = %s' % min_method)

if one_stage_fit:
    print('Using one stage fitting')
if two_stage_fit:
    print('Using two stage fitting')

#if use_jacobian:
#    print('Using the jacobian ')


# load the 4-D stacked image
hdr = None
if img_file_name:
    print('Reading the image')
    hdr = nb.load(img_file_name)
    shape = hdr.shape  # the image size
    img = hdr.get_data().astype(np.dtype('d'))

if text_file_name:
    img=np.genfromtxt(text_file_name, delimiter=',')
    shape=img.shape
shape3d = shape[0:-1]

# load the b-values
print('loading the b-values')
with open(bvalue_file_name, 'r') as bvalcsv:
    reader = csv.reader(bvalcsv)
    bvals_str = next(reader)[0].split(" ")
    bvals = np.array([float(i) for i in bvals_str])  # convert from strings to floats

if shape[-1] != len(bvals):
    print("The 4th dimension of the image and the number of bvalues do not matach!")
    sys.exit(2)
else:
    print('we have %d b-values, so on y va!' % len(bvals))
    print(bvals)

# if we have a mask file input, then load the image
if mask_file_name:
    mask_hdr = nb.load(mask_file_name)
    mask_shape = mask_hdr.shape
    if shape3d != mask_shape:
        print('The mask and image are not the same shape')
        print(shape3d,mask_shape)
        sys.exit(2)
    mask = mask_hdr.get_data()
else:
    mask = np.ones(shape3d)

if curve_name_file:
    with open(curve_name_file, 'r') as names:
        reader = csv.reader(names)
        curve_names_list = list(reader)
        curve_names=  [i[0] for i in curve_names_list]
    if shape3d[0] != len(curve_names):
        print("The number elements and the number of names do not match!")
        sys.exit(2)

# set up some bounds and initial guesses
bnds = ((0.5, 1.5), (0, 1), (0, 0.5), (0, 0.5))  # (S0, f, D, D*)
bnds_S0_D = (bnds[0], bnds[2])
bnds_f_Dstar = (bnds[1], bnds[3])
bnds_Dstar = (bnds[3],)
# initial guesses are from the typical values in "Quantitative Measurement of Brin Perfusion with Intravoxel Incoherent Motion MR Imaging"
S0_guess = 1.0  # img.ravel().max()
f_guess = 6e-2  # 6% or 0.06
D_guess = 0.9e-3  # note that D<D* because the slope is lower in the diffusion section
Dstar_guess = 7.0e-3  # note that D*>D because the slope is higher in the perfusion section
# noise_std /= img.ravel().max()

bvals_le_200 = bvals[bvals <= 200]
bvals_gt_200 = bvals[bvals > 200]
b_le_200_cutoff = len(bvals_le_200)

# # checking the gradient
# print('checking the full gradient')
# eps = np.sqrt(np.finfo('f').eps)
# # fprime_approx = op.approx_fprime([S0_guess, f_guess, D_guess, Dstar_guess], IVIM_fun_wrap, eps, bvals[10])
# # grdchk = op.check_grad(IVIM_fun_wrap, IVIM_grad_wrap, [S0_guess, f_guess, D_guess, Dstar_guess], bvals[10])
# for bv in bvals:
#     print(op.approx_fprime([S0_guess, f_guess, D_guess, Dstar_guess], IVIM_fun_wrap, eps, bv))
#     print(IVIM_grad_wrap([S0_guess, f_guess, D_guess, Dstar_guess], bv))
#     print(op.check_grad(IVIM_fun_wrap, IVIM_grad_wrap, [S0_guess, f_guess, D_guess, Dstar_guess], bv))
#
# print('checking the partial gradient')
# for bv in bvals:
#     #print(op.approx_fprime([S0_guess, f_guess, D_guess, Dstar_guess], IVIM_fun_wrap, eps, bv))
#     #print(IVIM_grad_wrap([S0_guess, f_guess, D_guess, Dstar_guess], bv))
#     #print(op.check_grad(IVIM_fun_wrap, IVIM_grad_wrap, [S0_guess, f_guess, D_guess, Dstar_guess], bv))
#     print('-')
#     print(op.approx_fprime([S0_guess,D_guess],IVIM_fun_lsqr_S0_D_sumsq, eps, bv, IVIM_fun_wrap([S0_guess, f_guess, D_guess, Dstar_guess], bv), [f_guess, Dstar_guess]))
#     print(IVIM_grad_lsqr_S0_D_sumsq([S0_guess, D_guess], bv, None, [f_guess, Dstar_guess]))
#     print(op.approx_fprime([f_guess,Dstar_guess],IVIM_fun_lsqr_f_Dstar_sumsq, eps, bv, IVIM_fun_wrap([S0_guess, f_guess, D_guess, Dstar_guess], bv), [S0_guess, D_guess]))
#     print(IVIM_grad_lsqr_f_Dstar_sumsq([f_guess, Dstar_guess], bv, None, [S0_guess, D_guess]))
#     #print(op.check_grad(IVIM_fun_wrap, IVIM_grad_wrap, [S0_guess, f_guess, D_guess, Dstar_guess], bv))


print('Beginning processing')

# allocate before because I can't figure out how to allocate during
fit = np.zeros(list(shape3d)+[4, ])  # convert tuple to a list and append 4 to the end
nit = np.zeros(shape3d)
success = np.zeros(shape3d)
fun = np.zeros(shape3d)
residual = np.zeros(shape)
curve = np.zeros(shape)

#normalize img but store the first value
img0 = np.zeros(shape3d)
np.copyto(img0, img[..., 0])
for item in np.ndindex(shape3d):
    if img[item][0] > 0:  # so we dont' divide by 0 or anything funky
        img[item] /= img[item][0]


# def mapprint(vec):
#     return vec
#
# def mapprintimg(pos,im):
#     return im[pos]
#
#
# print('parallel')
# pool=mp.Pool(8)
# test=pool.map(mapprint,shape3d,img)
# pool.close()
# print('end parallel')
#for item in np.ndindex(shape3d): #here's how to run a flat iterator!
#    print(item)
#    print(img[item])
#    print(img[item].shape)

#IVIM_Curve.IVIM_Curve.IVIM_fun_lsqr_S0_D_sumsq
if two_stage_fit:
    dc2 = ic.Diffusion_Curve(S0_reg_in=0., f_reg_in=0., D_reg_in=0., Dstar_reg_in=0.)
    fitfun_S0_D = ic.IVIM_Curve(fun_in=dc2.IVIM_fun_lsqr_S0_D_sumsq, method_in=min_method, bounds_in=bnds_S0_D)
    print('Fitting S0prime and D')
    fit[..., 0:3:2], nit, success, fun = ic.IVIM_array_fit(fitfun_S0_D, [S0_guess, D_guess], img[..., b_le_200_cutoff:], mask, [0, 0], bvals_gt_200)
    S0_prime = np.copy(fit[..., 0])  # save the intercept for the first values, let's call it S0'
    D = np.copy(fit[..., 2])  # save the slope for plotting later
    print('Fitting S0 and D*prime')
    fit[..., 0:4:3], nit, success, fun = ic.IVIM_array_fit(fitfun_S0_D, [S0_guess, Dstar_guess], img[..., :b_le_200_cutoff], mask, [0, 0], bvals_le_200)
    S0 = np.copy(fit[..., 0])  # save the intercept for plotting later
    Dstar_prime = np.copy(fit[..., 3])  # save the linear D* only fit
    print('Estimating f')
    fit[..., 1] = 1-S0_prime/fit[..., 0]  # arbitrary range, but we want to cap it to [0,1]
    fit[fit[..., 1] < 0, 1] = 0  # make sure we don't have f<0
    fit[fit[..., 1] > 1, 1] = 1  # make sure we don't have f>1
    f = np.copy(fit[..., 1])  # save the fraction for plotting later
    print('Fitting D*')
    fitfun_Dstar = ic.IVIM_Curve(fun_in=dc2.IVIM_fun_lsqr_Dstar_sumsq, method_in=min_method, bounds_in=bnds_Dstar)
    fit[..., 3:], nit, success, fun = ic.IVIM_array_fit(fitfun_Dstar, [Dstar_guess], img[..., :], mask, fit[..., 0:3], bvals)
    Dstar = np.copy(fit[..., 3])  # save D* for plotting later
    #print('Fitting S0, f and D*')
    # fitfun_f_Dstar = ic.IVIM_Curve(fun_in=dc2.IVIM_fun_lsqr_f_Dstar_sumsq, method_in=min_method, bounds_in=bnds_f_Dstar)
    # fit[..., 1:4:2], nit, success, fun = ic.IVIM_array_fit(fitfun_f_Dstar, [f_guess, Dstar_guess], img[..., :], mask, fit[..., 0:3:2], bvals)

if one_stage_fit:
    dc1 = ic.Diffusion_Curve(S0_reg_in=0.01, f_reg_in=0.01, D_reg_in=0.01, Dstar_reg_in=0.01)
    fitfun_S0_f_D_Dstar = ic.IVIM_Curve(fun_in=dc1.IVIM_fun_lsqr_sumsq, method_in=min_method, bounds_in=bnds)
    print('Fitting S0, f, D, and D*')
    if two_stage_fit:  # if we fit above, then use the fit as an inital guess
        fit, nit, success, fun = ic.IVIM_array_fit(fitfun_S0_f_D_Dstar, fit[..., :], img[..., :], mask, fit[..., :], bvals)
    else:
        fit, nit, success, fun = ic.IVIM_array_fit(fitfun_S0_f_D_Dstar, [S0_guess, f_guess, D_guess, Dstar_guess], img[..., :], mask, [0, 0, 0, 0], bvals)

# calculate the residual and the fitted curve
dc = ic.Diffusion_Curve()
if plot_folder_name:  # if we input the curve names then don't show the plots, just save them
    ml.use('agg')  # this doesn't seem to work here but I don't know if it's actually required anyway?
    import matplotlib.pylab as pl #import happens after the use('agg')
    pl.ioff()
else:
    import matplotlib.pylab as pl
#pl.figure()
for item in np.ndindex(shape3d):
    curve[item] = dc.IVIM_fun(bvals, fit[item][0], fit[item][1], fit[item][2], fit[item][3])*img0[item]
    residual[item] = img[item] - curve[item]
    if plot_folder_name or curve_name_file:
        D_line = dc.IVIM_fun(bvals, S0_prime[item], 0., D[item], 0.)*img0[item]
        Dstar_line = dc.IVIM_fun(bvals_le_200, S0[item], 1., 0., Dstar_prime[item])*img0[item]
        pl.plot(bvals, D_line, 'b--', label='D curve')
        pl.plot(bvals_le_200, Dstar_line, 'r--', label='D* curve')
        pl.plot(bvals, img[item]*img0[item], '.', label='Image values')
        pl.plot(bvals, curve[item], label='Fitted curve')
        pl.yscale('symlog') #to protect against 0 or negative values
        pl.xlabel(r'b-value $(s/mm^2)$')
        pl.ylabel(r'Signal intensity $(a.u.)$')
        pl.legend(loc='best')
        #pl.gca().text(0.25, 0.75, 'S0=%f f=%f D=%f D*=%f' %(fit[item][0], fit[item][1], fit[item][2], fit[item][3]))
        #ax = pl.axes()
        text_fit = 'S0={:.2e} f={:.2e}\nD={:.2e} D*={:.2e}'.format(fit[item][0], fit[item][1], fit[item][2], fit[item][3])
        #'S0=%f f=%f D=%f D*=%f' %(fit[item][0], fit[item][1], fit[item][2], fit[item][3])
        pl.gca().text(0.25, 0.85, text_fit, horizontalalignment='center',verticalalignment='center',transform=pl.gca().transAxes)
        if curve_name_file:
            pl.title(os.splitext(curve_names[item[0]])[0])
        if plot_folder_name and curve_name_file:  # I'm not sure what file type to save, so for now I'll save a few
            fig_save_name = os.join(plot_folder_name, os.splitext(curve_names[item[0]])[0])
            pl.savefig(fig_save_name+'.png')  # save as png
            pl.savefig(fig_save_name+'.svg')  # save as svg
            pl.savefig(fig_save_name+'.eps')  # save as eps
        pl.cla()



# output the images
# if output_fit_name:  # we always have this file because this check is done above
save_nii_or_txt(output_fit_name, fit, hdr)
#nb.save(nb.Nifti1Image(fit, hdr.get_affine()), output_fit_name)  # hdr.get_header()
if output_niterations_name:
    save_nii_or_txt(output_niterations_name, nit, hdr)
    #if ".nii" in output_niterations_name:
    #    nb.save(nb.Nifti1Image(nit, hdr.get_affine()), output_niterations_name)  # hdr.get_header()
    #else:
    #    np.savetxt(output_niterations_name,fit,delimiter=",")
if output_success_name:
    save_nii_or_txt(output_success_name, success, hdr)
    #if ".nii" in output_success_name:
    #    nb.save(nb.Nifti1Image(success, hdr.get_affine()), output_success_name)  # hdr.get_header()
    #else:
    #    np.savetxt(output_success_name,fit,delimiter=",")
if output_fitness_name:
    save_nii_or_txt(output_fitness_name, fun, hdr)
    #nb.save(nb.Nifti1Image(fun, hdr.get_affine()), output_fitness_name)  # hdr.get_header()
if output_residual_name:
    save_nii_or_txt(output_residual_name, residual, hdr)
    #nb.save(nb.Nifti1Image(residual, hdr.get_affine()), output_residual_name)
if output_curve_name:
    save_nii_or_txt(output_curve_name, curve, hdr)
    #nb.save(nb.Nifti1Image(curve, hdr.get_affine()), output_curve_name)


