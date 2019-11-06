#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 16:34:17 2019

@author: Javier Alejandro Acevedo Barroso
"""


## LLAMANDO LIBRERIAS DE LENSTRONOMY
import numpy as np                                  # Libreria numerica, definimos numeros
import time                                         # Libreria de simulacion.
import corner
import matplotlib.pyplot as plt
import matplotlib
import os
import copy
import astropy.io.fits as pyfits
import astropy
import lenstronomy.Util.simulation_util as sim_util
import lenstronomy.Util.image_util as image_util
import lenstronomy.Plots.output_plots as lens_plot
import matplotlib.pyplot as plt

from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF

from mpl_toolkits.axes_grid1 import make_axes_locatable

from lenstronomy.Util import constants

import lenstronomy.Util.param_util as param_util

from lenstronomy.ImSim.image_model import ImageModel

from lenstronomy.PointSource.point_source import PointSource

from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions

from lenstronomy.LightModel.light_model import LightModel

from lenstronomy.Sampling.parameters import Param

from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF

from lenstronomy.Cosmo.lens_cosmo import LensCosmo

from astropy.cosmology import FlatLambdaCDM

from astropy.io import fits

from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
import multiprocessing







#####data from the good teams file
# Condiciones observacionales, estas librerias nos determinan condiciones para simulacion de imagenes
background_rms = .05  # background noise per pixel
exp_time = 1200  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
numPix = 99  # cutout pixel size
#delta pix parametro a utilizar por el PSF creemos que se relaciona con el brillo
deltaPix = 0.08  # pixel size in arcsec (area per pixel = deltaPix**2)
fwhm = 0.1  # full width half max of PSF
psf_type = 'PIXEL'
kernel_size = 99

# Condiciones cosmologicas, distancias de la lente y del medio
z_lens = 0.858
z_source = 2.175



path = 'rung0/code1/f160w-seed3/drizzled_image/'

# Abrimos imagenes las definimos como matrices
#hdul = fits.open('lens-image.fits')
hdullens = fits.open(path+'lens-image.fits')
hdulnoise = fits.open(path+'noise_map.fits')
hdulpsf = fits.open(path+'psf.fits')

#image_data = hdul[0].data 
lens_data = hdullens[0].data
noise_data = hdulnoise[0].data
psf_data = hdulpsf[0].data


image_data = lens_data




kwargs_data = sim_util.data_configure_simple(numPix, deltaPix, exp_time, np.median(noise_data))
kwargs_psf = {'psf_type': 'PIXEL', 'pixel_size': deltaPix, 'kernel_point_source': psf_data}
kwargs_data['image_data'] = image_data


# Con la ayuda de IRAF deterinamos la ubicacion en pixeles, sabiendo que cada pixel son 0.08" para drizzzle images
# el centro esta ubicado en pixeles x = 49.919 y = 50.002
x_A = 0.08480
x_B = -1.16496
x_C = 1.21336
x_D = -0.27544

y_A = -1.09936
y_B = -0.24800
y_C = 0.22269
y_D = 1.05784

# Definimos valor de Anillo de Einstein Inicial
theta_E = 1.155297871

# Definimos ubicacion de las imagenes en la lente.
x_list = [x_A, x_B, x_C, x_D]
y_list = [y_A, y_B, y_C, y_D]

# Transformando los datos en legibles Lenstronomy.
x_image = np.array(x_list)
y_image = np.array(y_list)


# Definimos el tipo de modelo de lente que necesitamos, lens_model_list
# para esta primera simulacion escogemos SPEMD Y SHEAR, esto debido a que ya conocemos los resultados.
lens_model_list = ['SPEMD','SHEAR_GAMMA_PSI']
# definimos tambien la clase
lens_model_class = LensModel(lens_model_list=lens_model_list)

# Definimos modelo de luz de la fuente SOURCE AGN, tenemos la opcion de puntual, sersic y sersic ellipse
source_model_list = ['SERSIC_ELLIPSE']

# Definimos modelo de luz de la lente, LENS, es una galaxia no activa. Sabemos que es Elliptica debido a las
# respuestas del RUNG-0
lens_light_model_list = ['SERSIC_ELLIPSE']

# Usamos la libreria point source Desconocemos su utilidad, podemos inicialmente obviarlo
# Debemos solucionar la ecacion de lente para encontrar las ubicaciones de las imagenes en el 
# plano de la lente, sin embargo las ubicaciones ya la determinamos
point_source_list = ['LENSED_POSITION']

kwargs_model = {'lens_model_list': lens_model_list,
                'source_light_model_list': source_model_list,
                'lens_light_model_list': lens_light_model_list,
                'point_source_model_list': point_source_list,
                'additional_images_list': [False],
                'fixed_magnification_list': [False],
                             }
kwargs_numerics = {'supersampling_factor': 1, 'supersampling_convolution': False}

num_source_model = len(source_model_list)

kwargs_constraints = {'joint_source_with_point_source': [[0, 0]],
                              'num_point_source_list': [4],
                              'solver_type': 'PROFILE',  # 'PROFILE', 'PROFILE_SHEAR', 'ELLIPSE', 'CENTER'
                              }

kwargs_likelihood = {'check_bounds': True,
                             'force_no_add_image': False,
                             'source_marg': False,
                             'position_uncertainty': 0.004,
                             'check_solver': True,
                             'solver_tolerance': 0.001
                             }

image_band = [kwargs_data, kwargs_psf, kwargs_numerics]
multi_band_list = [image_band]
kwargs_data_joint = {'multi_band_list': multi_band_list, 'multi_band_type': 'multi-linear'}





# initial guess of non-linear parameters, we chose different starting parameters than the truth #
kwargs_lens_init = [{'theta_E': 1.2, 'e1': 0, 'e2': 0, 'gamma': 2., 'center_x': 0., 'center_y': 0},
    {'gamma_ext': 0.01, 'psi_ext': 0.}]
kwargs_source_init = [{'R_sersic': 0.03, 'n_sersic': 1., 'e1': 0, 'e2': 0, 'center_x': 0, 'center_y': 0}]
kwargs_lens_light_init = [{'R_sersic': 0.1, 'n_sersic': 1, 'e1': 0, 'e2': 0, 'center_x': 0, 'center_y': 0}]
kwargs_ps_init = [{'ra_image': x_image+0.01, 'dec_image': y_image-0.01}]

# initial spread in parameter estimation #
kwargs_lens_sigma = [{'theta_E': 0.3, 'e1': 0.2, 'e2': 0.2, 'gamma': .2, 'center_x': 0.1, 'center_y': 0.1},
    {'gamma_ext': 0.1, 'psi_ext': np.pi}]
kwargs_source_sigma = [{'R_sersic': 0.1, 'n_sersic': .5, 'center_x': .1, 'center_y': 0.1, 'e1': 0.2, 'e2': 0.2}]
kwargs_lens_light_sigma = [{'R_sersic': 0.1, 'n_sersic': 0.2, 'e1': 0.1, 'e2': 0.1, 'center_x': .1, 'center_y': 0.1}]
kwargs_ps_sigma = [{'ra_image': [0.02] * 4, 'dec_image': [0.02] * 4}]

# hard bound lower limit in parameter space #
kwargs_lower_lens = [{'theta_E': 0, 'e1': -0.5, 'e2': -0.5, 'gamma': 1.5, 'center_x': -10., 'center_y': -10},
    {'gamma_ext': 0., 'psi_ext': -np.pi}]
kwargs_lower_source = [{'R_sersic': 0.001, 'n_sersic': .5, 'e1': -0.5, 'e2': -0.5, 'center_x': -10, 'center_y': -10}]
kwargs_lower_lens_light = [{'R_sersic': 0.001, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -10, 'center_y': -10}]
kwargs_lower_ps = [{'ra_image': -10 * np.ones_like(x_image), 'dec_image': -10 * np.ones_like(y_image)}]

# hard bound upper limit in parameter space #
kwargs_upper_lens = [{'theta_E': 10, 'e1': 0.5, 'e2': 0.5, 'gamma': 2.5, 'center_x': 10., 'center_y': 10},
    {'gamma_ext': 0.3, 'psi_ext': np.pi}]
kwargs_upper_source = [{'R_sersic': 10, 'n_sersic': 5., 'e1': 0.5, 'e2': 0.5, 'center_x': 10, 'center_y': 10}]
kwargs_upper_lens_light = [{'R_sersic': 10, 'n_sersic': 5., 'e1': 0.5, 'e2': 0.5, 'center_x': 10, 'center_y': 10}]
kwargs_upper_ps = [{'ra_image': 10 * np.ones_like(x_image), 'dec_image': 10 * np.ones_like(y_image)}]

lens_params = [kwargs_lens_init, kwargs_lens_sigma, [{}, {'ra_0': 0.02, 'dec_0': 0.001}], kwargs_lower_lens, kwargs_upper_lens]
source_params = [kwargs_source_init, kwargs_source_sigma, [{}], kwargs_lower_source, kwargs_upper_source]
lens_light_params = [kwargs_lens_light_init, kwargs_lens_light_sigma, [{}], kwargs_lower_lens_light, kwargs_upper_lens_light]
ps_params = [kwargs_ps_init, kwargs_ps_sigma, [{}], kwargs_lower_ps, kwargs_upper_ps]

kwargs_params = {'lens_model': lens_params,
                'source_model': source_params,
                'lens_light_model': lens_light_params,
                'point_source_model': ps_params}

from lenstronomy.Workflow.fitting_sequence import FittingSequence
fitting_seq = FittingSequence(kwargs_data_joint, kwargs_model, kwargs_constraints, kwargs_likelihood, kwargs_params)

fitting_kwargs_list = [['update_settings', {'lens_add_fixed': [[0, ['gamma'], [2.0]]],
                  'ps_add_fixed' : [[0,['ra_image', 'dec_image'], [x_image,y_image] ]] }],
      ['PSO', {'sigma_scale': 1., 'n_particles': 200, 'n_iterations': 200, 'threadCount' : 1}],
      ['update_settings', {'lens_remove_fixed': [[0, ['gamma']]]}],
      ['PSO', {'sigma_scale': 0.1, 'n_particles': 200, 'n_iterations': 200, 'threadCount' : 1}],
                       ['MCMC', {'n_burn': 50, 'n_run': 5000, 'walkerRatio': 2, 'sigma_scale': .01}]
        ]

chain_list = fitting_seq.fit_sequence(fitting_kwargs_list)
kwargs_result = fitting_seq.best_fit()













##### VAMOS A GRAFICAR LO QUE NOS DA 

from lenstronomy.Plots.output_plots import ModelPlot
import lenstronomy.Plots.output_plots as out_plot

modelPlot = ModelPlot(multi_band_list, kwargs_model, kwargs_result, arrow_size=0.02, cmap_string="gist_heat")

param_class = fitting_seq.param_class
print(param_class.num_param())
#print(chain_list)

for i in range(len(chain_list)):
    out_plot.plot_chain_list(chain_list, i)

f, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=False, sharey=False)

modelPlot.data_plot(ax=axes[0,0])
modelPlot.model_plot(ax=axes[0,1])
modelPlot.normalized_residual_plot(ax=axes[0,2], v_min=-6, v_max=6)
modelPlot.source_plot(ax=axes[1, 0], deltaPix_source=0.01, numPix=100)
modelPlot.convergence_plot(ax=axes[1, 1], v_max=1)
modelPlot.magnification_plot(ax=axes[1, 2])
#f.tight_layout()
#f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.05)
#plt.show()
f.savefig('lens_model.png', dpi = 600)

f, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=False, sharey=False)

modelPlot.decomposition_plot(ax=axes[0,0], text='Lens light', lens_light_add=True, unconvolved=True)
modelPlot.decomposition_plot(ax=axes[1,0], text='Lens light convolved', lens_light_add=True)
modelPlot.decomposition_plot(ax=axes[0,1], text='Source light', source_add=True, unconvolved=True)
modelPlot.decomposition_plot(ax=axes[1,1], text='Source light convolved', source_add=True)
modelPlot.decomposition_plot(ax=axes[0,2], text='All components', source_add=True, lens_light_add=True, unconvolved=True)
modelPlot.decomposition_plot(ax=axes[1,2], text='All components convolved', source_add=True, lens_light_add=True, point_source_add=True)
f.tight_layout()
f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.05)
f.savefig('lens_results.png', dpi = 600)
print(kwargs_result)







































