#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 14:02:53 2024

@author: z0272571a
"""


import pyCATHY
from pyCATHY import cathy_tools
from pyCATHY.DA.cathy_DA import DA, dictObs_2pd
from pyCATHY.DA.perturbate import perturbate_parm
from pyCATHY.DA import perturbate
from pyCATHY.DA.observations import read_observations, prepare_observations, make_data_cov
from pyCATHY.DA import performance
from pyCATHY.ERT import petro_Archie as Archie
import pyCATHY.meshtools as cathy_meshtools
import pyvista as pv
import pyCATHY.plotters.cathy_plots as cplt 
import os 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# set some default plotting parameters for nicer looking plots
mpl.rcParams.update({"axes.grid":True, 
                     "grid.color":"gray", 
                     "grid.linestyle":'--',
                     'figure.figsize':(6,6)}
                   )
import pandas as pd
import utils
import pygimli as pg 
from pygimli.physics import ert


ensi=10

prjName = 'meshLi_withDA'
# prjName = 'meshCi_withDA'
prjName_without = 'meshLi_withoutDA'
#prjName = 'meshCi_withoutDA'

name_scenario = 'ic_test'
# name_scenario = 'ic'
# name_scenario = 'ic_ZROOT_NOupdate'
# name_scenario = 'ic_ZROOT_upd_ZROOT'

simu_solution = cathy_tools.CATHY(
                                    dirName='./solution_ERT',
                                    prj_name= 'ERT_dataset',
                                    notebook=True,
                                  )

simuWithDA = DA(
                        dirName='./DA_ERT',
                        prj_name= prjName + '_' + name_scenario, 
                        notebook=True,
                    )
simuWithoutDA = DA(
                        dirName='./DA_ERT',
                        prj_name= prjName_without + '_' + name_scenario, 
                        notebook=True,
                    )

results_withDA = simuWithDA.load_pickle_backup()
# results_withoutDA = simuWithoutDA.load_pickle_backup()


results_withDA['df_DA'].time


#%%
# from pyCATHY.plotters import cathy_plots as cplt

# cplt.show_vtk_TL(
#                 unit="pressure",
#                 notebook=False,
#                 path= os.path.join(simu_solution.workdir,
#                                    simu_solution.project_name,
#                                    "vtk",
#                                    ),
#                 show=False,
#                 x_units='days',
#                 clim = [0.55,0.70],
#                 savefig=True,
#             )
# #%%

# cplt.show_vtk_TL(
#                 unit="saturation",
#                 notebook=False,
#                 path= os.path.join(simu_solution.workdir,
#                                    simu_solution.project_name,
#                                    "vtk",
#                                    ),
#                 show=False,
#                 x_units='days',
#                 clim = [0.4,0.5],
#                 savefig=True,
#             )

#%%
# ensi = 2

# cplt.show_vtk_TL(
#                 unit="saturation",
#                 notebook=False,
#                 path= os.path.join(simuWithDA.workdir, simuWithDA.project_name,
#                                             'DA_Ensemble/cathy_' + str(ensi) + '/vtk'
#                                             ),
#                 show=False,
#                 x_units='days',
#                 clim = [0.4,0.5],
#                 savefig=True,
#             )


#%%
# ensi = 1
# pl = pv.Plotter(shape=(2, 2))
# axis_pl = [(0,0),(0,1),(1,0),(1,1)]
# for i, j in enumerate([0,2,5,9]):
#     mesh = pv.read(os.path.join(simuWithDA.workdir, simuWithDA.project_name,
#                                 'DA_Ensemble/cathy_' + str(ensi) + '/vtk',
#                                 'ER_converted' + str(j) + '_nearIntrp2_pg_msh.vtk',
#                                 )
    
#             )
#     pl.subplot(axis_pl[i][0],axis_pl[i][1])
#     pl.add_mesh(mesh,
#                 scalars='ER_converted' + str(j) + '_nearIntrp2_pg_msh',
#                 lighting=False,
#                 cmap="jet_r"
#                 )
    
#     pl.view_xz()
# pl.show()
# pl.screenshot(os.path.join(simuWithDA.workdir,
#                            simuWithDA.project_name,
#                            'ER_Ensemble' + str(ensi) + '.png',
#                             )
#               )
#%%
pl = pv.Plotter(shape=(2, 2))
axis_pl = [(0,0),(0,1),(1,0),(1,1)]
# for i, j in enumerate([0,2,5,9]):
for i, j in enumerate([0,1,2,3]):
    mesh = pv.read(os.path.join(simuWithDA.workdir, simuWithDA.project_name,
                                'DA_Ensemble/cathy_' + str(ensi) + '/vtk',
                                'ER_converted' + str(j) + '_nearIntrp2_pg_msh.vtk',
                                )
    
            )
    pl.subplot(axis_pl[i][0],axis_pl[i][1])
    pl.add_mesh(mesh,
                scalars='ER_converted' + str(j) + '_nearIntrp2_pg_msh',
                lighting=False,
                cmap="jet"
                )
    
    pl.view_xz()
pl.add_title(f'Predicted ens {ensi}')
pl.show()
pl.screenshot(os.path.join(simuWithDA.workdir,
                           simuWithDA.project_name,
                           'ER_Ensemble' + str(ensi) + '.png',
                            )
              )


#%%
# ensi=15
pl = pv.Plotter(shape=(2, 3))
axis_pl = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]
for i, j in enumerate([0,1,2,3,5,9]):
    mesh = pv.read(os.path.join(simuWithoutDA.workdir, simuWithoutDA.project_name,
                                'DA_Ensemble/cathy_' + str(ensi) + '/vtk',
                                'ER_converted' + str(j) + '_nearIntrp2_pg_msh.vtk',
                                )
    
            )
    pl.subplot(axis_pl[i][0],axis_pl[i][1])
    pl.add_mesh(mesh,
                scalars='ER_converted' + str(j) + '_nearIntrp2_pg_msh',
                lighting=False,
                cmap="jet"
                )
    
    pl.view_xz()
pl.add_title(f'Predicted ens {ensi}')
pl.show()
pl.screenshot(os.path.join(simuWithoutDA.workdir,
                           simuWithoutDA.project_name,
                           'ER_Ensemble' + str(ensi) + '.png',
                            )
              )

#%%

pl = pv.Plotter(shape=(2, 3))
axis_pl = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]
for i, j in enumerate([0,1,2,3,5,9]):
    mesh = pv.read(os.path.join(simu_solution.workdir, 'ERTsolution',
                                'meshLi' + str(j) + '.vtk',
                                )
    
            )
    pl.subplot(axis_pl[i][0],axis_pl[i][1])
    pl.add_mesh(mesh,
                scalars='saturation_nearIntrp2_pg_msh',
                lighting=False,
                cmap="jet_r"
                )
    
    pl.view_xz()
pl.add_title('Solution')
pl.show()
pl.screenshot(os.path.join(simu_solution.workdir,
                            'Saturation_solution.png',
                            )
              )


pl = pv.Plotter(shape=(2, 3))
axis_pl = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]
for i, j in enumerate([0,1,2,3,5,9]):
    mesh = pv.read(os.path.join(simu_solution.workdir, 'ERTsolution',
                                'meshLi' + str(j) + '.vtk',
                                )
    
            )
    pl.subplot(axis_pl[i][0],axis_pl[i][1])
    pl.add_mesh(mesh,
                scalars='ER',
                lighting=False,
                cmap="jet"
                )
    pl.view_xz()
pl.add_title('Solution')
pl.show()
pl.screenshot(os.path.join(simu_solution.workdir,
                            'ER_solution.png',
                            )
              )


#%%
DATA = []
dict_obs = {}
fig, ax = plt.subplots()

# Generate a colormap object
cmap = plt.get_cmap('Reds')
num_colors = len(results_withDA['df_DA'].time.unique()) - 1
colors = [cmap(i / num_colors) for i in range(num_colors)]

for i, tt in enumerate(results_withDA['df_DA'].time.unique()):
    # if i == 0:
    #     continue
    filename = os.path.join(simu_solution.workdir,
                            'ERTsolution', 
                            'meshLi' + str(i) + '.data')    
    rhoa = pg.load(filename)
    DATA.append(rhoa)
    # Plot with the corresponding color from the colormap
    ax.plot(rhoa['rhoa'].array(), color=colors[i-1])

plt.show()

    
# tl = ert.TimelapseERT(DATA)
# tl.invert(zWeight=0.3, lam=100)
# tl.showAllModels()
    