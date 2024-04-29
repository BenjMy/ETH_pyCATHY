#!/usr/bin/env python
# coding: utf-8

# ---
# title: Working with pyCATHY and DA
# subtitle: PART1 - DA and ERT assimilation
# license: CC-BY-4.0
# github: https://github.com/BenjMy/ETH_pyCATHY/
# subject: Tutorial
# authors:
#   - name: Benjamin Mary
#     email: benjamin.mary@ica.csic.es
#     corresponding: true
#     orcid: 0000-0001-7199-2885
#     affiliations:
#       - ICA-CSIC
# date: 2024/04/12
# ---

# ```{toc} Table of Contents
# :depth: 3
# ```

# In[1]:


import pyCATHY
from pyCATHY import cathy_tools
import pygimli as pg

import pygimli as pg
from pygimli.physics import ert
import pygimli.meshtools as mt


# pyCATHY holds for **python** CATHY. Essentially it wraps the CATHY core to update/modify on the input files and has some **potential applications to ease preprocessing step**. pyCATHY also includes modules for Data Assimilation

# In[2]:


from pyCATHY.DA.cathy_DA import DA, dictObs_2pd
from pyCATHY.DA.perturbate import perturbate_parm
from pyCATHY.DA import perturbate
from pyCATHY.DA.observations import read_observations, prepare_observations, make_data_cov
from pyCATHY.DA import performance
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
                     'figure.figsize':(10,10)}
                   )
import pandas as pd

from pyCATHY.importers import cathy_inputs as in_CT
import pyCATHY.meshtools as cathy_meshtools
import utils

# In[4]:


# Create a CATHY project
# -----------------------
simu_solution = cathy_tools.CATHY(
                                    dirName='./solution_ERT',
                                    prj_name= 'weill_dataset',
                                    notebook=True,
                                  )

simu_solution.create_mesh_vtk()

# In[7]:

import pyvista as pv
meshpath = os.path.join(simu_solution.workdir,
                        simu_solution.project_name,
                        'vtk',
                        simu_solution.project_name + '.vtk'
                       )
meshCATHY = pv.read(meshpath)

# pl = pv.Plotter()
# pl.add_mesh(meshCATHY)
# pl.add_bounding_box()
# pl.show_grid()
# pl.show()

#%%## Change atmospheric boundary conditions

netValue = -1e-7
rain = 4e-7
tatmbc = list(np.linspace(0,86400,10))

netValue_list = [netValue]*len(tatmbc)
netValue_list[0] = netValue + rain
netValue_list[1] = netValue + rain

simu_solution.update_atmbc(
                            HSPATM=1,
                            IETO=0,
                            time=tatmbc,
                            # VALUE=[None, None],
                            netValue=netValue_list,
                    )

simu_solution.update_ic(INDP=0,IPOND=0,
                        pressure_head_ini=-1,
                        )


simu_solution.update_parm(TIMPRTi=tatmbc)
simu_solution.update_soil(PMIN=-1e25)


# In[21]:
# ss

simu_solution.run_preprocessor()
# simu_solution.run_processor(IPRT1=2,
#                             TRAFLAG=0,
#                             VTKF=2,
#                             )

# In[22]:
# ### Interpolate CATHY mesh on PG mesh
# sss

dem_mat, str_hd_dem = in_CT.read_dem(
    os.path.join(simu_solution.workdir, 
                 simu_solution.project_name, 
                 "prepro/dem"),
    os.path.join(simu_solution.workdir, 
                 simu_solution.project_name, 
                 "prepro/dtm_13.val"),
)

# yshift = 8
# xshift = 8

idC = 8
idL = 8

minmax_dem_mat_Ci = [min(dem_mat[:,idC]), max(dem_mat[:,idC])]
minmax_dem_mat_Li = [min(dem_mat[idL,:]), max(dem_mat[idL,:])]

geomCi = mt.createPolygon([[0.0, -1], [0, minmax_dem_mat_Ci[0]], [10, minmax_dem_mat_Ci[1]], [10, -1]],
                          isClosed=True, marker=1, area=1e-2)
meshCi = mt.createMesh(geomCi, quality=34.3, area=3, smooth=[1, 10])

geomLi = mt.createPolygon([[0.0, -1], [0, minmax_dem_mat_Li[1]], [10, minmax_dem_mat_Li[0]], [10, -1]],
                          isClosed=True, marker=1, area=1e-2)
meshLi = mt.createMesh(geomLi, quality=34.3, area=3, smooth=[1, 10])


meshCi.exportVTK('meshCi.vtk')
meshCiPG_PGref = pv.read('meshCi.vtk')

meshLi.exportVTK('meshLi.vtk')
meshLiPG_PGref = pv.read('meshLi.vtk')


(meshCiPG, meshLiPG) = utils.define_mesh_transformations(meshCiPG_PGref,
                                                      meshLiPG_PGref,
                                                      idC=10, 
                                                      idL=10,
                                                    )


# def meshPGtoCATHY_transf()
# points = meshCiPG.points.copy()
# transform_matrix = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
# transformed_points_meshCi = np.dot(points, transform_matrix)
# transformed_points_meshCi[:,0] = idC/2
# transformed_points_meshCi.points = transformed_points_meshCi
# meshCiPG.points = transformed_points_meshCi

# points = meshLiPG.points.copy()
# transform_matrix = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
# transformed_points_meshLi = np.dot(points, transform_matrix)
# transformed_points_meshLi[:,1] = idL/2
# meshLiPG.points = transformed_points_meshLi

# mesh
# pl = pv.Plotter()
# pl.add_mesh(meshCiPG)
# pl.add_mesh(meshLiPG)
# pl.add_mesh(meshCATHY,opacity=0.5)
# pl.show_grid()
# pl.show()
    

ERT_meta_dict_meshLi={
                'forward_mesh_vtk_file': 'meshLi.vtk',
                'mesh_nodes_modif': meshLiPG.points
}

# In[24]:

meshpath = os.path.join(simu_solution.workdir,
                        simu_solution.project_name,
                        'vtk',
                        '110.vtk'
                       )
meshCATHY = pv.read(meshpath)
p = pv.Plotter(notebook=False)
p.add_mesh(meshCATHY, scalars='saturation',show_edges=True)
_ = p.add_bounding_box(line_width=5, color="black")
cpos = p.show()

        
meshLi_PG_withSaturation, scalar_new = cathy_meshtools.CATHY_2_pg(meshCATHY,
                                                       ERT_meta_dict_meshLi,
                                                        # show=True,
                                                      )


#%%

from pyCATHY.ERT import petro_Archie as Archie

scheme = ert.createData(elecs=np.linspace(start=0, 
                                          stop=10, 
                                          num=21),
                           schemeName='dd')




sc = {}
rFluid_Archie = 1/(588*(1e-6/1e-2))
sc = {
    "POROS": [simu_solution.soil_SPP['SPP_map']['POROS'].mean()],
    "rFluid_Archie": [rFluid_Archie],
    "a_Archie": [0.3],
    "m_Archie": [1.7],
    "n_Archie": [1.7],
    "pert_sigma_Archie": [0],
}

df_sw, _ = simu_solution.read_outputs(filename='sw')

for i in range(len(df_sw.index)):

    # read in CATHY mesh data
    # ------------------------------------------------------------------------
    path_CATHY = os.path.join(simu_solution.workdir, 
                              simu_solution.project_name , 
                              'vtk/'
                              )
    if i<10:    
        filename = "10" + str(i) + ".vtk"
    elif i<100:
        filename = "1" + str(i) + ".vtk"
    elif i<200:
        newnb = [int(x) for x in str(i)]
        filename = "2" + str(newnb[1]) + str(newnb[2])  + ".vtk"
    elif i<300:
        newnb = [int(x) for x in str(i)]
        filename = "3" + str(newnb[1]) + str(newnb[2])  + ".vtk"
    
    meshCATHY = pv.read(path_CATHY+filename)
            
    meshLi_PG_withSaturation, scalar_new = cathy_meshtools.CATHY_2_pg(meshCATHY,
                                                                      ERT_meta_dict_meshLi,
                                                                      # show=True,
                                                                      )
    saturation = np.array(meshLi_PG_withSaturation[scalar_new])
    ER_converted_ti_noNoise = Archie.Archie_rho_DA(
                                                   sat = [saturation],
                                                   rFluid_Archie=sc['rFluid_Archie'], 
                                                   porosity=sc['POROS'], 
                                                   a_Archie=sc['a_Archie'], 
                                                   m_Archie=sc['m_Archie'], 
                                                   n_Archie=sc['n_Archie'],
                                                   pert_sigma_Archie=[0]
                                                   )
    rhomap = np.array(ER_converted_ti_noNoise)
    data = ert.simulate(meshLi, 
                        scheme=scheme, 
                        res=rhomap, 
                        noiseLevel=1,
                        noiseAbs=1e-6, 
                        seed=1337
                        )
    
    data.save('ERTsolution/ERT_Li_' + str(i) + '.data')




#%%

# pl = pv.Plotter()
# pl.add_mesh(mesh_new_attr,
#             scalars=scalar_new,
#             clim=[0.413, 0.505],
#             )

# pl.add_mesh(meshCATHY,
#             opacity=0.2,
#             scalars='saturation',
#             clim=[0.413, 0.505],
#             )
# pl.show_grid()
# pl.show()
