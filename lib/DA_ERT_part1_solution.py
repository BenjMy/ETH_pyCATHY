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
from pyCATHY.ERT import petro_Archie as Archie
from pyCATHY.importers import cathy_inputs as in_CT
import pyCATHY.meshtools as cathy_meshtools

import utils


# In[ ]:


import pygimli as pg
from pygimli.physics import ert
import pygimli.meshtools as mt


# In[2]:


import os
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib as mpl
# set some default plotting parameters for nicer looking plots
mpl.rcParams.update({"axes.grid":True, 
                     "grid.color":"gray", 
                     "grid.linestyle":'--',
                     'figure.figsize':(10,10)}
                   )
import pandas as pd


# We build a **twin simulation experiment** which means that the field observations are generated from a synthetic hydrological model of reference. Here the model mesh is based on Weill et al, a 3D regular slope (structured mesh). The atmospheric conditions describe a rain event followed by drainage and ET. Two ERT profiles (unstructured 2D mesh) are defined in both longitudinal and transversal direction of the hydrological mesh. The saturation predicted is interpolated into the ERT meshes and then converted into apparent resistivities using Archie law to generated the observations. To account for typical field errors, the apparent resistivities are noised. 
# 
# The notebooks does **not** describe: 
# - **Preprocessing step**: build a mesh, inputs preparations, ...
# 
# The notebooks describe: 
# - Create [synthetic ERT dataset](#fwd_mod_sol)
# - Add [noise](#add_noise_ERT). 
# 

# ## Create a pyCATHY object

# In[3]:


# Create a CATHY project
# -----------------------
simu_solution = cathy_tools.CATHY(
                                    dirName='./solution_ERT',
                                    prj_name= 'ERT_dataset',
                                    notebook=True,
                                  )

simu_solution.create_mesh_vtk()


# ## Check the mesh
# 
# :::{tip} **Problem statement**
# 
# In this example we use the defaut mesh (from Weill et al.) i.e. a geometry of the idealized hillslope. The domain is 10 m long, 10 m wide, and 10 m deep at its lowest point (i.e., at the outlet). The slopes are 10% and 1% along the x and y directions, respectively. The surface is discretized into 50 × 50 grid cells (i.e., a DEM resolution of 0.2 m). The subsurface domain is discretized vertically into 32 layers of varying thickness, with each layer parallel to the surface except for the last one, which has a flat base. 
# :::
# 
# 

# In[ ]:





# In[4]:


meshpath = os.path.join(simu_solution.workdir,
                        simu_solution.project_name,
                        'vtk',
                        simu_solution.project_name + '.vtk'
                       )
meshCATHY = pv.read(meshpath)


# In[5]:


np.linspace(0,86400,10)


# ## Change atmospheric boundary conditions and run simulation
# 
# A rain event that last 5.3 hours of intensity **4e-7** m/s. Then evapotranspiration **-1e-7** m/s.

# In[7]:


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
simu_solution.update_parm(TIMPRTi=tatmbc)


# Homogeneoous pressure head initial (**-1m**) conditions all over the domain. No ponding. 

# In[8]:


simu_solution.update_ic(INDP=0,IPOND=0,
                        pressure_head_ini=-1,
                        )


# Switching boundary condition set at soil PMIN=-1e25 --> soil process is controlled via Feddes parameters

# In[9]:


simu_solution.update_soil(PMIN=-1e25)


# run prepocessor and processor

# In[10]:


simu_solution.run_preprocessor()
simu_solution.run_processor(IPRT1=3)
#simu_solution.run_processor(IPRT1=2,TRAFLAG=0,VTKF=2,)


# ## Create 2D pygimli mesh (on top of CATHY mesh)

# In[11]:


dem_mat, str_hd_dem = in_CT.read_dem(
                                    os.path.join(simu_solution.workdir, 
                                                 simu_solution.project_name, 
                                                 "prepro/dem"),
                                    os.path.join(simu_solution.workdir, 
                                                 simu_solution.project_name, 
                                                 "prepro/dtm_13.val"),
                                    )


idC = 8
idL = 8

import pyCATHY.meshtools as mtCT
layers_top, layers_bottom = mtCT.get_layer_depths(simu_solution.dem_parameters)



minmax_dem_mat_Ci = [min(dem_mat[:,idC]), max(dem_mat[:,idC])]
minmax_dem_mat_Li = [min(dem_mat[idL,:]), max(dem_mat[idL,:])]
safe_depth = -0.15

geomCi = mt.createPolygon([[0.0, -1], [0, minmax_dem_mat_Ci[0]+ safe_depth], [10, minmax_dem_mat_Ci[1]+ safe_depth], [10, -1]],
                          isClosed=True, marker=1, area=1e-2)
meshCi = mt.createMesh(geomCi, quality=34.3, area=3, smooth=[1, 10])

geomLi = mt.createPolygon([[0.0, -1], [0, minmax_dem_mat_Li[1]+ safe_depth], [10, minmax_dem_mat_Li[0]+ safe_depth], [10, -1]],
                          isClosed=True, marker=1, area=1e-2)
meshLi = mt.createMesh(geomLi, quality=34.3, area=3, smooth=[1, 10])


# In[1]:


dem_mat


# In[12]:


pg.show(meshLi)


# ### Export and read with pyvista

# In[13]:


meshCi.exportVTK('meshCi.vtk')
meshCiPG_PGref = pv.read('meshCi.vtk')

meshLi.exportVTK('meshLi.vtk')
meshLiPG_PGref = pv.read('meshLi.vtk')


(meshCiPG, meshLiPG) = utils.define_mesh_transformations(meshCiPG_PGref,
                                                         meshLiPG_PGref,
                                                         idC=idC, 
                                                         idL=idL,
                                                         )
meshCiPG


# In[14]:


pl = pv.Plotter()
pl.add_mesh(meshCiPG)
pl.add_mesh(meshLiPG)
pl.add_mesh(meshCATHY,opacity=0.5)
pl.show_grid()
pl.show()


# ## Interpolate CATHY mesh on PG mesh

# In[15]:


ERT_meta_dict_meshLi={
                'forward_mesh_vtk_file': 'meshLi.vtk',
                'mesh_nodes_modif': meshLiPG.points
}

ERT_meta_dict_meshCi={
                'forward_mesh_vtk_file': 'meshCi.vtk',
                'mesh_nodes_modif': meshCiPG.points
}


meshpath = os.path.join(simu_solution.workdir,
                        simu_solution.project_name,
                        'vtk',
                        '110.vtk'
                       )
meshCATHY = pv.read(meshpath)

p = pv.Plotter(notebook=True)
p.add_mesh(meshCATHY, scalars='saturation',show_edges=True)
_ = p.add_bounding_box(line_width=5, color="black")
cpos = p.show()


# ## Forward modelling of ER apparent datasets

# ### Create ERT scheme

# In[16]:


scheme = ert.createData(elecs=np.linspace(start=0, 
                                          stop=10, 
                                          num=72),
                           schemeName='dd')


# ### Define Archie Parameters

# In[17]:


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


# (fwd_mod_sol)=
# ### ERT simulate
# 
# Steps:
# - Read the hydrological mesh containing the `saturation` water
# - Interpolate the `saturation` to pygimli mesh
# - Convert saturation to electrical resistivity (`ER_hydro`) using **Archie's law**
# - Simulate Apparent resistivities (`ERa`) using the `ER_hydro` as resistivity map

# In[29]:


# mesh2use = pg.load(mesh['forward_mesh_vtk_file'])
# mesh2use


# In[33]:


for mesh_dir_i in ['meshLi','meshCi']:
    #mesh = eval('ERT_meta_dict_'+ mesh_dir_i)['forward_mesh_vtk_file']
    mesh = pg.load(eval('ERT_meta_dict_'+ mesh_dir_i)['forward_mesh_vtk_file'])
    print('+'*20)   
    for i in range(len(tatmbc)):
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
        
        meshCATHY = pv.read(path_CATHY+filename)       
        mesh_PG_withSaturation, scalar_new = cathy_meshtools.CATHY_2_pg(meshCATHY,
                                                                        eval('ERT_meta_dict_'+ mesh_dir_i),
                                                                          # show=True,
                                                                          )
        saturation = np.array(mesh_PG_withSaturation[scalar_new])
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
        mesh_PG_withSaturation['ER'] = rhomap
        
        data = ert.simulate(mesh, 
                            scheme=scheme, 
                            res=rhomap, 
                            noiseLevel=1,
                            noiseAbs=1e-6, 
                            seed=1337
                            )
        data.save('ERTsolution/' + mesh_dir_i + str(i) + '.data')
        mesh_PG_withSaturation.save('ERTsolution/' + mesh_dir_i + str(i) + '.vtk')        
        # mesh_PG_withSaturation.exportVTK('ERTsolution/' + mesh_dir_i + str(i) + '.vtk')