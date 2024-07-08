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
from pyCATHY.DA.cathy_DA import DA, dictObs_2pd
from pyCATHY.DA.observations import read_observations, prepare_observations, make_data_cov
from pyCATHY.DA.perturbate import perturbate_parm
from pyCATHY.DA import perturbate

import utils


# In[2]:


import pygimli as pg
from pygimli.physics import ert
import pygimli.meshtools as mt


# In[3]:


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
import ipywidgets as widgets


# The notebooks does **not** describe: 
# - **Preprocessing step**: build a mesh, inputs preparations, ...
# 
# The notebooks describe: 
# 
# Create [synthetic soil moisture content dataset](#fwd_mod_sol) and add [noise](#add_noise_SMC). 
# 
# 1. [**Prepare for Data Assimilation**](prep_DA)
#    - 2.1 Read observations
#    - 2.1.2 Create covariance matrices
#    - 2.2 Perturbate
#    - 2.3 Define mapping operator
#    
# 2. **Simulation**: [solve the surface-subsurface flow](#solve).

#%%

import tomllib
with open("scenario.toml", "rb") as f:
    scenario = tomllib.load(f)['scenario']
scenario
# name_scenario = 'ic_test'
# name_scenario = 'ic'
name_scenario = 'ZROOT'
# name_scenario = 'ic_ZROOT_NOupdate'
# name_scenario = 'ic_ZROOT_upd_ZROOT'

# In[4]:


# prjName = 'meshLi_withDA'
prjName = 'meshLi_withoutDA'

# prjName = 'meshCi_withDA'
#prjName = 'meshCi_withoutDA'


# ### Set ERT data error estimate

# In[5]:


ERT_noise = 50
if 'without' in prjName:
    ERT_noise = 1e99


# In[6]:


# Create a CATHY project
# -----------------------
simu_solution = cathy_tools.CATHY(
                                    dirName='./solution_ERT/',
                                    prj_name= 'ERT_dataset',
                                    notebook=True,
                                  )
simu_solution.workdir


# - **ERT P1**: Longitudinal profile
# - **ERT P2**: Transversal profile 
# 
# ```{caution}
#     Project names: 
#     
#       - "without DA" means that the level of noise set to the observed data is infinite (1e99) (open loop run)
#         See the [import SMC section](#import_SMC_obs)
#       - "with DA" with a noise level according to sensors uncertainties
#       
#     Choose and comment the simulation you need to run!
# ```

# In[7]:


# Create a CATHY project
# -----------------------
simuWithDA = DA(
                        dirName='./DA_ERT',
                        #prj_name= choose_simulation.value,
                        prj_name= prjName + '_' + name_scenario, 
                        notebook=True,
                    )

simuWithDA.run_preprocessor()
simuWithDA.run_processor(IPRT1=3)


# In[8]:

import os
import shutil

# Define paths
origin_path = os.path.join(simu_solution.workdir, simu_solution.project_name)
dest_path = os.path.join(simuWithDA.workdir, simuWithDA.project_name)

# Function to copy files recursively
def copy_files_recursively(src, dst):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        src_item = os.path.join(src, item)
        dst_item = os.path.join(dst, item)
        if os.path.isdir(src_item):
            copy_files_recursively(src_item, dst_item)
        else:
            shutil.copy2(src_item, dst_item)

# Function to remove all files in a folder
def remove_all_files_in_folder(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            os.remove(os.path.join(root, file))

# Copy files from origin to destination
copy_files_recursively(origin_path, dest_path)

# Remove all files in vtk and outputs folders in the destination path
for folder in ['vtk', 'outputs']:
    remove_all_files_in_folder(os.path.join(dest_path, folder))

print("Files copied and vtk/outputs folders cleaned.")


simuWithDA.update_soil(PMIN=-1e25)


simu_solution.update_ic(INDP=0,IPOND=0,
                        pressure_head_ini=-1,
                        )


df_atmbc = simuWithDA.read_inputs('atmbc')

# In[9]:


meshCiPG_PGref = pv.read('../solution_ERT/meshCi.vtk')
meshLiPG_PGref = pv.read('../solution_ERT/meshLi.vtk')

yshift = 8
xshift = 8


(meshCiPG, meshLiPG) = utils.define_mesh_transformations(meshCiPG_PGref,
                                                          meshLiPG_PGref,
                                                          idC=yshift, 
                                                          idL=xshift,
                                                        )
meshCiPG_PGref.points
# meshCiPG.points
meshCiPG.points

meshLiPG_PGref.points
meshLiPG.points

# In[12]:


pt1, _ = simu_solution.find_nearest_node([0,yshift,meshLiPG.points[:,2].max()])
pt2, _ = simu_solution.find_nearest_node([5,yshift,meshLiPG.points[:,2].max()])
pt3, _ = simu_solution.find_nearest_node([10,yshift,meshLiPG.points[:,2].max()])


pt1, _ = simu_solution.find_nearest_node([xshift,0,meshLiPG.points[:,2].max()])
pt2, _ = simu_solution.find_nearest_node([xshift,5,meshLiPG.points[:,2].max()])
pt3, _ = simu_solution.find_nearest_node([xshift,10,meshLiPG.points[:,2].max()])



pl = pv.Plotter(notebook=True)
mesh = pv.read(os.path.join(simu_solution.workdir,
                                simu_solution.project_name,
                                'vtk/100.vtk',
                               )
       )
pl.add_mesh(mesh,
           opacity=0.2
           )
# pl.add_mesh(meshCiPG,
#               color='red'
#               )
pl.add_points(mesh.points[pt1],
              color='red',
              render_points_as_spheres=True,
              point_size=10
              )

pl.add_points(mesh.points[pt2],
              color='red',
              render_points_as_spheres=True,
              point_size=10
              )

pl.add_points(mesh.points[pt3],
              color='red',
              render_points_as_spheres=True,
              point_size=10
              )

pl.add_mesh(meshLiPG,
              color='b'
              )
pl.show_grid()
pl.show()

nodeIds = [pt1,pt2,pt3]


# In[10]:

df_sw, sw_times = simu_solution.read_outputs('sw')

fig, ax = plt.subplots()
for nn in nodeIds:
    ax.plot(df_sw.index,
            df_sw[nn],
            label=nn
            )
ax.set_xlabel('Time (s)')
ax.set_ylabel('Saturation (-)')
plt.legend()


# In[11]:

SPP_df, FP_df =  simu_solution.read_inputs('soil',MAXVEG=1)
SPP_df.head()
POROS = SPP_df['POROS'].mean()

# In[22]:
scenario_selected = scenario[name_scenario]

#%%
# Define mapping operator
# ----------------------
# rFluid_Archie = 1/(infitration_log_selec['water_EC_(microS/cm)']*(1e-6/1e-2))
print('Unknown fluid conductivity')
rFluid_Archie = 1/(588*(1e-6/1e-2))
rFluid_Archie
simuWithDA.set_Archie_parm(
                            rFluid_Archie=[rFluid_Archie],
                            a_Archie=[0.3],
                            m_Archie=[1.7],
                            n_Archie=[1.7],
                            pert_sigma_Archie=[0],
                            )
# In[13]:


# Round elements in ERT_times to the nearest hundred ! this is a bug in the simulation
sw_times_rounded = [round(swt, -2) for swt in sw_times]
common_elements = sorted(set(sw_times_rounded) & set(df_atmbc.time))
indices_sw_times = [sw_times_rounded.index(element) for element in common_elements]
sw_times_common_rounded = [sw_times_rounded[idx] for idx in indices_sw_times]


# In[ ]:





# In[14]:


print(simuWithDA.project_name)
if 'meshLi' in simuWithDA.project_name:
    meshERT = meshLiPG
    # meshERT = meshLiPG_PGref.points
    forward_mesh_vtk_file ='../solution_ERT/meshLi.vtk' # path to the ERT mesh (vtk file compatible with pygimli or resipy)
else:
    meshERT = meshCiPG
    # meshERT = meshCiPG_PGref
    forward_mesh_vtk_file ='../solution_ERT/meshCi.vtk'

# meshLiPG_PGref.points
# meshLiPG.points

# In[15]:


dict_obs = {} # initiate the dictionnary

# ERT observations metadata
# -------------------------
metadata_ERT    = {
                    'data_type': '$ERT$', # units
                    'units': '$\Ohm$', # units transfer_resistances
                    'forward_mesh_vtk_file':forward_mesh_vtk_file, # path to the ERT mesh (vtk file compatible with pygimli or resipy)
                    # 'sequenceERT': sequenceERT, # path to the ERT sequence  (file compatible with pygimli or resipy)
                    # 'instrument': 'Syscal', # Instrument
                    'data_format': 'pygimli', # format (raw or preprocessed)
                    'dataErr': ERT_noise, # error in %
                    'fwdNoiseLevel': 5, # error in %
                    'mesh_nodes_modif': meshERT.points, 
        
                }

# sw_times_common_rounded.insert(0,0)

for i, tt in enumerate(sw_times_common_rounded):

    # if i == 1:
    #     pass
    
    # .data file observation generated by pygimli
    filename = os.path.join(simu_solution.workdir,
                            'ERTsolution', 
                            'meshLi' + str(i) + '.data'
                            )
    
    data_measure = read_observations(
                                    dict_obs,
                                    filename, 
                                    data_type = 'ERT', 
                                    data_err = metadata_ERT['dataErr'], # instrumental error
                                    show=True,
                                    tA=tt,
                                    obs_cov_type='data_err', #data_err
                                    # elecs=elecs,
                                    meta=metadata_ERT,
                                    # datetime=ERT_datetimes[i]
                                    ) # data_err  reciprocal_err
    
data_measure_df = dictObs_2pd(dict_obs) 
data_measure_df.head()

data_measure_df
# In[16]:


data_cov, data_pert, stacked_data_cov = make_data_cov(
                                                        simuWithDA,
                                                        dict_obs,
                                                        list_assimilated_obs = 'all',
                                                        nb_assimilation_times=len(dict_obs)
                                                        )

simuWithDA.stacked_data_cov = stacked_data_cov
simuWithDA.stacked_data_cov[0]


# In[17]:


# simuWithDA.update_soil()


# ## Define parameter perturbation
# 
# :::{tip} **Parameter perturbation**
# 
# In this example we use an ensemble size (`NENS`) of **32** realizations. According to the scenario previously defined **IC** the initial condition in pressure head, and **ZROOT** the root depth are perturbated using a Gaussian sampling centered on their respective mean and std values (also defined in the scenario)
# 
# 
# The loop over the list of perturbated parameters is needed to call the function `perturbate_var` as many times as variables to perturbate. This will return a dictionnary merging all variable perturbate to parse into `run_DA_sequential`.
#     
# :::

# In[18]:


# NENS = 32
NENS = 32
list_pert = perturbate.perturbate(
                                    simuWithDA, 
                                    # scenario['ic_ZROOT_NOupdate'], 
                                    scenario_selected, 
                                    NENS,
                                 )

var_per_dict_stacked = {}
for dp in list_pert:

    var_per_dict_stacked = perturbate_parm(
                                var_per_dict_stacked,
                                parm=dp, 
                                type_parm = dp['type_parm'], # can also be VAN GENUCHTEN PARAMETERS
                                mean =  dp['mean'],
                                sd =  dp['sd'],
                                sampling_type =  dp['sampling_type'],
                                ensemble_size =  dp['ensemble_size'], # size of the ensemble
                                per_type= dp['per_type'],
                                savefig= os.path.join(simuWithDA.workdir,
                                                      simuWithDA.project_name,
                                                      simuWithDA.project_name + dp['savefig'])
                                )

simuWithDA.update_parm(
                        IPRT1=2,
                        TRAFLAG=0,
                        DTMIN=1e-2,DTMAX=1e3,DELTAT=1,
                        VTKF=2,
                        )



# (solve)=
# ## Solve Data Assimilation Sequentially

# In[ ]:


simuWithDA.run_DA_sequential(
                              parallel=True,    
                              dict_obs= dict_obs,
                              list_assimilated_obs='all', # default
                              list_parm2update=scenario_selected['listUpdateParm'],
                              DA_type='enkf_Evensen2009_Sakov',
                              dict_parm_pert=var_per_dict_stacked,
                              open_loop_run=False,
                              threshold_rejected=80,
                              # damping=1                   
                            )


# In[ ]:




