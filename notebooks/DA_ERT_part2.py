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
import utils

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

# ```{caution}
#     Project names: 
#     
#       - "without DA" means that the level of noise set to the observed data is infinite (1e99) (open loop run)
#         See the [import SMC section](#import_SMC_obs)
#       - "with DA" with a noise level according to sensors uncertainties
#       
#     Choose and comment the simulation you need to run!
# ```

# In[3]:


# Create a CATHY project
# -----------------------
simuWithDA = DA(
                        dirName='./DA_ERT',
                        prj_name= 'Weill_example_withoutDA', # without DA means that the level of noise set to data covariance is infinite (open loop run); 
                        #prj_name= 'Weill_example_withDA', 
                        notebook=True,
                    )


# (fwd_mod_sol)= 
# ## Forward model of the solution
# Run the simulation to get the model solution. This step is only necessary for this tutorial to generate synthetic observations of soil moisture content

# ### Change atmospheric boundary conditions

# In[4]:


# Create a CATHY project
# -----------------------
simu_solution = cathy_tools.CATHY(
                                    dirName='./solution_ERT',
                                    prj_name= 'weill_dataset',
                                    notebook=True,
                                  )


# In[5]:


simuWithDA.run_preprocessor()
simu_solution.run_preprocessor()


# In[6]:


simuWithDA.run_processor(IPRT1=3)
simu_solution.run_processor(IPRT1=3)


# In[7]:


netValue = -1e-7
rain = 4e-7
tatmbc = list(np.linspace(0,86400,10))

netValue_list = [netValue]*len(tatmbc)
netValue_list[0] = netValue + rain
netValue_list[1] = netValue + rain

# In[8]:


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

simuWithDA.update_atmbc(
                        HSPATM=1,
                        IETO=0,
                        time=tatmbc,
                        # VALUE=[None, None],
                        netValue=netValue_list,
                )


# In[9]:


simu_solution.update_parm(TIMPRTi=tatmbc)
simuWithDA.update_parm(TIMPRTi=tatmbc)

# simu_solution.read_inputs('atmbc')
# simuWithDA.read_inputs('atmbc')
simu_solution.parm
simu_solution.atmbc


simu_solution.update_soil(PMIN=-1e25)
simuWithDA.update_soil(PMIN=-1e25)


# In[10]:


# simu_solution.run_preprocessor()
# simu_solution.run_processor(IPRT1=2,
#                             TRAFLAG=0
#                             )


# ### Get soil moisture data at 3 different depths

# In[11]:

meshCiPG_PGref = pv.read('meshCi.vtk')
meshLiPG_PGref = pv.read('meshLi.vtk')

yshift = 8
xshift = 8


(meshCiPG, meshLiPG) = utils.define_mesh_transformations(meshCiPG_PGref,
                                                          meshLiPG_PGref,
                                                          idC=yshift, 
                                                          idL=xshift,
                                                        )

# In[12]:


pt1, _ = simu_solution.find_nearest_node([0,yshift,meshLiPG.points[:,2].max()])
pt2, _ = simu_solution.find_nearest_node([5,yshift,meshLiPG.points[:,2].max()])
pt3, _ = simu_solution.find_nearest_node([10,yshift,meshLiPG.points[:,2].max()])


pt1, _ = simu_solution.find_nearest_node([xshift,0,meshLiPG.points[:,2].max()])
pt2, _ = simu_solution.find_nearest_node([xshift,5,meshLiPG.points[:,2].max()])
pt3, _ = simu_solution.find_nearest_node([xshift,10,meshLiPG.points[:,2].max()])



pl = pv.Plotter(notebook=False)
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




SPP_df, FP_df =  simu_solution.read_inputs('soil',NVEG=1)
SPP_df.head()


# In[16]:

POROS = SPP_df['POROS'].mean()


# In[22]:


scenario = {
    
            # scenario without parameter update
            # -------------------------------------
            'ic_ZROOT_NOupdate': {'per_type': [None,None],
                                 'per_name':['ic', 'ZROOT'],
                                 'per_nom':[-1,0.4],
                                 'per_mean':[-1,0.4],    
                                 'per_sigma': [1.75,5e-3],
                                 'per_bounds': [None,{'min':0,'max':1.5}],
                                 'sampling_type': ['normal']*2,
                                 'transf_type':[None,None],
                                'listUpdateParm': ['St. var.'],
                                },
    
            # scenario with parameter update
            # -------------------------------------
            'ic_ZROOT_upd_ZROOT': 
                                                        {'per_type': [None,None], 
                                                         'per_name':['ic', 'ZROOT'],
                                                         'per_nom':[-1,0.4],
                                                         'per_mean':[-1,0.4],    
                                                         'per_sigma': [1.75,5e-3],
                                                         'per_bounds': [None,{'min':0,'max':1.5}],
                                                         'sampling_type': ['normal']*2,
                                                         'transf_type':[None,None],
                                                         'listUpdateParm': ['St. var.', 'ZROOT']
                                                         },    
            }
print(scenario)



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

#%%


# Round elements in ERT_times to the nearest hundred ! this is a bug in the simulation
sw_times_rounded = [round(swt, -2) for swt in sw_times]
common_elements = sorted(set(sw_times_rounded) & set(tatmbc))
indices_sw_times = [sw_times_rounded.index(element) for element in common_elements]
sw_times_common_rounded = [sw_times_rounded[idx] for idx in indices_sw_times]

# len(sw_times_common_rounded)
# len(df_sw.index)
# df_sw.index

# ERTtoAssimilate_common = ERTtoAssimilate[:, indices_sw_times]



# In[23]:


# pts_data_err = 5e-3
pts_data_err = 1e99 # this correspond to the simulation "withoutDA"


# In[24]:


dict_obs = {} # initiate the dictionnary

# ERT observations metadata
# -------------------------
metadata_ERT    = {
                    'data_type': '$ERT$', # units
                    'units': '$\Ohm$', # units transfer_resistances
                    'forward_mesh_vtk_file': '../solution_ERT/meshLi.vtk', # path to the ERT mesh (vtk file compatible with pygimli or resipy)
                    # 'sequenceERT': sequenceERT, # path to the ERT sequence  (file compatible with pygimli or resipy)
                    # 'instrument': 'Syscal', # Instrument
                    'data_format': 'pygimli', # format (raw or preprocessed)
                    'dataErr': pts_data_err, # error in %
                    'fwdNoiseLevel': 5, # error in %
                    'mesh_nodes_modif': meshLiPG.points, 
        
                }

# sw_times_common_rounded.insert(0,0)

for i, tt in enumerate(sw_times_common_rounded):

    # if i == 1:
    #     pass
    
    # .data file observation generated by pygimli
    filename = os.path.join(simu_solution.workdir,
                            'ERTsolution', 
                            'ERT_Li_' + str(i) + '.data'
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




data_cov, data_pert, stacked_data_cov = make_data_cov(
                                                        simuWithDA,
                                                        dict_obs,
                                                        list_assimilated_obs = 'all',
                                                        nb_assimilation_times=len(dict_obs)
                                                        )

simuWithDA.stacked_data_cov = stacked_data_cov


# In[29]:


simuWithDA.update_soil()


# In[30]:

NENS = 3
list_pert = perturbate.perturbate(
                                    simuWithDA, 
                                    scenario['ic_ZROOT_NOupdate'], 
                                    NENS,
                                 )

# In[32]:


var_per_dict_stacked = {}
for dp in list_pert:
    # need to call perturbate_var as many times as variable to perturbate
    # return a dict merging all variable perturbate to parse into prepare_DA
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

# In[34]:


simuWithDA.update_parm(
                        IPRT1=2,
                        TRAFLAG=0,
                        DTMIN=1e-2,DTMAX=1e3,DELTAT=1
                        )


# In[35]:


simuWithDA.run_DA_sequential(
                              parallel=True,    
                              dict_obs= dict_obs,
                              list_assimilated_obs='all', # default
                              list_parm2update=scenario['ic_ZROOT_NOupdate']['listUpdateParm'],
                              DA_type='enkf_Evensen2009_Sakov',
                              dict_parm_pert=var_per_dict_stacked,
                              open_loop_run=False,
                              threshold_rejected=80,
                              # damping=1                   
                            )

# In[ ]:



