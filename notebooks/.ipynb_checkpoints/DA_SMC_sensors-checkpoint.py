#!/usr/bin/env python
# coding: utf-8

# ---
# title: Working with pyCATHY and DA
# subtitle: DA and SWC sensors assimilation
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

# In[1]:


import pyCATHY
from pyCATHY import cathy_tools
import pygimli as pg


# pyCATHY holds for **python** CATHY. Essentially it wraps the CATHY core to update/modify on the input files and has some **potential applications to ease preprocessing step**. pyCATHY also includes modules for Data Assimilation

# In[35]:


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
mpl.rcParams.update({"axes.grid":True, "grid.color":"gray", "grid.linestyle":'--','figure.figsize':(10,10)})
import pandas as pd
# ss
# sssss
# The notebooks does **not** describe: 
# - **Preprocessing step**: build a mesh, inputs preparations, ...
# 
# The notebooks describe: 
# 
# 1. **Prepare for Data Assimilation**
#    - 2.1 Read observations
#    - 2.1.2 Create matrice covariances
#    - 2.2 Perturbate
#    - 2.3 Define mapping operator
#    
# 2. **Simulation**: solve the surface-subsurface flow.
# 
# 3. **Plot outputs**: analysis of the results
#    - Saturation with uncertainties
#    - Assimilation performance

# In[3]:


# Create a CATHY project
# -----------------------
simuWithDA = DA(
                        dirName='./DA_SMC',
                        prj_name= 'Weill_example',
                        notebook=True,
                    )


# ## Forward model of the solution
# Run the simulation to get the model solution. This step is only necessary for this tutorial to generate synthetic observations of soil moisture content

# In[4]:


# Fetch dataset
# import git
# import shutil
# repo_path = './solution_SMC_tmp/'

# if os.path.exists(repo_path):
#     shutil.rmtree(repo_path)
# git.Repo.clone_from('https://github.com/CATHY-Org/weill_dataset.git', repo_path)

# # Define the source and destination paths for the folder to be copied
# source_folder = os.path.join(repo_path,'raw')
# destination_folder = './solution_SMC/weill_dataset'

# # Copy the folder
# if os.path.exists(destination_folder):
#     shutil.rmtree(destination_folder)
# shutil.copytree(source_folder, destination_folder)
# shutil.rmtree(repo_path)


# In[5]:


# Create a CATHY project
# -----------------------
simu_solution = cathy_tools.CATHY(
                                    dirName='./solution_SMC',
                                    prj_name= 'weill_dataset',
                                    notebook=True,
                                  )

# In[6]:


simu_solution.run_preprocessor()
# simu_solution.update_parm()
# tparm = simu_solution.parm['(TIMPRT(I),I=1,NPRT)']
# tatmbc = [0] + tparm

netValue = -1e-7
rain = 4e-7
tatmbc = list(np.linspace(0,86400,10))
# tatmbc = [int(ta) for ta in tatmbc]

netValue_list = [netValue]*len(tatmbc)
netValue_list[0] = netValue + rain
netValue_list[1] = netValue + rain
# sf
# netValue_list[1] = netValue + rain

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

simu_solution.update_parm(TIMPRTi=tatmbc)
simuWithDA.update_parm(TIMPRTi=tatmbc)

# simu_solution.read_inputs('atmbc')
# simuWithDA.read_inputs('atmbc')
simu_solution.parm
simu_solution.atmbc


simu_solution.update_soil(PMIN=-1e25)
simuWithDA.update_soil(PMIN=-1e25)


simu_solution.run_processor(IPRT1=3,verbose=True)


simu_solution.update_parm(
                            IPRT1=2,
                            TRAFLAG=0,
                            # DTMIN=1e-2,
                            # DTMAX=1e3,
                            # DELTAT=1e-2
                            )
# simu_solution.parm
simu_solution.run_processor(verbose=True)


# ### Get soil moisture data at 3 different depths


# In[7]:


node_pos = [5,5,2] 
depths = [0.05,0.15,0.25,0.75]

nodeId, closest_pos = simu_solution.find_nearest_node(node_pos)

for d in depths:
    node_pos = [5,5,closest_pos[0][2]-d] 
    nodeId_tmp, closest_pos_tmp = simu_solution.find_nearest_node(node_pos)
    nodeId.append(nodeId_tmp)
    closest_pos.append(closest_pos_tmp)

nodeIds = np.vstack(nodeId)
closest_positions = np.vstack(closest_pos)
print(closest_positions)


# In[8]:


pl = pv.Plotter(notebook=True)
mesh = pv.read(os.path.join(simu_solution.workdir,
                                simu_solution.project_name,
                                'vtk/100.vtk',
                               )
       )
pl.add_mesh(mesh,
           opacity=0.2
           )
pl.add_points(closest_positions,
             color='red'
             )
pl.show_grid()
pl.show()


# In[9]:


sw, sw_times = simu_solution.read_outputs('sw')
print(np.shape(sw))

fig, ax = plt.subplots()
for nn in nodeIds:
    ax.plot(sw_times,sw[:,nn],label=nn)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Saturation (-)')
plt.legend()


# Insert errors into the soil moisture

# In[10]:


SPP_df, FP_df =  simu_solution.read_inputs('soil',MAXVEG=1)
SPP_df.head()


# In[12]:


POROS = SPP_df['POROS'].mean()
sw2SMC = sw*POROS


# In[13]:


noise = np.random.normal(loc=0, 
                         scale=0.001, 
                         size=sw.shape
                        )  # introducing Gaussian noise with mean 0 and standard deviation 0.01
sw2SMC_with_errors = sw2SMC + noise


# In[14]:


fig, ax = plt.subplots()
for nn in nodeIds:
    ax.plot(sw_times,sw2SMC_with_errors[:,nn],color='k')
    ax.plot(sw_times,sw2SMC[:,nn],marker='.',color='grey')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Soil Moisture Content (%)')


# In[15]:


SMCtoAssimilate = np.hstack(sw2SMC_with_errors[:,nodeIds])


# In[16]:


sensors_names = ['SMC'+str(i) for i in range(5)]


# In[48]:
    
# sw_times_rounded = [round(swt / 100) * 100 for swt in sw_times] 
# common_elements = list(set(sw_times_rounded) & set(tatmbc))
# common_elements_sort = sorted(common_elements)
# indices_sw_times = [sw_times_rounded.index(element) for element in common_elements_sort]

# sw_times_common_rounded = [sw_times_rounded[itt] for itt in indices_sw_times]
# SMCtoAssimilate_common = SMCtoAssimilate[:,indices_sw_times]


# Round elements in sw_times to the nearest hundred ! this is a bug in the simulation
sw_times_rounded = [round(swt, -2) for swt in sw_times]
common_elements = sorted(set(sw_times_rounded) & set(tatmbc))
indices_sw_times = [sw_times_rounded.index(element) for element in common_elements]
sw_times_common_rounded = [sw_times_rounded[idx] for idx in indices_sw_times]
SMCtoAssimilate_common = SMCtoAssimilate[:, indices_sw_times]



SMC_sensors_df = pd.DataFrame(SMCtoAssimilate_common,
                              columns=sw_times_common_rounded,
                              index=sensors_names
                             )
# SMC_sensors_df = SMC_sensors_df.iloc[:, :-1] # drop duplicate collumn
SMC_sensors_df = SMC_sensors_df.T
SMC_sensors_df.head()
# dd

# ## 1. Prepare Data Assimilation
# 
# First we define a dictionnary `scenario` describing which and how of the model parameters are **perturbated** and **updated**. One example case is considered, where initial condition and root depth is perturbated and updated sequentially: 
# `ic_ZROOT_upd_ZROOT`
# 
# :::{admonition} Feddes parameters perturbation
# In order to perturbate Feddes parameters we use typical range of possible values. 
# For **ZROOT** the perturbation must be bounded within the limit of the simulation region. 
# This condition is applied thanks to the dictionnary key `per_bounds`.
# :::
# 
# - `per_type`
# - `per_name`
# - `per_nom`
# - `per_sigma`
# - `sampling_type`
# 

# In[28]:


scenario = {
    
            # scenario without parameter update
            # -------------------------------------
            'ic_ZROOT_NOupdate': {'per_type': [None,None],
                                 'per_name':['ic', 'ZROOT'],
                                 'per_nom':[-5,0.4],
                                 'per_mean':[-5,0.4],    
                                 'per_sigma': [1.75,5e-3],
                                 'per_bounds': [None,{'min':0,'max':0.5}],
                                 'sampling_type': ['normal']*2,
                                 'transf_type':[None,None],
                                'listUpdateParm': ['St. var.'],
                                },
    
            # scenario with parameter update
            # -------------------------------------
            'ic_ZROOT_upd_ZROOT': 
                                                        {'per_type': [None,None], 
                                                         'per_name':['ic', 'ZROOT'],
                                                         'per_nom':[-5,0.4],
                                                         'per_mean':[-5,0.4],    
                                                         'per_sigma': [1.75,5e-3],
                                                         'per_bounds': [None,{'min':0,'max':0.5}],
                                                         'sampling_type': ['normal']*2,
                                                         'transf_type':[None,None],
                                                         'listUpdateParm': ['St. var.', 'ZROOT']
                                                         },    
            }
print(scenario)


# ### 2.1 Import SMC observations
# 
# ```{tip}
#     need to call `read_observations` as many times as variable to perturbate 
#     return a dict merging all variable perturbate to parse into prepare_DA
# ```

# In[54]:


dict_obs = {} # initiate the dictionnary
data_ass_time_s = sw_times
pts_data_err = 5
dict_obs



# In[55]:


for i in range(len(SMC_sensors_df.columns)):
    for j, tt in enumerate(SMC_sensors_df.index):
        dict_obs = read_observations( 
                                        dict_obs,
                                        obs_2_add=SMC_sensors_df['SMC'+str(i)].iloc[j], 
                                        mesh_nodes = nodeIds[i],
                                        data_type='swc',
                                        data_err=pts_data_err,
                                        #date_range=[args.startD,args.endD],
                                        colname=' m³/m³ Water Content',
                                        tA=tt
                                        )
        # print(dict_obs)
    # mesh_nodes.append(mesh_node_pos)

data_measure_df = dictObs_2pd(dict_obs) 
data_measure_df.columns

#%% 

data_cov, data_pert, stacked_data_cov = make_data_cov(
                                                        simuWithDA,
                                                        dict_obs,
                                                        list_assimilated_obs = 'all',
                                                        nb_assimilation_times=len(dict_obs)
                                                        )

simuWithDA.stacked_data_cov = stacked_data_cov

# ### 2.2 Perturbate

# In[19]:


simuWithDA.run_preprocessor()
simuWithDA.run_processor(IPRT1=3)


# In[29]:


#simuWithDA.read_inputs('soil',NVEG=1)


# In[26]:


#simuWithDA.update_dem_parameters()
#simuWithDA.update_veg_map()
simuWithDA.update_soil()

simuWithDA.soil_SPP

# In[30]:


help(perturbate.perturbate)
NENS = 32
list_pert = perturbate.perturbate(
                                    simuWithDA, 
                                    scenario['ic_ZROOT_NOupdate'], 
                                    NENS,
                                 )

# In[31]:


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
                                savefig= os.path.join(simuWithDA.project_name,
                                                      simuWithDA.project_name + dp['savefig'])
                                )


# ### 2.3 Define mapping operator 
# For assimilation of soil moisture content data, the default mapping to get the water saturation is: porosity*SM
# Please refer to ?? to see the mapping tools of pyCATHY

# ### 3. Run sequential DA
# 
# Simply use `run_DA_sequential()` with the `simu_DA` object
# 
# Required arguments are:
# - **dict_obs**: dictionnary of observations
# - **list_assimilated_obs**: list of observation to assimilate 
# - **list_parm2update**: list of parameters to update 
# 
# Possible **optionnal** arguments are: 
# - **parallel**: if True use multiple cores to run many realisations at the same time
# - **DA_type**: type of data assimilation
# - **threshold_rejected**: threshold above which the simulation stops (i.e. ensemble of rejected realisation too big)
# - **damping**: add damping to inflate the covariance matrice
# 
# 
# ```{tip}
#     During the execution **useful informations are displayed on the console** in order to follow the state of the DA. You can for example appreciated how many ensemble are rejected.
# ```
# 

# In[ ]:

# ddd
parallel = True

#%%

# simuWithDA.update_atmbc()
# simuWithDA.read_inputs('atmbc')
# simu_solution.read_inputs('atmbc')
# simuWithDA.update_atmbc(HSPATM=0,
#                               IETO=0,
#                               time=atmbc_TOI,
#                               netValue=v_atmbc_TOI,
#                               )


simuWithDA.update_parm(
                        IPRT1=2,
                        TRAFLAG=0,
                        DTMIN=1e-2,
                        DTMAX=1e3,
                        DELTAT=1
                        )

# In[34]:


simuWithDA.run_DA_sequential(
                              parallel=parallel,    
                              dict_obs= dict_obs,
                              list_assimilated_obs='all', # default
                              list_parm2update=scenario['ic_ZROOT_NOupdate']['listUpdateParm'],
                              DA_type='enkf_Evensen2009_Sakov', #'pf_analysis', # default
                              dict_parm_pert=var_per_dict_stacked,
                              open_loop_run=False,
                              threshold_rejected=80,
                              # damping=1                   
                            )


# ### 4. Plot outputs

# In[ ]:

# Create a CATHY project
# -----------------------
simuWithDA = DA(
                        dirName='./DA_SMC',
                        prj_name= 'Weill_example',
                        notebook=True,
                    )

results = simuWithDA.load_pickle_backup()

observations = dictObs_2pd(results['dict_obs'])

nodes_of_interest = observations.mesh_nodes
nodes_of_interest = list(np.hstack(np.unique(nodes_of_interest)))


fig, axs = plt.subplots(1,3,figsize=(15,3),
                       sharey=True,
                       )


results['df_DA'].columns


ind2replace = np.where(results['df_DA'].time==1)[0]
results['df_DA'].time
assimilation_times = observations.index.get_level_values(1).unique().to_list()
unique_times = results['df_DA']["time"].unique()
results['df_DA']["assimilation_times"] = results['df_DA']["time"].map(dict(zip(unique_times, assimilation_times)))


from datetime import datetime
# # Get today's date
start_date = datetime.today().date()


for axi, NOI in zip(axs, nodes_of_interest):
    cplt.DA_plot_time_dynamic(results['df_DA'],
                              'sw',
                              NOI,
                              ax=axi,
                              keytime='assimilation_times',
                              )  
    
    
    