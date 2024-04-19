#!/usr/bin/env python
# coding: utf-8

# ---
# title: Working with pyCATHY and DA
# subtitle: PART2 - plotting the results
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

# The notebooks describe: 
# 
# **Plot outputs**: analysis of the results
#    - [Saturation with uncertainties](plot_states)
#    - [Parameters convergence](Parm_evol)
#    - [Assimilation performance](DA_perf)

# In[1]:


import pyCATHY
from pyCATHY import cathy_tools
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
                     'figure.figsize':(6,6)}
                   )
import pandas as pd


# (plot_results)=
# # Analysis of the results

# ## Reload solution

# In[2]:


simu_solution = cathy_tools.CATHY(
                                    dirName='./solution_SMC',
                                    prj_name= 'weill_dataset',
                                    notebook=True,
                                  )
sw_SOL, sw_SOL_times = simu_solution.read_outputs('sw')
psi_SOL = simu_solution.read_outputs('psi')

rootMap, rootMap_hd = simu_solution.read_inputs('root_map')

soil_SPP_SOL, soil_FP_SOL = simu_solution.read_inputs('soil',
                                                         MAXVEG=int(np.max(rootMap)),
                                                    )
PERMX_SOL = soil_SPP_SOL['PERMX'].unique()[0]
POROS_SOL = soil_SPP_SOL['POROS'].unique()[0]


# ## Reload DA results

# In[3]:


simuWithDA = DA(
                        dirName='./DA_SMC',
                        prj_name= 'Weill_example',
                        notebook=True,
                    )

#%%
# import pyvista as pv
# pl= pv.Plotter()
# cplt.show_vtk(
#                 filename=os.path.join(
#                                         simuWithDA.workdir,
#                                         simuWithDA.project_name,
#                                         'DA_Ensemble/cathy_31/vtk/100.vtk',
#                         ),
#                 ax=pl,   
#             )
# pl.show()

# In[4]:


results = simuWithDA.load_pickle_backup()


# In[5]:


observations = dictObs_2pd(results['dict_obs'])


# In[6]:


fig, ax = plt.subplots()
observations.xs('swc',level=0).plot(y='data',ax=ax,label='swc')
observations.xs('swc1',level=0).plot(y='data',ax=ax,label='swc1')
observations.xs('swc2',level=0).plot(y='data',ax=ax)


# In[7]:


assimilation_times = observations.index.get_level_values(1).unique().to_list()


# In[8]:


nodes_of_interest = observations.mesh_nodes
nodes_of_interest = list(np.hstack(np.unique(nodes_of_interest)))
nodes_of_interest


# In[9]:


#nodes_of_interest_pos
#sensors_pos = infiltration_util.get_sensors_pos()


# (plot_states)=
# ## Plot model saturation with uncertainties 

# In[10]:


parm_df = pd.DataFrame.from_dict(results['dict_parm_pert'],
                                 orient='index').T


# In[11]:


fig, ax = plt.subplots()
ax.hist(parm_df['ic']['ini_perturbation'])


# In[12]:


obs2plot = observations.xs('swc',level=0)
obs2plot['saturation'] = obs2plot['data']/0.55


# In[21]:


# dict(zip(unique_times, assimilation_times))


# In[23]:


unique_times = results['df_DA']["time"].unique() -1  #+ 3
results['df_DA']["assimilation_times"] = results['df_DA']["time"].map(dict(zip(unique_times, assimilation_times)))
results['df_DA'].head()


# In[19]:


fig, axs = plt.subplots(1,3,figsize=(15,3),
                       sharey=True
                       )
for axi, NOI in zip(axs, nodes_of_interest):
    cplt.DA_plot_time_dynamic(results['df_DA'],
                              'sw',
                              NOI,
                              ax=axi,
                              keytime='assimilation_times',
                              )  
    axi.plot(sw_SOL_times,
            sw_SOL[:,NOI],
            color='r',
            marker= '.'
            )
    
    axi.plot(obs2plot.index, 
             obs2plot['saturation'].values,
             color='green'
            )
    


# In[15]:


fig, axs = plt.subplots(1,3,figsize=(15,3),
                       sharey=True
                       )
for axi, NOI in zip(axs, nodes_of_interest):
    cplt.DA_plot_time_dynamic(results['df_DA'],
                              'sw',
                              NOI,
                              ax=axi,
                              )  


# (Parm_evol)=
# ## Plot parameters convergence

# In[16]:


fig, axs = plt.subplots(2,1,figsize=[11,4],
                        # sharex=True
                        )
cplt.DA_plot_parm_dynamic_scatter(parm = 'ZROOT', 
                                  dict_parm_pert=results['dict_parm_pert'], 
                                  list_assimilation_times = assimilation_times,
                                  ax=axs[0],
                                          )   
axs[0].plot(np.linspace(1,len(assimilation_times), len(assimilation_times)),
            [POROS_SOL]*len(assimilation_times))


# (DA_perf)= 
# ## Plot DA performance

# In[ ]:


get_ipython().run_cell_magic('capture', '', "fig, ax = plt.subplots(2,figsize=[11,4])\ncplt.DA_RMS(results['df_performance'],'swc',ax=ax)\ncplt.DA_RMS(results['df_performance'],'swc1',ax=ax)\ncplt.DA_RMS(results['df_performance'],'swc2',ax=ax)\n")

