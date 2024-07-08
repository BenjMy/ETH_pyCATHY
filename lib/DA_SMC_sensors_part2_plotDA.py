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

# name_scenario = 'ic_test'
# name_scenario = 'ic'
# name_scenario = 'ZROOT'
name_scenario = 'ZROOT_WITH_UPDATE'
# name_scenario = 'ic_ZROOT_NOupdate'
# name_scenario = 'ic_ZROOT_upd_ZROOT'


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


# In[3]:


pl = pv.Plotter(notebook=True)
cplt.show_vtk(unit="pressure", 
              timeStep=5, 
              path=simu_solution.workdir + "/weill_dataset/vtk/",
              ax=pl,
              )
pl.show()

#%%

simu_solution.show_input('atmbc') 

# Assuming simu_solution is already defined
df_atmbc = simu_solution.read_inputs('atmbc')

# Convert time from seconds to hours
df_atmbc['time'] = df_atmbc['time'] / 3600

# Plot the data
ax = df_atmbc.set_index('time').plot.bar()

# Set integer ticks on x-axis
plt.xticks(range(len(df_atmbc)), [int(time) for time in df_atmbc['time']])

# Color negative values red
for container in ax.containers:
    for bar in container:
        if bar.get_height() < 0:
            bar.set_color('darkred')
        else:
            bar.set_color('darkblue')


# Show the plot
plt.xlabel('Time (hours)')
plt.ylabel('Value')
plt.title('Atmospheric Boundary Conditions')
plt.savefig(simu_solution.workdir + "/weill_dataset/atmbc.png",
            dpi=300,
            )

# ## Reload DA results

# In[4]:


simuWithDA = DA(
                        dirName='./DA_SMC',
                        prj_name= 'SMC_withDA_' + name_scenario,
                        notebook=True,
                    )
simuWithoutDA = DA(
                        dirName='./DA_SMC',
                        prj_name= 'SMC_withoutDA_' + name_scenario,
                        notebook=True,
                    )


# In[5]:


results_withDA = simuWithDA.load_pickle_backup()
results_withoutDA = simuWithoutDA.load_pickle_backup()


# In[6]:


observations = dictObs_2pd(results_withDA['dict_obs'])


# In[7]:


fig, ax = plt.subplots()
observations.xs('swc',level=0).plot(y='data',ax=ax,label='swc')
observations.xs('swc1',level=0).plot(y='data',ax=ax,label='swc1')
observations.xs('swc2',level=0).plot(y='data',ax=ax)
#observations.xs('swc3',level=0).plot(y='data',ax=ax)


# In[8]:


assimilation_times = observations.index.get_level_values(1).unique().to_list()


# In[9]:


nodes_of_interest = observations.mesh_nodes
nodes_of_interest = list(np.hstack(np.unique(nodes_of_interest)))
nodes_of_interest


# In[10]:


#nodes_of_interest_pos
#sensors_pos = infiltration_util.get_sensors_pos()


# (plot_states)=
# ## Plot model saturation with uncertainties 

# In[11]:


parm_df = pd.DataFrame.from_dict(results_withDA['dict_parm_pert'],
                                 orient='index').T


# In[12]:

try:
    fig, ax = plt.subplots()
    ax.hist(parm_df['ic']['ini_perturbation'])
except:
    pass

# In[13]:


#obs2plot = observations.xs('swc',level=0)
obs2plot = observations
obs2plot['saturation'] = obs2plot['data']/POROS_SOL


# In[14]:


unique_times = results_withDA['df_DA']["time"].unique()
results_withDA['df_DA']["assimilation_times"] = results_withDA['df_DA']["time"].map(dict(zip(unique_times, 
                                                                               assimilation_times))
                                                                     )


# In[15]:


fig, axs = plt.subplots(1,3,figsize=(15,4),
                       sharey=True
                       )
for axi, NOI in zip(axs, nodes_of_interest):
    cplt.DA_plot_time_dynamic(results_withDA['df_DA'],
                              'sw',
                              NOI,
                              ax=axi,
                              keytime='assimilation_times',
                              )  
    axi.plot(sw_SOL_times,
            sw_SOL.iloc[:,NOI],
            color='r',
            marker= '.'
            )
    
    
observations.xs('swc',level=0).plot(y='saturation',ax=axs[0],label='swc')
observations.xs('swc1',level=0).plot(y='saturation',ax=axs[1],label='swc1')
observations.xs('swc2',level=0).plot(y='saturation',ax=axs[2])



# In[16]:


unique_times = results_withoutDA['df_DA']["time"].unique()
results_withoutDA['df_DA']["assimilation_times"] = results_withoutDA['df_DA']["time"].map(dict(zip(unique_times, 
                                                                               assimilation_times))
                                                                     )


# In[17]:

for i, SI in enumerate([simuWithoutDA,simuWithDA]):
    
    if i==0:
        df_DA = results_withoutDA['df_DA']
    else:
        df_DA = results_withDA['df_DA']

        
    fig, axs = plt.subplots(1,3,
                            figsize=(8,3),
                           sharey=True
                           )
    for axi, NOI in zip(axs, nodes_of_interest):
        cplt.DA_plot_time_dynamic(df_DA,
                                  'sw',
                                  NOI,
                                  ax=axi,
                                  keytime='assimilation_times',
                                  )  
        axi.plot(sw_SOL_times,
                sw_SOL.iloc[:,NOI],
                color='r',
                marker= '.'
                )
        axi.legend().set_visible(False)
        
    # observations.xs('swc',level=0).plot(y='saturation',ax=axs[0],label='swc')
    # observations.xs('swc1',level=0).plot(y='saturation',ax=axs[1],label='swc1')
    # observations.xs('swc2',level=0).plot(y='saturation',ax=axs[2])
    
    fig.savefig(os.path.join(SI.workdir,
                             SI.project_name,
                             SI.project_name + '_sw.png'),
                dpi=300,
                bbox_inches = 'tight'
                )



#%%
for i, SI in enumerate([simuWithoutDA,simuWithDA]):
    
    if i==0:
        df_DA = results_withoutDA['df_DA']
    else:
        df_DA = results_withDA['df_DA']

        
    fig, axs = plt.subplots(1,3,
                            figsize=(8,3),
                           sharey=True
                           )
    for axi, NOI in zip(axs, nodes_of_interest):
        cplt.DA_plot_time_dynamic(df_DA,
                                  'psi',
                                  NOI,
                                  ax=axi,
                                  keytime='assimilation_times',
                                  )  
        axi.plot(sw_SOL_times,
                psi_SOL.iloc[:,NOI],
                color='r',
                marker= '.'
                )
        axi.legend().set_visible(False)
        
    # observations.xs('swc',level=0).plot(y='saturation',ax=axs[0],label='swc')
    # observations.xs('swc1',level=0).plot(y='saturation',ax=axs[1],label='swc1')
    # observations.xs('swc2',level=0).plot(y='saturation',ax=axs[2])
    
    fig.savefig(os.path.join(SI.workdir,
                             SI.project_name,
                             SI.project_name + '_psi.png'),
                dpi=300,
                bbox_inches = 'tight'
                )



# In[18]:


fig, ax = plt.subplots(2,figsize=[11,4])
cplt.DA_RMS(results_withDA['df_performance'],'swc',ax=ax)
cplt.DA_RMS(results_withDA['df_performance'],'swc1',ax=ax)
cplt.DA_RMS(results_withDA['df_performance'],'swc2',ax=ax)


# (Parm_evol)=
# ## Plot parameters convergence

# In[19]:


fig, axs = plt.subplots(2,1,figsize=[11,4],
                        # sharex=True
                        )
cplt.DA_plot_parm_dynamic_scatter(parm = 'ZROOT0', 
                                  dict_parm_pert=results_withDA['dict_parm_pert'], 
                                  list_assimilation_times = assimilation_times,
                                  ax=axs[0],
                                          )   
# axs[0].plot(np.linspace(1,len(assimilation_times), len(assimilation_times)),
#             [POROS_SOL]*len(assimilation_times))


# In[ ]:





# In[ ]:





# In[ ]:




