"""
Weil et al example
==================

Weill, S., et al. « Coupling Water Flow and Solute Transport into a Physically-Based Surface–Subsurface Hydrological Model ». 
Advances in Water Resources, vol. 34, no 1, janvier 2011, p. 128‑36. DOI.org (Crossref), 
https://doi.org/10.1016/j.advwatres.2010.10.001.

The CATHY gitbucket repository provides the Weill et al. dataset example to test the installation. On top of that, we provide a computational notebook code to reproduce the results using the **pyCATHY wrapper** (https://github.com/BenjMy/pycathy_wrapper). 

The notebook illustrate how to work interactively: execute single cell, see partial results at different processing steps (preprocessing, processing, output)... You can share it to work collaboratively on it by sharing the link and execute it from another PC without any installation required.


*Estimated time to run the notebook = 5min*

"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pyCATHY.meshtools as mt
from pyCATHY import cathy_tools
from pyCATHY.importers import cathy_inputs as in_CT
from pyCATHY.importers import cathy_outputs as out_CT
from pyCATHY.plotters import cathy_plots as cplt

import pyCATHY.petro as petro
import utils

import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# import utils_Bousval
import pyvista as pv

#%% Init CATHY model
# ------------------------
path2prj = "./solution_ET/"  # add your local path here
simu = cathy_tools.CATHY(dirName=path2prj, 
                         prj_name="ZROOT_spatially_from_weill"
                         )
figpath = path2prj
dimveg = 2

#%% spatially variable atmospheric boundary condition inputs

DEM, dem_header = simu.read_inputs('dem')

DEM_new = np.ones(np.shape(DEM)) #np.mean(DEM)*
DEM_new[-1,-1] = DEM_new[-1,-1] - 1e-3

# from scipy.ndimage import gaussian_filter
# Apply Gaussian blur to reduce slope
# sigma = 10.0  # Adjust the sigma value as needed for desired smoothing
# smoothed_dem = gaussian_filter(DEM, sigma=sigma)

simu.update_prepo_inputs(DEM_new)
simu.run_preprocessor()

interval = 5
ncycles = 7
t_atmbc = np.linspace(1e-3,36e3*ncycles,interval*ncycles)
v_atmbc = np.array([-2e-7]*len(t_atmbc))

for i in range(ncycles):
    if i % 2 != 0:
        v_atmbc[i*interval+1:(i+1)*interval] = 0

# v_atmbc[31:40] = 0
# v_atmbc[51:60] = 0

# simu.run_processor(IPRT=3)
simu.create_mesh_vtk()
#%%
simu.update_atmbc(
                    HSPATM=1,
                    IETO=0,
                    time=t_atmbc,
                    netValue=v_atmbc
                  )

fig, ax= plt.subplots()
simu.show_input('atmbc',ax=ax)
fig.savefig(os.path.join(simu.workdir + "/ZROOT_spatially_from_weill/", 'atmbc.png'), 
            dpi=300)

#%%
FP_map = {
         'PCANA': [0.0],
         'PCREF': [-4.0],
         'PCWLT': [-200],
         'ZROOT': [0.25],
         'PZ': [1.0],
         'OMGC': [1.0]
         }
dim2veg_dict = {key: val * dimveg for key, val in FP_map.items()}

if dimveg>2:
    dim2veg_dict['ZROOT'] = list(np.linspace(0.25,1.25,9))
else:
    dim2veg_dict['ZROOT'] = [0.25,1.25]
    

#%%
indice_veg = np.ones((np.shape(simu.DEM)), dtype=int)

if dimveg>2:
    indice_veg = utils.gradient_root(indice_veg)
else:
    indice_veg = utils.root_updownhill_zones(indice_veg)
    
simu.update_veg_map(indice_veg)

fig, ax = plt.subplots()
simu.show_input('root_map',ax=ax)
fig.savefig(os.path.join(simu.workdir,
                         simu.project_name,
                         simu.project_name+'root_map.png'
                         )
            , dpi=300)

#%%
try:
    SPP_map, _ = simu.read_inputs('soil',MAXVEG=1)
except:
    SPP_map, _ = simu.read_inputs('soil',MAXVEG=2)

simu.update_soil(
                PMIN=-1e35,
                FP_map=dim2veg_dict,
                SPP_map=SPP_map,
                )

#%%

simu.update_ic(INDP=4,
                IPOND=0,
                WTPOSITION=2 #.015
                )
#%%
simu.update_parm(
                          TIMPRTi=list(t_atmbc),
                      )

#%%
simu.run_processor(
                    DTMIN=1e-2,
                    DELTAT=1,
                    DTMAX=1e3,
                    IPRT1=2,
                    TRAFLAG=0,
                    VTKF=2,
                    verbose=True
                   )

#%%

dtcoupling = simu.read_outputs('dtcoupling')

fig, ax = plt.subplots()
dtcoupling.plot(y='Atmpot-vf', ax=ax, color='k')
dtcoupling.plot(y='Atmact-vf', ax=ax, color='k', linestyle='--')
# ax.set_ylim(-1e-9,-5e-9)
ax.set_xlabel('Time (s)')
ax.set_ylabel('ET (m)')
plt.tight_layout()
fig.savefig(os.path.join(simu.workdir,
                         simu.project_name,
                         simu.project_name+'dtcoupling.png'
                         )
            , dpi=300)

# ETact_ref = dtcoupling

#%%

fig, axs = plt.subplots(4,4,
                        figsize=(8,5),
                        sharey=True,
                        sharex=True,
                        )
axs = axs.ravel()
for i in range(4*4):
    cmap = simu.show('spatialET',
                ax=axs[i],   
                ti=i+2, 
                scatter=True, 
                vmin=0,
                vmax=1e-9,
                colorbar=False
            )
    axs[i].set_ylabel('')
    axs[i].set_xlabel('')
    axs[i].set_title(f'#{i}')
    
cbar = fig.colorbar(cmap, ax=axs, 
                    orientation='vertical',
                    fraction=0.02, 
                    pad=0.04
                    )
cbar.set_label('ET (m/s)')
    
fig.savefig(os.path.join(simu.workdir,
                         simu.project_name,
                         simu.project_name+'spatialET.png'
                         )
            , dpi=300,
            bbox_inches='tight')

#%%

cplt.show_vtk(
    unit="pressure",
    timeStep=30,
    notebook=False,
    path=simu.workdir + "/ZROOT_spatially_from_weill/vtk/",
    savefig=True,
)


#%% Read outputs 
sw, sw_times = simu.read_outputs('sw')
df_psi = simu.read_outputs('psi')
# df_psi.index
#%% Choose 2 points uphill and downhill

depths = [0.05,0.15,0.25,0.75]

nodeIds,closest_positions = utils.get_mesh_node(simu,
                                        node_pos = [5,2.5,2],
                                        depths = depths
                                         )

nodeIds2,closest_positions2 = utils.get_mesh_node(simu,
                                        node_pos = [5,7.5,2] ,
                                        depths = depths
                                         )

#%%

pl = pv.Plotter(notebook=True)
mesh = pv.read(os.path.join(simu.workdir,
                                simu.project_name,
                                'vtk/121.vtk',
                                )
        )
pl.add_mesh(mesh,
            opacity=0.1
            )
pl.add_points(closest_positions,
              color='red'
              )
pl.add_points(closest_positions2,
              color='red'
              )
pl.show_grid()
pl.show()


#%%


# Generate a colormap
colors = plt.cm.tab10(np.linspace(0, 1, len(nodeIds)))

fig, ax = plt.subplots()

# Plot the first set of data with '.' markers
for i, nn in enumerate(nodeIds):
    ax.plot(sw.index / 86400, sw.iloc[:, nn], label=depths[i], marker='.', color=colors[i])

# Plot the second set of data with 'v' markers
for i, nn in enumerate(nodeIds2):
    ax.plot(sw.index / 86400, sw.iloc[:, nn], label=depths[i], marker='v', color=colors[i])

ax.set_xlabel('Time (Days)')
ax.set_ylabel('Saturation (-)')

# Create a unique legend
handles, labels = ax.get_legend_handles_labels()
unique_labels = {label: handle for handle, label in zip(handles, labels)}
ax.legend(unique_labels.values(), unique_labels.keys())

plt.show()
fig.savefig(os.path.join(simu.workdir,
                         simu.project_name,
                         simu.project_name+'sw.png'
                         )
            , dpi=300)


#%%

# Generate a colormap
colors = plt.cm.tab10(np.linspace(0, 1, len(nodeIds)))

fig, ax = plt.subplots()

# Plot the first set of data with '.' markers
for i, nn in enumerate(nodeIds):
    ax.plot(df_psi.index / 86400, df_psi.iloc[:, nn], label=depths[i], marker='.', color=colors[i])

# Plot the second set of data with 'v' markers
for i, nn in enumerate(nodeIds2):
    ax.plot(df_psi.index / 86400, df_psi.iloc[:, nn], label=depths[i], marker='v', color=colors[i])

ax.set_xlabel('Time (Days)')
ax.set_ylabel('Pressure head (m)')

# Create a unique legend
handles, labels = ax.get_legend_handles_labels()
unique_labels = {label: handle for handle, label in zip(handles, labels)}
ax.legend(unique_labels.values(), unique_labels.keys())

plt.show()
plt.title('Zone 1 (.) / Zone 2 (v)')

fig.savefig(os.path.join(simu.workdir,
                         simu.project_name,
                         simu.project_name+'psi.png'
                         )
            , dpi=300)



