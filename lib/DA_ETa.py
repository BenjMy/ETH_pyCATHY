"""
Weil et al example
==================

Weill, S., et al. « Coupling Water Flow and Solute Transport into a Physically-Based Surface–Subsurface Hydrological Model ». 
Advances in Water Resources, vol. 34, no 1, janvier 2011, p. 128‑36. DOI.org (Crossref), 
https://doi.org/10.1016/j.advwatres.2010.10.001.

The CATHY gitbucket repository provides the Weill et al. dataset example to test the installation. On top of that, we provide a computational notebook code to reproduce the results using the **pyCATHY wrapper** (https://github.com/BenjMy/pycathy_wrapper). 
The notebook illustrate how to work interactively: execute single cell, see partial results at different processing steps (preprocessing, processing, output)... You can share it to work collaboratively on it by sharing the link and execute it from another PC without any installation required.

Notebook objectives:
    - Use the Weill et al example to show the application of DA for the assimilation of actual ET (synthetic) into a spatially heteregeneous atmbc domain.
    Atmbc are limited to evapotranspiration (no rain). 
    The atmbc parameters are perturbated using Evenson's theory with respect to a given scenario parameters (parameters 
    for the mean and standart deviation of the normal distribution, and time correlation length). Each mesh node is perturbated individually.
    Observations (the actual ET infered from the reference model) are given on each nodes for 5 times. 

*Estimated time to run the notebook = 25min*

"""

#%% Import libraries 

import os
import matplotlib.pyplot as plt
from pyCATHY.DA.cathy_DA import DA
from pyCATHY.DA import perturbate
from pyCATHY.DA.cathy_DA import dictObs_2pd
from pyCATHY.DA.observations import read_observations, prepare_observations, make_data_cov
from pyCATHY.plotters import cathy_plots as cplt
import pyCATHY.meshtools as mt
from pyCATHY.importers import cathy_outputs as out_CT

from pyCATHY.cathy_utils import change_x2date, MPa2m, kPa2m
from pyCATHY import cathy_tools

import numpy as np
import pandas as pd
import matplotlib as mpl
# set some default plotting parameters for nicer looking plots
mpl.rcParams.update({"axes.grid":True, "grid.color":"gray", "grid.linestyle":'--','figure.figsize':(10,10)})
import utils

from scenarii_weilletal_DA_ET import load_scenario 
import argparse
import matplotlib.dates as mdates

import shutil
import glob

#%%

def get_cmd():
    parser = argparse.ArgumentParser(description='ProcessDA')
    # parser.add_argument('sc', type=int, help='scenario nb')
    #  #Feddes_irr, ref_scenario_irr_het_soil_f5, hetsoil_irr, Archie_Pert_scenarii, freq
    # -------------------------------------------------------------------------------------------------------
    parser.add_argument('-study','--study', type=str, 
                        help='study selection', 
                        required=False, 
                        # default='ETdim2'
                        default='ZROOTdim2'
                        )  
    parser.add_argument('-sc','--sc', type=int,
                        help='scenario nb', 
                        required=False, 
                        default=0
                        )
    parser.add_argument('-nens','--nens', type=int, 
                        help='nb of ensemble', 
                        required=False,
                        default=3
                        )
    parser.add_argument('-DAtype',type=str, 
                        help='type of DA',
                        default='enkf_Evensen2009'
                        # default='enkf_Evensen2009_Sakov'
                        )
    parser.add_argument('-DAloc',type=int, 
                        help='DA localisation',
                        # default='enkf_Evensen2009'
                        default=1
                        )
    parser.add_argument('-damping',type=float, 
                        help='damping factor',
                        default=1
                        )
    parser.add_argument('-dataErr',type=float, 
                        help='error data',
                        # default=5e-2
                        default=5
                        )
    parser.add_argument('-parallel',type=int, 
                        help='parallel computing',
                        default=1
                        )
    parser.add_argument('-refModel',type=str, 
                        help='name of the ref model',
                        # default='atmbc_spatially_from_weill'
                        default='ZROOT_spatially_from_weill'                       
                        # default='atmbc_spatially_WT0.5_from_weill'                       
                        ) #ZROOT_spatially_from_weill
    args = parser.parse_args()

    return(args)


#%%
# ----------------------------------------------------------------------------
#  Build project name
# ----------------------------------------------------------------------------

args = get_cmd()
results_df, matching_index = utils.backup_simulog_DA(args,
                                                    filename='DA_ET_log.csv'
                                                    )
# prj_name = utils_Bousval.build_prj_name_DA(vars(args))
prj_name_DA = 'DA_ET_' + str(matching_index)

#%%
expected_value = 1e-7  #expected mean value of ETa
percentage_error = args.dataErr 
absolute_error = expected_value * (percentage_error / 100)


#%%
# ----------------------------------------------------------------------------
# Read DA scenario
# ----------------------------------------------------------------------------

scenarii = load_scenario(study=args.study)
scenario =  scenarii[list(scenarii)[args.sc]]

#%% 
# ----------------------------------------------------------------------------
# Init CATHY model: copy reference model input files into DA folder
# ----------------------------------------------------------------------------

path2prjSol = "./solution_ET/"  # add your local path here
path2prj = "./DA_ET/"  # add your local path here

simu_ref = cathy_tools.CATHY(dirName=path2prjSol, 
                             prj_name=args.refModel
                             )
src_path = os.path.join(simu_ref.workdir,simu_ref.project_name)
dst_path = os.path.join(path2prj,prj_name_DA)
shutil.copytree(src_path, dst_path,  dirs_exist_ok=True)
vtk_files = glob.glob(os.path.join(dst_path, '**', '*.vtk'), recursive=True)
for vtk_file in vtk_files:
    try:
        os.remove(vtk_file)
    except: 
        pass

simu = DA(dirName=path2prj, 
          prj_name=prj_name_DA
          )

#%% 
# ----------------------------------------------------------------------------
# Update CATHY inputs
# ----------------------------------------------------------------------------
# simu.update_dem_parameters()
# simu.update_prepo_inputs()
DEM, _ = simu.read_inputs('dem')
# simu.update_dem(DEM)
# simu.update_veg_map()
# simu_ref.update_veg_map()
# utils_Bousval.get_ZROOTdim441_FeddesParm_dict(simu_ref,simu, 
#                                               study=args.study
#                                               )



#%% Plot actual ET
# --------------------
dtcoupling = simu_ref.read_outputs('dtcoupling')
ET_act_obs = dtcoupling[['Time','Atmact-vf']]

df_fort777 = out_CT.read_fort777(os.path.join(simu_ref.workdir,
                                              simu_ref.project_name,
                                              'fort.777'),
                                  )

df_fort777 = df_fort777.set_index('time_sec')
ET_nn = df_fort777.set_index('SURFACE NODE').index[0]
df_fort777.set_index('SURFACE NODE').loc[ET_nn]['ACT. ETRA']

#%% actual ET 1D
dtcoupling = simu_ref.read_outputs('dtcoupling')

fig, ax = plt.subplots()

# dtcoupling.columns
# dtcoupling.plot(y='Atmpot-vf', ax=ax, color='k')
dtcoupling.plot(y='Atmact-vf', ax=ax, color='k', linestyle='--')
# ax.set_ylim(-1e-9,-5e-9)
ax.set_xlabel('Time (s)')
ax.set_ylabel('ET (m)')
plt.tight_layout()

#%% Plot ATMBC
# --------------------
fig, ax = plt.subplots(figsize=(6,2))
simu_ref.show_input('atmbc',ax=ax)
plt.tight_layout()
fig.savefig(os.path.join(simu.workdir,simu.project_name,'atmbc.png'),
            dpi=350,
            bbox_inches='tight'
            )

#%% Plot spatial ATMBC
# --------------------
try:
    v_atmbc = simu_ref.atmbc['atmbc_df'].set_index('time').loc[simu_ref.atmbc['atmbc_df'].index[1]]
    v_atmbc_mat = np.reshape(v_atmbc,[21,21])
    fig, ax = plt.subplots(figsize=(4,4))
    img = ax.imshow(v_atmbc_mat)
    plt.colorbar(img)
    fig.savefig(os.path.join(simu.workdir,
                             simu.project_name,
                             'spatial_ETp.png')
                )
except:
    pass


#%%

#%% Initial conditions
# --------------------
df_psi = simu_ref.read_outputs('psi')

simu.run_preprocessor(verbose=True)
simu.run_processor(IPRT1=3)

simu.update_ic(INDP=1,
                pressure_head_ini=list(df_psi.iloc[0])
                )

simu.read_inputs('atmbc')
print(simu.atmbc['time'])
simu.atmbc['VALUE']
df_fort777.index.unique()
df_fort777_new = list(df_fort777.index.unique()[1:])

#%% Read observations = actual ET
data_measure = {}

date_string = '2023-01-01 08:00:00.00000'
pd_datetime = pd.to_datetime(date_string)


for i, tobsi in enumerate(df_fort777_new):  
    if i==0:
        continue
    else:
        for node_obsi in range(int(df_fort777['SURFACE NODE'].max())-1):
            data_measure = read_observations(
                                            data_measure,
                                            df_fort777.loc[tobsi].iloc[node_obsi]['ACT. ETRA'], 
                                            data_type = 'ETact', 
                                            data_err = absolute_error, # instrumental error
                                            show=True,
                                            tA=simu.atmbc['time'][i],
                                            obs_cov_type='data_err', #data_err
                                            mesh_nodes=node_obsi, 
                                            datetime=pd_datetime,
                                            ) 

#%% Get Nodes of interest (1 in each zone)

(nodes2plots, 
 nodespos2plots, 
 nodes2plots_surf, 
 nodespos2plots_surf) = utils.get_NOIs(
                                      simu=simu_ref,
                                      depths = [0,1],
                                      maxdem=DEM.max()
                                    )

#%%
data_measure_df = dictObs_2pd(data_measure) 
data_measure_df.index
data_measure_df.iloc[0]

_, FP = simu.read_inputs('soil')
simu.update_soil(PMIN=-1e+35)
# simu.soil_SPP

fig, ax = plt.subplots(figsize=(6,2))
data_measure_df.xs(f'ETact{nodes2plots_surf[0]}', level=0).plot(y='data',
                                                              ax=ax,
                                                              marker='.',
                                                              color='r',
                                                              label='ZROOT:' + str(FP['ZROOT'].iloc[1])
                                                              )
data_measure_df.xs(f'ETact{nodes2plots_surf[1]}', level=0).plot(y='data',
                                                                ax=ax,marker='.',
                                                                color='b',
                                                                label='ZROOT:' + str(FP['ZROOT'].iloc[0])
                                                                )

ax.set_ylabel('ETobs (m/s)')
# data_measure_df.xs('ETact400', level=0)
fig.savefig(os.path.join(simu.workdir,simu.project_name,'atmbc_pernodes.png'),
            dpi=350,
            bbox_inches='tight'
            )

#%% Add covariance localisation



#%%

measurements = data_measure_df['data'].unstack(level='assimilation time').to_numpy()
# ground_truth = np.mean(measurements, axis=0)
# errors = measurements - ground_truth

# errors2d = np.array([errors]*440)
# errors2d_reshaped = errors2d.reshape((34,440,440))

errors2d = np.array([measurements]*440)
errors2d_reshaped = errors2d.reshape((34,440,440))


# stacked_data_cov = []
# for i in range(len(data_measure_df.index.get_level_values(1).unique())):
#     stacked_data_cov.append(np.cov(errors2d_reshaped[i]))


stacked_data_cov = []
for i in range(len(data_measure_df.index.get_level_values(1).unique())):
    matrix = np.zeros((440,440))
    np.fill_diagonal(matrix, args.dataErr)
    # if args.dataErr==1e+99:
        # np.fill_diagonal(matrix, 1e99)
    stacked_data_cov.append(matrix)


# fig, ax = plt.subplots()
# im1 = ax.imshow(stacked_data_cov[0], 
#                   cmap='viridis', 
#                   interpolation='none',
#                   )
# plt.colorbar(im1, ax=ax)

# fig.savefig(os.path.join(simu.workdir,
#                          simu.project_name,
#                          'stacked_data_cov.png'
#                          ),
#             dpi=350,
#             bbox_inches='tight'
#             )


# stacked_data_cov = [error_covariance_matrix]*len(data_measure_df.index.get_level_values(1).unique())
# simu.stacked_data_cov = stacked_data_cov
# df0 = data_measure_df['data'].xs('ETact',level=0)
# df1 = data_measure_df['data'].xs('ETact1',level=0)
# df400 = data_measure_df['data'].xs('ETact400',level=0)

# data_cov, data_pert, stacked_data_cov = make_data_cov(
#                                                         simu,
#                                                         data_measure,
#                                                         list_assimilated_obs='all',
#                                                         nb_assimilation_times=len(data_measure)
#                                                         )

#%%  define covariance matrice for each assimilation times

if args.DAloc==1:
    
    import matplotlib.patches as patches
    
    # Define size and sigma for the Gaussian covariance matrix
    size = stacked_data_cov[0].shape[0]
    sigma = 100.0
    cov_matrix = utils.gaussian_covariance_matrix(size, sigma)
    localized_cov_matrix, zone1,zone2 = utils.make_data_localisation(cov_matrix,
                                                         length_scale=1e3,
                                                         )
    # stacked_data_cov = [localized_cov_matrix*absolute_error]*len(stacked_data_cov)
    stacked_data_cov = [localized_cov_matrix*args.dataErr]*len(stacked_data_cov)
    fig, ax = plt.subplots()
    im1 = ax.imshow(stacked_data_cov[0], 
                      cmap='viridis', 
                      interpolation='none',
                      )
    plt.colorbar(im1, ax=ax)
    rect1 = patches.Rectangle((zone1[0] - 0.5, zone1[0] - 0.5), 
                              len(zone1), len(zone1), 
                              linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect1)
    rect2 = patches.Rectangle((zone2[0] - 0.5, zone2[0] - 0.5), 
                              len(zone2), len(zone2),
                              linewidth=2, edgecolor='blue', facecolor='none')
    ax.add_patch(rect2)
    
    fig.savefig(os.path.join(simu.workdir,
                             simu.project_name,
                             'localisation.png'
                             ),
                dpi=350,
                bbox_inches='tight'
                )
    simu.stacked_data_cov = stacked_data_cov
else:
    
    fig, ax = plt.subplots()
    im1 = ax.imshow(stacked_data_cov[0], 
                      cmap='viridis', 
                      interpolation='none',
                      )
    plt.colorbar(im1, ax=ax)
    fig.savefig(os.path.join(simu.workdir,
                             simu.project_name,
                             'No_localisation.png'
                             ),
                dpi=350,
                bbox_inches='tight'
                )
    simu.stacked_data_cov = stacked_data_cov


#%% perturbated variable 

list_pert = perturbate.perturbate(simu, 
                                  scenario, 
                                  args.nens
                                  )   

#%% Parameters perturbation
# stop
var_per_dict_stacked = {}
for dp in list_pert:
    
    if len(list_pert)>10:
        savefig = False
    else:
        savefig = os.path.join(
                                simu.workdir,
                                simu.project_name,
                                simu.project_name + dp['savefig']
                                )
    np.random.seed(1)
    # need to call perturbate_var as many times as variable to perturbate
    # return a dict merging all variable perturbate to parse into prepare_DA
    var_per_dict_stacked = perturbate.perturbate_parm(
                                                    var_per_dict_stacked,
                                                    parm=dp, 
                                                    type_parm = dp['type_parm'], # can also be VAN GENUCHTEN PARAMETERS
                                                    mean =  dp['mean'],
                                                    sd =  dp['sd'],
                                                    sampling_type =  dp['sampling_type'],
                                                    ensemble_size =  dp['ensemble_size'], # size of the ensemble
                                                    per_type= dp['per_type'],
                                                    savefig=savefig
                                                    )
    
#%%
try:
    utils.plot_atmbc_pert(simu,var_per_dict_stacked)
except:
    pass


#%% Run DA sequential

# DTMIN = 1e-2
# DELTAT = 1e-1

DTMIN = 1e-2
DELTAT = 10
DTMAX = 1e2

# stop
simu.run_DA_sequential(
                          VTKF=2,
                          TRAFLAG=0,
                          DTMIN=DTMIN,
                          DELTAT=DELTAT,
                          DTMAX=DTMAX,
                          parallel=True,    
                          dict_obs= data_measure,
                          list_assimilated_obs='all', # default
                          list_parm2update=scenarii[list(scenarii)[args.sc]]['listUpdateParm'],
                          DA_type=args.DAtype, #'pf_analysis', # default
                          dict_parm_pert=var_per_dict_stacked,
                          open_loop_run=False,
                          threshold_rejected=80,
                          damping=args.damping                    
                          )

# args.parallel
