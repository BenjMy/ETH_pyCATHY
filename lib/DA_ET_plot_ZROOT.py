"""
Data assimilation of actual ET observation into hydrological model
Reference model is based on weill et al. with spatially heterogeneous ETp
python file to run weiletal_ET_ref_atmbc_spatially_variable
"""

#%% Import libraries 
import os
import numpy as np

from pyCATHY import CATHY
from pyCATHY.plotters import cathy_plots as pltCT
from pyCATHY.importers import cathy_outputs as out_CT

import matplotlib.pyplot as plt
import pandas as pd
from pyCATHY.plotters import cathy_plots as cplt

import matplotlib.dates as mdates
from pyCATHY.cathy_utils import change_x2date, MPa2m, kPa2m
from pyCATHY.DA.cathy_DA import dictObs_2pd
import pyCATHY.meshtools as mt

import utils
import argparse


def get_cmd():
    parser = argparse.ArgumentParser(description='plot_results')
    parser.add_argument('-idsimu','--idsimu', 
                        type=int, 
                        help='study id',
                        required=False,
                        default=2
                        )  #default='ZROOTdim2') #default='ET_scenarii')
    args = parser.parse_args()

    return(args)

#%% Build project name
args = get_cmd()
#%%
plt.close('all')
idsimu=args.idsimu
date_string = '2023-01-01 08:00:00.00000'
start_date = pd.to_datetime(date_string)
#%%
results_df = pd.read_csv('DA_ET_log.csv',index_col=0)
simu2plot = results_df.loc[idsimu]
prj_name_ref = simu2plot['refModel']
prj_name_DA = 'DA_ET_' + str(idsimu)

path2prj_ref = "./solution_ET/"  # add your local path here
simu_ref = CATHY(dirName=path2prj_ref, 
                 prj_name=prj_name_ref
                 )

path2prj = "./DA_ET/"  # add your local path here
simu = CATHY(dirName=path2prj, 
             prj_name=prj_name_DA
             )
# fig_path = os.path.join(simu.workdir,'BousvalDA' + str(idsimu))

#%%

# ----------------------------------------------------------------------------
# Update CATHY inputs
# ----------------------------------------------------------------------------
simu.update_dem_parameters()
simu.update_prepo_inputs()
DEM, _ = simu.read_inputs('dem')
simu.update_dem(DEM)
simu.update_veg_map()
simu_ref.update_veg_map()
# utils_Bousval.get_ZROOTdim441_FeddesParm_dict(simu_ref,
#                                               simu, 
#                                               study='ZROOTdim441'
#                                               )


#%% Load simulation results
backupDA =  os.path.join(simu.workdir,prj_name_DA,prj_name_DA+ '_df.pkl')
results = simu.load_pickle_backup(backupDA)
dem, dem_hd= simu_ref.read_inputs('dem')
test = results['df_DA'].set_index(['time','Ensemble_nb'])

#%% Get point of interest (1 per zone)

(nodes2plots, 
 nodespos2plots, 
 nodes2plots_surf, 
 nodespos2plots_surf) = utils.get_NOIs(simu,
                                        depths = [0,0.5],
                                        maxdem=dem.max()
                                        )

#%%

import pyvista as pv
mesh = pv.read(os.path.join(
                    simu_ref.workdir,
                    simu_ref.project_name,
                    # 'DA_Ensemble/cathy_32/'
                    'vtk/100.vtk',                    
                    )
    )
plotter = pv.Plotter()
plotter.add_mesh(mesh, smooth_shading=True,
                  opacity=0.2,
                  show_edges=True,
                  color='white'
                  )
plotter.show_bounds(location='origin')


plotter.add_points(nodespos2plots,
                    render_points_as_spheres=True, 
                    point_size=20.0)

plotter.show()


#%%
veg_map, hd_veg_map = simu_ref.read_inputs('root_map')

fig, axs = plt.subplots(1,2,figsize=(6,3))

cmap = axs[0].imshow(dem)
axs[0].set_title('DEM')
plt.colorbar(cmap,ax=axs[0])

cmap= axs[1].imshow(veg_map)
plt.colorbar(cmap,ax=axs[1])
axs[1].set_title('Veg map')

markers = ['.','v']*2
colors = ['r','r','blue','blue']*2
labels = ['0.1m','1m']*2
for i, nn in enumerate(nodes2plots):
    if i<2:
        label=labels[i]
    else:
        label= ''
    axs[1].scatter(nodespos2plots[i,0]*2,nodespos2plots[i,1]*2,
                   label=label, 
                   marker=markers[i], 
                   color=colors[i]
                   )
    axs[0].scatter(nodespos2plots[i,0]*2,nodespos2plots[i,1]*2,
                   # label=nodes2plots[i], 
                   marker=markers[i], 
                   color=colors[i]
                   )    
plt.tight_layout()
plt.legend()
fig.savefig(os.path.join(simu.workdir,simu.project_name,'DEM_VegMap.png'),
            dpi=300,
            bbox_inches='tight')


#%% read psi and sw
sw, t = simu_ref.read_outputs('sw')
psi = simu_ref.read_outputs('psi')

sw_datetimes = change_x2date(t,start_date)

#%% Show solution spatial actual ET

mosaic = '''aaac
           bbbc'''
fig, axs = plt.subplot_mosaic(mosaic,
                              layout='constrained',
                              figsize=(10,5))
colors_surf = ['r', 'b',]
colors2 = ['r','r','b','b']
markers = ['.','v']*2


SPP, FP = simu_ref.read_inputs('soil',
                               MAXVEG=len(np.unique(veg_map))
                               )

simu_ref.show('spatialET',
            ax=axs['c'],   
            ti=i+1, 
            scatter=True, 
            # vmax=1e-9,
        )
axs['c'].invert_yaxis()

_, FP = simu.read_inputs('soil')

for i, nn in enumerate(nodes2plots_surf):
    axs['c'].scatter(nodespos2plots_surf[i][0],
                     nodespos2plots_surf[i][1],
                     label='ZROOT:' + str(FP['ZROOT'].iloc[1]),
                      color=colors_surf[i],
                      marker='v',
                   )
for i, nn in enumerate(nodes2plots):
    axs['a'].scatter(sw_datetimes,
                sw.iloc[:,nn],
                label=labels[i],
                marker=markers[i],
                color=colors2[i],
            )
axs['a'].set_ylabel('SW (-)')
axs['a'].legend()

for i, nn in enumerate(nodes2plots):
    axs['b'].scatter(sw_datetimes,
                psi.iloc[:,nn],
                label='ZROOT:' + str(FP['ZROOT'].iloc[0]),
                marker=markers[i],
                color=colors2[i],
            )
axs['b'].set_ylabel(r'$\Psi_{soil}$ (m)')
fig.savefig(os.path.join(simu.workdir,simu.project_name,
                         'states_dyn_scatter.png'),
            dpi=300,
            bbox_inches='tight')


#%%  reload observations 
#--------------------------------
NENS = len(results['df_DA']['Ensemble_nb'].unique())
observations = dictObs_2pd(results['dict_obs'])
startDate = observations['datetime'].min()

#%% reload atmbc
#--------------------------------
df_atmbc = simu.read_inputs('atmbc')
atmbc_times = df_atmbc.time.unique()
nnod = len(df_atmbc)/len(atmbc_times)
ntimes = len(df_atmbc.time.unique())
nodenb = np.tile(np.arange(0,nnod),ntimes)
df_atmbc['nodeNb'] = nodenb
df_atmbc.set_index(['time','nodeNb'], inplace=True)


#%% Plot observations
#--------------------------------

fig, ax = plt.subplots(
                        figsize=(6,3),
                        sharex=True,
                       )


for i in range(int(len(nodes2plots)/2)):
    observations.xs('ETact' + str(nodes2plots_surf[i]), level=0).plot(y='data',
                                                                    ax=ax,
                                                                    marker='.',
                                                                    # label='node' + str(nodes2plots_surf[i]),
                                                                    label='',
                                                                    c=colors_surf[i]
                                                                )

df_atmbc_abs = df_atmbc*-1
df_atmbc.xs(0,level=1).plot(ax=ax,
                            c='k',
                            legend=None,
                            )

# axs[1].set_ylabel('ETp [m/s]')
ax.set_ylabel('ETact [m/s]')
# axs[1].set_xlabel('Time [s]')
plt.tight_layout()
plt.legend(['ETa_{up}','ETa_{downhill}','ETp'])
fig.savefig(os.path.join(simu.workdir,simu.project_name,
                         'ETact_dynamic.png'),
            dpi=300,
            bbox_inches='tight'
            )

#%%get assimilation times 
assimilationTimes = observations.xs('ETact', level=0).index.to_list()
obsAssimilated = observations.index.get_level_values(0).unique().values

#%% Plot performance
# -------------------------
fig, ax = plt.subplots(2,figsize=[11,4])
pltCT.DA_RMS(results['df_performance'],'ETact',ax=ax)

#%% Plot state dynamic
# -------------------------
# read psi and sw 
obs2plot_selec = observations.xs('ETact')[['data','data_err']]
psi = simu_ref.read_outputs('psi')
sw, sw_times = simu_ref.read_outputs('sw')
tass = results['df_DA'].time.unique()

#%% Plot states_dyn sw 
fig, axs = plt.subplots(2,2, 
                        sharex=True,sharey=True,
                        # figsize=(5,7)
                        )
# simu.read_inputs('soil',MAXVEG=2)
# simu.update_dem_parameters()
# simu.update_veg_map()
# simu.update_soil()

#%%

l_tda = len(results['df_DA'].time.unique())
nens = len(results['df_DA'].Ensemble_nb.unique())
grid3d = simu_ref.read_outputs('grid3d')
grid3d_nnod3 = grid3d['nnod3']

# grid3d_nnod3*l_tda*nens - len(results['df_DA'])
# 15352960 - 15353856

#%%
ZROOTid = [0,0,1,1]
sw_datetimes = change_x2date(sw_times,start_date)
axs= axs.ravel()

for i, nn in enumerate(nodes2plots):
    
    axs[i].plot(sw_datetimes[1:],
                sw.iloc[1:,nn],
                'r',
                label='ref',
                linestyle='--',
                marker='.'
                )
    pltCT.DA_plot_time_dynamic(results['df_DA'],
                                'sw',
                                nn,
                                savefig=False,
                                ax=axs[i],
                                start_date=start_date,
                                atmbc_times=atmbc_times
                                ) 
    axs[i].legend().remove()
    axs[i].set_ylabel('sw (-)')

    dd = simu.DEM.max() - np.round(simu.grid3d['mesh3d_nodes'][nn][0][2]*10)/10
    zr= str(FP['ZROOT'].iloc[ZROOTid[i]])

    axs[i].set_title('$Z_{root}$:' + str(zr) + 'm '
                     'SMC:' + str(dd) + 'm'
                     )


    if i == len(axs):
        axs[i].set_xlabel('Assimiliation time')
    # axs
    plt.legend(['solution','pred. ensemble'])
    plt.tight_layout()
fig.savefig(os.path.join(simu.workdir,simu.project_name,'states_dyn.png'),dpi=300)

#%% Plot states_dyn psi  
fig, axs = plt.subplots(2,2, 
                        sharex=True,sharey=True,
                        # figsize=(5,7)
                        )
# simu.read_inputs('soil',MAXVEG=2)
# simu.update_dem_parameters()
# simu.update_veg_map()
# simu.update_soil()


ZROOTid = [0,0,1,1]
sw_datetimes = change_x2date(sw_times,start_date)
axs= axs.ravel()
for i, nn in enumerate(nodes2plots):
    axs[i].plot(sw_datetimes[1:],psi.iloc[1:,nn],'r',label='ref',linestyle='--',marker='.')
    pltCT.DA_plot_time_dynamic(results['df_DA'],
                                'psi',
                                nn,
                                savefig=False,
                                ax=axs[i],
                                start_date=start_date,
                                atmbc_times=atmbc_times
                                )  
    axs[i].legend().remove()
    axs[i].set_ylabel(r'$\psi_{soil}$ (m)')
    
    dd = simu.DEM.max() - np.round(simu.grid3d['mesh3d_nodes'][nn][0][2]*10)/10
    zr= str(FP['ZROOT'].iloc[ZROOTid[i]])

    axs[i].set_title('$Z_{root}$:' + str(zr) + 'm '
                     'SMC:' + str(dd) + 'm'
                     )

    if i == len(axs):
        axs[i].set_xlabel('Assimiliation time')
    # axs
    plt.legend(['solution','pred. ensemble'])
    plt.tight_layout()
fig.savefig(os.path.join(simu.workdir,simu.project_name,'states_dyn_psi.png'),dpi=300)
    
    
#%% Plot parameters dynamic
# -------------------------
veg_map, veg_map_hd = simu_ref.read_inputs('root_map')
df_SPP, df_FP = simu_ref.read_inputs('soil',MAXVEG=len(np.unique(veg_map)))
results['dict_parm_pert'].keys()
    
for kk in results['dict_parm_pert'].keys():
    try:
        fig, ax = plt.subplots(figsize=[11,4])
        pltCT.DA_plot_parm_dynamic_scatter(parm = kk, 
                                            dict_parm_pert=results['dict_parm_pert'], 
                                            list_assimilation_times = assimilationTimes,
                                            ax=ax,
                                                  )
        zone_nb = int(''.join(filter(str.isdigit, kk)))
        nb_ass_times= len(results['dict_parm_pert'][kk]['ini_perturbation'])
        if 'ZROOT' in kk:
            ax.plot(np.arange(0,nb_ass_times,1),[df_FP['ZROOT'].iloc[zone_nb]]*nb_ass_times,
                    color='red',linestyle='--')
        elif 'PCREF' in kk:
            ax.plot(np.arange(0,nb_ass_times,1),[df_FP['PCREF'].iloc[zone_nb]]*nb_ass_times,
                    color='red',linestyle='--')
        fig.savefig( os.path.join(simu.workdir,simu.project_name,kk + '.png'),dpi=300)
    except:
        pass
