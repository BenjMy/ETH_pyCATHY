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

import utils_Bousval
import argparse


def get_cmd():
    parser = argparse.ArgumentParser(description='plot_results')
    parser.add_argument('-idsimu','--idsimu', 
                        type=int, 
                        help='study id',
                        required=False,
                        default=13
                        )  #default='ZROOTdim2') #default='ET_scenarii')
    args = parser.parse_args()

    return(args)

#%% Build project name
args = get_cmd()
#%%
plt.close('all')
idsimu=args.idsimu
# nodes_interest = [132,140,300,308]
# nodes_interest = [132,140]

date_string = '2023-01-01 08:00:00.00000'
start_date = pd.to_datetime(date_string)


#%%
results_df = pd.read_csv('DA_ET_log.csv',index_col=0)
# simu2plot = results_df[results_df.index==idsimu]
simu2plot = results_df.loc[idsimu]
# prj_name = 'simuID' + str(idsimu) 
# prj_name_ref = 'ZROOT_spatially_from_weill'
prj_name_ref = simu2plot['refModel']
prj_name_DA = 'DA_ET_' + str(idsimu)

path2prj = "../pyCATHY/DA_ET_test/"  # add your local path here
simu_ref = CATHY(dirName=path2prj, 
                 prj_name=prj_name_ref
                 )
simu = CATHY(dirName=path2prj, prj_name=prj_name_DA)

fig_path = os.path.join(simu.workdir,'BousvalDA' + str(idsimu))

#%%

# results['df_performance']

# results['df_performance'].columns

# results['df_DA'].columns

#%%
# backup =  os.path.join(simuInfiltration.workdir,'SimuID' + str(idsimu), 'SimuID' + str(idsimu) + '.pkl')
# simupa = simuInfiltration.load_pickle_backup(backup)

backupDA =  os.path.join(simu.workdir,prj_name_DA,prj_name_DA+ '_df.pkl')
results = simu.load_pickle_backup(backupDA)

dem, dem_hd= simu_ref.read_inputs('dem')

# stop
# results.keys()
# test.columns
test = results['df_DA'].set_index(['time','Ensemble_nb'])
# test.xs(test.index.get_level_values(1)[0],level=1).mean('sw_bef_update_').plot()

# test.xs(test.index.get_level_values(0)[-1],level=0)
# stop
#%%

# import pyvista as pv
# mesh = pv.read(os.path.join(
#                     simu_ref.workdir,
#                     # simu_ref.project_name,
#                     # 'DA_Ensemble/cathy_32/'
#                     '100.vtk',                    
#                     )
#     )
# pts2plot = np.array(mesh.points[nodes_interest])#.to_numpy()

# plotter = pv.Plotter()
# plotter.add_mesh(mesh, smooth_shading=True,
#                  opacity=0.2,
#                  show_edges=True,
#                  color='white'
#                  )
# plotter.show_bounds(location='origin')


# plotter.add_points(pts2plot,
#                    render_points_as_spheres=True, 
#                    point_size=20.0)

# plotter.show()
alt =1
pts2plot = np.array([[1       , 9       , alt ],
                    [9.        , 1.        , alt ],
                    [9.        , 9.        , alt ],
                    ]
                    )
#%%
# stop
depths = [0,1] #,-0.1]
nodes2plots = []
nodespos2plots = []
for pp in pts2plot:
    nodes2plots_tmp = []
    nodespos2plots_tmp = []
    for dd in depths:
        print(pp[2]-dd)
        nodes2plots_tmp.append(simu.find_nearest_node([pp[0],
                                                        pp[1],
                                                        pp[2]-dd
                                                        ]
                                                        )[0]
                           )
        nodespos2plots_tmp.append(simu.find_nearest_node([pp[0],
                                pp[1],
                                pp[2]-dd
                                ]
                                )[1]
                           )
    nodes2plots.append(np.vstack(nodes2plots_tmp))
    nodespos2plots.append(np.vstack(nodespos2plots_tmp))

nodes2plots = np.vstack(nodes2plots)
nodespos2plots = np.vstack(nodespos2plots)

#%%
# nodes2plots_150cm = np.vstack(nodes2plots)+441*5
# nodes2plots_all = list(np.hstack(np.r_[np.vstack(nodes2plots),nodes2plots_150cm]))


# pts2plot = np.array(mesh.points[list(nodes2plots_all)])#.to_numpy()
# pts2plot = np.vstack(pts2plot)
# plotter = pv.Plotter()
# plotter.add_mesh(mesh, smooth_shading=True,
#                   opacity=0.2,
#                   show_edges=True,
#                   color='white'
#                   )
# plotter.show_bounds(location='origin')


# plotter.add_points(pts2plot,
#                     render_points_as_spheres=True, 
#                     point_size=20.0)

# plotter.show()

#%%

# dem = simu.read_inputs('dem')
dem, _ = simu_ref.read_inputs('dem')

# dem, _ = simu.read_inputs('dem')


grid3d = simu.read_outputs('grid3d')
grid3d['mesh3d_nodes'][0]

simu_ref.workdir
simu_ref.project_name

grid3d = simu_ref.read_outputs('grid3d')
grid3d['mesh3d_nodes'][0]
# grid3d['mesh3d_nodes'][200]
grid3d['mesh3d_nodes'][5292]


#%%
veg_map, hd_veg_map = simu_ref.read_inputs('root_map')

fig, axs = plt.subplots(1,2,figsize=(6,3))

cmap = axs[0].imshow(dem)
axs[0].set_title('DEM')
plt.colorbar(cmap,ax=axs[0])

cmap= axs[1].imshow(veg_map)
plt.colorbar(cmap,ax=axs[1])
axs[1].set_title('Veg map')

for i, nn in enumerate(nodes2plots):
    axs[0].scatter(nodespos2plots[i,0]*2,nodespos2plots[i,1]*2,label=nodes2plots[i])
    axs[1].scatter(nodespos2plots[i,0]*2,nodespos2plots[i,1]*2,label=nodes2plots[i])
    # axs[1].scatter(nodespos2plots[i,0]*2,nodespos2plots[i,1]*2,label=nodes2plots[i])
    
plt.tight_layout()
plt.legend()
fig.savefig(os.path.join(simu.workdir,simu.project_name,'DEM_VegMap.png'),
            dpi=300,
            bbox_inches='tight')

#%%

# fig, axs = plt.subplots(1,4,figsize=(8,5)) #,sharey=True)

# for i in range(4):
#     simu_ref.show('spatialET',
#                 ax=axs[i],   
#                 ti=i+2, 
#                 scatter=True, 
#                 # vmin=0,
#                 # vmax=1e-9,
#             )

#%%
# stop
sw, t = simu_ref.read_outputs('sw')
psi = simu_ref.read_outputs('psi')

sw_datetimes = change_x2date(t,start_date)

mosaic = '''aaac
           bbbc'''
fig, axs = plt.subplot_mosaic(mosaic,
                              layout='constrained',
                              figsize=(10,5))

# colors = ['b','orange','g','r']
# colors2 = ['b','g','orange','r']
colors_surf = ['b', 'g','orange']
colors2 = ['b','b','g','g','orange','orange']
markers = ['v','+','v','+','v','+']

# fig, axs = plt.subplots(3,1)
# simu_ref.update_veg_map()

SPP, FP = simu_ref.read_inputs('soil',
                               MAXVEG=len(np.unique(veg_map))
                               )


veg_map_ZROOT = np.copy(veg_map)  # Make a copy to preserve the original veg_map
for zone_index, zroot_value in zip(FP.index,FP['ZROOT']):
    veg_map_ZROOT[veg_map == (zone_index + 1)] = zroot_value
   
# veg_map_ZROOT = np.copy(veg_map)
# for FPi in FP.index:
#     print(FPi)
#     veg_map_ZROOT[veg_map_ZROOT==FPi+1]=FP['ZROOT'].iloc[FPi]
    
# cmap = axs['c'].imshow(veg_map_ZROOT)
# cbar = plt.colorbar(cmap,ax=axs['c'])
# cbar.set_label('ZROOT (m)')

simu_ref.show('spatialET',
            ax=axs['c'],   
            ti=i+1, 
            scatter=True, 
            # vmax=1e-9,
        )
axs['c'].invert_yaxis()

nodes2plots_surf = [nodes2plots[0][0], nodes2plots[2][0], nodes2plots[4][0]]
nodespos2plots_surf = [nodespos2plots[0][0:2],
                       nodespos2plots[2][0:2], 
                       nodespos2plots[4][0:2]
                       ]

for i, nn in enumerate(nodes2plots_surf):
    axs['c'].scatter(nodespos2plots_surf[i][0],
                     nodespos2plots_surf[i][1],
                     # label=str(nodes2plots_surf[i])+str(nodespos2plots_surf[0]),
                       color=colors_surf[i],
                      marker='v',
                   )
for i, nn in enumerate(nodes2plots):
    axs['a'].scatter(sw_datetimes,
                sw.iloc[:,nn],
                label=str(nodes2plots[i])+str(nodespos2plots[i,:]),
                marker=markers[i],
                color=colors2[i],
            )
axs['a'].set_ylabel('SW (-)')
# axs['a'].legend()

for i, nn in enumerate(nodes2plots):
    axs['b'].scatter(sw_datetimes,
                psi.iloc[:,nn],
                label=str(nodes2plots[i])+str(nodespos2plots[i,:]),
                marker=markers[i],
                color=colors2[i],
            )
axs['b'].set_ylabel('PSI (m)')
# axs[2].legend()
fig.savefig(os.path.join(simu.workdir,simu.project_name,
                         'states_dyn_scatter.png'),
            dpi=300,
            bbox_inches='tight')


#%%
results.keys()
results['df_DA'].columns
NENS = len(results['df_DA']['Ensemble_nb'].unique())
#%%
results['dict_obs']
observations = dictObs_2pd(results['dict_obs'])
startDate = observations['datetime'].min()
observations.xs('ETact')

#%%
df_atmbc = simu.read_inputs('atmbc')
atmbc_times = df_atmbc.time.unique()
nnod = len(df_atmbc)/len(atmbc_times)
ntimes = len(df_atmbc.time.unique())
nodenb = np.tile(np.arange(0,nnod),ntimes)
df_atmbc['nodeNb'] = nodenb
df_atmbc.set_index(['time','nodeNb'], inplace=True)

#%% Prepare statistics SW map
#------------------------------------------------------------------------------
# read all ET ensemble map for each assimilation time step
# df_SW_All_Ensemble = []

# # results['df_DA'].columns

# grid3d = simu.read_outputs('grid3d')
# nnod3 = grid3d['nnod3']

# nodes_array = np.tile(np.arange(0,nnod3),
#                         (
#                         len(results['df_DA'].time.unique())*
#                         len(results['df_DA'].Ensemble_nb.unique())
#                         )
#                         )
# results['df_DA']['node_id'] = nodes_array


# # results['df_DA'].groupby(['node_id','Ensemble_nb']).mean(dim=1).plot(x='time',
# #                                                                 y='psi_bef_update'
# #                                                                 )

# mean_df = results['df_DA'].groupby(['node_id','time']).mean()
# mean_df = mean_df.reset_index()
# # mean_df.reset_index().set_index('time')
# results['df_DA'].columns
# mean_df.plot(x='time',y='psi_bef_update')
# mean_df.plot(x='time',y='sw_bef_update_')

#%%

# fig, axs = plt.subplots(1,4,figsize=(8,5),sharey=True)

# tt = np.linspace(1,len(atmbc_times)-1,4)
# tt = [int(tti) for tti in tt]
# for i, tti in enumerate(tt):
#     cmap = simu.show('WT',
#                 ax=axs[i],   
#                 ti=tti, 
#                 # scatter=True, 
#                 # vmin=1e-10,
#                 # vmax=8e-9,
#                 # colorbar=False,
#             )
    
#     axs[i].set_title('atmbc t:' + str(tti))
#     axs[i].set_aspect('equal')
    

# # Create a new axes for the colorbar
# cbar_ax = fig.add_axes([0.5, 0.15, 0.3, 0.01])  # [left, bottom, width, height]
# cbar = plt.colorbar(cmap, cax=cbar_ax, 
#                     orientation='horizontal')
# cbar.set_label('actual ET (m/s)')
# plt.tight_layout()
# fig.savefig(os.path.join(simu.workdir,simu.project_name,'simu_DA_spatial_WT.png'),
#             dpi=350,
#             bbox_inches='tight')

#%% Plot median ensemble SW map
#------------------------------------------------------------------------------
# fig, axes = plt.subplots(2,2)
# axes = axes.ravel()
# for i, time_value in enumerate(ET_times_sec):
#     data_subset = mean_df.xs(time_value,level=0)
#     data_matrix = data_subset.pivot_table(index='Y', 
#                                           columns='X', 
#                                           values='ACT. ETRA').values
#     cmap = axes[i].imshow(data_matrix,
#                           extent=[df_fort777.X.min(),df_fort777.X.max(),
#                                   df_fort777.Y.min(),df_fort777.Y.max()
#                                   ],
#                            origin='lower', 
#                            aspect='auto',
#                            clim=[1e-7,2e-7],
#                           )
#     # Setting titles and labels
#     axes[i].set_title(f'Time: {time_value}')
#     axes[i].set_xlabel('X')
#     axes[i].set_ylabel('Y')

# plt.colorbar(cmap)
# plt.tight_layout()
# plt.show()

#%% Plot mean ensemble SW map
#------------------------------------------------------------------------------

# fig, axes = plt.subplots(2,2)
# axes = axes.ravel()
# for i, time_value in enumerate(ET_times_sec):
#     data_subset = median_df.xs(time_value,level=0)
#     data_matrix = data_subset.pivot_table(index='Y', 
#                                           columns='X', 
#                                           values='ACT. ETRA').values
#     cmap = axes[i].imshow(data_matrix,
#                           extent=[df_fort777.X.min(),df_fort777.X.max(),
#                                   df_fort777.Y.min(),df_fort777.Y.max()
#                                   ],
#                            origin='lower', 
#                            aspect='auto',
#                            vmin=1e-7,vmax=2e-7,
#                           )
#     # Setting titles and labels
#     axes[i].set_title(f'Time: {time_value}')
#     axes[i].set_xlabel('X')
#     axes[i].set_ylabel('Y')

# plt.colorbar(cmap)
# plt.tight_layout()
# plt.show()

# df_ET_All_Ensemble.xs(ET_times[0],level=1)
# df_ET_All_Ensemble.to_xarray()




#%%

# fig, axs = plt.subplots(1,4,figsize=(8,5),sharey=True)

# tt = np.linspace(1,len(atmbc_times)-1,4)
# tt = [int(tti) for tti in tt]
# for i, tti in enumerate(tt):
#     cmap = simu.show('spatialET',
#                 ax=axs[i],   
#                 ti=tti, 
#                 scatter=True, 
#                 # vmin=1e-10,
#                 # # vmax=8e-9,
#                 # colorbar=False,
#             )
    
#     axs[i].set_title('atmbc t:' + str(tti))
#     axs[i].set_aspect('equal')
    

# # Create a new axes for the colorbar
# cbar_ax = fig.add_axes([0.5, 0.15, 0.3, 0.01])  # [left, bottom, width, height]
# cbar = plt.colorbar(cmap, cax=cbar_ax, orientation='horizontal')
# cbar.set_label('actual ET (m/s)')
# plt.tight_layout()
# fig.savefig(os.path.join(simu.workdir,simu.project_name,'simu_DA_spatial_act_ET.png'),
#             dpi=350,
#             bbox_inches='tight')

#%%
observations.xs('ETact', level=0)['data']
fig, axs = plt.subplots(2,1,
                        figsize=(6,3),
                        sharex=True,
                       )


for i in range(int(len(nodes2plots)/2)):
    observations.xs('ETact' + str(nodes2plots_surf[i]), level=0).plot(y='data',
                                                                    ax=axs[0],
                                                                    marker='.',
                                                                    label='node' + str(nodes2plots_surf[i]),
                                                                    c=colors_surf[i]
                                                                    )
    df_atmbc.xs(nodes2plots_surf[i],level=1).plot(ax=axs[1],
                                                c=colors_surf[i],
                                                legend=None,
                                                )

# observations.xs('ETact' + str(nodes2plots[2][0]), level=0).plot(y='data',
#                                                                 ax=ax,
#                                                                 marker='.',
#                                                                 label='node' + str(nodes2plots[2]),
#                                                                 c='b'
#                                                                 )
# df_atmbc.xs(nodes2plots[2][0],level=1).plot(ax=ax,
#                                             c='b')
axs[0].set_ylabel('ETp [m/s]')
axs[1].set_ylabel('ETact [m/s]')
axs[1].set_xlabel('Time [s]')
plt.tight_layout()
fig.savefig(os.path.join(simu.workdir,simu.project_name,
                         'ETact_dynamic.png'),
            dpi=300,
            bbox_inches='tight'
            )

# fig.savefig(os.path.join(simu.workdir,simu.project_name,
#                          'ETp_VS_ETa.png'),
#             dpi=300,
#             bbox_inches='tight')


#%%
assimilationTimes = observations.xs('ETact', level=0).index.to_list()
obsAssimilated = observations.index.get_level_values(0).unique().values


#%%

# fig, axs = plt.subplots(1,4,figsize=(8,5)) #,sharey=True)

# for i in range(4):
#     simu_ref.show('spatialET',
#                 ax=axs[i],   
#                 ti=i+2, 
#                 scatter=True, 
#                 vmin=0,
#                 vmax=1e-9,
#             )

#%%

# cplt.show_vtk(
#     unit="pressure",
#     timeStep=1,
#     notebook=False,
#     path=simu.workdir + prj_name_ref,
#     savefig=True,
# )

#%% Plot state dynamic
# -------------------------

#%%

#%% Plot parameters dynamic
# -------------------------
fig, ax = plt.subplots(2,figsize=[11,4])
pltCT.DA_RMS(results['df_performance'],'ETact',ax=ax)

#%%
# plt.close('all')
obs2plot_selec = observations.xs('ETact')[['data','data_err']]
    
psi = simu_ref.read_outputs('psi')
sw, sw_times = simu_ref.read_outputs('sw')
tass = results['df_DA'].time.unique()
np.shape(sw)



#%%
# nodes_interest_ZROOT = [240,241]
fig, axs = plt.subplots(len(nodes2plots),1, 
                        sharex=True,sharey=True,
                        figsize=(5,10)
                        )#,figsize=(6,7))

sw_datetimes = change_x2date(sw_times,start_date)


for i, nn in enumerate(nodes2plots):
    
    axs[i].plot(sw_datetimes[1:],sw.iloc[1:,nn],'r',label='ref',linestyle='--',marker='.')
    pltCT.DA_plot_time_dynamic(results['df_DA'],
                                'sw',
                                nn,
                                savefig=False,
                                ax=axs[i],
                                start_date=start_date,
                                atmbc_times=atmbc_times
                                )  
    axs[i].set_xlabel('Assimiliation time')
    plt.legend()
    plt.tight_layout()
    # print(simu.project_name)
    fig.savefig(os.path.join(simu.workdir,simu.project_name,'states_dyn.png'),dpi=300)

#%%
# nodes_interest_ZROOT = [240,241]
fig, axs = plt.subplots(len(nodes2plots),1, 
                        sharex=True,sharey=True,
                        figsize=(5,10)
                        )#,figsize=(6,7))

sw_datetimes = change_x2date(sw_times,start_date)


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
    axs[i].set_xlabel('Assimiliation time')
    plt.legend()
    plt.tight_layout()
    print(simu.project_name)
    fig.savefig(os.path.join(simu.workdir,simu.project_name,'states_dyn_psi.png'),dpi=300)
    
    
#%%

# test = results['df_DA'].set_index(['time','Ensemble_nb'])
# test.columns
# test.index[0]
# test.xs(test.index[0])['sw_bef_update_']

#%%
# # fig, axs = plt.subplots(len(nodes_interest),1,figsize=(6,7))
# fig, ax = plt.subplots(figsize=(6,7))

# for i, nn in enumerate(nodes_interest):
#     # fig, ax = plt.subplots()
#     # plt.plot(sw_times,sw[:,nn],'r',label='ref',linestyle='--',marker='.')
    
#     # fig, ax = plt.subplots(figsize=(6,3))
    
#     # axs[i].plot(np.arange(-1,6),sw[:,nn],'r',label='ref',linestyle='--',marker='.')
#     ax.plot(np.arange(-1,6),sw[:,nn],'r',label='ref',linestyle='--',marker='.')
#     pltCT.DA_plot_time_dynamic(results['df_DA'],
#                                 'sw',
#                                 nn,
#                                 savefig=False,
#                                 ax=ax,
#                                 # start_date=today,
#                                 # atmbc_times=second_ax_data['time']
#                                 )  
#     plt.legend()
#     plt.tight_layout()
#     # fig.savefig()
#     print(simu.project_name)
#     fig.savefig(os.path.join(simu.workdir,simu.project_name,'states_dyn.png'),dpi=300)

#%% Plot parameters dynamic
# -------------------------

veg_map, veg_map_hd = simu_ref.read_inputs('root_map')
df_SPP, df_FP = simu_ref.read_inputs('soil',MAXVEG=len(np.unique(veg_map)))


results['dict_parm_pert'].keys()
# try:
    
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
        # if 'WT' in kk:
        #     ax.plot(np.arange(0,nb_ass_times,1),[df_FP['WT'].iloc[zone_nb]]*nb_ass_times,
        #             color='red',linestyle='--')
        if 'ZROOT' in kk:
            ax.plot(np.arange(0,nb_ass_times,1),[df_FP['ZROOT'].iloc[zone_nb]]*nb_ass_times,
                    color='red',linestyle='--')
        elif 'PCREF' in kk:
            ax.plot(np.arange(0,nb_ass_times,1),[df_FP['PCREF'].iloc[zone_nb]]*nb_ass_times,
                    color='red',linestyle='--')
            
        fig.savefig( os.path.join(simu.workdir,simu.project_name,kk + '.png'),dpi=300)
    except:
        pass
# except:
#%%

# fig, axs = plt.subplots(3,3)
# axs = axs.ravel()
# for kk in results['dict_parm_pert'].keys():
     
#     i = 0
#     # zone_nb = int(''.join(filter(str.isdigit, kk)))
#     # nb_ass_times= len(results['dict_parm_pert'][kk]['ini_perturbation'])
#     if 'ZROOT' in kk:
        
#         pltCT.DA_plot_parm_dynamic_scatter(parm = kk, 
#                                             dict_parm_pert=results['dict_parm_pert'], 
#                                             list_assimilation_times = assimilationTimes,
#                                             ax=axs[i],
#                                                   )
#     i =+ 1   
    
#%%
dict_parm_pert = results['dict_parm_pert']

if 'atmbc0' in dict_parm_pert.keys():
        
    fig, axs = plt.subplots(len(nodes2plots_surf),1,figsize=(10,5))
    
    for i, nn in enumerate(nodes2plots_surf):
        
        times_2plot = np.unique(dict_parm_pert['atmbc'+str(nn)]['data2perturbate']['time'])
        atmbc_2plot = dict_parm_pert['atmbc'+str(nn)]['data2perturbate']['VALUE']
        atmbc_pert_2plot = dict_parm_pert['atmbc'+str(nn)]['time_variable_perturbation']
        axs[i].plot(times_2plot,atmbc_2plot,color='r',marker='v',label='Node' + str(nn) + ', atmbc ref')
        axs[i].plot(times_2plot,atmbc_pert_2plot, marker='.', label='Node' + str(nn) + ', Ensemble')
        axs[i].set_xlabel('Assimilation time')
        axs[i].set_ylabel('net atmbc (m/s)')
        
        plt.legend()
        
    fig.savefig(os.path.join(simu.workdir,simu.project_name,'atmbc.png'),dpi=300)

#%%

# results['df_DA'].set_index(['time','Ensemble_nb']).index

