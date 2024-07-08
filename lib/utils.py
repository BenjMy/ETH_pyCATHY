import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import os 

#%%
def define_mesh_transformations(meshCiPG,meshLiPG,
                                idC=10, 
                                idL=10,
                                ):

    points = meshCiPG.points.copy()
    meshCiPG_new = meshCiPG.copy()

    transform_matrix = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    transformed_points_meshCi = np.dot(points, transform_matrix)
    transformed_points_meshCi[:,0] = idC/2
    transformed_points_meshCi.points = transformed_points_meshCi
    meshCiPG_new.points = transformed_points_meshCi
    
    points = meshLiPG.points.copy()
    meshLiPG_new = meshLiPG.copy()

    transform_matrix = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    transformed_points_meshLi = np.dot(points, transform_matrix)
    transformed_points_meshLi[:,1] = idL/2
    meshLiPG_new.points = transformed_points_meshLi
    
    return meshCiPG_new, meshLiPG_new
     
     
def backup_simulog_DA(args,filename='DAlog.csv'):
    results_df = pd.read_csv(filename,index_col=0)
    now = datetime.now()
    results_df_cols = vars(args).keys()
    results_df_new = pd.DataFrame([vars(args)])
    cols2check = list(vars(args).keys())
    
    values = results_df_new[cols2check].values
    matching_index = results_df.index[(results_df[cols2check] == values).all(axis=1)].tolist()
    if matching_index:
        now = datetime.now()
        results_df.loc[matching_index, 'datetime'] = now
        matching_index = matching_index[0]
    else:
        results_df_new['datetime']=now
        results_df = pd.concat([results_df,results_df_new],ignore_index=True)
        matching_index = len(results_df)-1
    results_df.to_csv(filename)
    return results_df, matching_index


def get_mesh_node(simu,
                  node_pos = [5,2.5,2],
                  depths = [0.05,0.15,0.25,0.75]
                  ):
    
    _ , closest_pos_notopo = simu.find_nearest_node(node_pos)
    closest_pos = []
    nodeId = []
    for d in depths:
        node_pos_withdepth = [node_pos[0],node_pos[1],closest_pos_notopo[0][2]-d] 
        nodeId_tmp, closest_pos_tmp = simu.find_nearest_node(node_pos_withdepth)
        nodeId.append(nodeId_tmp)
        closest_pos.append(closest_pos_tmp)
    nodeIds = np.vstack(nodeId)
    closest_positions = np.vstack(closest_pos)
    return nodeIds,closest_positions


def get_NOIs(simu,
             depths = [0,1],
             maxdem=1):
    '''
    Get nodes of interests for the synthetic case with 2 zones ZROOT
    '''
    
    alt = maxdem
    pts2plot = np.array([[5       , 8       , alt ],
                        [5.        , 2.        , alt ],
                        ]
                        )
    nodes2plots = []
    nodespos2plots = []
    for pp in pts2plot:
        nodes2plots_tmp = []
        nodespos2plots_tmp = []
        for dd in depths:
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
    
    
    nodes2plots_surf = [nodes2plots[0][0], 
                        nodes2plots[2][0]
                        ]
    nodespos2plots_surf = [nodespos2plots[0][0:2],
                           nodespos2plots[2][0:2], 
                           ]
    return nodes2plots, nodespos2plots, nodes2plots_surf, nodespos2plots_surf


def root_updownhill_zones(indice_veg):
    indice_veg[:int(len(indice_veg)/2),:]=2
    return indice_veg


def gradient_root(indice_veg):
    rows, cols = 3,3
    # rows, cols = 1, 1
    # Calculate the size of each part
    part_size = 21 // rows
    # Define the values for each part
    values = np.arange(1, rows * cols + 1)
    # Shuffle the values randomly
    # np.random.shuffle(values)
    # Populate each part with its respective value
    for i in range(rows):
        for j in range(cols):
            row_start = i * part_size
            row_end = (i + 1) * part_size
            col_start = j * part_size
            col_end = (j + 1) * part_size
            value = values[i * rows + j]
            # Check if the value is already used in the matrix
            while value in indice_veg:
                np.random.shuffle(values)
                value = values[i * rows + j]
            indice_veg[row_start:row_end, col_start:col_end] = value
            
# Function to create a Gaussian covariance matrix
def gaussian_covariance_matrix(size, sigma):
    """Create a Gaussian covariance matrix."""
    cov_matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            cov_matrix[i, j] = np.exp(-((i - j)**2) / (2 * sigma**2))
    return cov_matrix


def localization_function(distance, length_scale):
    return np.exp(-(distance**2) / (2 * length_scale**2))


def make_data_localisation(cov_matrix,length_scale):
    zone1 = np.arange(0,int(len(cov_matrix)/2))
    zone2 = np.arange(int(len(cov_matrix)/2),len(cov_matrix))
    localized_cov_matrix = np.zeros_like(cov_matrix)
    localized_matrix = np.zeros_like(cov_matrix)
    
    # Populate the localized covariance matrix
    for i in range(cov_matrix.shape[0]):
        for j in range(cov_matrix.shape[1]):
            if (i in zone1 and j in zone1) or (i in zone2 and j in zone2):
                distance = abs(i - j)
                localized_cov_matrix[i, j] = cov_matrix[i, j] * localization_function(distance, length_scale)
                localized_matrix[i, j] = localization_function(distance, length_scale)
    return localized_cov_matrix, zone1, zone2


def plot_atmbc_pert(simu,
                    var_per_dict_stacked,
                    ):
    times_2plot = var_per_dict_stacked['atmbc0']['data2perturbate']['time']
    atmbc_2plot = var_per_dict_stacked['atmbc0']['data2perturbate']['VALUE']
    atmbc_pert_2plot = var_per_dict_stacked['atmbc0']['time_variable_perturbation']
    fig, ax = plt.subplots()
    ax.plot(times_2plot,atmbc_2plot,color='r',marker='v',label='Node0, atmbc ref')
    ax.plot(times_2plot,atmbc_pert_2plot, marker='.', label='Node0, Ensemble')
    ax.set_xlabel('Assimilation time')
    ax.set_ylabel('net atmbc (m/s)')
    plt.legend()
    fig.savefig( os.path.join(simu.workdir,simu.project_name,'time_variable_perturbation.png'),dpi=300)