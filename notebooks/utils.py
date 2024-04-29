import numpy as np

def define_mesh_transformations(meshCiPG,meshLiPG,
                                idC=10, 
                                idL=10,
                                ):

    points = meshCiPG.points.copy()
    transform_matrix = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    transformed_points_meshCi = np.dot(points, transform_matrix)
    transformed_points_meshCi[:,0] = idC/2
    transformed_points_meshCi.points = transformed_points_meshCi
    meshCiPG.points = transformed_points_meshCi
    
    points = meshLiPG.points.copy()
    transform_matrix = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    transformed_points_meshLi = np.dot(points, transform_matrix)
    transformed_points_meshLi[:,1] = idL/2
    meshLiPG.points = transformed_points_meshLi
    
    return meshCiPG, meshLiPG
     
