'''   
Model perturbations are: 
------------------------
'''

import numpy as np

def load_scenario(study='hetsoil',**kwargs):
    
    if 'test' in study:
        scenarii = test()
    elif 'ZROOTdim2' in study:
        scenarii = ZROOTdim2_scenarii()
    elif 'ETdim441' in study:
        scenarii = ETdim441_scenarii()
    # elif 'VGP' in study:
    #     scenarii = VGP_scenarii()
    # elif 'ZROOTdim2' in study:
        # scenarii = ZROOTdim2_scenarii()
    return scenarii
        

# ic
# -------------------
pert_nom_ic = -1.5
pert_sigma_ic = 0.75

# ZROOT
# -------------------
pert_nom_ZROOT = 1
pert_sigma_ZROOT = 0.35
minZROOT = 0
maxZROOT = 2


# Water table
# -------------------
pert_nom_WT = 0.5
pert_sigma_WT = 0.75
minWT = 0.5
maxWT = 2

# Atmbc
# -------------------
pert_sigma_atmbcETp = 1e-7
pert_nom_atmbcETp= -2e-7
time_decorrelation_len = 40e3

# Ks
# -------------------
pert_nom_ks = 1.880e-04
pert_sigma_ks = 1.75


def ETdim441_scenarii():
    scenarii = {
                                               
                                                
            'WTD_ZROOT_zones_withUpd': 
                                                {'per_type': [None,None],
                                                 'per_name':['WTPOSITION', 'ZROOT'],
                                                 'per_nom':[pert_nom_WT,pert_nom_ZROOT],
                                                 'per_mean':[pert_nom_WT,pert_nom_ZROOT],    
                                                 'per_sigma': [pert_sigma_WT,pert_sigma_ZROOT],
                                                 'per_bounds': [{'min':minWT,'max':maxWT},
                                                                {'min':minZROOT,'max':maxZROOT}
                                                                ],
                                                 'sampling_type': ['normal','normal'],
                                                 'transf_type':[None,None,None],
                                                 'listUpdateParm': ['St. var.', 'ZROOT'],
                                                 'listObAss': ['RS_ET'],
                                                 },
                                                
            
            'ET_WTD_ZROOT_withZROOTUpdate': 
                                                {'per_type': [None,None,None],
                                                  'per_name':['WTPOSITION', 'ZROOT','atmbc'],
                                                  'per_nom':[pert_nom_WT,pert_nom_ZROOT,pert_nom_atmbcETp],
                                                  'per_mean':[pert_nom_WT,pert_nom_ZROOT,pert_nom_atmbcETp],    
                                                  'per_sigma': [pert_sigma_WT,pert_sigma_ZROOT,pert_sigma_atmbcETp],
                                                  'per_bounds': [{'min':minWT,'max':maxWT},{'min':minZROOT,'max':maxZROOT},None],
                                                  'sampling_type': ['normal','normal','normal'],
                                                  'transf_type':[None,None,None],
                                                  'time_decorrelation_len': [None,None,time_decorrelation_len],
                                                  'listUpdateParm': ['St. var.', 'ZROOT'],
                                                  'listObAss': ['RS_ET'],
                                                  },
            
            }
    return scenarii
        

#            }

def ZROOTdim2_scenarii(nzones = 2):
    
    nnod = 441
    # nzones = 9
    # pert_nom_ZROOT_IND_tmp = list(np.linspace(0.25,1.25,9))
    # pert_nom_ZROOT_IND = [ pz + 0.25 for pz in pert_nom_ZROOT_IND_tmp]
    pert_nom_ZROOT_IND = [0.25,1.25]
    
    scenarii = {

            

            # # NON biased ZROOT normal distribution
            # # ------------------------------------
            'ZROOT_pert2zones_withUpd': 
                                                {'per_type': [None],
                                                 'per_name': ['ZROOT']*nzones,
                                                 'per_nom':
                                                            [pert_nom_ZROOT_IND]
                                                            ,
                                                 'per_mean':
                                                            [pert_nom_ZROOT_IND]
                                                            ,    
                                                 'per_sigma':
                                                              [[pert_sigma_ZROOT]*nzones]
                                                            ,
                                                 'per_bounds': 
                                                                [[{'min':minZROOT,'max':maxZROOT}]*nzones]
                                                            ,
                                                 'sampling_type': 
                                                                  [['normal']*nzones]
                                                                ,
                                                 'transf_type':
                                                                [[None]*nzones]
                                                            ,
                                                 'listUpdateParm': ['St. var.', 'ZROOT'],
                                                 'listObAss': ['RS_ET'],
                                                 },
                                                        
                                                
            # # NON biased ZROOT normal distribution
            # # ------------------------------------
            'WTD_ZROOT_pert2zones_withUpd': 
                                                {'per_type': [None,None],
                                                 'per_name':['WTPOSITION'] + ['ZROOT']*nzones,
                                                 'per_nom':[pert_nom_WT,
                                                            pert_nom_ZROOT_IND
                                                            ],
                                                 'per_mean':[pert_nom_WT,
                                                            pert_nom_ZROOT_IND
                                                            ],    
                                                 'per_sigma': [pert_sigma_WT,
                                                              [pert_sigma_ZROOT]*nzones
                                                            ],
                                                 'per_bounds': [{'min':minWT,'max':maxWT},
                                                                [{'min':minZROOT,'max':maxZROOT}]*nzones
                                                            ],
                                                 'sampling_type': ['normal',
                                                                  ['normal']*nzones
                                                                ],
                                                 'transf_type':[None,
                                                                [None]*nzones
                                                            ],
                                                 'listUpdateParm': ['St. var.', 'ZROOT'],
                                                 'listObAss': ['RS_ET'],
                                                 },
                                                        

            # # UNBIASED ZROOT + ATMBC normal distribution
            # # ------------------------------------
            'ET_WTD_ZROOT_pert9zones_withUpd': 
                                                {'per_type': [None,None,None],
                                                  'per_name':['WTPOSITION','ZROOT','atmbc'],
                                                  'per_nom':[pert_nom_WT,pert_nom_ZROOT_IND,pert_nom_atmbcETp],
                                                  'per_mean':[pert_nom_WT,pert_nom_ZROOT_IND,pert_nom_atmbcETp],    
                                                  'per_sigma': [pert_sigma_WT,[pert_sigma_ZROOT]*nzones,pert_sigma_atmbcETp],
                                                  'per_bounds': [{'min':minWT,'max':maxWT},[{'min':minZROOT,'max':maxZROOT}]*nzones,None],
                                                  'sampling_type': ['normal',['normal']*nzones,'normal'],
                                                  'transf_type':[None,[None]*nzones,None],
                                                  'time_decorrelation_len': [None,None,time_decorrelation_len],
                                                  'listUpdateParm': ['St. var.', 'ZROOT'],
                                                  'listObAss': ['RS_ET'],
                                                  },

            # # BIASED ZROOT normal distribution
            # # ------------------------------------
                                                
            }
        
    
    return scenarii
    
    



def test():
# ------------------------------------------------------------- #
# Testing
# usually use an smaller ensemble size
# ------------------------------------------------------------- #
    scenarii = {
                    # 'ic_pert_narrow': {'per_type': [None],
                    #         'per_name':['ic'],
                    #         'per_nom':[-5],
                    #         'per_mean':[-5],
                    #         'per_sigma': [1e-6],
                    #         'transf_type':[None],
                    #         'listUpdateParm': ['St. var.'],
                    #         'listObAss': ['RS_ET'],
                    #         },
    
                    # 'ic_pert_large': {'per_type': [None],
                    #         'per_name':['ic'],
                    #         'per_nom':[0],
                    #         'per_mean':[0],
                    #         'per_sigma': [2.75],
                    #         'transf_type':[None],
                    #         'listUpdateParm': ['St. var.'],
                    #         'listObAss': ['RS_ET'],
                    #         },
                    'ZROOT_pert_narrow': {'per_type': [None],
                            'per_name':['ZROOT'],
                            'per_nom':[1],
                            'per_mean':[1],
                            'per_sigma':[1e-6],
                            'per_bounds': [{'min':0,'max':1}],
                            'transf_type':[None],
                            'listUpdateParm': ['St. var.'],
                            'listObAss': ['RS_ET'],
                            },

                    'ZROOT_pert': {'per_type': [None],
                            'per_name':['ZROOT'],
                            'per_nom':[pert_nom_ZROOT],
                            'per_mean':[pert_nom_ZROOT],
                            'per_sigma':[pert_sigma_ZROOT],
                            'per_bounds': [{'min':0,'max':2}],
                            'transf_type':[None],
                            'listUpdateParm': ['St. var.'],
                            'listObAss': ['RS_ET'],
                            },
                    
                    'WTD_withUpd': 
                                    {'per_type': [None],
                                     'per_name':['WTPOSITION'],
                                     'per_nom':[0],
                                     'per_mean':[0],    
                                     'per_sigma': [0.75],
                                     'per_bounds': [{'min':0,'max':1}],
                                     'sampling_type': ['normal'],
                                     'transf_type':[None],
                                     'listUpdateParm': ['St. var.'],
                                     'listObAss': ['RS_ET'],
                                     },
                                    
                    'ET_pert': {
                                            'per_type': [None],
                                            'per_name':['atmbc'],
                                            'per_nom':[pert_nom_atmbcETp],
                                            'per_mean':[pert_nom_atmbcETp],
                                            'per_sigma': [pert_sigma_atmbcETp],
                                            'time_decorrelation_len': [40e3],
                                            'transf_type':[None],
                                            'listUpdateParm': ['St. var.'], 
                                            'listAssimilatedObs': ['RS_ET'],
                                            },
                    
                    'ET_pert_withZROOTUpdate': {
                                            'per_type': [None],
                                            'per_name':['atmbc'],
                                            'per_nom':[pert_nom_atmbcETp],
                                            'per_mean':[pert_nom_atmbcETp],
                                            'per_sigma': [pert_sigma_atmbcETp],
                                            'time_decorrelation_len': [40e3],
                                            'transf_type':[None],
                                            'listUpdateParm': ['St. var.', 'ZROOT'],
                                            'listAssimilatedObs': ['RS_ET'],
                                            },
                    
                    # 'ks_pert_narrow': {'per_type': [None],
                    #         'per_name':['ks'],
                    #         'per_nom':[pert_nom_ks],
                    #         'per_mean':[pert_nom_ks],
                    #         'per_sigma':[1e-6],
                    #         'transf_type':[None],
                    #         'listUpdateParm': ['St. var.'],
                    #         'listObAss': ['RS_ET'],
                    #         },
                }
    
    return scenarii



# def ZROOTdim2_scenarii():
#     scenarii = {
        
#             'WTD_ZROOT_pert2zones_withUpd': 
#                                                 {'per_type': [None,None,'additive'],
#                                                  'per_name':['WTPOSITION', 'ZROOT','PCREF'],
#                                                  'per_nom':[1,[pert_nom_ZROOT,pert_nom_ZROOT],[-5,-5]],
#                                                  'per_mean':[1,[pert_nom_ZROOT,pert_nom_ZROOT],[-5,-5]],    
#                                                  'per_sigma': [1.75,[pert_sigma_ZROOT,pert_sigma_ZROOT],[2.55,2.55]],
#                                                  'per_bounds': [{'min':0,'max':2},[{'min':0,'max':2},{'min':0,'max':2}],None],
#                                                  'sampling_type': ['normal','normal','normal'],
#                                                  'transf_type':[None,[None,None],[None,None]],
#                                                  'listUpdateParm': ['St. var.', 'ZROOT','PCREF'],
#                                                  'listObAss': ['RS_ET'],
#                                                  },
                                                
            
#             'ET_WTD_ZROOT_pert2zones_withZROOTUpdate': {
#                                                 'per_type': [None,None,'additive',None],
#                                                  'per_name':['WTPOSITION', 'ZROOT','PCREF','atmbc'],
#                                                  'per_nom':[1,[pert_nom_ZROOT,pert_nom_ZROOT],[-5,-5],pert_nom_atmbcETp],
#                                                  'per_mean':[1,[pert_nom_ZROOT,pert_nom_ZROOT],[-5,-5],pert_nom_atmbcETp],    
#                                                  'per_sigma': [1.75,[pert_sigma_ZROOT,pert_sigma_ZROOT],[2.55,2.55],pert_sigma_atmbcETp],
#                                                  'per_bounds': [{'min':0,'max':2},[{'min':0,'max':2},{'min':0,'max':2}],None],
#                                                  'sampling_type': ['normal','normal','normal','normal'],
#                                                  'transf_type':[None,[None,None],[None,None],None],
#                                                  'time_decorrelation_len': [None,None,None,40e3],
#                                                  'listUpdateParm': ['St. var.', 'ZROOT','PCREF'],
#                                                  'listObAss': ['RS_ET'],
#                                                  },
                                                 
                                     
