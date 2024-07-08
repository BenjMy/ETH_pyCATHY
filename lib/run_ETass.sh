# run ET assimilation
# -----------------------


						# Surface-Subsurface hydrological simulation - full AREA 
						#--------------------------------------------------------
# reference models

python weilletal_ref.py
python fwd_weilletal_ET_ref_ZROOT_spatially_variable.py
python fwd_weilletal_ET_ref_atmbc_spatially_variable.py

						# DATA ASSIMILATION 
						# -----------------

# Varying root depths (2 zones) assimilation of actual ET

python DA_ETa.py -study ZROOTdim2 -sc 0 -nens 64 -DAtype enkf_analysis_inflation_multiparm -damping 1 -dataErr 1e99 -refModel ZROOT_spatially_from_weill -DAloc 0
python DA_ETa.py -study ZROOTdim2 -sc 0 -nens 64 -DAtype enkf_analysis_inflation_multiparm -damping 1 -dataErr 1e-11 -refModel ZROOT_spatially_from_weill -DAloc 0
python DA_ETa.py -study ZROOTdim2 -sc 0 -nens 64 -DAtype enkf_Evensen2009 -damping 1 -dataErr 1e-11 -refModel ZROOT_spatially_from_weill -DAloc 0
python DA_ETa.py -study ZROOTdim2 -sc 0 -nens 64 -DAtype enkf_analysis_inflation_multiparm -damping 1 -dataErr 1e-11 -refModel ZROOT_spatially_from_weill -DAloc 1
python DA_ETa.py -study ZROOTdim2 -sc 0 -nens 64 -DAtype enkf_Evensen2009 -damping 1 -dataErr 1e-11 -refModel ZROOT_spatially_from_weill -DAloc 1

python DA_ETa.py -study ZROOTdim2 -sc 1 -nens 64 -DAtype enkf_Evensen2009 -damping 1 -dataErr 1e99 -refModel ZROOT_spatially_from_weill -DAloc 0
python DA_ETa.py -study ZROOTdim2 -sc 1 -nens 64 -DAtype enkf_Evensen2009 -damping 1 -dataErr 0.5 -refModel ZROOT_spatially_from_weill -DAloc 0
python DA_ETa.py -study ZROOTdim2 -sc 1 -nens 64 -DAtype enkf_analysis_inflation_multiparm -damping 1 -dataErr 0.5 -refModel ZROOT_spatially_from_weill -DAloc 0


python synthetic_DA_ET.py -study ZROOTdim441 -sc 0 -nens 32 -DAtype enkf_analysis_inflation_multiparm -damping 1 -dataErr 1e99 -refModel ZROOT_spatially_from_weill -DAloc 0


python synthetic_DA_ET.py -study ZROOTdim441 -sc 0 -nens 32 -DAtype enkf_analysis_inflation_multiparm -damping 1 -dataErr 0.5 -refModel ZROOT_spatially_from_weill -DAloc 0



python synthetic_DA_ET.py -study ZROOTdim441 -sc 0 -nens 32 -DAtype enkf_analysis_inflation_multiparm -damping 1 -dataErr 0.5 -refModel ZROOT_spatially_from_weill
python synthetic_DA_ET.py -study ZROOTdim441 -sc 0 -nens 32 -DAtype enkf_analysis_inflation_multiparm -damping 1 -dataErr 5 -refModel ZROOT_spatially_from_weill -DAloc 1
python synthetic_DA_ET.py -study ZROOTdim441 -sc 0 -nens 32 -DAtype enkf_analysis_inflation_multiparm -damping 1.5 -dataErr 0.5 -refModel ZROOT_spatially_from_weill -DAloc 1


python synthetic_DA_ET.py -study ZROOTdim441 -sc 0 -nens 32 -DAtype enkf_analysis_inflation_multiparm -damping 1 -dataErr 0.5 -refModel ZROOT_spatially_from_weill -DAloc 0
python synthetic_DA_ET.py -study ZROOTdim441 -sc 0 -nens 32 -DAtype enkf_analysis_inflation_multiparm -damping 1 -dataErr 5 -refModel ZROOT_spatially_from_weill -DAloc 0
python synthetic_DA_ET.py -study ZROOTdim441 -sc 0 -nens 32 -DAtype enkf_analysis_inflation_multiparm -damping 1.5 -dataErr 0.5 -refModel ZROOT_spatially_from_weill -DAloc 0



# Varying atmbc assimilation of actual ET
python synthetic_DA_ET.py -study ETdim2 -sc 0 -nens 32 -DAtype enkf_Evensen2009 -damping 1 -dataErr 1e-3 -refModel atmbc_spatially_timely_from_weill



