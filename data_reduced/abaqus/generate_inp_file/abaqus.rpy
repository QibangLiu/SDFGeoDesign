# -*- coding: mbcs -*-
#
# Abaqus/CAE Release 2024 replay file
# Internal Version: 2023_09_21-07.55.25 RELr426 190762
# Run by qibang on Thu Jan 16 10:32:32 2025
#

# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...
from abaqus import *
from abaqusConstants import *
session.Viewport(name='Viewport: 1', origin=(1.36719, 1.36719), width=201.25, 
    height=135.625)
session.viewports['Viewport: 1'].makeCurrent()
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
execfile(
    '/work/hdd/bdsy/qibang/repository_Wbdsy/GeoSDF2D/data_reduced/abaqus/generate_inp_file/abaqus_script_inp_20-30.py', 
    __main__.__dict__)
