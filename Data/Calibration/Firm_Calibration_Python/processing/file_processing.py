'''
-------------------------------------------------------------------------------
Last updated 5/26/2015
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
    Packages
-------------------------------------------------------------------------------
'''
import os
import sys
'''
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
'''



def get_file(dirct, contains=[""]):
    for filename in os.listdir(dirct):
        if all(item in filename for item in contains):
            return filename
