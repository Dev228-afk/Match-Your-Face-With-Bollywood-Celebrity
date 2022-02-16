# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 14:44:00 2022

@author: DEV
"""

import os

def execute_system():
    bash1= 'python src/01-generate_image_in_pkl.py'
    bash2= 'python src/02-feature_extractor.py'
    
    os.system(bash1)
    os.system(bash2)
    print("Scessfully Executed")
    
if __name__ == '__main__':
    execute_system()