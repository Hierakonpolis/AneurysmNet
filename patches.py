#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 20:23:02 2020

@author: cat
"""

import dataset as D

dataroot='/scratch/project_2003143/ADAM_release_subjs'
D.ExtractPatches(dataroot, '/scratch/project_2003143/patches64', 100,size=64)