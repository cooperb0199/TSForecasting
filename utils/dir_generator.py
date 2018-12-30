#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 17:41:00 2018

@author: bencooper
"""

import os

class DirGen:
    @staticmethod
    def create_dir(path):
        if os.path.isdir(path) == False:
            os.makedirs(path)
        