# -*- coding: utf-8 -*-
"""
Created on Tue Aug 03 15:33:33 2020

@author: Jonathan Ting
"""

import pandas as pd
import numpy as np

pandas.set_option('display.float_format', lambda x:'%f'%x)
data = pd.read_csv('./gapminder.csv', low_memory=False)

print(len(data))
print(len(data.columns))
print(data.columns)
