# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 10:00:16 2018

@author: aoanng
"""

from pylab import *
import treePlotter
from ID3Tree import *
mpl.rcParams['font.sans-serif'] = ['SimHei']  # Specify default font
mpl.rcParams['axes.unicode_minus'] = False  # Solve the problem that the negative sign '-' is displayed as a square when saving the image
##################################

# Construction of test decision tree
myDat, labels = createDataSet()
myTree = createTree(myDat, labels)
# Draw decision tree

treePlotter.createPlot(myTree)