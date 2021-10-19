#!/usr/bin/env python3
import math
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
# set up file
input_file_name = 'real_pred.csv'
f_in = open('%s' %input_file_name,'r')
f1 = f_in.readlines()
real=[]
pred=[]
# real real,pred data
for line in f1:
    real.append(float(line.split(',')[0]))
    pred.append(float(line.split(',')[1]))
# calculate metrics
r,_   = pearsonr(real,pred)
rms  = math.sqrt(mean_squared_error(real,pred))
print('rms = %.2f%s. r = %.2f' %(rms,'%',r))
