#!/usr/bin/env python3
import math
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
input_file_name = 'real_pred.csv'
f_in = open('%s' %input_file_name,'r')
f1 = f_in.readlines()
real=[]
pred=[]
for line in f1:
    real.append(float(line.split(',')[0]))
    pred.append(float(line.split(',')[1]))
    #print(line.split(',')[0], line.split(',')[1])
#print('real:')
#print(real)
#print('pred:')
#print(pred)

r,_   = pearsonr(real,pred)
rho,_ = spearmanr(real,pred)
rms  = math.sqrt(mean_squared_error(real,pred))
print('rms = %.2f. r = %.2f' %(rms,r))
