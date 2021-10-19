#!/usr/bin/env python3
import sys
import math
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
# set up files
input_file_name = 'real_pred.csv'
f_in = open('%s' %input_file_name,'r')
f1 = f_in.readlines()
PCE_threshold = float(sys.argv[1])
real=[]
pred=[]
# read real,pred
for line in f1:
    real.append(float(line.split(',')[0]))
    pred.append(float(line.split(',')[1]))
# initialize
TP = 0.0
FP = 0.0
TN = 0.0
FN = 0.0
# calculate how many points fulfill condition
for i in range(len(pred)):
    if real[i] >= PCE_threshold and pred[i] >= PCE_threshold: TP = TP + 1.0
    if real[i] <  PCE_threshold and pred[i] >  PCE_threshold: FP = FP + 1.0
    if real[i] <= PCE_threshold and pred[i] <= PCE_threshold: TN = TN + 1.0
    if real[i] >  PCE_threshold and pred[i] <  PCE_threshold: FN = FN + 1.0
print('TP:', TP)
print('FP:', FP)
print('TN:', TN)
print('FN:', FN)
# print metrics
if TP >0:
    Accuracy = (TP+TN)/(TP+TN+FP+FN)
    Precision = TP/(TP+FP)
    Recall = TP/(TP+FN)
    F = 2.0 * (Precision*Recall)/(Precision+Recall)
    print('%10s %.2f' %('Accuracy',Accuracy))
    print('%10s %.2f' %('Precision',Precision))
    print('%10s %.2f' %('Recall',Recall))
    print('%10s %.2f' %('F1-score',F))
elif TP==0:
    F = 0
    print('NO TP found')
