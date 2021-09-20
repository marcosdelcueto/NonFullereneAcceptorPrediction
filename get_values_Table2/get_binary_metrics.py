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

PCE_median = 3.475
TP = 0.0
FP = 0.0
TN = 0.0
FN = 0.0
for i in range(len(pred)):
    if real[i] > PCE_median and pred[i] > PCE_median: TP = TP + 1.0
    if real[i] < PCE_median and pred[i] > PCE_median: FP = FP + 1.0
    if real[i] < PCE_median and pred[i] < PCE_median: TN = TN + 1.0
    if real[i] > PCE_median and pred[i] < PCE_median: FN = FN + 1.0
print('TP:', TP)
print('FP:', FP)
print('TN:', TN)
print('FN:', FN)
if TP >0:
    Accuracy = (TP+TN)/(TP+TN+FP+FN)
    Precision = TP/(TP+FP)
    Recall = TP/(TP+FN)
    F = 2.0 * (Precision*Recall)/(Precision+Recall)
    print('%10s %.2f' %('Accuracy',Accuracy))
    print('%10s %.2f' %('Precision',Precision))
    print('%10s %.2f' %('Recall',Recall))
    print('%10s %.2f' %('F1-score',F))
else:
    F = 0
    print('no TP found. F:', F)

