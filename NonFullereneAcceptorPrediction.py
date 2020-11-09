#!/usr/bin/env python3
# Marcos del Cueto
# Department of Chemistry and Materials Innovation Factory, University of Liverpool
# For any queries about the code: m.del-cueto@liverpool.ac.uk
#################################################################################
import re
import sys
import ast
import math
import functools
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib import ticker, gridspec, cm
from math import sqrt
from time import time
from scipy.optimize import differential_evolution
from scipy.stats import pearsonr, spearmanr
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import KNeighborsRegressor, DistanceMetric
#import rdkit
#from rdkit import Chem, DataStructs
#from rdkit.Chem import rdMolDescriptors
#################################################################################
######## START CUSTOMIZABLE PARAMETERS ########
input_file_name = 'inputNonFullereneAcceptorPrediction.inp'  # name of input file
# The rest of input options are inside the specified file
########  END CUSTOMIZABLE PARAMETERS  ########
#################################################################################


#################################################################################
###### START MAIN ######
def main(alpha,gamma_el,gamma_d,gamma_a,C,epsilon,alpha_lim,gamma_el_lim,gamma_d_lim,gamma_a_lim,C_lim,epsilon_lim):
    # Read data
    df=pd.read_csv(db_file,index_col=0)
    # Preprocess data
    #df=preprocess_smiles(df) (not needed, we're reading directly FP)
    #print('xcols:',xcols)
    xcols_flat = [item for sublist in xcols for item in sublist]
    #print('xcols_flat:',xcols_flat)
    X=df[xcols_flat].values
    y=df[ycols].values
    for i in range(Ndata):
        X_d=[]
        X_a=[]
        X1=X[i][0][1:-1].split()
        X2=X[i][1][1:-1].split()
        for j in range(FP_length):
            X_d.append(int(float(X1[j])))
            X_a.append(int(float(X2[j])))
        X[i][0]=X_d
        X[i][1]=X_a
    X=preprocess_fn(X)
    # Get optimimum hyperparameters
    if optimize_hyperparams==True:
        fixed_hyperparams = []
        condition=None
        # Use just structural descriptors
        for i in range(len(elec_descrip)):
            if gamma_el[i]==0.0:
                condition = 'structure'
                print('Optimize hyperparameters using only structural descriptors')
                if print_log==True: f_out.write('Optimize hyperparameters using only structural descriptors \n')
                fixed_hyperparams = [gamma_el]
                if ML=='kNN': 
                    hyperparams = [gamma_d,gamma_a]                
                    bounds = [gamma_d_lim] + [gamma_a_lim]
                elif ML=='KRR':
                    hyperparams = [gamma_d,gamma_a,alpha]
                    bounds = [gamma_d_lim] + [gamma_a_lim] + [alpha_lim]
                elif ML=='SVR':
                    hyperparams = [gamma_d,gamma_a,C,epsilon]
                    bounds = [gamma_d_lim] + [gamma_a_lim] + [C_lim] + [epsilon_lim]
                break
        # Use just electronic (physical) descriptors
        if gamma_d==0.0 and gamma_a==0.0 and condition != 'structure':
            condition='electronic'
            print('Optimize hyperparameters using only electronic (physical) descriptors')
            if print_log==True: f_out.write('Optimize hyperparameters using only electronic (physical) descriptors \n')
            fixed_hyperparams =[gamma_d,gamma_a]
            if ML=='kNN': 
                hyperparams=[gamma_el]
                bounds = gamma_el_lim
            if ML=='KRR': 
                hyperparams=[gamma_el,alpha]
                bounds = gamma_el_lim + [alpha_lim]
            if ML=='SVR': 
                hyperparams=[gamma_el,C,epsilon]
                bounds = gamma_el_lim + [C_lim] + [epsilon_lim]
        # Use both electronic and structural descriptors
        elif condition != 'structure':
            condition='structure_and_electronic'
            print('Optimize hyperparameters using both structural and electronic (physical) descriptors')
            if print_log==True: f_out.write('Optimize hyperparameters using both structural and electronic (physical) descriptors \n')
            fixed_hyperparams = []
            if ML=='kNN':
                hyperparams=[gamma_el,gamma_d,gamma_a]
                bounds = gamma_el_lim + [gamma_d_lim] + [gamma_a_lim]
            if ML=='KRR':
                hyperparams=[gamma_el,gamma_d,gamma_a,alpha]
                bounds = gamma_el_lim + [gamma_d_lim] + [gamma_a_lim] + [alpha_lim]
            if ML=='SVR':
                hyperparams=[gamma_el,gamma_d,gamma_a,C,epsilon]
                bounds = gamma_el_lim + [gamma_d_lim] + [gamma_a_lim] + [C_lim] + [epsilon_lim]
        # Print some info before running ML
        print('######################################')
        print('######################################')
        print('Input options printed above')
        print('######################################')
        print('Now starting ML')
        print('This may take several minutes')
        print('######################################')
        # Set differential evolution parameters
        if ML=='KRR' or ML=='SVR':
            # Add final_call=False to fixed_hyperparameters to indicate that validaton is coming (only relevant for CV='groups')
            fixed_hyperparams.append(False)
            mini_args = (X, y, condition,fixed_hyperparams)
            solver = differential_evolution(func_ML,bounds,args=mini_args,popsize=15,tol=0.01,polish=False,workers=NCPU,updating='deferred')
            # print best hyperparams
            best_hyperparams = solver.x
            best_rmse = solver.fun
            print('#######################################')
            print('Best hyperparameters:',best_hyperparams)
            print('Best rmse:', best_rmse)
            print('#######################################')
            if print_log==True: f_out.write('#######################################\n')
            if print_log==True: f_out.write('Best hyperparameters: %s \n' %(str(best_hyperparams)))
            if print_log==True: f_out.write('Best rmse: %s \n' %(str(best_rmse)))
            if print_log==True: f_out.write('#######################################\n')
            if print_log==True: f_out.flush()
            # Once optimized hyperparams are found, do final calculation using those values
            hyperparams=best_hyperparams.tolist()
            print('###############################################')
            print('Doing final call with optimized hyperparameters')
            print('###############################################')
            if print_log==True: f_out.write('###############################################\n')
            if print_log==True: f_out.write('Doing final call with optimized hyperparameters\n')
            if print_log==True: f_out.write('###############################################\n')
            if print_log==True: f_out.flush()
            flat_hyperparams = hyperparams
            # Change final_call=True in fixed_hyperparameters, to indicate that it is final ML call to calculate rmse of novel group (only relevant if CV='groups')
            fixed_hyperparams[-1]=True
            func_ML(flat_hyperparams,X,y,condition,fixed_hyperparams)
        elif ML=='kNN':
            for k in range(len(Neighbors)):
                #print('kNN, for k = %i' %(Neighbors[k]))
                if condition=='electronic':
                    #hyperparams=[gamma_el,gamma_d,gamma_a]
                    fixed_hyperparams.append(Neighbors[k])
                    # Add final_call=False to fixed_hyperparameters to indicate that validaton is coming (only relevant for CV='groups')
                    fixed_hyperparams.append(False)
                    flat_hyperparams = hyperparams[0] + hyperparams[1:]
                    best_rmse = func_ML(flat_hyperparams,X,y,condition,fixed_hyperparams)
                    best_hyperparams = flat_hyperparams
                    if k==0:
                        total_best_hyperparams = best_hyperparams
                        total_best_rmse = best_rmse
                        best_k = Neighbors[k]
                    else:
                        if best_rmse < total_best_rmse:
                            total_best_hyperparams = best_hyperparams
                            total_best_rmse = best_rmse
                            best_k = Neighbors[k]
                if condition=='structure' or condition=='structure_and_electronic':
                    fixed_hyperparams.append(Neighbors[k])
                    # Add final_call=False to fixed_hyperparameters to indicate that validaton is coming (only relevant for CV='groups')
                    fixed_hyperparams.append(False)
                    mini_args = (X, y, condition,fixed_hyperparams)
                    solver = differential_evolution(func_ML,bounds,args=mini_args,popsize=15,tol=0.01,polish=False,workers=NCPU,updating='deferred')
                    # print best hyperparams
                    best_hyperparams = solver.x
                    best_rmse = solver.fun
                    if k==0:
                        total_best_hyperparams = best_hyperparams
                        total_best_rmse = best_rmse
                        best_k = Neighbors[k]
                    else:
                        if best_rmse < total_best_rmse:
                            total_best_hyperparams = best_hyperparams
                            total_best_rmse = best_rmse
                            best_k = Neighbors[k]
            print('#######################################')
            print('Best hyperparameters:', total_best_hyperparams)
            print('Best kNN, k =', best_k)
            print('Best rmse:', total_best_rmse)
            print('#######################################')
            if print_log==True: f_out.write('#######################################\n')
            if print_log==True: f_out.write('Best hyperparameters: %s \n' %(str(total_best_hyperparams)))
            if print_log==True: f_out.write('Best kNN, k = : %i \n' %(best_k))
            if print_log==True: f_out.write('Best rmse: %s \n' %(str(total_best_rmse)))
            if print_log==True: f_out.write('#######################################\n')
            if print_log==True: f_out.flush()
            # Once optimized hyperparams are found, do final calculation using those values
            if type(total_best_hyperparams) is list: 
                hyperparams=total_best_hyperparams
            else:
                hyperparams=total_best_hyperparams.tolist()
            fixed_hyperparams[-2]=best_k
            # Change final_call=True in fixed_hyperparameters, to indicate that it is final ML call to calculate rmse of novel group (only relevant if CV='groups')
            fixed_hyperparams[-1]=True
            print('###############################################')
            print('Doing final call with optimized hyperparameters')
            print('###############################################')
            if print_log==True: f_out.write('###############################################\n')
            if print_log==True: f_out.write('Doing final call with optimized hyperparameters\n')
            if print_log==True: f_out.write('###############################################\n')
            if print_log==True: f_out.flush()
            flat_hyperparams = hyperparams
            func_ML(flat_hyperparams,X,y,condition,fixed_hyperparams)
    ## Use initial hyperparameters
    elif optimize_hyperparams==False:
        condition='structure_and_electronic'
        print('Hyperparameters are not being optimized')
        # Print some info before running ML
        print('######################################')
        print('######################################')
        print('Input options printed above')
        print('######################################')
        print('Now starting ML')
        print('This may take several minutes')
        print('######################################')
        if print_log==True: f_out.write('Hyperparameters are not being optimized \n')
        fixed_hyperparams = []
        if ML=='kNN' and len(Neighbors)>1:
            print('ERROR: Not optimizing parameters, but more than one possible value in "Neighbors" for kNN')
            sys.exit()
        if ML=='kNN': 
            hyperparams=[gamma_el,gamma_d,gamma_a]
            fixed_hyperparams.append(Neighbors[-1])
        # Add final_call=True in fixed_hyperparameters, to indicate that it is final ML call to calculate rmse of novel group (only relevant if CV='groups')
        fixed_hyperparams.append(True)
        if ML=='KRR': hyperparams=[gamma_el,gamma_d,gamma_a,alpha]
        if ML=='SVR': hyperparams=[gamma_el,gamma_d,gamma_a,C,epsilon]
        flat_hyperparams = hyperparams[0] + hyperparams[1:]
        func_ML(flat_hyperparams,X,y,condition,fixed_hyperparams)
###### END MAIN ######
#################################################################################

#################################################################################
###### START OTHER FUNCTIONS ######
### Function reading input parameters
def read_initial_values(inp):
    # open input file
    input_file_name = inp
    f_in = open('%s' %input_file_name,'r')
    f1 = f_in.readlines()
    # initialize arrays
    input_info = []
    var_name = []
    var_value = []
    # read info before comments. Ignore commented lines and blank lines
    for line in f1:
        if not line.startswith("#") and line.strip(): 
            input_info.append(line.split('#',1)[0].strip())

    # read names and values of variables
    number_elec_descrip=0
    for i in range(len(input_info)):
        var_name.append(input_info[i].split('=')[0].strip())
        var_value.append(input_info[i].split('=')[1].strip())
        if re.match(r'xcols_elec.+', input_info[i]) != None: number_elec_descrip=number_elec_descrip+1
    # close input file
    f_in.close()
    # assign input variables    
    ML = ast.literal_eval(var_value[var_name.index('ML')])               # 'kNN' or 'KRR' or 'SVR'
    Neighbors = ast.literal_eval(var_value[var_name.index('Neighbors')]) # number of nearest-neighbors (only used for kNN)
    alpha  = ast.literal_eval(var_value[var_name.index('alpha')])        # kernel hyperparameter (only used for KRR)
    gamma_el = ast.literal_eval(var_value[var_name.index('gamma_el')])   # hyperparameter with weight of d_el
    gamma_d = ast.literal_eval(var_value[var_name.index('gamma_d')])     # hyperparameter with weight of d_fp_d
    gamma_a = ast.literal_eval(var_value[var_name.index('gamma_a')])     # hyperparameter with weight of d_fp_a
    C = ast.literal_eval(var_value[var_name.index('C')])                 # SVR hyperparameter
    epsilon = ast.literal_eval(var_value[var_name.index('epsilon')])     # SVR hyperparameter
    optimize_hyperparams = ast.literal_eval(var_value[var_name.index('optimize_hyperparams')])# whether hyperparameters are optimized (T) or just use initial values (F). If hyperparam=0.0, then that one is not optimized
    alpha_lim  = ast.literal_eval(var_value[var_name.index('alpha_lim')])       # range in which alpha hyperparam is optimized (only used for KRR)
    gamma_el_lim = ast.literal_eval(var_value[var_name.index('gamma_el_lim')])  # range in which gamma_el is optimized
    gamma_d_lim = ast.literal_eval(var_value[var_name.index('gamma_d_lim')])    # range in which gamma_el is optimized
    gamma_a_lim = ast.literal_eval(var_value[var_name.index('gamma_a_lim')])    # range in which gamma_el is optimized
    C_lim = ast.literal_eval(var_value[var_name.index('C_lim')])                # range in which C is optimized
    epsilon_lim = ast.literal_eval(var_value[var_name.index('epsilon_lim')])    # range in which epsilon is optimized
    db_file = ast.literal_eval(var_value[var_name.index('db_file')])            # name of input file with database
    acceptor_label_column = ast.literal_eval(var_value[var_name.index('acceptor_label_column')])
    CV = ast.literal_eval(var_value[var_name.index('CV')])
    #elec_descrip = ast.literal_eval(var_value[var_name.index('elec_descrip')]) # number of electronic descriptors: they must match the number in 'xcols', and be followed by the two structural descriptors
    xcols = []
    elec_descrip = []
    xcols.append(ast.literal_eval(var_value[var_name.index('xcols_struc')]))  # specify which descriptors are used
    for i in range(number_elec_descrip):
        xcols.append(ast.literal_eval(var_value[var_name.index('xcols_elec'+str(i))]))              # specify which descriptors are used
        elec_descrip.append(len(xcols[i+1]))
    # If we are using CV='groups', add temporarily the AcceptorLabel descriptor to identify each group
    if CV=='groups' or CV=='logo':
        for i in range(len(xcols)):
            if i==1:
                xcols[1].insert(0, acceptor_label_column)
                elec_descrip[0] = elec_descrip[0] +1 # account for the label descriptor
    ycols = ast.literal_eval(var_value[var_name.index('ycols')])              # specify which is target property
    Ndata = ast.literal_eval(var_value[var_name.index('Ndata')])              # number of d/a pairs
    print_log = ast.literal_eval(var_value[var_name.index('print_log')])      # choose whether information is also written into a log file (Default: True)
    log_name = ast.literal_eval(var_value[var_name.index('log_name')])        # name of log file
    print_progress_every_x_percent = ast.literal_eval(var_value[var_name.index('print_progress_every_x_percent')])        # when to print progress
    NCPU = ast.literal_eval(var_value[var_name.index('NCPU')])                # select number of CPUs (-1 means all CPUs in a node)
    FP_length = ast.literal_eval(var_value[var_name.index('FP_length')])      # select number of CPUs (-1 means all CPUs in a node)
    weight_RMSE = ast.literal_eval(var_value[var_name.index('weight_RMSE')])  # select number of CPUs (-1 means all CPUs in a node)
    kfold = ast.literal_eval(var_value[var_name.index('kfold')])
    Nlast = ast.literal_eval(var_value[var_name.index('Nlast')])
    plot_target_predictions = ast.literal_eval(var_value[var_name.index('plot_target_predictions')])
    plot_kNN_distances = ast.literal_eval(var_value[var_name.index('plot_kNN_distances')])
    groups_acceptor_labels = ast.literal_eval(var_value[var_name.index('groups_acceptor_labels')])
    group_test = ast.literal_eval(var_value[var_name.index('group_test')])
    prediction_csv_file_name = ast.literal_eval(var_value[var_name.index('prediction_csv_file_name')])
    columns_labels_prediction_csv = ast.literal_eval(var_value[var_name.index('columns_labels_prediction_csv')])
    predict_unknown = ast.literal_eval(var_value[var_name.index('predict_unknown')])

    # Perform sanity check to see that the dimension of gamma_el and gamma_el_lim is the same as the number of xcols_elecX
    if number_elec_descrip != len(gamma_el) or number_elec_descrip != len(gamma_el_lim):
        print('INPUT ERROR: there is some incoherence with the number of electronic descriptor groups')
        print('Number of xcols_elecX:', number_elec_descrip)
        print('Length of gamma_el', len(gamma_el))
        print('Length of gamma_el_lim', len(gamma_el_lim))
        sys.exit()

    # open log file to write intermediate information
    if print_log==True:
        f_out = open('%s' %log_name,'w')
    else:
        f_out=None
    print('##### START PRINT INPUT OPTIONS ######')
    print('######################################')
    print('########### Parallelization ##########')
    print('######################################')
    print('NCPU = ', NCPU)
    print('######################################')
    print('############ Verbose options #########')
    print('######################################')
    print('print_log = ', print_log)
    print('log_name = ', log_name)
    print('######################################')
    print('######### Data base options ##########')
    print('######################################')
    print('db_file = ', db_file)
    print('Ndata = ', Ndata)
    print('xcols = ', xcols)
    print('ycols = ', ycols)
    print('FP_length = ', FP_length)
    print('predict_unknown = ', predict_unknown)
    if prediction_csv_file_name != None:
        print('#######################################')
        print('####### Output prediction csv #########')
        print('#######################################')
        print('prediction_csv_file_name',prediction_csv_file_name)
        print('columns_labels_prediction_csv',columns_labels_prediction_csv)
    print('######################################')
    print('# Machine Learning Algorithm options #')
    print('######################################')
    print('plot_target_predictions =', plot_target_predictions)
    print('ML =', ML)
    print('### Cross Validation #################')
    print('CV =', CV)
    if CV == 'kf': print('kfold =', kfold)
    if CV == 'last': print('Nlast =', Nlast)
    if CV == 'groups' or CV=='logo':
        print('acceptor_label_column =', acceptor_label_column)
        print('groups_acceptor_labels =', groups_acceptor_labels)
        print('group_test =', group_test)
    print('### General hyperparameters ##########')
    print('optimize_hyperparams = ', optimize_hyperparams)
    print('gamma_el = ', gamma_el)
    print('gamma_d = ', gamma_d)
    print('gamma_a = ', gamma_a)
    print('gamma_el_lim = ', gamma_el_lim)
    print('gamma_d_lim = ', gamma_d_lim)
    print('gamma_a_lim = ', gamma_a_lim)
    print('weight_RMSE = ', weight_RMSE)
    if ML=='kNN':
       print('### k-Nearest Neighbors ("kNN") ######')
       print('Neighbors = ', Neighbors)
       print('plot_kNN_distances =', plot_kNN_distances)
    elif ML=='KRR':
        print('### Kernel Ridge Regression ("KRR") ##')
        print('alpha = ', alpha)
        print('alpha_lim = ', alpha_lim)
    elif ML=='SVR':
        print('### Support Vector Regression ("SVR") ########')
        print('C = ', C)
        print('epsilon = ', epsilon)
        print('C_lim = ', C_lim)
        print('epsilon_lim = ', epsilon_lim)
    print('####### END PRINT INPUT OPTIONS ######')

    if print_log==True: 
        f_out.write('##### START PRINT INPUT OPTIONS ######\n')
        f_out.write('######################################\n')
        f_out.write('########### Parallelization ##########\n')
        f_out.write('######################################\n')
        f_out.write('NCPU %s\n' % str(NCPU))
        f_out.write('######################################\n')
        f_out.write('############ Verbose options #########\n')
        f_out.write('######################################\n')
        f_out.write('print_log %s\n' % str(print_log))
        f_out.write('log_name %s\n' % str(log_name))
        f_out.write('print_progress_every_x_percent %s\n' % str(print_progress_every_x_percent))
        f_out.write('######################################\n')
        f_out.write('######### Data base options ##########\n')
        f_out.write('######################################\n')
        f_out.write('db_file %s\n' % str(db_file))
        f_out.write('Ndata %s\n' % str(Ndata))
        f_out.write('xcols %s\n' % str(xcols))
        f_out.write('ycols %s\n' % str(ycols))
        f_out.write('FP_length %s\n' % str(FP_length))
        f_out.write('predict_unknown %s\n' % str(predict_unknown))
        if prediction_csv_file_name != None:
            f_out.write('#######################################\n')
            f_out.write('####### Output prediction csv #########\n')
            f_out.write('#######################################\n')
            f_out.write('prediction_csv_file_name %s\n' % str(prediction_csv_file_name))
            f_out.write('columns_labels_prediction_csv %s\n' % str(columns_labels_prediction_csv))
        f_out.write('######################################\n')
        f_out.write('# Machine Learning Algorithm options #\n')
        f_out.write('######################################\n')
        f_out.write('plot_target_predictions %s\n' % str(plot_target_predictions))
        f_out.write('ML %s\n' % str(ML))
        f_out.write('### Cross Validation #################\n')
        f_out.write('CV %s\n' % str(CV))

        if CV=='kf': f_out.write('kfold %s\n' % str(kfold))
        if CV=='last': f_out.write('Nlast %s\n' % str(Nlast))
        if CV=='groups' or CV=='logo':
            f_out.write('acceptor_label_column %s\n' % str(acceptor_label_column))
            f_out.write('groups_acceptor_labels %s\n' % str(groups_acceptor_labels))
            f_out.write('group_test %s\n' % str(group_test))
        f_out.write('### General hyperparameters ##########\n')
        f_out.write('optimize_hyperparams %s\n' % str(optimize_hyperparams))
        f_out.write('gamma_el %s\n' % str(gamma_el))
        f_out.write('gamma_d %s\n' % str(gamma_d))
        f_out.write('gamma_a %s\n' % str(gamma_a))
        f_out.write('gamma_el_lim %s\n' % str(gamma_el_lim))
        f_out.write('gamma_d_lim %s\n' % str(gamma_d_lim))
        f_out.write('gamma_a_lim %s\n' % str(gamma_a_lim))
        f_out.write('weight_RMSE %s\n' % str(weight_RMSE))
        if ML=='kNN':
            f_out.write('### k-Nearest Neighbors ("kNN") ######\n')
            f_out.write('Neighbors %s\n' % str(Neighbors))
            f_out.write('plot_kNN_distances %s\n' % str(plot_kNN_distances))
        elif ML=='KRR':
            f_out.write('### Kernel Ridge Regression ("KRR") ##\n')
            f_out.write('alpha %s\n' % str(alpha))
            f_out.write('alpha_lim %s\n' % str(alpha_lim))
        elif ML=='SVR':
            f_out.write('### Support Vector Regression ("SVR") ########\n')
            f_out.write('C %s\n' % str(C))
            f_out.write('epsilon %s\n' % str(epsilon))
            f_out.write('C_lim %s\n' % str(C_lim))
            f_out.write('epsilon_lim %s\n' % str(epsilon_lim))
        f_out.write('####### END PRINT INPUT OPTIONS ######\n')

    return (ML,Neighbors,alpha,gamma_el,gamma_d,gamma_a,C,epsilon,optimize_hyperparams,alpha_lim,gamma_el_lim,gamma_d_lim,gamma_a_lim,C_lim,epsilon_lim,db_file,elec_descrip,xcols,ycols,Ndata,print_log,log_name,NCPU,f_out,FP_length,weight_RMSE,CV,kfold,plot_target_predictions,plot_kNN_distances,print_progress_every_x_percent,number_elec_descrip,groups_acceptor_labels,group_test,acceptor_label_column,Nlast,prediction_csv_file_name,columns_labels_prediction_csv,predict_unknown)

### Preprocess function to scale data ###
def preprocess_fn(X):
    '''
    Function to preprocess raw data

    Parameters
    ----------
    X: np.array.
        raw data array.

    Returns
    -------
    X: np.array.
        processed data array.
    '''
    X_el=[[] for j in range(Ndata)]
    X_fp_d=[]
    X_fp_a=[]
    elec_descrip_total=0
    for k in elec_descrip:
        elec_descrip_total=elec_descrip_total+k
    for i in range(Ndata):
        X_fp_d.append(X[i][0])
        X_fp_a.append(X[i][1])
        for j in range(2,elec_descrip_total+2):
            X_el[i].append(X[i][j])
    save_X_el = list(X_el[:])
    xscaler = StandardScaler()
    if CV == 'groups' or CV=='logo':
        for i in range(Ndata):
            X_el[i] = X_el[i][1:]
        X_el = xscaler.fit_transform(X_el)
        new_X_el = []
        total_elec_descrip = 0
        for j in range(len(elec_descrip)):
            total_elec_descrip = total_elec_descrip + elec_descrip[j]
        for i in range(Ndata):
            new_list = []
            new_list.append(save_X_el[i][0])
            for j in range(total_elec_descrip-1):
                new_list.append(X_el[i][j])
            new_X_el.append(new_list)
        X_el = list(new_X_el)
    else:
        X_el = xscaler.fit_transform(X_el)
    X = np.c_[ X_el,X_fp_d,X_fp_a]

    return X

### Function to calculate custom metric for electronic and structural properties ###
def custom_distance(X1,X2,gamma_el,gamma_d,gamma_a):
    d_el = []
    distance = 0.0
    # Calculate distances for FP
    elec_descrip_total=0
    for k in elec_descrip:
        elec_descrip_total=elec_descrip_total+k
    if CV=='groups' or CV=='logo': elec_descrip_total=elec_descrip_total-1
    ndesp1 = elec_descrip_total + FP_length
    T_d = ( np.dot(np.transpose(X1[elec_descrip_total:ndesp1]),X2[elec_descrip_total:ndesp1]) ) / ( np.dot(np.transpose(X1[elec_descrip_total:ndesp1]),X1[elec_descrip_total:ndesp1]) + np.dot(np.transpose(X2[elec_descrip_total:ndesp1]),X2[elec_descrip_total:ndesp1]) - np.dot(np.transpose(X1[elec_descrip_total:ndesp1]),X2[elec_descrip_total:ndesp1]) )
    T_a = ( np.dot(np.transpose(X1[ndesp1:]),X2[ndesp1:]) ) / ( np.dot(np.transpose(X1[ndesp1:]),X1[ndesp1:]) + np.dot(np.transpose(X2[ndesp1:]),X2[ndesp1:]) - np.dot(np.transpose(X1[ndesp1:]),X2[ndesp1:]) )
    d_fp_d = 1 - T_d
    d_fp_a = 1 - T_a

    distance = distance + gamma_d*d_fp_d + gamma_a*d_fp_a

    # Calculate distance for electronic properties
    ini = 0
    fin = elec_descrip[0]
    for j in range(len(elec_descrip)):
        d_el.append(0.0)
        for i in range(ini,fin):
            d_el[j] = d_el[j] + (X1[i]-X2[i])**2
        d_el[j] = d_el[j]**(1.0/2.0)

        distance = distance + gamma_el[j] * d_el[j]

        if j < len(elec_descrip)-1:
            ini = elec_descrip[j]
            fin = elec_descrip[j] + elec_descrip[j+1]
    return distance


### ML Function to calculate rmse and r ###
def func_ML(hyperparams,X,y,condition,fixed_hyperparams):
    final_call = fixed_hyperparams[-1]
    # Assign hyperparameters
    if condition=='structure':
        gamma_el = fixed_hyperparams[0]
        gamma_d = hyperparams[0]
        gamma_a = hyperparams[1]

        if ML=='KRR': 
            alpha = hyperparams[2]
        if ML=='SVR':
            C = hyperparams[2]
            epsilon = hyperparams[3]
    elif condition=='electronic':
        gamma_el = []
        counter=0
        for i in range(len(elec_descrip)):
            gamma_el.append(hyperparams[i])
            counter=counter+1
        gamma_d = fixed_hyperparams[0]
        gamma_a = fixed_hyperparams[1]
        if ML=='KRR':
            alpha = hyperparams[i+1]
        if ML=='SVR':
            C = hyperparams[i+1]
            epsilon = hyperparams[i+2]
    elif condition=='structure_and_electronic':
        gamma_el = []
        for i in range(len(elec_descrip)):
            gamma_el.append(hyperparams[i])
        gamma_d = hyperparams[i+1]
        gamma_a = hyperparams[i+2]
        if ML=='KRR':
            alpha = hyperparams[i+3]
        if ML=='SVR':
            C = hyperparams[i+3]
            epsilon = hyperparams[i+4]
    # Build kernel function and assign ML parameters
    if ML=='kNN':
        neighbor_value=fixed_hyperparams[-2]
        ML_algorithm = KNeighborsRegressor(n_neighbors=neighbor_value, weights='distance', metric=custom_distance,metric_params={"gamma_el":gamma_el,"gamma_d":gamma_d,"gamma_a":gamma_a})
    elif ML=='KRR':
        kernel = build_hybrid_kernel(gamma_el=gamma_el,gamma_d=gamma_d,gamma_a=gamma_a)
        ML_algorithm = KernelRidge(alpha=alpha, kernel=kernel)
    elif ML=='SVR':
        ML_algorithm = SVR(kernel=functools.partial(kernel_SVR, gamma_el=gamma_el, gamma_d=gamma_d, gamma_a=gamma_a), C=C, epsilon=epsilon)
    # Initialize values
    y_predicted = []
    y_real = []
    counter = 0
    progress_count = 1
    kNN_distances = []
    kNN_error = []
    test_indeces=[]
    #################################################################
    #################################################################
    # Do LEAVE-ONE-GROUP-OUT (LOGO)
    if CV == 'logo':
        sizes = [0,7,4,5,4,25,20]
        if final_call == False:
            rms_total = 0
            error_logo = 0.0
            for m in range(1,len(groups_acceptor_labels)): # Note: we're ignoring group 0 . UNCOMMENT THIS LINE
            #for m in range(1,4): # Note: we're ignoring group 0  . THIS LINE JUST FOR TESTING AND MAKE IT FASTER
                y_real = []
                y_predicted = []
                print('############')
                print('MAIN M GROUP', m)
                print('############')
                new_rms = 0.0
                for n in range(1,len(groups_acceptor_labels)): # UNCOMMENT THIS LINE
                #for n in range(1,4): # THIS LINE JUST FOR TESTING AND MAKE IT FASTER
                    if m != n:
                        X_test = []
                        y_test = []
                        X_train = []
                        y_train = []
                        print('TEST n', n)
                        for i in range(len(X)):
                            if X[i][0] in groups_acceptor_labels[n]:
                                new_X = np.delete(X[i],0)
                                X_test.append(new_X)
                                y_test.append(y[i].tolist())
                                counter = counter+1
                                if prediction_csv_file_name != None:
                                    test_indeces.append(i)
                                #print(X[i][0])
                            elif X[i][0] in groups_acceptor_labels[m]:
                                pass
                            else:
                                new_X = np.delete(X[i],0)
                                X_train.append(new_X)
                                y_train.append(y[i])
                        y_pred = ML_algorithm.fit(X_train, y_train).predict(X_test)
                        # add predicted values in this LOO to list with total
                        y_predicted.append(y_pred.tolist())
                        y_real.append(y_test)
                        partial_rms = sqrt(mean_squared_error(y_test,y_pred.tolist()))
                        new_rms = new_rms + partial_rms
                        print('partial_rms:', partial_rms)
                        #########################################
                        y_pred = [item for sublist in y_pred.tolist() for item in sublist]
                        y_test = [item for sublist in y_test for item in sublist]
                        logo_weight_A = 1.0          ### Weight A
                        logo_weight_B = 1/((len(sizes)-2)*sizes[n]) ### Weight B
                        sum_sizes_n = 0
                        for i in range(len(sizes)):
                            if i != m: sum_sizes_n = sum_sizes_n + sizes[i]
                        print('sum_sizes_n', sum_sizes_n)
                        logo_weight_C = 1/(sum_sizes_n) ### Weight C
                        print('logo_weights A, B, C:', logo_weight_A, logo_weight_B, logo_weight_C)
                        logo_weight = logo_weight_A
                        error_logo = error_logo +  logo_weight * squared_error(y_pred,y_test)
                        print('y_pred:', y_pred)
                        print('y_test',y_test)
                        print('error_logo',error_logo)
                        #########################################
                print('new_rms:', new_rms, new_rms/5)
                # Put results in a 1D list
                y_real = [item for dummy in y_real for item in dummy ]
                y_predicted = [item for dummy in y_predicted for item in dummy ]
                y_real = [item for dummy in y_real for item in dummy ]
                y_predicted = [item for dummy in y_predicted for item in dummy ]
                print('#### TEST1 ####:', len(y_real))
                print('y_real:', y_real)
                print('y_predicted:', y_predicted)
                # Calculate rmse and r
                if weight_RMSE == 'PCE2':
                    weights = np.square(y_real) / np.linalg.norm(np.square(y_real)) #weights proportional to PCE**2 
                elif weight_RMSE == 'PCE':
                    weights = y_real / np.linalg.norm(y_real) # weights proportional to PCE
                elif weight_RMSE == 'linear':
                    weights = np.ones_like(y_real)
                ### Check if all predicted y values are identical. If so, set r=1 to avoid NaN
                #all_identical = 0
                #for i in range(len(y_predicted)-1):
                    #if y_predicted[i] == y_predicted[i+1]: all_identical = all_identical+1
                #if all_identical == len(y_predicted)-1:
                    #r = np.float64(1.0)
                #else:
                    #r, _ = pearsonr(y_real, y_predicted)
                r, _ = pearsonr(y_real, y_predicted)
                rms  = sqrt(mean_squared_error(y_real, y_predicted,sample_weight=weights))
                #print('TEST rmse, size:', rms, sizes[m])
                rms_total = rms_total + rms*sizes[m]
                y_real_array=np.array(y_real)
                y_predicted_array=np.array(y_predicted)
                #print('TEST m=%i, rmse=%f' %(m, rms))
            rms = rms_total
            print('TOTAL rmse:', rms)
            print('New', ML, 'call:')
            if ML=='kNN': print('kNN, for k = %i' %(neighbor_value))
            print('gamma_el:', gamma_el, 'gamma_d:', gamma_d, 'gamma_a:', gamma_a, 'r:', r,'rmse:',rms,flush=True)
            if ML=='KRR' or ML=='SVR': print('hyperparameters:', ML_algorithm.get_params())
            if print_log==True: 
                f_out.write('New %s call: \n' %(ML))
                f_out.write('gamma_el: %s, gamma_d: %f gamma_a: %f, r: %f, rmse: %f \n' %(str(gamma_el), gamma_d, gamma_a, r, rms))
                if ML=='KRR' or ML=='SVR': f_out.write('hyperparameters: %s \n' %(str(ML_algorithm.get_params())))
                f_out.flush()
        ###########################################
        rms = error_logo
        print('FINAL LOGO ERROR:', rms)
        ###########################################
        if final_call == True:
            y_real = []
            y_predicted = []
            for m in range(1,len(groups_acceptor_labels)):
                X_test = []
                y_test = []
                X_train = []
                y_train = []
                print('TEST m', m)
                counter1=0
                counter2=0
                for i in range(len(X)):
                    if X[i][0] in groups_acceptor_labels[m]:
                        new_X = np.delete(X[i],0)
                        X_test.append(new_X)
                        y_test.append(y[i].tolist())
                        counter = counter+1
                        if prediction_csv_file_name != None:
                            test_indeces.append(i)
                        counter1=counter1+1
                    else:
                        new_X = np.delete(X[i],0)
                        X_train.append(new_X)
                        y_train.append(y[i])
                        counter2=counter2+1
                print('TEST counter1: %i, counter2: %i' %(counter1, counter2))
                print('TEST sizes:', len(X_train), len(X_test))
                #y_pred = ML_algorithm.fit(X_train, y_train).predict(X_test)
                stime = time()
                print('X_train:')
                #print(X_train)
                print('len(X_train)',len(X_train))
                print('y_train:')
                #print(y_train)
                print('len(y_train)',len(y_train))
                ML_algorithm.fit(X_train, y_train)
                print("Time for KRR fitting: %.3f" % (time() - stime))
                stime = time()
                print('X_test:')
                #print(X_test)
                print('len(X_test)',len(X_test))
                y_pred = ML_algorithm.predict(X_test)
                print("Time for GPR prediction: %.3f" % (time() - stime))
                y_predicted.append(y_pred.tolist())
                y_real.append(y_test)
            # Put results in a 1D list
            y_real = [item for dummy in y_real for item in dummy ]
            y_predicted = [item for dummy in y_predicted for item in dummy ]
            y_real = [item for dummy in y_real for item in dummy ]
            y_predicted = [item for dummy in y_predicted for item in dummy ]
            print('y_real:', y_real)
            print('y_predicted:', y_predicted)
            # Plot predictions
            y_real_array=np.array(y_real)
            y_predicted_array=np.array(y_predicted)
            if plot_target_predictions != None: 
                plot_scatter(y_real_array, y_predicted_array,'plot_target_predictions',plot_target_predictions)
            # Calculate rmse and r
            if weight_RMSE == 'PCE2':
                weights = np.square(y_real) / np.linalg.norm(np.square(y_real)) #weights proportional to PCE**2 
            elif weight_RMSE == 'PCE':
                weights = y_real / np.linalg.norm(y_real) # weights proportional to PCE
            elif weight_RMSE == 'linear':
                weights = np.ones_like(y_real)
            ### Check if all predicted y values are identical. If so, set r=1 to avoid NaN
            r, _ = pearsonr(y_real, y_predicted)
            rms  = sqrt(mean_squared_error(y_real, y_predicted,sample_weight=weights))
            y_real_array=np.array(y_real)
            y_predicted_array=np.array(y_predicted)
            print('TEST m=%i, r=%f, rmse=%f' %(m, r, rms))
            print('Final', ML, 'call:')
            if ML=='kNN': print('kNN, for k = %i' %(neighbor_value))
            print('gamma_el:', gamma_el, 'gamma_d:', gamma_d, 'gamma_a:', gamma_a, 'r:', r,'rmse:',rms,flush=True)
            if ML=='KRR' or ML=='SVR': print('hyperparameters:', ML_algorithm.get_params())
            if print_log==True: 
                f_out.write('New %s call: \n' %(ML))
                f_out.write('gamma_el: %s, gamma_d: %f gamma_a: %f, r: %f, rmse: %f \n' %(str(gamma_el), gamma_d, gamma_a, r, rms))
                if ML=='KRR' or ML=='SVR': f_out.write('hyperparameters: %s \n' %(str(ML_algorithm.get_params())))
                f_out.flush()
    #################################################################
    #################################################################
    elif CV !='logo':
    #################################################################
    # Do novel-group validation
        if CV == 'groups':
            X_test = []
            y_test = []
            X_train = []
            y_train = []
            for i in range(len(X)):
                if X[i][0] in groups_acceptor_labels[group_test]:
                    new_X = np.delete(X[i],0)
                    X_test.append(new_X)
                    y_test.append(y[i].tolist())
                    counter = counter+1
                    if prediction_csv_file_name != None:
                        test_indeces.append(i)
                    print ('TEST:', X[i][0])
                else:
                    new_X = np.delete(X[i],0)
                    X_train.append(new_X)
                    y_train.append(y[i])
            for i in range(len(X_train)):
                X_train[i] = X_train[i].tolist()
                y_train[i] = y_train[i].tolist()
            for i in range(len(X_test)):
                X_test[i] = X_test[i].tolist()
            y_train = [item for dummy in y_train for item in dummy ]
            if final_call == True:
                y_pred = ML_algorithm.fit(X_train, y_train).predict(X_test)
                # if kNN: calculate lists with kNN_distances and kNN_error
                if ML=='kNN':
                    provi_kNN_dist=ML_algorithm.kneighbors(X_test)
                    for i in range(len(provi_kNN_dist[0])):
                        kNN_dist=np.mean(provi_kNN_dist[0][i])
                        kNN_distances.append(kNN_dist)
                    error = [sqrt((float(i - j))**2) for i, j in zip(y_pred, y_test)]
                    kNN_error.append(error)
                # add predicted values in this LOO to list with total
                y_predicted.append(y_pred.tolist())
                y_real.append(y_test)
                y_real_array=np.array(y_real)
                y_predicted_array=np.array(y_predicted)
                if plot_target_predictions != None: 
                    plot_scatter(y_real_array, y_predicted_array,'plot_target_predictions',plot_target_predictions)
            elif final_call == False:
                ###################################################
                # For novel-group validation, we need to minimize RMSE of validation group within Train
                X_train=np.array(X_train)
                y_train=np.array(y_train)
                loo = LeaveOneOut()
                validation=loo.split(X_train)
                y_total_pred = []
                y_total_valid = []
                for train_index, valid_index in validation:
                    X_new_train,X_new_valid=X_train[train_index],X_train[valid_index]
                    y_new_train,y_new_valid=y_train[train_index],y_train[valid_index]
                    y_pred = ML_algorithm.fit(X_new_train, y_new_train).predict(X_new_valid)
                    y_total_valid.append(y_new_valid)
                    y_total_pred.append(y_pred)
                ###################################################
                # add predicted values in this LOO to list with total
                y_predicted.append(y_total_pred)
                y_real.append(y_total_valid)
        #################################################################
        # Do regular cross-validation
        elif CV == 'kf' or CV == 'loo':
            # Cross-Validation
            if CV=='kf':
                kf = KFold(n_splits=kfold,shuffle=True)
                validation=kf.split(X)
            elif CV=='loo':
                loo = LeaveOneOut()
                validation=loo.split(X)
            for train_index, test_index in validation:
                # Print progress
                if CV=='loo':
                    if counter == 0: print('Progress %.0f%s' %(0.0, '%'))
                    prog = (counter+1)*100/Ndata
                    if prog >= print_progress_every_x_percent*progress_count:
                        print('Progress %.0f%s' %(prog, '%'), flush=True)
                        progress_count = progress_count + 1
                if CV=='kf':
                    print('Step',counter," / ", kfold,flush=True)
                counter=counter+1
                # assign train and test indeces
                X_train,X_test=X[train_index],X[test_index]
                y_train,y_test=y[train_index],y[test_index]
                # predict y values
                y_pred = ML_algorithm.fit(X_train, y_train.ravel()).predict(X_test)
                # if kNN: calculate lists with kNN_distances and kNN_error
                if ML=='kNN':
                    provi_kNN_dist=ML_algorithm.kneighbors(X_test)
                    for i in range(len(provi_kNN_dist[0])):
                        kNN_dist=np.mean(provi_kNN_dist[0][i])
                        kNN_distances.append(kNN_dist)
                    error = [sqrt((float(i - j))**2) for i, j in zip(y_pred, y_test)]
                    kNN_error.append(error)
                # add predicted values in this LOO to list with total
                y_predicted.append(y_pred.tolist())
                y_real.append(y_test.tolist())
                if prediction_csv_file_name != None:
                    for i in test_index:
                        test_indeces.append(i)
        #################################################################
        # Do last X% validation
        elif CV =='last':
            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=Nlast,random_state=None,shuffle=False)
            # predict y values
            y_pred = ML_algorithm.fit(X_train, y_train.ravel()).predict(X_test)
            # if kNN: calculate lists with kNN_distances and kNN_error
            if ML=='kNN':
                provi_kNN_dist=ML_algorithm.kneighbors(X_test)
                for i in range(len(provi_kNN_dist[0])):
                    kNN_dist=np.mean(provi_kNN_dist[0][i])
                    kNN_distances.append(kNN_dist)
                error = [sqrt((float(i - j))**2) for i, j in zip(y_pred, y_test)]
                kNN_error.append(error)
            # add predicted values in this LOO to list with total
            y_predicted.append(y_pred.tolist())
            y_real.append(y_test.tolist())
            if prediction_csv_file_name != None:
                for i in range(Ndata-Nlast,Ndata):
                    test_indeces.append(i)
        #################################################################
        # Put results in a 1D list
        y_real = [item for dummy in y_real for item in dummy ]
        y_predicted = [item for dummy in y_predicted for item in dummy ]
        y_real = [item for dummy in y_real for item in dummy ]
        if final_call==True: 
            y_predicted = y_predicted
        else: 
            y_predicted = [item for dummy in y_predicted for item in dummy ]
        # Calculate rmse and r
        if weight_RMSE == 'PCE2':
            weights = np.square(y_real) / np.linalg.norm(np.square(y_real)) #weights proportional to PCE**2 
        elif weight_RMSE == 'PCE':
            weights = y_real / np.linalg.norm(y_real) # weights proportional to PCE
        elif weight_RMSE == 'linear':
            weights = np.ones_like(y_real)
        r, _ = pearsonr(y_real, y_predicted)
        rms  = sqrt(mean_squared_error(y_real, y_predicted,sample_weight=weights))
        y_real_array=np.array(y_real)
        y_predicted_array=np.array(y_predicted)
        #######################################
        if prediction_csv_file_name != None:
            df=pd.read_csv(db_file,index_col=0)
            counter_i = 0
            data=[]
            for i in test_indeces:
                df2 = df.loc[i]
                value=[]
                for j in columns_labels_prediction_csv:
                    value.append(df2[j])
                data_row=[i,value,y_real_array[counter_i],y_predicted_array[counter_i]]
                data.append(data_row)
                output_df = pd.DataFrame(data,columns=['Index','Labels','Real_target','Predicted_target'])
                output_df.to_csv (prediction_csv_file_name, index = False, header=True)
                counter_i = counter_i+1
        #######################################
        # Print plots
        if plot_target_predictions != None: 
            plot_scatter(y_real_array, y_predicted_array,'plot_target_predictions',plot_target_predictions)
        if ML=='kNN' and plot_kNN_distances != None: 
            kNN_distances_array=np.array(kNN_distances)
            kNN_error_flat = [item for dummy in kNN_error for item in dummy]
            kNN_error_array=np.array(kNN_error_flat)
            plot_scatter(kNN_distances_array, kNN_error_array, 'plot_kNN_distances', plot_kNN_distances)
        # Print results
        if predict_unknown == True:
            r=0.0
            rms=0.0
        print('New', ML, 'call:')
        if ML=='kNN': print('kNN, for k = %i' %(neighbor_value))
        print('gamma_el:', gamma_el, 'gamma_d:', gamma_d, 'gamma_a:', gamma_a, 'r:', r,'rmse:',rms,flush=True)
        if ML=='KRR' or ML=='SVR': print('hyperparameters:', ML_algorithm.get_params())
        if print_log==True: 
            f_out.write('New %s call: \n' %(ML))
            f_out.write('gamma_el: %s, gamma_d: %f gamma_a: %f, r: %f, rmse: %f \n' %(str(gamma_el), gamma_d, gamma_a, r, rms))
            if ML=='KRR' or ML=='SVR': f_out.write('hyperparameters: %s \n' %(str(ML_algorithm.get_params())))
            f_out.flush()
    return rms 

### SVR kernel function
def kernel_SVR(_x1, _x2, gamma_el, gamma_d, gamma_a):
    # WARNING: FUNCTION WOULD NEED UPDATING FOR CV='groups' and 'logo' [1-OFF ERROR]
    # Initialize kernel values
    K_el   = []
    K_fp_d = 1.0
    K_fp_a = 1.0
    size_matrix1=_x1.shape[0]
    size_matrix2=_x2.shape[0]

    elec_descrip_total=0
    for k in elec_descrip:
        elec_descrip_total=elec_descrip_total+k
    if CV=='groups': elec_descrip_total=elec_descrip_total-1

    ### K_el ###
    ini = 0
    fin = elec_descrip[0]
    K = 1.0
    for k in range(len(elec_descrip)):
        K_el.append(1.0)
        if gamma_el[k] != 0.0:
            # define Xi_el
            Xi_el = [[] for j in range(size_matrix1)]
            for i in range(size_matrix1):
                for j in range(elec_descrip[k]):
                    Xi_el[i].append(_x1[i][j])
            Xi_el = np.array(Xi_el)
            # define Xj_el
            Xj_el = [[] for j in range(size_matrix2)]
            for i in range(size_matrix2):
                for j in range(elec_descrip[k]):
                    Xj_el[i].append(_x2[i][j])
            Xj_el = np.array(Xj_el)
            # calculate K_el
            D_el  = euclidean_distances(Xi_el, Xj_el)
            D2_el = np.square(D_el)
            K_el[k] = np.exp(-gamma_el[k]*D2_el)
            K = K * K_el[k]
            if k < len(elec_descrip)-1:
                ini = elec_descrip[k]
                fin = elec_descrip[k] + elec_descrip[k+1]
    ### K_fp_d ###
    if gamma_d != 0.0:
        # define Xi_fp_d
        Xi_fp_d = [[] for j in range(size_matrix1)]
        for i in range(size_matrix1):
            for j in range(FP_length):
                Xi_fp_d[i].append(_x1[i][j+elec_descrip_total])
        Xi_fp_d = np.array(Xi_fp_d)
        # define Xj_fp_d
        Xj_fp_d = [[] for j in range(size_matrix2)]
        for i in range(size_matrix2):
            for j in range(FP_length):
                Xj_fp_d[i].append(_x2[i][j+elec_descrip_total])
        Xj_fp_d = np.array(Xj_fp_d)
        # calculate K_fp_d
        Xii_d = np.repeat(np.linalg.norm(Xi_fp_d, axis=1, keepdims=True)**2, size_matrix2, axis=1)
        Xjj_d = np.repeat(np.linalg.norm(Xj_fp_d, axis=1, keepdims=True).T**2, size_matrix1, axis=0)
        T_d = np.dot(Xi_fp_d, Xj_fp_d.T) / (Xii_d + Xjj_d - np.dot(Xi_fp_d, Xj_fp_d.T))
        #print('Tanimoto D')
        D_fp_d  = 1 - T_d
        D2_fp_d = np.square(D_fp_d)
        K_fp_d = np.exp(-gamma_d*D2_fp_d)
    ### K_fp_a ###
    if gamma_a != 0.0:
        # define Xi_fp_a
        Xi_fp_a = [[] for j in range(size_matrix1)]
        for i in range(size_matrix1):
            for j in range(FP_length,2*FP_length):
                Xi_fp_a[i].append(_x1[i][j+elec_descrip_total])
        Xi_fp_a = np.array(Xi_fp_a)
        # define Xj_fp_a
        Xj_fp_a = [[] for j in range(size_matrix2)]
        for i in range(size_matrix2):
            for j in range(FP_length,2*FP_length):
                Xj_fp_a[i].append(_x2[i][j+elec_descrip_total])
        Xj_fp_a = np.array(Xj_fp_a)
        # calculate K_fp_a
        Xii_a = np.repeat(np.linalg.norm(Xi_fp_a, axis=1, keepdims=True)**2, size_matrix2, axis=1)
        Xjj_a = np.repeat(np.linalg.norm(Xj_fp_a, axis=1, keepdims=True).T**2, size_matrix1, axis=0)
        T_a = np.dot(Xi_fp_a, Xj_fp_a.T) / (Xii_a + Xjj_a - np.dot(Xi_fp_a, Xj_fp_a.T))
        D_fp_a  = 1 - T_a
        D2_fp_a = np.square(D_fp_a)
        K_fp_a = np.exp(-gamma_a*D2_fp_a)
    # Calculate final kernel
    K = K * K_fp_d * K_fp_a
    #K = np.exp(-gamma_el*np.square(euclidean_distances(Xi_el, Xj_el)) - gamma_d*np.square(1-(np.dot(Xi_fp_d, Xj_fp_d.T) / (Xii_d + Xjj_d - np.dot(Xi_fp_d, Xj_fp_d.T)))) - gamma_a*np.square(1-(np.dot(Xi_fp_a, Xj_fp_a.T) / (Xii_a + Xjj_a - np.dot(Xi_fp_a, Xj_fp_a.T)))))
    #print('K just before return:', K)
    return K

### KRR Kernel function ###
def gaussian_kernel(Xi, Xj, gamma):
    '''
    Function to compute a gaussian kernel.

    Parameters
    ----------
    Xi: np.array.
        training data array
    Xj: np.array.
        training/testing data array.
    gamma: float.
        hyperparameter.

    Returns
    -------
    K: np.array.
        Kernel matrix.
    '''

    m1 = Xi.shape[0]
    m2 = Xi.shape[0]
    X1 = Xi[:,np.newaxis,:]
    X1 = np.repeat(X1, m2, axis=1)
    X2 = Xj[np.newaxis,:,:]
    X2 = np.repeat(X2, m1, axis=0)
    D2 = np.sum((X1 - X2)**2, axis=2)
    K = np.exp(-gamma * D2)

    return K

### KRR Kernel function ###
def tanimoto_kernel(Xi, Xj, gamma):
    '''
    Function to compute a Tanimoto kernel.

    Parameters
    ----------
    Xi: np.array.
        training data array
    Xj: np.array.
        training/testing data array.
    gamma: float.
        hyperparameter.

    Returns
    -------
    K: np.array.
        Kernel matrix.
    '''

    m1 = Xi.shape[0]
    m2 = Xj.shape[0]
    Xii = np.repeat(np.linalg.norm(Xi, axis=1, keepdims=True)**2, m2, axis=1)
    Xjj = np.repeat(np.linalg.norm(Xj, axis=1, keepdims=True).T**2, m1, axis=0)
    T = np.dot(Xi, Xj.T) / (Xii + Xjj - np.dot(Xi, Xj.T))
    K = np.exp(-gamma * (1 - T)**2)

    return K

### KRR Kernel function ###
def build_hybrid_kernel(gamma_el,gamma_d,gamma_a):
    '''
    Parameters
    ----------
    gamma_el: float.
        gaussian kernel hyperparameter.
    gamma_d: float.
        Donor Tanimoto kernel hyperparameter.
    gamma_a: float.
        Acceptor Tanimoto kernel hyperparameter.

    Returns
    -------
    hybrid_kernel: callable.
        function to compute the hybrid gaussian/Tanimoto kernel given values.
    '''

    def hybrid_kernel(_x1, _x2):
        '''
        Function to compute a hybrid gaussian/Tanimoto.

        Parameters
        ----------
        _x1: np.array.
            data point.
        _x2: np.array.
            data point.

        Returns
        -------
        K: np.array.
            Kernel matrix element.
        '''
        # Split electronic data from fingerprints
        elec_descrip_total=0
        for k in elec_descrip:
            elec_descrip_total=elec_descrip_total+k
        if CV=='groups': elec_descrip_total=elec_descrip_total-1
        ndesp1 = elec_descrip_total + FP_length

        # Calculate electronic kernel
        Xi_el = []
        Xj_el = []
        K_el = []
        ini = 0
        if CV=='groups':
            fin = elec_descrip[0]-1
        else:
            fin = elec_descrip[0]
        K = 1.0
        for i in range(len(elec_descrip)):
            Xi_el.append(_x1[ini:fin].reshape(1,-1))
            Xj_el.append(_x2[ini:fin].reshape(1,-1))
            K_el.append(1.0)
            if gamma_el[i] != 0.0: K_el[i] = gaussian_kernel(Xi_el[i], Xj_el[i], gamma_el[i])
            K = K * K_el[i]
            if i < len(elec_descrip)-1:
                ini = elec_descrip[i]
                fin = elec_descrip[i] + elec_descrip[i+1]
        # Calculate structure kernel
        Xi_fp_d = _x1[elec_descrip_total:ndesp1].reshape(1,-1)
        Xi_fp_a = _x1[ndesp1:].reshape(1,-1)
        Xj_fp_d = _x2[elec_descrip_total:ndesp1].reshape(1,-1)
        Xj_fp_a = _x2[ndesp1:].reshape(1,-1)
        K_fp_d = 1.0
        K_fp_a = 1.0
        if gamma_d != 0.0: K_fp_d = tanimoto_kernel(Xi_fp_d, Xj_fp_d, gamma_d)
        if gamma_a != 0.0: K_fp_a = tanimoto_kernel(Xi_fp_a, Xj_fp_a, gamma_a)
        # Element-wise multiplication
        K = K * K_fp_d * K_fp_a
        return K

    return hybrid_kernel

### visualization and calculate pearsonr and spearmanr ###
def plot_scatter(x, y, plot_type, plot_name):
    # general plot options
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 1)
    r, _ = pearsonr(x, y)
    rmse  = sqrt(mean_squared_error(x,y))
    #rho, _ = spearmanr(x, y)
    print('TEST', x, y)
    ma = np.max([x.max(), y.max()]) + 1
    mi = y.min() - 1
    print('ma, mi:', ma, mi)
    ax = plt.subplot(gs[0])
    ax.scatter(x, y, color="b")
    ax.tick_params(axis='both', which='major', direction='in', labelsize=20, pad=10, length=5)
    #######################################################################
    ##### SAVE #####
    # options for plot_target_predictions
    #if plot_type == 'plot_target_predictions':
        #ax.set_xlabel(r"PCE / %", size=24, labelpad=10)
        #ax.set_ylabel(r'PCE$^{%s}$ / %s' %(ML,"%"), size=24, labelpad=10)
        #ax.set_xlim(0, ma)
        #ax.set_ylim(0, ma)
        #ax.set_aspect('equal')
        #ax.plot(np.arange(0, ma + 0.1, 0.1), np.arange(0, ma + 0.1, 0.1), color="k", ls="--")
        #ax.annotate(u'$r$ = %.2f' % r, xy=(0.15,0.85), xycoords='axes fraction', size=22)
    #######################################################################
    #######################################################################
    # options for plot_target_predictions
    if plot_type == 'plot_target_predictions':
        ax.set_xlabel(r"PCE / %", size=24, labelpad=10)
        ax.set_ylabel(r'PCE$^{%s}$ / %s' %(ML,"%"), size=24, labelpad=10)
        #ax.set_xlim(0, ma)
        #ax.set_ylim(0, ma)
        ax.set_xlim(mi, ma)
        ax.set_ylim(mi, ma)
        ax.set_aspect('equal')
        ax.plot(np.arange(0, ma + 0.1, 0.1), np.arange(0, ma + 0.1, 0.1), color="k", ls="--")

        m, b = np.polyfit(x, y, 1)
        ax.plot(x, m*x + b,color="k",linewidth=3)
        print('m:', m, 'b:', b)

        ax.annotate(u'$r$ = %.2f' % r, xy=(0.15,0.85), xycoords='axes fraction', size=18)
        ax.annotate(u'$rmse$ = %.2f' % rmse, xy=(0.15,0.75), xycoords='axes fraction', size=18)
    # options for plot_kNN_distances
    elif plot_type == 'plot_kNN_distances':
        ax.set_xlabel(r"Distance", size=24, labelpad=10)
        ax.set_ylabel(r"RMSE$^{%s}$" %ML, size=24, labelpad=10)
    # extra options in common for all plot types
    xtickmaj = ticker.MaxNLocator(5)
    xtickmin = ticker.AutoMinorLocator(5)
    ytickmaj = ticker.MaxNLocator(5)
    ytickmin = ticker.AutoMinorLocator(5)
    ax.xaxis.set_major_locator(xtickmaj)
    ax.xaxis.set_minor_locator(xtickmin)
    ax.yaxis.set_major_locator(ytickmaj)
    ax.yaxis.set_minor_locator(ytickmin)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(axis='both', which='minor', direction='in', labelsize=20, pad=10, length=2)
    # save plot into corresponding file
    plt.savefig(plot_name,dpi=600,bbox_inches='tight')
    return

### ###
def squared_error(x1,x2):
    if len(x1) != len(x2): # sanity check
        print('ERROR: length of predicted and real values does not have the same length')
        sys.exit()
    sq_error = 0.0
    for i in range(len(x1)):
        sq_error = sq_error + (x1[i]-x2[i])**2
    return sq_error

#### Function to get FP from smiles (not used) ###
#def preprocess_smiles(X):
    #'''
    #Function to preprocess SMILES.

    #Parameters
    #----------
    #df: Pandas DataFrame.

#### Function to get FP from smiles (not used) ###
#def preprocess_smiles(X):
    #'''
    #Function to preprocess SMILES.

    #Parameters
    #----------
    #df: Pandas DataFrame.
        #input data DataFrame.

    #Returns
    #-------
    #df: Pandas DataFrame.
        #preprocessed data DataFrame.
    #'''
    #X["Smiles"]  = X["Smiles"].map(polish_smiles)
    #X["DonorFP"] = X["Smiles"].map(get_fp_bitvect)
    #return X

#### Function to get FP from smiles (not used) ###
#def polish_smiles(smiles, kekule=False):
    #'''
    #Function to polish a SMILES string through the RDKit.

    #Parameters
    #----------
    #smiles: str.
        #SMILES string.
    #kekule: bool (default: False).
        #whether to return Kekule SMILES.

    #Returns
    #-------
    #polished: str.
        #SMILES string.
    #'''
    #mol = Chem.MolFromSmiles(smiles)
    #polished = Chem.MolToSmiles(mol, kekuleSmiles=kekule)
    #return polished

#### Function to get FP from smiles (not used) ###
#def get_fp_bitvect(smiles):
    #'''
    #Function to convert a SMILES string to a Morgan fingerprint through the
    #RDKit.

    #Parameters
    #----------
    #smiles: str.
        #SMILES string.

    #Returns
    #-------
    #fp_arr: np.array.
        #Bit vector corresponding to the Morgan fingerpring.
    #'''
    #mol = Chem.MolFromSmiles(smiles)
    #fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2)
    #fp_arr = np.zeros((1,))
    #DataStructs.ConvertToNumpyArray(fp, fp_arr)
    #return fp_arr

### Function just used to test custom metrics (not used) ###
#def mimic_minkowski(X1,X2):
    #distance=0.0
    #print('X1:',flush=True)
    #print(X1,flush=True)
    #print('X2:',flush=True)
    #print(X2,flush=True)
    #for i in range(len(X1)):
        #distance=distance+(X1[i]-X2[i])**2
    #distance=distance**(1.0/2.0)
    #return distance

##### END OTHER FUNCTIONS ######
################################################################################
################################################################################
################################################################################
### Run main program ###
start = time()
# Read input values
all_rmse_values = []
all_r_values = []
(ML,Neighbors,alpha,gamma_el,gamma_d,gamma_a,C,epsilon,optimize_hyperparams,alpha_lim,gamma_el_lim,gamma_d_lim,gamma_a_lim,C_lim,epsilon_lim,db_file,elec_descrip,xcols,ycols,Ndata,print_log,log_name,NCPU,f_out,FP_length,weight_RMSE,CV,kfold,plot_target_predictions,plot_kNN_distances,print_progress_every_x_percent,number_elec_descrip,groups_acceptor_labels,group_test,acceptor_label_column,Nlast,prediction_csv_file_name,columns_labels_prediction_csv,predict_unknown) = read_initial_values(input_file_name)
# Execute main function
main(alpha,gamma_el,gamma_d,gamma_a,C,epsilon,alpha_lim,gamma_el_lim,gamma_d_lim,gamma_a_lim,C_lim,epsilon_lim)
# Print running time and close log file
time_taken = time()-start
if print_log==True: print('######################################')
if print_log==True: print('Output information can be reviewed in %s' %(log_name))
if print_log==True: print('######################################')
print ('Process took %0.2f seconds' %time_taken,flush=True)
if print_log==True: f_out.write('Process took %0.2f seconds\n' %(time_taken))
if print_log==True: f_out.close()
