# Non-Fullerene Acceptor Prediction
This repository contains the database and code for **Can machine learning methods predict non-fullerene acceptors for organic solar cells with completely new chemistries?** by _Z-W Zhao, M del Cueto and A Troisi_

Code is based in our previous [MLPhotovoltaics](https://github.com/marcosdelcueto/MachineLearning_AcceptorDonor) program, with the main addition of performing _novel-group validation_.

Options for this novel-group validation are controlled in input file:
- **CV='groups'**
- **acceptor\_label\_column**: allows to set the name of the column that contains the acceptor labels
- **groups\_acceptor\_labels**: allows to assign pairs whose acceptor has a specific label to a group
- **group\_test**: select which of the previous groups is used as test. The rest will be used as training

---

## Prerequisites
The necessary packages (with the tested versions with Python 3.8.5) are specified in the file requirements.txt. These packages can be installed with pip:
```
pip3 install -r requirements.txt
```

## Usage
All input parameters are specified in file: **inputNonFullereneAcceptorPrediction.inp**. Input options in this file are separated in different groups:

- **Parallelization**: only relevant when trying to use differential evolution algorithm to optimize hyperparameters
- **Verbose options**: allows some flexibility for how much information to print to standard output and log file
- **Data base options**: allows to select how many donor/acceptor pairs are used, as well as which descriptors are considered
- **Output prediction csv**: allows to print the actual and predicted target properties values of the test points
- **Machine Learning Algorithm options**: allows to select what ML algorithm is used (whether kNN, KRR or SVR), as well as cross validation method, hyperparameters etc.

To execute program, make sure that you have all necessary python packages installed, and that all necessary files are present: the database (**database.csv**), input file (**inputNonFullereneAcceptorPrediction.inp**) and program (**NonFullereneAcceptorPrediction.py**). Finally, simply run:

```
./MLPhotovoltaics.py
```

## Example inputs
The rmse values with different validation and descriptors, shown in Table 1 of the manuscript, can be reproduced using the hyperparameters shown in Table S3 and Table S4 of the Supporting Information.

We provide example inputs for kNN and KRR to calculate the rmse with leave-one-out (loo) cross-validation and novel-group validation for Group 5 and Group 6, in the directories:

- kNN_loo
- kNN_group5
- kNN_group6
- KRR_loo
- KRR_group5
- KRR_group6

---

### Contributors

[Zhi-Wen Zhao](https://github.com/amiswen) and [Marcos del Cueto](https://github.com/marcosdelcueto)
