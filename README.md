# Non-Fullerene Acceptor Prediction
This repository contains the database and code for **Training Machine Learning Models to Predict Compounds with Completely New Chemistries: Application to Non-Fullerene Acceptors** by _Z-W Zhao_, _M del Cueto_ and _A Troisi_

Code is based in our previous [MLPhotovoltaics](https://github.com/marcosdelcueto/MachineLearning_AcceptorDonor) program, with the main addition of performing _LOO-extrapolation_ and _LOGO-extrapolation_. More details on these method can be found in manuscript. These two cross-validations can be controlled with the following input keywords:

- **CV='groups'** (LOO-extrapolation) OR **CV='logo'** (LOGO-extrapolation)
- **acceptor\_label\_column**: allows to set the name of the column that contains the acceptor labels
- **groups\_acceptor\_labels**: allows to assign pairs whose acceptor has a specific label to a group
- **group\_test**: select which of the previous groups is used as test. The rest will be used as training

---

## Prerequisites
The necessary packages (with the tested versions with Python 3.8.5) are specified in the file requirements.txt. These packages can be installed with pip:
```
pip3 install -r requirements.txt
```
---

## Usage
All input parameters are specified in file: **inputNonFullereneAcceptorPrediction.inp**. Input options in this file are separated in different groups:

- **Parallelization**: only relevant when trying to use differential evolution algorithm to optimize hyperparameters
- **Verbose options**: allows some flexibility for how much information to print to standard output and log file
- **Data base options**: allows to select how many donor/acceptor pairs are used, as well as which descriptors are considered
- **Output prediction csv**: allows to print the actual and predicted target properties values of the test points
- **Machine Learning Algorithm options**: allows to select what ML algorithm is used, as well as cross validation method, hyperparameters etc.

To execute the program, make sure that you have all necessary python packages installed, and that all necessary files are present: the database (**database.csv**), input file (**inputNonFullereneAcceptorPrediction.inp**) and program (**NonFullereneAcceptorPrediction.py**). Finally, simply run:

```
python NonFullereneAcceptorPrediction.py
```

---

## Examples
The folders reproduce_Table1 and reproduce_Table2 contain all necessary files to reproduce the main results of the manuscript. To do this, simply execute the bash scripts: reproduce_Table1.sh and reproduce_Table2.sh inside their respective folders.

---

## License
**Authors**: [Zhi-Wen Zhao](https://github.com/amiswen), [Marcos del Cueto](https://github.com/marcosdelcueto) and Alessandro Troisi

Licensed under the [MIT License](LICENSE.md) 
