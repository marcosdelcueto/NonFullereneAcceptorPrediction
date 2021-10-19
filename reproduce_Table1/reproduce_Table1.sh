#!/bin/bash

# Clean intermediate files in case they were created in a previous run
rm -f pred_G?.csv intermediate1.csv intermediate2.csv real_pred.csv log_NonFullereneAcceptorPrediction.log

##### LOO-interpolation #####
# Doing LOO-interpolation may take more than 1h, so by default do not do it
# If one wants to also do LOO-interpolation, set do_LOO_interpolation=True
do_LOO_interpolation=False
if [[ ${do_LOO_interpolation} = True ]]; then
    for file in `echo "LOO_interpolation_chem LOO_interpolation_phys"`
    do
    	 # Set up name of input file
    	 input="input_files/input_${file}.inp"
    	 # Run prediction code
    	 python3 NonFullereneAcceptorPrediction.py ${input} > /dev/null
    	 # Handle output and put it in format 'real_value,pred_value'
    	 mv pred_G1.csv > intermediate1.csv
    	 grep -v "Index" intermediate1.csv > intermediate2.csv
    	 cut -d ',' -f 3,4 intermediate2.csv > real_pred.csv
    	 echo "###################################################"
    	 echo "${file}"
    	 # Run code to calculate rmse and r of real,pred.csv
    	 python3 get_r_rmse.py
    	 echo "###################################################"
    	 # Clean intermediate files
    	 rm -f pred_G?.csv intermediate1.csv intermediate2.csv real_pred.csv log_NonFullereneAcceptorPrediction.log
    done
fi

##### LOO-extrapolation and LOGO-extrapolation #####
for file in `echo "LOO_extrapolation_chem LOO_extrapolation_phys LOGO_extrapolation_chem LOGO_extrapolation_phys"`
do
    for i in `seq 1 5`
    do
		  # Set up name of input file
	     input="input_files/input_${file}_G${i}.inp"
		  # Run prediction code
		  python3 NonFullereneAcceptorPrediction.py ${input} > /dev/null
    done
    # Handle output and put it in format 'real_value,pred_value'
    cat  pred_G?.csv > intermediate1.csv
    grep -v "Index" intermediate1.csv > intermediate2.csv
    cut -d ',' -f 3,4 intermediate2.csv > real_pred.csv
    echo "###################################################"
	 echo "${file}"
	 # Run code to calculate rmse and r of real,pred.csv
    python3 get_r_rmse.py
    echo "###################################################"
	 # Clean intermediate files
    rm -f pred_G?.csv intermediate1.csv intermediate2.csv real_pred.csv log_NonFullereneAcceptorPrediction.log
 done