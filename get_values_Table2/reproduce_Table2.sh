#!/bin/bash

# Clean intermediate files in case they were created in a previous run
rm -f pred_G?.csv intermediate1.csv intermediate2.csv real_pred.csv log_NonFullereneAcceptorPrediction.log

for file in `echo "LOO_extrapolation LOGO_extrapolation"`
do
    for i in `seq 1 5`
    do
		 # Set up name of input file
		 input="input_files/input_${file}_phys_G${i}.inp"
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
    python3 get_binary_metrics.py
    echo "###################################################"
	 # Clean intermediate files
    rm -f pred_G?.csv intermediate1.csv intermediate2.csv real_pred.csv log_NonFullereneAcceptorPrediction.log
done
