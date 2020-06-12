# NonFullereneAcceptorPrediction
This is an almost exact copy of 'MachineLearning_AcceptorDonor'. There is one extra option: CV='groups'. This still needs to be tested and as of now requires some precautions:

- "ACCEPTORNo" needs to be added as first descriptor in xcols_elec0 (this is just used to select test set and is not actually used in ML). Remember to remove this descriptor if using CV='loo' or CV='kf'

- Line 481-491 of the code can be editted to select what acceptor labels correspond to each group, as well as which acceptor group is used as test group
