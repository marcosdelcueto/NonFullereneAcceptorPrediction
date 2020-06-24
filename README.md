# NonFullereneAcceptorPrediction
This is an almost exact copy of 'MachineLearning_AcceptorDonor'. There is one extra option: CV='groups', along with the variables: 'groups\_acceptor\_labels' and 'group_test' to select the acceptor labels corresponding to each group, as well as select which of those groups will be use as a test group. This still needs to be tested and as of now requires some precautions:

- "ACCEPTORNo" needs to be added as first descriptor in xcols_elec0 (this is just used to select test set and is not actually used in ML). Remember to remove this descriptor if using CV='loo' or CV='kf'
