# NonFullereneAcceptorPrediction
This is an almost exact copy of 'MachineLearning_AcceptorDonor'. There are some extra options than can be specified in the input file to make a train/test split by groups:

- CV='groups'
- acceptor_label_column: allows to set the name of the column that contains the acceptor labels
- groups_acceptor_labels: allows to assign pairs whose acceptor has a specific label to a group
- group_test: select which of the previous groups is used as test. The rest will be used as training
