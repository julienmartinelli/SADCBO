#!/bin/bash

# SYNTAX for EXP: funcname_reldim-zdimidx1&zdimidx2...-zirrelevant
# for instance, the hartmann function with 4 dimensions being relevant,
# dimensions 0 and 3 being the designs, 1 and 2 being z relevant,
# and 3 additionnal irrelevant dimensions
# is written Hartmann_4-0&3-3

#EXP="Hartmann_6-1&4&5-4 Hartmann_4-0&3-3 Ackley_5-0&1-8 EggHolder_2-0-4"
EXP="Hartmann_4-0&3-3"
#SADCBO hyperparameters: eta=0.8, Q=10, gamma=(1-0.2)=0.8
METHODS="Sensitivitycontextual-0.8-10-0.2"
KERNELS="RBF"
ACQFS="UCB0.2"
N_REP=10
BUDGET=40
SEED=20
RESULTFOLDER="test"
N_INIT=5 # number of initial points will be N_INIT * RELDIM

mkdir -p $RESULTFOLDER
echo -e "FUNCTIONS=$EXP\nMETHODS=$METHODS\nKERNELS=$KERNELS\nACQFS=$ACQFS\nN_REPS=$N_REP\nN_INIT=$N_INIT\nBUDGET=$BUDGET\nSEED=$SEED" > "$RESULTFOLDER/config_test.txt"
python3.11 main.py -n $N_REP -ni $N_INIT -b $BUDGET -k $KERNELS -a $ACQFS -e $EXP -se $SEED -m $METHODS -s $RESULTFOLDER