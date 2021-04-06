#!/bin/bash
#    source /afs/desy.de/user/a/anstein/miniconda3/etc/profile.d/conda.sh
# source /afs/desy.de/user/a/anstein/miniconda3/bin/activate
# cd /afs/desy.de/user/a/anstein/private/aisafety/SF/VHcc-cTagSF/Analyzer/condorDESY
# echo "arguments: " $1 $2 $3 $4
    conda activate my-env
    
#    module load conda
#    source activate my-env
    
    python3 dummy.py
    
    xrdcp -d 1 -f root://grid-cms-xrootd.physik.rwth-aachen.de:1094//store/user/anovak/PFNano/106X_v2_17/GluGluHToBB_M-125_13TeV_powheg_MINLO_NNLOPS_pythia8/RunIIFall17PFNanoAODv2-PU2017_12Apr2018_94X_mc2017_realistic_v14-v1PFNanoV2/210101_174027/0000/nano_mc2017_3-16.root /dev/null