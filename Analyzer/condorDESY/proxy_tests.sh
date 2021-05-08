#!/bin/bash

    echo $HOSTNAME
    echo "Who am I?"
    whoami

    echo "   Current dir:"
    pwd
    echo "CONDOR_SCRATCH_DIR = $_CONDOR_SCRATCH_DIR"
    ls -lh

    echo "all arguments:" $@

    echo "setting up the environment"
    source /cvmfs/grid.cern.ch/centos7-umd4-ui-4_200423/etc/profile.d/setup-c7-ui-example.sh
    source /cvmfs/cms.cern.ch/common/crab-setup.sh prod
    source /cvmfs/cms.cern.ch/cmsset_default.sh
    cd /cvmfs/cms.cern.ch/slc7_amd64_gcc900/cms/cmssw/CMSSW_11_3_0_pre3/src
    eval "$(scramv1 runtime -sh)"
    
    echo "successfully set up the enviroment"

    echo "changing to scratch"
    cd $_CONDOR_SCRATCH_DIR

    if [ -f "x509up_u38320" ]; then
       export X509_USER_PROXY=x509up_u38320
    fi
    voms-proxy-info
    xrdcp -d 1 -f root://grid-cms-xrootd.physik.rwth-aachen.de:1094//store/user/anovak/PFNano/106X_v2_17/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIIFall17PFNanoAODv2-PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v1PFNanoV2/210101_174326/0001/nano_mc2017_1-1708.root /dev/null
    echo "PATH: $PATH"
    echo "LD: $LD_LIBRARY_PATH"
    echo "PY: $PYTHONPATH"

    

    ls -lh 

    echo "which Python"
    which python

    echo "which Root"
    which root

    #which gcc

    echo "Now copying output"
    
    eval `scram unsetenv -sh`
    # replace env-name on the right hand side of this line with the name of your conda environment
    ENVNAME=my-env
    # if you need the environment directory to be named something other than the environment name, change this line
    ENVDIR=$ENVNAME

    # these lines handle setting up the environment; you shouldn't have to modify them
    export PATH
    mkdir $ENVDIR
    tar -xzf /nfs/dust/cms/user/anstein/$ENVNAME.tar.gz -C $ENVDIR
    . $ENVDIR/bin/activate
    python3 -c 'import torch; print(); print(torch.__version__); print()'
    #eval `scram unsetenv -sh`; gfal-copy -p myFile.root srm://grid-srm.physik.rwth-aachen.de:8443/srm/managerv2?SFN=/pnfs/physik.rwth-aachen.de/cms/store/user/userbname/myFile.root

    #rc=$?
    #if [[ $rc != 0 ]]
    #then
    #   echo "Copy failed!"
    #   exit $rc
    #fi

    echo "Clean up after yourself"
    rm *.root *.pem *.pcm *.so *.tar.gz *.py *.cc *.h
    rm -r ./aux ./cfg ./plugins ./python

    ls -lh

    echo "all done!"
    
