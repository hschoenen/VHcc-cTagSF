#!/bin/bash
    
    # choose name of adversarial mode
    ADVERSARIAL_MODEL_NAME="fgsm-0_125"
    WM="_DeepJet_Run2_COMPLETE" # do nominal and a adversarial model at the same time
#   WM="_DeepJet_Run2_COMPARE" # does nominal and adversarial in one go
#   WM="_DeepJet_Run2_nominal"
#   WM="_DeepJet_Run2_adversarial_eps0p01"
#   WM="_multi_nominal_5,15,30"
#   WM="_DeepJet_Run2_COMPARESHARPNESSAWARE" # does nominal and adversarial in one go
#   WM="_ParT_COMPARE"
    
    echo $HOSTNAME
    echo "Who am I?"
    whoami
    echo "hostname"
    hostname -f
    echo "tokens"
    tokens
    echo "klist -f"
    klist -f
    echo "try aklog"
    aklog
    echo "and again: tokens"
    tokens
    echo "as well as: klist -f"
    klist -f
    echo "   Current dir:"
    pwd
    echo "CONDOR_SCRATCH_DIR = $_CONDOR_SCRATCH_DIR"
    ls -lh
    
    # if only checking with nominal samples, targets are not required (will not apply FGSM attack, i.e. don't need truth)
    TARGETSNECESSARY="no"
    STOREINTERESTINGINPUTS="no"

    # edited to personal directory
    export OUTPUTDIR=/nfs/dust/cms/user/hschonen/DataMC/${ADVERSARIAL_MODEL_NAME}/2017_$4${WM}/
	OUTPUTNAME=outTree.root

	CONDOR_CLUSTER_ID=$1
	CONDOR_PROCESS_ID=$2
	INPFILE=$3

        if  [[ $4 == "Wc" ]]; then
             PYFILE="WcSelection_new.py"
        elif  [[ $4 == "DY" ]]; then
             PYFILE="DYJetSelection_new.py"
        elif  [[ $4 == "TT" ]]; then
             PYFILE="TTbSelection.py"
        elif  [[ $4 == "TTNoMu" ]]; then
             PYFILE="TTbNoMuSelection.py"
        elif  [[ $4 == "WcNoMu" ]]; then
             PYFILE="WcNoMuSelection.py"
        fi
        
        
        export PATH=/afs/desy.de/common/passwd:/usr/local/bin:/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/sbin:/cvmfs/grid.cern.ch/emi3ui-latest/bin:/cvmfs/grid.cern.ch/emi3ui-latest/sbin:/cvmfs/grid.cern.ch/emi3ui-latest/usr/bin:/cvmfs/grid.cern.ch/emi3ui-latest/usr/sbin:$PATH
        echo "    echo PATH:"
        echo $PATH
        echo "    arguments: " $1 $2 $3
        echo "    username and group"
        id -n -u
        id -n -g
        echo "    pwd"
        pwd
        echo "copy scripts to scratch"
        # edited to personal directories
        cp -r /afs/desy.de/user/h/hschonen/aisafety/VHcc-cTagSF/Analyzer/${PYFILE} /afs/desy.de/user/h/hschonen/aisafety/VHcc-cTagSF/Analyzer/condorDESY/customDeepJetTaggerInference.py /afs/desy.de/user/h/hschonen/aisafety/VHcc-cTagSF/Analyzer/condorDESY/attacks.py /afs/desy.de/user/h/hschonen/aisafety/VHcc-cTagSF/Analyzer/condorDESY/definitions.py /afs/desy.de/user/h/hschonen/aisafety/VHcc-cTagSF/Analyzer/condorDESY/focal_loss.py /afs/desy.de/user/h/hschonen/aisafety/VHcc-cTagSF/Analyzer/condorDESY/pytorch_deepjet.py /afs/desy.de/user/h/hschonen/aisafety/VHcc-cTagSF/Analyzer/condorDESY/pytorch_deepjet_transformer.py /afs/desy.de/user/h/hschonen/aisafety/VHcc-cTagSF/Analyzer/condorDESY/pytorch_deepjet_run2.py /afs/desy.de/user/h/hschonen/aisafety/VHcc-cTagSF/Analyzer/condorDESY/helpers_advertorch.py /afs/desy.de/user/h/hschonen/aisafety/VHcc-cTagSF/Analyzer/condorDESY/attacks_ParT.py /afs/desy.de/user/h/hschonen/aisafety/VHcc-cTagSF/Analyzer/condorDESY/definitions_ParT.py /afs/desy.de/user/h/hschonen/aisafety/VHcc-cTagSF/Analyzer/condorDESY/ParT.py /afs/desy.de/user/h/hschonen/aisafety/VHcc-cTagSF/Analyzer/nuSolutions.py /afs/desy.de/user/h/hschonen/aisafety/VHcc-cTagSF/Analyzer/scalefactors* $_CONDOR_SCRATCH_DIR

        echo "setting up the grid commands"
        source /cvmfs/grid.cern.ch/centos7-umd4-ui-4_200423/etc/profile.d/setup-c7-ui-example.sh
        echo "    which xrdcp"
        which xrdcp
        
        
        echo "set up proxy"
        # edited to personal voms
        if [ -f "x509up_u38609" ]; then
           export X509_USER_PROXY=x509up_u38609
        fi
        echo "    voms-proxy-info -all"
        voms-proxy-info -all
        
        INPPREFIX="root://grid-cms-xrootd.physik.rwth-aachen.de:1094/"
        INPPREFIX2="root://dcache-cms-xrootd.desy.de:1094/"
        echo "copy actual input file"
        xrdcp ${INPPREFIX}${INPFILE} ./infile.root
        ## Did we find IP address? Use exit status of the grep command ##
        if [ $? -eq 0 ]
        then
          echo "Success: copied file from Aachen T2."
        else
          echo "Failure: could not find file on Aachen T2. Use DESY T2 instead."
          INPPREFIX=${INPPREFIX2}
          xrdcp ${INPPREFIX}${INPFILE} ./infile.root
          echo "Success: copied file from DESY T2."
        fi
        
        echo "    echo PATH:"
        echo $PATH
        echo "    content of pwd"
        ls -lh
        
        echo "    which python3"
        which python3
        
        ENVNAME=deepjet-env
        ENVDIR=$ENVNAME
        export PATH
        echo "    echo PATH:"
        echo $PATH
        mkdir $ENVDIR
        echo "setup conda"
        # edited to personal directory
        tar -xzf /nfs/dust/cms/user/hschonen/${ENVNAME}.tar.gz -C ${ENVDIR}
        source ${ENVNAME}/bin/activate
        echo "    which python3"
        which python3
        conda-unpack
        which python3
        echo "    echo PATH:"
        echo $PATH
        echo "start with custom tagger"
        
        # execute customDeepJetTaggerInference.py
        python3 customDeepJetTaggerInference.py ${INPPREFIX}${INPFILE} ${WM} ${OUTPUTDIR} ${TARGETSNECESSARY} ${STOREINTERESTINGINPUTS}
        
        echo "setting up the environment (CMSSW)"
        cd /cvmfs/cms.cern.ch/slc6_amd64_gcc630/cms/cmssw/CMSSW_10_2_0_pre6/src
        source /cvmfs/cms.cern.ch/cmsset_default.sh
        which xrdcp
        eval `scramv1 runtime -sh`

        echo "    echo PATH:"
        echo $PATH
        source /cvmfs/grid.cern.ch/etc/profile.d/setup-cvmfs-ui.sh
        echo "changing to scratch dir again"
        cd $_CONDOR_SCRATCH_DIR
        echo "    pwd and ls"
        pwd
        ls -ls
        
        # execute WcSelection_new.py, DYJetSelection_new.py or TTbSelection.py"
        echo "running python script (Analyzer)"
        python ${PYFILE} ${INPFILE}
        
        rc=$?
        if [[ $rc == 99 ]]               
        then  
            echo "Output file already exists. Aborting job with exit code 0." $rc  
            exit 0       
        fi
        if [[ $rc != 0 ]]
        then
            echo "got exit code from python code: " $rc
            exit $rc
        fi
        echo "done running, now copying output to DUST"

        echo "copying output"
        SAMPNAME=$(bash dirName.sh)
        FLNAME=$(bash flName.sh)
        mkdir -p ${OUTPUTDIR}${SAMPNAME}
        until cp -vf ${OUTPUTNAME} ${OUTPUTDIR}${SAMPNAME}"/outTree_"${FLNAME}".root"; do
            echo "copying output failed. Retrying..."
            sleep 60
        done
        echo "copied output successfully"
        
        echo "Clean up after yourself"
        rm x509up_u38609
        rm *.root *.py *.cc *.npy
        rm -r ./${ENVDIR} ./scalefactors*

        echo "all done!"
