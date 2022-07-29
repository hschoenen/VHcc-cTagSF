#!/bin/bash
    
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
    
    # OLD! Custom ~DeepCSV
    # adjust the weighting method, look up the definitions in customTaggerInference.py
#    WM="_notflat_200_gamma25.0_alphaNone"  # example for single weighting method alone (using raw/Noise/FGSM inputs)
    WM="_multi_basic_5,10,100"  # example for three epochs of one weighting method

    # NEW! Custom ~DeepJet
 #   WM="_nominal_best"
#    WM="_multi_nominal_5,15,30"

    # OLD!
#	export OUTPUTDIR=/nfs/dust/cms/user/spmondal/ctag_condor/210225_2017_SemiT_$4/
#    export OUTPUTDIR=/nfs/dust/cms/user/anstein/ctag_condor/210402_2017_$4_minimal/
#    export OUTPUTDIR=/nfs/dust/cms/user/anstein/ctag_condor/210708_2017_$4${WM}/
#    export OUTPUTDIR=/nfs/dust/cms/user/anstein/ctag_condor/210714_2017_$4${WM}/

    # NEW!
    export OUTPUTDIR=/nfs/dust/cms/user/anstein/ctag_condor/220729_2017_$4${WM}/
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
        
        
        #which xrdcp
        
        export PATH=/afs/desy.de/common/passwd:/usr/local/bin:/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/sbin:/cvmfs/grid.cern.ch/emi3ui-latest/bin:/cvmfs/grid.cern.ch/emi3ui-latest/sbin:/cvmfs/grid.cern.ch/emi3ui-latest/usr/bin:/cvmfs/grid.cern.ch/emi3ui-latest/usr/sbin:$PATH
        echo "    echo PATH:"
        echo $PATH
        echo "    arguments: " $1 $2 $3
        echo "    username and group"
        id -n -u
        id -n -g
        echo "    pwd"
        pwd
        #echo "creating tempdir and copy"
        #tmp_dir=$(mktemp -d)
        #cp -r ../${PYFILE} customTaggerInference.py ../nuSolutions.py ../scalefactors* $tmp_dir
        echo "copy scripts to scratch"
        # Note: for new tagger, the inference will run on already preprocessed samples
        cp -r /afs/desy.de/user/a/anstein/private/aisafety/SF/VHcc-cTagSF/Analyzer/${PYFILE} /afs/desy.de/user/a/anstein/private/aisafety/SF/VHcc-cTagSF/Analyzer/condorDESY/customTaggerInference.py /afs/desy.de/user/a/anstein/private/aisafety/SF/VHcc-cTagSF/Analyzer/condorDESY/customDeepJetTaggerInference.py /afs/desy.de/user/a/anstein/private/aisafety/SF/VHcc-cTagSF/Analyzer/condorDESY/focal_loss.py /afs/desy.de/user/a/anstein/private/aisafety/SF/VHcc-cTagSF/Analyzer/nuSolutions.py /afs/desy.de/user/a/anstein/private/aisafety/SF/VHcc-cTagSF/Analyzer/scalefactors* $_CONDOR_SCRATCH_DIR
        #echo "changing to tempdir (first time)"
        #cd $tmp_dir
                
        
        #echo "setting up the environment"
        #cd /cvmfs/cms.cern.ch/slc6_amd64_gcc630/cms/cmssw/CMSSW_10_2_0_pre6/src
        #source /cvmfs/cms.cern.ch/cmsset_default.sh
        #which xrdcp
        #eval `scramv1 runtime -sh`
        
        echo "setting up the grid commands"
        source /cvmfs/grid.cern.ch/centos7-umd4-ui-4_200423/etc/profile.d/setup-c7-ui-example.sh
        #source /cvmfs/cms.cern.ch/common/crab-setup.sh prod
        #source /cvmfs/cms.cern.ch/cmsset_default.sh
        #cd /cvmfs/cms.cern.ch/slc7_amd64_gcc900/cms/cmssw/CMSSW_11_3_0_pre3/src
        #eval "$(scramv1 runtime -sh)"
        echo "    which xrdcp"
        which xrdcp
        
        #echo "changing to scratch"
        #cd $_CONDOR_SCRATCH_DIR
        echo "set up proxy"
        if [ -f "x509up_u38320" ]; then
           export X509_USER_PROXY=x509up_u38320
        fi
        echo "    voms-proxy-info -all"
        voms-proxy-info -all
        #echo "test if copy with proxy works"
        #xrdcp -d 1 -f root://grid-cms-xrootd.physik.rwth-aachen.de:1094//store/user/anovak/PFNano/106X_v2_17/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIIFall17PFNanoAODv2-PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v1PFNanoV2/210101_174326/0001/nano_mc2017_1-1708.root /dev/null
        INPPREFIX="root://grid-cms-xrootd.physik.rwth-aachen.de:1094/"
        #INPPREFIX=""
        echo "copy actual input file"
        xrdcp ${INPPREFIX}${INPFILE} ./infile.root
        
        echo "    echo PATH:"
        echo $PATH
        #source /cvmfs/grid.cern.ch/etc/profile.d/setup-cvmfs-ui.sh
        #echo "changing to tempdir (second time)"
        #cd $tmp_dir
        #pwd
        echo "    content of pwd"
        ls -lh
        
        
        
        
        
        echo "    which python3"
        which python3
        #eval `scram unsetenv -sh`
        ENVNAME=my-env
 #       ENVNAME=deepjet-env
        ENVDIR=$ENVNAME

        export PATH
        echo "    echo PATH:"
        echo $PATH
        mkdir $ENVDIR
        echo "setup conda"
        tar -xzf /nfs/dust/cms/user/anstein/${ENVNAME}.tar.gz -C ${ENVDIR}
        #./${ENVDIR}/bin/activate
        source ${ENVNAME}/bin/activate
        echo "    which python3"
        which python3
        conda-unpack
        which python3
        echo "    echo PATH:"
        echo $PATH
        echo "start with custom tagger"
        python3 customTaggerInference.py ${INPPREFIX}${INPFILE} ${WM} ${OUTPUTDIR}
 #       python3 customDeepJetTaggerInference.py ${INPPREFIX}${INPFILE} ${WM} ${OUTPUTDIR}
        
        
        
        
        
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
        
        
        #path_without_pref=${INPFILE#"$INPPREFIX"}
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

#        python -c "import sys,ROOT; f=ROOT.TFile(''); sys.exit(int(f.IsZombie() and 99))"
#        rc=$?
#        if [[ $rc != 0 ]]
#        then
#            echo "copy failed (either bad output from cp or file is Zombie)"
#            exit $rc
#        fi

        #echo "delete tmp dir"
        #cd $TMP
        #rm -r $tmp_dir
        
        echo "Clean up after yourself"
        rm x509up_u38320
        #rm *.root *.pem *.pcm *.so *.tar.gz *.py *.cc *.h *.npy
        rm *.root *.py *.cc *.npy
        rm -r ./${ENVDIR} ./scalefactors*

        echo "all done!"
