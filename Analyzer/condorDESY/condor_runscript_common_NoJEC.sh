#	export OUTPUTDIR=/nfs/dust/cms/user/spmondal/ctag_condor/210225_2017_SemiT_$4/
#    export OUTPUTDIR=/nfs/dust/cms/user/anstein/ctag_condor/210402_2017_$4_minimal/
    export OUTPUTDIR=/nfs/dust/cms/user/anstein/ctag_condor/210430_2017_$4_PFNano_test/
	OUTPUTNAME=outTree.root

	CONDOR_CLUSTER_ID=$1
	CONDOR_PROCESS_ID=$2
	INPFILE=$3

        if  [[ $4 == "Wc" ]]; then
             PYFILE="WcSelection_new.py"
        elif  [[ $4 == "DY" ]]; then
             PYFILE="DYJetSelection.py"
        elif  [[ $4 == "TT" ]]; then
             PYFILE="TTbSelection.py"
	elif  [[ $4 == "TTNoMu" ]]; then
             PYFILE="TTbNoMuSelection.py"
        elif  [[ $4 == "WcNoMu" ]]; then
             PYFILE="WcNoMuSelection.py"
        fi
        
        
        #which xrdcp
        
        export PATH=/afs/desy.de/common/passwd:/usr/local/bin:/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/sbin:/cvmfs/grid.cern.ch/emi3ui-latest/bin:/cvmfs/grid.cern.ch/emi3ui-latest/sbin:/cvmfs/grid.cern.ch/emi3ui-latest/usr/bin:/cvmfs/grid.cern.ch/emi3ui-latest/usr/sbin:$PATH
        echo "echo PATH:"
        echo $PATH
        echo "arguments: " $1 $2 $3
        echo "username and group"
        id -n -u
        id -n -g
        
        echo "creating tempdir and copy"
        tmp_dir=$(mktemp -d)
        cp -r ../${PYFILE} ../nuSolutions.py ../scalefactors* $tmp_dir
        echo "changing to tempdir (first time)"
        cd $tmp_dir
        
        which xrdcp
        
        
        echo "setting up the environment"
        cd /cvmfs/cms.cern.ch/slc6_amd64_gcc630/cms/cmssw/CMSSW_10_2_0_pre6/src
        source /cvmfs/cms.cern.ch/cmsset_default.sh
        which xrdcp
        eval `scramv1 runtime -sh`
        which xrdcp
        voms-proxy-info -all
        xrdcp root://grid-cms-xrootd.physik.rwth-aachen.de:1094/${INPFILE} $tmp_dir/infile.root
        export X509_USER_PROXY=$5
        xrdcp root://grid-cms-xrootd.physik.rwth-aachen.de:1094/${INPFILE} $tmp_dir/infile.root
        #export X509_USER_PROXY=x509up_u38320
        #xrdcp root://grid-cms-xrootd.physik.rwth-aachen.de:1094/${INPFILE} $tmp_dir/infile.root
        #voms-proxy-info -all -file $5
        echo "echo PATH:"
        echo $PATH
        source /cvmfs/grid.cern.ch/etc/profile.d/setup-cvmfs-ui.sh
        which xrdcp
        echo "changing to tempdir (second time)"
        cd $tmp_dir
        pwd
        ls
        
        #export X509_USER_PROXY=$5
        #export X509_USER_PROXY=${HOME}/private/x509up_u38320
        #voms-proxy-info -all
        #voms-proxy-info -all -file $5
        #pip install --upgrade pip
        #pip install wheel
        #pip install xrootd --user
        #pip install xrootd.whl --user
        #xrdcp root://xrootd-cms.infn.it//${INPFILE} ./infile.root
        #xrdcp -d 1 -f root://grid-cms-xrootd.physik.rwth-aachen.de:1094/${INPFILE} /dev/null
        #xrdcp root://grid-cms-xrootd.physik.rwth-aachen.de:1094/${INPFILE} ./infile.root
        #xrdfs grid-cms-xrootd.physik.rwth-aachen.de ls -l -u /store/user/anovak/PFNano/106X_v2_17/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIIFall17PFNanoAODv2-PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v1PFNanoV2/210101_174326/0001
        #xrdcp -d 3 -f root://grid-cms-xrootd.physik.rwth-aachen.de:1094/${INPFILE} /dev/null
        #ls
        echo "running python script"
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

        echo "delete tmp dir"
        cd $TMP
        rm -r $tmp_dir

        echo "all done!"
