	OUTPUTDIR=/nfs/dust/cms/user/anstein/ctag_condor/systPlots_221010_2017_inputs_/Plots_221010_inputs_30bins/
	OUTPUTNAME=output_2017_PFNano

	CONDOR_CLUSTER_ID=$1
	CONDOR_PROCESS_ID=$2

        export PATH=/afs/desy.de/common/passwd:/usr/local/bin:/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/sbin:/cvmfs/grid.cern.ch/emi3ui-latest/bin:/cvmfs/grid.cern.ch/emi3ui-latest/sbin:/cvmfs/grid.cern.ch/emi3ui-latest/usr/bin:/cvmfs/grid.cern.ch/emi3ui-latest/usr/sbin:$PATH
        echo "echo PATH:"
        echo $PATH
        echo "arguments: " $1 $2 $3
        echo "username and group"
        id -n -u
        id -n -g
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
        echo "creating tempdir and copy"
        tmp_dir=$(mktemp -d)
# [AS: 13.09.2022]        cp -r ../Stacker_old.py cmdList.txt ../Deep*.root ../samplesDict.py $tmp_dir
# [AS: 13.09.2022]        cp -r ../Stacker.py cmdList.txt ../Deep*.root ../samplesDict.py $tmp_dir
        cp -r ../Stacker.py cmdList.txt ../Deep*.root ../samplesDict.py $tmp_dir

        echo "setting up the environment"
        cd /cvmfs/cms.cern.ch/slc7_amd64_gcc820/cms/cmssw/CMSSW_11_1_0_p3_ROOT618/src/
        source /afs/desy.de/user/a/anstein/public/to_source/cmsset_default.sh
        eval `scramv1 runtime -sh`
        echo "echo PATH:"
        echo $PATH
        source /cvmfs/grid.cern.ch/etc/profile.d/setup-cvmfs-ui.sh

        echo "changing to tempdir"
        cd $tmp_dir
        pwd
        ls
	
    	echo "running python script"
        python -c $'import Stacker; f=open("cmdList.txt","r"); ln=f.readlines(); thisline=ln['$2$'];\nfor cmd in thisline.split("NEWLINE"): exec(cmd)'
       	rc=$?
        if [[ $rc != 0 ]]
        then
            echo "got exit code from python code: " $rc
            exit $rc
        fi
      	echo "done running, now copying output to DUST"

        echo "copying output"
        mkdir -p ${OUTPUTDIR}
	    until cp -vr ${OUTPUTNAME}* ${OUTPUTDIR}; do
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
    
