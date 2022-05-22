xrdcp -d 1 -f root://grid-cms-xrootd.physik.rwth-aachen.de:1094//store/user/anovak/PFNano/106X_v2_17/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIIFall17PFNanoAODv2-PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v1PFNanoV2/210101_174326/0001/nano_mc2017_1-1708.root /dev/null
# without any modifications, no xrdcp command can be found (makes sense)
which xrdcp
# this is what is done in the Analyzer's runscript
export PATH=/afs/desy.de/common/passwd:/usr/local/bin:/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/sbin:/cvmfs/grid.cern.ch/emi3ui-latest/bin:/cvmfs/grid.cern.ch/emi3ui-latest/sbin:/cvmfs/grid.cern.ch/emi3ui-latest/usr/bin:/cvmfs/grid.cern.ch/emi3ui-latest/usr/sbin:$PATH
cd /cvmfs/cms.cern.ch/slc6_amd64_gcc630/cms/cmssw/CMSSW_10_2_0_pre6/src
source /cvmfs/cms.cern.ch/cmsset_default.sh
eval `scramv1 runtime -sh`
# now xrcdp itself works, but fails to use the proxy
voms-proxy-info -all
xrdcp -d 1 -f root://grid-cms-xrootd.physik.rwth-aachen.de:1094//store/user/anovak/PFNano/106X_v2_17/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIIFall17PFNanoAODv2-PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v1PFNanoV2/210101_174326/0001/nano_mc2017_1-1708.root /dev/null
export X509_USER_PROXY=/afs/desy.de/user/a/anstein/private/x509up_u38320
voms-proxy-info -all
xrdcp -d 1 -f root://grid-cms-xrootd.physik.rwth-aachen.de:1094//store/user/anovak/PFNano/106X_v2_17/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIIFall17PFNanoAODv2-PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v1PFNanoV2/210101_174326/0001/nano_mc2017_1-1708.root /dev/null
export X509_USER_PROXY=$1
xrdcp -d 1 -f root://grid-cms-xrootd.physik.rwth-aachen.de:1094//store/user/anovak/PFNano/106X_v2_17/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIIFall17PFNanoAODv2-PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v1PFNanoV2/210101_174326/0001/nano_mc2017_1-1708.root /dev/null
voms-proxy-info -all
echo "echo PATH:"
echo $PATH
# when activating grid commands, xrdcp doesn't work anymore
source /cvmfs/grid.cern.ch/etc/profile.d/setup-cvmfs-ui.sh
xrdcp -d 1 -f root://grid-cms-xrootd.physik.rwth-aachen.de:1094//store/user/anovak/PFNano/106X_v2_17/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIIFall17PFNanoAODv2-PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v1PFNanoV2/210101_174326/0001/nano_mc2017_1-1708.root /dev/null
voms-proxy-info -all
#compgen  -abckA function
#compgen  -abckA function | less
#compgen  -abckA function | grep -i --color xrd
#xrdcp --help
#xrdfs --help
#xrdcp -d 1 -f root://grid-cms-xrootd.physik.rwth-aachen.de:1094//store/user/anovak/PFNano/106X_v2_17/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIIFall17PFNanoAODv2-PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v1PFNanoV2/210101_174326/0001/nano_mc2017_1-1708.root /dev/null
#xrdfs grid-cms-xrootd.physik.rwth-aachen.de ls -l -u /store/user/anovak/PFNano/106X_v2_17/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIIFall17PFNanoAODv2-PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v1PFNanoV2/210101_174326/0001
