import os,sys
from random import random

# use like that:
# python make_submit.py filelists2017_PFNanoDeepJetINPUTS/inputs_DY_minimal DY 1
# python make_submit.py filelists2017_PFNanoParT/inputs_DY_minimal DY 1

inputdir = "inputs"
sel = "Wc"
for_testing = 0

if len(sys.argv) > 1:
    inputdir = sys.argv[1]
if len(sys.argv) > 2:
    sel = sys.argv[2]
if len(sys.argv) > 3:
    for_testing = int(sys.argv[3])

if for_testing == 1:
    appendix = '_for_testing'
else:
    appendix = ''
f2 = open("cmdList_sim_%s_PFNanoParTINPUTS%s.txt" % (sel, appendix),'w')
fDATA = open("cmdList_data_%s_PFNanoParTINPUTS%s.txt" % (sel, appendix),'w')
for fl in [i for i in os.listdir(inputdir) if os.path.isfile(os.path.join(inputdir,i))]:
#  if "Single" in fl or "Double" in fl or "EGamma" in fl or "MuonEG" in fl:
    f3 = open(inputdir.rstrip('/')+"/"+fl,'r')
    for i,line in enumerate(f3):
        if for_testing == 1 and i > 0:
            break
        if 'Wc' in sel and ('DY' in fl or 'TT' in fl or 'ST' in fl) and '2018' in inputdir:
            if random() > 1: continue
        if "Single" in line or "Double" in line or "EGamma" in line or "MuonEG" in line:
            #fDATA.write(sel+" "+line.split("root://grid-cms-xrootd.physik.rwth-aachen.de:1094/")[1])
            fDATA.write(sel+" "+line.split(".de:1094/")[1])
        else:    
            #f2.write(sel+" "+line.split("root://grid-cms-xrootd.physik.rwth-aachen.de:1094/")[1])
            f2.write(sel+" "+line.split(".de:1094/")[1])
    f3.close()
    #f2.write('\n')
f2.close()
