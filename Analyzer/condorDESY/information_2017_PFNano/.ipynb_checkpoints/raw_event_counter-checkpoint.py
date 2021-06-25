import json
import numpy as np
import uproot4 as uproot

allEvents = []

PFNanopath = 'DoubleMuon.txt'
Postprpath = '/afs/desy.de/user/a/anstein/private/aisafety/SF/VHcc-cTagSF/Analyzer/condorDESY/filelists2017Postpro/filelists/DoubleMuon.txt'
with open(Postprpath, 'r') as infile:
    paths = infile.read().splitlines()
    for path in paths:
        #file = uproot.open(path)
        try:
            file = uproot.open(path)
        except:
            print('file',path,'not readable')
        else:
            nEvents = file['Events'].num_entries

            allEvents.append(nEvents)

    
print(sum(allEvents))