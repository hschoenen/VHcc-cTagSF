import sys
import os
import uproot4 as uproot
import numpy as np
import awkward1 as ak

import gc

import torch
import torch.nn as nn

from sklearn import metrics
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import time

print("Torch version =",torch.__version__)

minima = np.load('/nfs/dust/cms/user/anstein/additional_files/default_value_studies_minima.npy')
defaults = minima - 0.001

def cleandataset(f, defaults, isMC):
    print('Doing cleaning, isMC = ',isMC)
    feature_names = [k for k in f['Events'].keys() if  (('Jet_eta' == k) or ('Jet_pt' == k) or ('Jet_DeepCSV' in k))]
    # tagger output to compare with later and variables used to get the truth output
    feature_names.extend(('Jet_btagDeepB_b','Jet_btagDeepB_bb', 'Jet_btagDeepC','Jet_btagDeepL'))
    if isMC == True:
        feature_names.extend(('Jet_nBHadrons', 'Jet_hadronFlavour'))
    
    
    # go through a specified number of events, and get the information (awkward-arrays) for the keys specified above
    for data in f['Events'].iterate(feature_names, step_size=f['Events'].num_entries, library='ak'):
        break
        
    
    # creating an array to store all the columns with their entries per jet, flatten per-event -> per-jet
    datacolumns = np.zeros((len(feature_names)+1, len(ak.flatten(data['Jet_pt'], axis=1))))
   

    for featureindex in range(len(feature_names)):
        a = ak.flatten(data[feature_names[featureindex]], axis=1) # flatten along first inside to get jets
        
        datacolumns[featureindex] = ak.to_numpy(a)

    if isMC == True:
        nbhad = ak.to_numpy(ak.flatten(data['Jet_nBHadrons'], axis=1))
        hadflav = ak.to_numpy(ak.flatten(data['Jet_hadronFlavour'], axis=1))

        target_class = np.full_like(hadflav, 3)                                                      # udsg
        target_class = np.where(hadflav == 4, 2, target_class)                                       # c
        target_class = np.where(np.bitwise_and(hadflav == 5, nbhad > 1), 1, target_class)            # bb
        target_class = np.where(np.bitwise_and(hadflav == 5, nbhad <= 1), 0, target_class)           # b, lepb

   

        datacolumns[len(feature_names)] = ak.to_numpy(target_class) 

    datavectors = datacolumns.transpose()
    
    
    #print(i)
    for j in range(len(datavectors[0])):
        datavectors[datavectors[:, j] == np.nan]  = defaults[j]
        datavectors[datavectors[:, j] <= -np.inf] = defaults[j]
        datavectors[datavectors[:, j] >= np.inf]  = defaults[j]
        datavectors[datavectors[:, j] == -999]  = defaults[j]  # this one line is new and the reason for that is that there can be "original" -999 defaults in the inputs that should now also move into the new
                                                               # default bin, it was not necessary in my old clean_1_2.py code, because I could just leave them where they are, here they need to to be modified
    
    datavecak = ak.from_numpy(datavectors)
    
    #print(len(datavecak),"entries before cleaning step 1")
    
    #datavecak = datavecak[datavecak[:, 67] >= 0.]
    #datavecak = datavecak[datavecak[:, 67] <= 1.]
    #datavecak = datavecak[datavecak[:, 68] >= 0.]
    #datavecak = datavecak[datavecak[:, 68] <= 1.]
    #datavecak = datavecak[datavecak[:, 69] >= 0.]
    #datavecak = datavecak[datavecak[:, 69] <= 1.]
    #datavecak = datavecak[datavecak[:, 70] >= 0.]
    #datavecak = datavecak[datavecak[:, 70] <= 1.]

    

    # check jetNSelectedTracks, jetNSecondaryVertices > 0
    #datavecak = datavecak[(datavecak[:, 63] > 0) | (datavecak[:, 64] > 0)]  # keep those where at least any of the two variables is > 0, they don't need to be > 0 simultaneously
    #print(len(datavecak),"entries after cleaning step 1")

    alldata = ak.to_numpy(datavecak)
    
        
    
    for track0_vars in [6,12,22,29,35,42,50]:
        alldata[:,track0_vars][alldata[:,64] <= 0] = defaults[track0_vars]
    for track0_1_vars in [7,13,23,30,36,43,51]:
        alldata[:,track0_1_vars][alldata[:,64] <= 1] = defaults[track0_1_vars]
    for track01_2_vars in [8,14,24,31,37,44,52]:
        alldata[:,track01_2_vars][alldata[:,64] <= 2] = defaults[track01_2_vars]
    for track012_3_vars in [9,15,25,32,38,45,53]:
        alldata[:,track012_3_vars][alldata[:,64] <= 3] = defaults[track012_3_vars]
    for track0123_4_vars in [10,16,26,33,39,46,54]:
        alldata[:,track0123_4_vars][alldata[:,64] <= 4] = defaults[track0123_4_vars]
    for track01234_5_vars in [11,17,27,34,40,47,55]:
        alldata[:,track01234_5_vars][alldata[:,64] <= 5] = defaults[track01234_5_vars]
    alldata[:,18][alldata[:,65] <= 0] = defaults[18]
    alldata[:,19][alldata[:,65] <= 1] = defaults[19]
    alldata[:,20][alldata[:,65] <= 2] = defaults[20]
    alldata[:,21][alldata[:,65] <= 3] = defaults[21]

    for AboveCharm_vars in [41,48,49,56]:
        alldata[:,AboveCharm_vars][alldata[:,AboveCharm_vars]==-1] = defaults[AboveCharm_vars] 
    
    
    datacls = [i for i in range(0,67)]
    if isMC == True:
        datacls.append(73)
    dataset = alldata[:, datacls]
    
    #DeepCSV_dataset = alldata[:, 67:71]
    
    return dataset

def preprocess(rootfile, isMC):
    print('Doing starting clean/prep, isMC: ',isMC)
    minima = np.load('/nfs/dust/cms/user/anstein/additional_files/default_value_studies_minima.npy')
    defaults = minima - 0.001
    dataset_input_target = cleandataset(uproot.open(rootfile), defaults, isMC)
    print(len(dataset_input_target))
    #sys.exit()
    inputs = torch.Tensor(dataset_input_target[:,0:67])
    # targets only make sense for MC, but it does no harm when calling it on Data (the last column is different though)
    targets = torch.Tensor(dataset_input_target[:,-1]).long()
            
    scalers = []
    
    for i in range(0,67): # do not apply scaling to default values, which were set to -999
        scaler = StandardScaler().fit(inputs[:,i][inputs[:,i]!=defaults[i]].reshape(-1,1))
        inputs[:,i]   = torch.Tensor(scaler.transform(inputs[:,i].reshape(-1,1)).reshape(1,-1))
        scalers.append(scaler)

    return inputs, targets, scalers

def predict(inputs, targets, scalers, method):
    #inputs, targets, scalers = preprocess(rootfile)
    #print(targets[:100])
    #sys.exit()
    
    device = torch.device("cpu")
    
    model = nn.Sequential(nn.Linear(67, 100),
                      nn.ReLU(),
                      nn.Dropout(0.1),
                      nn.Linear(100, 100),
                      nn.ReLU(),
                      nn.Dropout(0.1),
                      nn.Linear(100, 100),
                      nn.ReLU(),
                      nn.Dropout(0.1),
                      nn.Linear(100, 100),
                      nn.ReLU(),
                      nn.Dropout(0.1),
                      nn.Linear(100, 100),
                      nn.ReLU(),
                      nn.Linear(100, 4),
                      nn.Softmax(dim=1))

    if method == '_new':
        #allweights = compute_class_weight(
        #       'balanced',
        #        classes=np.array([0,1,2,3]), 
        #        y=targets.numpy().astype(int))
        #class_weights = torch.FloatTensor(allweights).to(device)
        #del allweights
        #gc.collect()

        #criterion = nn.CrossEntropyLoss(weight=class_weights)
        criterion = nn.CrossEntropyLoss()
        modelPath = f'/nfs/dust/cms/user/anstein/pretrained_models/model_all_TT_350_epochs_v10_GPU_weighted_new_49_datasets_with_default_0.001.pt'
        
    else:
        criterion = nn.CrossEntropyLoss()
        modelPath = f'/nfs/dust/cms/user/anstein/pretrained_models/model_all_TT_350_epochs_v10_GPU_weighted_as_is_49_datasets_with_default_0.001.pt'
    
    checkpoint = torch.load(modelPath, map_location=torch.device(device))
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)

    #evaluate network on inputs
    model.eval()
    return model(inputs).detach().numpy()

    
def calcBvsL(predictions):  # P(b)+P(bb)
    return predictions[:,0]+predictions[:,1]
    
def calcCvsB(predictions):  # P(c)/(P(b)+P(bb)+P(c))
    return (predictions[:,2])/(predictions[:,0]+predictions[:,1]+predictions[:,2])
    
def calcCvsL(predictions):  # P(c)/(P(udsg)+P(c))
    return (predictions[:,2])/(predictions[:,3]+predictions[:,2])



if __name__ == "__main__":
    fullName, weightingMethod, condoroutdir = sys.argv[1], sys.argv[2], sys.argv[3]
    
    '''
    JECNameList = ["nom","jesTotalUp","jesTotalDown","jerUp","jerDown"]
    fileName = str(sys.argv[1])
    fullName = fileName
    isLocal = False
    if len(sys.argv) > 3:
        JECidx = int(sys.argv[4])
    else:
        JECidx = 0
    JECName = JECNameList[JECidx]

    maxEvents=-1

    print("#########"*10)
    print("start_time : ",time.ctime())
    print("processing on : ",fullName)
    
    debug = False
    isNano = False
    pref = ""
    '''
    parentDir = ""
    era = 2016

    #pnfspref = "/pnfs/desy.de/cms/tier2/"

    #if os.path.isfile(fullName):
    #    pref = ""
    #elif os.path.isfile(pnfspref+fullName):
    #    pref = pnfspref    
    #elif fullName.startswith("root:"):
    #    pref = ""
    #    #print("Input file name is in AAA format.")
    #else:
    #    pref = "root://xrootd-cms.infn.it//"
    #    #print("Forcing AAA.")
    #    if not fullName.startswith("/store/"):
    #        fileName = "/" + '/'.join(fullName.split('/')[fullName.split('/').index("store"):])
    #print("Will open file %s."%(pref+fileName))
    
    print("Will open file %s."%(fullName))

    #parentDirList = ["VHcc_2017V5_Dec18/","NanoCrabProdXmas/","/2016/","2016_v2/","/2017/","2017_v2","/2018/","VHcc_2016V4bis_Nov18/"]
    parentDirList = ["/106X_v2_17/","/106X_v2_17rsb2/","/106X_v2_17rsb3/"]
    for iParent in parentDirList:
        if iParent in fullName: parentDir = iParent
    if parentDir == "": fullName.split('/')[8]+"/"

    if "2017" in fullName: era = 2017
    if "2018" and not "2017" in fullName: era = 2018   # this is needed because both 2017 and 2018 appear in the new file names
    '''
    #if "spmondal" in fullName and fullName.startswith('/pnfs/'):
    ##    parentDir = 'VHbbPostNano2016_V5_CtagSF/'
        #parentDir = fullName.split('/')[8]+"/"
        #if "2017" in fullName: era = 2017
        #if "/2017/" in fullName: parentDir = "2017/"

    #if "VHcc_2017V5_Dec18" in fullName and fullName.startswith('/pnfs/'):
        #parentDir = 'VHcc_2017V5_Dec18/'
        #era = 2017
    #if fullName.startswith('/store/'):
        #if "lmastrol" in fullName:
            #pref = "/pnfs/desy.de/cms/tier2"
        #else:
            #pref = "root://xrootd-cms.infn.it//"
            #parentDir="NanoCrabProdXmas/"
            #isNano = True
    #elif fullName.startswith('root:'):
        #pref = ""
    #else:
        #pref = "file:"

    #iFile = TFile.Open(pref+fileName)

    #inputTree = iFile.Get("Events")
    #inputTree.SetBranchStatus("*",1)
    '''
    sampName=fullName.split(parentDir)[1].split('/')[0]
    channel=sampName
    sampNo=fullName.split(parentDir)[1].split('/')[1].split('_')[-1]
    dirNo=fullName.split(parentDir)[1].split('/')[3][-1]
    flNo=fullName.split(parentDir)[1].split('/')[-1].rstrip('.root').split('_')[-1]
    outNo= "%s_%s_%s"%(sampNo,dirNo,flNo)

    if "_" in channel: channel=channel.split("_")[0]
    # channel="Generic"
    if not 'Single' in channel and not 'Double' in channel and not 'EGamma' in channel:
        isMC = True
    else:
        isMC = False
    print("Using channel =",channel, "; isMC:", isMC, "; era: %d"%era)
    #print("Using jet pT with JEC correction:", JECName)
    #print("Output file will be numbered", outNo, "by condor.")

    #if not isMC and JECName!="nom":
    #    print("Cannot run data with JEC systematics!! Exiting.")
    #    sys.exit()
    '''
    #dirName=open("dirName.sh",'w')
    #if isMC and JECName!="nom":
    #    sampName += "_"+JECName
    #dirName.write("echo \""+sampName+"\"")
    #dirName.close()

    #flName=open("flName.sh",'w')
    #flName.write("echo \"%s\""%outNo)
    #flName.close()

    #if "OUTPUTDIR" in os.environ:
    #    condoroutdir = os.environ["OUTPUTDIR"]
    #    condoroutfile = "%s/%s/outTree_%s.root"%(condoroutdir,sampName,outNo)
    #    if os.path.isfile("%s/%s/outTree_%s.root"%(condoroutdir,sampName,outNo)):
    #        print("Output file already exists. Aborting job.")
    #        print("Outfile file: %s"%condoroutfile)
    #        sys.exit(99)
    '''    
    
    #inputs, targets, scalers = preprocess(fullName, isMC)
    inputs, targets, scalers = preprocess('infile.root', isMC)
    
    if weightingMethod == "_both":
        methods = ["_as_is","_new"]
    else:
        methods = weightingMethod
    
    for wm in methods:
        #outputPredsdir = "%s/%s/outPreds_%s%s.npy"%(condoroutdir,sampName,outNo,wm)
        #outputBvsLdir = "%s/%s/outBvsL_%s%s.npy"%(condoroutdir,sampName,outNo,wm)
        outputPredsdir = "outPreds_%s%s.npy"%(outNo,wm)
        outputBvsLdir = "outBvsL_%s%s.npy"%(outNo,wm)
        outputCvsBdir = "outCvsB_%s%s.npy"%(outNo,wm)
        outputCvsLdir = "outCvsL_%s%s.npy"%(outNo,wm)

        #print("Saving into %s/%s"%(condoroutdir,sampName))

        

        predictions = predict(inputs, targets, scalers, wm)
        #print(predictions[:100,:])
        np.save(outputPredsdir, predictions)
        bvl = calcBvsL(predictions)
        #print(bvl[:100])
        np.save(outputBvsLdir, bvl)
        del bvl
        gc.colect()
        cvb = calcCvsB(predictions)
        #print(bvl[:100])
        np.save(outputCvsBdir, cvb)
        del cvb
        gc.colect()
        cvl = calcCvsL(predictions)
        #print(bvl[:100])
        np.save(outputCvsLdir, cvl)
        del cvl
        gc.colect()
        
        
        