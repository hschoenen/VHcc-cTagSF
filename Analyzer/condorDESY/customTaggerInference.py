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
default = 0.001
#defaults_per_variable = minima - 0.001
defaults_per_variable = minima - default

def cleandataset(f, defaults_per_variable, isMC):
    print('Doing cleaning, isMC = ',isMC)
    feature_names = [k for k in f['Events'].keys() if  (('Jet_eta' == k) or ('Jet_pt' == k) or ('Jet_DeepCSV' in k))]
    # tagger output to compare with later and variables used to get the truth output
    feature_names.extend(('Jet_btagDeepB_b','Jet_btagDeepB_bb', 'Jet_btagDeepC','Jet_btagDeepL'))
    if isMC == True:
        feature_names.extend(('Jet_nBHadrons', 'Jet_hadronFlavour'))
    #print(feature_names)
    #print(len(feature_names))
    
    # go through a specified number of events, and get the information (awkward-arrays) for the keys specified above
    for data in f['Events'].iterate(feature_names, step_size=f['Events'].num_entries, library='ak'):
        break
        
    
    # creating an array to store all the columns with their entries per jet, flatten per-event -> per-jet
    datacolumns = np.zeros((len(feature_names)+1, len(ak.flatten(data['Jet_pt'], axis=1))))
    #print(len(datacolumns))

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

        #print(np.unique(target_class))

        #datacolumns[len(feature_names)] = ak.to_numpy(target_class)
        datacolumns[len(feature_names)] = target_class
        #print(np.unique(datacolumns[len(feature_names)]))
        
    datavectors = datacolumns.transpose()
    #print(np.unique(datavectors[:,len(feature_names)]))
    
    #print(i)
    for j in range(67):
        datavectors[:, j][datavectors[:, j] == np.nan]  = defaults_per_variable[j]
        datavectors[:, j][datavectors[:, j] <= -np.inf] = defaults_per_variable[j]
        datavectors[:, j][datavectors[:, j] >= np.inf]  = defaults_per_variable[j]
        datavectors[:, j][datavectors[:, j] == -999]  = defaults_per_variable[j]  # this one line is new and the reason for that is that there can be "original" -999 defaults in the inputs that should now also move into the new
                                                               # default bin, it was not necessary in my old clean_1_2.py code, because I could just leave them where they are, here they need to to be modified
        #print(np.unique(datavectors[:,-1]))
    #print(np.unique(datavectors[:,-1]))
    datavecak = ak.from_numpy(datavectors)
    #print(ak.unique(datavecak[:,-1]))
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
    #print(np.unique(alldata[:,-1]))
        
    
    for track0_vars in [6,12,22,29,35,42,50]:
        alldata[:,track0_vars][alldata[:,64] <= 0] = defaults_per_variable[track0_vars]
    for track0_1_vars in [7,13,23,30,36,43,51]:
        alldata[:,track0_1_vars][alldata[:,64] <= 1] = defaults_per_variable[track0_1_vars]
    for track01_2_vars in [8,14,24,31,37,44,52]:
        alldata[:,track01_2_vars][alldata[:,64] <= 2] = defaults_per_variable[track01_2_vars]
    for track012_3_vars in [9,15,25,32,38,45,53]:
        alldata[:,track012_3_vars][alldata[:,64] <= 3] = defaults_per_variable[track012_3_vars]
    for track0123_4_vars in [10,16,26,33,39,46,54]:
        alldata[:,track0123_4_vars][alldata[:,64] <= 4] = defaults_per_variable[track0123_4_vars]
    for track01234_5_vars in [11,17,27,34,40,47,55]:
        alldata[:,track01234_5_vars][alldata[:,64] <= 5] = defaults_per_variable[track01234_5_vars]
    alldata[:,18][alldata[:,65] <= 0] = defaults_per_variable[18]
    alldata[:,19][alldata[:,65] <= 1] = defaults_per_variable[19]
    alldata[:,20][alldata[:,65] <= 2] = defaults_per_variable[20]
    alldata[:,21][alldata[:,65] <= 3] = defaults_per_variable[21]

    for AboveCharm_vars in [41,48,49,56]:
        alldata[:,AboveCharm_vars][alldata[:,AboveCharm_vars]==-1] = defaults_per_variable[AboveCharm_vars] 
    
    
    datacls = [i for i in range(0,67)]
    if isMC == True:
        datacls.append(73)
    dataset = alldata[:, datacls]
    #print(np.unique(dataset[:,-1]))
    
    #DeepCSV_dataset = alldata[:, 67:71]
    
    return dataset

def preprocess(rootfile, isMC):
    print('Doing starting clean/prep, isMC: ',isMC)
    minima = np.load('/nfs/dust/cms/user/anstein/additional_files/default_value_studies_minima.npy')
    defaults_per_variable = minima - 0.001
    dataset_input_target = cleandataset(uproot.open(rootfile), defaults_per_variable, isMC)
    print(len(dataset_input_target))
    #print(np.unique(dataset_input_target[:,-1]))
    #sys.exit()
    inputs = torch.Tensor(dataset_input_target[:,0:67])
    # targets only make sense for MC, but it does no harm when calling it on Data (the last column is different though)
    targets = torch.Tensor(dataset_input_target[:,-1]).long()
    #print(torch.unique(targets))        
    scalers = []
    
    for i in range(0,67): # do not apply scaling to default values, which were set to -999
        scaler = StandardScaler().fit(inputs[:,i][inputs[:,i]!=defaults_per_variable[i]].reshape(-1,1))
        inputs[:,i]   = torch.Tensor(scaler.transform(inputs[:,i].reshape(-1,1)).reshape(1,-1))
        scalers.append(scaler)

    return inputs, targets, scalers

def apply_noise(sample, scalers, magn=1e-2,offset=[0]):
    device = torch.device("cpu")
    #scalers = torch.load(scalers_file_paths[s])
    #test_inputs =  torch.load(test_input_file_paths[s]).to(device).float()
    #val_inputs =  torch.load(val_input_file_paths[s]).to(device).float()
    #train_inputs =  torch.load(train_input_file_paths[s]).to(device).float()
    #test_targets =  torch.load(test_target_file_paths[s]).to(device)
    #val_targets =  torch.load(val_target_file_paths[s]).to(device)
    #train_targets =  torch.load(train_target_file_paths[s]).to(device)            
    #all_inputs = torch.cat((test_inputs,val_inputs,train_inputs))

    noise = torch.Tensor(np.random.normal(offset,magn,(len(sample),67))).to(device)
    xadv = sample + noise
    #all_inputs_noise = all_inputs + noise
    #xadv = scalers[variable].inverse_transform(all_inputs_noise[:,variable].cpu())
    integervars = [59,63,64,65,66]
    for variable in integervars:
        xadv[:,variable] = sample[:,variable]


    for i in range(67):
        defaults = abs(scalers[i].inverse_transform(sample[:,i].cpu()) - defaults_per_variable[i]) < 0.001   # "floating point error" --> allow some error margin
        if np.sum(defaults) != 0:
            xadv[:,i][defaults] = sample[:,i][defaults]

    return xadv

def fgsm_attack(epsilon=1e-2,sample=None,targets=None,reduced=True, scalers=None):
    device = torch.device("cpu")
    xadv = sample.clone().detach()
    
    # calculate the gradient of the model w.r.t. the *input* tensor:
    # first we tell torch that x should be included in grad computations
    xadv.requires_grad = True
    
    # from the undisturbed predictions, both the model and the criterion are already available and can be used here again; it's just that they were each part of a function, so not
    # automatically in the global scope
    
    global model
    global criterion
    
    # then we just do the forward and backwards pass as usual:
    preds = model(xadv)
    #print(targets)
    #print(torch.unique(targets))
    #print(preds)
    loss = criterion(preds, targets).mean()
    model.zero_grad()
    loss.backward()
    
    with torch.no_grad():
        #now we obtain the gradient of the input. It has the same dimensions as the tensor xadv, and it "points" in the direction of increasing loss values.
        dx = torch.sign(xadv.grad.detach())
        
        #so, we take a step in that direction!
        xadv += epsilon*torch.sign(dx)
        
        #remove the impact on selected variables. This is nessecary to avoid problems that occur otherwise in the input shapes.
        if reduced:
            integervars = [59,63,64,65,66]
            for variable in integervars:
                xadv[:,variable] = sample[:,variable]

            for i in range(67):
                defaults = abs(scalers[i].inverse_transform(sample[:,i].cpu()) - defaults_per_variable[i]) < 0.001   # "floating point error" --> allow some error margin
                if np.sum(defaults) != 0:
                    xadv[:,i][defaults] = sample[:,i][defaults]
        return xadv.detach()

def predict(inputs, targets, scalers, method):
    #inputs, targets, scalers = preprocess(rootfile)
    #print(targets[:100])
    #sys.exit()
    
    device = torch.device("cpu")
    global model
    global criterion
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
        #these classweights have been derived once for TTtoSemileptonic (the ones used for training)
        #class_weights = torch.FloatTensor(np.array([ 0.37333512, 24.65012434,  2.25474568,  1.1942229 ])).to(device)
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
    
    
    #inputs, targets, scalers = preprocess(fullName, isMC)
    inputs, targets, scalers = preprocess('infile.root', isMC)
    
    if weightingMethod == "_both":
        methods = ["_as_is","_new"]
    else:
        methods = [weightingMethod]
    
    for wm in methods:
        #outputPredsdir = "%s/%s/outPreds_%s%s.npy"%(condoroutdir,sampName,outNo,wm)
        #outputBvsLdir = "%s/%s/outBvsL_%s%s.npy"%(condoroutdir,sampName,outNo,wm)
        outputPredsdir = "outPreds_%s%s.npy"%(outNo,wm)
        outputBvsLdir = "outBvsL_%s%s.npy"%(outNo,wm)
        outputCvsBdir = "outCvsB_%s%s.npy"%(outNo,wm)
        outputCvsLdir = "outCvsL_%s%s.npy"%(outNo,wm)
        
        noise_outputPredsdir = "noise_outPreds_%s%s.npy"%(outNo,wm)
        noise_outputBvsLdir = "noise_outBvsL_%s%s.npy"%(outNo,wm)
        noise_outputCvsBdir = "noise_outCvsB_%s%s.npy"%(outNo,wm)
        noise_outputCvsLdir = "noise_outCvsL_%s%s.npy"%(outNo,wm)
        
        fgsm_outputPredsdir = "fgsm_outPreds_%s%s.npy"%(outNo,wm)
        fgsm_outputBvsLdir = "fgsm_outBvsL_%s%s.npy"%(outNo,wm)
        fgsm_outputCvsBdir = "fgsm_outCvsB_%s%s.npy"%(outNo,wm)
        fgsm_outputCvsLdir = "fgsm_outCvsL_%s%s.npy"%(outNo,wm)

        #print("Saving into %s/%s"%(condoroutdir,sampName))

        

        predictions = predict(inputs, targets, scalers, wm)
        #print(predictions[:100,:])
        np.save(outputPredsdir, predictions)
        bvl = calcBvsL(predictions)
        #print(bvl[:100])
        np.save(outputBvsLdir, bvl)
        del bvl
        gc.collect()
        cvb = calcCvsB(predictions)
        # handle division by 0 and assign -1 (if either CvsB or CvsL would be undefined)
        cvb[(predictions[:,0]+predictions[:,1]+predictions[:,2]) == 0] = -1
        cvb[(predictions[:,3]+predictions[:,2]) == 0] = -1
        #print(bvl[:100])
        np.save(outputCvsBdir, cvb)
        del cvb
        gc.collect()
        cvl = calcCvsL(predictions)
        cvl[(predictions[:,0]+predictions[:,1]+predictions[:,2]) == 0] = -1
        cvl[(predictions[:,3]+predictions[:,2]) == 0] = -1
        #print(bvl[:100])
        np.save(outputCvsLdir, cvl)
        del cvl
        del predictions
        gc.collect()
        
        if isMC == True:

            noise_preds = predict(apply_noise(inputs, scalers, magn=1e-2,offset=[0]), targets, scalers, wm)
            np.save(noise_outputPredsdir, noise_preds)
            noise_bvl = calcBvsL(noise_preds)
            #print(bvl[:100])
            np.save(noise_outputBvsLdir, noise_bvl)
            del noise_bvl
            gc.collect()
            noise_cvb = calcCvsB(noise_preds)
            # handle division by 0 and assign -1 (if either CvsB or CvsL would be undefined)
            noise_cvb[(noise_preds[:,0]+noise_preds[:,1]+noise_preds[:,2]) == 0] = -1
            noise_cvb[(noise_preds[:,3]+noise_preds[:,2]) == 0] = -1
            #print(bvl[:100])
            np.save(noise_outputCvsBdir, noise_cvb)
            del noise_cvb
            gc.collect()
            noise_cvl = calcCvsL(noise_preds)
            noise_cvl[(noise_preds[:,0]+noise_preds[:,1]+noise_preds[:,2]) == 0] = -1
            noise_cvl[(noise_preds[:,3]+noise_preds[:,2]) == 0] = -1
            #print(bvl[:100])
            np.save(noise_outputCvsLdir, noise_cvl)
            del noise_cvl
            del noise_preds
            gc.collect()


            fgsm_preds = predict(fgsm_attack(epsilon=1e-2,sample=inputs,targets=targets,reduced=True, scalers=scalers), targets, scalers, wm)
            np.save(fgsm_outputPredsdir, fgsm_preds)
            fgsm_bvl = calcBvsL(fgsm_preds)
            #print(bvl[:100])
            np.save(fgsm_outputBvsLdir, fgsm_bvl)
            del fgsm_bvl
            gc.collect()
            fgsm_cvb = calcCvsB(fgsm_preds)
            # handle division by 0 and assign -1 (if either CvsB or CvsL would be undefined)
            fgsm_cvb[(fgsm_preds[:,0]+fgsm_preds[:,1]+fgsm_preds[:,2]) == 0] = -1
            fgsm_cvb[(fgsm_preds[:,3]+fgsm_preds[:,2]) == 0] = -1
            #print(bvl[:100])
            np.save(fgsm_outputCvsBdir, fgsm_cvb)
            del fgsm_cvb
            gc.collect()
            fgsm_cvl = calcCvsL(fgsm_preds)
            fgsm_cvl[(fgsm_preds[:,0]+fgsm_preds[:,1]+fgsm_preds[:,2]) == 0] = -1
            fgsm_cvl[(fgsm_preds[:,3]+fgsm_preds[:,2]) == 0] = -1
            #print(bvl[:100])
            np.save(fgsm_outputCvsLdir, fgsm_cvl)
            del fgsm_cvl
            del fgsm_preds
            gc.collect()
