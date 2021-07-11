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

from focal_loss import FocalLoss, focal_loss

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
        datavectors[:, j][datavectors[:, j] == -999]  = defaults_per_variable[j] 
        # this one line is new and the reason for that is that there can be "original" -999 defaults in the inputs that should now also move into the new
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
    # targets only make sense for MC, but nothing 'breaks' when calling it on Data (the last column is different though)
    targets = torch.Tensor(dataset_input_target[:,-1]).long()
    #print(torch.unique(targets))        
    scalers = []
    
    for i in range(0,67): # use already calculated scalers (same for all files),
        # for the calculation, only train samples and only non-defaults were used
        #scaler = StandardScaler().fit(inputs[:,i][inputs[:,i]!=defaults_per_variable[i]].reshape(-1,1))
        scaler = torch.load(f'/nfs/dust/cms/user/anstein/additional_files/scalers/scaler_{i}_with_default_{default}.pt')
        inputs[:,i]   = torch.Tensor(scaler.transform(inputs[:,i].reshape(-1,1)).reshape(1,-1))
        scalers.append(scaler)

    return inputs, targets, scalers

def apply_noise(sample, scalers, magn=1e-2,offset=[0]):
    with torch.no_grad():
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
    # maybe add sample weights here as well for the ptetaflavloss weighting method
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
    with torch.no_grad():
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


        # ==========================================================================
        #
        #                               NEW: may_21
        #

        elif method == '_ptetaflavloss20':
            criterion = nn.CrossEntropyLoss(reduction='none')
            modelPath = f'/nfs/dust/cms/user/anstein/pretrained_models/model_124_epochs_v10_GPU_weighted_ptetaflavloss_20_datasets_with_default_0.001_-1.pt'

        elif method == '_ptetaflavloss278':
            criterion = nn.CrossEntropyLoss(reduction='none')
            modelPath = f'/nfs/dust/cms/user/anstein/pretrained_models/model_1_epochs_v10_GPU_weighted_ptetaflavloss_278_datasets_with_default_0.001_-1.pt'

        #
        #
        #
        # --------------------------------------------------------------------------


        # ==========================================================================
        #
        #                               NEW: as of June, 16th
        #

        elif method == '_ptetaflavloss_focalloss':
            # for focal loss: parameters
            alpha = None  # weights are handled differently, not with the focal loss but with sample weights if wanted
            gamma = 2
            criterion = FocalLoss(alpha, gamma, reduction='none')
            modelPath = f'/nfs/dust/cms/user/anstein/pretrained_models/model_200_epochs_v10_GPU_weighted_ptetaflavloss_focalloss_278_datasets_with_default_0.001_-1.pt'

        elif method == '_flatptetaflavloss_focalloss':
            # for focal loss: parameters
            alpha = None  # weights are handled differently, not with the focal loss but with sample weights if wanted
            gamma = 2
            criterion = FocalLoss(alpha, gamma, reduction='none')
            modelPath = f'/nfs/dust/cms/user/anstein/pretrained_models/model_200_epochs_v10_GPU_weighted_flatptetaflavloss_focalloss_278_datasets_with_default_0.001_-1.pt'

        #
        #
        #
        # --------------------------------------------------------------------------



        # ==========================================================================
        #
        #                               NEW: as of June, 25th
        #

        elif method == '_notflat_250_gamma2.0_alphaNone':
            # for focal loss: parameters
            alpha = None
            gamma = 2.0
            criterion = FocalLoss(alpha, gamma, reduction='none')
            modelPath = f'/nfs/dust/cms/user/anstein/pretrained_models/model_250_epochs_v10_GPU_weighted_ptetaflavloss_focalloss_278_datasets_with_default_0.001_-1.pt'

        elif method == '_flat_230_gamma2.0_alphaNone':
            # for focal loss: parameters
            alpha = None
            gamma = 2.0
            criterion = FocalLoss(alpha, gamma, reduction='none')
            modelPath = f'/nfs/dust/cms/user/anstein/pretrained_models/model_230_epochs_v10_GPU_weighted_flatptetaflavloss_focalloss_278_datasets_with_default_0.001_-1.pt'

        elif method == '_notflat_100_gamma20.0_alphaNone':
            # for focal loss: parameters
            alpha = None
            gamma = 20.0
            criterion = FocalLoss(alpha, gamma, reduction='none')
            modelPath = f'/nfs/dust/cms/user/anstein/pretrained_models/model_100_epochs_v10_GPU_weighted_ptetaflavloss_focalloss_gamma20.0_278_datasets_with_default_0.001_-1.pt'

        elif method == '_notflat_100_gamma4.0_alpha0.4,0.4,0.2,0.2':
            # for focal loss: parameters
            alpha = torch.Tensor([0.4,0.4,0.2,0.2])
            gamma = 4.0
            criterion = FocalLoss(alpha, gamma, reduction='none')
            modelPath = f'/nfs/dust/cms/user/anstein/pretrained_models/model_100_epochs_v10_GPU_weighted_ptetaflavloss_focalloss_gamma4.0_alpha0.4,0.4,0.2,0.2_278_datasets_with_default_0.001_-1.pt'

        elif method == '_flat_200_gamma25.0_alphaNone':
            # for focal loss: parameters
            alpha = None
            gamma = 25.0
            criterion = FocalLoss(alpha, gamma, reduction='none')
            modelPath = f'/nfs/dust/cms/user/anstein/pretrained_models/model_200_epochs_v10_GPU_weighted_flatptetaflavloss_focalloss_gamma25.0_278_datasets_with_default_0.001_-1.pt'

        #
        #
        #
        # --------------------------------------------------------------------------

        # ==========================================================================
        #
        #                               NEW: as of July, 8th
        #

        elif method == '_notflat_200_gamma25.0_alphaNone_adv_tr_eps0.01':
            # for focal loss: parameters
            alpha = None
            gamma = 25.0
            criterion = FocalLoss(alpha, gamma, reduction='none')
            modelPath = f'/nfs/dust/cms/user/anstein/pretrained_models/adv_tr/model_200_epochs_v10_GPU_weighted_ptetaflavloss_focalloss_gamma25.0_adv_tr_eps0.01_278_datasets_with_default_0.001_-1.pt'
            
        elif method == '_notflat_200_gamma25.0_alphaNone':
            # for focal loss: parameters
            alpha = None
            gamma = 25.0
            criterion = FocalLoss(alpha, gamma, reduction='none')
            modelPath = f'/nfs/dust/cms/user/anstein/pretrained_models/basic_tr/model_200_epochs_v10_GPU_weighted_ptetaflavloss_focalloss_gamma25.0_278_datasets_with_default_0.001_-1.pt'
            
        
        # special cases that can handle different epochs (checkpoints)    
        elif method.startswith('adv'):
            epoch = method.split('adv_tr_eps0.01_')[-1]
            # for focal loss: parameters
            alpha = None
            gamma = 25.0
            criterion = FocalLoss(alpha, gamma, reduction='none')
            modelPath = f'/nfs/dust/cms/user/anstein/pretrained_models/adv_tr/model_{epoch}_epochs_v10_GPU_weighted_ptetaflavloss_focalloss_gamma25.0_adv_tr_eps0.01_278_datasets_with_default_0.001_-1.pt'
            
        elif method.startswith('basic'):
            epoch = method.split('basic_')[-1]
            # for focal loss: parameters
            alpha = None
            gamma = 25.0
            criterion = FocalLoss(alpha, gamma, reduction='none')
            modelPath = f'/nfs/dust/cms/user/anstein/pretrained_models/basic_tr/model_{epoch}_epochs_v10_GPU_weighted_ptetaflavloss_focalloss_gamma25.0_278_datasets_with_default_0.001_-1.pt'
            

        #
        #
        #
        # --------------------------------------------------------------------------

        # old
        else:
            criterion = nn.CrossEntropyLoss()
            modelPath = f'/nfs/dust/cms/user/anstein/pretrained_models/model_all_TT_350_epochs_v10_GPU_weighted_as_is_49_datasets_with_default_0.001.pt'

        checkpoint = torch.load(modelPath, map_location=torch.device(device))
        model.load_state_dict(checkpoint["model_state_dict"])

        model.to(device)

        #evaluate network on inputs
        model.eval()
        return model(inputs).detach().numpy()


def calcBvsL(matching_predictions):
    global n_jets
    #matching_predictions = np.where(np.tile((matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,3] != 0), (4,1)).transpose(), matching_predictions, (-1.0)*np.ones((n_jets,4)))
    
    custom_BvL = np.where(((matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,3]) != 0) & (matching_predictions[:,0] >= 0) & (matching_predictions[:,0] <= 1) & (matching_predictions[:,1] >= 0) & (matching_predictions[:,1] <= 1) & (matching_predictions[:,2] >= 0) & (matching_predictions[:,2] <= 1) & (matching_predictions[:,3] >= 0) & (matching_predictions[:,3] <= 1), (matching_predictions[:,0]+matching_predictions[:,1])/(matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,3]), (-1.0)*np.ones(n_jets))
    
    custom_BvL[(custom_BvL < 0.000001) & (custom_BvL > -0.000001)] = 0.000001
    custom_BvL[(np.isnan(custom_BvL)) & (np.isinf(custom_BvL))] = -1.0
    custom_BvL[custom_BvL > 0.99999] = 0.99999
    
    return custom_BvL

def calcBvsC(matching_predictions):
    global n_jets
    #matching_predictions = np.where(np.tile((matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,2] != 0) , (4,1)).transpose(), matching_predictions, (-1.0)*np.ones((n_jets,4)))
    
    custom_BvC = np.where(((matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,2]) != 0) & (matching_predictions[:,0] >= 0) & (matching_predictions[:,0] <= 1) & (matching_predictions[:,1] >= 0) & (matching_predictions[:,1] <= 1) & (matching_predictions[:,2] >= 0) & (matching_predictions[:,2] <= 1) & (matching_predictions[:,3] >= 0) & (matching_predictions[:,3] <= 1), (matching_predictions[:,0]+matching_predictions[:,1])/(matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,2]), (-1.0)*np.ones(n_jets))
    
    custom_BvC[(custom_BvC < 0.000001) & (custom_BvC > -0.000001)] = 0.000001
    custom_BvC[(np.isnan(custom_BvC)) & (np.isinf(custom_BvC))] = -1.0
    custom_BvC[custom_BvC > 0.99999] = 0.99999
    
    return custom_BvC
    
def calcCvsB(matching_predictions):
    global n_jets
    #matching_predictions = np.where(np.tile((matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,2] != 0), (4,1)).transpose(), matching_predictions, (-1.0)*np.ones((n_jets,4)))
    
    custom_CvB = np.where(((matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,2]) != 0) & (matching_predictions[:,0] >= 0) & (matching_predictions[:,0] <= 1) & (matching_predictions[:,1] >= 0) & (matching_predictions[:,1] <= 1) & (matching_predictions[:,2] >= 0) & (matching_predictions[:,2] <= 1) & (matching_predictions[:,3] >= 0) & (matching_predictions[:,3] <= 1), (matching_predictions[:,2])/(matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,2]), (-1.0)*np.ones(n_jets))
    
    custom_CvB[(custom_CvB < 0.000001) & (custom_CvB > -0.000001)] = 0.000001
    custom_CvB[(np.isnan(custom_CvB)) & (np.isinf(custom_CvB))] = -1.0
    custom_CvB[custom_CvB > 0.99999] = 0.99999
    
    return custom_CvB
    
def calcCvsL(matching_predictions):
    global n_jets
    #matching_predictions = np.where(np.tile((matching_predictions[:,2]+matching_predictions[:,3] != 0), (4,1)).transpose(), matching_predictions, (-1.0)*np.ones((n_jets,4)))
    
    custom_CvL = np.where(((matching_predictions[:,2]+matching_predictions[:,3]) != 0) & (matching_predictions[:,0] >= 0) & (matching_predictions[:,0] <= 1) & (matching_predictions[:,1] >= 0) & (matching_predictions[:,1] <= 1) & (matching_predictions[:,2] >= 0) & (matching_predictions[:,2] <= 1) & (matching_predictions[:,3] >= 0) & (matching_predictions[:,3] <= 1), (matching_predictions[:,2])/(matching_predictions[:,2]+matching_predictions[:,3]), (-1.0)*np.ones(n_jets))
    
    custom_CvL[(custom_CvL < 0.000001) & (custom_CvL > -0.000001)] = 0.000001
    custom_CvL[(np.isnan(custom_CvL)) & (np.isinf(custom_CvL))] = -1.0
    custom_CvL[custom_CvL > 0.99999] = 0.99999
    
    return custom_CvL

    
def calcBvsL_legacy(predictions):  # P(b)+P(bb)/(P(b)+P(bb)+P(udsg))
    bvsl = (predictions[:,0]+predictions[:,1])/(1-predictions[:,2])
    bvsl[bvsl < 0.000001] = 0.000001
    bvsl[bvsl > 0.99999] = 0.99999
    return bvsl    
def calcBvsC_legacy(predictions):  # P(b)+P(bb)/(P(b)+P(bb)+P(c))
    bvsc = (predictions[:,0]+predictions[:,1])/(1-predictions[:,3])
    bvsc[bvsc < 0.000001] = 0.000001
    bvsc[bvsc > 0.99999] = 0.99999
    return bvsc
    
def calcCvsB_legacy(predictions):  # P(c)/(P(b)+P(bb)+P(c))
    cvsb =  (predictions[:,2])/(predictions[:,0]+predictions[:,1]+predictions[:,2])
    cvsb[cvsb < 0.000001] = 0.000001
    cvsb[cvsb > 0.99999] = 0.99999
    cvsb[np.isnan(cvsb)] = -1
    return cvsb
    
def calcCvsL_legacy(predictions):  # P(c)/(P(udsg)+P(c))
    cvsl = (predictions[:,2])/(predictions[:,3]+predictions[:,2])
    cvsl[cvsl < 0.000001] = 0.000001
    cvsl[cvsl > 0.99999] = 0.99999
    cvsl[np.isnan(cvsl)] = -1
    return cvsl


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
    
    global n_jets
    
    #inputs, targets, scalers = preprocess(fullName, isMC)
    inputs, targets, scalers = preprocess('infile.root', isMC)
    
    n_jets = len(targets)
    
    #if weightingMethod == "_both":
    #    methods = ["_as_is","_new"]
    #else:
    #    methods = [weightingMethod]
    
    # to check multiple epochs of a given weighting method at once (using always 3 epochs should make sense, as previous tests were done on raw/noise/FGSM = 3 different sets)
    if weightingMethod.startswith('_multi_'):
        letters = ['A','B','C']  # using the same three letters all the time means that the Analyzer code does not need to be updated for every possible epoch
        if 'basic' in weightingMethod:
            # basic training on raw inputs only
            wmethods = ['basic_'+e for e in (weightingMethod.split('_basic_')[-1]).split(',')]
        else:
            # adversarial training
            wmethods = ['adv_tr_eps0.01_'+e for e in (weightingMethod.split('_adv_tr_eps0.01_')[-1]).split(',')]
        print('Will run with these weighting methods & epochs:', wmethods)
        
        for i,wm in enumerate(wmethods):
            outputPredsdir = f"{letters[i]}_outPreds_%s.npy"%(outNo)
            outputCvsBdir  = f"{letters[i]}_outCvsB_%s.npy"%(outNo)
            outputCvsLdir  = f"{letters[i]}_outCvsL_%s.npy"%(outNo)
            outputBvsCdir  = f"{letters[i]}_outBvsC_%s.npy"%(outNo)
            outputBvsLdir  = f"{letters[i]}_outBvsL_%s.npy"%(outNo)
            
            predictions = predict(inputs, targets, scalers, wm)
            
            bvl = calcBvsL(predictions)
            print('Raw bvl, bvc, cvb, cvl')
            print(min(bvl), max(bvl))
            np.save(outputBvsLdir, bvl)
            del bvl
            gc.collect()

            bvc = calcBvsC(predictions)
            print(min(bvc), max(bvc))
            np.save(outputBvsCdir, bvc)
            del bvc
            gc.collect()

            cvb = calcCvsB(predictions)
            
            print(min(cvb), max(cvb))
            np.save(outputCvsBdir, cvb)
            del cvb
            gc.collect()
            cvl = calcCvsL(predictions)
            
            print(min(cvl), max(cvl))
            np.save(outputCvsLdir, cvl)
            del cvl
            gc.collect()

            predictions[:,0][predictions[:,0] > 0.99999] = 0.99999
            predictions[:,1][predictions[:,1] > 0.99999] = 0.99999
            predictions[:,2][predictions[:,2] > 0.99999] = 0.99999
            predictions[:,3][predictions[:,3] > 0.99999] = 0.99999
            predictions[:,0][predictions[:,0] < 0.000001] = 0.000001
            predictions[:,1][predictions[:,1] < 0.000001] = 0.000001
            predictions[:,2][predictions[:,2] < 0.000001] = 0.000001
            predictions[:,3][predictions[:,3] < 0.000001] = 0.000001
            print('Raw b, bb, c, l min and max (after cutting over-/underflow)')
            print(min(predictions[:,0]), max(predictions[:,0]))
            print(min(predictions[:,1]), max(predictions[:,1]))
            print(min(predictions[:,2]), max(predictions[:,2]))
            print(min(predictions[:,3]), max(predictions[:,3]))
            np.save(outputPredsdir, predictions)
            del predictions
            gc.collect()
            
    # just one weighting method at a given epoch, but with Noise or FGSM attack applied to MC
    else:

        wm = weightingMethod
        #for wm in methods:  # was formerly using a loop over all weighting methods, but this would require using also multiple w.m. in the Analyzer
        #outputPredsdir = "%s/%s/outPreds_%s%s.npy"%(condoroutdir,sampName,outNo,wm)
        #outputBvsLdir = "%s/%s/outBvsL_%s%s.npy"%(condoroutdir,sampName,outNo,wm)

        # new version doesn't store the w.m. in the filename
        outputPredsdir = "outPreds_%s.npy"%(outNo)
        outputCvsBdir = "outCvsB_%s.npy"%(outNo)
        outputCvsLdir = "outCvsL_%s.npy"%(outNo)
        outputBvsCdir = "outBvsC_%s.npy"%(outNo)
        outputBvsLdir = "outBvsL_%s.npy"%(outNo)

        noise_outputPredsdir = "noise_outPreds_%s.npy"%(outNo)
        noise_outputCvsBdir = "noise_outCvsB_%s.npy"%(outNo)
        noise_outputCvsLdir = "noise_outCvsL_%s.npy"%(outNo)
        noise_outputBvsCdir = "noise_outBvsC_%s.npy"%(outNo)
        noise_outputBvsLdir = "noise_outBvsL_%s.npy"%(outNo)

        fgsm_outputPredsdir = "fgsm_outPreds_%s.npy"%(outNo)
        fgsm_outputCvsBdir = "fgsm_outCvsB_%s.npy"%(outNo)
        fgsm_outputCvsLdir = "fgsm_outCvsL_%s.npy"%(outNo)
        fgsm_outputBvsCdir = "fgsm_outBvsC_%s.npy"%(outNo)
        fgsm_outputBvsLdir = "fgsm_outBvsL_%s.npy"%(outNo)

        #print("Saving into %s/%s"%(condoroutdir,sampName))



        predictions = predict(inputs, targets, scalers, wm)
        #print(predictions[:100,:])
        #hist, bin_edges = np.histogram(predictions[:,0],bins=20)
        #print('Flavour b predictions: bin_edges and histogram')
        #print(bin_edges)
        #print(hist)
        #del hist
        #del bin_edges
        #gc.collect()
        #hist, bin_edges = np.histogram(predictions[:,1],bins=20)
        #print('Flavour bb predictions: bin_edges and histogram')
        #print(bin_edges)
        #print(hist)
        #del hist
        #del bin_edges
        #gc.collect()
        #hist, bin_edges = np.histogram(predictions[:,2],bins=20)
        #print('Flavour c predictions: bin_edges and histogram')
        #print(bin_edges)
        #print(hist)
        #del hist
        #del bin_edges
        #gc.collect()
        #hist, bin_edges = np.histogram(predictions[:,3],bins=20)
        #print('Flavour udsg predictions: bin_edges and histogram')
        #print(bin_edges)
        #print(hist)
        #del hist
        #del bin_edges
        #gc.collect()
        bvl = calcBvsL(predictions)
        #hist, bin_edges = np.histogram(bvl)
        #print('bvl: bin_edges and histogram')
        #print(bin_edges)
        #print(hist)
        #del hist
        #del bin_edges
        #gc.collect()
        #print(bvl[:100])
        print('Raw bvl, bvc, cvb, cvl')
        print(min(bvl), max(bvl))
        np.save(outputBvsLdir, bvl)
        del bvl
        gc.collect()

        bvc = calcBvsC(predictions)
        print(min(bvc), max(bvc))
        np.save(outputBvsCdir, bvc)
        del bvc
        gc.collect()

        cvb = calcCvsB(predictions)
        #hist, bin_edges = np.histogram(cvb)
        #print('cvb: bin_edges and histogram before assigning -1')
        #print(bin_edges)
        #print(hist)
        #
        #
        #
        # NOTE:
        #
        # The -1 bins are now already assigned inside the functions for BvsL, CvsB and CvsL !!
        #
        # handle division by 0 and assign -1 (if either CvsB or CvsL would be undefined)
        #cvb[(predictions[:,0]+predictions[:,1]+predictions[:,2]) == 0] = -1
        #cvb[(predictions[:,3]+predictions[:,2]) == 0] = -1
        #hist, bin_edges = np.histogram(cvb,bins=20)
        #print('cvb: bin_edges and histogram after assigning -1')
        #print(bin_edges)
        #print(hist)
        #del hist
        #del bin_edges
        #gc.collect()
        #print(bvl[:100])
        print(min(cvb), max(cvb))
        np.save(outputCvsBdir, cvb)
        del cvb
        gc.collect()
        cvl = calcCvsL(predictions)
        #hist, bin_edges = np.histogram(cvl)
        #print('cvl: bin_edges and histogram before assigning -1')
        #print(bin_edges)
        #print(hist)
        #cvl[(predictions[:,0]+predictions[:,1]+predictions[:,2]) == 0] = -1
        #cvl[(predictions[:,3]+predictions[:,2]) == 0] = -1
        #hist, bin_edges = np.histogram(cvl,bins=20)
        #print('cvl: bin_edges and histogram after assigning -1')
        #print(bin_edges)
        #print(hist)
        #del hist
        #del bin_edges
        #gc.collect()
        #print(bvl[:100])
        print(min(cvl), max(cvl))
        np.save(outputCvsLdir, cvl)
        del cvl
        gc.collect()


        #print(min(predictions[:,0]), max(predictions[:,0]))
        #print(min(predictions[:,1]), max(predictions[:,1]))
        #print(min(predictions[:,2]), max(predictions[:,2]))
        #print(min(predictions[:,3]), max(predictions[:,3]))
        predictions[:,0][predictions[:,0] > 0.99999] = 0.99999
        predictions[:,1][predictions[:,1] > 0.99999] = 0.99999
        predictions[:,2][predictions[:,2] > 0.99999] = 0.99999
        predictions[:,3][predictions[:,3] > 0.99999] = 0.99999
        predictions[:,0][predictions[:,0] < 0.000001] = 0.000001
        predictions[:,1][predictions[:,1] < 0.000001] = 0.000001
        predictions[:,2][predictions[:,2] < 0.000001] = 0.000001
        predictions[:,3][predictions[:,3] < 0.000001] = 0.000001
        print('Raw b, bb, c, l min and max (after cutting over-/underflow)')
        print(min(predictions[:,0]), max(predictions[:,0]))
        print(min(predictions[:,1]), max(predictions[:,1]))
        print(min(predictions[:,2]), max(predictions[:,2]))
        print(min(predictions[:,3]), max(predictions[:,3]))
        np.save(outputPredsdir, predictions)
        del predictions
        gc.collect()

        if isMC == True:

            noise_preds = predict(apply_noise(inputs, scalers, magn=1e-2,offset=[0]), targets, scalers, wm)
            #hist, bin_edges = np.histogram(noise_preds[:,0],bins=20)
            #print('Flavour b noise_preds: bin_edges and histogram')
            #print(bin_edges)
            #print(hist)
            #del hist
            #del bin_edges
            #gc.collect()
            #hist, bin_edges = np.histogram(noise_preds[:,1],bins=20)
            #print('Flavour bb noise_preds: bin_edges and histogram')
            #print(bin_edges)
            #print(hist)
            #del hist
            #del bin_edges
            #gc.collect()
            #hist, bin_edges = np.histogram(noise_preds[:,2],bins=20)
            #print('Flavour c noise_preds: bin_edges and histogram')
            #print(bin_edges)
            #print(hist)
            #del hist
            #del bin_edges
            #gc.collect()
            #hist, bin_edges = np.histogram(noise_preds[:,3],bins=20)
            #print('Flavour udsg noise_preds: bin_edges and histogram')
            #print(bin_edges)
            #print(hist)
            #del hist
            #del bin_edges
            #gc.collect()
            noise_bvl = calcBvsL(noise_preds)
            #print(bvl[:100])
            print('Noise bvl, bvc, cvb, cvl')
            print(min(noise_bvl), max(noise_bvl))
            np.save(noise_outputBvsLdir, noise_bvl)
            del noise_bvl
            gc.collect()

            noise_bvc = calcBvsC(noise_preds)
            print(min(noise_bvc), max(noise_bvc))
            np.save(noise_outputBvsCdir, noise_bvc)
            del noise_bvc
            gc.collect()

            noise_cvb = calcCvsB(noise_preds)
            #hist, bin_edges = np.histogram(noise_cvb)
            #print('noise_cvb: bin_edges and histogram before assigning -1')
            #print(bin_edges)
            #print(hist)
            ## handle division by 0 and assign -1 (if either CvsB or CvsL would be undefined)
            #noise_cvb[(noise_preds[:,0]+noise_preds[:,1]+noise_preds[:,2]) == 0] = -1
            #noise_cvb[(noise_preds[:,3]+noise_preds[:,2]) == 0] = -1
            #hist, bin_edges = np.histogram(noise_cvb,bins=20)
            #print('noise_cvb: bin_edges and histogram after assigning -1')
            #print(bin_edges)
            #print(hist)
            #del hist
            #del bin_edges
            #gc.collect()
            #print(bvl[:100])
            print(min(noise_cvb), max(noise_cvb))
            np.save(noise_outputCvsBdir, noise_cvb)
            del noise_cvb
            gc.collect()
            noise_cvl = calcCvsL(noise_preds)
            #hist, bin_edges = np.histogram(noise_cvl)
            #print('noise_cvl: bin_edges and histogram before assigning -1')
            #print(bin_edges)
            #print(hist)
            #noise_cvl[(noise_preds[:,0]+noise_preds[:,1]+noise_preds[:,2]) == 0] = -1
            #noise_cvl[(noise_preds[:,3]+noise_preds[:,2]) == 0] = -1
            #hist, bin_edges = np.histogram(noise_cvl,bins=20)
            #print('noise_cvl: bin_edges and histogram after assigning -1')
            #print(bin_edges)
            #print(hist)
            #del hist
            #del bin_edges
            #gc.collect()
            #print(bvl[:100])
            print(min(noise_cvl), max(noise_cvl))
            np.save(noise_outputCvsLdir, noise_cvl)
            del noise_cvl
            noise_preds[:,0][noise_preds[:,0] > 0.99999] = 0.99999
            noise_preds[:,1][noise_preds[:,1] > 0.99999] = 0.99999
            noise_preds[:,2][noise_preds[:,2] > 0.99999] = 0.99999
            noise_preds[:,3][noise_preds[:,3] > 0.99999] = 0.99999
            noise_preds[:,0][noise_preds[:,0] < 0.000001] = 0.000001
            noise_preds[:,1][noise_preds[:,1] < 0.000001] = 0.000001
            noise_preds[:,2][noise_preds[:,2] < 0.000001] = 0.000001
            noise_preds[:,3][noise_preds[:,3] < 0.000001] = 0.000001
            print('Noise b, bb, c, l min and max (after cutting over-/underflow)')
            print(min(noise_preds[:,0]), max(noise_preds[:,0]))
            print(min(noise_preds[:,1]), max(noise_preds[:,1]))
            print(min(noise_preds[:,2]), max(noise_preds[:,2]))
            print(min(noise_preds[:,3]), max(noise_preds[:,3]))
            np.save(noise_outputPredsdir, noise_preds)
            del noise_preds
            gc.collect()


            fgsm_preds = predict(fgsm_attack(epsilon=1e-2,sample=inputs,targets=targets,reduced=True, scalers=scalers), targets, scalers, wm)
            #hist, bin_edges = np.histogram(fgsm_preds[:,0],bins=20)
            #print('Flavour b fgsm_preds: bin_edges and histogram')
            #print(bin_edges)
            #print(hist)
            #hist, bin_edges = np.histogram(fgsm_preds[:,1],bins=20)
            #print('Flavour bb fgsm_preds: bin_edges and histogram')
            #print(bin_edges)
            #print(hist)
            #hist, bin_edges = np.histogram(fgsm_preds[:,2],bins=20)
            #print('Flavour c fgsm_preds: bin_edges and histogram')
            #print(bin_edges)
            #print(hist)
            #hist, bin_edges = np.histogram(fgsm_preds[:,3],bins=20)
            #print('Flavour udsg fgsm_preds: bin_edges and histogram')
            #print(bin_edges)
            #print(hist)
            fgsm_bvl = calcBvsL(fgsm_preds)
            #print(bvl[:100])
            print('FGSM bvl, bvc, cvb, cvl')
            print(min(fgsm_bvl), max(fgsm_bvl))
            np.save(fgsm_outputBvsLdir, fgsm_bvl)
            del fgsm_bvl
            gc.collect()

            fgsm_bvc = calcBvsC(fgsm_preds)
            print(min(fgsm_bvc), max(fgsm_bvc))
            np.save(fgsm_outputBvsCdir, fgsm_bvc)
            del fgsm_bvc
            gc.collect()

            fgsm_cvb = calcCvsB(fgsm_preds)
            #hist, bin_edges = np.histogram(fgsm_cvb)
            #print('fgsm_cvb: bin_edges and histogram before assigning -1')
            #print(bin_edges)
            #print(hist)
            ## handle division by 0 and assign -1 (if either CvsB or CvsL would be undefined)
            #fgsm_cvb[(fgsm_preds[:,0]+fgsm_preds[:,1]+fgsm_preds[:,2]) == 0] = -1
            #fgsm_cvb[(fgsm_preds[:,3]+fgsm_preds[:,2]) == 0] = -1
            #hist, bin_edges = np.histogram(fgsm_cvb,bins=20)
            #print('fgsm_cvb: bin_edges and histogram after assigning -1')
            #print(bin_edges)
            #print(hist)
            #del hist
            #del bin_edges
            #gc.collect()
            #print(bvl[:100])
            print(min(fgsm_cvb), max(fgsm_cvb))
            np.save(fgsm_outputCvsBdir, fgsm_cvb)
            del fgsm_cvb
            gc.collect()
            fgsm_cvl = calcCvsL(fgsm_preds)
            #hist, bin_edges = np.histogram(fgsm_cvl)
            #print('fgsm_cvl: bin_edges and histogram before assigning -1')
            #print(bin_edges)
            #print(hist)
            #fgsm_cvl[(fgsm_preds[:,0]+fgsm_preds[:,1]+fgsm_preds[:,2]) == 0] = -1
            #fgsm_cvl[(fgsm_preds[:,3]+fgsm_preds[:,2]) == 0] = -1
            #hist, bin_edges = np.histogram(fgsm_cvl,bins=20)
            #print('fgsm_cvl: bin_edges and histogram after assigning -1')
            #print(bin_edges)
            #print(hist)
            #del hist
            #del bin_edges
            #gc.collect()
            #print(bvl[:100])
            print(min(fgsm_cvl), max(fgsm_cvl))
            np.save(fgsm_outputCvsLdir, fgsm_cvl)
            del fgsm_cvl
            fgsm_preds[:,0][fgsm_preds[:,0] > 0.99999] = 0.99999
            fgsm_preds[:,1][fgsm_preds[:,1] > 0.99999] = 0.99999
            fgsm_preds[:,2][fgsm_preds[:,2] > 0.99999] = 0.99999
            fgsm_preds[:,3][fgsm_preds[:,3] > 0.99999] = 0.99999
            fgsm_preds[:,0][fgsm_preds[:,0] < 0.000001] = 0.000001
            fgsm_preds[:,1][fgsm_preds[:,1] < 0.000001] = 0.000001
            fgsm_preds[:,2][fgsm_preds[:,2] < 0.000001] = 0.000001
            fgsm_preds[:,3][fgsm_preds[:,3] < 0.000001] = 0.000001
            print('FGSM b, bb, c, l min and max (after cutting over-/underflow)')
            print(min(fgsm_preds[:,0]), max(fgsm_preds[:,0]))
            print(min(fgsm_preds[:,1]), max(fgsm_preds[:,1]))
            print(min(fgsm_preds[:,2]), max(fgsm_preds[:,2]))
            print(min(fgsm_preds[:,3]), max(fgsm_preds[:,3]))
            np.save(fgsm_outputPredsdir, fgsm_preds)
            del fgsm_preds
            gc.collect()
