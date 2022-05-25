import sys
import os
# Done: use up-to-date versions
#import uproot4 as uproot
import uproot
import numpy as np
#import awkward1 as ak
import awkward as ak

import gc
# Done: New version of torch!
import torch
import torch.nn as nn
# Done: import the relevant models (from lxplus)
from pytorch_deepjet import DeepJet
from pytorch_deepjet_run2 import DeepJet_Run2
from pytorch_deepjet_transformer import DeepJetTransformer

from definitions import *

#from sklearn import metrics
#from sklearn.utils.class_weight import compute_class_weight
#from sklearn.model_selection import train_test_split
# Not yet sure if I will need this one (b/c DeepJet inputs are already processed how the tagger "sees" them)
#from sklearn.preprocessing import StandardScaler

import time
# Not (yet) for DeepJet
#from focal_loss import FocalLoss, focal_loss

def cross_entropy_one_hot(input, target):
    _, labels = target.max(dim=1)
    return nn.CrossEntropyLoss()(input, labels)
# we don't need one-hot-encoded targets, using 0 to 5 works as well
def cross_entropy(input, target):
    return nn.CrossEntropyLoss()(input, target)


print("Torch version =",torch.__version__)
# Hopefully nothing like that is necessary
#minima = np.load('/nfs/dust/cms/user/anstein/additional_files/default_value_studies_minima.npy')
#default = 0.001
#defaults_per_variable = minima - 0.001
#defaults_per_variable = minima - default
# ToDo: Instead we can use the definitions directly

from attacks import apply_noise, fgsm_attack

# ToDo: Instead of cleaning, we "only" need to transform the dataset into tensors that can be inserted into the model
'''
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
'''
def pfnano_to_array(rootfile, isMC):
    print('Doing cleaning, isMC = ',isMC)
    #feature_names = [k for k in f['Events'].keys() if  (('Jet_eta' == k) or ('Jet_pt' == k) or ('Jet_DeepJet' in k))]
    # Global
    feature_names = ['Jet_pt', 'Jet_eta',
                    'Jet_DeepJet_nCpfcand','Jet_DeepJet_nNpfcand',
                    'Jet_DeepJet_nsv','Jet_DeepJet_npv',
                    'Jet_DeepCSV_trackSumJetEtRatio',
                    'Jet_DeepCSV_trackSumJetDeltaR',
                    'Jet_DeepCSV_vertexCategory',
                    'Jet_DeepCSV_trackSip2dValAboveCharm',
                    'Jet_DeepCSV_trackSip2dSigAboveCharm',
                    'Jet_DeepCSV_trackSip3dValAboveCharm',
                    'Jet_DeepCSV_trackSip3dSigAboveCharm',
                    'Jet_DeepCSV_jetNSelectedTracks',
                    'Jet_DeepCSV_jetNTracksEtaRel']
    # CPF
    cpf = [[f'Jet_DeepJet_Cpfcan_BtagPf_trackEtaRel_{i}',
            f'Jet_DeepJet_Cpfcan_BtagPf_trackPtRel_{i}',
            f'Jet_DeepJet_Cpfcan_BtagPf_trackPPar_{i}',
            f'Jet_DeepJet_Cpfcan_BtagPf_trackDeltaR_{i}',
            f'Jet_DeepJet_Cpfcan_BtagPf_trackPParRatio_{i}',
            f'Jet_DeepJet_Cpfcan_BtagPf_trackSip2dVal_{i}',
            f'Jet_DeepJet_Cpfcan_BtagPf_trackSip2dSig_{i}',
            f'Jet_DeepJet_Cpfcan_BtagPf_trackSip3dVal_{i}',
            f'Jet_DeepJet_Cpfcan_BtagPf_trackSip3dSig_{i}',
            f'Jet_DeepJet_Cpfcan_BtagPf_trackJetDistVal_{i}',
            f'Jet_DeepJet_Cpfcan_ptrel_{i}',
            f'Jet_DeepJet_Cpfcan_drminsv_{i}',
            f'Jet_DeepJet_Cpfcan_VTX_ass_{i}',
            f'Jet_DeepJet_Cpfcan_puppiw_{i}',
            f'Jet_DeepJet_Cpfcan_chi2_{i}',
            f'Jet_DeepJet_Cpfcan_quality_{i}'] for i in range(25)]
    feature_names.extend([item for sublist in cpf for item in sublist])
    # NPF
    npf = [[f'Jet_DeepJet_Npfcan_ptrel_{i}',
            f'Jet_DeepJet_Npfcan_deltaR_{i}',
            f'Jet_DeepJet_Npfcan_isGamma_{i}',
            f'Jet_DeepJet_Npfcan_HadFrac_{i}',
            f'Jet_DeepJet_Npfcan_drminsv_{i}',
            f'Jet_DeepJet_Npfcan_puppiw_{i}'] for i in range(25)]
    feature_names.extend([item for sublist in npf for item in sublist])
    # VTX
    vtx = [[f'Jet_DeepJet_sv_pt_{i}',
            f'Jet_DeepJet_sv_deltaR_{i}',
            f'Jet_DeepJet_sv_mass_{i}',
            f'Jet_DeepJet_sv_ntracks_{i}',
            f'Jet_DeepJet_sv_chi2_{i}',
            f'Jet_DeepJet_sv_normchi2_{i}',
            f'Jet_DeepJet_sv_dxy_{i}',
            f'Jet_DeepJet_sv_dxysig_{i}',
            f'Jet_DeepJet_sv_d3d_{i}',
            f'Jet_DeepJet_sv_d3dsig_{i}',
            f'Jet_DeepJet_sv_costhetasvpv_{i}',
            f'Jet_DeepJet_sv_enratio_{i}'] for i in range(4)]
    feature_names.extend([item for sublist in vtx for item in sublist])
    
    number_of_features = len(feature_names)
    
    if isMC == True:
        # flavour definition for PFNano based on: https://indico.cern.ch/event/739204/#3-deepjet-overview
        # ToDo: find out how to read the angular distance between nu from b/c hadron and jet (< 0.4 to be lepb)
        feature_names.extend(('Jet_nBHadrons', 'Jet_hadronFlavour', 'Jet_partonFlavour'))
        # probably need to do something like here:
        # https://github.com/hqucms/DNNTuples/blob/93024f14ac05480dbc6ae9c9453678e8a5b66d25/BTagHelpers/src/FlavorDefinition.cc#L5-L90
        # or https://github.com/CMSDeepFlavour/DeepNTuples/blob/master/DeepNtuplizer/src/helpers.cc#L24-L134
        # need to go through genparticles, find all neutrinos from b or c decay
        # in a given event, cross these neutrinos with jets via DeltaR function (a.k.a metric_table)
        # if neutrino is contained in AK4 jet (and hadronFlavour==5 & nBhadrons==1) --> lepb
        # tbf this might be better suited for PFNano itself
        
        # In PFNano: collection is called GenPart
        ## GenPart_genPartIdxMother to check decay chain (necessary to find leptonic nu)
        ## GenPart_pdgId to check what particle it is
        ## comes with GenPart_eta and GenPart_phi --> good, can calculate DeltaR
        
        # feature_names.extend('nu_b_c_jet_deltaR')
        
    
    # go through a specified number of events, and get the information (awkward-arrays) for the keys specified above
    for data in rootfile['Events'].iterate(feature_names, step_size=rootfile['Events'].num_entries, library='ak'):
        break
    
    # creating an array to store all the columns with their entries per jet, flatten per-event -> per-jet
    # this works ONLY because the number of jets per event will be accessible in the analyzer
    datacolumns = np.zeros((number_of_features+1, len(ak.flatten(data['Jet_pt'], axis=1))))
    #print(len(datacolumns))

    for featureindex in range(number_of_features):
        a = ak.flatten(data[feature_names[featureindex]], axis=1) # flatten along first inside to get jets
        datacolumns[featureindex] = ak.to_numpy(a)

    if isMC == True:
        nbhad = ak.to_numpy(ak.flatten(data['Jet_nBHadrons'], axis=1))
        hadflav = ak.to_numpy(ak.flatten(data['Jet_hadronFlavour'], axis=1))
        partonflav = ak.to_numpy(ak.flatten(data['Jet_partonFlavour'], axis=1))

        target_class = np.full_like(hadflav, 3)                                                                                                    # c
        target_class = np.where(np.bitwise_and(hadflav == 5, nbhad > 1), 1, target_class)                                                          # bb
        # ToDo!!!
        #target_class = np.where(np.bitwise_and(hadflav == 5, nbhad == 1, nu_b_c_jet_deltaR < 0.4), 2, target_class)                                # lepb
        #target_class = np.where(np.bitwise_and(hadflav == 5, nbhad == 1, nu_b_c_jet_deltaR >= 0.4), 0, target_class)                               # b
        target_class = np.where(np.bitwise_and(hadflav == 0, partonflav == 21), 5, target_class)                                                   # g
        target_class = np.where(np.bitwise_and(hadflav == 0, np.bitwise_or(partonflav == 1, partonflav == 2, partonflav == 3)), 4, target_class)   # uds

        datacolumns[number_of_features] = target_class
        
    datavectors = datacolumns.transpose()
    # shape of datavectors: number of jets, number of features  +    1
    #                                             inputs           target
    #                                       (both data and MC)   (MC only)
    # Maybe ToDo: wondering whether we need to clean the features like it was done for DeepCSV, the ShallowTagInfos are contained in DeepJet inputs!
    return datavectors

# ToDo: merge with above function to get correct datastructure
def preprocess(rootfile_path, isMC):
    print('Doing starting clean/prep, isMC: ',isMC)
    #minima = np.load('/nfs/dust/cms/user/anstein/additional_files/default_value_studies_minima.npy')
    #defaults_per_variable = minima - 0.001
    #dataset_input_target = cleandataset(uproot.open(rootfile_path), defaults_per_variable, isMC)
    #print(len(dataset_input_target))
    #print(np.unique(dataset_input_target[:,-1]))
    #sys.exit()
    
    dataset_input_target = pfnano_to_array(uproot.open(rootfile_path), isMC)
    
    inputs = torch.Tensor(dataset_input_target[:,0:-1])
    # ToDo: need glob, cpf, npf, vtx as separate inputs
    # split first 4 times (n_features_in_group * n_candidates_per_group) for each group
    # then reshape each tensor into rectangular format (n_cands * n_features), glob has one dimension less
    
    slice_glob = cands_per_variable['glob'] * vars_per_candidate['glob']
    slice_cpf = cands_per_variable['cpf'] * vars_per_candidate['cpf']
    slice_npf = cands_per_variable['npf'] * vars_per_candidate['npf']
    slice_vtx = cands_per_variable['vtx'] * vars_per_candidate['vtx']
    
    glob, cpf, npf, vtx = inputs[:,0:slice_glob], inputs[:,slice_glob:slice_glob+slice_cpf], inputs[:,slice_glob+slice_cpf:slice_glob+slice_cpf+slice_npf], inputs[:,slice_glob+slice_cpf+slice_npf:slice_glob+slice_cpf+slice_npf+slice_vtx]
    # targets only make sense for MC,
    # but nothing 'breaks' when calling it on Data (the last column is different though, it's all Zeros, see definition above)
    targets = torch.Tensor(dataset_input_target[:,-1]).long()

    print(glob.shape)
    print(cpf.shape)
    print(npf.shape)
    print(vtx.shape)
    # OLD: return inputs, targets, scalers
    cpf = cpf.reshape((-1,cands_per_variable['cpf'],vars_per_candidate['cpf']))
    npf = npf.reshape((-1,cands_per_variable['npf'],vars_per_candidate['npf']))
    vtx = vtx.reshape((-1,cands_per_variable['vtx'],vars_per_candidate['vtx']))
    return glob,cpf,npf,vtx, targets

# ToDo: modify paths, import correct model
def predict(glob,cpf,npf,vtx, model_name):
    with torch.no_grad():
        device = torch.device("cpu")
        
        # Done: use the DF model from external module
        if 'DeepJet_Run2' in model_name:
            tagger = 'DF_Run2'
            model = DeepJet_Run2(num_classes = 6)
        elif 'DeepJetTransformer' in model_name:
            tagger = 'DF_Transformer'
            model = DeepJetTransformer(num_classes = 4)
        elif 'DeepJet' in model_name:
            tagger = 'DF'
            model = DeepJet(num_classes = 6)
        
        if 'nominal' in model_name:
            modelpath = f'/nfs/dust/cms/user/anstein/DeepJet/Train_{tagger}/nominal/checkpoint_best_loss.pth'
        elif 'adversarial_eps0p01' in model_name:
            modelpath = f'/nfs/dust/cms/user/anstein/DeepJet/Train_{tagger}/adversarial_eps0p01/checkpoint_best_loss.pth'
        

        checkpoint = torch.load(modelpath, map_location=torch.device(device))
        #model.load_state_dict(checkpoint["model_state_dict"])
        model.load_state_dict(checkpoint["state_dict"])

        model.to(device)

        #evaluate network on inputs
        model.eval()
        print('successfully loaded model and checkpoint')
        #sys.exit()
        return model(glob,cpf,npf,vtx).detach().numpy()


# ToDo: revisit calculation, use DeepJet output nodes (6! not 4)
def calcBvsL(matching_predictions):
    global n_jets
    #matching_predictions = np.where(np.tile((matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,3] != 0), (4,1)).transpose(), matching_predictions, (-1.0)*np.ones((n_jets,4)))
    
    custom_BvL = np.where(((matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,3]) != 0) & (matching_predictions[:,0] >= 0) & (matching_predictions[:,0] <= 1) & (matching_predictions[:,1] >= 0) & (matching_predictions[:,1] <= 1) & (matching_predictions[:,2] >= 0) & (matching_predictions[:,2] <= 1) & (matching_predictions[:,3] >= 0) & (matching_predictions[:,3] <= 1), (matching_predictions[:,0]+matching_predictions[:,1])/(matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,3]), (-1.0)*np.ones(n_jets))
    
    custom_BvL[(custom_BvL < 0.000001) & (custom_BvL > -0.000001)] = 0.000001
    custom_BvL[(np.isnan(custom_BvL)) & (np.isinf(custom_BvL))] = -1.0
    custom_BvL[custom_BvL > 0.99999] = 0.99999
    
    return custom_BvL

# ToDo: revisit calculation, use DeepJet output nodes (6! not 4)
def calcBvsC(matching_predictions):
    global n_jets
    #matching_predictions = np.where(np.tile((matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,2] != 0) , (4,1)).transpose(), matching_predictions, (-1.0)*np.ones((n_jets,4)))
    
    custom_BvC = np.where(((matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,2]) != 0) & (matching_predictions[:,0] >= 0) & (matching_predictions[:,0] <= 1) & (matching_predictions[:,1] >= 0) & (matching_predictions[:,1] <= 1) & (matching_predictions[:,2] >= 0) & (matching_predictions[:,2] <= 1) & (matching_predictions[:,3] >= 0) & (matching_predictions[:,3] <= 1), (matching_predictions[:,0]+matching_predictions[:,1])/(matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,2]), (-1.0)*np.ones(n_jets))
    
    custom_BvC[(custom_BvC < 0.000001) & (custom_BvC > -0.000001)] = 0.000001
    custom_BvC[(np.isnan(custom_BvC)) & (np.isinf(custom_BvC))] = -1.0
    custom_BvC[custom_BvC > 0.99999] = 0.99999
    
    return custom_BvC
    
# ToDo: revisit calculation, use DeepJet output nodes (6! not 4)
def calcCvsB(matching_predictions):
    global n_jets
    #matching_predictions = np.where(np.tile((matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,2] != 0), (4,1)).transpose(), matching_predictions, (-1.0)*np.ones((n_jets,4)))
    
    custom_CvB = np.where(((matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,2]) != 0) & (matching_predictions[:,0] >= 0) & (matching_predictions[:,0] <= 1) & (matching_predictions[:,1] >= 0) & (matching_predictions[:,1] <= 1) & (matching_predictions[:,2] >= 0) & (matching_predictions[:,2] <= 1) & (matching_predictions[:,3] >= 0) & (matching_predictions[:,3] <= 1), (matching_predictions[:,2])/(matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,2]), (-1.0)*np.ones(n_jets))
    
    custom_CvB[(custom_CvB < 0.000001) & (custom_CvB > -0.000001)] = 0.000001
    custom_CvB[(np.isnan(custom_CvB)) & (np.isinf(custom_CvB))] = -1.0
    custom_CvB[custom_CvB > 0.99999] = 0.99999
    
    return custom_CvB
    
# ToDo: revisit calculation, use DeepJet output nodes (6! not 4)
def calcCvsL(matching_predictions):
    global n_jets
    #matching_predictions = np.where(np.tile((matching_predictions[:,2]+matching_predictions[:,3] != 0), (4,1)).transpose(), matching_predictions, (-1.0)*np.ones((n_jets,4)))
    
    custom_CvL = np.where(((matching_predictions[:,2]+matching_predictions[:,3]) != 0) & (matching_predictions[:,0] >= 0) & (matching_predictions[:,0] <= 1) & (matching_predictions[:,1] >= 0) & (matching_predictions[:,1] <= 1) & (matching_predictions[:,2] >= 0) & (matching_predictions[:,2] <= 1) & (matching_predictions[:,3] >= 0) & (matching_predictions[:,3] <= 1), (matching_predictions[:,2])/(matching_predictions[:,2]+matching_predictions[:,3]), (-1.0)*np.ones(n_jets))
    
    custom_CvL[(custom_CvL < 0.000001) & (custom_CvL > -0.000001)] = 0.000001
    custom_CvL[(np.isnan(custom_CvL)) & (np.isinf(custom_CvL))] = -1.0
    custom_CvL[custom_CvL > 0.99999] = 0.99999
    
    return custom_CvL

'''    
# ignore
def calcBvsL_legacy(predictions):  # P(b)+P(bb)/(P(b)+P(bb)+P(udsg))
    bvsl = (predictions[:,0]+predictions[:,1])/(1-predictions[:,2])
    bvsl[bvsl < 0.000001] = 0.000001
    bvsl[bvsl > 0.99999] = 0.99999
    return bvsl    

# ignore
def calcBvsC_legacy(predictions):  # P(b)+P(bb)/(P(b)+P(bb)+P(c))
    bvsc = (predictions[:,0]+predictions[:,1])/(1-predictions[:,3])
    bvsc[bvsc < 0.000001] = 0.000001
    bvsc[bvsc > 0.99999] = 0.99999
    return bvsc
    
# ignore
def calcCvsB_legacy(predictions):  # P(c)/(P(b)+P(bb)+P(c))
    cvsb =  (predictions[:,2])/(predictions[:,0]+predictions[:,1]+predictions[:,2])
    cvsb[cvsb < 0.000001] = 0.000001
    cvsb[cvsb > 0.99999] = 0.99999
    cvsb[np.isnan(cvsb)] = -1
    return cvsb
    
# ignore
def calcCvsL_legacy(predictions):  # P(c)/(P(udsg)+P(c))
    cvsl = (predictions[:,2])/(predictions[:,3]+predictions[:,2])
    cvsl[cvsl < 0.000001] = 0.000001
    cvsl[cvsl > 0.99999] = 0.99999
    cvsl[np.isnan(cvsl)] = -1
    return cvsl
'''

# ToDo: * rename some variables
#       * add more nodes (6 instead of 4)
if __name__ == "__main__":
    fullName, model_name, condoroutdir = sys.argv[1], sys.argv[2], sys.argv[3]
    #predict('dummy',model_name)
    #sys.exit()
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
    # default era, will be overwritten
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
    
    # ToDo: modify once I have my private samples ready
    #parentDirList = ["VHcc_2017V5_Dec18/","NanoCrabProdXmas/","/2016/","2016_v2/","/2017/","2017_v2","/2018/","VHcc_2016V4bis_Nov18/"]
    parentDirList = ["/106X_v2_17/","/106X_v2_17rsb2/","/106X_v2_17rsb3/"]
    for iParent in parentDirList:
        if iParent in fullName: parentDir = iParent
    if parentDir == "": fullName.split('/')[8]+"/"
    
    # ToDo: check if still ok for my private samples
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
    # Commented out while using local files
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
    '''
    # ToDo: pass this as an argument instead of using global variable
    #global n_jets
    
    #inputs, targets, scalers = preprocess(fullName, isMC)
    # ToDo: get inputs, targets but no scalers necessary
    # OLD: inputs, targets, scalers = preprocess('infile.root', isMC)
    # NEW: glob,cpf,npf,vtx, targets = preprocess('infile.root', isMC)
    # WIP tests
    glob,cpf,npf,vtx, targets = preprocess('~/private/pfnano_dev/CMSSW_10_6_20/src/PhysicsTools/PFNano/test/nano106Xv8_on_mini106X_2017_mc_NANO_py_NANO_AddDeepJet.root', True)
    # sys.exit()
    n_jets = len(targets)
    
    #if weightingMethod == "_both":
    #    methods = ["_as_is","_new"]
    #else:
    #    methods = [weightingMethod]
    
    # to check multiple epochs of a given weighting method at once (using always 3 epochs should make sense, as previous tests were done on raw/noise/FGSM = 3 different sets)
    # Done: don't call it "weighting method"
    if model_name.startswith('_multi_'):
        letters = ['A','B','C']  # using the same three letters all the time means that the Analyzer code does not need to be updated for every possible epoch
        if 'nominal' in model_name:
            # basic training on raw inputs only
            models = ['nominal_'+e for e in (model_name.split('_basic_')[-1]).split(',')]
        else:
            # adversarial training
            models = ['adversarial_eps0p01_'+e for e in (model_name.split('_adversarial_eps0p01_')[-1]).split(',')]
        print('Will run with these weighting methods & epochs:', models)
        
        for i,model_i in enumerate(models):
            outputPredsdir = f"{letters[i]}_outPreds_%s.npy"%(outNo)
            outputCvsBdir  = f"{letters[i]}_outCvsB_%s.npy"%(outNo)
            outputCvsLdir  = f"{letters[i]}_outCvsL_%s.npy"%(outNo)
            outputBvsCdir  = f"{letters[i]}_outBvsC_%s.npy"%(outNo)
            outputBvsLdir  = f"{letters[i]}_outBvsL_%s.npy"%(outNo)
            
            # Done: independent of scalers!
            predictions = predict((glob,cpf,npf,vtx), model_i)
            
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
            
            
            # Done: switch to 6 output nodes
            predictions[:,0][predictions[:,0] > 0.99999] = 0.99999
            predictions[:,1][predictions[:,1] > 0.99999] = 0.99999
            predictions[:,2][predictions[:,2] > 0.99999] = 0.99999
            predictions[:,3][predictions[:,3] > 0.99999] = 0.99999
            predictions[:,4][predictions[:,4] > 0.99999] = 0.99999
            predictions[:,5][predictions[:,5] > 0.99999] = 0.99999
            predictions[:,0][predictions[:,0] < 0.000001] = 0.000001
            predictions[:,1][predictions[:,1] < 0.000001] = 0.000001
            predictions[:,2][predictions[:,2] < 0.000001] = 0.000001
            predictions[:,3][predictions[:,3] < 0.000001] = 0.000001
            predictions[:,4][predictions[:,4] < 0.000001] = 0.000001
            predictions[:,5][predictions[:,5] < 0.000001] = 0.000001
            print('Raw b, bb, lepb, c, uds, g min and max (after cutting over-/underflow)')
            print(min(predictions[:,0]), max(predictions[:,0]))
            print(min(predictions[:,1]), max(predictions[:,1]))
            print(min(predictions[:,2]), max(predictions[:,2]))
            print(min(predictions[:,3]), max(predictions[:,3]))
            print(min(predictions[:,4]), max(predictions[:,4]))
            print(min(predictions[:,5]), max(predictions[:,5]))
            np.save(outputPredsdir, predictions)
            del predictions
            gc.collect()
            
    # just one weighting method at a given epoch, but with Noise or FGSM attack applied to MC
    else:        
        # Can stay
        # new version doesn't store the w.m. in the filename
        '''
        outputPredsdir = "outPreds_%s.npy"%(outNo)
        outputCvsBdir = "outCvsB_%s.npy"%(outNo)
        outputCvsLdir = "outCvsL_%s.npy"%(outNo)
        outputBvsCdir = "outBvsC_%s.npy"%(outNo)
        outputBvsLdir = "outBvsL_%s.npy"%(outNo)
        # Can stay
        noise_outputPredsdir = "noise_outPreds_%s.npy"%(outNo)
        noise_outputCvsBdir = "noise_outCvsB_%s.npy"%(outNo)
        noise_outputCvsLdir = "noise_outCvsL_%s.npy"%(outNo)
        noise_outputBvsCdir = "noise_outBvsC_%s.npy"%(outNo)
        noise_outputBvsLdir = "noise_outBvsL_%s.npy"%(outNo)
        # Can stay
        fgsm_outputPredsdir = "fgsm_outPreds_%s.npy"%(outNo)
        fgsm_outputCvsBdir = "fgsm_outCvsB_%s.npy"%(outNo)
        fgsm_outputCvsLdir = "fgsm_outCvsL_%s.npy"%(outNo)
        fgsm_outputBvsCdir = "fgsm_outBvsC_%s.npy"%(outNo)
        fgsm_outputBvsLdir = "fgsm_outBvsL_%s.npy"%(outNo)

        #print("Saving into %s/%s"%(condoroutdir,sampName))
        '''

        # Done: independent of scalers!
        # OLD: predictions = predict(inputs, targets, scalers, wm)
        predictions = predict(glob,cpf,npf,vtx, model_name)
        sys.exit()
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


        # Done: switch to 6 output nodes
        predictions[:,0][predictions[:,0] > 0.99999] = 0.99999
        predictions[:,1][predictions[:,1] > 0.99999] = 0.99999
        predictions[:,2][predictions[:,2] > 0.99999] = 0.99999
        predictions[:,3][predictions[:,3] > 0.99999] = 0.99999
        predictions[:,4][predictions[:,4] > 0.99999] = 0.99999
        predictions[:,5][predictions[:,5] > 0.99999] = 0.99999
        predictions[:,0][predictions[:,0] < 0.000001] = 0.000001
        predictions[:,1][predictions[:,1] < 0.000001] = 0.000001
        predictions[:,2][predictions[:,2] < 0.000001] = 0.000001
        predictions[:,3][predictions[:,3] < 0.000001] = 0.000001
        predictions[:,4][predictions[:,4] < 0.000001] = 0.000001
        predictions[:,5][predictions[:,5] < 0.000001] = 0.000001
        print('Raw b, bb, lepb, c, uds, g min and max (after cutting over-/underflow)')
        print(min(predictions[:,0]), max(predictions[:,0]))
        print(min(predictions[:,1]), max(predictions[:,1]))
        print(min(predictions[:,2]), max(predictions[:,2]))
        print(min(predictions[:,3]), max(predictions[:,3]))
        print(min(predictions[:,4]), max(predictions[:,4]))
        print(min(predictions[:,5]), max(predictions[:,5]))
        np.save(outputPredsdir, predictions)
        del predictions
        gc.collect()

        if isMC == True:
            # Done: use new function with new arguments
            # OLD: noise_preds = predict(apply_noise(inputs, scalers, magn=1e-2,offset=[0]), targets, scalers, wm)
            # ToDo: rewrite apply_noise and call multiple times for four sets of inputs
            noise_preds = predict((apply_noise(glob,magn=1e-2,offset=[0],restrict_impact=0.2,var_group='glob'),
                                   apply_noise(cpf, magn=1e-2,offset=[0],restrict_impact=0.2,var_group='cpf'),
                                   apply_noise(npf, magn=1e-2,offset=[0],restrict_impact=0.2,var_group='npf'),
                                   apply_noise(vtx, magn=1e-2,offset=[0],restrict_impact=0.2,var_group='vtx')
                                  ),
                                  model_name)
            
            noise_bvl = calcBvsL(noise_preds)
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
            print(min(noise_cvb), max(noise_cvb))
            np.save(noise_outputCvsBdir, noise_cvb)
            del noise_cvb
            gc.collect()
            
            noise_cvl = calcCvsL(noise_preds)
            print(min(noise_cvl), max(noise_cvl))
            np.save(noise_outputCvsLdir, noise_cvl)
            del noise_cvl
            gc.collect()
            
            # Done: switch to 6 output nodes
            noise_preds[:,0][noise_preds[:,0] > 0.99999] = 0.99999
            noise_preds[:,1][noise_preds[:,1] > 0.99999] = 0.99999
            noise_preds[:,2][noise_preds[:,2] > 0.99999] = 0.99999
            noise_preds[:,3][noise_preds[:,3] > 0.99999] = 0.99999
            noise_preds[:,4][noise_preds[:,4] > 0.99999] = 0.99999
            noise_preds[:,5][noise_preds[:,5] > 0.99999] = 0.99999
            noise_preds[:,0][noise_preds[:,0] < 0.000001] = 0.000001
            noise_preds[:,1][noise_preds[:,1] < 0.000001] = 0.000001
            noise_preds[:,2][noise_preds[:,2] < 0.000001] = 0.000001
            noise_preds[:,3][noise_preds[:,3] < 0.000001] = 0.000001
            noise_preds[:,4][noise_preds[:,4] < 0.000001] = 0.000001
            noise_preds[:,5][noise_preds[:,5] < 0.000001] = 0.000001
            print('Noise b, bb, lepb, c, uds, g min and max (after cutting over-/underflow)')
            print(min(noise_preds[:,0]), max(noise_preds[:,0]))
            print(min(noise_preds[:,1]), max(noise_preds[:,1]))
            print(min(noise_preds[:,2]), max(noise_preds[:,2]))
            print(min(noise_preds[:,3]), max(noise_preds[:,3]))
            print(min(noise_preds[:,4]), max(noise_preds[:,4]))
            print(min(noise_preds[:,5]), max(noise_preds[:,5]))
            np.save(noise_outputPredsdir, noise_preds)
            del noise_preds
            gc.collect()

            # ToDo: use new function with new arguments
            # OLD: fgsm_preds = predict(fgsm_attack(epsilon=1e-2,sample=inputs,targets=targets,reduced=True, scalers=scalers), targets, scalers, wm)
            fgsm_preds = predict((fgsm_attack(epsilon=1e-2,sample=(glob,cpf,npf,vtx),targets=targets,
                                             thismodel=thismodel,thiscriterion=thiscriterion,reduced=True,restrict_impact=0.2)
                                 ),
                                 model_name)
            
            fgsm_bvl = calcBvsL(fgsm_preds)
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
            print(min(fgsm_cvb), max(fgsm_cvb))
            np.save(fgsm_outputCvsBdir, fgsm_cvb)
            del fgsm_cvb
            gc.collect()
            
            fgsm_cvl = calcCvsL(fgsm_preds)
            print(min(fgsm_cvl), max(fgsm_cvl))
            np.save(fgsm_outputCvsLdir, fgsm_cvl)
            del fgsm_cvl
            gc.collect()
            
            # Done: switch to 6 output nodes
            fgsm_preds[:,0][fgsm_preds[:,0] > 0.99999] = 0.99999
            fgsm_preds[:,1][fgsm_preds[:,1] > 0.99999] = 0.99999
            fgsm_preds[:,2][fgsm_preds[:,2] > 0.99999] = 0.99999
            fgsm_preds[:,3][fgsm_preds[:,3] > 0.99999] = 0.99999
            fgsm_preds[:,4][fgsm_preds[:,4] > 0.99999] = 0.99999
            fgsm_preds[:,5][fgsm_preds[:,5] > 0.99999] = 0.99999
            fgsm_preds[:,0][fgsm_preds[:,0] < 0.000001] = 0.000001
            fgsm_preds[:,1][fgsm_preds[:,1] < 0.000001] = 0.000001
            fgsm_preds[:,2][fgsm_preds[:,2] < 0.000001] = 0.000001
            fgsm_preds[:,3][fgsm_preds[:,3] < 0.000001] = 0.000001
            fgsm_preds[:,4][fgsm_preds[:,4] < 0.000001] = 0.000001
            fgsm_preds[:,5][fgsm_preds[:,5] < 0.000001] = 0.000001
            print('FGSM b, bb, lepb, c, uds, g min and max (after cutting over-/underflow)')
            print(min(fgsm_preds[:,0]), max(fgsm_preds[:,0]))
            print(min(fgsm_preds[:,1]), max(fgsm_preds[:,1]))
            print(min(fgsm_preds[:,2]), max(fgsm_preds[:,2]))
            print(min(fgsm_preds[:,3]), max(fgsm_preds[:,3]))
            print(min(fgsm_preds[:,4]), max(fgsm_preds[:,4]))
            print(min(fgsm_preds[:,5]), max(fgsm_preds[:,5]))
            np.save(fgsm_outputPredsdir, fgsm_preds)
            del fgsm_preds
            gc.collect()
