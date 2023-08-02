import sys
import os

import uproot
import numpy as np
import awkward as ak

import gc

import torch
import torch.nn as nn

from pytorch_deepjet import DeepJet
from pytorch_deepjet_run2 import DeepJet_Run2
from pytorch_deepjet_transformer import DeepJetTransformer
from ParT import ParticleTransformer

import definitions
from attacks import apply_noise, fgsm_attack
import definitions_ParT
from attacks_ParT import first_order_attack

import time

# choose model
adversarial_model_name = 'fgsm-0_05'

def cross_entropy_one_hot(input, target):
    _, labels = target.max(dim=1)
    return nn.CrossEntropyLoss()(input, labels)
# we don't need one-hot-encoded targets, using 0 to 5 works as well
def cross_entropy(input, target):
    return nn.CrossEntropyLoss()(input, target)

print("Torch version =",torch.__version__)

# empty for usage in job (will be put into scratch-dir)
save_to = ''
# not empty for testing purposes, don't clutter working dir
#save_to = '/nfs/dust/cms/user/anstein/DeepJet/test_outputs_for_BTV_meeting_adversarial/'
#save_to = '/nfs/dust/cms/user/anstein/DeepJet/test_outputs_for_BTV_meeting_nominal/'


def pfnano_to_array(rootfile, isMC, deepjet=True):
    print('Doing cleaning, isMC = ',isMC)
    print('Is deepjet?', deepjet)
    
    if deepjet:
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
                        'Jet_DeepCSV_jetNTracksEtaRel'
                        ]
    
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
    
    else: # ParT
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
                        'Jet_DeepCSV_jetNTracksEtaRel'
                        ]
    
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
                f'Jet_ParT_Cpfcan_distminsv_{i}',
                f'Jet_DeepJet_Cpfcan_VTX_ass_{i}',
                f'Jet_DeepJet_Cpfcan_puppiw_{i}',
                f'Jet_DeepJet_Cpfcan_chi2_{i}',
                f'Jet_DeepJet_Cpfcan_quality_{i}'] for i in range(25)]
        feature_names.extend([item for sublist in cpf for item in sublist])
        # NPF
        npf = [[f'Jet_DeepJet_Npfcan_ptrel_{i}',
                f'Jet_ParT_Npfcan_etarel_{i}',
                f'Jet_ParT_Npfcan_phirel_{i}',
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
                f'Jet_ParT_sv_etarel_{i}',
                f'Jet_ParT_sv_phirel_{i}',
                f'Jet_DeepJet_sv_ntracks_{i}',
                f'Jet_DeepJet_sv_chi2_{i}',
                f'Jet_DeepJet_sv_normchi2_{i}',
                f'Jet_DeepJet_sv_dxy_{i}',
                f'Jet_DeepJet_sv_dxysig_{i}',
                f'Jet_DeepJet_sv_d3d_{i}',
                f'Jet_DeepJet_sv_d3dsig_{i}',
                f'Jet_DeepJet_sv_costhetasvpv_{i}',
                f'Jet_DeepJet_sv_enratio_{i}'] for i in range(4)] + \
              [[f'Jet_ParT_sv_pt_4',
                f'Jet_ParT_sv_deltaR_4',
                f'Jet_ParT_sv_mass_4',
                f'Jet_ParT_sv_etarel_4',
                f'Jet_ParT_sv_phirel_4',
                f'Jet_ParT_sv_ntracks_4',
                f'Jet_ParT_sv_chi2_4',
                f'Jet_ParT_sv_normchi2_4',
                f'Jet_ParT_sv_dxy_4',
                f'Jet_ParT_sv_dxysig_4',
                f'Jet_ParT_sv_d3d_4',
                f'Jet_ParT_sv_d3dsig_4',
                f'Jet_ParT_sv_costhetasvpv_4',
                f'Jet_ParT_sv_enratio_4']]
        feature_names.extend([item for sublist in vtx for item in sublist])
        
        # CPF 4-vec
        cpf_4v = [[f'Jet_ParT_Cpfcan_pt_{i}',
                   f'Jet_ParT_Cpfcan_eta_{i}',
                   f'Jet_ParT_Cpfcan_phi_{i}',
                   f'Jet_ParT_Cpfcan_e_{i}'] for i in range(25)]
        feature_names.extend([item for sublist in cpf_4v for item in sublist])
        # NPF 4-vec
        npf_4v = [[f'Jet_ParT_Npfcan_pt_{i}',
                   f'Jet_ParT_Npfcan_eta_{i}',
                   f'Jet_ParT_Npfcan_phi_{i}',
                   f'Jet_ParT_Npfcan_e_{i}'] for i in range(25)]
        feature_names.extend([item for sublist in npf_4v for item in sublist])
        # VTX 4-vec
        vtx_4v = [[f'Jet_DeepJet_sv_pt_{i}',
                   f'Jet_ParT_sv_eta_{i}',
                   f'Jet_ParT_sv_phi_{i}',
                   f'Jet_ParT_sv_e_{i}'] for i in range(4)] + \
                 [[f'Jet_ParT_sv_pt_4',
                   f'Jet_ParT_sv_eta_4',
                   f'Jet_ParT_sv_phi_4',
                   f'Jet_ParT_sv_e_4']]
        feature_names.extend([item for sublist in vtx_4v for item in sublist])
        
    #print(feature_names)
    number_of_features = len(feature_names)
    
    if isMC == True and targets_necessary:
        # flavour definition for PFNano based on: https://indico.cern.ch/event/739204/#3-deepjet-overview
        if 'Jet_FlavSplit' in rootfile['Events'].keys():
            feature_names.extend(['Jet_FlavSplit'])
        else:
            feature_names.extend(['Jet_hadronFlavour','Jet_partonFlavour','Jet_nBHadrons'])
     
    print('Events:', rootfile['Events'].num_entries)
    
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

    if isMC == True and targets_necessary:
        if 'Jet_FlavSplit' in rootfile['Events'].keys():
            flavsplit = ak.to_numpy(ak.flatten(data['Jet_FlavSplit'], axis=1))
            # if the list specified below was exhaustive, the -1 would get overwritten all the time
            #target_class = np.full_like(flavsplit, -1)                                                         # initialize
            # but it isn't the case, there are undefined jet flavors, therefore set to something that could be used later
            target_class = np.full_like(flavsplit, 1)                                                         # initialize
            target_class = np.where(flavsplit == 500, 0, target_class)                                                       # b
            target_class = np.where(np.bitwise_or(flavsplit == 510, flavsplit == 511), 1, target_class)                      # bb
            target_class = np.where(np.bitwise_or(flavsplit == 520, flavsplit == 521), 2, target_class)                      # leptonicb
            target_class = np.where(np.bitwise_or(flavsplit == 400, flavsplit == 410, flavsplit == 411), 3, target_class)    # c
            target_class = np.where(np.bitwise_or(flavsplit == 1, flavsplit == 2), 4, target_class)                          # uds
            target_class = np.where(flavsplit == 0, 5, target_class)                                                         # g
            del flavsplit
            gc.collect()
        else: # backup case for samples that don't have fine grained target definition available
            hadronFlav = ak.to_numpy(ak.flatten(data['Jet_hadronFlavour'], axis=1))
            partonFlav = ak.to_numpy(ak.flatten(data['Jet_partonFlavour'], axis=1))
            nBHadrons = ak.to_numpy(ak.flatten(data['Jet_nBHadrons'], axis=1))
            target_class = np.full_like(hadronFlav, 2)                                                         # initialize
            target_class = np.where(np.bitwise_and(hadronFlav == 5, nBHadrons == 1), 0, target_class)                        # b
            target_class = np.where(np.bitwise_and(hadronFlav == 5, nBHadrons > 1.5), 1, target_class)                       # bb
            #target_class = np.where(np.bitwise_or(flavsplit == 520, flavsplit == 521), 2, target_class)                     # leptonicb
            target_class = np.where(hadronFlav == 4, 3, target_class)                                                        # c
            target_class = np.where(np.bitwise_and(hadronFlav != 5, hadronFlav != 4), 4, target_class)                       # uds
            target_class = np.where(np.bitwise_and(hadronFlav != 5, hadronFlav != 4, partonFlav == 21), 5, target_class)     # g
            del hadronFlav
            del partonFlav
            del nBHadrons
            gc.collect()
        datacolumns[number_of_features] = target_class
        
    datavectors = datacolumns.transpose()
    print('Jets:', len(datavectors))    
    # shape of datavectors: number of jets, number of features  +    1
    #                                             inputs           target
    #                                       (both data and MC)   (MC only)
    # Maybe ToDo: wondering whether we need to clean the features like it was done for DeepCSV, the ShallowTagInfos are contained in DeepJet inputs!
    return datavectors


def preprocess(rootfile_path, isMC, deepjet=True):
    print('Doing starting clean/prep, isMC: ',isMC)
    dataset_input_target = pfnano_to_array(uproot.open(rootfile_path), isMC, deepjet)
    
    # targets only make sense for MC,
    # but nothing 'breaks' when calling it on Data (the last column is different though, it's all Zeros, see definition above)
    targets = torch.Tensor(dataset_input_target[:,-1]).long()
    inputs = torch.Tensor(dataset_input_target[:,0:-1])
    
    del dataset_input_target
    gc.collect()
    print(inputs[0])
    print(len(inputs[0]))
    if deepjet:
        slice_glob = definitions.cands_per_variable['glob'] * definitions.vars_per_candidate['glob']
        slice_cpf = definitions.cands_per_variable['cpf'] * definitions.vars_per_candidate['cpf']
        slice_npf = definitions.cands_per_variable['npf'] * definitions.vars_per_candidate['npf']
        slice_vtx = definitions.cands_per_variable['vtx'] * definitions.vars_per_candidate['vtx']
        glob = inputs[:,0:slice_glob]
        cpf  = inputs[:,slice_glob:slice_glob+slice_cpf]
        npf  = inputs[:,slice_glob+slice_cpf:slice_glob+slice_cpf+slice_npf]
        vtx  = inputs[:,slice_glob+slice_cpf+slice_npf:slice_glob+slice_cpf+slice_npf+slice_vtx]
    else:
        slice_glob = definitions_ParT.cands_per_variable['glob'] * definitions_ParT.vars_per_candidate['glob']
        slice_cpf = definitions_ParT.cands_per_variable['cpf'] * definitions_ParT.vars_per_candidate['cpf']
        slice_npf = definitions_ParT.cands_per_variable['npf'] * definitions_ParT.vars_per_candidate['npf']
        slice_vtx = definitions_ParT.cands_per_variable['vtx'] * definitions_ParT.vars_per_candidate['vtx']
        slice_cpf_4v = definitions_ParT.cands_per_variable['cpf_pts'] * definitions_ParT.vars_per_candidate['cpf_pts']
        slice_npf_4v = definitions_ParT.cands_per_variable['npf_pts'] * definitions_ParT.vars_per_candidate['npf_pts']
        slice_vtx_4v = definitions_ParT.cands_per_variable['vtx_pts'] * definitions_ParT.vars_per_candidate['vtx_pts']
        print(slice_glob+slice_cpf+slice_npf+slice_vtx+slice_cpf_4v+slice_npf_4v+slice_vtx_4v)
        cpf  = inputs[:,slice_glob:slice_glob+slice_cpf]
        npf  = inputs[:,slice_glob+slice_cpf:slice_glob+slice_cpf+slice_npf]
        vtx  = inputs[:,slice_glob+slice_cpf+slice_npf:slice_glob+slice_cpf+slice_npf+slice_vtx]
        cpf_4v  = inputs[:,slice_glob+slice_cpf+slice_npf+slice_vtx:slice_glob+slice_cpf+slice_npf+slice_vtx+slice_cpf_4v]
        npf_4v  = inputs[:,slice_glob+slice_cpf+slice_npf+slice_vtx+slice_cpf_4v:slice_glob+slice_cpf+slice_npf+slice_vtx+slice_cpf_4v+slice_npf_4v]
        vtx_4v  = inputs[:,slice_glob+slice_cpf+slice_npf+slice_vtx+slice_cpf_4v+slice_npf_4v:slice_glob+slice_cpf+slice_npf+slice_vtx+slice_cpf_4v+slice_npf_4v+slice_vtx_4v]

    if deepjet:
        cpf = cpf.reshape((-1,definitions.cands_per_variable['cpf'],definitions.vars_per_candidate['cpf']))
        npf = npf.reshape((-1,definitions.cands_per_variable['npf'],definitions.vars_per_candidate['npf']))
        vtx = vtx.reshape((-1,definitions.cands_per_variable['vtx'],definitions.vars_per_candidate['vtx']))
    else:
        cpf = cpf.reshape((-1,definitions_ParT.cands_per_variable['cpf'],definitions_ParT.vars_per_candidate['cpf']))
        npf = npf.reshape((-1,definitions_ParT.cands_per_variable['npf'],definitions_ParT.vars_per_candidate['npf']))
        vtx = vtx.reshape((-1,definitions_ParT.cands_per_variable['vtx'],definitions_ParT.vars_per_candidate['vtx']))
        cpf_4v = cpf_4v.reshape((-1,definitions_ParT.cands_per_variable['cpf_pts'],definitions_ParT.vars_per_candidate['cpf_pts']))
        npf_4v = npf_4v.reshape((-1,definitions_ParT.cands_per_variable['npf_pts'],definitions_ParT.vars_per_candidate['npf_pts']))
        vtx_4v = vtx_4v.reshape((-1,definitions_ParT.cands_per_variable['vtx_pts'],definitions_ParT.vars_per_candidate['vtx_pts']))
        
    if deepjet:
        return glob,cpf,npf,vtx, targets
    else:
        return cpf,npf,vtx,cpf_4v,npf_4v,vtx_4v, targets

def get_model(model_name, device):
    if 'DeepJet_Run2' in model_name:
        tagger = 'DF_Run2'
        model = DeepJet_Run2(num_classes = 6)
    elif 'DeepJetTransformer' in model_name:
        tagger = 'DF_Transformer'
        model = DeepJetTransformer(num_classes = 4)
    elif 'DeepJet' in model_name:
        tagger = 'DF'
        model = DeepJet(num_classes = 6)
    elif 'ParT' in model_name:
        tagger = 'ParT'
        model = ParticleTransformer(num_classes = 6,
                            num_enc = 3,
                            num_head = 8,
                            embed_dim = 128,
                            cpf_dim = 17,
                            npf_dim = 8,
                            vtx_dim = 14,
                            for_inference = False)
    else:
        tagger = 'DF_Run2'
        model = DeepJet_Run2(num_classes = 6)
        
    # edited to personal directories
    modelpath = f'/nfs/dust/cms/user/hschonen/auxiliary/models/{model_name}/checkpoint_best_loss.pth'
    #if 'nominal' in model_name and 'ParT' not in model_name:
        #modelpath = f'/nfs/dust/cms/user/hschonen/DeepJet/Train_{tagger}/nominal/checkpoint_best_loss.pth'
    #elif 'adversarial_eps0p01' in model_name:
        #modelpath = f'/nfs/dust/cms/user/hschonen/DeepJet/Train_{tagger}/fgsm/checkpoint_best_loss.pth'
    #elif 'adversarial_eps0p005' in model_name:
        #modelpath = f'/nfs/dust/cms/user/anstein/DeepJet/Train_{tagger}/adversarial_eps0p005/checkpoint_best_loss.pth'
    #elif 'GSAM' in model_name and 'GSAM2' not in model_name:
        #modelpath = f'/nfs/dust/cms/user/anstein/DeepJet/SharpnessAware/DeepJet_GSAM.pth'
    #elif 'GSAM2' in model_name:
        #modelpath = f'/nfs/dust/cms/user/anstein/DeepJet/SharpnessAware/DeepJet_GSAM2.pth'
    #elif 'ParT' in model_name and 'ngm' not in model_name:
        #modelpath = f'/nfs/dust/cms/user/anstein/ParT/nominal/checkpoint_epoch_20.pth'
    #elif 'ParT' in model_name and 'ngm' in model_name:
        #modelpath = f'/nfs/dust/cms/user/anstein/ParT/ngm_adversarial/checkpoint_epoch_20.pth'
        
    checkpoint = torch.load(modelpath, map_location=torch.device(device))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    return model

def predict(glob,cpf,npf,vtx,cpf_4v,npf_4v,vtx_4v, model_name, device):
    with torch.no_grad():
        model = get_model(model_name, device)
        #evaluate network on inputs
        model.eval()
        if 'ParT' in model_name:
            return nn.Softmax(dim=1)(model((cpf,npf,vtx,cpf_4v,npf_4v,vtx_4v))).detach().numpy(), model
        else:
            return nn.Softmax(dim=1)(model(glob,cpf,npf,vtx)).detach().numpy(), model

def calcBvsL(matching_predictions):
    custom_BvL = np.where( ((matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,2]) >= 0) \
                          & (matching_predictions[:,3] < 1.) \
                          & (matching_predictions[:,3] >= 0.)  \
                          & (matching_predictions[:,4]+matching_predictions[:,5] >= 0),
                          (matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,2])/(matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,2]+matching_predictions[:,4]+matching_predictions[:,5]),
                          (-1.0)*np.ones(n_jets))
    custom_BvL[(custom_BvL < 0.000001) & (custom_BvL > -0.000001)] = 0.000001
    custom_BvL[(np.isnan(custom_BvL)) & (np.isinf(custom_BvL))] = -1.0
    print('Number of BvL values below 0:',sum(custom_BvL[custom_BvL < 0.]))
    return custom_BvL


def calcBvsC(matching_predictions):
    custom_BvC = np.where( ((matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,2]) > 0.) \
                          & ((matching_predictions[:,3]) > 0.) \
                          & (matching_predictions[:,4]+matching_predictions[:,5] >= 0.),
                    (matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,2])/(matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,2]+matching_predictions[:,3]),
                          (-1.0)*np.ones(n_jets))
    
    custom_BvC[(custom_BvC < 0.000001) & (custom_BvC > -0.000001)] = 0.000001
    custom_BvC[(np.isnan(custom_BvC)) & (np.isinf(custom_BvC))] = -1.0
    return custom_BvC
    
    
def calcCvsB(matching_predictions):
    custom_CvB = np.where( ((matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,2]) > 0.) \
                          & (matching_predictions[:,3] > 0.) \
                          & (matching_predictions[:,4]+matching_predictions[:,5] >= 0.),
                          (matching_predictions[:,3])/(matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,2]+matching_predictions[:,3]),
                          (-1.0)*np.ones(n_jets))
    
    custom_CvB[(custom_CvB < 0.000001) & (custom_CvB > -0.000001)] = 0.000001
    custom_CvB[(np.isnan(custom_CvB)) & (np.isinf(custom_CvB))] = -1.0
    return custom_CvB
    
    
def calcCvsL(matching_predictions):    
    custom_CvL = np.where( ((matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,2]) < 1.) \
                          & ((matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,2]) >= 0.)  \
                          & (matching_predictions[:,3] >= 0) \
                          & (matching_predictions[:,4]+matching_predictions[:,5] >= 0),
                          (matching_predictions[:,3])/(matching_predictions[:,3]+matching_predictions[:,4]+matching_predictions[:,5]),
                          (-1.0)*np.ones(n_jets))
    
    custom_CvL[(custom_CvL < 0.000001) & (custom_CvL > -0.000001)] = 0.000001
    custom_CvL[(np.isnan(custom_CvL)) & (np.isinf(custom_CvL))] = -1.0
    return custom_CvL


if __name__ == "__main__":
    fullName, model_name, condoroutdir, targets_necessary, store_interesting_inputs = sys.argv[1], sys.argv[2], sys.argv[3], True if (sys.argv[4]=="yes") else False, True if (sys.argv[5]=="yes") else False
    
    print(targets_necessary, store_interesting_inputs)
    
    parentDir = ""
    # default era, will be overwritten
    era = 2016
    
    print("Will open file %s."%(fullName))
    
    # edited for new files
    parentDirList = ["/nanotest_add_DeepJet/","/PFNano/","/RunIISummer20UL17MiniAODv2/","/PFNano_ParT/"]
    for iParent in parentDirList:
        if iParent in fullName: parentDir = iParent
    if parentDir == "": fullName.split('/')[8]+"/"
    
    if "2017" in fullName: era = 2017
    if "2018" and not "2017" in fullName: era = 2018   # this is needed because both 2017 and 2018 appear in the new file names
    
    sampName=fullName.split(parentDir)[1].split('/')[0]
    channel=sampName
    print('sampName:', sampName)

    sampNo=fullName.split(parentDir)[1].split('/')[1]
    dirNo=fullName.split(parentDir)[1].split('/')[3][-1]
    flNo=fullName.split(parentDir)[1].split('/')[-1].rstrip('.root').split('_')[-1]
    outNo= "%s_%s_%s"%(sampNo,dirNo,flNo)
    print('outNo:', outNo)
    if "_" in channel: channel=channel.split("_")[0]
    # channel="Generic"
    if not 'Single' in channel and not 'Double' in channel and not 'EGamma' in channel and not 'MET' in channel:
        isMC = True
    else:
        isMC = False
    print("Using channel =",channel, "; isMC:", isMC, "; era: %d"%era)
    
    device = torch.device("cpu")
    
    # if targets are not necessary, targets will default to placeholder value for all jets
    if 'ParT' not in model_name:
        glob,cpf,npf,vtx, targets = preprocess('infile.root', isMC)
    else:
        cpf,npf,vtx,cpf_4v,npf_4v,vtx_4v, targets = preprocess('infile.root', isMC, deepjet=False)
    n_jets = len(targets)
    
    # activate this to debug with small set of jets
    #n_jets = 3000
    
    # default: no
    if targets_necessary:
        outputTargetsdir  = "outTargets_%s.npy"%(outNo)
        np.save(save_to+outputTargetsdir, targets)
    # default: no
    if store_interesting_inputs:
        inputsdir  = "inputsCENTRAL_%s.npy"%(outNo)
        inputsdirADVADV  = "inputsADV_ADV_%s.npy"%(outNo)
        inputsdirADVNOM  = "inputsADV_NOM_%s.npy"%(outNo)
        interesting_arrays = np.zeros((len(interesting_inputs), n_jets))
        for i,name in enumerate(interesting_inputs):
            #continue
            var_group = get_group(name)
            feature_index, cand_ind = get_group_index_from_name(name)
            if var_group == 'glob':
                this_column = glob[:,feature_index].detach().numpy()
            elif var_group == 'cpf':
                this_column = cpf[:,cand_ind, feature_index].detach().numpy()
            elif var_group == 'npf':
                this_column = npf[:,cand_ind, feature_index].detach().numpy()
            elif var_group == 'vtx':
                this_column = vtx[:,cand_ind, feature_index].detach().numpy()
            interesting_arrays[i] = this_column
            del this_column
        np.save(save_to+inputsdir, interesting_arrays)
        if not isMC:
            np.save(save_to+inputsdirADVNOM, interesting_arrays)
            np.save(save_to+inputsdirADVADV, interesting_arrays)
        del interesting_arrays
        gc.collect()
        
    # to check multiple epochs of a given weighting method at once (using always 3 epochs should make sense, as previous tests were done on raw/noise/FGSM = 3 different sets)
    # default: no
    if model_name.startswith('_multi_'):
        letters = ['A','B','C']  # using the same three letters all the time means that the Analyzer code does not need to be updated for every possible epoch
        if 'nominal' in model_name:
            # nominal training on raw inputs only
            models = ['nominal_'+e for e in (model_name.split('_basic_')[-1]).split(',')]
        else:
            # adversarial training
            epsilon_decimals = (model_name.split('eps0p')[-1]).split('_')[0]
            models = [f'adversarial_eps0p{epsilon_decimals}_'+e for e in (model_name.split(f'_adversarial_eps0p{epsilon_decimals}_')[-1]).split(',')]
        print('Will run with these weighting methods & epochs:', models)
        
        for i,model_i in enumerate(models):
            outputPredsdir = f"{letters[i]}_outPreds_%s.npy"%(outNo)
            outputCvsBdir  = f"{letters[i]}_outCvsB_%s.npy"%(outNo)
            outputCvsLdir  = f"{letters[i]}_outCvsL_%s.npy"%(outNo)
            outputBvsCdir  = f"{letters[i]}_outBvsC_%s.npy"%(outNo)
            outputBvsLdir  = f"{letters[i]}_outBvsL_%s.npy"%(outNo)
            # probably need to use chunks due to memory constraints here
            n_chunks = len(range(0,n_jets,2000))
            for i,k in enumerate(range(0,n_jets,2000)):
                #print(i,k)
                if i == 0:
                    predictions, _ = predict(glob[k:k+2000],cpf[k:k+2000],npf[k:k+2000],vtx[k:k+2000], model_i, device)
                elif i == n_chunks-1:
                    current_predictions, _ = predict(glob[k:n_jets],cpf[k:n_jets],npf[k:n_jets],vtx[k:n_jets], model_i, device)
                    np.concatenate((predictions,current_predictions))
                    del current_predictions
                    gc.collect()
                else:
                    current_predictions, _ = predict(glob[k:k+2000],cpf[k:k+2000],npf[k:k+2000],vtx[k:k+2000], model_i, device)
                    np.concatenate((predictions,current_predictions))
                    del current_predictions
                    gc.collect()
            bvl = calcBvsL(predictions)
            print('Raw bvl, bvc, cvb, cvl')
            print(min(bvl), max(bvl))
            np.save(save_to+outputBvsLdir, bvl)
            del bvl
            gc.collect()
            bvc = calcBvsC(predictions)
            print(min(bvc), max(bvc))
            np.save(save_to+outputBvsCdir, bvc)
            del bvc
            gc.collect()
            cvb = calcCvsB(predictions)
            print(min(cvb), max(cvb))
            np.save(save_to+outputCvsBdir, cvb)
            del cvb
            gc.collect()
            cvl = calcCvsL(predictions)
            print(min(cvl), max(cvl))
            np.save(save_to+outputCvsLdir, cvl)
            del cvl
            gc.collect()
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
            np.save(save_to+outputPredsdir, predictions)
            del predictions
            gc.collect()
    elif 'COMPARE' in model_name:
        if 'SHARPNESSAWARE' in model_name:
            models = ['_DeepJet_Run2_GSAM','_DeepJet_Run2_GSAM2']
            short_names = ['','ADV_']
        elif 'ParT' in model_name:
            models = ['_ParT_nominal','_ParT_ngm']
            short_names = ['','ADV_']
        else:
            models = ['_DeepJet_Run2_nominal','_DeepJet_Run2_adversarial_eps0p01']
            short_names = ['','ADV_']
        print('Will run with these models:', models)
        for i,model_i in enumerate(models):
            outputPredsdir = f"{short_names[i]}outPreds_%s.npy"%(outNo)
            outputCvsBdir  = f"{short_names[i]}outCvsB_%s.npy"%(outNo)
            outputCvsLdir  = f"{short_names[i]}outCvsL_%s.npy"%(outNo)
            outputBvsCdir  = f"{short_names[i]}outBvsC_%s.npy"%(outNo)
            outputBvsLdir  = f"{short_names[i]}outBvsL_%s.npy"%(outNo)
            # probably need to use chunks due to memory constraints here
            n_chunks = len(range(0,n_jets,2000))
            if 'ParT' in model_name:
                for i,k in enumerate(range(0,n_jets,2000)):
                    if i == 0:
                        predictions, _ = predict(None,cpf[k:k+2000],npf[k:k+2000],vtx[k:k+2000],cpf_4v[k:k+2000],npf_4v[k:k+2000],vtx_4v[k:k+2000], model_i, device)
                    elif i == n_chunks-1:
                        current_predictions, _ = predict(None,cpf[k:n_jets],npf[k:n_jets],vtx[k:n_jets],cpf_4v[k:n_jets],npf_4v[k:n_jets],vtx_4v[k:n_jets], model_i, device)
                        predictions = np.concatenate((predictions,current_predictions))
                        del current_predictions
                        gc.collect()
                    else:
                        current_predictions, _ = predict(None,cpf[k:k+2000],npf[k:k+2000],vtx[k:k+2000],cpf_4v[k:k+2000],npf_4v[k:k+2000],vtx_4v[k:k+2000], model_i, device)
                        predictions = np.concatenate((predictions,current_predictions))
                        del current_predictions
                        gc.collect()
            else:
                for i,k in enumerate(range(0,n_jets,2000)):
                    if i == 0:
                        predictions, _ = predict(glob[k:k+2000],cpf[k:k+2000],npf[k:k+2000],vtx[k:k+2000],None,None,None, model_i, device)
                    elif i == n_chunks-1:
                        current_predictions, _ = predict(glob[k:n_jets],cpf[k:n_jets],npf[k:n_jets],vtx[k:n_jets],None,None,None, model_i, device)
                        predictions = np.concatenate((predictions,current_predictions))
                        del current_predictions
                        gc.collect()
                    else:
                        current_predictions, _ = predict(glob[k:k+2000],cpf[k:k+2000],npf[k:k+2000],vtx[k:k+2000],None,None,None, model_i, device)
                        predictions = np.concatenate((predictions,current_predictions))
                        del current_predictions
                        gc.collect()
            # calculate and save discriminators
            bvl = calcBvsL(predictions)
            print('Raw bvl, bvc, cvb, cvl')
            print(min(bvl), max(bvl))
            np.save(save_to+outputBvsLdir, bvl)
            del bvl
            gc.collect()
            bvc = calcBvsC(predictions)
            print(min(bvc), max(bvc))
            np.save(save_to+outputBvsCdir, bvc)
            del bvc
            gc.collect()
            cvb = calcCvsB(predictions)
            print(min(cvb), max(cvb))
            np.save(save_to+outputCvsBdir, cvb)
            del cvb
            gc.collect()
            cvl = calcCvsL(predictions)
            print(min(cvl), max(cvl))
            np.save(save_to+outputCvsLdir, cvl)
            del cvl
            gc.collect()
            # constraining predictions is fine, they are not touched by Stacker
            # but we expect (want) them to be probabilities
            predictions[:,0][predictions[:,0] > 0.999999] = 0.999999
            predictions[:,1][predictions[:,1] > 0.999999] = 0.999999
            predictions[:,2][predictions[:,2] > 0.999999] = 0.999999
            predictions[:,3][predictions[:,3] > 0.999999] = 0.999999
            predictions[:,4][predictions[:,4] > 0.999999] = 0.999999
            predictions[:,5][predictions[:,5] > 0.999999] = 0.999999
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
            np.save(save_to+outputPredsdir, predictions)
            del predictions
            gc.collect()         
        # Also do attack with both models, mainly to store attacked samples
        if store_interesting_inputs and isMC:
            epsilon_factors = {
                'glob' : torch.Tensor(np.load(epsilons_per_feature['glob']).transpose()).to(device),
                'cpf' : torch.Tensor(np.load(epsilons_per_feature['cpf']).transpose()).to(device),
                'npf' : torch.Tensor(np.load(epsilons_per_feature['npf']).transpose()).to(device),
                'vtx' : torch.Tensor(np.load(epsilons_per_feature['vtx']).transpose()).to(device),
            }
            interesting_ADV_ADV_arrays = np.zeros((len(interesting_inputs), n_jets))
            interesting_ADV_NOM_arrays = np.zeros((len(interesting_inputs), n_jets))       
            # Code to get distorted inputs, two different models
            n_chunks = len(range(0,n_jets,2000))
            #print(n_chunks)
            for m,model_m in enumerate(models):
                print('Will apply attack to MC, with FGSM targeted to disturb', model_m)
                thismodel = get_model(model_m, device)
                for i,k in enumerate(range(0,n_jets,2000)):
                    #print(i,k)
                    if i == 0:
                        glob_fgsm, cpf_fgsm, npf_fgsm, vtx_fgsm = fgsm_attack(epsilon=1e-2,sample=(glob[k:k+2000],cpf[k:k+2000],npf[k:k+2000],vtx[k:k+2000]),
                                                                                  targets=targets[k:k+2000],thismodel=thismodel,thiscriterion=cross_entropy,reduced=True,restrict_impact=-1, epsilon_factors=epsilon_factors)
                        for i,name in enumerate(interesting_inputs):
                            var_group = get_group(name)
                            feature_index, cand_ind = get_group_index_from_name(name)
                            if var_group == 'glob':
                                this_column = glob_fgsm[:,feature_index].detach().numpy()
                            elif var_group == 'cpf':
                                this_column = cpf_fgsm[:,cand_ind, feature_index].detach().numpy()
                            elif var_group == 'npf':
                                this_column = npf_fgsm[:,cand_ind, feature_index].detach().numpy()
                            elif var_group == 'vtx':
                                this_column = vtx_fgsm[:,cand_ind, feature_index].detach().numpy()
                            if 'adv' in model_m:
                                interesting_ADV_ADV_arrays[i,k:k+2000] = this_column
                            else:
                                interesting_ADV_NOM_arrays[i,k:k+2000] = this_column
                            del this_column
                            gc.collect()
                          #  fgsm_preds, _ = predict(glob_fgsm, cpf_fgsm, npf_fgsm, vtx_fgsm, model_name)
                        del glob_fgsm
                        del cpf_fgsm
                        del npf_fgsm
                        del vtx_fgsm
                        gc.collect()
                    elif i == n_chunks-1:
                        glob_fgsm, cpf_fgsm, npf_fgsm, vtx_fgsm = fgsm_attack(epsilon=1e-2,sample=(glob[k:n_jets],cpf[k:n_jets],npf[k:n_jets],vtx[k:n_jets]),
                                                                              targets=targets[k:n_jets],thismodel=thismodel,thiscriterion=cross_entropy,reduced=True,restrict_impact=-1, epsilon_factors=epsilon_factors)
                        for i,name in enumerate(interesting_inputs):
                            var_group = get_group(name)
                            feature_index, cand_ind = get_group_index_from_name(name)
                            if var_group == 'glob':
                                this_column = glob_fgsm[:,feature_index].detach().numpy()
                            elif var_group == 'cpf':
                                this_column = cpf_fgsm[:,cand_ind, feature_index].detach().numpy()
                            elif var_group == 'npf':
                                this_column = npf_fgsm[:,cand_ind, feature_index].detach().numpy()
                            elif var_group == 'vtx':
                                this_column = vtx_fgsm[:,cand_ind, feature_index].detach().numpy()
                            if 'adv' in model_m:
                                interesting_ADV_ADV_arrays[i,k:n_jets] = this_column
                            else:
                                interesting_ADV_NOM_arrays[i,k:n_jets] = this_column
                            del this_column
                            gc.collect()
                        del glob_fgsm
                        del cpf_fgsm
                        del npf_fgsm
                        del vtx_fgsm
                        gc.collect()
                    else:
                        glob_fgsm, cpf_fgsm, npf_fgsm, vtx_fgsm = fgsm_attack(epsilon=1e-2,sample=(glob[k:k+2000],cpf[k:k+2000],npf[k:k+2000],vtx[k:k+2000]),
                                                                              targets=targets[k:k+2000],thismodel=thismodel,thiscriterion=cross_entropy,reduced=True,restrict_impact=-1, epsilon_factors=epsilon_factors)
                        for i,name in enumerate(interesting_inputs):
                            var_group = get_group(name)
                            feature_index, cand_ind = get_group_index_from_name(name)
                            if var_group == 'glob':
                                this_column = glob_fgsm[:,feature_index].detach().numpy()
                            elif var_group == 'cpf':
                                this_column = cpf_fgsm[:,cand_ind, feature_index].detach().numpy()
                            elif var_group == 'npf':
                                this_column = npf_fgsm[:,cand_ind, feature_index].detach().numpy()
                            elif var_group == 'vtx':
                                this_column = vtx_fgsm[:,cand_ind, feature_index].detach().numpy()
                            if 'adv' in model_m:
                                interesting_ADV_ADV_arrays[i,k:k+2000] = this_column
                            else:
                                interesting_ADV_NOM_arrays[i,k:k+2000] = this_column
                            del this_column
                            gc.collect()
                        del glob_fgsm
                        del cpf_fgsm
                        del npf_fgsm
                        del vtx_fgsm
                        gc.collect()
                if 'adv' in model_m:
                    np.save(save_to+inputsdirADVADV, interesting_ADV_ADV_arrays)
                else:
                    np.save(save_to+inputsdirADVNOM, interesting_ADV_NOM_arrays)
                    
    elif 'COMPLETE' in model_name:
        models = ['nominal',adversarial_model_name]
        short_names = ['','ADV_']
        print('Will run with these models:', models)
        
        for i,model_i in enumerate(models):
            outputPredsdir = f"{short_names[i]}outPreds_%s.npy"%(outNo)
            outputCvsBdir  = f"{short_names[i]}outCvsB_%s.npy"%(outNo)
            outputCvsLdir  = f"{short_names[i]}outCvsL_%s.npy"%(outNo)
            outputBvsCdir  = f"{short_names[i]}outBvsC_%s.npy"%(outNo)
            outputBvsLdir  = f"{short_names[i]}outBvsL_%s.npy"%(outNo)
            # probably need to use chunks due to memory constraints here
            n_chunks = len(range(0,n_jets,2000))
            # get model predictions
            for i,k in enumerate(range(0,n_jets,2000)):
                if i == 0:
                    predictions, _ = predict(glob[k:k+2000],cpf[k:k+2000],npf[k:k+2000],vtx[k:k+2000],None,None,None, model_i, device)
                elif i == n_chunks-1:
                    current_predictions, _ = predict(glob[k:n_jets],cpf[k:n_jets],npf[k:n_jets],vtx[k:n_jets],None,None,None, model_i, device)
                    predictions = np.concatenate((predictions,current_predictions))
                    del current_predictions
                    gc.collect()
                else:
                    current_predictions, _ = predict(glob[k:k+2000],cpf[k:k+2000],npf[k:k+2000],vtx[k:k+2000],None,None,None, model_i, device)
                    predictions = np.concatenate((predictions,current_predictions))
                    del current_predictions
                    gc.collect()
            # calculate discriminator values
            bvl = calcBvsL(predictions)
            print('Raw bvl, bvc, cvb, cvl')
            print(min(bvl), max(bvl))
            np.save(save_to+outputBvsLdir, bvl)
            del bvl
            gc.collect()
            bvc = calcBvsC(predictions)
            print(min(bvc), max(bvc))
            np.save(save_to+outputBvsCdir, bvc)
            del bvc
            gc.collect()
            cvb = calcCvsB(predictions)
            print(min(cvb), max(cvb))
            np.save(save_to+outputCvsBdir, cvb)
            del cvb
            gc.collect()
            cvl = calcCvsL(predictions)
            print(min(cvl), max(cvl))
            np.save(save_to+outputCvsLdir, cvl)
            del cvl
            gc.collect()
            # constraining predictions is fine (not touched by Stacker), but we expect (want) them to be probabilities
            predictions[:,0][predictions[:,0] > 0.999999] = 0.999999
            predictions[:,1][predictions[:,1] > 0.999999] = 0.999999
            predictions[:,2][predictions[:,2] > 0.999999] = 0.999999
            predictions[:,3][predictions[:,3] > 0.999999] = 0.999999
            predictions[:,4][predictions[:,4] > 0.999999] = 0.999999
            predictions[:,5][predictions[:,5] > 0.999999] = 0.999999
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
            np.save(save_to+outputPredsdir, predictions)
            del predictions
            gc.collect()        
            
    # just one training at a given epoch, but with Noise or FGSM attack applied to MC
    else:        
        # Can stay
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

        print("Saving into %s/%s"%(condoroutdir,sampName))
        

        #predictions, thismodel = predict(glob,cpf,npf,vtx, model_name)
        # probably need to use chunks due to memory constraints here
        n_chunks = len(range(0,n_jets,2000))
        #print(n_chunks)
        for i,k in enumerate(range(0,n_jets,2000)):
            #print(i,k)
            if i == 0:
                predictions, thismodel = predict(glob[k:k+2000],cpf[k:k+2000],npf[k:k+2000],vtx[k:k+2000], model_name, device)
            elif i == n_chunks-1:
                current_predictions, _ = predict(glob[k:n_jets],cpf[k:n_jets],npf[k:n_jets],vtx[k:n_jets], model_name, device)
                predictions = np.concatenate((predictions,current_predictions))
                del current_predictions
                gc.collect()
            else:
                current_predictions, _ = predict(glob[k:k+2000],cpf[k:k+2000],npf[k:k+2000],vtx[k:k+2000], model_name, device)
                predictions = np.concatenate((predictions,current_predictions))
                del current_predictions
                gc.collect()
        #print(n_jets, 'matches', len(predictions))
        
        #sys.exit()
        bvl = calcBvsL(predictions)
        print('Raw bvl, bvc, cvb, cvl')
        print(min(bvl), max(bvl))
        np.save(save_to+outputBvsLdir, bvl)
        hist, bin_edges = np.histogram(bvl, bins=20)
        print(hist, bin_edges)
        del bvl
        gc.collect()
        
        bvc = calcBvsC(predictions)
        print(min(bvc), max(bvc))
        np.save(save_to+outputBvsCdir, bvc)
        del bvc
        gc.collect()
        
        cvb = calcCvsB(predictions)
        print(min(cvb), max(cvb))
        np.save(save_to+outputCvsBdir, cvb)
        del cvb
        gc.collect()

        cvl = calcCvsL(predictions)
        print(min(cvl), max(cvl))
        np.save(save_to+outputCvsLdir, cvl)
        del cvl
        gc.collect()


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
        np.save(save_to+outputPredsdir, predictions)
        del predictions
        gc.collect()

        if isMC == True:
            
            n_chunks = len(range(0,n_jets,2000))
            #print(n_chunks)
            for i,k in enumerate(range(0,n_jets,2000)):
                #print(i,k)
                if i == 0:
                    noise_preds, _ = predict(apply_noise(glob[k:k+2000],magn=1e-2,offset=[0],restrict_impact=0.2,var_group='glob'),
                                                     apply_noise(cpf[k:k+2000], magn=1e-2,offset=[0],restrict_impact=0.2,var_group='cpf'),
                                                     apply_noise(npf[k:k+2000], magn=1e-2,offset=[0],restrict_impact=0.2,var_group='npf'),
                                                     apply_noise(vtx[k:k+2000], magn=1e-2,offset=[0],restrict_impact=0.2,var_group='vtx'),
                                                     model_name, device)
                elif i == n_chunks-1:
                    current_noise_preds, _ = predict(apply_noise(glob[k:n_jets],magn=1e-2,offset=[0],restrict_impact=0.2,var_group='glob'),
                                                     apply_noise(cpf[k:n_jets], magn=1e-2,offset=[0],restrict_impact=0.2,var_group='cpf'),
                                                     apply_noise(npf[k:n_jets], magn=1e-2,offset=[0],restrict_impact=0.2,var_group='npf'),
                                                     apply_noise(vtx[k:n_jets], magn=1e-2,offset=[0],restrict_impact=0.2,var_group='vtx'),
                                                     model_name, device)
                    noise_preds = np.concatenate((noise_preds,current_noise_preds))
                    del current_noise_preds
                    gc.collect()
                else:
                    current_noise_preds, _ = predict(apply_noise(glob[k:k+2000],magn=1e-2,offset=[0],restrict_impact=0.2,var_group='glob'),
                                                     apply_noise(cpf[k:k+2000], magn=1e-2,offset=[0],restrict_impact=0.2,var_group='cpf'),
                                                     apply_noise(npf[k:k+2000], magn=1e-2,offset=[0],restrict_impact=0.2,var_group='npf'),
                                                     apply_noise(vtx[k:k+2000], magn=1e-2,offset=[0],restrict_impact=0.2,var_group='vtx'),
                                                     model_name, device)
                    noise_preds = np.concatenate((noise_preds,current_noise_preds))
                    del current_noise_preds
                    gc.collect()
                
                
            noise_bvl = calcBvsL(noise_preds)
            print('Noise bvl, bvc, cvb, cvl')
            print(min(noise_bvl), max(noise_bvl))
            np.save(save_to+noise_outputBvsLdir, noise_bvl)
            del noise_bvl
            gc.collect()

            noise_bvc = calcBvsC(noise_preds)
            print(min(noise_bvc), max(noise_bvc))
            np.save(save_to+noise_outputBvsCdir, noise_bvc)
            del noise_bvc
            gc.collect()

            noise_cvb = calcCvsB(noise_preds)
            print(min(noise_cvb), max(noise_cvb))
            np.save(save_to+noise_outputCvsBdir, noise_cvb)
            del noise_cvb
            gc.collect()
            
            noise_cvl = calcCvsL(noise_preds)
            print(min(noise_cvl), max(noise_cvl))
            np.save(save_to+noise_outputCvsLdir, noise_cvl)
            del noise_cvl
            gc.collect()
            
            
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
            np.save(save_to+noise_outputPredsdir, noise_preds)
            del noise_preds
            gc.collect()

            #sys.exit()
            
            n_chunks = len(range(0,n_jets,2000))
            #print(n_chunks)
            for i,k in enumerate(range(0,n_jets,2000)):
                #print(i,k)
                if i == 0:
                    glob_fgsm, cpf_fgsm, npf_fgsm, vtx_fgsm = fgsm_attack(epsilon=1e-2,sample=(glob[k:k+2000],cpf[k:k+2000],npf[k:k+2000],vtx[k:k+2000]),
                                                                          targets=targets[k:k+2000],thismodel=thismodel,thiscriterion=cross_entropy,reduced=True,restrict_impact=0.2)
                    fgsm_preds, _ = predict(glob_fgsm, cpf_fgsm, npf_fgsm, vtx_fgsm, model_name, device)
                    del glob_fgsm
                    del cpf_fgsm
                    del npf_fgsm
                    del vtx_fgsm
                    gc.collect()
                elif i == n_chunks-1:
                    glob_fgsm, cpf_fgsm, npf_fgsm, vtx_fgsm = fgsm_attack(epsilon=1e-2,sample=(glob[k:n_jets],cpf[k:n_jets],npf[k:n_jets],vtx[k:n_jets]),
                                                                          targets=targets[k:n_jets],thismodel=thismodel,thiscriterion=cross_entropy,reduced=True,restrict_impact=0.2)
                    current_fgsm_preds, _ = predict(glob_fgsm, cpf_fgsm, npf_fgsm, vtx_fgsm, model_name, device)
                    del glob_fgsm
                    del cpf_fgsm
                    del npf_fgsm
                    del vtx_fgsm
                    gc.collect()
                    fgsm_preds = np.concatenate((fgsm_preds,current_fgsm_preds))
                    del current_fgsm_preds
                    gc.collect()
                else:
                    glob_fgsm, cpf_fgsm, npf_fgsm, vtx_fgsm = fgsm_attack(epsilon=1e-2,sample=(glob[k:k+2000],cpf[k:k+2000],npf[k:k+2000],vtx[k:k+2000]),
                                                                          targets=targets[k:k+2000],thismodel=thismodel,thiscriterion=cross_entropy,reduced=True,restrict_impact=0.2)
                    current_fgsm_preds, _ = predict(glob_fgsm, cpf_fgsm, npf_fgsm, vtx_fgsm, model_name, device)
                    del glob_fgsm
                    del cpf_fgsm
                    del npf_fgsm
                    del vtx_fgsm
                    gc.collect()
                    fgsm_preds = np.concatenate((fgsm_preds,current_fgsm_preds))
                    del current_fgsm_preds
                    gc.collect()
                    
            
            fgsm_bvl = calcBvsL(fgsm_preds)
            print('FGSM bvl, bvc, cvb, cvl')
            print(min(fgsm_bvl), max(fgsm_bvl))
            np.save(save_to+fgsm_outputBvsLdir, fgsm_bvl)
            hist, bin_edges = np.histogram(fgsm_bvl, bins=20)
            print(hist, bin_edges)
            del fgsm_bvl
            gc.collect()

            fgsm_bvc = calcBvsC(fgsm_preds)
            print(min(fgsm_bvc), max(fgsm_bvc))
            np.save(save_to+fgsm_outputBvsCdir, fgsm_bvc)
            del fgsm_bvc
            gc.collect()

            fgsm_cvb = calcCvsB(fgsm_preds)
            print(min(fgsm_cvb), max(fgsm_cvb))
            np.save(save_to+fgsm_outputCvsBdir, fgsm_cvb)
            del fgsm_cvb
            gc.collect()
            
            fgsm_cvl = calcCvsL(fgsm_preds)
            print(min(fgsm_cvl), max(fgsm_cvl))
            np.save(save_to+fgsm_outputCvsLdir, fgsm_cvl)
            del fgsm_cvl
            gc.collect()
            
            
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
            np.save(save_to+fgsm_outputPredsdir, fgsm_preds)
            del fgsm_preds
            gc.collect()
