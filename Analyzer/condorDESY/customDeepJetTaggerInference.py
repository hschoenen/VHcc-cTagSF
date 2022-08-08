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

from definitions import *
from attacks import apply_noise, fgsm_attack



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

# empty for usage in job (will be put into scratch-dir)
save_to = ''
# not empty for testing purposes, don't clutter working dir
#save_to = '/nfs/dust/cms/user/anstein/DeepJet/test_outputs_for_BTV_meeting_adversarial/'
#save_to = '/nfs/dust/cms/user/anstein/DeepJet/test_outputs_for_BTV_meeting_nominal/'

def pfnano_to_array(rootfile, isMC):
    print('Doing cleaning, isMC = ',isMC)
    
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
    
    number_of_features = len(feature_names)
    
    if isMC == True:
        # flavour definition for PFNano based on: https://indico.cern.ch/event/739204/#3-deepjet-overview
        feature_names.extend(['Jet_FlavSplit'])
        
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

    if isMC == True:
        flavsplit = ak.to_numpy(ak.flatten(data['Jet_FlavSplit'], axis=1))
        # if the list specified below was exhaustive, the -1 would get overwritten all the time
        #target_class = np.full_like(flavsplit, -1)                                                         # initialize
        # but it isn't the case, there are undefined jet flavors, therefore set to something that could be used later (light)
        target_class = np.full_like(flavsplit, 1)                                                         # initialize
        target_class = np.where(flavsplit == 500, 0, target_class)                                                       # b
        target_class = np.where(np.bitwise_or(flavsplit == 510, flavsplit == 511), 1, target_class)                      # bb
        target_class = np.where(np.bitwise_or(flavsplit == 520, flavsplit == 521), 2, target_class)                      # leptonicb
        target_class = np.where(np.bitwise_or(flavsplit == 400, flavsplit == 410, flavsplit == 411), 3, target_class)    # c
        target_class = np.where(np.bitwise_or(flavsplit == 1, flavsplit == 2), 4, target_class)                          # uds
        target_class = np.where(flavsplit == 0, 5, target_class)                                                         # g

        datacolumns[number_of_features] = target_class
        
    datavectors = datacolumns.transpose()
    print('Jets:', len(datavectors))    
    # shape of datavectors: number of jets, number of features  +    1
    #                                             inputs           target
    #                                       (both data and MC)   (MC only)
    # Maybe ToDo: wondering whether we need to clean the features like it was done for DeepCSV, the ShallowTagInfos are contained in DeepJet inputs!
    return datavectors


def preprocess(rootfile_path, isMC):
    print('Doing starting clean/prep, isMC: ',isMC)
    #minima = np.load('/nfs/dust/cms/user/anstein/additional_files/default_value_studies_minima.npy')
    #defaults_per_variable = minima - 0.001
    #dataset_input_target = cleandataset(uproot.open(rootfile_path), defaults_per_variable, isMC)
    #print(len(dataset_input_target))
    #print(np.unique(dataset_input_target[:,-1]))
    #sys.exit()
    
    dataset_input_target = pfnano_to_array(uproot.open(rootfile_path), isMC)
    
    # targets only make sense for MC,
    # but nothing 'breaks' when calling it on Data (the last column is different though, it's all Zeros, see definition above)
    targets = torch.Tensor(dataset_input_target[:,-1]).long()
    
    inputs = torch.Tensor(dataset_input_target[:,0:-1])
    
    del dataset_input_target
    gc.collect()
    
    
    slice_glob = cands_per_variable['glob'] * vars_per_candidate['glob']
    slice_cpf = cands_per_variable['cpf'] * vars_per_candidate['cpf']
    slice_npf = cands_per_variable['npf'] * vars_per_candidate['npf']
    slice_vtx = cands_per_variable['vtx'] * vars_per_candidate['vtx']
    
    glob = inputs[:,0:slice_glob]
    cpf  = inputs[:,slice_glob:slice_glob+slice_cpf]
    npf  = inputs[:,slice_glob+slice_cpf:slice_glob+slice_cpf+slice_npf]
    vtx  = inputs[:,slice_glob+slice_cpf+slice_npf:slice_glob+slice_cpf+slice_npf+slice_vtx]

    #print(glob.shape)
    #print(cpf.shape)
    #print(npf.shape)
    #print(vtx.shape)
    
    cpf = cpf.reshape((-1,cands_per_variable['cpf'],vars_per_candidate['cpf']))
    npf = npf.reshape((-1,cands_per_variable['npf'],vars_per_candidate['npf']))
    vtx = vtx.reshape((-1,cands_per_variable['vtx'],vars_per_candidate['vtx']))
    return glob,cpf,npf,vtx, targets


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
        elif 'adversarial_eps0p005' in model_name:
            modelpath = f'/nfs/dust/cms/user/anstein/DeepJet/Train_{tagger}/adversarial_eps0p005/checkpoint_best_loss.pth'

        checkpoint = torch.load(modelpath, map_location=torch.device(device))
        model.load_state_dict(checkpoint["state_dict"])

        model.to(device)

        #evaluate network on inputs
        model.eval()
        print('successfully loaded model and checkpoint')
        # note: unlike most other models, the pytorch version of DeepJet does not output "probabilities", but the last step needs to be applied on top of the logits
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
    custom_BvL[custom_BvL > 0.99999] = 0.99999
    
    return custom_BvL


def calcBvsC(matching_predictions):
    custom_BvC = np.where( ((matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,2]) > 0.) \
                          & ((matching_predictions[:,3]) > 0.) \
                          & (matching_predictions[:,4]+matching_predictions[:,5] >= 0.),
                          (matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,2])/(matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,2]+matching_predictions[:,3]),
                          (-1.0)*np.ones(n_jets))
    
    custom_BvC[(custom_BvC < 0.000001) & (custom_BvC > -0.000001)] = 0.000001
    custom_BvC[(np.isnan(custom_BvC)) & (np.isinf(custom_BvC))] = -1.0
    custom_BvC[custom_BvC > 0.99999] = 0.99999
    
    return custom_BvC
    
    
def calcCvsB(matching_predictions):
    custom_CvB = np.where( ((matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,2]) > 0.) \
                          & (matching_predictions[:,3] > 0.) \
                          & (matching_predictions[:,4]+matching_predictions[:,5] >= 0.),
                          (matching_predictions[:,3])/(matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,2]+matching_predictions[:,3]),
                          (-1.0)*np.ones(n_jets))
    
    custom_CvB[(custom_CvB < 0.000001) & (custom_CvB > -0.000001)] = 0.000001
    custom_CvB[(np.isnan(custom_CvB)) & (np.isinf(custom_CvB))] = -1.0
    custom_CvB[custom_CvB > 0.99999] = 0.99999
    
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
    custom_CvL[custom_CvL > 0.99999] = 0.99999
    
    return custom_CvL


if __name__ == "__main__":
    fullName, model_name, condoroutdir = sys.argv[1], sys.argv[2], sys.argv[3]
    
    parentDir = ""
    # default era, will be overwritten
    era = 2016

    
    print("Will open file %s."%(fullName))
    
    # ToDo: modify once I have my private samples ready
    #parentDirList = ["VHcc_2017V5_Dec18/","NanoCrabProdXmas/","/2016/","2016_v2/","/2017/","2017_v2","/2018/","VHcc_2016V4bis_Nov18/"]
    #parentDirList = ["/106X_v2_17/","/106X_v2_17rsb2/","/106X_v2_17rsb3/"]
    parentDirList = ["/nanotest_add_DeepJet/","/PFNano/"]
    for iParent in parentDirList:
        if iParent in fullName: parentDir = iParent
    if parentDir == "": fullName.split('/')[8]+"/"
    
    
    if "2017" in fullName: era = 2017
    if "2018" and not "2017" in fullName: era = 2018   # this is needed because both 2017 and 2018 appear in the new file names
    
    sampName=fullName.split(parentDir)[1].split('/')[0]
    channel=sampName
    print('sampName:', sampName)
#    sampNo=fullName.split(parentDir)[1].split('/')[1].split('_')[-1]
# new version to prevent overwriting of files for PFNano filenames
# splitting here at the underscore would destroy uniqueness (example: Data for DY, DoubleMuon)
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
    
    
    # OLD: inputs, targets, scalers = preprocess(fullName, isMC)
    # OLD: inputs, targets, scalers = preprocess('infile.root', isMC)
    # Wanted, once I can start using the runscript:
    # NEW:
    glob,cpf,npf,vtx, targets = preprocess('infile.root', isMC)
    # WIP tests
    #glob,cpf,npf,vtx, targets = preprocess('~/private/pfnano_dev/CMSSW_10_6_20/src/PhysicsTools/PFNano/test/nano106Xv8_on_mini106X_2017_mc_NANO_py_NANO_AddDeepJet.root', True)
    #glob,cpf,npf,vtx, targets = preprocess('root://grid-cms-xrootd.physik.rwth-aachen.de:1094//store/user/anstein/PFNano/DoubleMuon/Run2017B-09Aug2019_UL2017-v1_PFtestNano/220806_223005/0000/nano_data2017_1.root', isMC)
    #glob,cpf,npf,vtx, targets = preprocess('root://grid-cms-xrootd.physik.rwth-aachen.de:1094//store/user/anstein/PFNano/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer19UL17MiniAOD-106X_mc2017_realistic_v6-v2_PFtestNano/220807_184642/0000/nano_mc2017_1-1.root', isMC)
    #sys.exit()
    n_jets = len(targets)
    
    
    outputTargetsdir  = "outTargets_%s.npy"%(outNo)
    np.save(save_to+outputTargetsdir, targets)
    
    # to check multiple epochs of a given weighting method at once (using always 3 epochs should make sense, as previous tests were done on raw/noise/FGSM = 3 different sets)
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
            n_chunks = len(range(0,n_jets,15000))
            #print(n_chunks)
            for i,k in enumerate(range(0,n_jets,15000)):
                print(i,k)
                if i == 0:
                    predictions, _ = predict(glob[k:k+15000],cpf[k:k+15000],npf[k:k+15000],vtx[k:k+15000], model_i)
                elif i == n_chunks-1:
                    current_predictions, _ = predict(glob[k:n_jets],cpf[k:n_jets],npf[k:n_jets],vtx[k:n_jets], model_i)
                    np.concatenate((predictions,current_predictions))
                else:
                    current_predictions, _ = predict(glob[k:k+15000],cpf[k:k+15000],npf[k:k+15000],vtx[k:k+15000], model_i)
                    np.concatenate((predictions,current_predictions))
            #print(n_jets, 'matches', len(predictions))
            
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
        n_chunks = len(range(0,n_jets,15000))
        #print(n_chunks)
        for i,k in enumerate(range(0,n_jets,15000)):
            #print(i,k)
            if i == 0:
                predictions, thismodel = predict(glob[k:k+15000],cpf[k:k+15000],npf[k:k+15000],vtx[k:k+15000], model_name)
            elif i == n_chunks-1:
                current_predictions, _ = predict(glob[k:n_jets],cpf[k:n_jets],npf[k:n_jets],vtx[k:n_jets], model_name)
                predictions = np.concatenate((predictions,current_predictions))
            else:
                current_predictions, _ = predict(glob[k:k+15000],cpf[k:k+15000],npf[k:k+15000],vtx[k:k+15000], model_name)
                predictions = np.concatenate((predictions,current_predictions))
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
            
            n_chunks = len(range(0,n_jets,8000))
            #print(n_chunks)
            for i,k in enumerate(range(0,n_jets,8000)):
                #print(i,k)
                if i == 0:
                    noise_preds, _ = predict(apply_noise(glob[k:k+8000],magn=1e-2,offset=[0],restrict_impact=0.2,var_group='glob'),
                                                     apply_noise(cpf[k:k+8000], magn=1e-2,offset=[0],restrict_impact=0.2,var_group='cpf'),
                                                     apply_noise(npf[k:k+8000], magn=1e-2,offset=[0],restrict_impact=0.2,var_group='npf'),
                                                     apply_noise(vtx[k:k+8000], magn=1e-2,offset=[0],restrict_impact=0.2,var_group='vtx'),
                                                     model_name)
                elif i == n_chunks-1:
                    current_noise_preds, _ = predict(apply_noise(glob[k:n_jets],magn=1e-2,offset=[0],restrict_impact=0.2,var_group='glob'),
                                                     apply_noise(cpf[k:n_jets], magn=1e-2,offset=[0],restrict_impact=0.2,var_group='cpf'),
                                                     apply_noise(npf[k:n_jets], magn=1e-2,offset=[0],restrict_impact=0.2,var_group='npf'),
                                                     apply_noise(vtx[k:n_jets], magn=1e-2,offset=[0],restrict_impact=0.2,var_group='vtx'),
                                                     model_name)
                    noise_preds = np.concatenate((noise_preds,current_noise_preds))
                else:
                    current_noise_preds, _ = predict(apply_noise(glob[k:k+8000],magn=1e-2,offset=[0],restrict_impact=0.2,var_group='glob'),
                                                     apply_noise(cpf[k:k+8000], magn=1e-2,offset=[0],restrict_impact=0.2,var_group='cpf'),
                                                     apply_noise(npf[k:k+8000], magn=1e-2,offset=[0],restrict_impact=0.2,var_group='npf'),
                                                     apply_noise(vtx[k:k+8000], magn=1e-2,offset=[0],restrict_impact=0.2,var_group='vtx'),
                                                     model_name)
                    noise_preds = np.concatenate((noise_preds,current_noise_preds))
                
                
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

            
            n_chunks = len(range(0,n_jets,8000))
            #print(n_chunks)
            for i,k in enumerate(range(0,n_jets,8000)):
                #print(i,k)
                if i == 0:
                    glob_fgsm, cpf_fgsm, npf_fgsm, vtx_fgsm = fgsm_attack(epsilon=1e-2,sample=(glob[k:k+8000],cpf[k:k+8000],npf[k:k+8000],vtx[k:k+8000]),
                                                                          targets=targets[k:k+8000],thismodel=thismodel,thiscriterion=cross_entropy,reduced=True,restrict_impact=0.2)
                    fgsm_preds, _ = predict(glob_fgsm, cpf_fgsm, npf_fgsm, vtx_fgsm, model_name)
                    del glob_fgsm
                    del cpf_fgsm
                    del npf_fgsm
                    del vtx_fgsm
                    gc.collect()
                elif i == n_chunks-1:
                    glob_fgsm, cpf_fgsm, npf_fgsm, vtx_fgsm = fgsm_attack(epsilon=1e-2,sample=(glob[k:n_jets],cpf[k:n_jets],npf[k:n_jets],vtx[k:n_jets]),
                                                                          targets=targets[k:n_jets],thismodel=thismodel,thiscriterion=cross_entropy,reduced=True,restrict_impact=0.2)
                    current_fgsm_preds, _ = predict(glob_fgsm, cpf_fgsm, npf_fgsm, vtx_fgsm, model_name)
                    del glob_fgsm
                    del cpf_fgsm
                    del npf_fgsm
                    del vtx_fgsm
                    gc.collect()
                    fgsm_preds = np.concatenate((fgsm_preds,current_fgsm_preds))
                else:
                    glob_fgsm, cpf_fgsm, npf_fgsm, vtx_fgsm = fgsm_attack(epsilon=1e-2,sample=(glob[k:k+8000],cpf[k:k+8000],npf[k:k+8000],vtx[k:k+8000]),
                                                                          targets=targets[k:k+8000],thismodel=thismodel,thiscriterion=cross_entropy,reduced=True,restrict_impact=0.2)
                    current_fgsm_preds, _ = predict(glob_fgsm, cpf_fgsm, npf_fgsm, vtx_fgsm, model_name)
                    del glob_fgsm
                    del cpf_fgsm
                    del npf_fgsm
                    del vtx_fgsm
                    gc.collect()
                    fgsm_preds = np.concatenate((fgsm_preds,current_fgsm_preds))
                    
            
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
