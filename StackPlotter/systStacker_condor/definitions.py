import numpy as np

cands_per_variable = {
    'glob' : 1,
    'cpf' : 25,
    'npf' : 25,
#    'vtx' : 5,
    'vtx' : 4,
    #'pxl' : ,
}
vars_per_candidate = {
    'glob' : 15,
    'cpf' : 16,
#    'npf' : 8,
    'npf' : 6,
#    'vtx' : 14,
    'vtx' : 12,
    #'pxl' : ,
}
defaults_per_variable = {
    'glob' : [0 for i in range(vars_per_candidate['glob'])],
    'cpf' : [0 for i in range(vars_per_candidate['cpf'])],
    'npf' : [0 for i in range(vars_per_candidate['npf'])],
    'vtx' : [0 for i in range(vars_per_candidate['vtx'])],
    #'pxl' : ,
}
integer_variables_by_candidate = {
    'glob' : [2,3,4,5,8,13,14],
    'cpf' : [12,13,14,15], # in principle also 14 because apparently, chi2 is stored as Uint 
#    'npf' : [4],
    'npf' : [2],
#    'vtx' : [5],
    'vtx' : [3],
    #'pxl' : ,
}
epsilons_per_feature = {
    'glob' : '/afs/desy.de/user/h/hschonen/aisafety/VHcc-cTagSF/epsilons/global_epsilons.npy',
    'cpf' : '/afs/desy.de/user/h/hschonen/aisafety/VHcc-cTagSF/epsilons/cpf_epsilons.npy',
    'npf' : '/afs/desy.de/user/h/hschonen/aisafety/VHcc-cTagSF/epsilons/npf_epsilons.npy',
    'vtx' : '/afs/desy.de/user/h/hschonen/aisafety/VHcc-cTagSF/epsilons/vtx_epsilons.npy',
    #'pxl' : ,
}
# Global
global_names = ['Jet_pt', 'Jet_eta',
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
feature_names = global_names.copy()
# CPF
cpf = [['Jet_DeepJet_Cpfcan_BtagPf_trackEtaRel_{}'.format(i),
        'Jet_DeepJet_Cpfcan_BtagPf_trackPtRel_{}'.format(i),
        'Jet_DeepJet_Cpfcan_BtagPf_trackPPar_{}'.format(i),
        'Jet_DeepJet_Cpfcan_BtagPf_trackDeltaR_{}'.format(i),
        'Jet_DeepJet_Cpfcan_BtagPf_trackPParRatio_{}'.format(i),
        'Jet_DeepJet_Cpfcan_BtagPf_trackSip2dVal_{}'.format(i),
        'Jet_DeepJet_Cpfcan_BtagPf_trackSip2dSig_{}'.format(i),
        'Jet_DeepJet_Cpfcan_BtagPf_trackSip3dVal_{}'.format(i),
        'Jet_DeepJet_Cpfcan_BtagPf_trackSip3dSig_{}'.format(i),
        'Jet_DeepJet_Cpfcan_BtagPf_trackJetDistVal_{}'.format(i),
        'Jet_DeepJet_Cpfcan_ptrel_{}'.format(i),
        'Jet_DeepJet_Cpfcan_drminsv_{}'.format(i),
        'Jet_DeepJet_Cpfcan_VTX_ass_{}'.format(i),
        'Jet_DeepJet_Cpfcan_puppiw_{}'.format(i),
        'Jet_DeepJet_Cpfcan_chi2_{}'.format(i),
        'Jet_DeepJet_Cpfcan_quality_{}'.format(i)] for i in range(25)]
feature_names.extend([item for sublist in cpf for item in sublist])
# NPF
npf = [['Jet_DeepJet_Npfcan_ptrel_{}'.format(i),
        'Jet_DeepJet_Npfcan_deltaR_{}'.format(i),
        'Jet_DeepJet_Npfcan_isGamma_{}'.format(i),
        'Jet_DeepJet_Npfcan_HadFrac_{}'.format(i),
        'Jet_DeepJet_Npfcan_drminsv_{}'.format(i),
        'Jet_DeepJet_Npfcan_puppiw_{}'.format(i)] for i in range(25)]
feature_names.extend([item for sublist in npf for item in sublist])
# VTX
vtx = [['Jet_DeepJet_sv_pt_{}'.format(i),
        'Jet_DeepJet_sv_deltaR_{}'.format(i),
        'Jet_DeepJet_sv_mass_{}'.format(i),
        'Jet_DeepJet_sv_ntracks_{}'.format(i),
        'Jet_DeepJet_sv_chi2_{}'.format(i),
        'Jet_DeepJet_sv_normchi2_{}'.format(i),
        'Jet_DeepJet_sv_dxy_{}'.format(i),
        'Jet_DeepJet_sv_dxysig_{}'.format(i),
        'Jet_DeepJet_sv_d3d_{}'.format(i),
        'Jet_DeepJet_sv_d3dsig_{}'.format(i),
        'Jet_DeepJet_sv_costhetasvpv_{}'.format(i),
        'Jet_DeepJet_sv_enratio_{}'.format(i)] for i in range(4)]
feature_names.extend([item for sublist in vtx for item in sublist])


cpf_main_names = ['Jet_DeepJet_Cpfcan_BtagPf_trackEtaRel',
        'Jet_DeepJet_Cpfcan_BtagPf_trackPtRel',
        'Jet_DeepJet_Cpfcan_BtagPf_trackPPar',
        'Jet_DeepJet_Cpfcan_BtagPf_trackDeltaR',
        'Jet_DeepJet_Cpfcan_BtagPf_trackPParRatio',
        'Jet_DeepJet_Cpfcan_BtagPf_trackSip2dVal',
        'Jet_DeepJet_Cpfcan_BtagPf_trackSip2dSig',
        'Jet_DeepJet_Cpfcan_BtagPf_trackSip3dVal',
        'Jet_DeepJet_Cpfcan_BtagPf_trackSip3dSig',
        'Jet_DeepJet_Cpfcan_BtagPf_trackJetDistVal',
        'Jet_DeepJet_Cpfcan_ptrel',
        'Jet_DeepJet_Cpfcan_drminsv',
        'Jet_DeepJet_Cpfcan_VTX_ass',
        'Jet_DeepJet_Cpfcan_puppiw',
        'Jet_DeepJet_Cpfcan_chi2',
        'Jet_DeepJet_Cpfcan_quality']
npf_main_names = ['Jet_DeepJet_Npfcan_ptrel',
        'Jet_DeepJet_Npfcan_deltaR',
        'Jet_DeepJet_Npfcan_isGamma',
        'Jet_DeepJet_Npfcan_HadFrac',
        'Jet_DeepJet_Npfcan_drminsv',]
vtx_main_names = ['Jet_DeepJet_sv_pt',
        'Jet_DeepJet_sv_deltaR',
        'Jet_DeepJet_sv_mass',
        'Jet_DeepJet_sv_ntracks',
        'Jet_DeepJet_sv_chi2',
        'Jet_DeepJet_sv_normchi2',
        'Jet_DeepJet_sv_dxy',
        'Jet_DeepJet_sv_dxysig',
        'Jet_DeepJet_sv_d3d',
        'Jet_DeepJet_sv_d3dsig',
        'Jet_DeepJet_sv_costhetasvpv',
        'Jet_DeepJet_sv_enratio']

interesting_inputs = [
    'Jet_pt', 'Jet_eta',
   # 'Jet_DeepJet_nCpfcand','Jet_DeepJet_nNpfcand', 'Jet_DeepJet_nsv',
    'Jet_DeepCSV_trackSip2dSigAboveCharm', 'Jet_DeepCSV_trackSip3dSigAboveCharm',
    'Jet_DeepJet_Cpfcan_BtagPf_trackDeltaR_0',
    #'Jet_DeepJet_Cpfcan_BtagPf_trackDeltaR_5','Jet_DeepJet_Cpfcan_BtagPf_trackDeltaR_10','Jet_DeepJet_Cpfcan_BtagPf_trackDeltaR_24',
    'Jet_DeepJet_Cpfcan_BtagPf_trackSip2dSig_0',
    #'Jet_DeepJet_Cpfcan_BtagPf_trackSip2dSig_5','Jet_DeepJet_Cpfcan_BtagPf_trackSip2dSig_10','Jet_DeepJet_Cpfcan_BtagPf_trackSip2dSig_24',
    'Jet_DeepJet_Npfcan_ptrel_0',
    #'Jet_DeepJet_Npfcan_ptrel_5','Jet_DeepJet_Npfcan_ptrel_10','Jet_DeepJet_Npfcan_ptrel_24',
    'Jet_DeepJet_Npfcan_deltaR_0',
    #'Jet_DeepJet_Npfcan_deltaR_5','Jet_DeepJet_Npfcan_deltaR_10','Jet_DeepJet_Npfcan_deltaR_24',
   # 'Jet_DeepJet_sv_dxy_0','Jet_DeepJet_sv_dxy_1','Jet_DeepJet_sv_dxy_2','Jet_DeepJet_sv_dxy_3',
    'Jet_DeepJet_sv_mass_0',
    #'Jet_DeepJet_sv_mass_1','Jet_DeepJet_sv_mass_2','Jet_DeepJet_sv_mass_3',
    'Jet_DeepJet_sv_dxysig_0',
    #'Jet_DeepJet_sv_dxysig_1','Jet_DeepJet_sv_dxysig_2','Jet_DeepJet_sv_dxysig_3',
   # 'Jet_DeepJet_sv_d3d_0','Jet_DeepJet_sv_d3d_1','Jet_DeepJet_sv_d3d_2','Jet_DeepJet_sv_d3d_3',
   # 'Jet_DeepJet_sv_d3dsig_0','Jet_DeepJet_sv_d3dsig_1','Jet_DeepJet_sv_d3dsig_2','Jet_DeepJet_sv_d3dsig_3',
    ]

def list_flattener(input_list):
    return [item for sublist in input_list for item in sublist]

cpf_flat = list_flattener(cpf)
npf_flat = list_flattener(npf)
vtx_flat = list_flattener(vtx)

def get_group(name):
    if name in global_names:
        return 'glob'
    elif name in cpf_flat:
        return 'cpf'
    elif name in npf_flat:
        return 'npf'
    elif name in vtx_flat:
        return 'vtx'
    
def get_group_index_from_name(name):
    if name in global_names:
        return global_names.index(name), None
    elif name in cpf_flat:
        main_name = ''
        for item in name.split('_')[:-1]:
            main_name += item + '_'
        main_name = main_name[:-1]
        can_index = int(name.split('_')[-1])
        return cpf_main_names.index(main_name), can_index
    elif name in npf_flat:
        main_name = ''
        for item in name.split('_')[:-1]:
            main_name += item + '_'
        main_name = main_name[:-1]
        can_index = int(name.split('_')[-1])
        return npf_main_names.index(main_name), can_index
    elif name in vtx_flat:
        main_name = ''
        for item in name.split('_')[:-1]:
            main_name += item + '_'
        main_name = main_name[:-1]
        can_index = int(name.split('_')[-1])
        return vtx_main_names.index(main_name), can_index

def get_range_from_name(name):
    if name in global_names:
        ranges = np.load('/afs/desy.de/user/h/hschonen/aisafety/VHcc-cTagSF/Analyzer/condorDESY/auxiliary/global_ranges.npy')[global_names.index(name)]
    elif name in cpf_flat:
        main_name = ''
        for item in name.split('_')[:-1]:
            main_name += item + '_'
        main_name = main_name[:-1]
        can_index = int(name.split('_')[-1])
        ranges = np.load('/afs/desy.de/user/h/hschonen/aisafety/VHcc-cTagSF/Analyzer/condorDESY/auxiliary/cpf_ranges.npy')[cpf_main_names.index(main_name)][can_index]
    elif name in npf_flat:
        main_name = ''
        for item in name.split('_')[:-1]:
            main_name += item + '_'
        main_name = main_name[:-1]
        can_index = int(name.split('_')[-1])
        ranges = np.load('/afs/desy.de/user/h/hschonen/aisafety/VHcc-cTagSF/Analyzer/condorDESY/auxiliary/npf_ranges.npy')[npf_main_names.index(main_name)][can_index]
    elif name in vtx_flat:
        main_name = ''
        for item in name.split('_')[:-1]:
            main_name += item + '_'
        main_name = main_name[:-1]
        can_index = int(name.split('_')[-1])
        ranges = np.load('/afs/desy.de/user/h/hschonen/aisafety/VHcc-cTagSF/Analyzer/condorDESY/auxiliary/vtx_ranges.npy')[vtx_main_names.index(main_name)][can_index]
    return ranges