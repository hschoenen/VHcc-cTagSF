from definitions import get_range_from_name
import sys
import numpy as np
sys.path.append('~/aisafety/VHcc-cTagSF/Analyzer/condorDESY/auxiliary/')

# choose adversarial model
adversarial_model_name = 'fgsm-0_1'

outDir = "output_2017_PFNano" #"190928_2017"
DYPath = f"/nfs/dust/cms/user/hschonen/DataMC/{adversarial_model_name}/2017_DY_DeepJet_Run2_COMPLETE/"
WcPath = f"/nfs/dust/cms/user/hschonen/DataMC/{adversarial_model_name}/2017_Wc_DeepJet_Run2_COMPLETE/" 

systs = [
         "central",
         "PUWeight_up","PUWeight_down","MuIDSF_up","MuIDSF_down","EleIDSF_up","EleIDSF_down",
         "LHEScaleWeight_muR_up","LHEScaleWeight_muR_down","LHEScaleWeight_muF_up","LHEScaleWeight_muF_down",
         "PSWeightISR_up","PSWeightISR_down","PSWeightFSR_up","PSWeightFSR_down",
         #"jesTotalUp","jesTotalDown","jerUp","jerDown",
         "XSec_WJets_up","XSec_WJets_down","XSec_DYJets_up","XSec_DYJets_down","XSec_ST_up","XSec_ST_down",
         "XSec_ttbar_up", "XSec_ttbar_down", "XSec_BRUnc_DYJets_b_up", "XSec_BRUnc_DYJets_b_down", "XSec_BRUnc_DYJets_c_up", "XSec_BRUnc_DYJets_c_down", "XSec_BRUnc_WJets_c_up", "XSec_BRUnc_WJets_c_down"
         ]

SFfilesDeepCSV = ["","DeepCSV_ctagSF_MiniAOD94X_2017_pTincl_v3_2.root"]
SFfilesDeepJet = ["","DeepJet_ctagSF_MiniAOD94X_2017_pTincl_v3_2.root"]
SFhistSuff = [""] #"_ValuesSystOnlyUp","_ValuesSystOnlyDown"]   # "" for nominal

plotExtra = False
plotsysts = False
plotBinSlices = False #True  # needed if one wants to derive SFs later
validateSFs = False
addsel = '' #'&& jet_CvsL[max(0.,muJet_idx)] > 0.8 && jet_CvsB[max(0.,muJet_idx)] > 0.1'
 #'&& jet_Pt[max(0.,muJet_idx)] > 80 && jet_Pt[max(0.,muJet_idx)] < 10000'

muBiasTestIndex = '(muJet_idx==0?1:0)'
# muBiasTestIndex = 'getBJetIdx(jet_DeepFlavCvsB,muJet_idx)'
# muBiasTestIndex = 'getCJetIdx(jet_DeepFlavCvsB,jet_DeepFlavCvsL,muJet_idx)'
'''
nBinDisc = 60  # was 30 before, now using 60 to be consistent with previous runs (that also plotted the bin slices and therefore used 60 bins)
if plotBinSlices: nBinDisc = 60 
'''
nBinDisc = 30  # back to 30 for the moment, to account for low stats with stat. fluctuations for Wc & TTSEMI sel, 
# to be revisited once all crab jobs finished
if plotBinSlices: nBinDisc = 30 # 30 is anyway used for BTV-20-001, so should be fine

plotBinnedKins = False
normMCtodata = True # should make blue and red line in ratio overlap at exactly 1
plot2D = False

outDir = outDir.rstrip('/')

interesting_feature = 'Jet_DeepJet_Npfcan_deltaR_0'
interesting_ranges = get_range_from_name(interesting_feature)
interesting_feature_LEFT_BOUND,interesting_feature_RIGHT_BOUND = str(interesting_ranges[0]) , str(interesting_ranges[1])

def applyCuts(ln,reg="",pt_eta_domain=[0,0,0,0]):
    ln = ln.replace('ZMASSCUT','[85,95,\"invert\"]')
    ln = ln.replace('CVXBINNING','varBin1=[-0.2,0.,0.2,0.4,0.6,0.8,1.],varBin2=[-0.2,0.,0.2,0.4,0.6,0.8,1.]')
    ln = ln.replace('JETIDX',muBiasTestIndex)
    if "central" in syst:
        ln = ln.replace('TTSEMIWEIGHT','MCWeightName="eventWeightUnsigned",DataWeightName="eventWeightUnsigned",yTitle="Jet yield",outDir="OUTDIR_SYSTNAME",rootPath="WCPATH"')
        #ln = ln.replace('TTWEIGHT','MCWeightName="eventWeight",DataWeightName="eventWeight",yTitle="Jet yield",outDir="OUTDIR_SYSTNAME",rootPath="TTPATH"')
        ln = ln.replace('WCWEIGHT','MCWeightName="eventWeight",DataWeightName="eventWeight",yTitle="Jet yield, OS-SS subtracted",outDir="OUTDIR_SYSTNAME",rootPath="WCPATH"')
        ln = ln.replace('DYWEIGHT','MCWeightName="eventWeight",DataWeightName="eventWeight",yTitle="Jet yield",outDir="OUTDIR_SYSTNAME",rootPath="DYPATH"')
    elif syst.startswith("je"):
        ln = ln.replace('TTSEMIWEIGHT','MCWeightName="eventWeightUnsigned",DataWeightName="eventWeightUnsigned",yTitle="Jet yield",outDir="OUTDIR_SYSTNAME",pathSuff="_SYSTNAME",rootPath="WCPATH"')
        #ln = ln.replace('TTWEIGHT','MCWeightName="eventWeight",DataWeightName="eventWeight",yTitle="Jet yield",outDir="OUTDIR_SYSTNAME",pathSuff="_SYSTNAME",rootPath="TTPATH"')
        ln = ln.replace('WCWEIGHT','MCWeightName="eventWeight",DataWeightName="eventWeight",yTitle="Jet yield, OS-SS subtracted",outDir="OUTDIR_SYSTNAME",pathSuff="_SYSTNAME",rootPath="WCPATH"')
        ln = ln.replace('DYWEIGHT','MCWeightName="eventWeight",DataWeightName="eventWeight",yTitle="Jet yield",outDir="OUTDIR_SYSTNAME",rootPath="DYPATH",pathSuff="_SYSTNAME"')
    elif "XSec" in syst:
        ln = ln.replace('TTSEMIWEIGHT','MCWeightName="eventWeightUnsigned",DataWeightName="eventWeightUnsigned",yTitle="Jet yield",outDir="OUTDIR_SYSTNAME",useXSecUnc="SYSTNAME",rootPath="WCPATH"')
        #ln = ln.replace('TTWEIGHT','MCWeightName="eventWeight",DataWeightName="eventWeight",yTitle="Jet yield",outDir="OUTDIR_SYSTNAME",useXSecUnc="SYSTNAME",rootPath="TTPATH"')
        ln = ln.replace('WCWEIGHT','MCWeightName="eventWeight",DataWeightName="eventWeight",yTitle="Jet yield, OS-SS subtracted",outDir="OUTDIR_SYSTNAME",useXSecUnc="SYSTNAME",rootPath="WCPATH"')
        ln = ln.replace('DYWEIGHT','MCWeightName="eventWeight",DataWeightName="eventWeight",yTitle="Jet yield",outDir="OUTDIR_SYSTNAME",rootPath="DYPATH",useXSecUnc="SYSTNAME"')
    else:
        ln = ln.replace('TTSEMIWEIGHT','MCWeightName="eventWeightUnsigned*SYSTNAME",DataWeightName="eventWeightUnsigned",yTitle="Jet yield",outDir="OUTDIR_SYSTNAME",rootPath="WCPATH"')
        #ln = ln.replace('TTWEIGHT','MCWeightName="eventWeight*SYSTNAME",DataWeightName="eventWeight",yTitle="Jet yield",outDir="OUTDIR_SYSTNAME",rootPath="TTPATH"')
        ln = ln.replace('WCWEIGHT','MCWeightName="eventWeight*SYSTNAME",DataWeightName="eventWeight",yTitle="Jet yield, OS-SS subtracted",outDir="OUTDIR_SYSTNAME",rootPath="WCPATH"')
        ln = ln.replace('DYWEIGHT','MCWeightName="eventWeight*SYSTNAME",DataWeightName="eventWeight",yTitle="Jet yield",outDir="OUTDIR_SYSTNAME",rootPath="DYPATH"')
    
    ln = ln.replace('NBINDISC',str(nBinDisc))
    ln = ln.replace('OUTDIR',outDir)
    ln = ln.replace('SYSTNAME',syst)
    ln = ln.replace('DYPATH',DYPath)
    #ln = ln.replace('TTPATH',TTPath)
    ln = ln.replace('WCPATH',WcPath)

    ln = ln.replace('ESEL','filePre="Wc_e", selections="is_E == 1 && jetMuPt_by_jetPt < 0.6 && jet_nJet < 4 && diLepVeto == 0 QCDSELE UNISEL"')
    ln = ln.replace('MSEL','filePre="Wc_m", selections="is_M == 1 && jetMuPt_by_jetPt < 0.4 && jet_nJet < 4 && diLepVeto == 0 && (Z_Mass_best < 80 || Z_Mass_best > 100) && (Z_Mass_min > 12 || Z_Mass_min < 0) && jet_muplusneEmEF[muJet_idx] < 0.7 && jetMu_iso > 0.5 QCDSELM UNISEL"')
    #ln = ln.replace('MSEL','filePre="Wc_m", selections="is_M == 1 && jetMuPt_by_jetPt < 0.4 && jet_nJet < 4 && diLepVeto == 0 && Z_Mass_best > 0 &&  Z_Mass_min > 12 && jet_muplusneEmEF[muJet_idx] < 0.7 && jetMu_iso > 0.5 QCDSELM UNISEL"')
   
    ln = ln.replace('TTSEMISELE','filePre="TT_semie", selections="is_E == 1 && jetMuPt_by_jetPt < 0.6 && jet_nJet > 3 && diLepVeto == 0 && jetMu_Pt > 5 && jetMu_Pt < 25 QCDSELE UNISEL"')
    ln = ln.replace('TTSEMISELM','filePre="TT_semim", selections="is_M == 1 && jetMuPt_by_jetPt < 0.4 && jet_nJet > 3 && diLepVeto == 0 && (Z_Mass < 85 || Z_Mass > 95) && (Z_Mass > 12 || Z_Mass < 0) && jetMu_Pt > 5 && jetMu_Pt < 25 QCDSELM UNISEL"')

    ln = ln.replace('TTSELME','filePre="TT_me", selections="is_ME == 1 UNISEL"')
    ln = ln.replace('TTSELMM','filePre="TT_mm", selections="is_MM == 1 && (Z_Mass < 75 || Z_Mass > 105) && (Z_Mass > 12 || Z_Mass < 0) && met_Pt > 40 UNISEL"')
    ln = ln.replace('TTSELEE','filePre="TT_ee", selections="is_EE == 1 && (Z_Mass < 75 || Z_Mass > 105) && (Z_Mass > 12 || Z_Mass < 0) && met_Pt > 40 UNISEL"')
    
    ln = ln.replace('DYSELM','filePre="DY_m", selections="is_M == 1 && M_Pt[0] > 20 && Z_Pt > 15 UNISEL"')
    ln = ln.replace('DYSELE','filePre="DY_e", selections="is_E == 1 && E_Pt[0] > 27 && Z_Pt > 15 UNISEL"')
    
    ln = ln.replace('QCDSELE','&& E_RelIso[0] < 0.05 && (hardE_Jet_PtRatio  > 0.75 || hardE_Jet_PtRatio  < 0.) && abs(E_dz[0]) < 0.02 && abs(E_dxy[0]) < 0.01  && E_sip3d[0] < 2.5')
    ln = ln.replace('QCDSELM','&& M_RelIso[0] < 0.05 && (hardMu_Jet_PtRatio > 0.75 || hardMu_Jet_PtRatio < 0.) && abs(M_dz[0]) < 0.01 && abs(M_dxy[0]) < 0.002 && M_sip3d[0] < 2')
    
    ln = ln.replace('UNISEL',addsel)
    ln = ln.replace('UNICUT','')

    if not reg=="": ln = ln.replace('REG',reg)
    
    # selection for jet_pt and jet_eta
    if np.sum(pt_eta_domain)!=0:
        jet_index = 'muJet_idx'
        if 'DY_m' in ln:
            jet_index = '0'
        pt_eta_sel = f'selections="jet_Pt[{jet_index}] > {pt_eta_domain[0]} && jet_Pt[{jet_index}] < {pt_eta_domain[1]} && jet_Eta[{jet_index}] > {pt_eta_domain[2]} && jet_Eta[{jet_index}] < {pt_eta_domain[3]} && '
        ln = ln.replace('selections="',pt_eta_sel)
    return ln

arguments = '''
           "jet_CvsL[muJet_idx]","CvsL",6,-0.1,1,MSEL,dataset="smu",brName2D=["jet_CvsB[muJet_idx]"],brLabel2="CvsB",nbins2=6,CVXBINNING,drawStyle="",makeROOT=True,WCWEIGHT
          # "jet_CvsL[muJet_idx]","CvsL",6,-0.1,1,ESEL,dataset="sele",brName2D=["jet_CvsB[muJet_idx]"],brLabel2="CvsB",nbins2=6,CVXBINNING,drawStyle="",makeROOT=True,WCWEIGHT

            "jet_CvsL[muJet_idx]","CvsL",6,-0.1,1,TTSEMISELM,dataset="smu",brName2D=["jet_CvsB[muJet_idx]"],brLabel2="CvsB",nbins2=6,CVXBINNING,drawStyle="",makeROOT=True,TTSEMIWEIGHT
          #  "jet_CvsL[muJet_idx]","CvsL",6,-0.1,1,TTSEMISELE,dataset="sele",brName2D=["jet_CvsB[muJet_idx]"],brLabel2="CvsB",nbins2=6,CVXBINNING,drawStyle="",makeROOT=True,TTSEMIWEIGHT
           
          #  "jet_CvsL[muJet_idx]","CvsL",6,-0.1,1,TTSELMM,dataset="dmu",brName2D=["jet_CvsB[muJet_idx]"],brLabel2="CvsB",nbins2=6,CVXBINNING,drawStyle="",makeROOT=True,TTWEIGHT
          #  "jet_CvsL[muJet_idx]","CvsL",6,-0.1,1,TTSELEE,dataset="deg",brName2D=["jet_CvsB[muJet_idx]"],brLabel2="CvsB",nbins2=6,CVXBINNING,drawStyle="",makeROOT=True,TTWEIGHT
          #  "jet_CvsL[muJet_idx]","CvsL",6,-0.1,1,TTSELME,dataset="mue",brName2D=["jet_CvsB[muJet_idx]"],brLabel2="CvsB",nbins2=6,CVXBINNING,drawStyle="",makeROOT=True,TTWEIGHT
           
            "jet_CvsL[0]","CvsL",6,-0.1,1,DYSELM,dataset="dmu",brName2D=["jet_CvsB[0]"],brLabel2="CvsB",nbins2=6,CVXBINNING,drawStyle="",makeROOT=True,DYWEIGHT
          #  "jet_CvsL[0]","CvsL",6,-0.1,1,DYSELE,dataset="deg",brName2D=["jet_CvsB[0]"],brLabel2="CvsB",nbins2=6,CVXBINNING,drawStyle="",makeROOT=True,DYWEIGHT
'''

plot1D = '''
          # All jets
          # Nominal Deepjet Model
         "jet_CustomCvsL[muJet_idx]",r"DeepJet (Nominal Training) CvsL",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
         "jet_CustomCvsB[muJet_idx]",r"DeepJet (Nominal Training) CvsB",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
         "jet_CustomCvsL[muJet_idx]",r"DeepJet (Nominal Training) CvsL",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
         "jet_CustomCvsB[muJet_idx]",r"DeepJet (Nominal Training) CvsB",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
         "jet_CustomCvsL[0]",r"DeepJet (Nominal Training) CvsL",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
         "jet_CustomCvsB[0]",r"DeepJet (Nominal Training) CvsB",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
         "jet_CustomBvsL[muJet_idx]",r"DeepJet (Nominal Training) BvsL",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
         "jet_CustomBvsC[muJet_idx]",r"DeepJet (Nominal Training) BvsC",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
         "jet_CustomBvsL[muJet_idx]",r"DeepJet (Nominal Training) BvsL",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
         "jet_CustomBvsC[muJet_idx]",r"DeepJet (Nominal Training) BvsC",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
         "jet_CustomBvsL[0]",r"DeepJet (Nominal Training) BvsL",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
         "jet_CustomBvsC[0]",r"DeepJet (Nominal Training) BvsC",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
          # Adversarial Deepjet Model
        "jet_CustomADVCvsL[muJet_idx]",r"DeepJet (Adversarial Training) CvsL",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
        "jet_CustomADVCvsB[muJet_idx]",r"DeepJet (Adversarial Training) CvsB",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
        "jet_CustomADVCvsL[muJet_idx]",r"DeepJet (Adversarial Training) CvsL",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
        "jet_CustomADVCvsB[muJet_idx]",r"DeepJet (Adversarial Training) CvsB",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
        "jet_CustomADVCvsL[0]",r"DeepJet (Adversarial Training) CvsL",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
        "jet_CustomADVCvsB[0]",r"DeepJet (Adversarial Training) CvsB",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
        "jet_CustomADVBvsL[muJet_idx]",r"DeepJet (Adversarial Training) BvsL",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
        "jet_CustomADVBvsC[muJet_idx]",r"DeepJet (Adversarial Training) BvsC",NBINDISC,-0.2,1,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
        "jet_CustomADVBvsL[muJet_idx]",r"DeepJet (Adversarial Training) BvsL",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
        "jet_CustomADVBvsC[muJet_idx]",r"DeepJet (Adversarial Training) BvsC",NBINDISC,-0.2,1,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
        "jet_CustomADVBvsL[0]",r"DeepJet (Adversarial Training) BvsL",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
        "jet_CustomADVBvsC[0]",r"DeepJet (Adversarial Training) BvsC",NBINDISC,-0.2,1,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT
'''

onlyCentral = '''
        #    "jetMu_Pt",r"p^{soft #mu}_{T} [GeV] (mu)",25,0,25,MSEL,dataset="smu",WCWEIGHT
        #    "jetMu_Pt",r"p^{soft #mu}_{T} [GeV] (e)",25,0,25,ESEL,dataset="sele",WCWEIGHT
        #    "jet_Pt[muJet_idx]",r"p^{jet}_{T} [GeV] (mu)",25,20,120,MSEL,dataset="smu",makeROOT=True,WCWEIGHT
        #    "jet_Pt[muJet_idx]",r"p^{jet}_{T} [GeV] (e)",25,20,120,ESEL,dataset="sele",makeROOT=True,WCWEIGHT
        #    "jet_Eta[muJet_idx]",r"#eta_{jet} (mu)",20,-2.8,2.8,MSEL,dataset="smu",WCWEIGHT
        #    "jet_Eta[muJet_idx]",r"#eta_{jet} (e)",20,-2.8,2.8,ESEL,dataset="sele",WCWEIGHT
        #    "jet_Phi[muJet_idx]",r"#phi_{jet} (mu)",20,-3.2,3.2,MSEL,dataset="smu",WCWEIGHT
        #    "jet_Phi[muJet_idx]",r"#phi_{jet} (e)",20,-3.2,3.2,ESEL,dataset="sele",WCWEIGHT            
            
        #    "nTightMu",r"Number of tight #mu", 5,0,5,MSEL,dataset="smu",WCWEIGHT
        #    "Z_Mass_withJet","M_{#mu,jet}",40,0,120,MSEL,dataset="smu",WCWEIGHT
              
        #     "M_RelIso[0]","Rel Iso (mu)",40,0,0.08,MSEL,dataset="smu",makeROOT=True,nminus1=True,WCWEIGHT
        #     "E_RelIso[0]","Rel Iso (e)",40,0,0.08,ESEL,dataset="sele",makeROOT=True,nminus1=True,WCWEIGHT
        #     "M_dz[0]",r"M_dz (mu)",40,0,0.02,MSEL,dataset="smu",makeROOT=True,nminus1=True,WCWEIGHT
        #     "E_dz[0]",r"E_dz (e)",40,0,0.04,ESEL,dataset="sele",makeROOT=True,nminus1=True,WCWEIGHT
             "M_dxy[0]",r"M_dxy (mu)",40,0,0.004,MSEL,dataset="smu",makeROOT=True,nminus1=True,WCWEIGHT
        #     "E_dxy[0]",r"E_dxy (e)",40,0,0.02,ESEL,dataset="sele",makeROOT=True,nminus1=True,WCWEIGHT

        #     "jet_nJet",r"nJet (mu)",6,1,7,MSEL,dataset="smu",makeROOT=True,nminus1=True,WCWEIGHT
        #     "jet_nJet",r"nJet (e)",6,1,7,ESEL,dataset="sele",makeROOT=True,nminus1=True,WCWEIGHT
            # "jet_nJet",r"nJet (mu)",6,1,7,TTSEMISELM,dataset="smu",nminus1=True,TTSEMIWEIGHT
            # "jet_nJet",r"nJet (e)",6,1,7,TTSEMISELE,dataset="sele",nminus1=True,TTSEMIWEIGHT


        #    "jetMu_Pt",r"p^{soft #mu}_{T} [GeV] (mu)",25,0,25,TTSEMISELM,dataset="smu",TTSEMIWEIGHT
        #    "jetMu_Pt",r"p^{soft #mu}_{T} [GeV] (e)",25,0,25,TTSEMISELE,dataset="sele",TTSEMIWEIGHT
        #    "jet_Pt[muJet_idx]",r"p^{jet}_{T} [GeV] (mu)",25,20,120,TTSEMISELM,dataset="smu",makeROOT=True,TTSEMIWEIGHT
        #    "jet_Pt[muJet_idx]",r"p^{jet}_{T} [GeV] (e)",25,20,120,TTSEMISELE,dataset="sele",makeROOT=True,TTSEMIWEIGHT
        #    "jet_Eta[muJet_idx]",r"#eta_{jet} (mu)",20,-2.8,2.8,TTSEMISELM,dataset="smu",TTSEMIWEIGHT
        #    "jet_Eta[muJet_idx]",r"#eta_{jet} (e)",20,-2.8,2.8,TTSEMISELE,dataset="sele",TTSEMIWEIGHT
        #    "jet_Phi[muJet_idx]",r"#phi_{jet} (mu)",20,-3.2,3.2,TTSEMISELM,dataset="smu",TTSEMIWEIGHT
        #    "jet_Phi[muJet_idx]",r"#phi_{jet} (e)",20,-3.2,3.2,TTSEMISELE,dataset="sele",TTSEMIWEIGHT           
              
          
          
        #     "jetMu_Pt",r"p^{soft #mu}_{T} [GeV] (#mu #mu)",25,0,25,TTSELMM,dataset="dmu",TTWEIGHT
        #     "jetMu_Pt",r"p^{soft #mu}_{T} [GeV] (e e)",25,0,25,TTSELEE,dataset="deg",TTWEIGHT
        #     "jetMu_Pt",r"p^{soft #mu}_{T} [GeV] (#mu e)",25,0,25,TTSELME,dataset="mue",TTWEIGHT
        #    "jet_Pt[muJet_idx]",r"p^{jet}_{T} [GeV] (#mu #mu)",25,20,120,TTSELMM,dataset="dmu",makeROOT=True,TTWEIGHT
        #    "jet_Pt[muJet_idx]",r"p^{jet}_{T} [GeV] (e e)",25,20,120,TTSELEE,dataset="deg",makeROOT=True,TTWEIGHT
        #    "jet_Pt[muJet_idx]",r"p^{jet}_{T} [GeV] (#mu e)",25,20,120,TTSELME,dataset="mue",makeROOT=True,TTWEIGHT
        #    "jet_Eta[muJet_idx]",r"#eta_{jet} (#mu #mu)",20,-2.8,2.8,TTSELMM,dataset="dmu",TTWEIGHT
        #    "jet_Eta[muJet_idx]",r"#eta_{jet} (e e)",20,-2.8,2.8,TTSELEE,dataset="deg",TTWEIGHT
        #    "jet_Eta[muJet_idx]",r"#eta_{jet} (#mu e)",20,-2.8,2.8,TTSELME,dataset="mue",TTWEIGHT
        #    "jet_Phi[muJet_idx]",r"#phi_{jet} (#mu #mu)",20,-3.2,3.2,TTSELMM,dataset="dmu",TTWEIGHT
        #    "jet_Phi[muJet_idx]",r"#phi_{jet} (e e)",20,-3.2,3.2,TTSELEE,dataset="deg",TTWEIGHT
        #    "jet_Phi[muJet_idx]",r"#phi_{jet} (#mu e)",20,-3.2,3.2,TTSELME,dataset="mue",TTWEIGHT      
         
         
       #     "jet_Phi[0]",r"#phi_{jet}",20,-3.2,3.2,DYSELM,dataset="dmu",DYWEIGHT
       #     "jet_Eta[0]",r"#eta_{jet}",20,-2.8,2.8,DYSELM,dataset="dmu",DYWEIGHT
       #    "jet_Pt[0]",r"p^{jet}_{T} [GeV]",25,20,120,DYSELM,dataset="dmu",makeROOT=True,DYWEIGHT              
            
        #     "jet_Phi[0]",r"#phi_{jet}",20,-3.2,3.2,DYSELE,dataset="deg",DYWEIGHT
        #     "jet_Eta[0]",r"#eta_{jet}",20,-2.8,2.8,DYSELE,dataset="deg",DYWEIGHT
        #     "jet_Pt[0]",r"p^{jet}_{T} [GeV]",25,20,120,DYSELE,dataset="deg",makeROOT=True,DYWEIGHT
              
'''

onlyKins =  '''

'''

# open job list file
cmdList = open("cmdList.txt","w")

if addsel!='': print("WARNING: YOU HAVE A CUSTOM SELECTION APPLIED!")
for systname in systs:
    global syst
    syst=systname
    
    # false
    if plotsysts and plot2D:
        args=[applyCuts(line.strip()) for line in arguments.split("\n") if not line.strip()=="" and not line.strip().startswith("#")]
        for i, line in enumerate(args):
            cmdList.write("Stacker.plotStack("+line.strip()+")\n")
    # false
    if plotBinSlices:        
        if not plotsysts and not "central" in systname: continue
        #varBin1=[-0.2,0.,0.2,0.4,0.6,0.8,1.]
        #varBin2=[-0.2,0.,0.2,0.4,0.6,0.8,1.]
        varBin1=[-0.2,0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.]
        varBin2=[-0.2,0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.]
        args=[line.strip() for line in plot1D.split("\n") if not line.strip()=="" and not line.strip().startswith("#")]
        # this adds the jobs for the stacked histograms, but without the eventual additional specifications (like normalization) to the command list
        # and if this for some reason runs after the later jobs, this overwrites the actually normalized plots
        #for i, line in enumerate(args):
        #    cmdList.write("Stacker.plotStack("+applyCuts(line).strip()+")\n")
        for i in range(1,len(varBin1)-1):            
            for iline, line in enumerate(args):
                deepsuff = ""
                if line.strip().startswith('"jet_DeepFlav'): deepsuff="DeepFlav"
                if line.strip().startswith('"jet_Custom'): deepsuff="Custom"
                if line.strip().startswith('"jet_CustomNoise'): deepsuff="CustomNoise"
                if line.strip().startswith('"jet_CustomFGSM'): deepsuff="CustomFGSM"
                if line.strip().startswith('"jet_Custom_A_'): deepsuff="Custom_A_"
                if line.strip().startswith('"jet_Custom_B_'): deepsuff="Custom_B_"
                if line.strip().startswith('"jet_Custom_C_'): deepsuff="Custom_C_"
                if line.strip().startswith('"jet_%sCvsL'%deepsuff): continue
                for sel in ["ESEL","MSEL","TTSELEE","TTSELMM","TTSELME","TTSEMISELE","TTSEMISELM"]:
                    line = line.replace(sel,sel+'+" && jet_%sCvsL[muJet_idx] >= %f && jet_%sCvsL[muJet_idx] < %f"'%(deepsuff,varBin1[i],deepsuff,varBin1[i+1]))
                for sel in ["DYSELE","DYSELM"]:
                    line = line.replace(sel,sel+'+" && jet_%sCvsL[0] >= %.2f && jet_%sCvsL[0] < %.2f"'%(deepsuff,varBin1[i],deepsuff,varBin1[i+1]))
                line += ',filePost="%sCvsL_%.2f-%.2f"'%(deepsuff,varBin1[i],varBin1[i+1])
                # cmdList.write("Stacker.plotStack("+applyCuts(line).strip()+",makePNG=False)\n")   
        for i in range(1,len(varBin2)-1):
            for iline, line in enumerate(args):
                deepsuff = ""
                if line.strip().startswith('"jet_DeepFlav'): deepsuff="DeepFlav"
                if line.strip().startswith('"jet_Custom'): deepsuff="Custom"
                if line.strip().startswith('"jet_CustomNoise'): deepsuff="CustomNoise"
                if line.strip().startswith('"jet_CustomFGSM'): deepsuff="CustomFGSM"
                if line.strip().startswith('"jet_Custom_A_'): deepsuff="Custom_A_"
                if line.strip().startswith('"jet_Custom_B_'): deepsuff="Custom_B_"
                if line.strip().startswith('"jet_Custom_C_'): deepsuff="Custom_C_"
                if line.strip().startswith('"jet_%sCvsB'%deepsuff): continue
                for sel in ["ESEL","MSEL","TTSELEE","TTSELMM","TTSELME","TTSEMISELE","TTSEMISELM"]:
                    line = line.replace(sel,sel+'+" && jet_%sCvsB[muJet_idx] >= %f && jet_%sCvsB[muJet_idx] < %f"'%(deepsuff,varBin2[i],deepsuff,varBin2[i+1]))
                for sel in ["DYSELE","DYSELM"]:
                    line = line.replace(sel,sel+'+" && jet_%sCvsB[0] >= %f && jet_%sCvsB[0] < %f"'%(deepsuff,varBin2[i],deepsuff,varBin2[i+1]))
                line += ',filePost="%sCvsB_%.2f-%.2f"'%(deepsuff,varBin2[i],varBin2[i+1])
                cmdList.write("Stacker.plotStack("+applyCuts(line).strip()+",makePNG=False)\n")

    # true
    if "central" in systname:
        moreargs = ""
        moreargs+=",drawDataMCRatioLine=True"
        # true
        if normMCtodata:
            moreargs+=",normTotalMC=True"
        # false
        if plotExtra:
            args=[applyCuts(line.strip()) for line in onlyCentral.split("\n")+plot1D.split('\n') if not line.strip()=="" and not line.strip().startswith("#")]
            for i, line in enumerate(args):
                cmdList.write("Stacker.plotStack("+line.strip()+moreargs+")\n")
        # false
        if validateSFs:
            args=[applyCuts(line.strip()) for line in plot1D.split('\n') if not line.strip()=="" and not line.strip().startswith("#")]
            for i, line in enumerate(args):
                if "DeepFlavCvs" in line or "btagDeepFlav" in line: SFfiles = SFfilesDeepJet
                else: SFfiles = SFfilesDeepCSV
                for SF in SFfiles:
                    for histsuff in SFhistSuff:
                        if SF == "" and histsuff != "": continue
                        cmdList.write("Stacker.plotStack(%s,SFfile=\"%s\",SFhistSuff=\"%s\",drawDataMCRatioLine=True)\n"%(line.strip(),SF,histsuff))
        else:
            args=[applyCuts(line.strip()) for line in plot1D.split('\n') if not line.strip()=="" and not line.strip().startswith("#")]
            for i, line in enumerate(args):
                cmdList.write("Stacker.plotStack(%s)\n"%(line.strip()+moreargs))
        pt_binning = [0,50,100,2000]
        eta_binning = [-3,-1,0,1,3]
        for i in range(len(pt_binning)-1):
            for j in range(len(eta_binning)-1):
                args=[applyCuts(line.strip(), pt_eta_domain=[pt_binning[i],pt_binning[i+1],eta_binning[j],eta_binning[j+1]]) for line in plot1D.split('\n') if not line.strip()=="" and not line.strip().startswith("#")]
                for k, line in enumerate(args):
                    cmdList.write("Stacker.plotStack(%s)\n"%(line.strip()+moreargs))
cmdList.close()
