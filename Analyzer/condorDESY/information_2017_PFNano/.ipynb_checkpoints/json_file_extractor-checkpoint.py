import json

with open('ajson.json') as json_file:
    json_paths = json.load(json_file)
    # SingleMuon files
    with open('SingleMuon.txt','w') as singlemuon_file:
        for entry in json_paths['SingleMuon']:
            singlemuon_file.write(entry+'\n')
    # DoubleMuon files
    with open('DoubleMuon.txt','w') as doublemuon_file:
        for entry in json_paths['DoubleMuon']:
            doublemuon_file.write(entry+'\n')
    with open('DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8.txt','w') as DYJetsToLL_file:
        for entry in json_paths['DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8']:
            DYJetsToLL_file.write(entry+'\n')
    with open('W1JetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8.txt','w') as txtfile:
        for entry in json_paths['W1JetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8']:
            txtfile.write(entry+'\n')
    with open('W2JetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8.txt','w') as txtfile:
        for entry in json_paths['W2JetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8']:
            txtfile.write(entry+'\n')
    with open('W3JetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8.txt','w') as txtfile:
        for entry in json_paths['W3JetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8']:
            txtfile.write(entry+'\n')
    with open('W4JetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8.txt','w') as txtfile:
        for entry in json_paths['W4JetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8']:
            txtfile.write(entry+'\n')
    with open('WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8.txt','w') as txtfile:
        for entry in json_paths['W1JetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8']:
            txtfile.write(entry+'\n')
            
with open('v2x17.json') as json_file:
    json_paths = json.load(json_file)
    # TTtoSemileptonic files
    with open('TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8.txt','w') as txtfile:
        for entry in json_paths['TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8']:
            txtfile.write(entry+'\n')
    with open('TTToHadronic_TuneCP5_13TeV-powheg-pythia8.txt','w') as txtfile:
        for entry in json_paths['TTToHadronic_TuneCP5_13TeV-powheg-pythia8']:
            txtfile.write(entry+'\n')
    with open('TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8.txt','w') as txtfile:
        for entry in json_paths['TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8']:
            txtfile.write(entry+'\n')
    with open('ST_s-channel_4f_hadronicDecays_TuneCP5_13TeV-amcatnlo-pythia8.txt','w') as txtfile:
        for entry in json_paths['ST_s-channel_4f_hadronicDecays_TuneCP5_13TeV-amcatnlo-pythia8']:
            txtfile.write(entry+'\n')
    with open('ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8.txt','w') as txtfile:
        for entry in json_paths['ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8']:
            txtfile.write(entry+'\n')
    with open('ST_t-channel_antitop_4f_inclusiveDecays_TuneCP5_13TeV-powhegV2-madspin-pythia8.txt','w') as txtfile:
        for entry in json_paths['ST_t-channel_antitop_4f_inclusiveDecays_TuneCP5_13TeV-powhegV2-madspin-pythia8']:
            txtfile.write(entry+'\n')
    with open('ST_t-channel_top_4f_inclusiveDecays_TuneCP5_13TeV-powhegV2-madspin-pythia8.txt','w') as txtfile:
        for entry in json_paths['ST_t-channel_top_4f_inclusiveDecays_TuneCP5_13TeV-powhegV2-madspin-pythia8']:
            txtfile.write(entry+'\n')
    with open('ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8.txt','w') as txtfile:
        for entry in json_paths['ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8']:
            txtfile.write(entry+'\n')
    with open('ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8.txt','w') as txtfile:
        for entry in json_paths['ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8']:
            txtfile.write(entry+'\n')
    #with open('ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8.txt','w') as txtfile:
    #    for entry in json_paths['ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8']:
    #        txtfile.write(entry+'\n')
    #with open('ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8.txt','w') as txtfile:
    #    for entry in json_paths['ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8']:
    #        txtfile.write(entry+'\n')
