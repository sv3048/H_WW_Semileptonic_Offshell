import uproot
import hist
import matplotlib.pyplot as plt
import mplhep as hep
import os
import shutil

#plots = ["h_Lepton_pt_selection", "h_Jet_pt_selection", "h_Higgs_mass", "h_Higgs_pt", "h_Nominal_WTagger_selection"]
plots = ["h_Lepton_pt_selection"]

group_map = {
    "W + Jets": [
        "WJetsToLNu-LO",
        "WJetsToLNu_HT70To100",
        "WJetsToLNu_HT100To200",
        "WJetsToLNu_HT200To400",
        "WJetsToLNu_HT400To600",
        "WJetsToLNu_HT600To800",
        "WJetsToLNu_HT800To1200",
        "WJetsToLNu_HT1200To2500",
        "WJetsToLNu_HT2500ToInf"
    ],
    "Top": [
        "TTToSemiLeptonic",
        "TTTo2L2Nu",
        "TTWJetsToLNu",
        "ST_s-channel",
        "ST_t-channel_antitop",
        "ST_t-channel_top",
        "ST_tW_antitop",
        "ST_tW_top"
    ],
    "DY": [
        "DYJetsToLL_M-50",
        "DYJetsToLL_M-50_HT-70to100",
        "DYJetsToLL_M-50_HT-100to200",
        "DYJetsToLL_M-50_HT-200to400",
        "DYJetsToLL_M-50_HT-400to600",
        "DYJetsToLL_M-50_HT-600to800",
        "DYJetsToLL_M-50_HT-800to1200",
        "DYJetsToLL_M-50_HT-1200to2500",
        "DYJetsToLL_M-50_HT-2500toInf",
        "DYJetsToLL_M-10to50-LO",
        "DY_else"
    ],
     "Multiboson": [
        "WW",
        "ggH_bonly_on",
        "ggH_bonly_off",
        "qqWWqq",
        "Vg",
        "WGToLNuG",
        "ZGToLLG",
        "WZTo3LNu_mllmin0p1",
        "ZZ",
        "WZ",
        "WmToLNu_ZTo2J_QCD",
        "WpToLNu_ZTo2J_QCD",
        "VVV",
        "WWewk",
        "VBF_V",
        "ggH_sonly_on",
        #"ggH_sonly_off",
        "ggH_sand_on"
    ],
    
    # Add other groups as needed
}
uproot_file = uproot.open("output_step2.root")
output_path = "/eos/user/s/sverma/www/Marc_TNP_Plots/H_WW/28Oct_step2"
os.makedirs(output_path, exist_ok=True)
shutil.copy("index.php", output_path)

def get_integral(histogram):
    s = histogram.sum()
    return s.value if hasattr(s, "value") else s

for plot in plots:
    print(plot)
    hep.style.use("CMS")
    fig, ax = plt.subplots()
    hep.cms.label("Preliminary", com=13, lumi=59.7, data=True, loc=0, ax=ax)
    ax.set_ylabel("Events")
    ax.set_yscale('log')
    ax.set_xlabel(f"{plot}")
    histo_list = []
    legend_list = []
    for group, samples in group_map.items():
        group_hist = None
        for sample in samples:
            try:
                h = uproot_file[f"{sample}/{plot}"].to_hist()
                h *= 59.7  # Scale by luminosity
                if group_hist is None:
                    group_hist = h
                else:
                    group_hist += h
            except KeyError:
                print(f"WARNING: Skipping {sample}/{plot}. Not found.")
                continue
        if group_hist is not None:
            integral = get_integral(group_hist)
            histo_list.append(group_hist)
            legend_list.append(f"{group}: {integral:.2f}")

    hep.histplot(histo_list, ax=ax, stack=True, histtype='fill', label=legend_list)

    # --- Overlay signal ---
    try:
        hist_signal = uproot_file[f"ggH_sonly_off/{plot}"].to_hist()
        hist_signal *= 59.7
        integral_signal = get_integral(hist_signal)
        hep.histplot(hist_signal, ax=ax, histtype='step', linewidth=3, color='green', label=f"Signal: {integral_signal:.2f}")
    except KeyError:
        print(f"WARNING: Signal ggH_sonly_off/{plot} not found.")

    # --- Overlay SBI ---
    try:
        hist_sbi = uproot_file[f"ggH_sand_off/{plot}"].to_hist()
        hist_sbi *= 59.7
        integral_sbi = get_integral(hist_sbi)
        hep.histplot(hist_sbi, ax=ax, histtype='step', linewidth=3, color='blue', label=f"SBI: {integral_sbi:.2f}")
    except KeyError:
        print(f"WARNING: SBI ggH_sand_off/{plot} not found.")

    ax.legend()
    ax.set_ylabel("Events")
    ax.set_yscale('log')
    ax.set_xlabel(f"{plot}")
    ax.set_xlim(0, 900)  # <-- Add this line to limit x-axis
    plt.savefig(f'{output_path}/{plot}.png')











# # ==============================================================================
# # 🚀 PURE PYTHON PLOTTING SCRIPT (Uproot & Hist Workflow)
# # ==============================================================================

# # ROOT imports 
# import os, ROOT
# import cmsstyle as CMS
# import narf

# # python imports 
# import matplotlib.pyplot as plt  # matplotlib library
# import mplhep as hep  # HEP (CMS) extensions/styling on top of mpl

# # For constructing examples
# import hist  # histogramming library
# import numpy as np 
# import uproot
# import math
# import shutil

# import narf

# # # Temporary patch in case root_to_hist is missing
# # if not hasattr(narf, "root_to_hist"):
# #     import hist
# #     def root_to_hist(root_hist):
# #         return hist.export_root.import_hist(root_hist)
# #     narf.root_to_hist = root_to_hist



# #plots = ["h_Jet_pt","h_Nominal_WTagger","h_Lepton_pt_selection","h_Lepton_eta_selection","h_Lepton_phi_selection","h_Jet_pt_selection","h_Jet_eta_selection","h_Jet_phi_selection","h_Nominal_WTagger_selection","h_Higgs_pt","h_Higgs_eta","h_Higgs_phi","h_Higgs_mass"]
# plots = ["h_Lepton_pt_selection"]

# bkg_samples = ["W + Jets", "Top", "DY", "Multiboson", "ggH_sonly_on", "ggH_sand_on"]
# bkg_samples.reverse()
# #Multi_bosons = ["WW","Vg","VgS","VZ","VVV","WWewk","VBF_V"]
# Multi_bosons = ["WW","ggH_bonly_on","ggH_bonly_off", "qqWWqq","Vg", "WGToLNuG","ZGToLLG","WZTo3LNu_mllmin0p1","ZZ","WZ","WmToLNu_ZTo2J_QCD","WpToLNu_ZTo2J_QCD","VVV", "WWewk", "VBF_V", "ggH_sonly_on", "ggH_sonly_off", "ggH_sand_on"]
# Top = ["TTToSemiLeptonic","TTTo2L2Nu","TTWJetsToLNu","ST_s-channel","ST_t-channel_antitop","ST_t-channel_top","ST_tW_antitop","ST_tW_top"]
# DY = ["DYJetsToLL_M-50","DYJetsToLL_M-50_HT-70to100","DYJetsToLL_M-50_HT-100to200","DYJetsToLL_M-50_HT-200to400","DYJetsToLL_M-50_HT-400to600","DYJetsToLL_M-50_HT-600to800","DYJetsToLL_M-50_HT-800to1200","DYJetsToLL_M-50_HT-1200to2500","DYJetsToLL_M-50_HT-2500toInf","DYJetsToLL__M-10to50-LO","DY_else"]
# WJets = ["WJetsToLNu_LO","WJetsToLNu_HT-70to100","WJetsToLNu_HT-100to200","WJetsToLNu_HT-200to400","WJetsToLNu_HT-400to600","WJetsToLNu_HT-600to800","WJetsToLNu_HT-800to1200","WJetsToLNu_HT-1200to2500","WJetsToLNu_HT-2500toInf"]
# signal = ["ggH_sonly_off"]
# SBI = [""]

# # Replace PyROOT file opening with Uproot for stability
# uproot_file = uproot.open("output_bkg.root")

# output_path = "/eos/user/s/sverma/www/Marc_TNP_Plots/H_WW/28Oct"
# os.makedirs(output_path,exist_ok=True)
# shutil.copy("index.php", output_path)

# for plot in plots:
#     print(plot)
#     #Styling
#     hep.style.use("CMS")
#     fig, ax = plt.subplots()
#     hep.cms.label("Preliminary", com = 13, lumi = 59.7, data = True, loc=0, ax=ax);
#     ax.set_ylabel("Events")
#     ax.set_yscale('log')
#     ax.set_xlabel(f"{plot}")
#     histo_list = []
#     legend_list = []
#     for bkg in bkg_samples:
#         print(bkg)
        
#         current_bkg_hist = None # Will hold the hist object sum
#         histogram = None 
        
#         if bkg == "Multiboson":
#             try:
#                 # Get first histogram as hist object
#                 current_bkg_hist = uproot_file[f"WW/{plot}"].to_hist()
#             except KeyError:
#                 print(f"WARNING: Skipping Multiboson. WW/{plot} not found.")
#                 continue

#             for sample in Multi_bosons:
#                 if sample == "WW": 
#                     continue
#                 try:
#                     # Get subsequent histograms as hist objects
#                     h_sample = uproot_file[f"{sample}/{plot}"].to_hist()
#                     # Use robust Python addition
#                     current_bkg_hist += h_sample 
#                 except KeyError:
#                     print(f"WARNING: Skipping sample {sample}/{plot}. Not found.")
#                     continue
            
#             histogram = current_bkg_hist # Alias for integration below
            
#         else:
#             try:
#                 # Get single background histogram as hist object
#                 histogram = uproot_file[f"{bkg}/{plot}"].to_hist()
#             except KeyError:
#                 print(f"WARNING: Skipping {bkg}. Histogram not found.")
#                 continue

#         # Use Hist object's .sum() method for integral
        
#         # Before (Failing): integral = histogram.sum().value 

#         try:
#             # Try to access .value (for WeightedSum objects)
#             integral = histogram.sum().value
#         except AttributeError:
#             # If it fails, the object is a float, so use the sum directly
#             integral = histogram.sum()
                
#         # Append the Hist object (no conversion needed)
#         histo_list.append(histogram) 
#         legend_list.append(f"{bkg}: {integral:.2f}") 
        
#     hep.histplot(histo_list, ax=ax, stack=True, histtype='fill', label=legend_list)
    
#     # # --- Signal and SBI Handling ---
#     # try:
#     #     # Get signal histogram as hist object
#     #     hist_signal = uproot_file[f"ggH_sonly_off/{plot}"].to_hist()
#     #     integral_signal = hist_signal.sum() 
#     #     hep.histplot(hist_signal, ax=ax, histtype='step', linewidth=3,color='green', label=f"Signal: {integral_signal:.2f}")
#     # except KeyError:
#     #     print(f"WARNING: Signal ggH_sonly_off/{plot} not found. Skipping plot.")
        
#     # try:
#     #     # Get SBI histogram as hist object
#     #     hist_sbi = uproot_file[f"ggH_sand_off/{plot}"].to_hist()
#     #     integral_sbi = hist_sbi.sum() 
#     #     hep.histplot(hist_sbi, ax=ax, histtype='step', linewidth=3,color='blue', label=f"SBI: {integral_sbi:.2f}")
#     # except KeyError:
#     #     print(f"WARNING: SBI ggH_sand_off/{plot} not found. Skipping plot.")
        
#     ax.legend()
#     plt.savefig(f'{output_path}/{plot}.png')




# # ROOT imports 
# import os, ROOT
# import cmsstyle as CMS
# import narf

# # python imports 
# import matplotlib.pyplot as plt  # matplotlib library
# import mplhep as hep  # HEP (CMS) extensions/styling on top of mpl

# # For constructing examples
# import hist  # histogramming library
# import numpy as np 
# import uproot
# import math
# import shutil

# import narf

# # # Temporary patch in case root_to_hist is missing
# # if not hasattr(narf, "root_to_hist"):
# #     import hist
# #     def root_to_hist(root_hist):
# #         return hist.export_root.import_hist(root_hist)
# #     narf.root_to_hist = root_to_hist



# #plots = ["h_Jet_pt","h_Nominal_WTagger","h_Lepton_pt_selection","h_Lepton_eta_selection","h_Lepton_phi_selection","h_Jet_pt_selection","h_Jet_eta_selection","h_Jet_phi_selection","h_Nominal_WTagger_selection","h_Higgs_pt","h_Higgs_eta","h_Higgs_phi","h_Higgs_mass"]
# plots = ["h_Lepton_pt_selection"]

# bkg_samples = ["W + Jets","Top","DY","Multiboson","ggH_sonly_on","ggH_sand_on"]
# bkg_samples.reverse()
# #Multi_bosons = ["WW","Vg","VgS","VZ","VVV","WWewk","VBF_V"]
# Multi_bosons = ["WW","ggH_bonly_on","ggH_bonly_off", "qqWWqq","Vg", "WGToLNuG","ZGToLLG","WZTo3LNu_mllmin0p1","ZZ","WZ","WmToLNu_ZTo2J_QCD","WpToLNu_ZTo2J_QCD","VVV", "WWewk", "VBF_V", "ggH_sonly_on", "ggH_sonly_off", "ggH_sand_on"]
# Top = ["TTToSemiLeptonic","TTTo2L2Nu","TTWJetsToLNu","ST_s-channel","ST_t-channel_antitop","ST_t-channel_top","ST_tW_antitop","ST_tW_top"]
# DY = ["DYJetsToLL_M-50","DYJetsToLL_M-50_HT-70to100","DYJetsToLL_M-50_HT-100to200","DYJetsToLL_M-50_HT-200to400","DYJetsToLL_M-50_HT-400to600","DYJetsToLL_M-50_HT-600to800","DYJetsToLL_M-50_HT-800to1200","DYJetsToLL_M-50_HT-1200to2500","DYJetsToLL_M-50_HT-2500toInf","DYJetsToLL__M-10to50-LO","DY_else"]
# WJets = ["WJetsToLNu_LO","WJetsToLNu_HT-70to100","WJetsToLNu_HT-100to200","WJetsToLNu_HT-200to400","WJetsToLNu_HT-400to600","WJetsToLNu_HT-600to800","WJetsToLNu_HT-800to1200","WJetsToLNu_HT-1200to2500","WJetsToLNu_HT-2500toInf"]
# signal = ["ggH_sonly_off"]
# SBI = ["ggH_sand_off"]

# #root_file = ROOT.TFile.Open("output_saved.root","READ")
# root_file = ROOT.TFile.Open("output.root","READ")

# output_path = "/eos/user/s/sverma/www/Marc_TNP_Plots/H_WW/28Oct"
# os.makedirs(output_path,exist_ok=True)
# shutil.copy("index.php", output_path)
# for plot in plots:
#     print(plot)
#     #Styling
#     hep.style.use("CMS")
#     fig, ax = plt.subplots()
#     hep.cms.label("Preliminary", com = 13, lumi = 59.7, data = True, loc=0, ax=ax);
#     ax.set_ylabel("Events")
#     ax.set_yscale('log')
#     ax.set_xlabel(f"{plot}")
#     histo_list = []
#     legend_list = []
#     for bkg in bkg_samples:
#         print(bkg)
#         if bkg == "Multiboson":
#                 h_ww = root_file.Get(f"WW/{plot}")
#                 histogram = h_ww.Clone()
#                 for sample in Multi_bosons:
#                     if sample == "WW": 
#                         continue
#                     h = root_file.Get(f"{sample}/{plot}")
#                     histogram.Add(h)
#         else:
#             histogram = root_file.Get(f"{bkg}/{plot}")
#         integral = histogram.Integral()
#         h= narf.root_to_hist(histogram)
#         histo_list.append(h)
#         legend_list.append(f"{bkg}: {integral:.2f}") 
#     hep.histplot(histo_list, ax=ax, stack=True, histtype='fill', label=legend_list)
#     hist_signal = root_file.Get(f"ggH_sonly_off/{plot}")
#     integral_signal = hist_signal.Integral()
#     h_signal = narf.root_to_hist(hist_signal)
#     hep.histplot(h_signal, ax=ax, histtype='step', linewidth=3,color='green', label=f"Signal: {integral_signal:.2f}")
#     hist_sbi = root_file.Get(f"ggH_sand_off/{plot}")
#     integral_sbi = hist_sbi.Integral()
#     h_sbi = narf.root_to_hist(hist_sbi)
#     hep.histplot(h_sbi, ax=ax, histtype='step', linewidth=3,color='blue', label=f"SBI: {integral_sbi:.2f}")
#     ax.legend()
#     plt.savefig(f'{output_path}/{plot}.png')  
    
        
        

