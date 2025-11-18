from Dataset import dataset
import ROOT

ROOT.gInterpreter.ProcessLine(".O3")
ROOT.gInterpreter.Declare('#include "SemiLeptonic.h"')

def makeRDF_ratioCheck(dataset_name):
    print(f"Processing {dataset_name}")
    
    files = dataset[dataset_name]["files"]
    isOffshell = dataset[dataset_name]["isOffshell"]
    
    # Get genEventSumw
    df_run = ROOT.RDataFrame("Runs", files)
    genEventSumw = df_run.Sum("genEventSumw").GetValue()
    
    df = ROOT.RDataFrame("Events", files)
    
    # Define LHE mWW and fill histogram BEFORE any filter
    df = df.Define("Lhe_mWW", "computeMWW(nLHEPart, LHEPart_pt, LHEPart_eta, LHEPart_phi, LHEPart_mass, LHEPart_pdgId, LHEPart_status)")
    h_mWW = df.Histo1D(("h_mWW_LHE", "LHE mWW", 100, 0, 1000), "Lhe_mWW", "genWeight")
    
    return h_mWW.GetPtr()

# Run for both samples
histograms = {}
histograms["ggH_sonly_off"] = makeRDF_ratioCheck("ggH_sonly_off")
histograms["ggH_sonly_on"] = makeRDF_ratioCheck("ggH_sonly_on")

# Check on-shell/off-shell ratio
for sample_name in ["ggH_sonly_off", "ggH_sonly_on"]:
    h = histograms[sample_name]
    
    int_on = h.Integral(h.FindBin(100), h.FindBin(160) - 1)
    int_off = h.Integral(h.FindBin(160), h.FindBin(1000))
    ratio = int_off / int_on if int_on > 0 else 0
    
    print(f"\n{sample_name}:")
    print(f"  On-shell (100-160):  {int_on:.2f}")
    print(f"  Off-shell (160-1000): {int_off:.2f}")
    print(f"  Ratio (off/on):      {ratio:.2f}\n")
