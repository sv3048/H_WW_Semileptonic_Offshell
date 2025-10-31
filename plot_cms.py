import uproot
import numpy as np
import matplotlib.pyplot as plt
import mplhep

# List your backgrounds and signals (update names as needed)
backgrounds = [
    "DY", "Top", "W + jets", "Vg", "VgS", "VZ", "VVV", "WWewk", "qqWWqq"
]
signals = [
    "ggH_sonly_on", "ggH_sonly_off", "ggH_sand_on", "ggH_sand_off"
]

# Color map for samples
color_map = {
    "DY": "#e41a1c",
    "Top": "#377eb8",
    "W + jets": "#984ea3",
    "Vg": "#ff7f00",
    "VgS": "#ffff33",
    "VZ": "#a65628",
    "VVV": "#f781bf",
    "WWewk": "#4daf4a",
    "qqWWqq": "#999999",
    "ggH_sonly_on": "#000000",
    "ggH_sonly_off": "#000000",
    "ggH_sand_on": "#000000",
    "ggH_sand_off": "#000000",
}

# Open your output ROOT file
f = uproot.open("output.root")

# Choose histogram name (update as needed)
hist_name = "h_cutflow_5"

# Stack backgrounds
bkg_counts = []
bkg_labels = []
edges = None
for bkg in backgrounds:
    hname = f"{bkg}_{hist_name}"
    if hname in f:
        h = f[hname]
        counts, edges = h.to_numpy()
        bkg_counts.append(counts)
        bkg_labels.append(bkg)

plt.figure(figsize=(7,6))
mplhep.histplot(
    bkg_counts,
    edges,
    stack=True,
    label=bkg_labels,
    color=[color_map[bkg] for bkg in bkg_labels]
)

# Overlay signals
for sig in signals:
    hname = f"{sig}_{hist_name}"
    if hname in f:
        h = f[hname]
        counts, edges = h.to_numpy()
        mplhep.histplot(
            counts,
            edges,
            label=sig,
            color=color_map[sig],
            linewidth=2
        )

mplhep.cms.label("Preliminary", data=True, lumi=59.7)
plt.xlabel("Cutflow Stage")
plt.ylabel("Events")
plt.legend()
plt.yscale("log")
plt.tight_layout()
plt.savefig("cutflow_stack.png")
plt.show()