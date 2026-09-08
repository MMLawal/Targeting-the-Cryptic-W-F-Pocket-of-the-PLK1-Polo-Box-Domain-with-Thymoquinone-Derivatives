import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# Input
# ==========================================================
df = pd.read_csv("all_admet_tq_derivative_data.csv")

compound_col = "Compound"

# ==========================================================
# Properties and thresholds
# ==========================================================
properties = [
    ("Density",0,1),
#   ("MR",40,130),
    ("QED",0.67,None),   
    ("nHA",0,12),
    ("nHD",0,7),
   # ("MW",100,600),
    ("nRot",0,11),
    ("nRing",0,6),
    ("MaxRing",0,18),
    ("nHet",1,15),
   # ("Vol",300,600),
    ("nRig",0,30),
    ("Flex",0,9),
  # ("TPSA",20,140),
    ("nStereo",0,2),
    ("logS",-6,0),
    ("pka_acidic",2,10),
    ("pka_basic",2,12),
  # ("BP",100,600),
    ("logP",-0.7,5),
    ("logD",1,3),
   #("MP",50,250),
    ("Aromatic heavy atoms",0,2),
    ("Fsp3",0.42,None),
    ("Heavy atoms",5,35),
]

labels = []
lower = []
upper = []

for name, lo, hi in properties:

    if hi is None:
        labels.append(f"{name}\n≥{lo}")
        lower.append(lo)

        # automatic upper limit from data
        upper.append(df[name].max()*1.05)

    else:
        labels.append(f"{name}\n({lo}-{hi})")
        lower.append(lo)
        upper.append(hi)

# ==========================================================
# Radar geometry
# ==========================================================
N = len(properties)

angles = np.linspace(0,2*np.pi,N,endpoint=False)

angles = np.concatenate((angles,[angles[0]]))

lower.append(lower[0])
upper.append(upper[0])

# ==========================================================
# Plot
# ==========================================================
fig = plt.figure(figsize=(8,8))

ax = plt.subplot(111,polar=True)

# ----------------------------------------------------------
# Threshold polygons
# ----------------------------------------------------------
ax.plot(
    angles,
    lower,
    color="forestgreen",
    linewidth=2,
    label="Lower limit"
)

ax.fill(
    angles,
    lower,
    color="forestgreen",
    alpha=0.08
)

ax.plot(
    angles,
    upper,
    color="blue",
    linewidth=2,
    label="Upper limit"
)

ax.fill(
    angles,
    upper,
    color="blue",
    alpha=0.08
)

# ----------------------------------------------------------
# Plot all compounds
# ----------------------------------------------------------
colors = plt.cm.tab20(np.linspace(0,1,20))

for i,row in df.iterrows():

    values=[]

    for p in properties:

        values.append(row[p[0]])

    values.append(values[0])

    ax.plot(
        angles,
        values,
        linewidth=1,
        alpha=0.20,
        color=colors[i%20]
    )

# ----------------------------------------------------------
# Labels
# ----------------------------------------------------------
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels,fontsize=18)

#plt.title(
  #  "Physicochemical Profile of Thymoquinone Derivatives",
   # fontsize=16,
   # weight="bold"
#)

plt.legend(
    loc='upper right',
    bbox_to_anchor=(1.25,1.10)
)

plt.tight_layout()

plt.savefig(
    "TQ_PhysChem_Radar.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()
