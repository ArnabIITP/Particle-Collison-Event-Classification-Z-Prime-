# Particle Collision Event Classification (Signal vs Background)

## 1) Project Summary (What this project does)
This project is a **particle-collision event classification** workflow implemented as a single Jupyter notebook.

**Goal:** Train a supervised ML model to distinguish a **signal** process from a **background** process using event-level and object-level features stored in ROOT ntuples.

- **Signal sample:** `...zprime3000_tt...` (Z′ → t\bar{t} at a 3000 GeV mass point, based on the filename)
- **Background sample:** `...ttbarWW...` (based on the filename)

**Data source:** These ROOT ntuples come from the ATLAS Open Data “Open Data for Education and Outreach” release on the CERN Open Data Portal:
- https://opendata.cern.ch/record/93913

The record requests that you cite it as:
“ATLAS collaboration (2025). ATLAS ROOT ntuple format MC simulation, 2015+2016 proton-proton collisions beta release, 2J2LMET30 skim. CERN Open Data Portal. DOI: 10.7483/OPENDATA.ATLAS.NNF8.76IX”.

**Output of the notebook:**
- A merged feature table for signal+background
- A trained **XGBoost classifier**
- ROC curve + AUC
- Feature importance plot
- A “bump hunt” style plot: an **approximate reconstructed invariant mass** after selecting events with high signal probability

This is a common HEP analysis pattern:
1. Read ROOT trees → build flat ML features
2. Train classifier (signal vs background)
3. Apply selection on classifier output
4. Make physics-motivated plots on selected events


## 2) Repository / Workspace Structure

```
Particle collision event classification/
├─ Particle collision.ipynb
├─ README.md
└─ Data/
   ├─ ODEO_FEB2025_v0_2J2LMET30_mc_301333.Pythia8EvtGen_A14NNPDF23LO_zprime3000_tt.2J2LMET30.root
   └─ ODEO_FEB2025_v0_2J2LMET30_mc_410081.MadGraphPythia8EvtGen_A14NNPDF23_ttbarWW.2J2LMET30.root
```

- `Particle collision.ipynb` contains the full analysis pipeline.
- `Data/` contains two ROOT files (signal and background).


## 3) Environment Requirements

### OS
- OS-independent (Windows/macOS/Linux). The notebook resolves paths relative to the repository folder.

### Python
- Python 3.9+ recommended (3.10/3.11 usually fine).

### Python packages (installed inside the notebook kernel)
The notebook includes a helper cell that installs missing packages directly into the current kernel:
- `uproot`
- `awkward`
- `scikit-learn`
- `xgboost`
- `vector` (currently imported/installed, not essential to the minimal pipeline)
- `pandas`, `numpy`, `matplotlib` (used heavily; typically preinstalled but required)

If you prefer installing outside the notebook, you can run:

```bash
python -m pip install uproot awkward scikit-learn xgboost vector pandas numpy matplotlib
```

**Note on Windows + XGBoost:**
- If `pip install xgboost` fails, you may need a newer pip (`python -m pip install --upgrade pip`) or a compatible Python version.


## 4) How to Run (Step-by-step)

### Step 0 — Verify the datasets exist
Confirm these files exist:
- `Data/ODEO_FEB2025_v0_2J2LMET30_mc_301333.Pythia8EvtGen_A14NNPDF23LO_zprime3000_tt.2J2LMET30.root`
- `Data/ODEO_FEB2025_v0_2J2LMET30_mc_410081.MadGraphPythia8EvtGen_A14NNPDF23_ttbarWW.2J2LMET30.root`

### Step 1 — Open the notebook
Open `Particle collision.ipynb` in VS Code (or Jupyter).

### Step 2 — Run cells in order
Run all cells from top to bottom.

The first cell:
- Locates the repository folder (using the current working directory, and falling back to walking up to find `.git`)
- Uses `DATA_DIR = <repo>/Data`
- Asserts both ROOT files exist

The second cell:
- Installs required Python packages into the notebook kernel if they are missing

The third cell:
- Opens the signal ROOT file with `uproot.open(...)` and lists the ROOT keys

After that, the notebook:
- Inspects ROOT keys
- Reads the TTree named `analysis`
- Builds features, trains the model, evaluates it, and produces plots


## 5) What is inside the ROOT files (high-level)

Each `.root` file is a Monte Carlo ntuple file that contains at least:
- A top-level object (file keys)
- A TTree named `analysis` (accessed as `file["analysis"]`)
- Branches (columns) inside the `analysis` tree used for event / jet / lepton variables

The notebook calls:
- `file.keys()` to list objects stored in the ROOT file
- `tree.keys()` to list branches inside the `analysis` tree
- `tree.num_entries` to count events


## 6) Dataset Names: what the filenames are telling you

Your two ROOT files are:

### 6.1 Signal
`ODEO_FEB2025_v0_2J2LMET30_mc_301333.Pythia8EvtGen_A14NNPDF23LO_zprime3000_tt.2J2LMET30.root`

Interpretation (based on common HEP naming conventions):
- `ODEO_FEB2025_v0`: dataset/production campaign label (February 2025, version v0)
- `2J2LMET30`: an analysis category/selection label meaning roughly “2 jets, 2 leptons, MET>30” (exact cut definition depends on the production)
- `mc_301333`: Monte Carlo dataset ID
- `Pythia8EvtGen`: generator chain used
- `A14NNPDF23LO`: tune/PDF configuration
- `zprime3000_tt`: physics process: Z′ with mass around 3000 (GeV), decaying to t\bar{t}

### 6.2 Background
`ODEO_FEB2025_v0_2J2LMET30_mc_410081.MadGraphPythia8EvtGen_A14NNPDF23_ttbarWW.2J2LMET30.root`

Interpretation:
- Same campaign and selection tag style as signal
- `mc_410081`: background MC dataset ID
- `MadGraphPythia8EvtGen`: generator chain used
- `ttbarWW`: physics process label (top pair plus WW, or a related combined sample)

For the official definition of the **2J2LMET30** skim in this release, see the CERN Open Data record. It describes the applied skimming selection at a high level (≥2 jets, ≥2 leptons, and MET ≥ 30 GeV; with additional ID requirements).

**Important:** Filenames encode production metadata but are not a guarantee of exact selection definitions. For authoritative definitions (cuts, object definitions, units), you’d normally check production documentation, ntuple schema docs, or metadata.


## 7) Notebook Pipeline (What each phase does)

The notebook is written as a linear analysis script. Here is what each logical phase does.

### Phase A — Load the signal ROOT file
- Builds paths to ROOT files.
- Opens the signal file: `uproot.open(sig_path)`.
- Selects the `analysis` TTree: `tree = file["analysis"]`.

### Phase B — Build signal features
The notebook builds a **pandas DataFrame** called `X_event` that holds one row per event.

1) **Event-level flat features** loaded directly from branches:
- `met`, `met_phi`, `jet_n`, `lep_n`, `n_sig_lep`

2) **Jet-derived features** loaded from jagged branches (`awkward`) then converted into flat per-event columns:
- Reads: `jet_pt`, `jet_eta`, `jet_phi`, `jet_btag_quantile`
- Creates:
  - Leading jet kinematics: `lead_jet_pt`, `lead_jet_eta`, `lead_jet_phi`
  - Subleading jet kinematics: `sublead_jet_pt`, `sublead_jet_eta`, `sublead_jet_phi`
  - A simple b-tag summary: `lead_jet_btag` (taken as the first btag quantile in the event)
  - Scalar HT: `ht = sum(jet_pt)`

3) **Lepton-derived features** (also jagged → padded → flattened):
- Reads: `lep_pt`, `lep_eta`, `lep_phi`, `lep_charge`
- Creates:
  - Leading lepton: `lead_lep_pt`, `lead_lep_eta`, `lead_lep_phi`, `lead_lep_charge`
  - Subleading lepton: `sublead_lep_pt`, `sublead_lep_eta`, `sublead_lep_phi`, `sublead_lep_charge`
  - Dilepton features (massless approximation):
    - `m_ll` = approximate invariant mass of two leading leptons
    - `dphi_ll` = |Δφ| between two leading leptons
  - Lepton scalar sum: `lt = sum(lep_pt)`
  - Event activity variable: `st = ht + lt + met`

**Padding logic:** for events with fewer than 2 jets or 2 leptons, the missing values are filled with 0 so the DataFrame stays rectangular.

### Phase C — Load the background ROOT file and build the same features
- Opens background file, gets `bg_tree = bg_file["analysis"]`.
- Builds `X_bg` with identical columns and construction logic.

### Phase D — Labels and merged dataset
- Signal labels: `y_sig = 1` for all signal events.
- Background labels: `y_bg = 0` for all background events.
- Merge feature tables:
  - `X_all = concat([X_event, X_bg])`
  - `y_all = concatenate([y_sig, y_bg])`

### Phase E — Train/test split
Splits into training and testing sets with stratification:
- `train_test_split(..., test_size=0.25, random_state=42, stratify=y_all)`

### Phase F — Train an XGBoost classifier
- Replaces infinities and NaNs with 0 (`X_train_f`, `X_test_f`).
- Trains `XGBClassifier` (tree-based gradient boosting).

Key hyperparameters used:
- `n_estimators=800`
- `max_depth=4`
- `learning_rate=0.05`
- `subsample=0.8`
- `colsample_bytree=0.8`
- `eval_metric="auc"`
- `tree_method="hist"`

### Phase G — Evaluate and interpret
- Computes ROC curve and AUC on the test set.
- Plots feature importance (built-in XGBoost importances).

### Phase H — “Bump hunt” mass reconstruction (approximate)
The notebook builds an approximate reconstructed mass using:
- Leading 2 jets
- Leading 2 leptons
- MET (treated as a massless 4-vector with pz=0)

This produces:
- `m_all`: concatenated reconstructed mass array for signal+background in the same order as `X_all`

Then it:
- Applies the trained model to all events: `proba_all = xgb.predict_proba(X_all_f)[:, 1]`
- Defines a tight selection: `sel = proba_all > 0.90`
- Plots mass distributions for selected signal and selected background

**Physics note:** this mass is **not a true invariant mass** reconstruction of a heavy resonance in dileptonic top decays (neutrinos have unknown pz). It’s a quick, approximate discriminant for demonstration.


## 8) Branch & Feature Glossary (What each variable means)

This section is the “dictionary” of variables used in the notebook.

### 8.1 Branches read directly from the ROOT `analysis` tree
These exist in the ROOT files (as branches).

**Event-level scalar branches**
- `met`: Missing transverse energy magnitude (float). Typically in GeV.
- `met_phi`: Azimuthal angle φ of the missing transverse momentum vector (float). Typically in radians.
- `jet_n`: Number of reconstructed jets in the event (int).
- `lep_n`: Number of reconstructed leptons in the event (int).
- `n_sig_lep`: Number of “signal leptons” passing tighter ID/isolation (int). Exact definition depends on ntuple schema.

**Jet-level jagged branches** (arrays per event; length varies event-by-event)
- `jet_pt`: Transverse momentum pT of each jet (array[float]). Typically in GeV.
- `jet_eta`: Pseudorapidity η of each jet (array[float]). Unitless.
- `jet_phi`: Azimuthal angle φ of each jet (array[float]). Typically in radians.
- `jet_btag_quantile`: A b-tagging related score expressed as a quantile/bin/category (array[float] or array[int] depending on schema).

**Lepton-level jagged branches** (arrays per event)
- `lep_pt`: Transverse momentum pT of each lepton (array[float]). Typically in GeV.
- `lep_eta`: Pseudorapidity η of each lepton (array[float]). Unitless.
- `lep_phi`: Azimuthal angle φ of each lepton (array[float]). Typically in radians.
- `lep_charge`: Electric charge of each lepton (array[int]), usually ±1.

### 8.2 Derived (engineered) per-event features created by the notebook
These are created inside pandas and do not exist as branches in the ROOT files.

**Jet-derived features**
- `lead_jet_pt`: pT of the leading (highest-pT) jet. 0 if fewer than 1 jet.
- `lead_jet_eta`: η of the leading jet. 0 if fewer than 1 jet.
- `lead_jet_phi`: φ of the leading jet. 0 if fewer than 1 jet.
- `lead_jet_btag`: b-tag quantile of (the first) jet in the event as used in the notebook. -1 if missing.
- `sublead_jet_pt`: pT of the 2nd jet. 0 if fewer than 2 jets.
- `sublead_jet_eta`: η of the 2nd jet. 0 if fewer than 2 jets.
- `sublead_jet_phi`: φ of the 2nd jet. 0 if fewer than 2 jets.
- `ht`: Scalar sum of jet pT over all jets in the event: `HT = Σ pT(jet)`.

**Lepton-derived features**
- `lead_lep_pt`: pT of the leading lepton. 0 if fewer than 1 lepton.
- `lead_lep_eta`: η of the leading lepton. 0 if fewer than 1 lepton.
- `lead_lep_phi`: φ of the leading lepton. 0 if fewer than 1 lepton.
- `lead_lep_charge`: charge of the leading lepton. 0 if fewer than 1 lepton.
- `sublead_lep_pt`: pT of the 2nd lepton. 0 if fewer than 2 leptons.
- `sublead_lep_eta`: η of the 2nd lepton. 0 if fewer than 2 leptons.
- `sublead_lep_phi`: φ of the 2nd lepton. 0 if fewer than 2 leptons.
- `sublead_lep_charge`: charge of the 2nd lepton. 0 if fewer than 2 leptons.
- `m_ll`: Approximate invariant mass of the two leading leptons using a massless approximation.
- `dphi_ll`: Absolute Δφ between the two leading leptons (wrapped into [-π, π]).
- `lt`: Scalar sum of lepton pT over all leptons in the event: `LT = Σ pT(lepton)`.
- `st`: Global activity variable: `ST = HT + LT + MET`.

### 8.3 Model and selection variables
- `y`: signal label vector for the signal sample (all ones)
- `y_bg`: background label vector for the background sample (all zeros)
- `y_all`: concatenated labels for the merged dataset
- `proba_all`: predicted probability that each event is signal (output of `predict_proba`)
- `sel`: boolean selection mask for “signal-like” events (`proba_all > 0.90`)

### 8.4 Mass reconstruction variables (approximate)
- `m_sig_all`: reconstructed mass array for signal events
- `m_bkg_all`: reconstructed mass array for background events
- `m_all`: concatenation of the two arrays aligned with `X_all`
- `m_sig_sel`: mass values for selected signal events
- `m_bkg_sel`: mass values for selected background events


## 9) Practical Notes and Common Pitfalls

### 9.1 Paths on Windows
The notebook hardcodes:
- `WORKSPACE_DIR = D:\Particle collision event classification`

If you cloned/copied the folder somewhere else, update that path or set it to `Path.cwd()`.

### 9.2 Units
The notebook assumes standard HEP conventions:
- pT, MET, HT, ST in **GeV**
- φ in **radians**
- η is unitless

If your ntuples use different units, the plots and derived features will scale accordingly.

### 9.3 Events with fewer than 2 jets / leptons
The notebook pads with missing values and then fills them with 0.
This is convenient for ML, but it can introduce artificial spikes at 0 in distributions.

### 9.4 Model training realism
This notebook is a demonstration workflow:
- It does not include systematic uncertainties, event weights, cross sections, or detector effects beyond what is in the ntuples.
- It trains on two MC samples only; a real analysis would consider multiple backgrounds, weighting, and dedicated validation.


## 10) How to extend the branch glossary to *all* branches
ROOT ntuples often contain many more branches than the small subset used here.

To list all branches in the `analysis` tree, run the notebook cell with:
- `tree.keys()` (for signal)
- `bg_tree.keys()` (for background)

Then you can extend Section 8 by documenting the additional branches.

If you want, tell me whether you want the README to document:
- **Only branches used in the notebook** (current behavior; accurate and manageable), or
- **Every branch in the tree** (can be very long; requires extracting full `tree.keys()` output).
