
# CSINet Adaptations for COST 2100 – Short README

If you only want to **run the test notebooks and reproduce the results**, follow these steps:

1. Install the Python dependencies as mentioned below
2. Run all cells in `dataset_analysis.ipynb` to download and understand the dataset
3. Run all cells in:
   - `CSINet-test.ipynb`
   - `CSINet-test-domainAware.ipynb`

These notebooks will automatically:

- Generate the output **CSV result files**
- Produce the corresponding **heatmaps**
- Using the ***already trained models***

To train the model and/or generate training and validation curves, use the training notebooks (see below).
Also, ```Report Analysis.pptx``` includes the presentation with the summary of the findings of this work.

---

# CSINet Adaptations for COST 2100 – Complete README

This repository contains a full pipeline for training, adapting, and evaluating **CSINet** models on the **COST 2100** MIMO channel dataset  
It includes:

1. Dataset download + exploratory analysis  
2. Baseline CSINet training (indoor / outdoor / joint)  
3. Cross-domain adaptation:
   - Simple fine‑tuning
   - Replay‑based fine‑tuning
   - Domain‑aware CSINet with lightweight adapters  
4. Unified evaluation of all methods (NMSE + heatmaps)

This README explains **how to run each notebook** and **how to interpret the outputs** (logs, checkpoints, CSV results, and figures)

---

# 1. Repository Structure

```
dataset/                  # COST 2100 .mat files
data_figs/                # Dataset analysis visualizations
logs/                     # Training and validation curves
checkpoints/              # All trained models
results/                  # NMSE CSV + angle-delay heatmaps

dataset_analysis.ipynb
CSINet-train.ipynb
CSINet-train-simpleFinetune.ipynb
CSINet-train-replay.ipynb
CSINet-train-domainAware.ipynb
CSINet-test.ipynb
CSINet-test-domainAware.ipynb
```

---

# 2. Environment & Dependencies

The requirements can be found in **`requirements.txt`** besides, you need to take into account:

- **Python 3.10** or greater
- Scientific libraries: NumPy, SciPy, Pandas, Matplotlib which are included in the requirements
- Jupyter and supporting tooling also included in the requirements

---

## GPU / CUDA Notes

PyTorch will install by default on CPU. The elaboration of this repo was run using Cuda on GPU
but due to the likelihood of incompatibilities, CPU run is safer

---

# 3. dataset_analysis.ipynb

This notebook:

1. Downloads dataset using `kagglehub`
2. Reshapes CSI into `(N, 2, 32, 32)`
3. Denormalizes CSI
4. Visualizes:
   - Delay-domain sparsity
   - Angle-domain sparsity
   - Heatmaps of indoor vs outdoor
5. Computes dispersion and sparsity metrics

### Generated outputs (in `data_figs/`)
- `Power_delay_profile.png`
- `Angle_delay.png`
- `heat_map_in_out_*.png`
- Metrics printed

---

# 4. Dataset: COST 2100

### Files downloaded:
```
DATA_Htrainin.mat
DATA_Hvalin.mat
DATA_Htestin.mat

DATA_Htrainout.mat
DATA_Hvalout.mat
DATA_Htestout.mat
```

### CSI format
Each `.mat` contains an `HT` matrix of shape:

```
(N_samples, 2048)
```

Where:

```
2048 = 2 × 32 × 32  
(Real/Imag) × (delay) × (antenna)
```

Reshaped into:

```
H = HT.reshape(-1, 2, 32, 32)
```

This is the **angle–delay domain**, standard for CSINet

---

# 5. CSINet Baseline Training – CSINet-train.ipynb

Trains models for:

```
scenario_type = "indoor"
scenario_type = "outdoor"
scenario_type = "indoor+outdoor"
```

### Behaviour:
- Implements CSINet architecture
- Trains in the selected scenario_type
- Reconstruction loss: MSE
- Early stopping on validation loss

### Execution
Running cells sequentially. You may select the `scenario_type` in cell 2

### Outputs

#### Checkpoints (in `checkpoints/`)
```
csinet_best_indoor.pt
csinet_best_outdoor.pt
csinet_best_indoor+outdoor.pt
```

#### Logs (in `logs/`)
```
train_losses_indoor.npy
val_losses_indoor.npy
learning_curve_indoor.png
...
```

---

# 6. Simple Finetuning – CSINet-train-simpleFinetune.ipynb

```
cross_type = "in2out"   # indoor → outdoor
cross_type = "out2in"   # outdoor → indoor
```

### Behaviour
- Loads pretrained source model from 5
- Finetunes on **target only**
- Demonstrates catastrophic forgetting
- Early stopping on target validation loss

### Execution
Running cells sequentially. You may select the `cross_type` in cell 2

### Outputs

#### Checkpoints (in `checkpoints/`)
```
csinet_best_finetune_in2out.pt
csinet_best_finetune_out2in.pt
```
#### Logs (in `logs/`)
```
train_losses_finetune_in2out.npy
val_losses_main_finetune_in2out.npy
val_losses_orig_finetune_in2out.npy
learning_curve_finetune_in2out.png
...
```

---

# 7. Replay-based Finetuning – CSINet-train-replay.ipynb

```
cross_type = "in2out"   # indoor → outdoor
cross_type = "out2in"   # outdoor → indoor
```

Adds **replay samples** from the source domain to mitigate forgetting.

### Behaviour
- Loads pretrained source model from 5
- Finetunes on **target only** using replay from source domain
- Demonstrates a method to tackle catastrophic forgetting
- Early stopping on target validation loss

### Execution
Running cells sequentially. Cell 2 allows defining the `cross_type` to select the finetuning 

### Outputs

#### Checkpoints (in `checkpoints/`)
```
csinet_best_replay_in2out.pt
csinet_best_replay_out2in.pt
```

#### Logs (in `logs/`)
```
csinet_best_replay_in2out.pt
train_losses_replay_in2out.npy
val_losses_main_replay_in2out.npy
val_losses_rep_replay_in2out.npy
learning_curve_replay_in2out.png
...
```

---

# 8. Domain-Aware CSINet – CSINet-train-domainAware.ipynb

```
cross_type = "in2out"   # indoor → outdoor
cross_type = "out2in"   # outdoor → indoor
```
### Behaviour
- Trains a **shared CSINet backbone** and **lightweight adapters**.
- Selects the specific decoders according to task
- Enables/Disables adapters according to task
- Demonstrates a method to tackle catastrophic forgetting
- Early stopping on validation loss


### Execution
Running cells sequentially. Cell 2 allows defining the `cross_type` to select the finetuning 

Two phases:

---

## Phase 1 — Train backbone only (source domain)

### Outputs Phase 1

#### Checkpoints (in `checkpoints/`)

```
csinet_domainaware_phase1_<src>.pt

```
#### Logs (in `logs/`)

```
train_losses_phase1_<src>.npy
val_losses_phase1_<src>.npy
```
---

## Phase 2 — Freeze backbone, train adapters (target domain)

#### Checkpoints (in `checkpoints/`)

```
csinet_domainaware_phase2_adapters_<src>_to_<tgt>.pt

```
#### Logs (in `logs/`)

```
train_losses_phase2_adapters_<src>_to_<tgt>.npy
val_losses_src_phase2_adapters_<src>_to_<tgt>.npy
val_losses_tgt_phase2_adapters_<src>_to_<tgt>.npy
learning_curve_phase2_adapters_<src>_to_<tgt>.png
```

Adapters allow **target specialization without forgetting**.

---

# 9. Evaluation – CSINet-test.ipynb

Evaluates all **non-adapter** models:

```
indoor
outdoor
indoor+outdoor
finetune_in2out
finetune_out2in
replay_in2out
replay_out2in
```

For each scenario:
- Computes NMSE (dB) in angle–delay domain
- Generates heatmaps

### Execution
Run each cell sequentially for automatic evaluation of all combinations

### Output CSV:
```
results/nmse_results.csv
```

### Output figures:
```
results/heatmaps_<where_evaluated>_csinet_<model>.png
...
```

---

# 10. Evaluation – CSINet-test-domainAware.ipynb

Evaluates the **domain-aware** model:

```
domainaware_in2out
domainaware_out2in
```

Routing:

- If evaluating source domain → **no adapters**
- If evaluating target domain → **adapters enabled**

### Execution
Run each cell sequentially for automatic evaluation of all combinations

### Output CSV entries added:
```
domainaware_in2out: Indoor, Outdoor NMSE
domainaware_out2in: Indoor, Outdoor NMSE
```

### Output heatmaps:
```
heatmaps_indoor_domainaware_in2out.png
heatmaps_outdoor_domainaware_in2out.png
heatmaps_indoor_domainaware_out2in.png
heatmaps_outdoor_domainaware_out2in.png
```

---

# 11. Recommended Execution Order

1. `dataset_analysis.ipynb`
2. `CSINet-train.ipynb`  
   - indoor  
   - outdoor  
   - indoor+outdoor
3. `CSINet-train-simpleFinetune.ipynb`
4. `CSINet-train-replay.ipynb`
5. `CSINet-train-domainAware.ipynb`
6. `CSINet-test.ipynb`
7. `CSINet-test-domainAware.ipynb`

---

# 12. How to Interpret the Results

### NMSE (dB)
- Lower = better
- Cross-domain degradation reveals dataset shift  
- Replay and domain-aware usually improve target NMSE while preserving source NMSE

### Heatmaps
- Bright areas = large magnitude in angle–delay  
- Compare reconstructed vs original  
- Blurry or shifted clusters reveal reconstruction errors

### Learning curves
- Check for:
  - Convergence
  - Overfitting
  - Catastrophic forgetting (finetune)
  - Replay stabilization
  - Adapter specialization

---

# 13. Final Outputs Summary

After running all notebooks you obtain:

```
results/
  nmse_results.csv
  heatmaps_*.png

checkpoints/
  csinet_best_indoor.pt
  csinet_best_outdoor.pt
  csinet_best_indoor+outdoor.pt
  csinet_best_finetune_*.pt
  csinet_best_replay_*.pt
  csinet_domainaware_phase*.pt

logs/
  training curves + losses
```
