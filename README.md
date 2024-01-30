# Towards Learning-Based Link Anomaly Detection in Continuous-Time Dynamic Graphs

## Overview
This repository contains the codebase supporting the findings of the paper "Towards Learning-Based Link Anomaly Detection in Continuous-Time Dynamic Graphs".
With the code provided in this repository, we explore performance of temporal graph learning models on link anomaly detection task.
See our paper for more details.

## Install Dependencies
Our implementation works with python >= 3.9 and can be installed as follows:

1. Set up a [conda](https://docs.conda.io/projects/conda/en/latest/index.html) environment.
```
conda create -n tgb_env python=3.9
conda activate tgb_env
```

2. Install external packages.
```
pip install pandas==1.5.3
pip install matplotlib==3.7.1
pip install clint==0.5.1
pip install mlflow==2.10.0
pip install omegaconf==2.3.0
```

Install Pytorch and PyG dependencies to run the examples.
```
pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cu117
pip install torch_geometric==2.3.0
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
```

3. Clone [TGB-link-anomaly-detection](https://github.com/timpostuvan/TGB-link-anomaly-detection) repository and install local dependencies under its root directory `/TGB-link-anomaly-detection`.
```
git clone https://github.com/timpostuvan/TGB-link-anomaly-detection.git
cd TGB-link-anomaly-detection
pip install -e .
```


## Running an Example
Running experiments generally require two steps: generating anomalies for the dataset, and training and evaluating the model. Additionally, some datasets (e.g., LANL and DARPA-THEIA) have to be first pre-processed with the scripts provided in [TGB-link-anomaly-detection](https://github.com/timpostuvan/TGB-link-anomaly-detection).
A simple example of training and evaluating the TGN model on Wikipedia dataset with injected temporal-structural-contextual anomalies can be run as follows:

1. Generate anomalies

To generate temporal-structural-contextual anomalies for the Wikipedia dataset from TGB, run the following command inside `/TGB-link-anomaly-detection/tgb/datasets/dataset_scripts` directory:
```
python link_anomaly_generator.py  \
--dataset_name tgbl-wiki  \
--val_ratio 0.15  \
--test_ratio 0.15  \
--anom_type temporal-structural-contextual  \
--output_root <OUTPUT-DIR>
```

The anomalies are generated for the validation and test splits according to the 70/15/15 data split. The data is saved under `<OUTPUT-DIR>` directory, which should be specified as an absolute path.

2. Train and evaluate the model

TGN model can be trained and evaluated by running the following command inside the root directory `/`:
```
python train_tgb_linkanomdet.py --config_path=experiments/example.yaml
```
Note that `<OUTPUT-DIR>` directory in the configuration file has to be substituted with the same directory that was specified when generating anomalies.

The results of the experiment and the best model checkpoints are saved in `<OUTPUT_DIR>/EXPERIMENTS/saved_results` and `<OUTPUT_DIR>/EXPERIMENTS/saved_models`, respectively.


## Experiments From the Paper
This section presents the commands to reproduce experiments from the paper.

### Experiments on Synthetic Graph and Real Graphs With Synthetic Anomalies
- Learning-based temporal graph models:
```
python train_tgb_linkanomdet.py --config_path=experiments/experiment.yaml
```

- EdgeBank models:
```
python train_tgb_linkanomdet_edge_bank.py --config_path=experiments/experiment_EdgeBank.yaml
```

### Experiments on Real Graphs With Organic Anomalies
- Learning-based temporal graph models:
```
python train_tgb_linkanomdet.py --config_path=experiments/experiment_LANL.yaml
python train_tgb_linkanomdet.py --config_path=experiments/experiment_DARPA_THEIA.yaml
```

- EdgeBank models:
```
python train_tgb_linkanomdet_edge_bank.py --config_path=experiments/experiment_EdgeBank_LANL.yaml
python train_tgb_linkanomdet_edge_bank.py --config_path=experiments/experiment_EdgeBank_DARPA_THEIA.yaml
```

### Experiment Comparing Synthetic and Organic Anomalies
- Learning-based temporal graph models:
```
python train_tgb_linkanomdet.py --config_path=experiments/experiment_synthetic_vs_organic_anomalies_LANL.yaml
python train_tgb_linkpred_for_linkanomdet.py --config_path=experiments/link_prediction_experiment_LANL.yaml
```

### Experiment Without Conditioning on Contextual Information
- Learning-based temporal graph models:
```
python train_tgb_linkanomdet_without_conditioning_on_context.py --config_path=experiments/experiment_without_conditioning_on_context.yaml
```

### Experiment Without Improved Training Regime
- Learning-based temporal graph models:
```
python train_tgb_linkanomdet_without_improved_training.py --config_path=experiments/experiment_without_improved_training.yaml
```

## Acknowledgments
The code is adapted from [TGB_Baselines repository](https://github.com/fpour/TGB_Baselines). If this code repository is useful for your research, please consider citing the original authors from [TGB](https://arxiv.org/pdf/2307.01026.pdf) paper as well.