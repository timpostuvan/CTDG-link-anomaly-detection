"""
Evaluate a TG model with TGB package.
NOTE: The task is Transductive Dynamic Link Anomaly Detection.
"""

import copy
import logging
import timeit
import time
import datetime
import sys
import os
import math
from tqdm import tqdm
import numpy as np
import warnings
import shutil
import json
import ray
import torch
import torch.nn as nn
import os.path as osp
import mlflow
import itertools
from types import SimpleNamespace
from omegaconf import OmegaConf, ListConfig

from models.TGAT import TGAT
from models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from models.CAWN import CAWN
from models.TCL import TCL
from models.GraphMixer import GraphMixer
from models.DyGFormer import DyGFormer
from models.modules import ContextualLinkPredictor
from utils.utils import (
    get_ray_head_ip,
    set_random_seed,
    convert_to_gpu,
    get_parameter_sizes,
    create_optimizer,
)
from utils.utils import get_neighbor_sampler, NegativeEdgeSampler
from evaluate_models_utils import evaluate_model_link_prediction
from utils.metrics import get_link_prediction_metrics
from utils.DataLoader import (
    get_idx_data_loader,
    get_link_prediction_data,
    get_link_anom_det_data_TRANS_TGB,
)
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import (
    get_config_for_anomaly_detection,
    update_config_anomaly_detection,
)

import tgb
from tgb.linkanomdet.evaluate import Evaluator
from evaluation.tgb_evaluate_linkanomdet import eval_linkanomdet_TGB


@ray.remote(
    num_cpus=int(os.environ.get("NUM_CPUS_PER_TASK", 4)),
    num_gpus=float(os.environ.get("NUM_GPUS_PER_TASK", 0.0)),
    memory=10 * 1024 * 1024 * 1024,  # 10 GB
)
def evaluate_anomaly_detection(args: SimpleNamespace):
    return evaluate_anomaly_detection_single(args=args)


def evaluate_anomaly_detection_single(args: SimpleNamespace):
    # Silence PyTorch warnings.
    warnings.filterwarnings("ignore")
    # Silence a warning because there is no git executable.
    os.environ["GIT_PYTHON_REFRESH"] = "quiet"

    # Different types of anomalies should be saved in separate experiments.
    args.experiment_name = f"{args.experiment_name}/{args.val_anom_type}-{args.test_anom_type}/{args.dataset_name}/{args.model_name}"

    # Get data for training, validation and testing.
    (
        node_raw_features,
        edge_raw_features,
        full_data,
        train_data,
        val_data,
        test_data,
        dataset,
    ) = get_link_anom_det_data_TRANS_TGB(
        dataset_name=args.dataset_name,
        dataset_root=args.dataset_root,
        absolute_path=True,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        node_feat_dim=args.node_feat_dim,
        edge_feat_dim=args.edge_feat_dim,
        val_anom_type=args.val_anom_type,
        test_anom_type=args.test_anom_type,
        anom_set_id=args.anom_set_id,
        device=args.device,
    )

    # Initialize training neighbor sampler to retrieve temporal graph.
    train_neighbor_sampler = get_neighbor_sampler(
        data=train_data,
        sample_neighbor_strategy=args.sample_neighbor_strategy,
        time_scaling_factor=args.time_scaling_factor,
        seed=0,
    )

    # Initialize validation and test neighbor sampler to retrieve temporal graph.
    full_neighbor_sampler = get_neighbor_sampler(
        data=full_data,
        sample_neighbor_strategy=args.sample_neighbor_strategy,
        time_scaling_factor=args.time_scaling_factor,
        seed=1,
    )

    # Get data loader.
    test_idx_data_loader = get_idx_data_loader(
        indices_list=list(range(len(test_data.src_node_ids))),
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Evaluating with an evaluator of TGB
    metric = dataset.eval_metric
    main_metric = "auc"
    evaluator = Evaluator(anom_label=0)

    start_run = timeit.default_timer()
    set_random_seed(seed=args.seed + args.run_id)

    args.save_model_name = f"{args.model_name}_{args.dataset_name}_lr_{args.learning_rate}_seed_{args.seed}_set_{args.anom_set_id}_run_{args.run_id}"

    # Set up logger.
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(f"{args.experiment_name}/{args.save_model_name}")
    logger.setLevel(logging.INFO)
    os.makedirs(
        f"{args.output_root}/logs/{args.experiment_name}/{args.save_model_name}/",
        exist_ok=True,
    )
    # Create file handler that logs debug and higher level messages.
    log_start_time = datetime.datetime.fromtimestamp(time.time()).strftime(
        "%Y-%m-%d_%H:%M:%S"
    )
    fh = logging.FileHandler(
        f"{args.output_root}/logs/{args.experiment_name}/{args.save_model_name}/{str(log_start_time)}.log"
    )
    fh.setLevel(logging.DEBUG)
    # Create console handler with a higher log level.
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    # Create formatter and add it to the handlers.
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # Add the handlers to logger.
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"********** Run {args.run_id + 1} starts. **********")

    logger.info(f"Configuration is {args}")

    # Create model.
    if args.model_name == "TGAT":
        dynamic_backbone = TGAT(
            node_raw_features=node_raw_features,
            edge_raw_features=edge_raw_features,
            neighbor_sampler=train_neighbor_sampler,
            time_feat_dim=args.time_feat_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            device=args.device,
        )
    elif args.model_name in ["JODIE", "DyRep", "TGN"]:
        # Four floats that represent the mean and standard deviation of source and destination node time shifts
        # in the training data, which is used for JODIE.
        (
            src_node_mean_time_shift,
            src_node_std_time_shift,
            dst_node_mean_time_shift_dst,
            dst_node_std_time_shift,
        ) = compute_src_dst_node_time_shifts(
            train_data.src_node_ids,
            train_data.dst_node_ids,
            train_data.node_interact_times,
        )
        dynamic_backbone = MemoryModel(
            node_raw_features=node_raw_features,
            edge_raw_features=edge_raw_features,
            neighbor_sampler=train_neighbor_sampler,
            time_feat_dim=args.time_feat_dim,
            model_name=args.model_name,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            src_node_mean_time_shift=src_node_mean_time_shift,
            src_node_std_time_shift=src_node_std_time_shift,
            dst_node_mean_time_shift_dst=dst_node_mean_time_shift_dst,
            dst_node_std_time_shift=dst_node_std_time_shift,
            device=args.device,
        )
    elif args.model_name == "CAWN":
        dynamic_backbone = CAWN(
            node_raw_features=node_raw_features,
            edge_raw_features=edge_raw_features,
            neighbor_sampler=train_neighbor_sampler,
            time_feat_dim=args.time_feat_dim,
            position_feat_dim=args.position_feat_dim,
            walk_length=args.walk_length,
            num_walk_heads=args.num_walk_heads,
            dropout=args.dropout,
            device=args.device,
        )
    elif args.model_name == "TCL":
        dynamic_backbone = TCL(
            node_raw_features=node_raw_features,
            edge_raw_features=edge_raw_features,
            neighbor_sampler=train_neighbor_sampler,
            time_feat_dim=args.time_feat_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            num_depths=args.num_neighbors + 1,
            dropout=args.dropout,
            device=args.device,
        )
    elif args.model_name == "GraphMixer":
        dynamic_backbone = GraphMixer(
            node_raw_features=node_raw_features,
            edge_raw_features=edge_raw_features,
            neighbor_sampler=train_neighbor_sampler,
            time_feat_dim=args.time_feat_dim,
            num_tokens=args.num_neighbors,
            num_layers=args.num_layers,
            dropout=args.dropout,
            device=args.device,
        )
    elif args.model_name == "DyGFormer":
        dynamic_backbone = DyGFormer(
            node_raw_features=node_raw_features,
            edge_raw_features=edge_raw_features,
            neighbor_sampler=train_neighbor_sampler,
            time_feat_dim=args.time_feat_dim,
            channel_embedding_dim=args.channel_embedding_dim,
            patch_size=args.patch_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            max_input_sequence_length=args.max_input_sequence_length,
            device=args.device,
        )
    else:
        raise ValueError(f"Wrong value for model_name {args.model_name}!")
    link_predictor = ContextualLinkPredictor(
        input_dim_node1=node_raw_features.embedding_dim,
        input_dim_node2=node_raw_features.embedding_dim,
        input_dim_context=edge_raw_features.embedding_dim,
        hidden_dim=node_raw_features.embedding_dim,
        output_dim=1,
    )
    model = nn.Sequential(dynamic_backbone, link_predictor)
    logger.info(f"model -> {model}")
    logger.info(
        f"model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, "
        f"{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB."
    )

    model = convert_to_gpu(model, device=args.device)

    load_model_folder = (
        f"{args.output_root}/saved_models/{args.experiment_name}/{args.save_model_name}"
    )
    early_stopping = EarlyStopping(
        patience=args.patience,
        save_model_folder=load_model_folder,
        save_model_name=args.save_model_name,
        logger=logger,
        model_name=args.model_name,
    )
    # Load the best model.
    early_stopping.load_checkpoint(model)

    # ========================================
    # ============== Final Test ==============
    # ========================================
    start_test = timeit.default_timer()
    test_metrics = eval_linkanomdet_TGB(
        model_name=args.model_name,
        model=model,
        neighbor_sampler=full_neighbor_sampler,
        evaluate_idx_data_loader=test_idx_data_loader,
        evaluate_data=test_data,
        evaluator=evaluator,
        metrics=metric,
        num_neighbors=args.num_neighbors,
        time_gap=args.time_gap,
        show_progress_bar=args.show_progress_bar,
    )

    # Log elapsed time for testing.
    test_time = timeit.default_timer() - start_test
    logger.info(f"Test elapsed time (s): {test_time:.4f}")
    mlflow.log_metric("Test/Elapsed time", test_time)

    # Log testing anomaly detection metrics.
    for metric_name, metric_score in test_metrics.items():
        logger.info(f"Test: {metric_name}: {metric_score: .4f}")
        mlflow.log_metric(f"Test/Anomaly {metric_name.replace('@', '_')}", metric_score)

    test_results = {
        f"test {metric_name}": metric_score
        for metric_name, metric_score in test_metrics.items()
    }

    # Save model results.
    result_json = {
        "data": args.dataset_name,
        "model": args.model_name,
        "run": args.run_id,
        "seed": args.seed,
        "test_time": test_time,
        **test_results,
    }
    result_json = json.dumps(result_json, indent=4)

    save_result_folder = f"{args.output_root}/saved_results/{args.experiment_name}"
    os.makedirs(save_result_folder, exist_ok=True)
    save_result_path = os.path.join(
        save_result_folder, f"evaluation_{args.save_model_name}.json"
    )

    with open(save_result_path, "w") as file:
        file.write(result_json)

    # Log total elapsed time.
    total_elapsed_time = timeit.default_timer() - start_run
    logger.info(f"run {args.run_id} total elapsed time (s): {total_elapsed_time:.4f}")
    mlflow.log_metric("Total elapsed time", total_elapsed_time)


def main():
    # Get the path to configuration file of the experiment.
    config_path = get_config_for_anomaly_detection().config_path
    config = OmegaConf.load(config_path)

    # Add missing common hyperparameters to the config.
    common_args: dict = update_config_anomaly_detection(config)

    if config.general.debug:
        # Run on local cluster.
        common_args["device"] = f"cuda" if torch.cuda.is_available() else "cpu"
    else:
        # Run on CPU cluster with Ray.
        common_args["device"] = "cpu"

        # Kubernetes cluster initialization.
        runtime_env = {
            "working_dir": os.getcwd(),
            "py_modules": ["../tgb/tgb"],
            "excludes": ["datasets/"],
        }

        head_ip = get_ray_head_ip()
        ray.init(f"ray://{head_ip}:10001", runtime_env=runtime_env)

    # Set up MLflow.
    mlflow.set_tracking_uri(f'file://{common_args["mlflow_tracking_uri"]}')

    # If config does not contain ´val_anom_type´ and ´test_anom_type´, set them to ´anom_type´.
    if config.data.get("val_anom_type") is None:
        if config.data.get("anom_type") is None:
            raise ValueError(
                "val_anom_type and anom_type are both unset! Please specify at least one of them."
            )
        else:
            config.data.val_anom_type = config.data.anom_type

    if config.data.get("test_anom_type") is None:
        if config.data.get("anom_type") is None:
            raise ValueError(
                "test_anom_type and anom_type are both unset! Please specify at least one of them."
            )
        else:
            config.data.test_anom_type = config.data.anom_type

    # Convert common grid search hyperparameters to lists.
    if not isinstance(config.data.dataset_name, ListConfig):
        config.data.dataset_name = ListConfig([config.data.dataset_name])

    if not isinstance(config.data.val_anom_type, ListConfig):
        config.data.val_anom_type = ListConfig([config.data.val_anom_type])

    if not isinstance(config.data.test_anom_type, ListConfig):
        config.data.test_anom_type = ListConfig([config.data.test_anom_type])

    if not isinstance(config.training.learning_rate, ListConfig):
        config.training.learning_rate = ListConfig([config.training.learning_rate])

    ray_runs_ids = []
    for (
        dataset_name,
        (val_anom_type, test_anom_type),
        learning_rate,
    ) in itertools.product(
        config.data.dataset_name,
        zip(config.data.val_anom_type, config.data.test_anom_type),
        config.training.learning_rate,
    ):
        for model_name in config.models.keys():
            # Create MLflow experiment so that all concurent runs can read the experiment ID.
            experiment_name = f'{common_args["experiment_name"]}/{val_anom_type}-{test_anom_type}/{dataset_name}/{model_name}'
            experiment_id = mlflow.get_experiment_by_name(experiment_name)
            if experiment_id is None:
                experiment_id = mlflow.create_experiment(experiment_name)

            # Convert model-specific hyperparameters to lists for grid search.
            for param in config.models[model_name].keys():
                if not isinstance(config.models[model_name][param], ListConfig):
                    config.models[model_name][param] = ListConfig(
                        [config.models[model_name][param]]
                    )

            # Cartesian product of all values of all hyperparameters
            all_model_configs = [
                dict(zip(config.models[model_name].keys(), x))
                for x in itertools.product(*config.models[model_name].values())
            ]

            for model_config in all_model_configs:
                for anom_set_id in range(common_args["num_anom_sets"]):
                    for run_id in range(common_args["num_runs"]):
                        args = copy.copy(common_args)
                        # Update run-specific hyperparameters.
                        args["dataset_name"] = dataset_name
                        args["val_anom_type"] = val_anom_type
                        args["test_anom_type"] = test_anom_type
                        args["learning_rate"] = learning_rate
                        args["model_name"] = model_name
                        args["anom_set_id"] = anom_set_id
                        args["run_id"] = run_id
                        args.update(model_config)

                        # Convert args from dictonary to SimpleNamespace.
                        args = SimpleNamespace(**args)

                        # Run experiment.
                        if config.general.debug:
                            evaluate_anomaly_detection_single(args=args)
                        else:
                            ray_runs_ids.append(
                                evaluate_anomaly_detection.remote(args=args)
                            )

    if not config.general.debug:
        _ = ray.get(ray_runs_ids)


if __name__ == "__main__":
    main()
