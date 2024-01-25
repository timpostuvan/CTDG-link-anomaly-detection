"""
Train a TG model without improved training regime and evaluate it with TGB package.
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
def anomaly_detection(args: SimpleNamespace):
    return anomaly_detection_single(args=args)


def anomaly_detection_single(args: SimpleNamespace):
    # Silence PyTorch warnings.
    warnings.filterwarnings("ignore")
    # Silence a warning because there is no git executable.
    os.environ["GIT_PYTHON_REFRESH"] = "quiet"

    # Different types of anomalies should be saved in separate experiments.
    args.experiment_name = (
        f"{args.experiment_name}/{args.anom_type}/{args.dataset_name}/{args.model_name}"
    )

    # Set up MLflow and the experiment.
    mlflow.set_tracking_uri(f"file://{args.mlflow_tracking_uri}")

    experiment_id = mlflow.get_experiment_by_name(args.experiment_name)
    if experiment_id is None:
        experiment_id = mlflow.create_experiment(args.experiment_name)
    else:
        experiment_id = experiment_id.experiment_id

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
        anom_type=args.anom_type,
        anom_set_id=args.anom_set_id,
        device=args.device,
    )

    # Ensure to only sample actual destination nodes as negatives.
    min_dst_idx = int(train_data.dst_node_ids.min())
    max_dst_idx = int(train_data.dst_node_ids.max())

    # Timespan of the training split
    min_train_timestamp = int(train_data.node_interact_times.min())
    max_train_timestamp = int(train_data.node_interact_times.max())

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

    # Get data loaders.
    train_idx_data_loader = get_idx_data_loader(
        indices_list=list(range(len(train_data.src_node_ids))),
        batch_size=args.batch_size,
        shuffle=False,
    )
    val_idx_data_loader = get_idx_data_loader(
        indices_list=list(range(len(val_data.src_node_ids))),
        batch_size=args.batch_size,
        shuffle=False,
    )
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

    # Define optimizer.
    optimizer = create_optimizer(
        model=model,
        node_feat_projection=node_raw_features,
        edge_feat_projection=edge_raw_features,
        optimizer_name=args.optimizer,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    model = convert_to_gpu(model, device=args.device)

    save_model_folder = (
        f"{args.output_root}/saved_models/{args.experiment_name}/{args.save_model_name}"
    )
    shutil.rmtree(save_model_folder, ignore_errors=True)
    os.makedirs(save_model_folder, exist_ok=True)

    early_stopping = EarlyStopping(
        patience=args.patience,
        save_model_folder=save_model_folder,
        save_model_name=args.save_model_name,
        logger=logger,
        model_name=args.model_name,
    )

    loss_func = nn.BCEWithLogitsLoss()

    # ================================================
    # ============== train & validation ==============
    # ================================================
    val_perf_list = []
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=args.save_model_name,
    ):
        # Log all arguments and model size to MLflow.
        mlflow.log_params({**vars(args), "num_parameters": get_parameter_sizes(model)})

        for epoch in range(args.num_epochs):
            start_epoch = timeit.default_timer()
            model.train()
            if args.model_name in [
                "DyRep",
                "TGAT",
                "TGN",
                "CAWN",
                "TCL",
                "GraphMixer",
                "DyGFormer",
            ]:
                # Training, only use training graph.
                model[0].set_neighbor_sampler(train_neighbor_sampler)
            if args.model_name in ["JODIE", "DyRep", "TGN"]:
                # Reinitialize memory of memory-based models at the start of each epoch.
                model[0].memory_bank.__init_memory_bank__()

            # Store train losses and metrics.
            train_losses, train_metrics = [], []
            train_idx_data_loader_tqdm = tqdm(
                train_idx_data_loader,
                ncols=120,
                disable=(not args.show_progress_bar),
            )
            for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
                (
                    batch_src_node_ids,
                    batch_dst_node_ids,
                    batch_node_interact_times,
                    batch_edge_ids,
                ) = (
                    train_data.src_node_ids[train_data_indices],
                    train_data.dst_node_ids[train_data_indices],
                    train_data.node_interact_times[train_data_indices],
                    train_data.edge_ids[train_data_indices],
                )
                batch_msg_feat = model[0].edge_raw_features[batch_edge_ids]

                batch_neg_src_node_ids = np.copy(batch_src_node_ids)
                batch_neg_dst_node_ids = np.copy(batch_dst_node_ids)
                batch_neg_node_interact_times = np.copy(batch_node_interact_times)
                batch_neg_msg_feat = batch_msg_feat.clone()

                # Create negative samples via structural (0), contextual (1), and temporal (2) perturbations.
                perturbation_type = np.zeros(shape=batch_src_node_ids.shape[0])

                # Create structural anomalies by randomly sampling destinations.
                structural_neg_dst_node_ids = np.random.randint(
                    min_dst_idx,
                    max_dst_idx + 1,
                    size=batch_neg_src_node_ids.shape[0],
                )
                structural_mask = perturbation_type == 0
                batch_neg_dst_node_ids[structural_mask] = structural_neg_dst_node_ids[
                    structural_mask
                ]

                # Create contextual anomalies by randomly permuting the edge messages.
                perm = torch.randperm(batch_msg_feat.shape[0])
                contextual_neg_msg_feat = batch_msg_feat[perm]
                contextual_mask = perturbation_type == 1
                batch_neg_msg_feat[contextual_mask] = contextual_neg_msg_feat[
                    contextual_mask
                ]

                # Create temporal anomalies by randomly sampling timestamps.
                temporal_neg_node_interact_times = np.random.randint(
                    min_train_timestamp,
                    max_train_timestamp + 1,
                    size=batch_neg_src_node_ids.shape[0],
                )
                temporal_mask = perturbation_type == 2
                batch_neg_node_interact_times[
                    temporal_mask
                ] = temporal_neg_node_interact_times[temporal_mask]

                if args.model_name in ["TGAT", "CAWN", "TCL"]:
                    # Get temporal embedding of source and destination nodes.
                    # Two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings = model[
                        0
                    ].compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_src_node_ids,
                        dst_node_ids=batch_dst_node_ids,
                        node_interact_times=batch_node_interact_times,
                        num_neighbors=args.num_neighbors,
                    )

                    # Get temporal embedding of negative source and negative destination nodes.
                    # Two Tensors, with shape (batch_size, node_feat_dim)
                    (
                        batch_neg_src_node_embeddings,
                        batch_neg_dst_node_embeddings,
                    ) = model[0].compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_neg_src_node_ids,
                        dst_node_ids=batch_neg_dst_node_ids,
                        node_interact_times=batch_neg_node_interact_times,
                        num_neighbors=args.num_neighbors,
                    )
                elif args.model_name in ["JODIE", "DyRep", "TGN"]:
                    # Note that negative nodes do not change the memories while the positive nodes change
                    # the memories during training.
                    # We need to first compute the embeddings of negative nodes for memory-based models.
                    # Get temporal embedding of negative source and negative destination nodes.
                    # Two Tensors, with shape (batch_size, node_feat_dim)
                    (
                        batch_neg_src_node_embeddings,
                        batch_neg_dst_node_embeddings,
                    ) = model[0].compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_neg_src_node_ids,
                        dst_node_ids=batch_neg_dst_node_ids,
                        node_interact_times=batch_neg_node_interact_times,
                        edge_ids=None,
                        edges_are_positive=False,
                        num_neighbors=args.num_neighbors,
                    )

                    # Get temporal embedding of source and destination nodes.
                    # Two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings = model[
                        0
                    ].compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_src_node_ids,
                        dst_node_ids=batch_dst_node_ids,
                        node_interact_times=batch_node_interact_times,
                        edge_ids=batch_edge_ids,
                        edges_are_positive=True,
                        num_neighbors=args.num_neighbors,
                    )
                elif args.model_name in ["GraphMixer"]:
                    # Get temporal embedding of source and destination nodes.
                    # Two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings = model[
                        0
                    ].compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_src_node_ids,
                        dst_node_ids=batch_dst_node_ids,
                        node_interact_times=batch_node_interact_times,
                        num_neighbors=args.num_neighbors,
                        time_gap=args.time_gap,
                    )

                    # Get temporal embedding of negative source and negative destination nodes.
                    # Two Tensors, with shape (batch_size, node_feat_dim)
                    (
                        batch_neg_src_node_embeddings,
                        batch_neg_dst_node_embeddings,
                    ) = model[0].compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_neg_src_node_ids,
                        dst_node_ids=batch_neg_dst_node_ids,
                        node_interact_times=batch_neg_node_interact_times,
                        num_neighbors=args.num_neighbors,
                        time_gap=args.time_gap,
                    )
                elif args.model_name in ["DyGFormer"]:
                    # Get temporal embedding of source and destination nodes.
                    # Two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings = model[
                        0
                    ].compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_src_node_ids,
                        dst_node_ids=batch_dst_node_ids,
                        node_interact_times=batch_node_interact_times,
                    )

                    # Get temporal embedding of negative source and negative destination nodes.
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    (
                        batch_neg_src_node_embeddings,
                        batch_neg_dst_node_embeddings,
                    ) = model[0].compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_neg_src_node_ids,
                        dst_node_ids=batch_neg_dst_node_ids,
                        node_interact_times=batch_neg_node_interact_times,
                    )
                else:
                    raise ValueError(f"Wrong value for model_name {args.model_name}!")
                # Get positive and negative probabilities, shape (batch_size, )
                positive_probabilities = (
                    model[1](
                        node_embedding1=batch_src_node_embeddings,
                        node_embedding2=batch_dst_node_embeddings,
                        context=batch_msg_feat,
                    )
                    .squeeze(dim=-1)
                    .sigmoid()
                )
                negative_probabilities = (
                    model[1](
                        node_embedding1=batch_neg_src_node_embeddings,
                        node_embedding2=batch_neg_dst_node_embeddings,
                        context=batch_neg_msg_feat,
                    )
                    .squeeze(dim=-1)
                    .sigmoid()
                )

                predicts = torch.cat(
                    [positive_probabilities, negative_probabilities], dim=0
                )
                labels = torch.cat(
                    [
                        torch.ones_like(positive_probabilities),
                        torch.zeros_like(negative_probabilities),
                    ],
                    dim=0,
                )

                loss = loss_func(input=predicts, target=labels)

                train_losses.append(loss.item())

                # Masks for calculation of AUC for each type of perturbation separately
                structural_mask = torch.tensor(
                    np.concatenate([structural_mask, structural_mask])
                )
                contextual_mask = torch.tensor(
                    np.concatenate([contextual_mask, contextual_mask])
                )
                temporal_mask = torch.tensor(
                    np.concatenate([temporal_mask, temporal_mask])
                )

                train_metrics.append(
                    get_link_prediction_metrics(
                        predicts=predicts,
                        labels=labels,
                        structural_mask=structural_mask,
                        contextual_mask=contextual_mask,
                        temporal_mask=temporal_mask,
                    )
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Log training loss for each batch.
                train_idx_data_loader_tqdm.set_description(
                    f"Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss.item()}"
                )
                if batch_idx % 100 == 0:
                    mlflow.log_metric(
                        "Train/Loss",
                        loss.item(),
                        step=math.ceil(epoch * len(train_idx_data_loader)) + batch_idx,
                    )

                if args.model_name in ["JODIE", "DyRep", "TGN"]:
                    # Detach the memories and raw messages of nodes in the memory bank after each batch
                    # so we don't back propagate to the start of time.
                    model[0].memory_bank.detach_memory_bank()

            # === validation
            # After one complete epoch, evaluate the model on the validation set.
            val_metrics = eval_linkanomdet_TGB(
                model_name=args.model_name,
                model=model,
                neighbor_sampler=full_neighbor_sampler,
                evaluate_idx_data_loader=val_idx_data_loader,
                evaluate_data=val_data,
                evaluator=evaluator,
                metrics=metric,
                num_neighbors=args.num_neighbors,
                time_gap=args.time_gap,
                show_progress_bar=args.show_progress_bar,
            )
            val_perf_list.append(val_metrics[main_metric])

            # Log average train loss and elapsed time for the epoch.
            epoch_time = timeit.default_timer() - start_epoch
            average_train_loss = np.mean(train_losses)
            logger.info(
                f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, average train loss: {average_train_loss:.4f}, elapsed time (s): {epoch_time:.4f}'
            )
            mlflow.log_metric("Train/Average loss", average_train_loss, step=epoch + 1)
            mlflow.log_metric("Train/Epoch elapsed time", epoch_time, step=epoch + 1)

            # Log average train link prediction metrics.
            for metric_name in train_metrics[0].keys():
                average_train_metric = np.mean(
                    [train_metric[metric_name] for train_metric in train_metrics]
                )
                logger.info(
                    f"Train link prediction {metric_name}, {average_train_metric:.4f}"
                )
                mlflow.log_metric(
                    f"Train/Link prediction {metric_name}",
                    average_train_metric,
                    step=epoch + 1,
                )

            # Log validation anomaly detection metrics.
            for metric_name, metric_score in val_metrics.items():
                logger.info(f"Validation: {metric_name}: {metric_score: .4f}")

                mlflow.log_metric(
                    f"Validation/Anomaly {metric_name.replace('@', '_')}",
                    metric_score,
                    step=epoch + 1,
                )

            # Select the best model based on main validation metric.
            val_metric_indicator = [(main_metric, val_metrics[main_metric], True)]
            early_stop = early_stopping.step(val_metric_indicator, model)

            if early_stop:
                break

        # Load the best model.
        early_stopping.load_checkpoint(model)

        # Log total elapsed time for training and validation.
        total_train_val_time = timeit.default_timer() - start_run
        logger.info(
            f"Total train & validation elapsed time (s): {total_train_val_time:.6f}"
        )

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
            mlflow.log_metric(
                f"Test/Anomaly {metric_name.replace('@', '_')}", metric_score
            )

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
            f"validation {main_metric}": val_perf_list,
            "test_time": test_time,
            "total_train_val_time": total_train_val_time,
            **test_results,
        }
        result_json = json.dumps(result_json, indent=4)

        save_result_folder = f"{args.output_root}/saved_results/{args.experiment_name}"
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(
            save_result_folder, f"{args.save_model_name}.json"
        )

        with open(save_result_path, "w") as file:
            file.write(result_json)

        # Log total elapsed time.
        total_elapsed_time = timeit.default_timer() - start_run
        logger.info(
            f"run {args.run_id} total elapsed time (s): {total_elapsed_time:.4f}"
        )
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

    # Convert common grid search hyperparameters to lists.
    if not isinstance(config.data.dataset_name, ListConfig):
        config.data.dataset_name = ListConfig([config.data.dataset_name])

    if not isinstance(config.data.anom_type, ListConfig):
        config.data.anom_type = ListConfig([config.data.anom_type])

    if not isinstance(config.training.learning_rate, ListConfig):
        config.training.learning_rate = ListConfig([config.training.learning_rate])

    ray_runs_ids = []
    for dataset_name, anom_type, learning_rate in itertools.product(
        config.data.dataset_name, config.data.anom_type, config.training.learning_rate
    ):
        for model_name in config.models.keys():
            # Create MLflow experiment so that all concurent runs can read the experiment ID.
            experiment_name = f'{common_args["experiment_name"]}/{anom_type}/{dataset_name}/{model_name}'
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
                        args["anom_type"] = anom_type
                        args["learning_rate"] = learning_rate
                        args["model_name"] = model_name
                        args["anom_set_id"] = anom_set_id
                        args["run_id"] = run_id
                        args.update(model_config)

                        # Convert args from dictonary to SimpleNamespace.
                        args = SimpleNamespace(**args)

                        # Run experiment.
                        if config.general.debug:
                            anomaly_detection_single(args=args)
                        else:
                            ray_runs_ids.append(anomaly_detection.remote(args=args))

    if not config.general.debug:
        _ = ray.get(ray_runs_ids)


if __name__ == "__main__":
    main()
