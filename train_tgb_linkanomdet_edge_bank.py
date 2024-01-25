"""
Train an EdgeBank model and evaluate it with TGB package.
NOTE: The task is Transductive Dynamic Link Anomaly Detection.
"""

import copy
import logging
import timeit
import time
import datetime
import os
from tqdm import tqdm
import numpy as np
import warnings
import json
import ray
import mlflow
import itertools
from types import SimpleNamespace
from omegaconf import OmegaConf, ListConfig

from models.EdgeBank import edge_bank_link_prediction
from utils.utils import (
    get_ray_head_ip,
    set_random_seed,
)
from utils.DataLoader import (
    get_idx_data_loader,
    get_link_anom_det_data_TRANS_TGB,
    Data,
)
from utils.load_configs import (
    get_config_for_anomaly_detection,
    update_config_anomaly_detection,
)

import tgb
from tgb.linkanomdet.evaluate import Evaluator


@ray.remote(
    num_cpus=int(os.environ.get("NUM_CPUS_PER_TASK", 4)),
    num_gpus=float(os.environ.get("NUM_GPUS_PER_TASK", 0.0)),
    memory=10 * 1024 * 1024 * 1024,  # 10 GB
)
def anomaly_detection_edge_bank(args: SimpleNamespace):
    return anomaly_detection_edge_bank_single(args=args)


def anomaly_detection_edge_bank_single(args: SimpleNamespace):
    # Silence PyTorch warnings.
    warnings.filterwarnings("ignore")
    # Silence a warning because there is no git executable.
    os.environ["GIT_PYTHON_REFRESH"] = "quiet"

    # Different types of anomalies should be saved in separate experiments.
    args.experiment_name = f"{args.experiment_name}/{args.val_anom_type}-{args.test_anom_type}/{args.dataset_name}/{args.model_name}"

    # Set up MLflow and the experiment.
    mlflow.set_tracking_uri(f"file://{args.mlflow_tracking_uri}")

    experiment_id = mlflow.get_experiment_by_name(args.experiment_name)
    if experiment_id is None:
        experiment_id = mlflow.create_experiment(args.experiment_name)
    else:
        experiment_id = experiment_id.experiment_id

    # Get data for training, validation and testing.
    (
        _,
        _,
        _,
        train_data,
        val_data,
        test_data,
        dataset,
    ) = get_link_anom_det_data_TRANS_TGB(
        dataset_name=args.dataset_name,
        absolute_path=True,
        node_feat_dim=args.node_feat_dim,
        edge_feat_dim=args.edge_feat_dim,
        dataset_root=args.dataset_root,
        val_anom_type=args.val_anom_type,
        test_anom_type=args.test_anom_type,
        anom_set_id=args.anom_set_id,
        device=args.device,
    )

    # Keep only positive edges in validation set for EdgeBank.
    val_positive_mask = val_data.labels == 1
    val_positive_data = Data(
        src_node_ids=val_data.src_node_ids[val_positive_mask],
        dst_node_ids=val_data.dst_node_ids[val_positive_mask],
        node_interact_times=val_data.node_interact_times[val_positive_mask],
        edge_ids=val_data.edge_ids[val_positive_mask],
        labels=val_data.labels[val_positive_mask],
    )

    # Generate the train_validation split of the data: needed for constructing the memory for EdgeBank.
    train_val_data = Data(
        src_node_ids=np.concatenate(
            [train_data.src_node_ids, val_positive_data.src_node_ids]
        ),
        dst_node_ids=np.concatenate(
            [train_data.dst_node_ids, val_positive_data.dst_node_ids]
        ),
        node_interact_times=np.concatenate(
            [train_data.node_interact_times, val_positive_data.node_interact_times]
        ),
        edge_ids=np.concatenate([train_data.edge_ids, val_positive_data.edge_ids]),
        labels=np.concatenate([train_data.labels, val_positive_data.labels]),
    )

    # Get data loaders.
    test_idx_data_loader = get_idx_data_loader(
        indices_list=list(range(len(test_data.src_node_ids))),
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Evaluating with an evaluator of TGB
    metric = dataset.eval_metric
    main_metric = "auc"
    evaluator = Evaluator(anom_label=0)

    set_random_seed(seed=args.seed + args.run_id)

    args.save_model_name = f"{args.model_name}_{args.dataset_name}_seed_{args.seed}_set_{args.anom_set_id}_run_{args.run_id}"

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

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=args.save_model_name,
    ):
        # Log all arguments and model size to MLflow.
        mlflow.log_params({**vars(args), "num_parameters": 0})

        start_test = timeit.default_timer()

        y_preds, y_labels = [], []
        test_idx_data_loader_tqdm = tqdm(
            test_idx_data_loader,
            ncols=120,
            disable=(not args.show_progress_bar),
        )
        for batch_idx, test_data_indices in enumerate(test_idx_data_loader_tqdm):
            (
                batch_src_node_ids,
                batch_dst_node_ids,
                batch_labels,
            ) = (
                test_data.src_node_ids[test_data_indices].reshape(-1),
                test_data.dst_node_ids[test_data_indices].reshape(-1),
                test_data.labels[test_data_indices].reshape(-1),
            )

            # In link anomaly detection task, the model does not know which links are benign and which are anomalous,
            # therefore, EdgeBank should be updated according to all edges. Specifically, the testing data before
            # the current batch has to be considered in historical data, which is similar to memory-based models.
            history_data = Data(
                src_node_ids=np.concatenate(
                    [
                        train_val_data.src_node_ids,
                        test_data.src_node_ids[: test_data_indices[0]],
                    ]
                ),
                dst_node_ids=np.concatenate(
                    [
                        train_val_data.dst_node_ids,
                        test_data.dst_node_ids[: test_data_indices[0]],
                    ]
                ),
                node_interact_times=np.concatenate(
                    [
                        train_val_data.node_interact_times,
                        test_data.node_interact_times[: test_data_indices[0]],
                    ]
                ),
                edge_ids=np.concatenate(
                    [
                        train_val_data.edge_ids,
                        test_data.edge_ids[: test_data_indices[0]],
                    ]
                ),
                labels=np.concatenate(
                    [train_val_data.labels, test_data.labels[: test_data_indices[0]]]
                ),
            )

            # Get prediction probabilities.
            batch_preds = edge_bank_link_prediction(
                history_data=history_data,
                edges=(batch_src_node_ids, batch_dst_node_ids),
                edge_bank_memory_mode=args.edge_bank_memory_mode,
                time_window_mode=args.time_window_mode,
                time_window_proportion=args.test_ratio,
            )

            y_preds.append(batch_preds)
            y_labels.append(batch_labels)

        # Compute evaluation metrics.
        input_dict = {
            "y_pred": np.concatenate(y_preds),
            "y_label": np.concatenate(y_labels),
            "eval_metric": metric,
        }
        test_metrics = evaluator.eval(input_dict)

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
            f"validation {main_metric}": [],
            "test_time": test_time,
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
        total_elapsed_time = timeit.default_timer() - start_test
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
        common_args["device"] = "cpu"
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

    ray_runs_ids = []
    for dataset_name, (val_anom_type, test_anom_type) in itertools.product(
        config.data.dataset_name,
        zip(config.data.val_anom_type, config.data.test_anom_type),
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
                        args["model_name"] = model_name
                        args["anom_set_id"] = anom_set_id
                        args["run_id"] = run_id
                        args.update(model_config)

                        # Convert args from dictonary to SimpleNamespace.
                        args = SimpleNamespace(**args)

                        # Run experiment.
                        if config.general.debug:
                            anomaly_detection_edge_bank_single(args=args)
                        else:
                            ray_runs_ids.append(
                                anomaly_detection_edge_bank.remote(args=args)
                            )

    if not config.general.debug:
        _ = ray.get(ray_runs_ids)


if __name__ == "__main__":
    main()
