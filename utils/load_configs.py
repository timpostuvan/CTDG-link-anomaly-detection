import os
import argparse
import sys
import torch
from omegaconf import DictConfig


def get_config_for_anomaly_detection():
    """
    Get the path to configuration file of the experiment for the link anomaly detection task.
    """
    # Arguments
    parser = argparse.ArgumentParser(
        "DyGLib: Interface for the link anomaly detection task"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to configuration file of the experiment",
    )
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit()

    return args


def update_config_anomaly_detection(config: DictConfig) -> dict:
    """
    Update the configuration file of the experiment with missing hyperparameters and
    check that the critical hyperparameters are specified for the link anomaly detection task.

    Args:
        config (DictConfig): Configuration file.

    Returns:
        dict: Updated configuration file.
    """

    # Default values of hyperparameters
    config_dict = dict(
        mlflow_tracking_uri="./mlruns",
        dataset_root="../../data",
        output_root=".",
        dataset_name="wikipedia",
        node_feat_dim=50,
        edge_feat_dim=50,
        anom_type=None,
        experiment_name="Experiment",
        batch_size=200,
        model_name="DyGFormer",
        num_neighbors=20,
        sample_neighbor_strategy="recent",
        time_scaling_factor=1e-6,
        num_walk_heads=8,
        num_heads=2,
        num_layers=2,
        walk_length=1,
        time_gap=2000,
        time_feat_dim=100,
        position_feat_dim=172,
        edge_bank_memory_mode="unlimited_memory",
        time_window_mode="fixed_proportion",
        patch_size=1,
        channel_embedding_dim=50,
        max_input_sequence_length=32,
        learning_rate=0.0001,
        dropout=0.1,
        num_epochs=50,
        optimizer="Adam",
        weight_decay=0.0,
        patience=20,
        val_ratio=0.15,
        test_ratio=0.15,
        num_runs=1,
        num_anom_sets=1,
        test_interval_epochs=10,
        seed=2023,
        show_progress_bar=False,
        debug=False,
    )

    # Check that groups of hyperparameters are specified.
    if config.get("general") is None:
        raise KeyError("No general information is specified.")

    if config.get("data") is None:
        raise KeyError("No information about dataset is specified.")

    if config.get("models") is None:
        raise KeyError("No information about models is specified.")

    if config.get("training") is None:
        raise KeyError("No information about training is specified.")

    # Check that the critical hyperparameters are specified.
    if config.general.get("output_root") is None:
        raise KeyError("Output directory is not specified.")

    if config.data.get("dataset_root") is None:
        raise KeyError("Dataset directory is not specified.")

    if config.data.get("dataset_name") is None:
        raise KeyError("Dataset name is not specified.")

    if len(config.models.keys()) == 0:
        raise KeyError("Model configurations are not specified.")
    else:
        config_dict["model_name"] = list(config.models.keys())

    # Update arguments with specified ones.
    config_dict.update(config.general)

    config_dict.update(config.data)

    config_dict.update(config.training)

    return config_dict


def get_anomaly_detection_args(is_evaluation: bool = False):
    """
    Get the args for the link anomaly detection task.

    Args:
        is_evaluation (bool, optional): Whether in the evaluation process. Defaults to False.
    """
    # Arguments
    parser = argparse.ArgumentParser(
        "DyGLib: Interface for the link anomaly detection task"
    )
    parser.add_argument(
        "--mlflow_tracking_uri",
        type=str,
        help="Where to store mlflow runs.",
        default=os.getcwd() + "/mlruns",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        help="Where to look for datasets.",
        required=True,
    )
    parser.add_argument(
        "--output_root",
        type=str,
        help="Where to save logs and models.",
        required=True,
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Dataset to be used",
        default="wikipedia",
        choices=[
            "synthetic",
            "wikipedia",
            "reddit",
            "mooc",
            "lastfm",
            "enron",
            "SocialEvo",
            "uci",
            "Flights",
            "CanParl",
            "USLegis",
            "UNtrade",
            "UNvote",
            "Contacts",
            "amazonreview",
            "stablecoin",
            "opensky",
            "redditcomments",
            "tgbl-wiki",
            "tgbl-review",
        ],
    )
    parser.add_argument(
        "--node_feat_dim",
        type=int,
        default=50,
        help="Dimension of node features after projection",
    )
    parser.add_argument(
        "--edge_feat_dim",
        type=int,
        default=50,
        help="Dimension of edge features after projection",
    )
    parser.add_argument(
        "--anom_type",
        type=str,
        help="Type of anomalies to inject",
        default=None,
        choices=[
            None,
            "temporal-structural-contextual",
            "structural-contextual",
            "temporal-contextual",
            "temporal",
            "contextual",
            "organic",
        ],
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="Name of the experiment",
        default="Experiment",
    )
    parser.add_argument("--batch_size", type=int, default=200, help="Batch size")
    parser.add_argument(
        "--model_name",
        type=str,
        default="DyGFormer",
        help="Name of the model, note that EdgeBank is only applicable for evaluation",
        choices=[
            "JODIE",
            "DyRep",
            "TGAT",
            "TGN",
            "CAWN",
            "EdgeBank",
            "TCL",
            "GraphMixer",
            "DyGFormer",
        ],
    )
    parser.add_argument("--gpu", type=int, default=0, help="ID of gpu to use")
    parser.add_argument(
        "--num_neighbors",
        type=int,
        default=20,
        help="Number of neighbors to sample for each node",
    )
    parser.add_argument(
        "--sample_neighbor_strategy",
        default="recent",
        choices=["uniform", "recent", "time_interval_aware"],
        help="How to sample historical neighbors",
    )
    parser.add_argument(
        "--time_scaling_factor",
        default=1e-6,
        type=float,
        help="The hyperparameter that controls the sampling preference with time interval, "
        "a large time_scaling_factor tends to sample more on recent links, 0.0 corresponds to uniform sampling, "
        "it works when sample_neighbor_strategy == time_interval_aware",
    )
    parser.add_argument(
        "--num_walk_heads",
        type=int,
        default=8,
        help="Number of heads used for the attention in walk encoder",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=2,
        help="Number of heads used in attention layer",
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="Number of model layers"
    )
    parser.add_argument(
        "--walk_length", type=int, default=1, help="Length of each random walk"
    )
    parser.add_argument(
        "--time_gap",
        type=int,
        default=2000,
        help="Time gap for neighbors to compute node features",
    )
    parser.add_argument(
        "--time_feat_dim", type=int, default=100, help="Dimension of the time embedding"
    )
    parser.add_argument(
        "--position_feat_dim",
        type=int,
        default=172,
        help="Dimension of the position embedding",
    )
    parser.add_argument(
        "--edge_bank_memory_mode",
        type=str,
        default="unlimited_memory",
        help="How memory of EdgeBank works",
        choices=["unlimited_memory", "time_window_memory", "repeat_threshold_memory"],
    )
    parser.add_argument(
        "--time_window_mode",
        type=str,
        default="fixed_proportion",
        help="How to select the time window size for time window memory",
        choices=["fixed_proportion", "repeat_interval"],
    )
    parser.add_argument("--patch_size", type=int, default=1, help="Patch size")
    parser.add_argument(
        "--channel_embedding_dim",
        type=int,
        default=50,
        help="Dimension of each channel embedding",
    )
    parser.add_argument(
        "--max_input_sequence_length",
        type=int,
        default=32,
        help="Maximal length of the input sequence of each node",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0001, help="Learning rate"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument(
        "--num_epochs", type=int, default=50, help="Number of epochs"
    )  # original value = 100
    parser.add_argument(
        "--optimizer",
        type=str,
        default="Adam",
        choices=["SGD", "Adam", "RMSprop"],
        help="Name of optimizer",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument(
        "--patience", type=int, default=20, help="Patience for early stopping"
    )  # original value = 20
    parser.add_argument(
        "--val_ratio", type=float, default=0.15, help="Ratio of validation set"
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.15, help="Ratio of test set"
    )
    parser.add_argument("--num_runs", type=int, default=1, help="Number of runs")
    parser.add_argument(
        "--num_anom_sets", type=int, default=1, help="Number of anomaly sets"
    )
    parser.add_argument(
        "--test_interval_epochs",
        type=int,
        default=10,
        help="How many epochs to perform testing once",
    )
    parser.add_argument(
        "--load_best_configs",
        action="store_true",
        default=False,
        help="Whether to load the best configurations",
    )
    parser.add_argument("--seed", type=int, default=2023, help="Random seed")
    parser.add_argument(
        "--show_progress_bar",
        action="store_true",
        default=False,
        help="Whether to show progress bar during training, validation and testing",
    )

    try:
        args = parser.parse_args()
        args.device = (
            f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
        )
    except:
        parser.print_help()
        sys.exit()

    if args.model_name == "EdgeBank":
        assert is_evaluation, "EdgeBank is only applicable for evaluation!"

    if args.load_best_configs:
        # The best configurations for link prediction task.
        # It might come handy at some point.
        load_link_prediction_best_configs(args=args)

    return args


def load_link_prediction_best_configs(args: argparse.Namespace):
    """
    Load the best configurations for the link prediction task.

    Args:
        args (argparse.Namespace): Arguments.
    """
    # Model specific settings
    if args.model_name == "TGAT":
        args.num_neighbors = 20
        args.num_layers = 2
        if args.dataset_name in ["enron", "CanParl", "UNvote"]:
            args.dropout = 0.2
        else:
            args.dropout = 0.1
        if args.dataset_name in ["reddit", "CanParl", "UNtrade"]:
            args.sample_neighbor_strategy = "uniform"
        else:
            args.sample_neighbor_strategy = "recent"
    elif args.model_name in ["JODIE", "DyRep", "TGN"]:
        args.num_neighbors = 10
        args.num_layers = 1
        if args.model_name == "JODIE":
            if args.dataset_name in ["mooc", "USLegis"]:
                args.dropout = 0.2
            elif args.dataset_name in ["lastfm"]:
                args.dropout = 0.3
            elif args.dataset_name in ["uci", "UNtrade"]:
                args.dropout = 0.4
            elif args.dataset_name in ["CanParl"]:
                args.dropout = 0.0
            else:
                args.dropout = 0.1
        elif args.model_name == "DyRep":
            if args.dataset_name in [
                "mooc",
                "lastfm",
                "enron",
                "uci",
                "CanParl",
                "USLegis",
                "Contacts",
            ]:
                args.dropout = 0.0
            else:
                args.dropout = 0.1
        else:
            assert args.model_name == "TGN"
            if args.dataset_name in ["mooc", "UNtrade"]:
                args.dropout = 0.2
            elif args.dataset_name in ["lastfm", "CanParl"]:
                args.dropout = 0.3
            elif args.dataset_name in ["enron", "SocialEvo"]:
                args.dropout = 0.0
            else:
                args.dropout = 0.1
        if args.model_name in ["TGN", "DyRep"]:
            if args.dataset_name in ["CanParl"] or (
                args.model_name == "TGN" and args.dataset_name == "UNvote"
            ):
                args.sample_neighbor_strategy = "uniform"
            else:
                args.sample_neighbor_strategy = "recent"
    elif args.model_name == "CAWN":
        args.time_scaling_factor = 1e-6
        if args.dataset_name in [
            "mooc",
            "SocialEvo",
            "uci",
            "Flights",
            "UNtrade",
            "UNvote",
            "Contacts",
        ]:
            args.num_neighbors = 64
        elif args.dataset_name in ["lastfm", "CanParl"]:
            args.num_neighbors = 128
        else:
            args.num_neighbors = 32
        if args.dataset_name in ["CanParl"]:
            args.dropout = 0.0
        else:
            args.dropout = 0.1
        args.sample_neighbor_strategy = "time_interval_aware"
    elif args.model_name == "EdgeBank":
        if args.negative_sample_strategy == "random":
            if args.dataset_name in ["wikipedia", "reddit", "uci", "Flights"]:
                args.edge_bank_memory_mode = "unlimited_memory"
            elif args.dataset_name in ["mooc", "lastfm", "enron", "CanParl", "USLegis"]:
                args.edge_bank_memory_mode = "time_window_memory"
                args.time_window_mode = "fixed_proportion"
            elif args.dataset_name in ["UNtrade", "UNvote", "Contacts"]:
                args.edge_bank_memory_mode = "time_window_memory"
                args.time_window_mode = "repeat_interval"
            else:
                assert args.dataset_name == "SocialEvo"
                args.edge_bank_memory_mode = "repeat_threshold_memory"
        elif args.negative_sample_strategy == "historical":
            if args.dataset_name in ["uci", "CanParl", "USLegis"]:
                args.edge_bank_memory_mode = "time_window_memory"
                args.time_window_mode = "fixed_proportion"
            elif args.dataset_name in [
                "mooc",
                "lastfm",
                "enron",
                "UNtrade",
                "UNvote",
                "Contacts",
            ]:
                args.edge_bank_memory_mode = "time_window_memory"
                args.time_window_mode = "repeat_interval"
            else:
                assert args.dataset_name in [
                    "wikipedia",
                    "reddit",
                    "SocialEvo",
                    "Flights",
                ]
                args.edge_bank_memory_mode = "repeat_threshold_memory"
        else:
            assert args.negative_sample_strategy == "inductive"
            if args.dataset_name in ["USLegis"]:
                args.edge_bank_memory_mode = "time_window_memory"
                args.time_window_mode = "fixed_proportion"
            elif args.dataset_name in ["uci", "UNvote"]:
                args.edge_bank_memory_mode = "time_window_memory"
                args.time_window_mode = "repeat_interval"
            else:
                assert args.dataset_name in [
                    "wikipedia",
                    "reddit",
                    "mooc",
                    "lastfm",
                    "enron",
                    "SocialEvo",
                    "Flights",
                    "CanParl",
                    "UNtrade",
                    "Contacts",
                ]
                args.edge_bank_memory_mode = "repeat_threshold_memory"
    elif args.model_name == "TCL":
        args.num_neighbors = 20
        args.num_layers = 2
        if args.dataset_name in ["SocialEvo", "uci", "UNtrade", "UNvote", "Contacts"]:
            args.dropout = 0.0
        elif args.dataset_name in ["CanParl"]:
            args.dropout = 0.2
        elif args.dataset_name in ["USLegis"]:
            args.dropout = 0.3
        else:
            args.dropout = 0.1
        if args.dataset_name in ["reddit", "CanParl", "USLegis", "UNtrade", "UNvote"]:
            args.sample_neighbor_strategy = "uniform"
        else:
            args.sample_neighbor_strategy = "recent"
    elif args.model_name == "GraphMixer":
        args.num_layers = 2
        if args.dataset_name in ["wikipedia"]:
            args.num_neighbors = 30
        elif args.dataset_name in ["reddit", "lastfm"]:
            args.num_neighbors = 10
        else:
            args.num_neighbors = 20
        if args.dataset_name in ["wikipedia", "reddit", "enron"]:
            args.dropout = 0.5
        elif args.dataset_name in ["mooc", "uci", "USLegis"]:
            args.dropout = 0.4
        elif args.dataset_name in ["lastfm", "UNvote"]:
            args.dropout = 0.0
        elif args.dataset_name in ["SocialEvo"]:
            args.dropout = 0.3
        elif args.dataset_name in ["Flights", "CanParl"]:
            args.dropout = 0.2
        else:
            args.dropout = 0.1
        if args.dataset_name in ["CanParl", "UNtrade", "UNvote"]:
            args.sample_neighbor_strategy = "uniform"
        else:
            args.sample_neighbor_strategy = "recent"
    elif args.model_name == "DyGFormer":
        args.num_layers = 2
        if args.dataset_name in ["reddit"]:
            args.max_input_sequence_length = 64
            args.patch_size = 2
        elif args.dataset_name in ["mooc", "enron", "Flights", "USLegis", "UNtrade"]:
            args.max_input_sequence_length = 256
            args.patch_size = 8
        elif args.dataset_name in ["lastfm"]:
            args.max_input_sequence_length = 512
            args.patch_size = 16
        elif args.dataset_name in ["CanParl"]:
            args.max_input_sequence_length = 2048
            args.patch_size = 64
        elif args.dataset_name in ["UNvote"]:
            args.max_input_sequence_length = 128
            args.patch_size = 4
        else:
            args.max_input_sequence_length = 32
            args.patch_size = 1
        assert args.max_input_sequence_length % args.patch_size == 0
        if args.dataset_name in ["reddit", "UNvote"]:
            args.dropout = 0.2
        elif args.dataset_name in ["enron", "USLegis", "UNtrade", "Contacts"]:
            args.dropout = 0.0
        else:
            args.dropout = 0.1
    else:
        raise ValueError(f"Wrong value for model_name {args.model_name}!")
