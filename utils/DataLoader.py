from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import pandas as pd


# TGB imports
from tgb.linkanomdet.dataset_pyg import PyGLinkAnomDetDataset


class CustomizedDataset(Dataset):
    def __init__(self, indices_list: list):
        """
        Customized dataset.
        :param indices_list: list, list of indices
        """
        super(CustomizedDataset, self).__init__()

        self.indices_list = indices_list

    def __getitem__(self, idx: int):
        """
        get item at the index in self.indices_list
        :param idx: int, the index
        :return:
        """
        return self.indices_list[idx]

    def __len__(self):
        return len(self.indices_list)


def get_idx_data_loader(indices_list: list, batch_size: int, shuffle: bool):
    """
    get data loader that iterates over indices
    :param indices_list: list, list of indices
    :param batch_size: int, batch size
    :param shuffle: boolean, whether to shuffle the data
    :return: data_loader, DataLoader
    """
    dataset = CustomizedDataset(indices_list=indices_list)

    data_loader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False
    )
    return data_loader


class Data:
    def __init__(
        self,
        src_node_ids: np.ndarray,
        dst_node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        edge_ids: np.ndarray,
        labels: np.ndarray,
    ):
        """
        Data object to store the nodes interaction information.
        :param src_node_ids: ndarray
        :param dst_node_ids: ndarray
        :param node_interact_times: ndarray
        :param edge_ids: ndarray
        :param labels: ndarray
        """
        self.src_node_ids = src_node_ids
        self.dst_node_ids = dst_node_ids
        self.node_interact_times = node_interact_times
        self.edge_ids = edge_ids
        self.labels = labels
        self.num_interactions = len(src_node_ids)
        self.unique_node_ids = set(src_node_ids) | set(dst_node_ids)
        self.num_unique_nodes = len(self.unique_node_ids)


class EmbeddingWithProjection(torch.nn.Module):
    def __init__(
        self,
        embedding: np.ndarray,
        embedding_dim: int,
        device: Optional[str] = "cpu",
    ):
        """
        Embedding layer that additionally performs also projection.

        Args:
            embedding (np.ndarray): Embeddings.
            embedding_dim (int): Embedding dimensionality after projection.
            device (Optional[str], optional): Device to put embedding on. Defaults to "cpu".
        """
        super(EmbeddingWithProjection, self).__init__()

        self.embedding = torch.from_numpy(embedding.astype(np.float32)).to(device)
        self.num_embeddings = self.embedding.shape[0]
        self.embedding_dim = embedding_dim
        self.device = device

        self.linear = torch.nn.Linear(
            in_features=embedding.shape[1],
            out_features=self.embedding_dim,
        )

    def forward(self, idx: torch.Tensor):
        x = self.embedding[idx]
        x = self.linear(x)
        return x

    def __getitem__(self, idx: torch.Tensor):
        return self.forward(idx)


def get_link_prediction_data(dataset_name: str, val_ratio: float, test_ratio: float):
    """
    generate data for link prediction task (inductive & transductive settings)
    :param dataset_name: str, dataset name
    :param val_ratio: float, validation data ratio
    :param test_ratio: float, test data ratio
    :return: node_raw_features, edge_raw_features, (np.ndarray),
            full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data, (Data object)
    """
    # Load data and train val test split
    graph_df = pd.read_csv(
        "./processed_data/{}/ml_{}.csv".format(dataset_name, dataset_name)
    )
    edge_raw_features = np.load(
        "./processed_data/{}/ml_{}.npy".format(dataset_name, dataset_name)
    )
    node_raw_features = np.load(
        "./processed_data/{}/ml_{}_node.npy".format(dataset_name, dataset_name)
    )

    NODE_FEAT_DIM = EDGE_FEAT_DIM = 172
    assert (
        NODE_FEAT_DIM >= node_raw_features.shape[1]
    ), f"Node feature dimension in dataset {dataset_name} is bigger than {NODE_FEAT_DIM}!"
    assert (
        EDGE_FEAT_DIM >= edge_raw_features.shape[1]
    ), f"Edge feature dimension in dataset {dataset_name} is bigger than {EDGE_FEAT_DIM}!"
    # padding the features of edges and nodes to the same dimension (172 for all the datasets)
    if node_raw_features.shape[1] < NODE_FEAT_DIM:
        node_zero_padding = np.zeros(
            (node_raw_features.shape[0], 172 - node_raw_features.shape[1])
        )
        node_raw_features = np.concatenate(
            [node_raw_features, node_zero_padding], axis=1
        )
    if edge_raw_features.shape[1] < EDGE_FEAT_DIM:
        edge_zero_padding = np.zeros(
            (edge_raw_features.shape[0], 172 - edge_raw_features.shape[1])
        )
        edge_raw_features = np.concatenate(
            [edge_raw_features, edge_zero_padding], axis=1
        )

    assert (
        NODE_FEAT_DIM == node_raw_features.shape[1]
        and EDGE_FEAT_DIM == edge_raw_features.shape[1]
    ), "Unaligned feature dimensions after feature padding!"

    # get the timestamp of validate and test set
    val_time, test_time = list(
        np.quantile(graph_df.ts, [(1 - val_ratio - test_ratio), (1 - test_ratio)])
    )

    src_node_ids = graph_df.u.values.astype(np.longlong)
    dst_node_ids = graph_df.i.values.astype(np.longlong)
    node_interact_times = graph_df.ts.values.astype(np.float64)
    edge_ids = graph_df.idx.values.astype(np.longlong)
    labels = graph_df.label.values

    full_data = Data(
        src_node_ids=src_node_ids,
        dst_node_ids=dst_node_ids,
        node_interact_times=node_interact_times,
        edge_ids=edge_ids,
        labels=labels,
    )

    # the setting of seed follows previous works
    random.seed(2020)

    # union to get node set
    node_set = set(src_node_ids) | set(dst_node_ids)
    num_total_unique_node_ids = len(node_set)

    # compute nodes which appear at test time
    test_node_set = set(src_node_ids[node_interact_times > val_time]).union(
        set(dst_node_ids[node_interact_times > val_time])
    )
    # sample nodes which we keep as new nodes (to test inductiveness), so then we have to remove all their edges from training
    new_test_node_set = set(
        random.sample(test_node_set, int(0.1 * num_total_unique_node_ids))
    )

    # mask for each source and destination to denote whether they are new test nodes
    new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
    new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values

    # mask, which is true for edges with both destination and source not being new test nodes (because we want to remove all edges involving any new test node)
    observed_edges_mask = np.logical_and(
        ~new_test_source_mask, ~new_test_destination_mask
    )

    # for train data, we keep edges happening before the validation time which do not involve any new node, used for inductiveness
    train_mask = np.logical_and(node_interact_times <= val_time, observed_edges_mask)

    train_data = Data(
        src_node_ids=src_node_ids[train_mask],
        dst_node_ids=dst_node_ids[train_mask],
        node_interact_times=node_interact_times[train_mask],
        edge_ids=edge_ids[train_mask],
        labels=labels[train_mask],
    )

    # define the new nodes sets for testing inductiveness of the model
    train_node_set = set(train_data.src_node_ids).union(train_data.dst_node_ids)
    assert len(train_node_set & new_test_node_set) == 0
    # new nodes that are not in the training set
    new_node_set = node_set - train_node_set

    val_mask = np.logical_and(
        node_interact_times <= test_time, node_interact_times > val_time
    )
    test_mask = node_interact_times > test_time

    # new edges with new nodes in the val and test set (for inductive evaluation)
    edge_contains_new_node_mask = np.array(
        [
            (src_node_id in new_node_set or dst_node_id in new_node_set)
            for src_node_id, dst_node_id in zip(src_node_ids, dst_node_ids)
        ]
    )
    new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
    new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

    # validation and test data
    val_data = Data(
        src_node_ids=src_node_ids[val_mask],
        dst_node_ids=dst_node_ids[val_mask],
        node_interact_times=node_interact_times[val_mask],
        edge_ids=edge_ids[val_mask],
        labels=labels[val_mask],
    )

    test_data = Data(
        src_node_ids=src_node_ids[test_mask],
        dst_node_ids=dst_node_ids[test_mask],
        node_interact_times=node_interact_times[test_mask],
        edge_ids=edge_ids[test_mask],
        labels=labels[test_mask],
    )

    # validation and test with edges that at least has one new node (not in training set)
    new_node_val_data = Data(
        src_node_ids=src_node_ids[new_node_val_mask],
        dst_node_ids=dst_node_ids[new_node_val_mask],
        node_interact_times=node_interact_times[new_node_val_mask],
        edge_ids=edge_ids[new_node_val_mask],
        labels=labels[new_node_val_mask],
    )

    new_node_test_data = Data(
        src_node_ids=src_node_ids[new_node_test_mask],
        dst_node_ids=dst_node_ids[new_node_test_mask],
        node_interact_times=node_interact_times[new_node_test_mask],
        edge_ids=edge_ids[new_node_test_mask],
        labels=labels[new_node_test_mask],
    )

    print(
        "The dataset has {} interactions, involving {} different nodes".format(
            full_data.num_interactions, full_data.num_unique_nodes
        )
    )
    print(
        "The training dataset has {} interactions, involving {} different nodes".format(
            train_data.num_interactions, train_data.num_unique_nodes
        )
    )
    print(
        "The validation dataset has {} interactions, involving {} different nodes".format(
            val_data.num_interactions, val_data.num_unique_nodes
        )
    )
    print(
        "The test dataset has {} interactions, involving {} different nodes".format(
            test_data.num_interactions, test_data.num_unique_nodes
        )
    )
    print(
        "The new node validation dataset has {} interactions, involving {} different nodes".format(
            new_node_val_data.num_interactions, new_node_val_data.num_unique_nodes
        )
    )
    print(
        "The new node test dataset has {} interactions, involving {} different nodes".format(
            new_node_test_data.num_interactions, new_node_test_data.num_unique_nodes
        )
    )
    print(
        "{} nodes were used for the inductive testing, i.e. are never seen during training".format(
            len(new_test_node_set)
        )
    )

    return (
        node_raw_features,
        edge_raw_features,
        full_data,
        train_data,
        val_data,
        test_data,
        new_node_val_data,
        new_node_test_data,
    )


def get_link_anom_det_data_TRANS_TGB(
    dataset_name: str,
    dataset_root: Optional[str] = "datasets",
    absolute_path: Optional[bool] = False,
    val_ratio: Optional[float] = 0.15,
    test_ratio: Optional[float] = 0.15,
    node_feat_dim: Optional[int] = 50,
    edge_feat_dim: Optional[int] = 50,
    val_anom_type: Optional[str] = None,
    test_anom_type: Optional[str] = None,
    anom_set_id: Optional[int] = 0,
    device: Optional[str] = "cpu",
) -> Tuple[EmbeddingWithProjection, EmbeddingWithProjection, Data, Data, Data, Data]:
    """
    Generate data for link anomaly detection task.
    Load the data with the help of TGB and generate required format for DyGLib.

    Args:
        dataset_name (str): Dataset name.
        val_ratio (Optional[float], optional): Ratio of validation data. Defaults to 0.15.
        test_ratio (Optional[float], optional): Ratio of test data. Defaults to 0.15.
        node_feat_dim (Optional[int], optional): Dimension of node features after projection. Defaults to 50.
        edge_feat_dim (Optional[int], optional): Dimension of edge features after projection. Defaults to 50.
        val_anom_type (Optional[str], optional): Type of anomalies to inject in validation set. Defaults to None.
        test_anom_type (Optional[str], optional): Type of anomalies to inject in test set. Defaults to None.
        anom_set_id (Optional[int], optional): ID of anomaly set (multiple sets of anomalies are supported).
                Defaults to 0.
        device (Optional[str], optional): Device to put embedding on. Defaults to "cpu".

    Returns:
        Tuple[EmbeddingWithProjection, EmbeddingWithProjection, Data, Data, Data, Data]:
            Raw node features with projection layer, raw edge features with projectin layer,
            full data, training data, validation data and test data.
    """
    # Data loading
    dataset = PyGLinkAnomDetDataset(
        name=dataset_name,
        root=dataset_root,
        absolute_path=absolute_path,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    dataset.load_anomalies(
        val_anom_type=val_anom_type,
        test_anom_type=test_anom_type,
        anom_set_id=anom_set_id,
    )
    data = dataset.get_TemporalData()
    # Get split masks.
    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask
    # Get split data.
    train_data = data[train_mask]
    val_data = data[val_mask]
    test_data = data[test_mask]

    # Load data and train/val/test split.
    edge_raw_features = data.msg.numpy()
    node_raw_features = np.zeros((data.dst.size(0), node_feat_dim))

    src_node_ids = (data.src.numpy() + 1).astype(np.longlong)
    dst_node_ids = (data.dst.numpy() + 1).astype(np.longlong)
    node_interact_times = data.t.numpy().astype(np.float64)
    edge_ids = np.array([i for i in range(1, len(data.src) + 1)]).astype(np.longlong)
    labels = data.y.numpy()

    # Since src_node_ids, dst_node_ids and edge_ids are increased by 1,
    # pad node_raw_features and edge_raw_features with a dummy vector.
    node_zero_padding = np.zeros((1, node_raw_features.shape[1]))
    node_raw_features = np.concatenate([node_zero_padding, node_raw_features], axis=0)

    edge_zero_padding = np.zeros((1, edge_raw_features.shape[1]))
    edge_raw_features = np.concatenate([edge_zero_padding, edge_raw_features], axis=0)

    # Node and edge features should be down-projected when retrieved.
    node_raw_features = EmbeddingWithProjection(
        embedding=node_raw_features,
        embedding_dim=node_feat_dim,
        device=device,
    )
    edge_raw_features = EmbeddingWithProjection(
        embedding=edge_raw_features,
        embedding_dim=edge_feat_dim,
        device=device,
    )

    full_data = Data(
        src_node_ids=src_node_ids,
        dst_node_ids=dst_node_ids,
        node_interact_times=node_interact_times,
        edge_ids=edge_ids,
        labels=labels,
    )

    train_data = Data(
        src_node_ids=src_node_ids[train_mask],
        dst_node_ids=dst_node_ids[train_mask],
        node_interact_times=node_interact_times[train_mask],
        edge_ids=edge_ids[train_mask],
        labels=labels[train_mask],
    )

    # Validation and test data
    val_data = Data(
        src_node_ids=src_node_ids[val_mask],
        dst_node_ids=dst_node_ids[val_mask],
        node_interact_times=node_interact_times[val_mask],
        edge_ids=edge_ids[val_mask],
        labels=labels[val_mask],
    )

    test_data = Data(
        src_node_ids=src_node_ids[test_mask],
        dst_node_ids=dst_node_ids[test_mask],
        node_interact_times=node_interact_times[test_mask],
        edge_ids=edge_ids[test_mask],
        labels=labels[test_mask],
    )

    print(
        "INFO: The dataset has {} interactions, involving {} different nodes".format(
            full_data.num_interactions, full_data.num_unique_nodes
        )
    )
    print(
        "INFO: The training dataset has {} interactions, involving {} different nodes".format(
            train_data.num_interactions, train_data.num_unique_nodes
        )
    )
    print(
        "INFO: The validation dataset has {} interactions, involving {} different nodes".format(
            val_data.num_interactions, val_data.num_unique_nodes
        )
    )
    print(
        "INFO: The test dataset has {} interactions, involving {} different nodes".format(
            test_data.num_interactions, test_data.num_unique_nodes
        )
    )

    return (
        node_raw_features,
        edge_raw_features,
        full_data,
        train_data,
        val_data,
        test_data,
        dataset,
    )
