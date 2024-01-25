import torch
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def get_link_prediction_metrics(
    predicts: torch.Tensor,
    labels: torch.Tensor,
    structural_mask: torch.Tensor,
    contextual_mask: torch.Tensor,
    temporal_mask: torch.Tensor,
):
    """
    get metrics for the link prediction task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :param structural_mask: Tensor, shape (num_samples, )
    :param contextual_mask: Tensor, shape (num_samples, )
    :param temporal_mask: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()
    structural_mask = structural_mask.cpu().numpy()
    contextual_mask = contextual_mask.cpu().numpy()
    temporal_mask = temporal_mask.cpu().numpy()

    average_precision = average_precision_score(y_true=labels, y_score=predicts)
    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

    structural_roc_auc = (
        roc_auc_score(y_true=labels[structural_mask], y_score=predicts[structural_mask])
        if not np.all(structural_mask == False)
        else 0.50
    )
    contextual_roc_auc = (
        roc_auc_score(y_true=labels[contextual_mask], y_score=predicts[contextual_mask])
        if not np.all(contextual_mask == False)
        else 0.50
    )

    temporal_roc_auc = (
        roc_auc_score(y_true=labels[temporal_mask], y_score=predicts[temporal_mask])
        if not np.all(temporal_mask == False)
        else 0.50
    )

    return {
        "average_precision": average_precision,
        "roc_auc": roc_auc,
        "structural_roc_auc": structural_roc_auc,
        "contextual_roc_auc": contextual_roc_auc,
        "temporal_roc_auc": temporal_roc_auc,
    }
