"""Different losses used in network training."""

import torch
import torch.nn.functional as F
from torch.nn import CosineSimilarity


def noise_loss(
    add_noise: torch.Tensor, mult_noise: torch.Tensor
) -> torch.Tensor:
    """Calculates the constraining loss for noise matrices."""
    add_noise_regul = torch.zeros_like(add_noise)
    mult_noise_regul = torch.ones_like(mult_noise)
    add_noise_loss = F.mse_loss(add_noise, add_noise_regul)
    mult_noise_loss = F.mse_loss(mult_noise, mult_noise_regul)

    return add_noise_loss + mult_noise_loss


def pseudo_loss(
    normal: torch.Tensor, anomalies: torch.Tensor, cosine_similarity: torch.nn.CosineSimilarity
) -> torch.Tensor:
    return (cosine_similarity(normal, anomalies[None, :, :]) - 1.0).square().mean()


def classif_loss(
    classif_normal: torch.Tensor,
    classif_anomalies: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Calculates the classification loss (normal vs. anomalies)."""
    loss_normal = F.binary_cross_entropy_with_logits(classif_normal, targets)
    loss_anomalies = F.binary_cross_entropy_with_logits(
        classif_anomalies, 0 * targets
    )

    return loss_normal + loss_anomalies


def kl_divergence(z_mean: torch.Tensor, z_logvar: torch.Tensor) -> torch.Tensor:
    """Calculates divergence between the embedded parameters and a gaussian."""
    kl_divergence_ = (
        0.5
        * torch.sum(torch.exp(z_logvar) + torch.pow(z_mean, 2) - 1.0 - z_logvar)
        / z_mean.size(0)
    )

    return kl_divergence_


def _make_index_shift(batch_size: int) -> torch.Tensor:
    return (
        torch.arange(batch_size * batch_size).reshape(batch_size, batch_size)
        + torch.arange(batch_size)
        .reshape(batch_size, 1)
        .expand(batch_size, batch_size)
    ) % batch_size


def _cossim(
    set1: torch.Tensor, set2: torch.Tensor, cosine_similarity: CosineSimilarity
) -> torch.Tensor:
    batch_size = set1.shape[0]
    shift_indexes = _make_index_shift(batch_size)

    return cosine_similarity(
        set1[shift_indexes], set2[None, :, :].repeat(batch_size, 1, 1)
    )

def _mean_cossim(
    set1: torch.Tensor, set2: torch.Tensor, cosine_similarity: CosineSimilarity
) -> torch.Tensor:
    return _cossim(set1, set2, cosine_similarity)[1:, :].mean()


def cossim_constraint_loss(
    normal_features: torch.Tensor,
    anomalies_features: torch.Tensor,
    cosine_similarity: CosineSimilarity,
) -> torch.Tensor:
    """Calculates the average complement of the tensors' cosine similarity."""
    return 1.0 - _mean_cossim(
        normal_features, anomalies_features, cosine_similarity
    )


def cossim_coercion_loss(
    normal_features: torch.Tensor, cosine_similarity: CosineSimilarity
) -> torch.Tensor:
    """Calculates the average complement of the tensors' intra cosine similarity."""
    return 1.0 - _mean_cossim(
        normal_features, normal_features, cosine_similarity
    )


def contrastive_loss(
    normal_features: torch.Tensor,
    anomalies_features: torch.Tensor,
    cosine_similarity: CosineSimilarity,
    tau: float,
) -> torch.Tensor:
    """Calculates a contrastive loss."""

    normal_similarities = _cossim(
        normal_features, normal_features, cosine_similarity
    )[1:, :].div(tau)

    anomalies_similarities = _cossim(
        normal_features, anomalies_features, cosine_similarity
    ).div(tau)
    
    contrastive_term = (
        anomalies_similarities.exp().sum(dim=0)
        + normal_similarities.exp().sum(dim=0)
    ).log()
        
    return contrastive_term.mean() - normal_similarities.mean()


def mean_contrastive_loss(
    normal_features: torch.Tensor,
    anomalies_features: torch.Tensor,
    cosine_similarity: CosineSimilarity,
    tau: float,
) -> torch.Tensor:
    """Calculates a contrastive loss w.r.t. the mean vectors."""

    mean_normal_features = normal_features.mean(0)
    mean_anomalies_features = anomalies_features.mean(0)

    anomalies_similarities: torch.Tensor = cosine_similarity(
        normal_features,
        mean_anomalies_features[None, None, :],
    )
    contrastive_term = anomalies_similarities.exp().sum(dim=0).log().mean()

    return contrastive_term - cosine_similarity(
        normal_features,
        mean_normal_features[None, None, :],
    ).mean()
