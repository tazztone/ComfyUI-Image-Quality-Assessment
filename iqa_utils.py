import numpy as np

def aggregate_scores(scores, method="mean"):
    """
    Aggregates a list of scores using the specified method.

    Args:
        scores (list or np.ndarray): List of numerical scores.
        method (str): Aggregation method. One of ["mean", "min", "max", "first"].

    Returns:
        float: The aggregated score.
    """
    if not scores:
        return 0.0

    # Ensure scores is a list or numpy array
    if not isinstance(scores, (list, np.ndarray, tuple)):
        try:
            scores = [float(scores)]
        except:
            return 0.0

    if method == "mean":
        return float(np.mean(scores))
    elif method == "min":
        return float(np.min(scores))
    elif method == "max":
        return float(np.max(scores))
    elif method == "first":
        return float(scores[0])

    # Default to mean
    return float(np.mean(scores))
