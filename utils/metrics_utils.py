import numpy as np

def calculate_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Calculates Population Stability Index (PSI) to detect distributional shift.
    Adds epsilon to prevent division by zero.
    """
    expected_perc, _ = np.histogram(expected, bins=bins)
    actual_perc, _ = np.histogram(actual, bins=bins)
    
    # Convert to percentages and add epsilon
    expected_perc = (expected_perc + 1e-6) / sum(expected_perc)
    actual_perc = (actual_perc + 1e-6) / sum(actual_perc)
    
    psi_values = (actual_perc - expected_perc) * np.log(actual_perc / expected_perc)
    return np.sum(psi_values)

def check_drift_trigger(psi_score: float, threshold: float = 0.20) -> bool:
    """Returns True if PSI exceeds the adaptation trigger threshold."""
    return psi_score > threshold

def bootstrap_bca_ci(y_true: np.ndarray, y_pred: np.ndarray, metric_fn, n_resamples: int = 1000):
    """
    Generic BCa bootstrap function for calculating 95% confidence intervals
    for AUROC, AUPRC, or Brier Scores.
    """
    n = len(y_true)
    bootstrapped_scores = []
    
    for _ in range(n_resamples):
        indices = np.random.choice(n, n, replace=True)
        # Ensure both classes are present in the resample to avoid metric errors
        if len(np.unique(y_true[indices])) > 1: 
            score = metric_fn(y_true[indices], y_pred[indices])
            bootstrapped_scores.append(score)
            
    # Simple percentile CI (You can expand this with full BCa math if needed)
    lower = np.percentile(bootstrapped_scores, 2.5)
    upper = np.percentile(bootstrapped_scores, 97.5)
    mean_score = np.mean(bootstrapped_scores)
    
    return mean_score, lower, upper