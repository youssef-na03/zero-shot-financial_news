def compute_relevance_score(label_scores: dict, target_labels: list = None) -> float:
    
    if not label_scores:
        return 0.0

    if target_labels:
        relevant_scores = [label_scores[label] for label in target_labels if label in label_scores]
    else:
        relevant_scores = list(label_scores.values())

    return sum(relevant_scores) / len(relevant_scores) if relevant_scores else 0.0


def rank_headlines(headlines: list, predictions: list, target_labels: list = None, top_n: int = 10):
    """
    Rank headlines based on their relevance scores.

    Args:
        headlines (list): Original list of headlines.
        predictions (list): List of label-score dictionaries.
        target_labels (list): Labels to prioritize when computing relevance.
        top_n (int): Number of top headlines to return.

    Returns:
        List[Tuple[str, float]]: List of (headline, relevance score) tuples.
    """
    scored = [(headline, compute_relevance_score(pred, target_labels)) for headline, pred in zip(headlines, predictions)]
    ranked = sorted(scored, key=lambda x: x[1], reverse=True)
    return ranked[:top_n]
