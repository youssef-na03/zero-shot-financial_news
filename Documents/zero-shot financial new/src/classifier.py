from transformers import pipeline
from src.config import MODEL_NAME, CANDIDATE_LABELS, MULTI_LABEL

classifier_pipeline = pipeline("zero-shot-classification", model=MODEL_NAME)

def classify_headline(headline: str):
    
    result = classifier_pipeline(headline, candidate_labels=CANDIDATE_LABELS, multi_label=MULTI_LABEL)
    return dict(zip(result["labels"], result["scores"]))


def classify_batch(headlines: list):
    return [classify_headline(h) for h in headlines]