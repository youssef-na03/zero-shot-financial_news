from data_loader import load_headlines_from_csv
from classifier import classify_batch
from scorer import rank_headlines
from config import CANDIDATE_LABELS

def run_pipeline(input_csv: str, text_column: str = "headline", top_n: int = 10):
    headlines = load_headlines_from_csv(input_csv, text_column)
    print(f"Loaded {len(headlines)} headlines.")

    predictions = classify_batch(headlines)
    print("Classification complete.")

    ranked = rank_headlines(headlines, predictions, target_labels=CANDIDATE_LABELS, top_n=top_n)

    print(f"\nTop {top_n} Financially Relevant Headlines:\n")
    for i, (headline, score) in enumerate(ranked, 1):
        print(f"{i:2d}. [{score:.3f}] {headline}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Zero-Shot Classification of Financial News Headlines")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to CSV file with financial news")
    parser.add_argument("--text_column", type=str, default="headline", help="Name of the column containing headlines")
    parser.add_argument("--top_n", type=int, default=10, help="Number of top-ranked headlines to show")

    args = parser.parse_args()
    run_pipeline(args.input_csv, args.text_column, args.top_n)
