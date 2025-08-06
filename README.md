# Zero-Shot Financial News Classifier


> A Python project that classifies **financial news headlines** using a **zero-shot learning** approach with the `facebook/bart-large-mnli` model.

---

## 📌 Features
- **Zero-shot classification** for financial topics — no labeled training data required.
- **Multi-label support** for more flexible categorization.
- **Easily configurable** candidate labels via `src/config.py`.
- Modular and extensible codebase.

---


![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Framework](https://img.shields.io/badge/transformers-HuggingFace-orange)
![Model](https://img.shields.io/badge/model-BART--MNLI-yellow)
![Status](https://img.shields.io/badge/status-active-success)

## 🛠️ Setup & Installation

Follow these steps to get the project up and running on your local machine.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```

2.  **Install dependencies:**
    Make sure you have Python 3.8+ installed.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Data**:
    Place your dataset (e.g., `cnbc_headlines.csv`) inside the `data/` directory.

## 🚀 Usage

To run the classification pipeline, execute the main script from the root directory:

```bash
python src/main.py --input_csv "data/cnbc_headlines.csv" --text_column "Headlin
es"
```

The script will load the data, run the classifier, and output the evaluation scores.

## ⚙️ Setup

### 1. Clone & Install Dependencies
git clone https://github.com/yourusername/zero-shot-financial-news-classifier.git
cd zero-shot-financial-news-classifier
pip install -r requirements.txt


## 📂 Project Structure

Here is an overview of the project's file structure:

```
.
├── data/
│   └── cnbc_headlines.csv
├── src/
│   ├── config.py         # Configuration for model and labels
│   ├── data_loader.py    # Data loading and preprocessing
│   ├── classifier.py     # Core zero-shot classification logic
│   ├── scorer.py         # Evaluation metrics
│   └── main.py           # Main script to run the pipeline
└── requirements.txt
```

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
