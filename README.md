# HDB Resale Price Prediction – Graph & Model Analysis

This repository contains code for analyzing and predicting HDB resale prices in Singapore using XGBoost for resale price per sqm prediction and a graph-based approach (GraphSAGE) for GNN-based price prediction.
## Files

- `hdb_resale_model_prediction.ipynb` – Preprocesses the training data and trains an XGBoost model to predict resale price per square meter.
  - **GPU:** Not required (CPU is sufficient).
- `hdb_resale_link_analysis.ipynb` – Builds the HDB resale graph and trains GraphSAGE embeddings for GNN-based price prediction.  
  - **Training** requires a GPU.  
  - **Inference** can run on CPU.

---

## Installation

1. Clone the repository and navigate into the folder.  
```bash
git clone <your-repo-url>
cd <repository-folder>
```
2. (Optional) Create and activate a virtual environment for the project.  
```bash
python -m venv .venv
# For Linux/macOS
source .venv/bin/activate
# For Windows
.venv\Scripts\activate
```
3. Install required packages using the `requirements.txt` file.  
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
This will install all Python dependencies needed to run both notebooks.

---

## How to Use

### 1. Model Prediction Notebook (`hdb_resale_model_prediction.ipynb`)
- **Purpose:** Performs exploratory data analysis (EDA) on the resale flat price dataset, prepares the final training dataset, and trains an XGBoost model for HDB resale price prediction.
- **GPU Usage:** Not required; CPU inference is sufficient.
- **You WILL need to run this notebook to create the data csv required before running the link analysis notebook**


### 2. Link Analysis Notebook (`hdb_resale_link_analysis.ipynb`)
- **Purpose:** Build the graph, train GraphSAGE embeddings, and evaluate models.  
- **GPU Usage:** Required for training embeddings.  
- **Inference:** Can be run on CPU using the Mapper + MLP pipeline.  

---

## Notes

- Training GraphSAGE embeddings requires a GPU.  
- Once embeddings are trained, inference can be performed entirely on CPU using the Mapper + MLP pipeline.  
- Ensure that any virtual environment is activated before running the notebooks.
- Make sure to run the Model Prediction Notebook first
