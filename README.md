# HDB Resale Price Prediction – Graph & Model Analysis

This repository contains code for analyzing and predicting HDB resale prices in Singapore using a **graph-based approach (GraphSAGE)** and downstream models.

## Files

- `hdb_resale_model_prediction.ipynb` – Uses precomputed GraphSAGE embeddings with downstream models (XGBoost, MLP) to predict resale price per sqm.  
  - **GPU is not required** for this notebook.

- `hdb_resale_link_analysis.ipynb` – Builds the HDB resale graph and trains GraphSAGE embeddings.  
  - **Training** requires a GPU.  
  - **Inference** can run on CPU.

---

## Installation

1. Clone the repository and navigate into the folder.  

2. (Optional) Create and activate a virtual environment for the project.  

3. Install required packages using the `requirements.txt` file.  

> This will install all Python dependencies needed to run both notebooks.

---

## How to Use

### 1. Model Prediction Notebook (`hdb_resale_model_prediction.ipynb`)
- **Purpose:** Load precomputed GraphSAGE embeddings and run downstream models (XGBoost, MLP) for predictions.  
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
