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

### 1. Clone the repository and navigate into the folder.  
```bash
git clone https://github.com/dacrylic/hdb_resale.git
cd hdb_resale
```
### 2. (Optional) Create and activate a virtual environment for the project.  
```bash
python -m venv .venv
# For Linux/macOS
source .venv/bin/activate
# For Windows
.venv\Scripts\activate
```
### 3. Install required packages using the `requirements.txt` file.  

### 3.1 Upgrade pip

Run:
```bash
pip install --upgrade pip
```
### 3.2 Install dependencies

Run:
```bash
pip install -r requirements.txt
```
---

### Important Notes

#### Handling CUDA packages and the +cu118 suffix

- The CUDA-enabled PyTorch wheels (e.g., `torch==2.7.1+cu118`) require specifying PyTorch’s special CUDA wheel index.
- To ensure package resolvers like `uv` or pip find these packages correctly, **your `requirements.txt` should start with:**
```bash
  --extra-index-url https://download.pytorch.org/whl/cu118
```
- This tells pip/uv where to find the CUDA builds. This has already been done in our requirements.txt.

---

#### If you are using `uv` (or other resolvers) and encounter version resolution issues:

Try installing CUDA-enabled PyTorch packages separately before running `uv` on the rest:
```bash
pip install torch==2.7.1+cu118 torchaudio==2.7.1+cu118 torchvision==0.22.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

uv pip install -r requirements.txt --no-deps
```
Or ensure the `--extra-index-url` line is present at the top of `requirements.txt`.

---

### macOS Users

- CUDA-enabled PyTorch packages are **not compatible** with macOS because CUDA is not supported on Mac.

- After installing dependencies as above, run the following commands to replace CUDA builds with CPU-only versions:
```bash
pip uninstall torch torchaudio torchvision

pip install torch==2.7.1 torchaudio torchvision
```
- **Note:**  
  `torch_geometric` does not provide pre-built CPU-only wheels for macOS. macOS users may need to build it from source or use CPU-only features without GPU acceleration.

---

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
