## 🖥️ Environment Setup

This project was developed and tested on:

- **MacBook Pro (M3 Pro Chip)**
- **macOS Sonoma 14.x**
- **Python 3.10+**
- No GPU or neural engine acceleration is used—only **CPUExecutionProvider** via ONNX Runtime.

You can set up the project environment using either `venv` or `conda`.

## 🔧 Option 1: Using Python's `venv` (Virtual Environment)

### Step 1: Clone the repository
git clone https://github.com/yourusername/quantized-finance-ml.git
cd quantized-finance-ml

### Step 2: Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        
- For macOS/Linux

.\venv\Scripts\activate        
- For Windows

### Step 3: Install dependencies
pip install -r requirements.txt


## Option 2: Using Conda
conda create -n quantized-ml python=3.10
conda activate quantized-ml
pip install -r requirements.txt

## Dependencies
You can install the dependencies via the included requirements.txt file, or manually with:
* pip install yfinance pandas numpy matplotlib scikit-learn onnx onnxruntime skl2onnx

## How To Run the Code
Each Python script is self-contained and follows a full pipeline:
1. Downloads historical stock data using yfinance
2. Performs feature engineering and preprocessing
3. Trains a classification model (Logistic Regression, SVM, or Random Forest)
4. Converts the model to ONNX format
5. Applies dynamic post-training quantization to INT8 precision
6. Evaluates and benchmarks model performance (accuracy, latency, size)
7. Saves output visualizations and metrics to file

### Run Logistic Regression Pipeline:
* python quantizedLogisticRegression.py
### Run Support Vector Machine Pipeline:
* python quantizedSvm.py
### Run Random Forest Pipeline:
* python quantizedRandomForest.py

## Outputs
Each script generates and displays the following:

1. Classification accuracy (FP32 vs INT8)
2. Average inference latency (milliseconds)
3. ONNX model file sizes
4. Confusion matrices
5. Actual vs Predicted plots
6. Rolling accuracy plots
7. Probability score visualizations
8. Cross-ticker performance bar charts
9. Output files will be saved in the working directory as .png charts and console logs.

## Notes
The project is configured to run on CPU-only machines using ONNX’s CPUExecutionProvider.

Tickers analyzed: AAPL, MSFT, SPY. You can modify the list in each script by changing the tickers = [...] variable.

Scripts are designed for easy reproducibility and can be reused or adapted for other time series datasets.

