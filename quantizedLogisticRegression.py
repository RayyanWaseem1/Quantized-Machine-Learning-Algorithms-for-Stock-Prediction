import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import os

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

!pip install onnx
!pip install onnxruntime
!pip install skl2onnx
import onnx
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from onnxruntime.quantization import quantize_dynamic, QuantType

"""Downloading Historical Data"""

def download_data(ticker = 'AAPL', period = '2y'):
  df = yf.download(ticker, period = period)
  df.dropna(inplace=True)
  return df

"""Feature Engineering"""

def features(df):
  df['Return'] = df['Close'].pct_change()
  df['MA5'] = df['Close'].rolling(window = 5).mean()
  df['MA10'] = df['Close'].rolling(window = 10).mean()
  df['MA20'] = df['Close'].rolling(window = 20).mean()
  df['Volatility'] = df['Close'].rolling(window = 5).std()
  df['Target'] = (df['Return'].shift(-1) > 0).astype(int)

  df.dropna(inplace = True)

  if len(df) == 0:
    raise ValueError("No data remains after feature engineering")

  features = ['Close', 'MA5', 'MA10', 'MA20', 'Volatility']
  return df[features], df['Target'], df.index

"""Data Preparation"""

def data_preparation(X, y, indices):
  if len(X) == 0:
    raise ValueError("Empty feature matrix")

  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)

  X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
      X_scaled, y, np.arange(len(indices)), test_size=0.2, random_state=42)

  #Get actual dates from original indices
  test_date_indices = [indices[i] for i in idx_test]
  train_date_indices = [indices[i] for i in idx_train]

  return (X_train, X_test, y_train, y_test, train_date_indices, test_date_indices), scaler

"""Logistic Regression Model Training"""

def train_logistic_regression(X_train, y_train):
  model = LogisticRegression(max_iter = 1000)
  model.fit(X_train, y_train)
  return model

"""Logistic Regression Export to ONNX

"""

def export_to_onnx(model, X_sample, filename = 'model.onnx'):
  initial_type = [('float_input', FloatTensorType([None, X_sample.shape[1]]))]
  onnx_model = convert_sklearn(model, initial_types=initial_type)
  with open(filename, 'wb') as f:
    f.write(onnx_model.SerializeToString())
  return filename

"""Quantize ONNX"""

def quantize_logistic_regression(input_onnx_path, output_onnx_path):
  quantize_dynamic(model_input = input_onnx_path,
                   model_output = output_onnx_path,
                   weight_type = QuantType.QInt8
                   )

"""Inference Benchmarking"""

def benchmark_logistic_regression(onnx_path, X_test, n_runs = 100):
  session = ort.InferenceSession(onnx_path, providers = ['CPUExecutionProvider'])
  input_name = session.get_inputs()[0].name

  #Session Warmup
  session.run(None, {input_name: X_test.astype(np.float32)})

  start_time = time.perf_counter()
  for _ in range(n_runs):
    session.run(None, {input_name: X_test.astype(np.float32)})
  end_time = time.perf_counter()

  avg_time_ms = ((end_time - start_time) / n_runs) * 1000
  return avg_time_ms

"""Logistic Model Evaluation"""

def evaluate_logistic(onnx_path, X_test, y_test):
  session = ort.InferenceSession(onnx_path, providers = ['CPUExecutionProvider'])
  input_name = session.get_inputs()[0].name
  predictions = session.run(None, {input_name: X_test.astype(np.float32)})[0]
  predictions = (predictions > 0.5).astype(int) #logistic output threshold at 0.5
  return predictions, accuracy_score(y_test, predictions)

"""Full Pipeline Execution"""

def run_ticker(ticker):
  print(f"\n ***** Processing {ticker} *****")
  print("\n")
  print("-- Downloading Data --")
  df = download_data(ticker)


  if len(df) < 30:
    print(f"Warning: Not enough data for {ticker}. Only {len(df)} rows available")


  print(" -- Feature Engineering -- ")
  try:
    X, y, dates = features(df)  # dates is the full DataFrame index
    print(f"Data Shape after feature engineering: {X.shape}")
  except ValueError as e:
    print(f"Error during feature engineering: {e}")
    return None

  print(" -- Preparing the Data -- ")
  try:
    (X_train, X_test, y_train, y_test, idx_train, idx_test), scaler = data_preparation(X, y, dates)
    print(f"Training Data Shape: {X_train.shape}")
    print(f"Test Data Shape: {X_test.shape}")
  except ValueError as e:
    print(f"Error during data preparation: {e}")
    return None

  print(" -- Training Logistic Regression Model -- ")
  model = train_logistic_regression(X_train, y_train)
  y_pred = model.predict(X_test)
  baseline_accuracy = accuracy_score(y_test, y_pred)
  print(f"Baseline Accuracy: {baseline_accuracy:.4f}")

  print(" -- Export to ONNX -- ")
  onnx_fp32_path = f"{ticker}_fp32.onnx"
  export_to_onnx(model, X_train, onnx_fp32_path)

  print(" -- Quantize ONNX -- ")
  onnx_int8_path = f"{ticker}_int8.onnx"
  quantize_logistic_regression(onnx_fp32_path, onnx_int8_path)

  print(" -- Benchmarking -- ")
  time_fp32 = benchmark_logistic_regression(onnx_fp32_path, X_test)
  time_int8 = benchmark_logistic_regression(onnx_int8_path, X_test)

  print(" -- Evaluation -- ")
  prediction_fp32, accuracy_fp32 = evaluate_logistic(onnx_fp32_path, X_test, y_test)
  prediction_int8, accuracy_int8 = evaluate_logistic(onnx_int8_path, X_test, y_test)

  print(" -- Quantized Model Size -- ")
  size_fp32 = os.path.getsize(onnx_fp32_path) / 1024
  size_int8 = os.path.getsize(onnx_int8_path) / 1024

  print(f"\n=== {ticker} Results ===")
  print(f"Baseline FP32 Accuracy:     {accuracy_fp32:.4f}")
  print(f"Quantized INT8 Accuracy:    {accuracy_int8:.4f}")
  print(f"FP32 Inference Time (ms):   {time_fp32:.2f}")
  print(f"INT8 Inference Time (ms):   {time_int8:.2f}")
  print(f"FP32 Model Size (KB):       {size_fp32:.2f}")
  print(f"INT8 Model Size (KB):       {size_int8:.2f}")

  return {
      'Ticker': ticker,
      'Accuracy FP32': accuracy_fp32,
      'Accuracy INT8': accuracy_int8,
      'Time FP32 (ms)': time_fp32,
      'Time INT8 (ms)': time_int8,
      'Size FP32 (KB)': size_fp32,
      'Size INT8 (KB)': size_int8,
      'test_dates': idx_test,  # Store the actual test dates
      'all_dates': dates.tolist(),
      'y_test': y_test,
      'prediction_int8': prediction_int8
  }

def main():
  tickers = ['AAPL', 'SPY', 'MSFT']
  all_results = []
  for ticker in tickers:
    try:
      results = run_ticker(ticker)
      if results is not None:
        all_results.append(results)
      else:
        print(f"Skipping {ticker} due to errors.")
    except Exception as e:
      print(f"An error occurred while processing {ticker}: {e}")

  if not all_results:
    print("No valid results to plot.")
    return

  labels = [result['Ticker'] for result in all_results]

  # Accuracy Comparison
  plt.figure(figsize=(10, 6))
  plt.bar(labels, [r['Accuracy FP32'] for r in all_results], width=0.4, label="FP32", align='edge')
  plt.bar(labels, [r['Accuracy INT8'] for r in all_results], width=-0.4, label="INT8", align='edge')
  plt.ylabel("Accuracy")
  plt.title("Model Accuracy by Ticker")
  plt.legend()
  plt.grid(True, linestyle='--', alpha=0.5)
  plt.tight_layout()
  plt.savefig("multi_accuracy.png")
  plt.show()

  # Inference Time Comparison
  plt.figure(figsize=(10, 6))
  plt.bar(labels, [r['Time FP32 (ms)'] for r in all_results], width=0.4, label="FP32", align='edge')
  plt.bar(labels, [r['Time INT8 (ms)'] for r in all_results], width=-0.4, label="INT8", align='edge')
  plt.ylabel("Avg Inference Time (ms)")
  plt.title("Inference Time by Ticker")
  plt.legend()
  plt.grid(True, linestyle='--', alpha=0.5)
  plt.tight_layout()
  plt.savefig("multi_time.png")
  plt.show()

  # Model Size Comparison
  plt.figure(figsize=(10, 6))
  plt.bar(labels, [r['Size FP32 (KB)'] for r in all_results], width=0.4, label="FP32", align='edge')
  plt.bar(labels, [r['Size INT8 (KB)'] for r in all_results], width=-0.4, label="INT8", align='edge')
  plt.ylabel("Model Size (KB)")
  plt.title("ONNX Model File Size by Ticker")
  plt.legend()
  plt.grid(True, linestyle='--', alpha=0.5)
  plt.tight_layout()
  plt.savefig("multi_size.png")
  plt.show()

  # Plot Actual vs Predicted for all tickers
  for result in all_results:
    try:
      ticker = result['Ticker']
      plt.figure(figsize=(12, 5))

      # Since we're getting persistent indexing issues with the dates,
      # let's simplify by just using a sequential x-axis (0, 1, 2, etc.)
      # and annotate with a few actual dates as tick labels

      # Get the test data
      y_test = np.array(result['y_test'])
      predictions = np.array(result['prediction_int8'])

      # Create simple sequential x-axis
      x_seq = np.arange(len(y_test))

      # Plot using sequential x-axis
      plt.plot(x_seq, y_test, label="Actual", linewidth=2, marker='o', markersize=5)
      plt.plot(x_seq, predictions, label="Predicted (INT8)", linestyle='--', marker='x', markersize=5)

      # Set title and labels
      plt.title(f"{ticker} - Actual vs Predicted Stock Movement (Quantized Model)")
      plt.xlabel("Trading Periods")
      plt.ylabel("Direction (Up=1, Down=0)")
      plt.legend()
      plt.grid(True, linestyle='--', alpha=0.5)
      plt.tight_layout()
      plt.savefig(f"{ticker}_prediction.png")
      plt.show()
    except Exception as e:
      print(f"Error plotting for {ticker}: {e}")
      import traceback
      traceback.print_exc()

if __name__ == '__main__':
  main()

