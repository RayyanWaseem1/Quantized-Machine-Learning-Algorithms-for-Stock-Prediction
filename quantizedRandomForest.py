# -*- coding: utf-8 -*-
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from time import perf_counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

!pip install onnx
!pip install onnxruntime
!pip install skl2onnx
import onnx
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from onnxruntime.quantization import quantize_dynamic, QuantType

np.random.seed(42)

"""Load and Prepare Data"""

def load_data(ticker = 'AAPL', period = '2y', interval = '1d'):
  #load stock data and prepare features
  df = yf.download(ticker, period = period, interval = interval)
  df.dropna(inplace = True)
  df['Return'] = df['Close'].pct_change()
  df['MA5'] = df['Close'].rolling(window=5).mean()
  df['MA10'] = df['Close'].rolling(window=10).mean()
  df['MA20'] = df['Close'].rolling(window=20).mean()
  df['Volatlility'] = df['Return'].rolling(window=20).std()

  #Adding RSI (relative strength index)
  delta = df['Close'].diff()
  gain = delta.where(delta > 0, 0)
  loss = -delta.where(delta < 0, 0)
  avg_gain = gain.rolling(window = 14).mean()
  avg_loss = loss.rolling(window = 14).mean()
  rs = avg_gain / avg_loss
  df['RSI'] = 100 - (100 / (1 + rs))

  #Adding MACD (moving average convergence divergence)
  ema12 = df['Close'].ewm(span = 12, adjust = False).mean()
  ema26 = df['Close'].ewm(span = 26, adjust = False).mean()
  df['MACD'] = ema12 - ema26
  df['MACD_signal'] = df['MACD'].ewm(span = 9, adjust = False).mean()

  #Target: 1 if tomorrow's return is positive, 0 otherwise
  df['Target'] = (df['Return'].shift(-1) > 0).astype(int)

  df.dropna(inplace = True)
  features = df[['MA5', 'MA10', 'MA20', 'Volatlility', 'RSI', 'MACD', 'MACD_signal']]
  target = df['Target'].values
  return features, target, df.index

"""Train Random Forest"""

def train_random_forest(X_train, y_train):
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)

  model = RandomForestClassifier(
      n_estimators = 100,
      max_depth = 10,
      min_samples_split = 5,
      min_samples_leaf = 2,
      random_state = 42
  )

  model.fit(X_train_scaled, y_train)
  return model, scaler

"""Convert to ONNX"""

def convert_to_onnx(model, X_sample, output_path):
  initial_type = [('float_input', FloatTensorType([None, X_sample.shape[1]]))]
  onnx_model = convert_sklearn(model, initial_types = initial_type)
  with open(output_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

"""Quantize ONNX Model"""

def quantize_model(input_model_path, output_model_path):
  quantize_dynamic(input_model_path, output_model_path, weight_type=QuantType.QUInt8)

"""Evaluate Random Forest"""

def evaluate_random_forest(onnx_path, X, y, scaler):
  sess = ort.InferenceSession(onnx_path)
  input_name = sess.get_inputs()[0].name

  X_scaled = scaler.transform(X)

  #Getting class probabilities and predictions
  try:
    outputs = sess.get_outputs()
    output_names = [output.name for output in outputs]

    #First output is labels and second is probabilities
    if len(output_names) > 1:
      label_name = output_names[0]
      prob_name = output_names[1]

      #run inference to get both outputs
      results = sess.run(output_names, {input_name: X_scaled.astype(np.float32)})
      predictions = results[0]
      probabilities = results[1]

      #for binary classificatio, get probability of class 1
      if isinstance(probabilities, np.ndarray) and probabilities.shape[1] == 2:
        prob_class1 = probabilities[:, 1]
      else:
        prob_class1 = np.array(predictions, dtype = float)

    else:
      #if only one output is available, likely its class prediction
      output_name = output_names[0]
      predictions = sess.run([output_name], {input_name: X_scaled.astype(np.float32)})[0]
      prob_class1 = np.array(predictions, dtype = float) #use predictions as probabilities

  except Exception as e:
    print(f"Error during inference: {e}")
    #fall back to just getting predictions
    output_name = sess.get_outputs()[0].name
    predictions = sess.run([output_name], {input_name: X_scaled.astype(np.float32)})[0]
    prob_class1 = np.array(predictions, dtype = float)

  accuracy = accuracy_score(y, predictions)
  return accuracy, predictions, prob_class1

"""Benchmark Inference Time"""

def benchmark_rf(onnx_path, X, scaler, n_runs = 100):
  sess = ort.InferenceSession(onnx_path)
  input_name = sess.get_inputs()[0].name

  try:
    output_name = sess.get_outputs()[1].name
  except IndexError:
    output_name = sess.get_outputs()[0].name

  X_scaled = scaler.transform(X)

  #Warm up run
  sess.run([output_name], {input_name: X_scaled.astype(np.float32)})

  start = perf_counter()
  for _ in range(n_runs):
    sess.run([output_name], {input_name: X_scaled.astype(np.float32)})
  end = perf_counter()

  avg_time = (end - start) / n_runs * 1000
  return avg_time

"""Run Full Pipeline"""

def run_rf(tickers=['AAPL', 'SPY', 'MSFT']):
  results = {
    'ticker': [],
    'accuracy_fp32': [],
    'accuracy_int8': [],
    'time_fp32': [],
    'time_int8': [],
    'size_fp32': [],
    'size_int8': [],
    'dates': [],
    'actual': [],
    'pred_fp32': [],
    'pred_int8': [],
    'prob_fp32': [],
    'prob_int8': []
  }

  # Ensure tickers is a list
  if not isinstance(tickers, list):
    tickers = [tickers]  # Convert to a list if it's a single ticker

  for ticker in tickers:
    print(f"\n==== Processing {ticker} ====")
    X, y, dates = load_data(ticker)

    # Check if data is empty using correct method for each data type
    if isinstance(X, pd.DataFrame) and X.empty:
      print(f"Data for {ticker} is empty. Skipping...")
      continue
    elif isinstance(X, np.ndarray) and X.size == 0:
      print(f"Data for {ticker} is empty. Skipping...")
      continue

    # Similarly for y
    if isinstance(y, pd.Series) and y.empty:
      print(f"Target data for {ticker} is empty. Skipping...")
      continue
    elif isinstance(y, np.ndarray) and y.size == 0:
      print(f"Target data for {ticker} is empty. Skipping...")
      continue

    X_train, X_test, y_train, y_test, train_dates, test_dates = train_test_split(
        X, y, dates, shuffle=False, test_size=0.2
    )

    print(f"Training Random Forest for {ticker}...")
    model, scaler = train_random_forest(X_train, y_train)

    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print("\nFeature Importance:")
    print(feature_importance)

    # Convert and Quantize Model
    onnx_fp32_path = f'{ticker}_rf_fp32.onnx'
    onnx_int8_path = f'{ticker}_rf_int8.onnx'

    # Scaled data for ONNX conversion
    X_train_scaled = scaler.transform(X_train)

    print(f"Converting to ONNX format...")
    convert_to_onnx(model, X_train_scaled, onnx_fp32_path)

    print(f"Quantizing model...")
    quantize_model(onnx_fp32_path, onnx_int8_path)

    # Evaluate - I noticed another issue here with function name
    print(f"Evaluating models...")
    accuracy_fp32, preds_fp32, prob_fp32 = evaluate_random_forest(onnx_fp32_path, X_test, y_test, scaler)
    accuracy_int8, preds_int8, prob_int8 = evaluate_random_forest(onnx_int8_path, X_test, y_test, scaler)

    # Benchmark - Another issue here with function name
    print(f"Benchmarking inference time...")
    time_fp32 = benchmark_rf(onnx_fp32_path, X_test, scaler, n_runs=100)
    time_int8 = benchmark_rf(onnx_int8_path, X_test, scaler, n_runs=100)

    # Sizes
    size_fp32 = os.path.getsize(onnx_fp32_path) / 1024  # KB
    size_int8 = os.path.getsize(onnx_int8_path) / 1024  # KB

    # Append results
    results['ticker'].append(ticker)
    results['accuracy_fp32'].append(accuracy_fp32)
    results['accuracy_int8'].append(accuracy_int8)
    results['time_fp32'].append(time_fp32)
    results['time_int8'].append(time_int8)
    results['size_fp32'].append(size_fp32)
    results['size_int8'].append(size_int8)
    results['dates'].append(test_dates)
    results['actual'].append(y_test)
    results['pred_fp32'].append(preds_fp32)
    results['pred_int8'].append(preds_int8)
    results['prob_fp32'].append(prob_fp32)
    results['prob_int8'].append(prob_int8)

    # Print Results
    print(f"\n==== Results for {ticker} ====")
    print(f"FP32 Accuracy: {accuracy_fp32:.4f}")
    print(f"INT8 Accuracy: {accuracy_int8:.4f}")
    print(f"FP32 Inference Time: {time_fp32:.4f} ms")
    print(f"INT8 Inference Time: {time_int8:.4f} ms")
    print(f"FP32 Model Size: {size_fp32:.2f} KB")
    print(f"INT8 Model Size: {size_int8:.2f} KB")
    print(f"Size Reduction: {(size_fp32 - size_int8) / size_fp32 * 100:.2f}%")
    print(f"Speed Improvement: {(time_fp32 - time_int8) / time_fp32 * 100:.2f}%")

    # Confusion matrix
    cm_fp32 = confusion_matrix(y_test, preds_fp32)
    cm_int8 = confusion_matrix(y_test, preds_int8)

    print(f"\nFP32 Confusion Matrix: \n{cm_fp32}")
    print(f"\nINT8 Confusion Matrix: \n{cm_int8}")

  # Create visualizations for each ticker
  for i, ticker in enumerate(results['ticker']):
    create_visualizations(ticker,
                         results['dates'][i],
                         results['actual'][i],
                         results['pred_fp32'][i],
                         results['pred_int8'][i],
                         results['prob_fp32'][i],
                         results['prob_int8'][i])

  # Create comparison charts across tickers
  create_comparison_charts(results['ticker'], results)

  # Creating summary dataframe
  summary_df = pd.DataFrame({
      'Ticker': results['ticker'],
      'FP32 Accuracy': results['accuracy_fp32'],
      'INT8 Accuracy': results['accuracy_int8'],
      'Accuracy Diff': [fp32 - int8 for fp32, int8 in zip(results['accuracy_fp32'], results['accuracy_int8'])],
      'FP32 Time (ms)': results['time_fp32'],
      'INT8 Time (ms)': results['time_int8'],
      'Speed Improvement': [(fp32 - int8) / fp32 * 100 for fp32, int8 in zip(results['time_fp32'], results['time_int8'])],
      'FP32 Size (KB)': results['size_fp32'],
      'INT8 Size (KB)': results['size_int8'],
      'Size Reduction': [(fp32 - int8) / fp32 * 100 for fp32, int8 in zip(results['size_fp32'], results['size_int8'])]
  })

  # Display
  summary_df.set_index('Ticker', inplace=True)
  print("\n==== Summary ====")
  print(summary_df)

  return summary_df, results

def create_visualizations(ticker, dates, actual, pred_fp32, pred_int8, prob_fp32, prob_int8):
  # Converting dates to pandas datetime
  dates_pd = pd.to_datetime(dates)

  # Visualization 1: Actual vs Predicted (Binary)
  plt.figure(figsize=(14, 8))

  # Create a 2x1 subplot layout
  plt.subplot(2, 1, 1)
  plt.title(f"{ticker} - Actual vs Predicted Stock Movement (FP32)", fontsize=14)
  plt.plot(dates_pd, actual, 'b-', label='Actual', linewidth=1.5)
  plt.step(dates_pd, pred_fp32, 'r--', label='Predicted (FP32)', linewidth=1)

  # Add scatter points to emphasize the discrete values
  plt.scatter(dates_pd, actual, color='blue', s=30, alpha=0.6)
  plt.scatter(dates_pd, pred_fp32, color='red', s=30, alpha=0.6, marker='x')

  plt.ylabel('Direction (Up=1, Down=0)')
  plt.legend(loc='best')
  plt.grid(True, linestyle='--', alpha=0.7)

  # Second subplot for INT8 model
  plt.subplot(2, 1, 2)
  plt.title(f"{ticker} - Actual vs Predicted Stock Movement (INT8)", fontsize=14)
  plt.plot(dates_pd, actual, 'b-', label='Actual', linewidth=1.5)
  plt.step(dates_pd, pred_int8, 'g--', label='Predicted (INT8)', linewidth=1)

  # Add scatter points
  plt.scatter(dates_pd, actual, color='blue', s=30, alpha=0.6)
  plt.scatter(dates_pd, pred_int8, color='green', s=30, alpha=0.6, marker='x')

  plt.xlabel('Date')
  plt.ylabel('Direction (Up=1, Down=0)')
  plt.legend(loc='best')
  plt.grid(True, linestyle='--', alpha=0.7)

  plt.tight_layout()
  plt.savefig(f"{ticker}_binary_predictions.png")
  plt.show()

  # Visualization 2: Probability Scores
  plt.figure(figsize=(14, 8))

  # First subplot for FP32 probabilities
  plt.subplot(2, 1, 1)
  plt.title(f"{ticker} - Prediction Probability Scores (FP32)", fontsize=14)

  # Create colored regions for decision boundary
  plt.axhspan(0.5, 1.0, alpha=0.2, color='green')
  plt.axhspan(0, 0.5, alpha=0.2, color='red')

  # Plot prediction probabilities
  plt.plot(dates_pd, prob_fp32, 'o-', color='purple', markersize=4)

  # Plot decision boundary
  plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.7)

  # Color-code points based on correct/incorrect predictions
  correct = pred_fp32 == actual
  incorrect = ~correct

  plt.scatter(dates_pd[correct], prob_fp32[correct], color='green', s=40, label='Correct Prediction')
  plt.scatter(dates_pd[incorrect], prob_fp32[incorrect], color='red', s=40, label='Incorrect Prediction')

  plt.ylabel('Probability of "Up" Movement')
  plt.ylim(-0.05, 1.05)
  plt.legend(loc='best')
  plt.grid(True, linestyle='--', alpha=0.7)

  # Second subplot for INT8 probabilities
  plt.subplot(2, 1, 2)
  plt.title(f"{ticker} - Prediction Probability Scores (INT8)", fontsize=14)

  # Create colored regions
  plt.axhspan(0.5, 1.0, alpha=0.2, color='green')
  plt.axhspan(0, 0.5, alpha=0.2, color='red')

  # Plot prediction probabilities
  plt.plot(dates_pd, prob_int8, 'o-', color='purple', markersize=4)

  # Plot decision boundary
  plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.7)

  # Color-code points based on correct/incorrect predictions
  correct = pred_int8 == actual
  incorrect = ~correct

  plt.scatter(dates_pd[correct], prob_int8[correct], color='green', s=40, label='Correct Prediction')
  plt.scatter(dates_pd[incorrect], prob_int8[incorrect], color='red', s=40, label='Incorrect Prediction')

  plt.xlabel('Date')
  plt.ylabel('Probability of "Up" Movement')
  plt.ylim(-0.05, 1.05)
  plt.legend(loc='best')
  plt.grid(True, linestyle='--', alpha=0.7)

  plt.tight_layout()
  plt.savefig(f"{ticker}_probability_scores.png")
  plt.show()

  # Visualization 3: Moving Average of Prediction Accuracy
  window_size = min(10, len(actual) // 5)  # Adaptive window size
  plt.figure(figsize=(14, 6))

  # Calculate rolling accuracy
  fp32_correct = (pred_fp32 == actual).astype(int)
  int8_correct = (pred_int8 == actual).astype(int)

  fp32_rolling_acc = pd.Series(fp32_correct).rolling(window=window_size).mean()
  int8_rolling_acc = pd.Series(int8_correct).rolling(window=window_size).mean()

  plt.title(f"{ticker} - {window_size}-Day Rolling Prediction Accuracy", fontsize=14)
  plt.plot(dates_pd, fp32_rolling_acc, 'r-', label='FP32 Rolling Accuracy', linewidth=2)
  plt.plot(dates_pd, int8_rolling_acc, 'g-', label='INT8 Rolling Accuracy', linewidth=2)

  plt.axhline(y=0.5, color='gray', linestyle='--', label='Random Guess (50%)')

  plt.xlabel('Date')
  plt.ylabel('Rolling Accuracy')
  plt.ylim(-0.05, 1.05)
  plt.legend(loc='best')
  plt.grid(True, linestyle='--', alpha=0.7)

  plt.tight_layout()
  plt.savefig(f"{ticker}_rolling_accuracy.png")
  plt.show()

def create_comparison_charts(tickers, results):
  x = np.arange(len(tickers))

  # Accuracy Comparison
  plt.figure(figsize=(10, 6))
  width = 0.35
  plt.bar(x - width/2, results['accuracy_fp32'], width, label='FP32')
  plt.bar(x + width/2, results['accuracy_int8'], width, label='INT8')

  plt.axhline(y=0.5, color='red', linestyle='--', label='Random Guess (50%)')

  plt.xlabel('Ticker')
  plt.ylabel('Accuracy')
  plt.title('Random Forest Model Accuracy by Ticker')
  plt.xticks(x, tickers)
  plt.legend()
  plt.grid(True, axis='y', linestyle='--', alpha=0.7)

  # Add value labels on bars
  for i, v in enumerate(results['accuracy_fp32']):
    plt.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
  for i, v in enumerate(results['accuracy_int8']):
    plt.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

  plt.tight_layout()
  plt.savefig("accuracy_comparison.png")
  plt.show()

  # Inference Time Comparison
  plt.figure(figsize=(10, 6))
  plt.bar(x - width/2, results['time_fp32'], width, label='FP32')
  plt.bar(x + width/2, results['time_int8'], width, label='INT8')

  plt.xlabel('Ticker')
  plt.ylabel('Inference Time (ms)')
  plt.title('Random Forest Model Inference Time by Ticker')
  plt.xticks(x, tickers)
  plt.legend()
  plt.grid(True, axis='y', linestyle='--', alpha=0.7)

  # Add value labels
  for i, v in enumerate(results['time_fp32']):
    plt.text(i - width/2, v + 0.1, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
  for i, v in enumerate(results['time_int8']):
    plt.text(i + width/2, v + 0.1, f'{v:.2f}', ha='center', va='bottom', fontsize=9)

  plt.tight_layout()
  plt.savefig("inference_time_comparison.png")
  plt.show()

  # Model Size Comparison
  plt.figure(figsize=(10, 6))
  plt.bar(x - width/2, results['size_fp32'], width, label='FP32')
  plt.bar(x + width/2, results['size_int8'], width, label='INT8')

  plt.xlabel('Ticker')
  plt.ylabel('Model Size (KB)')
  plt.title('ONNX Random Forest Model Size by Ticker')
  plt.xticks(x, tickers)
  plt.legend()
  plt.grid(True, axis='y', linestyle='--', alpha=0.7)

  # Add value labels
  for i, v in enumerate(results['size_fp32']):
    plt.text(i - width/2, v + 1, f'{v:.1f}', ha='center', va='bottom', fontsize=9)
  for i, v in enumerate(results['size_int8']):
    plt.text(i + width/2, v + 1, f'{v:.1f}', ha='center', va='bottom', fontsize=9)

  plt.tight_layout()
  plt.savefig("model_size_comparison.png")
  plt.show()

  # Size Reduction & Speed Improvement
  plt.figure(figsize=(10, 6))

  size_reduction = [(fp32 - int8) / fp32 * 100 for fp32, int8 in zip(results['size_fp32'], results['size_int8'])]
  speed_improvement = [(fp32 - int8) / fp32 * 100 for fp32, int8 in zip(results['time_fp32'], results['time_int8'])]

  plt.bar(x - width/2, size_reduction, width, label='Size Reduction')
  plt.bar(x + width/2, speed_improvement, width, label='Speed Improvement')

  plt.xlabel('Ticker')
  plt.ylabel('Percentage (%)')
  plt.title('Quantization Benefits by Ticker')
  plt.xticks(x, tickers)
  plt.legend()
  plt.grid(True, axis='y', linestyle='--', alpha=0.7)

  # Add value labels
  for i, v in enumerate(size_reduction):
    plt.text(i - width/2, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
  for i, v in enumerate(speed_improvement):
    plt.text(i + width/2, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)

  plt.tight_layout()
  plt.savefig("quantization_benefits.png")
  plt.show()

  # Accuracy vs Speed - Bubble chart
  plt.figure(figsize=(12, 8))

  # Normalize size for bubble size
  size_fp32_norm = [s/max(results['size_fp32'])*500 for s in results['size_fp32']]
  size_int8_norm = [s/max(results['size_fp32'])*500 for s in results['size_int8']]

  # Plot FP32 models
  plt.scatter(results['time_fp32'], results['accuracy_fp32'], s=size_fp32_norm,
              alpha=0.7, color='blue', label='FP32 Models')

  # Plot INT8 models
  plt.scatter(results['time_int8'], results['accuracy_int8'], s=size_int8_norm,
              alpha=0.7, color='red', label='INT8 Models')

  # Add labels for each point
  for i, ticker in enumerate(tickers):
    plt.annotate(ticker, (results['time_fp32'][i], results['accuracy_fp32'][i]),
                fontsize=9, color='blue')
    plt.annotate(ticker, (results['time_int8'][i], results['accuracy_int8'][i]),
                fontsize=9, color='red')

  plt.xlabel('Inference Time (ms)')
  plt.ylabel('Accuracy')
  plt.title('Accuracy vs Speed Tradeoff')
  plt.grid(True, linestyle='--', alpha=0.7)
  plt.legend()

  plt.tight_layout()
  plt.savefig("accuracy_vs_speed.png")
  plt.show()

if __name__ == "__main__":
  print("Quantized Random Forest Stock Predictor")
  print("======================================")

  tickers = ['AAPL', 'MSFT', 'SPY']
  print(f"Running prediction for tickers: {', '.join(tickers)}")
  df, results = run_rf(tickers)

