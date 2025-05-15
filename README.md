# Quantized-Machine-Learning-Algorithms-for-Stock-Prediction

## Introduction
In the contemporary landscape of quantitative finance and
predictive modeling, the ability to make timely and accurate
predictions on market behavior is a competitive necessity.
Financial time series forecasting, especially in the context of
short-term market analysis, remains a deeply challenging task,
given the stochastic nature of asset prices and the non-
stationarity that is inherent in real-world financial data. While
machine learning (ML) and artificial intelligence (AI) has
emerged as a transformative force and aid across many other
domains such as natural language processing, image
recognition, and healthcare and business analytics, its
application in financial modeling and analysis has remained
complicated by many domain-specific constraints. The
marriage between machine learning and financial analysis in
quantitative finance has yielded itself to a critical tradeoff
between model performance and computational efficiency with
algorithmic accuracy. These constraints, particularly when
operating in environments where access to specialized
hardware (e.g. GPUs) is perhaps economically unjustifiable,
dependent on the resources of the bank. In this context,
resource-constrained CPU-only environments - common in
cost-sensitive situations - pose a unique challenge and
opportunity for optimization.

This project is rooted in the growing need to develop
machine learning solution that can operate under these
constraints without compromising predictive quality. More
specifically, it explores the role of model quantization - a
technique that reduces the numerical precision of model
parameters and operations - as a mean to optimize a model’s
memory usage and inference speed harbored within standard
CPU architectures. Traditionally, models are trained and
deployed using 32-bit floating point (FP32) arithmetic, which,
while incredibly precise, can be both memory-intensive and
computationally expensive. Theoretically, quantization to 8-bit
integer (INT8) precision presents an attractive alternative,
which promises to reduce memory usage/memory footprints
and faster inference times, especially for real-time analysis
where latency is critical. However, while quantization has
demonstrated substantial gains in areas like GPU performance
especially in the realm of deep learning and neural networks, it
has hardly leant itself towards being systematically studied in
quantitative finance. This project most notably focused on
classical ML frameworks such as logistic regression, random
forests, and support vector machines as a way to measure
performance.

The central problem addressed in this project is whether
quantization can provide meaningful improvements in
inference efficiency - namely, in terms of memory usage and
latency - without sacrificing predictive accuracy. By focusing
on interpretable classification models trained on historical data
from Microsoft, Apple, and the Standard & Poor’s 500 (S&P
500), the project offers practical relevance.
The objective and goal of this study was twofold: first, to
establish baseline performance for each model using FP32
precision across all three datasets, and then second, to quantify
the effects of post-training quantization on model size,
inference speed, and market classification accuracy using the
Open Neural Network Exchange (ONNX) framework.
Through this dual-lens evaluation, the project sought out to
inform both the feasibility and limitations of deploying
quantized ML algorithms in CPU-bound, latency-sensitive
financial systems. Contrary to deep learning-based algorithms,
which often require an extensive computational appetite, the
selected models strike a balance between transparency,
performance, optimization, and scalability - making them ideal
candidates for quantization.

## Repository

## Methodology 
This section presents the end-to-end design and
implementation of the project, focused on evaluating the
performance of post-training quantized machine learning
models for short-term stock movement prediction. The primary
objective was to benchmark the memory, latency, and accuracy
characteristics of three baseline models - logistic regression,
support vector machine (SVM), and random forest - when
deployed in INT8 format using the ONNX runtime. The entire
methodology was implemented in Python, using open-source
libraries and executed on a CPU-only environment.

### Data Collection and Preprocessing
Feature engineering steps were tailored to each model
class:

* For all models, daily return was calculated as the percentage change in closing price. Rolling indicators such as the 5-, 10-, and 20-day moving averages (MA5, MA10, MA20), and 20-day volatility were added.
* The random forest model included additional technical indicators, such as Relative Strength Index (RSI) and Moving Average Convergence Divergence (MACD) to enhance nonlinear pattern detection.

The target variable was binary: 1 if the next day’s return
was positive, and 0 otherwise. Data was cleaned to remove
rows with missing values due to rolling calculations. For each
ticker, an 80/20 time-series split was applied, preserving
temporal structure. All features were standardized using
StandardScaler.

### Model Training
Each model was trained on the unquantized (FP32) feature
using scikit-learn:
* Logistic Regression was trained using (LogisticRegression(max_iter = 1000) with L2 regularization. This served as a transparent baseline model.
* Support Vector Machine used a linear kernel (SVC(kernel = ‘linear’, probability = True)) and was trained with default hyperparameters.
* Random Forest was configured with 100 trees, a maximum depth of 10, and minimum samples per leaf and split set to 2 and 5 respectively (RandomForestClassifier(…)).

Training was executed on the CPU using reproducible
random seeds. Models were evaluated on the test set using
classification accuracy.

### ONNX Conversion and Quantization
Each trained model was exported to ONNX format using
the skl2onnx package. This required defining an input type
schema using FloatTensorType, followed by serialization to
a .onnx file.

Post-training quantization was performed using ONNX
Runtime’s quantize_dynamic method with weight type
QuantType.QUInt8 (or QInt8 for logistic regression). This
converted model weights and operators to INT8 precision.

### Model Evaluation and Benchmarking
Model evaluation was carried out using ONNX Runtime’s
InferenceSession for both FP32 and quantized INT8 models.
The following performance metrics were recorded:
* Accuracy was calculated using accuracy_score from scikit-learn on the test set for both models.
* Inference latency was measured by running 100 forward passes and averaging the elapsed time using time.perf_counter() or time.time(). A warm-up pass was included to mitigate cold start effects.
*  Model size was calculated by inspecting the size of the .onnx files using os.path.getsize().

All inference was performed using the CPUExecutionProvider in ONNX Runtime. The SVM and
Random Forest implementations additionally extracted class
probabilities form the quantized models and compared them to
binary ground truth labels.

Confusion matrices were printed to visualize prediction
distribution across actual classes. Rolling prediction accuracy
was plotted using a sliding window approach to detect periods
of temporal degradation or improvement.

### Visualization and Analysis
The project produced detailed visual outputs for each
model and ticker:
* Actual vs. Predicted Direction plots for both FP32 and INT8 models
* Probability score distributions and correct/incorrect classification overlays
* Rolling accuracy plots showing fluctuations in predictive perfjoamcne over time
* Comparison bar charts for FP32 and INT8 accuracy, inference time, and model size
* Bubble charts and percentage improvement plots, highlighting the trade-offs between accuracy and efficiency.

This complete pipeline - ranging from data engineering and
model training to ONNX quantization and CPU benchmarking - provides a robust and reproducible framework for evaluation
quantized classical machine learning models in financial
forecasting applications. The code base can be reused across
domains where resource-efficiency and real-time inference are
necessary.
