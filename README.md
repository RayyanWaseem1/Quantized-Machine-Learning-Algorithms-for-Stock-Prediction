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

## Experiments
To evaluate the efficacy of post-training quantization for
predictive stock movement modeling in CPU-constrained
environments, a series of controlled experiments were
conducted. These experiments aimed to benchmark three
traditional machine learning models - logistic regression,
support vector machine, and random forest - before and after
quantization in terms of classification accuracy, inference
latency, and model size. This section outlines the experimental
procedures, environments, and evaluation protocols used to
validate the proposed solution.

### Experimental Setup 
All experiments were performed on a MacBook Pro with
the Apple M3 Pro Chip (11-core CPU, 18 GB unified RAM)
running macOS Sonoma 14.x. The models were developed and
executed using Python 3.10+, with primary dependencies
including:
* scikit-learn for model training and baseline testing
* yfinance for data acquisition
* skl2onnx for exporting models to ONNX format
* onnxruntime for CPU-based inference and quantization
* matplotlib and pandas for visualization and result aggregation

All inference tasks were executed on the CPUExecutionProvider within ONNX Runtime. No GPU
acceleration or neural engine inference was used to ensure
consistency with the project’s goal of evaluating performance
in CPU-only, resource-constrained environments.

### Dataset and Target Definition
Three publicly traded assets were selected for analysis:
Apple Inc. (AAPL), Microsoft Corp. (MSFT), and the S&P 500
Index ETF (SPY). Historical data was retrieved for a rolling
two-year window using the yfinance API, with a daily
resolution.

Each instance in the dataset represents a single trading data
and includes the following engineered features:
* Moving Averages: MA5, MA10, MA20
* Volatility: 20-day rolling standard deviation of returns
* Momentum Indicators (Random Forest only): RSI, MACD, and MACD signal line

The binary classification target was constructed as follows:
* Class 1 (Up): If the following day’s return was positive
* Class 0 (Down): If the following day’s return was negative or zero

The datasets were split chronologically into training and
testing using an 80/20 split, preserving temporal structure to
reflect realistic financial prediction constraints.

### Model Training and Optimization Pipeline
Each model was trained on the FP32 (full-precision)
version of the dataset:
* Logistic Regression was trained with max_iter=1000 using L2 regularization
* SVM used a linear kernel with probability = True to enable ONNX conversion
* Random Forest was trained with 100 estimators, max_depth=10, and minimum split size of 5

After training, models were converted to ONNX format
using skl2onnx.convert_sklearn with appropriate input shape
definitions via FloatTensorType.

Quantization was then applied using ONNX Runtime’s
quantize_dynamic function with QuantType.QUInt8 (or QInt8
for logistic regression) to convert all eligible weights and
activations to 8-bit integers.

### Evaluation Metrics
Three key metrics were used to evaluate performance:
1. Accuracy
   - Post-quantization classification performance was measured against test set ground truth using accuracy_score from scikit-learn. Both FP32 and INT8 versions of each model were evaluated to quantify any degradation in predictive power.
2. Inference Time
   - Each model’s inference latency was measured using time.perf_counter() (or time.time()) by averaging 100 forward passes on the test set. A warm-up run as included before benchmarking. This simulates real-time performance in latency-sensitive environments.
3. Model Size
   - File sizes of the ONNX-serialized models were captured using os.path.getsize() to assess memory footprint. Sizes were compared pre- and post-quantization in kilobytes (kb).
  
### Parameter Settings and Reproducibility
Random seeds were fixed across all scripts using
np.random.seed(42) to ensure consistency between runs. Each
pipeline was encapsulated in modular functions such as
run_ticker() and run_svm(), enabling full reproducibility. Each
function supports batch processing of multiple tickers and
outputs comparison plots and result summaries.


In summary, the experimental setup was carefully designed
to reflect realistic CPU-only deployment constraints, with a
focus on reproducibility, interpretability, and cross-model
benchmarking. By maintaining a uniform evaluation pipeline
and utilizing ONNX for model standardization, the
experiments effectively isolated the impact of quantization on
traditional ML models applied to real-world financial
forecasting. The results of these experiments are presented and
analyzed in the subsequent section.

## Results
This section presents the experimental results comparing
the performance of FP32 (full precision) models against their
post-training INT8 quantized versions for three widely used
machine learning algorithms: logistic regression, support
vector machine (SVM), and random forest. The evaluation was
performed across three major financial assets - Apple (AAPL),
Microsoft (MSFT), and the Standard and Poor’s ETF (SPY) -
with a focus on three core metrics: accuracy, inference time,
and model size.

### Logistic Regression
Logistic regression, a linear and interpretable model,
demonstrated exceptional robustness to quantization. Across all
three datasets - AAPL, MSFT, and SPY - the classification
accuracy was identical between the FP32 and INT8 models.
This consistent retention of of accuracy underscores logistic
regression’s stability under quantization and confirms its
reliability for binary financial classification tasks in resource
constrained environments.

Inference latency varied modestly across tickers. For
AAPL, quantization led to a noticeable speedup of
approximately 12.37% reducing inference time from 0.1099
milliseconds to 0.0963 milliseconds. Similarly, SPY observed
a modest improvement of 2.86%. However, the MSFT dataset
showed a slight degradation in latency, with the quantized
model running marginally slower than the FP32 version.
These results suggest that while quantization can improve
speed, the magnitude of gain may be dependent on data-
specific characteristics.

Contrary to expectations, quantized models exhibited larger
file sizes than their full precision counterparts. For all three
datasets, model size increased from 0.5732 KB (FP32) to
0.6836 KB (INT8), representing a size increase of
approximately 19%. This outcome challenges conventional
assumptions that quantization always reduces storage
requirements. A plausible explanation is that ONNX
serialization overhead becomes dominant in extremely small
models, as seen here.

### Support Vector Machine (SVM)
The linear-kernel SVM exhibited similarly favorable
behavior when subjected to quantization. In all three datasets,
classification accuracy remained exactly the same between
FP32 and INT8 formats, confirming the algorithm’s resilience
to reduced numerical precision. Given SVM’s sensitivity to
margin separation, this result is particularly encouraging and
highlights the generalization capabilities of the quantized
model.

Inference time consistently improved across all datasets.
The most substantial latency reduction occurred in the AAPL
model, where inference time decreased from 0.1844
milliseconds to 0.1585 milliseconds, yielding a 14.0% speed
improvement. MSFT and SPY saw more modest gains of
2.32% and 3.29%, respectively. These improvements validate
the benefit of quantization in real-time financial applications
where even small latency reductions can scale significantly
across high-frequency systems.

Despite these performance gains, model size again showed
a marginal increase after quantization. Each SVM model
increased by roughly 0.07 KB, resulting in an approximate
-0.8% to -0.9% size change. While the numerical change is
small, the decision remains counterintuitive, as quantized
models are theoretically expected to be smaller. These findings
suggest again, that ONNX conversion may have introduced
slight file-level overhead, due to its smaller model
architecture.

### Random Forest
Random forest, an ensemble learning algorithm known for
modeling non-linear relationships, showed high predictive
stability in both full-precision and quantized forms. Accuracy
remained unchanged across all datasets, demonstrating the
model’s insensitivity to quantization. This is consistent with
the inherent structure of decision trees, which operate on
thresholds and splits, making them naturally robust to reduced
numerical precision.

However, in contrast to logistic regression and SVM,
quantized yielded negligible or even negative effects on
inference speed. For AAPL and MSFT, quantized models were
marginally faster, with speed gains of just 0.12% and 0.83%,
respectively. In the case of SPY, the quantized model actually
ran slower, with a speed reduction of 4.76%. These mixed
results indicate that for tree based models, the benefits of
quantization in inference latency are less pronounced and may
depend heavily on the number of estimators, tree-depth, and
implementation-level optimizations.

The model size remained nearly constant post-quantization.
Across all datasets, the change in size was virtually negligible,
with differences in the range of 0.01 - 0.02 KB. This suggests
that the quantization had minimal impact on the memory
impact of the ensemble models, which are already relatively
large due to the aggregation of multiple trees. Therefore, while
quantization preserves accuracy in random forests, its practical
benefits in reducing model size or latency appear limited in this
particular context.

## Conclusion
The experimental results provide compelling evidence that
traditional machine learning models, when quantized using
ONNX, can be effectively deployed in CPU - only
environments without sacrificing accuracy. Logistic regression
and SVM models, in particular, demonstrated measurable
latency improvements, making them well-suited for low-
latency financial forecasting systems. However, the increase in
model size following quantization, especially for small models,
signals a cautionary note and suggests the need for alternative
quantization strategies.

While random forest models remained accurate, the
minimal gains in speed and size suggest that quantization may
not be sufficient to optimize ensemble models. These findings
support broader conclusions in the literature [2]-[4], which
suggest that the benefits of quantization are most pronounced
in simpler, linear architectures.

In conclusion, post-training quantization is a promising
avenue for improving model efficiency in financial time series
forecasting - but its real world advantages depend heavily on
model architecture, data characteristics, and serialization
format.
