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

#### For all models, daily return was calculated as the
percentage change in closing price. Rolling indicators
such as the 5-, 10-, and 20-day moving averages
(MA5, MA10, MA20), and 20-day volatility were
added
