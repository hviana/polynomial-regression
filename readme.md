> ## ⚠️🚨 IMPORTANT: THIS LIBRARY HAS BEEN DEPRECATED 🚨⚠️
> 
> ---
> 
> ### 🔄 This library has been replaced by a newer, more powerful version!
> 
> <table>
> <tr>
> <td>
> 
> ### ❌ OLD (This Repository)
> `@hviana/polynomial-regression`
> 
> </td>
> <td>
> 
> ### ✅ NEW (Use This Instead)
> `@hviana/multivariate-convolutional-regression`
> 
> </td>
> </tr>
> </table>
> 
> ---
> 
> ### 📦 Migration Links
> 
> | Platform | Link |
> |----------|------|
> | 🌐 **JSR Registry** | 👉 [https://jsr.io/@hviana/multivariate-convolutional-regression](https://jsr.io/@hviana/multivariate-convolutional-regression) |
> | 🐙 **GitHub Repository** | 👉 [https://github.com/hviana/multivariate-convolutional-regression](https://github.com/hviana/multivariate-convolutional-regression) |
> 
> ---
> 
> ### 🛑 Please migrate to the new library for:
> - ✨ New features and improvements
> - 🐛 Bug fixes and security updates
> - 📚 Better documentation
> - 🔧 Continued maintenance and support
> 
> ---

Model: # 🚀 Multivariate Polynomial Regression

**A high-performance TypeScript library for multivariate polynomial regression
with incremental online learning, Adam optimizer, and z-score normalization.**

---

## 📋 Table of Contents

- [Key Advantages](#-key-advantages)
- [Quick Start](#-quick-start)
- [API Reference](#-api-reference)
- [Configuration Parameters](#-configuration-parameters)
- [Use Case Optimization Guide](#-use-case-optimization-guide)
- [Mathematical Background](#-mathematical-background)
- [Performance Tips](#-performance-tips)

---

## ✨ Key Advantages

### 🎯 **Core Strengths**

| Feature                           | Benefit                                                                       |
| --------------------------------- | ----------------------------------------------------------------------------- |
| 🔄 **Online Learning**            | Incrementally update your model with new data without retraining from scratch |
| 📈 **Polynomial Features**        | Automatically captures non-linear relationships up to degree 10               |
| ⚡ **Adam Optimizer**             | State-of-the-art optimization with adaptive learning rates                    |
| 🛡️ **Robust to Outliers**         | Built-in z-score outlier detection and downweighting                          |
| 🌊 **Concept Drift Detection**    | ADWIN algorithm automatically detects and adapts to data distribution changes |
| 📊 **Uncertainty Quantification** | Predictions include confidence intervals and standard errors                  |

### 🏆 **Performance Features**

```
┌─────────────────────────────────────────────────────────────────┐
│  🧮 Float64Array          │  Maximum numerical precision        │
│  🔁 Object Pooling        │  Minimized garbage collection       │
│  📦 Buffer Preallocation  │  Zero-allocation hot paths          │
│  🎲 Xavier Initialization │  Optimal gradient flow              │
│  📉 Cosine LR Decay       │  Smooth convergence                 │
│  🔢 Welford's Algorithm   │  Numerically stable statistics      │
└─────────────────────────────────────────────────────────────────┘
```

### 🎨 **Why Choose This Library?**

1. **🔥 Production-Ready** - Handles edge cases, validates inputs, and provides
   meaningful error messages
2. **📱 Memory Efficient** - Object pooling and buffer reuse minimize memory
   footprint
3. **🎛️ Highly Configurable** - 12 tunable parameters for fine-grained control
4. **📖 Self-Documenting** - Rich TypeScript interfaces with comprehensive JSDoc
5. **🔄 Dual Training Modes** - Both online (streaming) and batch training
   supported
6. **📊 Full Observability** - Access to weights, normalization stats, and model
   summary

---

## 🚀 Quick Start

### Basic Usage

```typescript
import { MultivariatePolynomialRegression } from "jsr:@hviana/polynomial-regression@1.1.0";

// 1️⃣ Create a model
const model = new MultivariatePolynomialRegression({
  polynomialDegree: 2,
  learningRate: 0.01,
});

// 2️⃣ Train with batch data
const result = model.fitBatch({
  xCoordinates: [
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6],
  ],
  yCoordinates: [
    [5],
    [8],
    [13],
    [20],
    [29],
  ],
  epochs: 100,
});

console.log(`✅ Training complete! Final loss: ${result.finalLoss}`);

// 3️⃣ Make predictions
const predictions = model.predict(3);
predictions.predictions.forEach((p, i) => {
  console.log(
    `Step ${i + 1}: ${p.predicted[0].toFixed(2)} ± ${
      p.standardError[0].toFixed(2)
    }`,
  );
});
```

### Online Learning (Streaming Data)

```typescript
const model = new MultivariatePolynomialRegression();

// Stream data point by point
for (const dataPoint of dataStream) {
  const result = model.fitOnline({
    xCoordinates: [dataPoint.x],
    yCoordinates: [dataPoint.y],
  });

  if (result.driftDetected) {
    console.log("⚠️ Concept drift detected! Model adapting...");
  }

  if (result.converged) {
    console.log("✅ Model converged!");
    break;
  }
}
```

---

## 📚 API Reference

### Constructor

```typescript
const model = new MultivariatePolynomialRegression(config?: MultivariatePolynomialRegressionConfig);
```

### Methods

| Method                    | Description                            | Returns              |
| ------------------------- | -------------------------------------- | -------------------- |
| `fitOnline(input)`        | Incremental learning on streaming data | `FitResult`          |
| `fitBatch(input)`         | Batch training with mini-batches       | `BatchFitResult`     |
| `predict(steps)`          | Generate predictions with uncertainty  | `PredictionResult`   |
| `getModelSummary()`       | Get model state overview               | `ModelSummary`       |
| `getWeights()`            | Access weight matrices                 | `WeightInfo`         |
| `getNormalizationStats()` | Get normalization parameters           | `NormalizationStats` |
| `reset()`                 | Reset model to initial state           | `void`               |

---

## ⚙️ Configuration Parameters

### Overview

```typescript
interface MultivariatePolynomialRegressionConfig {
  polynomialDegree?: number; // Feature expansion degree
  learningRate?: number; // Base learning rate
  warmupSteps?: number; // LR warmup period
  totalSteps?: number; // Total training steps
  beta1?: number; // Adam β₁ parameter
  beta2?: number; // Adam β₂ parameter
  epsilon?: number; // Numerical stability
  regularizationStrength?: number; // L2 regularization
  batchSize?: number; // Mini-batch size
  convergenceThreshold?: number; // Early stopping threshold
  outlierThreshold?: number; // Outlier z-score threshold
  adwinDelta?: number; // Drift detection sensitivity
}
```

---

### 1️⃣ **polynomialDegree**

> 📐 Controls the complexity of feature expansion

| Property    | Value    |
| ----------- | -------- |
| **Type**    | `number` |
| **Range**   | 1 - 10   |
| **Default** | 2        |

#### 📖 Explanation

The polynomial degree determines how many polynomial features are generated from
your input. For inputs with `n` dimensions and degree `d`, the number of
features is:

$$C(n+d, d) = \frac{(n+d)!}{n! \cdot d!}$$

**Feature expansion example for 2D input `[x₁, x₂]`:**

| Degree | Features Generated                                  | Count |
| ------ | --------------------------------------------------- | ----- |
| 1      | `1, x₁, x₂`                                         | 3     |
| 2      | `1, x₁, x₂, x₁², x₁x₂, x₂²`                         | 6     |
| 3      | `1, x₁, x₂, x₁², x₁x₂, x₂², x₁³, x₁²x₂, x₁x₂², x₂³` | 10    |

#### 💡 Examples

```typescript
// 🔵 Linear relationships (simple, fast)
const linearModel = new MultivariatePolynomialRegression({
  polynomialDegree: 1,
});

// 🟢 Quadratic patterns (parabolas, ellipses)
const quadraticModel = new MultivariatePolynomialRegression({
  polynomialDegree: 2,
});

// 🟠 Complex non-linear patterns
const complexModel = new MultivariatePolynomialRegression({
  polynomialDegree: 4,
});
```

#### 🎯 Optimization Guide

| Use Case                | Recommended Degree | Rationale                      |
| ----------------------- | ------------------ | ------------------------------ |
| Linear trends           | 1                  | Minimal complexity, fastest    |
| Quadratic curves        | 2                  | Captures curvature efficiently |
| Sensor data             | 2-3                | Balances accuracy and speed    |
| Physics simulations     | 3-4                | Captures complex dynamics      |
| High-precision modeling | 4-6                | When accuracy is paramount     |
| Research/exploration    | 6-10               | Maximum flexibility            |

> ⚠️ **Warning**: Higher degrees exponentially increase feature count and
> training time. A 10D input with degree 5 produces 3,003 features!

---

### 2️⃣ **learningRate**

> 🎚️ Controls the step size during optimization

| Property    | Value    |
| ----------- | -------- |
| **Type**    | `number` |
| **Range**   | > 0      |
| **Default** | 0.001    |

#### 📖 Explanation

The learning rate determines how much weights are updated in response to
gradients:

```
W_new = W_old - learningRate × gradient
```

The library uses **cosine decay with warmup**:

- **Warmup**: `η_t = η × t / warmupSteps`
- **Decay**: `η_t = η × 0.5 × (1 + cos(π × progress))`

#### 💡 Examples

```typescript
// 🐢 Conservative learning (stable but slow)
const conservativeModel = new MultivariatePolynomialRegression({
  learningRate: 0.0001,
});

// 🚶 Standard learning rate
const standardModel = new MultivariatePolynomialRegression({
  learningRate: 0.001,
});

// 🏃 Aggressive learning (fast but may overshoot)
const aggressiveModel = new MultivariatePolynomialRegression({
  learningRate: 0.01,
});

// 🚀 Very fast initial convergence
const fastModel = new MultivariatePolynomialRegression({
  learningRate: 0.1,
  warmupSteps: 200, // Important: longer warmup for stability
});
```

#### 🎯 Optimization Guide

| Scenario                     | Recommended LR | Notes                       |
| ---------------------------- | -------------- | --------------------------- |
| Small dataset (<100 samples) | 0.01 - 0.1     | Faster convergence needed   |
| Medium dataset (100-10K)     | 0.001 - 0.01   | Standard range              |
| Large dataset (>10K)         | 0.0001 - 0.001 | Stability over speed        |
| Online/streaming             | 0.001 - 0.01   | Balance adaptation speed    |
| High polynomial degree       | 0.0001 - 0.001 | Prevent exploding gradients |
| Noisy data                   | 0.0001 - 0.001 | Smoother updates            |

---

### 3️⃣ **warmupSteps**

> 🌡️ Gradual learning rate increase at start

| Property    | Value              |
| ----------- | ------------------ |
| **Type**    | `number` (integer) |
| **Range**   | ≥ 0                |
| **Default** | 100                |

#### 📖 Explanation

During warmup, the learning rate linearly increases from near-zero to the full
rate:

```
η_t = learningRate × (t + 1) / warmupSteps
```

This prevents large initial weight updates when statistics are uncertain.

```
Learning Rate
     ^
  η  |           ╭──────────────╮
     |          ╱                ╲  ← Cosine decay
     |         ╱                  ╲
     |        ╱                    ╲
     |   ╱───╯                      ╲
   0 |──╱─────────────────────────────→ Steps
       ↑           ↑
    Warmup     Full rate
```

#### 💡 Examples

```typescript
// 🔥 No warmup (aggressive start)
const noWarmup = new MultivariatePolynomialRegression({
  warmupSteps: 0,
});

// 🌤️ Quick warmup (standard)
const quickWarmup = new MultivariatePolynomialRegression({
  warmupSteps: 50,
});

// 🌅 Gradual warmup (safe)
const gradualWarmup = new MultivariatePolynomialRegression({
  warmupSteps: 200,
});

// 🌄 Extended warmup (very stable)
const extendedWarmup = new MultivariatePolynomialRegression({
  warmupSteps: 500,
  learningRate: 0.01, // Can use higher LR with longer warmup
});
```

#### 🎯 Optimization Guide

| Scenario              | Recommended Steps | Notes                             |
| --------------------- | ----------------- | --------------------------------- |
| Quick experimentation | 0-20              | Speed over stability              |
| Standard training     | 50-100            | Good balance                      |
| High learning rate    | 100-300           | Prevents initial instability      |
| Large batches         | 50-100            | Statistics stabilize faster       |
| Online learning       | 100-200           | Gives time for stats to stabilize |
| Production systems    | 100-500           | Maximum stability                 |

---

### 4️⃣ **totalSteps**

> 📏 Total steps for learning rate schedule

| Property    | Value              |
| ----------- | ------------------ |
| **Type**    | `number` (integer) |
| **Range**   | > 0                |
| **Default** | 10000              |

#### 📖 Explanation

`totalSteps` defines the full cosine decay schedule length. After `totalSteps`,
the learning rate approaches zero.

```
Effective LR = baseLR × 0.5 × (1 + cos(π × (step - warmup) / (total - warmup)))
```

#### 💡 Examples

```typescript
// 📊 Short training (quick experiments)
const shortTraining = new MultivariatePolynomialRegression({
  totalSteps: 1000,
  warmupSteps: 50,
});

// 📈 Standard training
const standardTraining = new MultivariatePolynomialRegression({
  totalSteps: 10000,
  warmupSteps: 100,
});

// 📉 Extended training (complex patterns)
const extendedTraining = new MultivariatePolynomialRegression({
  totalSteps: 100000,
  warmupSteps: 1000,
});

// ♾️ Continuous online learning
const onlineLearning = new MultivariatePolynomialRegression({
  totalSteps: 1000000, // Very slow decay
  warmupSteps: 1000,
});
```

#### 🎯 Optimization Guide

| Training Mode         | Recommended Steps | Notes                    |
| --------------------- | ----------------- | ------------------------ |
| Batch (small data)    | 1,000 - 5,000     | Faster convergence       |
| Batch (large data)    | 10,000 - 50,000   | Full dataset exploration |
| Online (session)      | 5,000 - 20,000    | Typical session length   |
| Online (continuous)   | 100,000+          | Long-running systems     |
| Hyperparameter search | 500 - 2,000       | Quick evaluation         |

---

### 5️⃣ **beta1**

> 📊 Adam first moment decay rate

| Property    | Value    |
| ----------- | -------- |
| **Type**    | `number` |
| **Range**   | [0, 1]   |
| **Default** | 0.9      |

#### 📖 Explanation

β₁ controls the exponential moving average of gradients (momentum):

```
m_t = β₁ × m_{t-1} + (1 - β₁) × g_t
```

Higher values = more momentum = smoother but slower adaptation.

#### 💡 Examples

```typescript
// 🎯 Standard momentum
const standardMomentum = new MultivariatePolynomialRegression({
  beta1: 0.9,
});

// 🌊 High momentum (smoother updates)
const highMomentum = new MultivariatePolynomialRegression({
  beta1: 0.95,
});

// ⚡ Low momentum (faster adaptation)
const lowMomentum = new MultivariatePolynomialRegression({
  beta1: 0.8,
});

// 🔄 Very low momentum (online learning)
const adaptiveMomentum = new MultivariatePolynomialRegression({
  beta1: 0.5, // Quick response to changes
});
```

#### 🎯 Optimization Guide

| Scenario               | Recommended β₁ | Rationale           |
| ---------------------- | -------------- | ------------------- |
| Standard training      | 0.9            | Well-tested default |
| Noisy gradients        | 0.95 - 0.99    | More smoothing      |
| Concept drift expected | 0.5 - 0.8      | Faster adaptation   |
| Sparse updates         | 0.9 - 0.95     | Maintain momentum   |
| Fine-tuning            | 0.85 - 0.9     | Balanced response   |

---

### 6️⃣ **beta2**

> 📈 Adam second moment decay rate

| Property    | Value    |
| ----------- | -------- |
| **Type**    | `number` |
| **Range**   | [0, 1]   |
| **Default** | 0.999    |

#### 📖 Explanation

β₂ controls the exponential moving average of squared gradients (variance):

```
v_t = β₂ × v_{t-1} + (1 - β₂) × g_t²
```

This enables per-parameter learning rates, crucial for Adam's adaptivity.

#### 💡 Examples

```typescript
// 🎯 Standard variance tracking
const standard = new MultivariatePolynomialRegression({
  beta2: 0.999,
});

// 📊 Faster variance adaptation
const fasterVariance = new MultivariatePolynomialRegression({
  beta2: 0.99,
});

// 🔬 Very stable variance estimate
const stableVariance = new MultivariatePolynomialRegression({
  beta2: 0.9999,
});
```

#### 🎯 Optimization Guide

| Scenario        | Recommended β₂ | Rationale                 |
| --------------- | -------------- | ------------------------- |
| General use     | 0.999          | Default works well        |
| Sparse features | 0.999 - 0.9999 | Stable estimates          |
| Dense features  | 0.99 - 0.999   | Faster adaptation         |
| Online learning | 0.99 - 0.999   | Balance stability/speed   |
| Long training   | 0.999 - 0.9999 | Prevent late-stage issues |

---

### 7️⃣ **epsilon**

> 🔢 Numerical stability constant

| Property    | Value    |
| ----------- | -------- |
| **Type**    | `number` |
| **Range**   | > 0      |
| **Default** | 1e-8     |

#### 📖 Explanation

Epsilon prevents division by zero in the Adam update:

```
W -= lr × m̂ / (√v̂ + ε)
```

Also used in normalization to prevent division by zero standard deviation.

#### 💡 Examples

```typescript
// 🎯 Standard precision
const standard = new MultivariatePolynomialRegression({
  epsilon: 1e-8,
});

// 🔬 High precision (for well-scaled data)
const highPrecision = new MultivariatePolynomialRegression({
  epsilon: 1e-10,
});

// 🛡️ Safer for mixed-precision
const safer = new MultivariatePolynomialRegression({
  epsilon: 1e-7,
});

// 🔧 Very safe (noisy/unstable data)
const verySafe = new MultivariatePolynomialRegression({
  epsilon: 1e-6,
});
```

#### 🎯 Optimization Guide

| Scenario                | Recommended ε | Notes              |
| ----------------------- | ------------- | ------------------ |
| Standard Float64        | 1e-8          | Default is optimal |
| Large gradient variance | 1e-7 - 1e-6   | More stability     |
| Small gradients         | 1e-9 - 1e-8   | Better precision   |
| Mixed precision         | 1e-6 - 1e-7   | Prevent underflow  |

> 💡 **Tip**: Rarely needs adjustment. Change only if you see NaN or Infinity
> values.

---

### 8️⃣ **regularizationStrength**

> 🎛️ L2 regularization (weight decay) strength

| Property    | Value    |
| ----------- | -------- |
| **Type**    | `number` |
| **Range**   | ≥ 0      |
| **Default** | 1e-4     |

#### 📖 Explanation

L2 regularization adds a penalty term to prevent overfitting:

```
Loss = MSE + (λ/2) × ||W||²
Gradient += λ × W
```

Higher values push weights toward zero, reducing model complexity.

#### 💡 Examples

```typescript
// ❌ No regularization (risk of overfitting)
const noReg = new MultivariatePolynomialRegression({
  regularizationStrength: 0,
});

// 🎯 Light regularization (default)
const lightReg = new MultivariatePolynomialRegression({
  regularizationStrength: 1e-4,
});

// 🛡️ Moderate regularization
const moderateReg = new MultivariatePolynomialRegression({
  regularizationStrength: 1e-3,
});

// 🔒 Strong regularization (prevent overfitting)
const strongReg = new MultivariatePolynomialRegression({
  regularizationStrength: 1e-2,
});

// 💪 Very strong regularization
const veryStrongReg = new MultivariatePolynomialRegression({
  regularizationStrength: 0.1,
});
```

#### 🎯 Optimization Guide

| Scenario               | Recommended λ | Notes                               |
| ---------------------- | ------------- | ----------------------------------- |
| Large dataset          | 0 - 1e-5      | Less regularization needed          |
| Small dataset          | 1e-3 - 1e-2   | Prevent overfitting                 |
| High polynomial degree | 1e-3 - 1e-2   | More features = more regularization |
| Noisy data             | 1e-3 - 1e-2   | Smooth out noise                    |
| Clean data             | 1e-5 - 1e-4   | Preserve signal                     |
| Online learning        | 1e-4 - 1e-3   | Stabilize updates                   |

**Rule of thumb**: `λ ∝ degree² / dataset_size`

---

### 9️⃣ **batchSize**

> 📦 Mini-batch size for batch training

| Property    | Value              |
| ----------- | ------------------ |
| **Type**    | `number` (integer) |
| **Range**   | > 0                |
| **Default** | 32                 |

#### 📖 Explanation

During batch training, data is processed in mini-batches:

```
for each epoch:
    shuffle data
    for batch in batches(data, batchSize):
        accumulate gradients over batch
        update weights
```

| Batch Size     | Gradient Quality | Memory | Speed  |
| -------------- | ---------------- | ------ | ------ |
| Small (8-16)   | Noisy            | Low    | Slow   |
| Medium (32-64) | Balanced         | Medium | Medium |
| Large (128+)   | Smooth           | High   | Fast   |

#### 💡 Examples

```typescript
// 🔬 Small batches (more noise, better generalization)
const smallBatch = new MultivariatePolynomialRegression({
  batchSize: 8,
  learningRate: 0.001, // Lower LR for stability
});

// 🎯 Standard batch size
const standardBatch = new MultivariatePolynomialRegression({
  batchSize: 32,
});

// 📊 Large batches (smoother gradients)
const largeBatch = new MultivariatePolynomialRegression({
  batchSize: 128,
  learningRate: 0.004, // Can scale up LR
});

// 🚀 Full batch gradient descent
const fullBatch = new MultivariatePolynomialRegression({
  batchSize: 10000, // If dataset fits
});
```

#### 🎯 Optimization Guide

| Scenario                 | Recommended Size | Notes                    |
| ------------------------ | ---------------- | ------------------------ |
| Small dataset (<100)     | 8 - 16           | More updates per epoch   |
| Medium dataset           | 32 - 64          | Good balance             |
| Large dataset            | 64 - 256         | Leverage parallelism     |
| High noise               | 16 - 32          | Regularization via noise |
| Quick convergence needed | 64 - 128         | Smoother gradients       |
| Memory constrained       | 8 - 32           | Minimize footprint       |

**Linear scaling rule**: When doubling batch size, consider increasing learning
rate by ~√2.

---

### 🔟 **convergenceThreshold**

> 🎯 Early stopping threshold

| Property    | Value    |
| ----------- | -------- |
| **Type**    | `number` |
| **Range**   | > 0      |
| **Default** | 1e-6     |

#### 📖 Explanation

Training stops early when loss change falls below threshold:

```
if |loss_{t} - loss_{t-1}| < convergenceThreshold:
    stop training (converged)
```

Also uses patience mechanism: stops after 10 epochs without improvement.

#### 💡 Examples

```typescript
// 🔬 High precision (train longer)
const highPrecision = new MultivariatePolynomialRegression({
  convergenceThreshold: 1e-8,
});

// 🎯 Standard precision
const standard = new MultivariatePolynomialRegression({
  convergenceThreshold: 1e-6,
});

// ⚡ Quick convergence (stop early)
const quick = new MultivariatePolynomialRegression({
  convergenceThreshold: 1e-4,
});

// 🚀 Very quick (fast experiments)
const veryQuick = new MultivariatePolynomialRegression({
  convergenceThreshold: 1e-3,
});
```

#### 🎯 Optimization Guide

| Scenario              | Recommended Threshold | Notes             |
| --------------------- | --------------------- | ----------------- |
| Production model      | 1e-7 - 1e-6           | High quality      |
| Standard training     | 1e-6 - 1e-5           | Good balance      |
| Hyperparameter search | 1e-4 - 1e-3           | Quick evaluation  |
| Real-time systems     | 1e-4                  | Faster adaptation |
| Scientific computing  | 1e-8 - 1e-7           | Maximum precision |

---

### 1️⃣1️⃣ **outlierThreshold**

> 🚨 Z-score threshold for outlier detection

| Property    | Value    |
| ----------- | -------- |
| **Type**    | `number` |
| **Range**   | > 0      |
| **Default** | 3.0      |

#### 📖 Explanation

Samples with residual z-scores above threshold are downweighted:

```
z_score = |residual - mean_residual| / std_residual
if z_score > threshold:
    sample_weight = 0.1  // Downweight outlier
else:
    sample_weight = 1.0  // Normal weight
```

| Threshold | % Data as Outliers (Normal Distribution) |
| --------- | ---------------------------------------- |
| 2.0       | ~4.6%                                    |
| 2.5       | ~1.2%                                    |
| 3.0       | ~0.3%                                    |
| 3.5       | ~0.05%                                   |

#### 💡 Examples

```typescript
// 🚨 Aggressive outlier detection
const aggressive = new MultivariatePolynomialRegression({
  outlierThreshold: 2.0,
});

// 🎯 Standard detection
const standard = new MultivariatePolynomialRegression({
  outlierThreshold: 3.0,
});

// 🛡️ Conservative detection
const conservative = new MultivariatePolynomialRegression({
  outlierThreshold: 4.0,
});

// ❌ Disable outlier detection
const noOutlierDetection = new MultivariatePolynomialRegression({
  outlierThreshold: 100, // Effectively disabled
});
```

#### 🎯 Optimization Guide

| Data Quality       | Recommended Threshold | Notes                |
| ------------------ | --------------------- | -------------------- |
| Clean data         | 3.5 - 4.0             | Minimal intervention |
| Some noise         | 3.0                   | Default handles well |
| Noisy data         | 2.5 - 3.0             | More aggressive      |
| Many outliers      | 2.0 - 2.5             | Strong filtering     |
| Outliers are valid | 4.0+                  | Don't discard        |

> ⚠️ **Note**: Outlier detection only activates after 20 samples (needs
> statistics).

---

### 1️⃣2️⃣ **adwinDelta**

> 🌊 ADWIN drift detection sensitivity

| Property    | Value    |
| ----------- | -------- |
| **Type**    | `number` |
| **Range**   | (0, 1)   |
| **Default** | 0.002    |

#### 📖 Explanation

ADWIN (ADaptive WINdowing) detects concept drift by comparing window means:

```
Drift detected when: |μ_left - μ_right| ≥ √((2/m) × ln(2/δ))
```

Where δ (delta) controls sensitivity:

- **Smaller δ** = More sensitive = More false positives
- **Larger δ** = Less sensitive = May miss drift

#### 💡 Examples

```typescript
// 🔬 Highly sensitive (catch small changes)
const sensitive = new MultivariatePolynomialRegression({
  adwinDelta: 0.0001,
});

// 🎯 Standard sensitivity
const standard = new MultivariatePolynomialRegression({
  adwinDelta: 0.002,
});

// 🛡️ Conservative (only major shifts)
const conservative = new MultivariatePolynomialRegression({
  adwinDelta: 0.01,
});

// 😴 Low sensitivity (stable environments)
const lowSensitivity = new MultivariatePolynomialRegression({
  adwinDelta: 0.1,
});
```

#### 🎯 Optimization Guide

| Environment         | Recommended δ  | Notes                 |
| ------------------- | -------------- | --------------------- |
| Stable data         | 0.01 - 0.1     | Avoid false positives |
| Some drift expected | 0.002 - 0.01   | Default range         |
| Frequent drift      | 0.0005 - 0.002 | Quick detection       |
| Mission critical    | 0.0001 - 0.001 | Don't miss drift      |
| Batch training only | 0.1+           | Effectively disabled  |

---

## 🎨 Use Case Optimization Guide

### 📊 **Time Series Forecasting**

```typescript
const timeSeriesModel = new MultivariatePolynomialRegression({
  polynomialDegree: 2, // Capture trends and seasonality
  learningRate: 0.005, // Moderate learning
  warmupSteps: 50, // Quick warmup
  regularizationStrength: 1e-3, // Prevent overfitting
  outlierThreshold: 2.5, // Handle anomalies
  adwinDelta: 0.001, // Detect trend changes
});
```

### 🤖 **Sensor Data Processing**

```typescript
const sensorModel = new MultivariatePolynomialRegression({
  polynomialDegree: 3, // Capture non-linear sensor responses
  learningRate: 0.01, // Fast adaptation
  warmupSteps: 20, // Minimal warmup
  batchSize: 16, // Small batches for streaming
  outlierThreshold: 3.0, // Filter sensor glitches
  adwinDelta: 0.002, // Detect sensor drift
});
```

### 📈 **Financial Modeling**

```typescript
const financeModel = new MultivariatePolynomialRegression({
  polynomialDegree: 2, // Keep it simple
  learningRate: 0.001, // Stable learning
  warmupSteps: 200, // Long warmup
  totalSteps: 50000, // Extended training
  regularizationStrength: 1e-2, // Strong regularization
  outlierThreshold: 2.5, // Handle volatility spikes
  adwinDelta: 0.0005, // Sensitive to regime changes
});
```

### 🏭 **Industrial Process Control**

```typescript
const processModel = new MultivariatePolynomialRegression({
  polynomialDegree: 3, // Complex process dynamics
  learningRate: 0.005, // Moderate adaptation
  beta1: 0.8, // Faster momentum adaptation
  regularizationStrength: 1e-3, // Balance fit and stability
  convergenceThreshold: 1e-5, // High precision
  adwinDelta: 0.001, // Detect process changes
});
```

### 🎮 **Real-time Gaming/Simulation**

```typescript
const realtimeModel = new MultivariatePolynomialRegression({
  polynomialDegree: 2, // Speed over complexity
  learningRate: 0.02, // Fast learning
  warmupSteps: 10, // Minimal warmup
  totalSteps: 1000, // Short horizon
  batchSize: 8, // Small batches
  convergenceThreshold: 1e-3, // Quick convergence
  outlierThreshold: 4.0, // Lenient filtering
});
```

### 🔬 **Scientific Research**

```typescript
const researchModel = new MultivariatePolynomialRegression({
  polynomialDegree: 5, // High flexibility
  learningRate: 0.0005, // Very stable
  warmupSteps: 500, // Long warmup
  totalSteps: 100000, // Extended training
  beta2: 0.9999, // Stable variance
  regularizationStrength: 1e-5, // Minimal bias
  convergenceThreshold: 1e-8, // High precision
});
```

---

## 📐 Mathematical Background

### Polynomial Feature Expansion

For input vector **x** = [x₁, x₂, ..., xₙ] and degree d:

```
φ(x) = [1, x₁, x₂, ..., xₙ, x₁², x₁x₂, ..., x₁ᵈ, ..., xₙᵈ]
```

### Model Equation

```
ŷ = W · φ(x)
```

Where W ∈ ℝ^(m×k), m = output dimension, k = feature count

### Loss Function

```
L = (1/2n) Σ ||y - ŷ||² + (λ/2) ||W||²
    \_____________________/   \________/
           MSE Loss         L2 Regularization
```

### Adam Update Rules

```
m_t = β₁ · m_{t-1} + (1 - β₁) · g_t       # First moment
v_t = β₂ · v_{t-1} + (1 - β₂) · g_t²      # Second moment
m̂_t = m_t / (1 - β₁ᵗ)                     # Bias correction
v̂_t = v_t / (1 - β₂ᵗ)                     # Bias correction
W_t = W_{t-1} - η · m̂_t / (√v̂_t + ε)     # Update
```

---

## ⚡ Performance Tips

### 💾 Memory Optimization

```typescript
// For memory-constrained environments:
const memoryEfficientModel = new MultivariatePolynomialRegression({
  polynomialDegree: 2, // Fewer features
  batchSize: 16, // Smaller batches
});

// Call reset() when done to free memory
model.reset();
```

### 🚀 Speed Optimization

```typescript
// For maximum speed:
const fastModel = new MultivariatePolynomialRegression({
  polynomialDegree: 1, // Minimal features
  warmupSteps: 0, // No warmup
  convergenceThreshold: 1e-3, // Quick convergence
  batchSize: 128, // Larger batches
});
```

### 📊 Accuracy Optimization

```typescript
// For maximum accuracy:
const accurateModel = new MultivariatePolynomialRegression({
  polynomialDegree: 4, // More features
  learningRate: 0.0005, // Stable learning
  warmupSteps: 500, // Long warmup
  totalSteps: 100000, // Extended training
  convergenceThreshold: 1e-8, // High precision
  regularizationStrength: 1e-5, // Minimal bias
});
```

---

## 📄 License

MIT License - feel free to use in personal and commercial projects.

---

<div align="center">

**Made with ❤️ for the machine learning community**

[⬆ Back to Top](#-multivariatepolynomialregression)

</div>
