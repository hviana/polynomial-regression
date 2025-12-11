# 📈 Multivariate Polynomial Regression

A powerful, production-ready TypeScript library for **multivariate polynomial
regression** with **incremental online learning**. Train your model one sample
at a time, perfect for real-time data streams and adaptive systems.

---

## ✨ Features

| Feature                       | Description                                                                                                         |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| 🔄 **Online Learning**        | Incrementally train with new data using the Recursive Least Squares (RLS) algorithm—no need to retrain from scratch |
| 📊 **Multivariate Support**   | Handle multiple input features and multiple output targets simultaneously                                           |
| 🎯 **Polynomial Flexibility** | Configure any polynomial degree to capture complex non-linear relationships                                         |
| 📏 **Auto Normalization**     | Built-in min-max and z-score normalization with automatic statistics tracking                                       |
| 📉 **Confidence Intervals**   | Get prediction uncertainty estimates with configurable confidence levels                                            |
| ⚡ **Memory Efficient**       | Forgetting factor allows the model to adapt and forget old data gracefully                                          |
| 🛡️ **Numerical Stability**    | Regularization prevents matrix singularity issues                                                                   |

---

## 🚀 Quick Start

### Basic Usage

```typescript
import { MultivariatePolynomialRegression } from "jsr:@hviana/polynomial-regression";

// Create a model with default settings
const model = new MultivariatePolynomialRegression({
  polynomialDegree: 2,
  enableNormalization: true,
});

// Train incrementally with data
model.fitOnline({
  xCoordinates: [[1], [2], [3], [4], [5]],
  yCoordinates: [[2.1], [4.0], [5.9], [8.1], [10.0]],
});

// Predict future values
const result = model.predict({ futureSteps: 3 });

console.log(result.predictions);
// → Predicted values with confidence intervals
```

---

## 📖 Comprehensive Guide

### 🏗️ Creating a Model

The `MultivariatePolynomialRegression` class accepts an optional configuration
object:

```typescript
const model = new MultivariatePolynomialRegression({
  polynomialDegree: 3,
  enableNormalization: true,
  normalizationMethod: "z-score",
  forgettingFactor: 0.95,
  initialCovariance: 1000,
  regularization: 1e-6,
  confidenceLevel: 0.95,
});
```

---

## ⚙️ Configuration Parameters

### `polynomialDegree`

> **Type:** `number` &nbsp;|&nbsp; **Default:** `2` &nbsp;|&nbsp; **Range:**
> `≥ 1`

The degree of polynomial features to generate. Higher degrees capture more
complex patterns but risk overfitting.

```typescript
// Linear regression (degree 1): y = a + bx
{
  polynomialDegree: 1;
}

// Quadratic regression (degree 2): y = a + bx + cx²
{
  polynomialDegree: 2;
}

// Cubic regression (degree 3): y = a + bx + cx² + dx³
{
  polynomialDegree: 3;
}
```

| Degree | Use Case                       | Complexity |
| ------ | ------------------------------ | ---------- |
| 1      | Linear trends                  | Low        |
| 2      | Parabolic curves, acceleration | Medium     |
| 3      | S-curves, inflection points    | High       |
| 4+     | Complex oscillations           | Very High  |

> ⚠️ **Warning:** High polynomial degrees with limited data can lead to
> overfitting and numerical instability.

---

### `enableNormalization`

> **Type:** `boolean` &nbsp;|&nbsp; **Default:** `true`

Enables automatic input data normalization. Highly recommended when input
features have different scales.

```typescript
// Enable normalization (recommended)
{
  enableNormalization: true;
}

// Disable if data is already normalized
{
  enableNormalization: false;
}
```

**Why normalize?**

- Prevents numerical overflow with large values
- Improves convergence speed
- Ensures all features contribute equally

---

### `normalizationMethod`

> **Type:** `'none' | 'min-max' | 'z-score'` &nbsp;|&nbsp; **Default:**
> `'min-max'`

The method used to normalize input data.

| Method      | Formula                   | Output Range | Best For                  |
| ----------- | ------------------------- | ------------ | ------------------------- |
| `'min-max'` | `(x - min) / (max - min)` | [0, 1]       | Bounded data              |
| `'z-score'` | `(x - μ) / σ`             | Unbounded    | Gaussian-distributed data |
| `'none'`    | No transformation         | Original     | Pre-normalized data       |

```typescript
// Min-max: scales to [0, 1]
{
  normalizationMethod: "min-max";
}

// Z-score: centers around 0 with unit variance
{
  normalizationMethod: "z-score";
}
```

---

### `forgettingFactor`

> **Type:** `number` &nbsp;|&nbsp; **Default:** `0.99` &nbsp;|&nbsp; **Range:**
> `(0, 1]`

Controls how quickly the model "forgets" old data. This is the λ parameter in
the RLS algorithm.

```
λ = 1.0  → Never forget (standard least squares)
λ = 0.99 → Slight forgetting (recommended default)
λ = 0.95 → Moderate forgetting (faster adaptation)
λ = 0.90 → Aggressive forgetting (highly adaptive)
```

```typescript
// Stable environments (slow-changing data)
{
  forgettingFactor: 0.999;
}

// Dynamic environments (fast-changing data)
{
  forgettingFactor: 0.95;
}
```

> 💡 **Tip:** Lower values make the model more responsive to recent data but
> less stable. Use `0.95-0.99` for most applications.

---

### `initialCovariance`

> **Type:** `number` &nbsp;|&nbsp; **Default:** `1000`

Initial diagonal value for the covariance matrix. Represents initial uncertainty
about the model weights.

```typescript
// High uncertainty (learns faster initially)
{
  initialCovariance: 10000;
}

// Low uncertainty (more conservative updates)
{
  initialCovariance: 100;
}
```

| Value  | Effect                                         |
| ------ | ---------------------------------------------- |
| Higher | Faster initial learning, larger weight updates |
| Lower  | Slower initial learning, more stable           |

---

### `regularization`

> **Type:** `number` &nbsp;|&nbsp; **Default:** `1e-6`

Small value added to the covariance matrix diagonal to prevent numerical
singularity.

```typescript
// Standard regularization
{
  regularization: 1e-6;
}

// Stronger regularization (if experiencing instability)
{
  regularization: 1e-4;
}
```

> 🛡️ **Note:** Only increase this if you encounter numerical errors or warnings.

---

### `confidenceLevel`

> **Type:** `number` &nbsp;|&nbsp; **Default:** `0.95` &nbsp;|&nbsp; **Range:**
> `(0, 1)`

The confidence level for prediction intervals.

```typescript
// 95% confidence interval (standard)
{
  confidenceLevel: 0.95;
}

// 99% confidence interval (wider, more conservative)
{
  confidenceLevel: 0.99;
}

// 90% confidence interval (narrower)
{
  confidenceLevel: 0.90;
}
```

---

## 📚 API Reference

### `fitOnline(params)`

Trains the model incrementally with new data points.

```typescript
model.fitOnline({
  xCoordinates: number[][],  // Input features [samples][features]
  yCoordinates: number[][]   // Output targets [samples][outputs]
});
```

**Example: Single-variable regression**

```typescript
model.fitOnline({
  xCoordinates: [[1], [2], [3]],
  yCoordinates: [[2], [4], [6]],
});
```

**Example: Multivariate regression**

```typescript
// Predicting house price from size and bedrooms
model.fitOnline({
  xCoordinates: [
    [1500, 3], // 1500 sqft, 3 bedrooms
    [2000, 4], // 2000 sqft, 4 bedrooms
    [1200, 2], // 1200 sqft, 2 bedrooms
  ],
  yCoordinates: [
    [300000], // $300k
    [450000], // $450k
    [250000], // $250k
  ],
});
```

**Example: Multiple outputs**

```typescript
// Predicting both price and days-on-market
model.fitOnline({
  xCoordinates: [[1500, 3], [2000, 4]],
  yCoordinates: [
    [300000, 45], // $300k, 45 days
    [450000, 30], // $450k, 30 days
  ],
});
```

---

### `predict(params)`

Generates predictions with confidence intervals.

```typescript
const result = model.predict({
  futureSteps: number,       // Number of extrapolated predictions
  inputPoints?: number[][]   // Optional: specific points to predict
});
```

**Return type: `PredictionResult`**

```typescript
{
  predictions: SinglePrediction[],  // Array of predictions
  confidenceLevel: number,          // Confidence level used
  rSquared: number,                 // R² coefficient (0-1)
  rmse: number,                     // Root Mean Square Error
  sampleCount: number,              // Training samples seen
  isModelReady: boolean             // Sufficient training data?
}
```

**Each `SinglePrediction` contains:**

```typescript
{
  predicted: number[],      // Point predictions
  lowerBound: number[],     // Lower confidence bound
  upperBound: number[],     // Upper confidence bound
  standardError: number[]   // Prediction standard error
}
```

**Example: Extrapolation**

```typescript
// Predict the next 5 time steps
const result = model.predict({ futureSteps: 5 });

result.predictions.forEach((pred, i) => {
  console.log(`Step ${i + 1}:`);
  console.log(`  Predicted: ${pred.predicted[0].toFixed(2)}`);
  console.log(
    `  95% CI: [${pred.lowerBound[0].toFixed(2)}, ${
      pred.upperBound[0].toFixed(2)
    }]`,
  );
});
```

**Example: Specific input points**

```typescript
// Predict for specific input values
const result = model.predict({
  futureSteps: 0, // Ignored when inputPoints provided
  inputPoints: [[1800, 3], [2200, 5]],
});
```

---

### `getModelSummary()`

Returns comprehensive model statistics.

```typescript
const summary = model.getModelSummary();

console.log(summary);
// {
//   isInitialized: true,
//   inputDimension: 2,
//   outputDimension: 1,
//   polynomialDegree: 2,
//   polynomialFeatureCount: 6,
//   sampleCount: 100,
//   rSquared: 0.94,
//   rmse: 12.5,
//   normalizationEnabled: true,
//   normalizationMethod: 'min-max'
// }
```

---

### `getWeights()`

Returns the current model weight matrix.

```typescript
const weights = model.getWeights();
// Shape: [polynomialFeatureCount][outputDimension]
```

---

### `getNormalizationStats()`

Returns the computed normalization statistics.

```typescript
const stats = model.getNormalizationStats();
// {
//   min: [0, 0],
//   max: [100, 50],
//   mean: [50, 25],
//   std: [15.2, 8.7],
//   count: 100
// }
```

---

### `reset()`

Clears all training data and resets the model to its initial state.

```typescript
model.reset();
// Model is now empty and ready for fresh training
```

---

## 🎯 Real-World Examples

### 📈 Time Series Forecasting

```typescript
const model = new MultivariatePolynomialRegression({
  polynomialDegree: 3,
  forgettingFactor: 0.98, // Adapt to recent trends
});

// Simulate streaming data
const dataStream = [
  { time: 1, value: 10 },
  { time: 2, value: 22 },
  { time: 3, value: 35 },
  // ... more data points
];

for (const point of dataStream) {
  model.fitOnline({
    xCoordinates: [[point.time]],
    yCoordinates: [[point.value]],
  });
}

// Forecast next 10 time steps
const forecast = model.predict({ futureSteps: 10 });
```

---

### 🏠 House Price Prediction

```typescript
const model = new MultivariatePolynomialRegression({
  polynomialDegree: 2,
  normalizationMethod: "z-score",
});

// Train with historical data
model.fitOnline({
  xCoordinates: [
    [1500, 3, 1990], // sqft, bedrooms, year built
    [2200, 4, 2005],
    [1800, 3, 2010],
    [3000, 5, 2015],
  ],
  yCoordinates: [
    [320000],
    [485000],
    [410000],
    [650000],
  ],
});

// Predict price for a new house
const prediction = model.predict({
  futureSteps: 0,
  inputPoints: [[2000, 4, 2008]],
});

console.log(
  `Estimated price: $${
    prediction.predictions[0].predicted[0].toLocaleString()
  }`,
);
console.log(
  `Range: $${prediction.predictions[0].lowerBound[0].toLocaleString()} - $${
    prediction.predictions[0].upperBound[0].toLocaleString()
  }`,
);
```

---

### 🌡️ Sensor Data with Multiple Outputs

```typescript
const model = new MultivariatePolynomialRegression({
  polynomialDegree: 2,
  confidenceLevel: 0.99,
});

// Input: time, humidity → Output: temperature, pressure
model.fitOnline({
  xCoordinates: [[0, 45], [1, 48], [2, 52], [3, 55]],
  yCoordinates: [
    [20.1, 1013.2],
    [21.3, 1012.8],
    [22.8, 1012.1],
    [24.2, 1011.5],
  ],
});

const result = model.predict({ futureSteps: 2 });

result.predictions.forEach((pred, i) => {
  console.log(`Hour ${i + 4}:`);
  console.log(
    `  Temperature: ${pred.predicted[0].toFixed(1)}°C ± ${
      pred.standardError[0].toFixed(2)
    }`,
  );
  console.log(
    `  Pressure: ${pred.predicted[1].toFixed(1)} hPa ± ${
      pred.standardError[1].toFixed(2)
    }`,
  );
});
```

---

## 🔬 Understanding the Algorithm

### Recursive Least Squares (RLS)

The RLS algorithm updates model weights incrementally without storing all
historical data:

```
For each new sample (x, y):
  1. φ = polynomial_features(x)
  2. k = P·φ / (λ + φᵀ·P·φ)         ← Gain vector
  3. w = w + k·(y - φᵀ·w)           ← Weight update
  4. P = (P - k·φᵀ·P) / λ           ← Covariance update
```

**Key advantages:**

- O(n²) memory instead of O(n·m) for batch methods
- Constant-time updates regardless of history length
- Natural support for forgetting via λ parameter

---

## 🐛 Troubleshooting

| Problem          | Possible Cause             | Solution                    |
| ---------------- | -------------------------- | --------------------------- |
| Poor predictions | Insufficient training data | Train with more samples     |
| Numerical errors | Large input values         | Enable normalization        |
| Overfitting      | Polynomial degree too high | Reduce `polynomialDegree`   |
| Slow adaptation  | Forgetting factor too high | Decrease `forgettingFactor` |
| Unstable weights | Covariance singularity     | Increase `regularization`   |

---

## 📄 License

MIT © Henrique Emanoel Viana

---

<div align="center">

**Made with ❤️ for the data science community**

Henrique Emanoel Viana

</div>
