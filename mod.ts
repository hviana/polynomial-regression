// ============================================================================
// INTERFACES AND TYPES
// ============================================================================

/**
 * Configuration options for the MultivariatePolynomialRegression model.
 * All parameters have sensible defaults and validation ranges.
 *
 * @interface IConfiguration
 */
interface IConfiguration {
  /** Polynomial degree for feature expansion (default: 2, minimum: 1) */
  polynomialDegree: number;
  /** Whether to normalize input features (default: true) */
  enableNormalization: boolean;
  /** Normalization method to use (default: 'min-max') */
  normalizationMethod: "none" | "min-max" | "z-score";
  /** Initial learning rate for SGD (default: 0.01, range: (0, 1]) */
  learningRate: number;
  /** Learning rate decay factor per sample (default: 0.999, range: (0, 1]) */
  learningRateDecay: number;
  /** Momentum coefficient for velocity updates (default: 0.9, range: [0, 1)) */
  momentum: number;
  /** L2 regularization coefficient (default: 1e-6) */
  regularization: number;
  /** Maximum absolute gradient value for clipping (default: 1.0, range: (0, ∞)) */
  gradientClipValue: number;
  /** Confidence level for prediction intervals (default: 0.95, range: (0, 1)) */
  confidenceLevel: number;
  /** Mini-batch size for SGD updates (default: 1, minimum: 1) */
  batchSize: number;
}

/**
 * Represents a single prediction with confidence bounds.
 * All arrays have length equal to output dimension.
 *
 * @interface SinglePrediction
 */
interface SinglePrediction {
  /** Predicted values for each output dimension */
  predicted: number[];
  /** Lower bounds of confidence interval */
  lowerBound: number[];
  /** Upper bounds of confidence interval */
  upperBound: number[];
  /** Standard errors for each output dimension */
  standardError: number[];
}

/**
 * Complete result from a prediction operation.
 * Contains predictions, model metrics, and status information.
 *
 * @interface PredictionResult
 */
interface PredictionResult {
  /** Array of predictions, one per requested point */
  predictions: SinglePrediction[];
  /** Confidence level used for interval calculation */
  confidenceLevel: number;
  /** Current R-squared (coefficient of determination) */
  rSquared: number;
  /** Current RMSE (root mean squared error) */
  rmse: number;
  /** Total number of training samples seen */
  sampleCount: number;
  /** Whether the model has sufficient data for reliable predictions */
  isModelReady: boolean;
}

/**
 * Summary of current model state and statistics.
 *
 * @interface ModelSummary
 */
interface ModelSummary {
  /** Whether the model has been initialized with data */
  isInitialized: boolean;
  /** Number of input features */
  inputDimension: number;
  /** Number of output targets */
  outputDimension: number;
  /** Polynomial degree used for feature expansion */
  polynomialDegree: number;
  /** Total number of polynomial features (including bias) */
  polynomialFeatureCount: number;
  /** Total number of training samples processed */
  sampleCount: number;
  /** Current R-squared value */
  rSquared: number;
  /** Current RMSE value */
  rmse: number;
  /** Whether normalization is enabled */
  normalizationEnabled: boolean;
  /** Current normalization method */
  normalizationMethod: string;
}

/**
 * Statistics used for input normalization.
 * All arrays have length equal to input dimension.
 *
 * @interface NormalizationStats
 */
interface NormalizationStats {
  /** Minimum values observed for each input feature */
  min: number[];
  /** Maximum values observed for each input feature */
  max: number[];
  /** Running mean for each input feature */
  mean: number[];
  /** Running standard deviation for each input feature */
  std: number[];
  /** Total number of samples used to compute statistics */
  count: number;
}

/**
 * Interface for matrix/vector operations.
 * Provides efficient in-place operations on typed arrays.
 *
 * @interface IMatrixOperations
 */
interface IMatrixOperations {
  /** Compute dot product of two vectors */
  dotProduct(a: Float64Array, b: Float64Array): number;
  /** Multiply vector by scalar in-place */
  scalarMultiplyInPlace(arr: Float64Array, scalar: number): void;
  /** Add source to target in-place */
  addInPlace(target: Float64Array, source: Float64Array): void;
  /** Subtract source from target in-place */
  subtractInPlace(target: Float64Array, source: Float64Array): void;
  /** Clip values to range in-place */
  clipInPlace(arr: Float64Array, min: number, max: number): void;
  /** Copy source array contents to target */
  copyTo(source: Float64Array, target: Float64Array): void;
  /** Get array from pool or create new one */
  acquireArray(size: number): Float64Array;
  /** Return array to pool for reuse */
  releaseArray(arr: Float64Array): void;
}

/**
 * Interface for normalization strategies.
 * Implements Strategy pattern for different normalization methods.
 *
 * @interface INormalizationStrategy
 */
interface INormalizationStrategy {
  /**
   * Normalize a single value
   * @param value - Raw value to normalize
   * @param index - Feature index
   * @param stats - Current normalization statistics
   * @returns Normalized value
   */
  normalize(
    value: number,
    index: number,
    stats: INormalizationStatsInternal,
  ): number;

  /**
   * Denormalize a single value back to original scale
   * @param value - Normalized value
   * @param index - Feature index
   * @param stats - Current normalization statistics
   * @returns Denormalized value
   */
  denormalize(
    value: number,
    index: number,
    stats: INormalizationStatsInternal,
  ): number;
}

/**
 * Internal interface for normalization statistics storage.
 * Uses typed arrays for memory efficiency.
 *
 * @interface INormalizationStatsInternal
 */
interface INormalizationStatsInternal {
  /** Minimum values per dimension */
  min: Float64Array;
  /** Maximum values per dimension */
  max: Float64Array;
  /** Running mean per dimension */
  mean: Float64Array;
  /** M2 accumulator for Welford's algorithm */
  m2: Float64Array;
  /** Standard deviation per dimension */
  std: Float64Array;
  /** Sample count */
  count: number;
}

/**
 * Interface for the normalizer component.
 *
 * @interface INormalizer
 */
interface INormalizer {
  initialize(dimension: number): void;
  updateStatistics(x: Float64Array): void;
  normalizeInPlace(x: Float64Array, output: Float64Array): void;
  getStats(): NormalizationStats;
  isEnabled(): boolean;
  reset(): void;
}

/**
 * Interface for polynomial feature generation.
 *
 * @interface IPolynomialFeatureGenerator
 */
interface IPolynomialFeatureGenerator {
  initialize(inputDimension: number): void;
  generateFeaturesInPlace(
    normalizedInput: Float64Array,
    output: Float64Array,
  ): void;
  getFeatureCount(): number;
  getInputDimension(): number;
  getDegree(): number;
}

/**
 * Interface for weight management.
 *
 * @interface IWeightManager
 */
interface IWeightManager {
  initialize(featureCount: number, outputDimension: number): void;
  getWeight(outputIndex: number, featureIndex: number): number;
  setWeight(outputIndex: number, featureIndex: number, value: number): void;
  getWeightsForOutput(outputIndex: number, buffer: Float64Array): void;
  updateWeight(outputIndex: number, featureIndex: number, delta: number): void;
  getWeights(): number[][];
  isInitialized(): boolean;
  reset(): void;
}

/**
 * Interface for gradient computation and momentum.
 *
 * @interface IGradientManager
 */
interface IGradientManager {
  initialize(featureCount: number, outputDimension: number): void;
  computeGradient(
    features: Float64Array,
    error: number,
    weights: Float64Array,
    regularization: number,
    outputIndex: number,
  ): void;
  updateVelocityAndGetDelta(
    learningRate: number,
    outputIndex: number,
  ): Float64Array;
  reset(): void;
}

/**
 * Interface for prediction computation.
 *
 * @interface IPredictionEngine
 */
interface IPredictionEngine {
  initialize(featureCount: number, outputDimension: number): void;
  predict(
    features: Float64Array,
    weightManager: IWeightManager,
    output: Float64Array,
  ): void;
  getCriticalValue(sampleCount: number, confidenceLevel: number): number;
  calculateStandardError(
    features: Float64Array,
    residualVariance: number,
    sampleCount: number,
  ): number;
  reset(): void;
}

/**
 * Interface for statistics tracking.
 *
 * @interface IStatisticsTracker
 */
interface IStatisticsTracker {
  initialize(outputDimension: number): void;
  update(actual: Float64Array, predicted: Float64Array): void;
  getRSquared(): number;
  getRMSE(): number;
  getResidualVariance(numParameters: number): number;
  getSampleCount(): number;
  reset(): void;
}

// ============================================================================
// MATRIX OPERATIONS CLASS
// ============================================================================

/**
 * Efficient matrix and vector operations using typed arrays.
 * Implements object pooling to minimize garbage collection.
 * All operations are optimized for CPU cache efficiency.
 *
 * @class MatrixOperations
 * @implements {IMatrixOperations}
 *
 * @example
 * const matOps = new MatrixOperations();
 * const a = new Float64Array([1, 2, 3]);
 * const b = new Float64Array([4, 5, 6]);
 * const dot = matOps.dotProduct(a, b); // Returns 32
 */
class MatrixOperations implements IMatrixOperations {
  /** Object pool for array reuse, keyed by size */
  private readonly arrayPool: Map<number, Float64Array[]>;
  /** Maximum pool size per array length */
  private readonly maxPoolSize: number;

  /**
   * Creates a new MatrixOperations instance.
   *
   * @param maxPoolSize - Maximum number of arrays to keep per size (default: 20)
   */
  constructor(maxPoolSize: number = 20) {
    this.arrayPool = new Map();
    this.maxPoolSize = maxPoolSize;
  }

  /**
   * Acquire a Float64Array from the pool or create a new one.
   * Arrays from pool are zeroed before return.
   *
   * @param size - Required array length
   * @returns Float64Array of requested size
   *
   * Time complexity: O(n) for zeroing, O(1) amortized allocation
   * Space complexity: O(1) when reusing, O(n) for new allocation
   */
  acquireArray(size: number): Float64Array {
    const pool = this.arrayPool.get(size);
    if (pool !== undefined && pool.length > 0) {
      const arr = pool.pop()!;
      // Zero out the array before returning
      const len = arr.length;
      for (let i = 0; i < len; i++) {
        arr[i] = 0;
      }
      return arr;
    }
    return new Float64Array(size);
  }

  /**
   * Return a Float64Array to the pool for later reuse.
   * Arrays beyond pool capacity are discarded for GC.
   *
   * @param arr - Array to return to pool
   *
   * Time complexity: O(1)
   * Space complexity: O(1)
   */
  releaseArray(arr: Float64Array): void {
    const size = arr.length;
    let pool = this.arrayPool.get(size);
    if (pool === undefined) {
      pool = [];
      this.arrayPool.set(size, pool);
    }
    if (pool.length < this.maxPoolSize) {
      pool.push(arr);
    }
    // Otherwise let it be garbage collected
  }

  /**
   * Compute the dot product of two vectors.
   * Uses loop unrolling for better performance.
   *
   * Formula: result = Σ(aᵢ × bᵢ) for i = 0 to n-1
   *
   * @param a - First vector
   * @param b - Second vector (must be same length as a)
   * @returns Scalar dot product
   *
   * Time complexity: O(n)
   * Space complexity: O(1)
   *
   * @example
   * const a = new Float64Array([1, 2, 3]);
   * const b = new Float64Array([4, 5, 6]);
   * const result = matOps.dotProduct(a, b); // 1*4 + 2*5 + 3*6 = 32
   */
  dotProduct(a: Float64Array, b: Float64Array): number {
    let result = 0;
    const len = a.length;

    // Process 4 elements at a time for better CPU pipelining
    const unrolledLen = len - (len % 4);
    let i = 0;

    for (; i < unrolledLen; i += 4) {
      result += a[i] * b[i] +
        a[i + 1] * b[i + 1] +
        a[i + 2] * b[i + 2] +
        a[i + 3] * b[i + 3];
    }

    // Handle remaining elements
    for (; i < len; i++) {
      result += a[i] * b[i];
    }

    return result;
  }

  /**
   * Multiply all elements of an array by a scalar in-place.
   *
   * Formula: arrᵢ = arrᵢ × scalar for all i
   *
   * @param arr - Array to modify
   * @param scalar - Multiplication factor
   *
   * Time complexity: O(n)
   * Space complexity: O(1)
   */
  scalarMultiplyInPlace(arr: Float64Array, scalar: number): void {
    const len = arr.length;
    for (let i = 0; i < len; i++) {
      arr[i] *= scalar;
    }
  }

  /**
   * Add source vector to target vector element-wise in-place.
   *
   * Formula: targetᵢ = targetᵢ + sourceᵢ for all i
   *
   * @param target - Target array (will be modified)
   * @param source - Source array to add
   *
   * Time complexity: O(n)
   * Space complexity: O(1)
   */
  addInPlace(target: Float64Array, source: Float64Array): void {
    const len = target.length;
    for (let i = 0; i < len; i++) {
      target[i] += source[i];
    }
  }

  /**
   * Subtract source vector from target vector element-wise in-place.
   *
   * Formula: targetᵢ = targetᵢ - sourceᵢ for all i
   *
   * @param target - Target array (will be modified)
   * @param source - Source array to subtract
   *
   * Time complexity: O(n)
   * Space complexity: O(1)
   */
  subtractInPlace(target: Float64Array, source: Float64Array): void {
    const len = target.length;
    for (let i = 0; i < len; i++) {
      target[i] -= source[i];
    }
  }

  /**
   * Clip all values in array to specified range in-place.
   *
   * Formula: arrᵢ = max(min, min(max, arrᵢ)) for all i
   *
   * @param arr - Array to clip
   * @param min - Minimum allowed value
   * @param max - Maximum allowed value
   *
   * Time complexity: O(n)
   * Space complexity: O(1)
   */
  clipInPlace(arr: Float64Array, min: number, max: number): void {
    const len = arr.length;
    for (let i = 0; i < len; i++) {
      const val = arr[i];
      if (val < min) {
        arr[i] = min;
      } else if (val > max) {
        arr[i] = max;
      }
    }
  }

  /**
   * Copy contents from source array to target array.
   *
   * @param source - Source array to copy from
   * @param target - Target array to copy to
   *
   * Time complexity: O(n)
   * Space complexity: O(1)
   */
  copyTo(source: Float64Array, target: Float64Array): void {
    const len = source.length;
    for (let i = 0; i < len; i++) {
      target[i] = source[i];
    }
  }

  /**
   * Fill array with zeros in-place.
   *
   * @param arr - Array to zero
   *
   * Time complexity: O(n)
   * Space complexity: O(1)
   */
  zeroFill(arr: Float64Array): void {
    const len = arr.length;
    for (let i = 0; i < len; i++) {
      arr[i] = 0;
    }
  }

  /**
   * Clear the array pool to free memory.
   */
  clearPool(): void {
    this.arrayPool.clear();
  }
}

// ============================================================================
// NORMALIZATION STRATEGIES
// ============================================================================

/**
 * No-op normalization strategy that passes values through unchanged.
 *
 * @class NoNormalizationStrategy
 * @implements {INormalizationStrategy}
 */
class NoNormalizationStrategy implements INormalizationStrategy {
  /**
   * Returns value unchanged.
   *
   * @param value - Input value
   * @returns Same value
   *
   * Time complexity: O(1)
   * Space complexity: O(1)
   */
  normalize(value: number): number {
    return value;
  }

  /**
   * Returns value unchanged.
   *
   * @param value - Input value
   * @returns Same value
   *
   * Time complexity: O(1)
   * Space complexity: O(1)
   */
  denormalize(value: number): number {
    return value;
  }
}

/**
 * Min-Max normalization strategy.
 * Scales values to [0, 1] range based on observed min and max.
 *
 * Formula: normalized = (x - min) / (max - min)
 *
 * @class MinMaxNormalizationStrategy
 * @implements {INormalizationStrategy}
 *
 * @example
 * // If min=0, max=100, then value 50 normalizes to 0.5
 */
class MinMaxNormalizationStrategy implements INormalizationStrategy {
  /** Minimum range to prevent division by zero */
  private readonly minRange: number = 1e-10;

  /**
   * Normalize value using min-max scaling.
   *
   * Formula: (x - min) / (max - min)
   * Output range: [0, 1]
   *
   * @param value - Raw value to normalize
   * @param index - Feature index for statistics lookup
   * @param stats - Current normalization statistics
   * @returns Normalized value in [0, 1]
   *
   * Time complexity: O(1)
   * Space complexity: O(1)
   */
  normalize(
    value: number,
    index: number,
    stats: INormalizationStatsInternal,
  ): number {
    const range = stats.max[index] - stats.min[index];
    if (range < this.minRange) {
      return 0.5; // Return midpoint when range is effectively zero
    }
    return (value - stats.min[index]) / range;
  }

  /**
   * Denormalize value back to original scale.
   *
   * Formula: x = normalized × (max - min) + min
   *
   * @param value - Normalized value
   * @param index - Feature index for statistics lookup
   * @param stats - Current normalization statistics
   * @returns Value in original scale
   *
   * Time complexity: O(1)
   * Space complexity: O(1)
   */
  denormalize(
    value: number,
    index: number,
    stats: INormalizationStatsInternal,
  ): number {
    const range = stats.max[index] - stats.min[index];
    return value * range + stats.min[index];
  }
}

/**
 * Z-Score (standard score) normalization strategy.
 * Centers data around mean with unit standard deviation.
 *
 * Formula: normalized = (x - mean) / std
 *
 * @class ZScoreNormalizationStrategy
 * @implements {INormalizationStrategy}
 *
 * @example
 * // If mean=50, std=10, then value 60 normalizes to 1.0
 */
class ZScoreNormalizationStrategy implements INormalizationStrategy {
  /** Minimum std to prevent division by zero */
  private readonly minStd: number = 1e-10;

  /**
   * Normalize value using z-score standardization.
   *
   * Formula: (x - μ) / σ
   * Output: values centered at 0 with std ≈ 1
   *
   * @param value - Raw value to normalize
   * @param index - Feature index for statistics lookup
   * @param stats - Current normalization statistics
   * @returns Z-score normalized value
   *
   * Time complexity: O(1)
   * Space complexity: O(1)
   */
  normalize(
    value: number,
    index: number,
    stats: INormalizationStatsInternal,
  ): number {
    if (stats.std[index] < this.minStd) {
      return 0; // Return zero when std is effectively zero
    }
    return (value - stats.mean[index]) / stats.std[index];
  }

  /**
   * Denormalize value back to original scale.
   *
   * Formula: x = normalized × σ + μ
   *
   * @param value - Normalized value
   * @param index - Feature index for statistics lookup
   * @param stats - Current normalization statistics
   * @returns Value in original scale
   *
   * Time complexity: O(1)
   * Space complexity: O(1)
   */
  denormalize(
    value: number,
    index: number,
    stats: INormalizationStatsInternal,
  ): number {
    return value * stats.std[index] + stats.mean[index];
  }
}

// ============================================================================
// NORMALIZER CLASS
// ============================================================================

/**
 * Handles incremental normalization of input data.
 * Uses Welford's online algorithm for numerical stability.
 * Does not store individual data points - computes statistics incrementally.
 *
 * @class Normalizer
 * @implements {INormalizer}
 *
 * @example
 * const normalizer = new Normalizer(true, 'min-max');
 * normalizer.initialize(3); // 3 input features
 *
 * const input = new Float64Array([1.0, 2.0, 3.0]);
 * normalizer.updateStatistics(input);
 *
 * const output = new Float64Array(3);
 * normalizer.normalizeInPlace(input, output);
 */
class Normalizer implements INormalizer {
  /** Internal statistics storage */
  private stats: INormalizationStatsInternal | null = null;
  /** Current normalization strategy */
  private readonly strategy: INormalizationStrategy;
  /** Whether normalization is enabled */
  private readonly enabled: boolean;
  /** Input dimension */
  private dimension: number = 0;
  /** Value clamp bounds for normalized output */
  private readonly clampMin: number = -10;
  private readonly clampMax: number = 10;

  /**
   * Creates a new Normalizer instance.
   *
   * @param enabled - Whether normalization is enabled
   * @param method - Normalization method ('none', 'min-max', or 'z-score')
   */
  constructor(enabled: boolean, method: "none" | "min-max" | "z-score") {
    this.enabled = enabled;
    this.strategy = this.createStrategy(method);
  }

  /**
   * Factory method to create normalization strategy.
   *
   * @param method - Strategy name
   * @returns Appropriate strategy instance
   */
  private createStrategy(method: string): INormalizationStrategy {
    switch (method) {
      case "z-score":
        return new ZScoreNormalizationStrategy();
      case "min-max":
        return new MinMaxNormalizationStrategy();
      default:
        return new NoNormalizationStrategy();
    }
  }

  /**
   * Initialize normalizer for given input dimension.
   * Allocates statistics arrays and sets initial values.
   *
   * @param dimension - Number of input features
   *
   * Time complexity: O(dimension)
   * Space complexity: O(dimension)
   */
  initialize(dimension: number): void {
    this.dimension = dimension;

    const min = new Float64Array(dimension);
    const max = new Float64Array(dimension);
    const mean = new Float64Array(dimension);
    const m2 = new Float64Array(dimension);
    const std = new Float64Array(dimension);

    // Initialize min to +Infinity, max to -Infinity
    for (let i = 0; i < dimension; i++) {
      min[i] = Number.MAX_VALUE;
      max[i] = -Number.MAX_VALUE;
      std[i] = 1; // Default std to prevent division by zero
    }

    this.stats = { min, max, mean, m2, std, count: 0 };
  }

  /**
   * Update running statistics with new data point.
   * Uses Welford's online algorithm for numerically stable mean/variance.
   *
   * Welford's Algorithm:
   * - M₁ = x₁
   * - Mₖ = M_{k-1} + (xₖ - M_{k-1}) / k
   * - S₁ = 0
   * - Sₖ = S_{k-1} + (xₖ - M_{k-1}) × (xₖ - Mₖ)
   * - variance = Sₖ / (k - 1)
   *
   * @param x - Input vector to incorporate
   *
   * Time complexity: O(dimension)
   * Space complexity: O(1)
   */
  updateStatistics(x: Float64Array): void {
    if (this.stats === null || !this.enabled) {
      return;
    }

    this.stats.count++;
    const n = this.stats.count;

    for (let i = 0; i < this.dimension; i++) {
      const value = x[i];

      // Update min/max
      if (value < this.stats.min[i]) {
        this.stats.min[i] = value;
      }
      if (value > this.stats.max[i]) {
        this.stats.max[i] = value;
      }

      // Welford's online algorithm for mean and variance
      const delta = value - this.stats.mean[i];
      this.stats.mean[i] += delta / n;
      const delta2 = value - this.stats.mean[i];
      this.stats.m2[i] += delta * delta2;

      // Update standard deviation
      if (n > 1) {
        this.stats.std[i] = Math.sqrt(this.stats.m2[i] / (n - 1));
      }
    }
  }

  /**
   * Normalize input vector in-place to output buffer.
   * Applies clamping to prevent extreme outliers.
   *
   * @param x - Input vector
   * @param output - Output buffer for normalized values
   *
   * Time complexity: O(dimension)
   * Space complexity: O(1)
   */
  normalizeInPlace(x: Float64Array, output: Float64Array): void {
    if (!this.enabled || this.stats === null) {
      // Copy directly if normalization disabled
      for (let i = 0; i < this.dimension; i++) {
        output[i] = x[i];
      }
      return;
    }

    for (let i = 0; i < this.dimension; i++) {
      let normalized = this.strategy.normalize(x[i], i, this.stats);

      // Clamp to prevent extreme outliers from destabilizing training
      if (normalized < this.clampMin) {
        normalized = this.clampMin;
      } else if (normalized > this.clampMax) {
        normalized = this.clampMax;
      }

      output[i] = normalized;
    }
  }

  /**
   * Get current normalization statistics as plain objects.
   *
   * @returns NormalizationStats object
   *
   * Time complexity: O(dimension)
   * Space complexity: O(dimension)
   */
  getStats(): NormalizationStats {
    if (this.stats === null) {
      return { min: [], max: [], mean: [], std: [], count: 0 };
    }
    return {
      min: Array.from(this.stats.min),
      max: Array.from(this.stats.max),
      mean: Array.from(this.stats.mean),
      std: Array.from(this.stats.std),
      count: this.stats.count,
    };
  }

  /**
   * Check if normalization is enabled.
   *
   * @returns true if normalization is enabled
   */
  isEnabled(): boolean {
    return this.enabled;
  }

  /**
   * Reset statistics to initial state.
   *
   * Time complexity: O(dimension)
   * Space complexity: O(1)
   */
  reset(): void {
    if (this.dimension > 0) {
      this.initialize(this.dimension);
    }
  }
}

// ============================================================================
// POLYNOMIAL FEATURE GENERATOR
// ============================================================================

/**
 * Generates polynomial features from input vectors.
 * Precomputes exponent combinations for efficiency.
 *
 * For degree d and input [x₁, x₂], generates all monomials of total degree ≤ d:
 * degree=2: [1, x₁, x₂, x₁², x₁x₂, x₂²]
 *
 * Feature count formula: C(n+d, d) = (n+d)! / (d! × n!)
 *
 * @class PolynomialFeatureGenerator
 * @implements {IPolynomialFeatureGenerator}
 *
 * @example
 * const generator = new PolynomialFeatureGenerator(2);
 * generator.initialize(2); // 2 input features
 *
 * const input = new Float64Array([2.0, 3.0]);
 * const features = new Float64Array(generator.getFeatureCount());
 * generator.generateFeaturesInPlace(input, features);
 * // features = [1, 2, 3, 4, 6, 9] = [1, x₁, x₂, x₁², x₁x₂, x₂²]
 */
class PolynomialFeatureGenerator implements IPolynomialFeatureGenerator {
  /** Polynomial degree */
  private readonly degree: number;
  /** Input dimension */
  private inputDimension: number = 0;
  /** Total number of polynomial features */
  private featureCount: number = 0;
  /** Precomputed exponent matrix [feature_index][input_index] = exponent */
  private exponentMatrix: Uint8Array[] | null = null;

  /**
   * Creates a new PolynomialFeatureGenerator.
   *
   * @param degree - Maximum polynomial degree (≥ 1)
   * @throws Error if degree < 1
   */
  constructor(degree: number) {
    if (degree < 1) {
      throw new Error("Polynomial degree must be at least 1");
    }
    this.degree = degree;
  }

  /**
   * Calculate number of polynomial features using combinatorics.
   *
   * Formula: C(n+d, d) where n=inputDimension, d=degree
   * This equals the number of monomials of total degree ≤ d in n variables.
   *
   * @param inputDimension - Number of input features
   * @returns Number of polynomial features
   *
   * Time complexity: O(degree)
   * Space complexity: O(1)
   */
  private calculateFeatureCount(inputDimension: number): number {
    // C(n+d, d) = (n+d)! / (d! × n!)
    // Computed iteratively: C(n+d, d) = Π(n+i)/i for i=1 to d
    let result = 1;
    for (let i = 1; i <= this.degree; i++) {
      result = (result * (inputDimension + i)) / i;
    }
    return Math.round(result);
  }

  /**
   * Initialize generator for given input dimension.
   * Precomputes all exponent combinations for fast feature generation.
   *
   * @param inputDimension - Number of input features
   *
   * Time complexity: O(featureCount × inputDimension)
   * Space complexity: O(featureCount × inputDimension)
   */
  initialize(inputDimension: number): void {
    this.inputDimension = inputDimension;
    this.featureCount = this.calculateFeatureCount(inputDimension);

    // Precompute all exponent combinations
    this.exponentMatrix = [];
    const currentExponents = new Uint8Array(inputDimension);
    this.generateExponentCombinations(currentExponents, 0, this.degree);
  }

  /**
   * Recursively generate all exponent combinations.
   * Uses DFS to enumerate all valid monomial combinations.
   *
   * @param current - Current exponent array being built
   * @param index - Current dimension index
   * @param remaining - Remaining degree budget
   */
  private generateExponentCombinations(
    current: Uint8Array,
    index: number,
    remaining: number,
  ): void {
    if (index === this.inputDimension) {
      // Store a copy of the current exponent combination
      const copy = new Uint8Array(this.inputDimension);
      for (let i = 0; i < this.inputDimension; i++) {
        copy[i] = current[i];
      }
      this.exponentMatrix!.push(copy);
      return;
    }

    // Try all possible exponents for this dimension
    for (let exp = 0; exp <= remaining; exp++) {
      current[index] = exp;
      this.generateExponentCombinations(current, index + 1, remaining - exp);
    }
    current[index] = 0; // Reset for backtracking
  }

  /**
   * Generate polynomial features from normalized input.
   * Uses precomputed exponents for efficiency.
   *
   * @param normalizedInput - Normalized input vector
   * @param output - Output buffer for polynomial features
   *
   * Time complexity: O(featureCount × inputDimension)
   * Space complexity: O(1) - uses preallocated buffers
   *
   * @example
   * // For input [x₁, x₂] with degree 2:
   * // Output: [1, x₁, x₂, x₁², x₁x₂, x₂²]
   */
  generateFeaturesInPlace(
    normalizedInput: Float64Array,
    output: Float64Array,
  ): void {
    if (this.exponentMatrix === null) {
      return;
    }

    const numFeatures = this.exponentMatrix.length;
    const numDims = this.inputDimension;

    for (let f = 0; f < numFeatures; f++) {
      const exponents = this.exponentMatrix[f];
      let feature = 1.0;

      // Compute x₁^e₁ × x₂^e₂ × ... × xₙ^eₙ
      for (let d = 0; d < numDims; d++) {
        const exp = exponents[d];
        if (exp > 0) {
          // Inline power computation (faster than Math.pow for small exponents)
          const base = normalizedInput[d];
          let power = 1.0;
          for (let e = 0; e < exp; e++) {
            power *= base;
          }
          feature *= power;
        }
      }

      output[f] = feature;
    }
  }

  /**
   * Get the total number of polynomial features.
   *
   * @returns Feature count
   */
  getFeatureCount(): number {
    return this.featureCount;
  }

  /**
   * Get the input dimension.
   *
   * @returns Input dimension
   */
  getInputDimension(): number {
    return this.inputDimension;
  }

  /**
   * Get the polynomial degree.
   *
   * @returns Degree
   */
  getDegree(): number {
    return this.degree;
  }
}

// ============================================================================
// WEIGHT MANAGER
// ============================================================================

/**
 * Manages model weights with efficient flat storage.
 * Weights stored as contiguous Float64Array for cache efficiency.
 *
 * Storage layout: [output₀_weights..., output₁_weights..., ...]
 *
 * @class WeightManager
 * @implements {IWeightManager}
 */
class WeightManager implements IWeightManager {
  /** Flat weight storage */
  private weights: Float64Array | null = null;
  /** Number of polynomial features */
  private featureCount: number = 0;
  /** Number of output dimensions */
  private outputDimension: number = 0;
  /** Initialization flag */
  private initialized: boolean = false;

  /**
   * Initialize weights for given dimensions.
   * Uses Xavier/Glorot initialization for better convergence.
   *
   * Xavier initialization: w ~ U(-√(6/(n_in+n_out)), √(6/(n_in+n_out)))
   *
   * @param featureCount - Number of polynomial features (inputs)
   * @param outputDimension - Number of output targets
   *
   * Time complexity: O(featureCount × outputDimension)
   * Space complexity: O(featureCount × outputDimension)
   */
  initialize(featureCount: number, outputDimension: number): void {
    this.featureCount = featureCount;
    this.outputDimension = outputDimension;

    const totalSize = featureCount * outputDimension;
    this.weights = new Float64Array(totalSize);

    // Xavier/Glorot initialization for better gradient flow
    const scale = Math.sqrt(6.0 / (featureCount + outputDimension));

    for (let i = 0; i < totalSize; i++) {
      // Uniform distribution in [-scale, scale]
      this.weights[i] = (Math.random() * 2 - 1) * scale;
    }

    this.initialized = true;
  }

  /**
   * Get weight at specific position.
   *
   * @param outputIndex - Output dimension index
   * @param featureIndex - Feature index
   * @returns Weight value
   *
   * Time complexity: O(1)
   * Space complexity: O(1)
   */
  getWeight(outputIndex: number, featureIndex: number): number {
    if (this.weights === null) {
      return 0;
    }
    return this.weights[outputIndex * this.featureCount + featureIndex];
  }

  /**
   * Set weight at specific position.
   *
   * @param outputIndex - Output dimension index
   * @param featureIndex - Feature index
   * @param value - New weight value
   *
   * Time complexity: O(1)
   * Space complexity: O(1)
   */
  setWeight(outputIndex: number, featureIndex: number, value: number): void {
    if (this.weights === null) {
      return;
    }
    this.weights[outputIndex * this.featureCount + featureIndex] = value;
  }

  /**
   * Copy weights for a specific output dimension to buffer.
   *
   * @param outputIndex - Output dimension index
   * @param buffer - Destination buffer
   *
   * Time complexity: O(featureCount)
   * Space complexity: O(1)
   */
  getWeightsForOutput(outputIndex: number, buffer: Float64Array): void {
    if (this.weights === null) {
      return;
    }
    const offset = outputIndex * this.featureCount;
    for (let i = 0; i < this.featureCount; i++) {
      buffer[i] = this.weights[offset + i];
    }
  }

  /**
   * Update weight by subtracting delta.
   * w = w - delta
   *
   * @param outputIndex - Output dimension index
   * @param featureIndex - Feature index
   * @param delta - Value to subtract from weight
   *
   * Time complexity: O(1)
   * Space complexity: O(1)
   */
  updateWeight(outputIndex: number, featureIndex: number, delta: number): void {
    if (this.weights === null) {
      return;
    }
    this.weights[outputIndex * this.featureCount + featureIndex] -= delta;
  }

  /**
   * Get all weights as 2D array.
   *
   * @returns Weights array [outputDim][featureCount]
   *
   * Time complexity: O(outputDimension × featureCount)
   * Space complexity: O(outputDimension × featureCount)
   */
  getWeights(): number[][] {
    if (this.weights === null) {
      return [];
    }

    const result: number[][] = [];
    for (let o = 0; o < this.outputDimension; o++) {
      const row: number[] = [];
      const offset = o * this.featureCount;
      for (let f = 0; f < this.featureCount; f++) {
        row.push(this.weights[offset + f]);
      }
      result.push(row);
    }
    return result;
  }

  /**
   * Check if weights have been initialized.
   *
   * @returns true if initialized
   */
  isInitialized(): boolean {
    return this.initialized;
  }

  /**
   * Get number of features.
   *
   * @returns Feature count
   */
  getFeatureCount(): number {
    return this.featureCount;
  }

  /**
   * Get output dimension.
   *
   * @returns Output dimension
   */
  getOutputDimension(): number {
    return this.outputDimension;
  }

  /**
   * Reset weights to uninitialized state.
   *
   * Time complexity: O(1)
   * Space complexity: O(1)
   */
  reset(): void {
    this.weights = null;
    this.initialized = false;
    this.featureCount = 0;
    this.outputDimension = 0;
  }
}

// ============================================================================
// GRADIENT MANAGER
// ============================================================================

/**
 * Manages gradient computation and momentum updates for SGD.
 * Implements momentum-based gradient descent with gradient clipping.
 *
 * Momentum update formula:
 * v = μ × v + η × g
 * w = w - v
 *
 * @class GradientManager
 * @implements {IGradientManager}
 */
class GradientManager implements IGradientManager {
  /** Velocity storage for momentum */
  private velocity: Float64Array | null = null;
  /** Gradient buffer for current computation */
  private gradient: Float64Array | null = null;
  /** Number of polynomial features */
  private featureCount: number = 0;
  /** Number of output dimensions */
  private outputDimension: number = 0;
  /** Momentum coefficient */
  private readonly momentum: number;
  /** Gradient clipping threshold */
  private readonly clipValue: number;

  /**
   * Creates a new GradientManager.
   *
   * @param momentum - Momentum coefficient [0, 1)
   * @param clipValue - Maximum gradient magnitude
   */
  constructor(momentum: number, clipValue: number) {
    this.momentum = momentum;
    this.clipValue = clipValue;
  }

  /**
   * Initialize gradient buffers.
   *
   * @param featureCount - Number of features
   * @param outputDimension - Number of outputs
   *
   * Time complexity: O(featureCount × outputDimension)
   * Space complexity: O(featureCount × outputDimension)
   */
  initialize(featureCount: number, outputDimension: number): void {
    this.featureCount = featureCount;
    this.outputDimension = outputDimension;

    const totalSize = featureCount * outputDimension;
    this.velocity = new Float64Array(totalSize);
    this.gradient = new Float64Array(featureCount);
  }

  /**
   * Compute gradient for a single output dimension.
   * Applies L2 regularization and gradient clipping.
   *
   * Gradient formula:
   * g = -error × features + λ × weights
   *
   * The negative error is used because we want to minimize loss:
   * ∂L/∂w = -error × x + λw (for MSE loss with L2 regularization)
   *
   * @param features - Polynomial features (φ)
   * @param error - Prediction error (y - ŷ)
   * @param weights - Current weights for this output
   * @param regularization - L2 regularization coefficient (λ)
   * @param outputIndex - Output dimension index
   *
   * Time complexity: O(featureCount)
   * Space complexity: O(1)
   */
  computeGradient(
    features: Float64Array,
    error: number,
    weights: Float64Array,
    regularization: number,
    outputIndex: number,
  ): void {
    if (this.gradient === null) {
      return;
    }

    const len = this.featureCount;
    const negError = -error;
    const clipMin = -this.clipValue;
    const clipMax = this.clipValue;

    for (let i = 0; i < len; i++) {
      // g = -e×φ + λ×w
      // Negative error because gradient of (y - ŷ)² w.r.t. w is -2(y-ŷ)×x
      let grad = negError * features[i] + regularization * weights[i];

      // Apply gradient clipping to prevent exploding gradients
      if (grad < clipMin) {
        grad = clipMin;
      } else if (grad > clipMax) {
        grad = clipMax;
      }

      this.gradient[i] = grad;
    }
  }

  /**
   * Update velocity with momentum and return weight deltas.
   *
   * Momentum update:
   * v = μ × v + η × g
   *
   * @param learningRate - Current learning rate (η)
   * @param outputIndex - Output dimension index
   * @returns View into velocity array for this output (deltas to subtract)
   *
   * Time complexity: O(featureCount)
   * Space complexity: O(1)
   */
  updateVelocityAndGetDelta(
    learningRate: number,
    outputIndex: number,
  ): Float64Array {
    if (this.velocity === null || this.gradient === null) {
      return new Float64Array(0);
    }

    const offset = outputIndex * this.featureCount;
    const len = this.featureCount;

    for (let i = 0; i < len; i++) {
      // v = μ×v + η×g
      this.velocity[offset + i] = this.momentum * this.velocity[offset + i] +
        learningRate * this.gradient[i];
    }

    // Return subarray view (no copy)
    return this.velocity.subarray(offset, offset + len);
  }

  /**
   * Reset velocity and gradient to zero.
   *
   * Time complexity: O(n)
   * Space complexity: O(1)
   */
  reset(): void {
    if (this.velocity !== null) {
      const len = this.velocity.length;
      for (let i = 0; i < len; i++) {
        this.velocity[i] = 0;
      }
    }
    if (this.gradient !== null) {
      const len = this.gradient.length;
      for (let i = 0; i < len; i++) {
        this.gradient[i] = 0;
      }
    }
  }
}

// ============================================================================
// STATISTICS TRACKER
// ============================================================================

/**
 * Tracks model statistics incrementally using Welford's algorithm.
 * Computes R², RMSE, and residual variance without storing data.
 *
 * R² = 1 - SS_res / SS_tot
 * RMSE = √(Σ(y - ŷ)² / n)
 *
 * @class StatisticsTracker
 * @implements {IStatisticsTracker}
 */
class StatisticsTracker implements IStatisticsTracker {
  /** Total number of samples processed */
  private sampleCount: number = 0;
  /** Sum of squared residuals: Σ(y - ŷ)² */
  private ssRes: number = 0;
  /** Sum of squared total: Σ(y - ȳ)² */
  private ssTot: number = 0;
  /** Sum of squared errors for RMSE */
  private sumSquaredError: number = 0;
  /** Running mean of y values per dimension */
  private yMean: Float64Array | null = null;
  /** M2 accumulator for y variance (Welford's algorithm) */
  private yM2: Float64Array | null = null;
  /** Output dimension */
  private outputDimension: number = 0;

  /**
   * Initialize tracker for given output dimension.
   *
   * @param outputDimension - Number of output targets
   *
   * Time complexity: O(outputDimension)
   * Space complexity: O(outputDimension)
   */
  initialize(outputDimension: number): void {
    this.outputDimension = outputDimension;
    this.yMean = new Float64Array(outputDimension);
    this.yM2 = new Float64Array(outputDimension);
    this.resetCounters();
  }

  /**
   * Reset all counters to initial state.
   */
  private resetCounters(): void {
    this.sampleCount = 0;
    this.ssRes = 0;
    this.ssTot = 0;
    this.sumSquaredError = 0;
    if (this.yMean !== null) {
      for (let i = 0; i < this.outputDimension; i++) {
        this.yMean[i] = 0;
      }
    }
    if (this.yM2 !== null) {
      for (let i = 0; i < this.outputDimension; i++) {
        this.yM2[i] = 0;
      }
    }
  }

  /**
   * Update statistics with new prediction.
   * Uses Welford's algorithm for numerically stable variance calculation.
   *
   * @param actual - Actual target values
   * @param predicted - Predicted values
   *
   * Time complexity: O(outputDimension)
   * Space complexity: O(1)
   */
  update(actual: Float64Array, predicted: Float64Array): void {
    if (this.yMean === null || this.yM2 === null) {
      return;
    }

    this.sampleCount++;
    const n = this.sampleCount;

    // Recalculate SS_tot using M2 accumulators
    this.ssTot = 0;

    for (let i = 0; i < this.outputDimension; i++) {
      const y = actual[i];
      const yHat = predicted[i];
      const error = y - yHat;

      // Update sum of squared residuals: SS_res = Σ(y - ŷ)²
      this.ssRes += error * error;

      // Track for RMSE calculation
      this.sumSquaredError += error * error;

      // Welford's algorithm for running mean and M2
      const delta = y - this.yMean[i];
      this.yMean[i] += delta / n;
      const delta2 = y - this.yMean[i];
      this.yM2[i] += delta * delta2;

      // Accumulate SS_tot = Σ(y - ȳ)²
      this.ssTot += this.yM2[i];
    }
  }

  /**
   * Get current R-squared value.
   *
   * Formula: R² = 1 - (SS_res / SS_tot)
   *
   * R² measures the proportion of variance explained by the model.
   * Range: [0, 1] where 1 indicates perfect fit.
   *
   * @returns R-squared value clamped to [0, 1]
   *
   * Time complexity: O(1)
   * Space complexity: O(1)
   */
  getRSquared(): number {
    if (this.sampleCount < 2 || this.ssTot < 1e-10) {
      return 0;
    }

    const rSquared = 1 - (this.ssRes / this.ssTot);

    // Clamp to [0, 1] - can go negative if model is worse than mean
    if (rSquared < 0) {
      return 0;
    }
    if (rSquared > 1) {
      return 1;
    }
    return rSquared;
  }

  /**
   * Get current RMSE (Root Mean Squared Error).
   *
   * Formula: RMSE = √(Σ(y - ŷ)² / (n × output_dim))
   *
   * @returns RMSE value
   *
   * Time complexity: O(1)
   * Space complexity: O(1)
   */
  getRMSE(): number {
    if (this.sampleCount === 0) {
      return 0;
    }
    return Math.sqrt(
      this.sumSquaredError / (this.sampleCount * this.outputDimension),
    );
  }

  /**
   * Get residual variance for confidence interval estimation.
   *
   * Formula: σ² = SS_res / (n - p)
   * where p is the number of parameters (degrees of freedom adjustment)
   *
   * @param numParameters - Number of model parameters
   * @returns Estimated residual variance
   *
   * Time complexity: O(1)
   * Space complexity: O(1)
   */
  getResidualVariance(numParameters: number): number {
    const degreesOfFreedom = this.sampleCount - numParameters;
    if (degreesOfFreedom <= 0) {
      return 1; // Return 1 if insufficient data
    }
    return this.ssRes / degreesOfFreedom;
  }

  /**
   * Get total number of samples processed.
   *
   * @returns Sample count
   */
  getSampleCount(): number {
    return this.sampleCount;
  }

  /**
   * Reset all statistics to initial state.
   *
   * Time complexity: O(outputDimension)
   * Space complexity: O(1)
   */
  reset(): void {
    this.resetCounters();
  }
}

// ============================================================================
// PREDICTION ENGINE
// ============================================================================

/**
 * Handles prediction computation and confidence interval calculation.
 * Uses t-distribution for small samples, z-distribution for large samples.
 *
 * Prediction formula: ŷ = wᵀφ (dot product of weights and features)
 *
 * Confidence interval: ŷ ± t_{α/2} × SE
 * where SE is the standard error of prediction
 *
 * @class PredictionEngine
 * @implements {IPredictionEngine}
 */
class PredictionEngine implements IPredictionEngine {
  /** Feature count */
  private featureCount: number = 0;
  /** Output dimension */
  private outputDimension: number = 0;

  /**
   * T-distribution critical values for 95% confidence.
   * Key is degrees of freedom, value is critical value.
   */
  private static readonly T_CRITICAL_95: { [df: number]: number } = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    25: 2.060,
    30: 2.042,
    40: 2.021,
    50: 2.009,
    60: 2.000,
    80: 1.990,
    100: 1.984,
    120: 1.980,
  };

  /** Z critical value for 95% confidence (large samples) */
  private static readonly Z_CRITICAL_95: number = 1.96;

  /**
   * Initialize prediction engine.
   *
   * @param featureCount - Number of polynomial features
   * @param outputDimension - Number of output targets
   */
  initialize(featureCount: number, outputDimension: number): void {
    this.featureCount = featureCount;
    this.outputDimension = outputDimension;
  }

  /**
   * Compute prediction from polynomial features.
   *
   * Formula: ŷⱼ = Σᵢ(wⱼᵢ × φᵢ) for each output j
   *
   * @param features - Polynomial features (φ)
   * @param weightManager - Weight manager instance
   * @param output - Output buffer for predictions
   *
   * Time complexity: O(outputDimension × featureCount)
   * Space complexity: O(1)
   */
  predict(
    features: Float64Array,
    weightManager: IWeightManager,
    output: Float64Array,
  ): void {
    for (let o = 0; o < this.outputDimension; o++) {
      let sum = 0;
      for (let f = 0; f < this.featureCount; f++) {
        sum += weightManager.getWeight(o, f) * features[f];
      }
      output[o] = sum;
    }
  }

  /**
   * Get critical value for confidence interval calculation.
   * Uses t-distribution for small samples (n ≤ 30), z-distribution otherwise.
   *
   * @param sampleCount - Number of training samples
   * @param confidenceLevel - Desired confidence level (e.g., 0.95)
   * @returns Critical value
   *
   * Time complexity: O(1)
   * Space complexity: O(1)
   */
  getCriticalValue(sampleCount: number, confidenceLevel: number): number {
    // Scale factor for different confidence levels (approximate)
    // Based on ratio to 95% critical values
    let scaleFactor = 1.0;
    if (confidenceLevel !== 0.95) {
      if (confidenceLevel >= 0.99) {
        scaleFactor = 1.32; // ≈ z_0.005 / z_0.025
      } else if (confidenceLevel >= 0.90) {
        scaleFactor = 0.84; // ≈ z_0.05 / z_0.025
      } else if (confidenceLevel >= 0.80) {
        scaleFactor = 0.66; // ≈ z_0.10 / z_0.025
      }
    }

    // Use z-distribution for large samples
    if (sampleCount > 120) {
      return PredictionEngine.Z_CRITICAL_95 * scaleFactor;
    }

    // Find closest t-critical value
    const df = Math.max(1, sampleCount - this.featureCount);

    // Check exact match first
    if (PredictionEngine.T_CRITICAL_95[df] !== undefined) {
      return PredictionEngine.T_CRITICAL_95[df] * scaleFactor;
    }

    // Find closest available df
    const availableDFs = Object.keys(PredictionEngine.T_CRITICAL_95)
      .map((k) => parseInt(k, 10))
      .sort((a, b) => a - b);

    for (let i = 0; i < availableDFs.length; i++) {
      if (availableDFs[i] >= df) {
        return PredictionEngine.T_CRITICAL_95[availableDFs[i]] * scaleFactor;
      }
    }

    // Fall back to z for very large df
    return PredictionEngine.Z_CRITICAL_95 * scaleFactor;
  }

  /**
   * Calculate standard error for a prediction.
   *
   * Simplified estimation: SE = √(σ² × (1 + 1/n + leverage))
   * where leverage ≈ ||φ||² / n
   *
   * @param features - Polynomial features used for prediction
   * @param residualVariance - Estimated residual variance
   * @param sampleCount - Number of training samples
   * @returns Standard error estimate
   *
   * Time complexity: O(featureCount)
   * Space complexity: O(1)
   */
  calculateStandardError(
    features: Float64Array,
    residualVariance: number,
    sampleCount: number,
  ): number {
    if (sampleCount < 2) {
      return 1; // Return default if insufficient data
    }

    // Calculate approximate leverage: h ≈ ||φ||² / n
    let featureSumSq = 0;
    for (let i = 0; i < this.featureCount; i++) {
      featureSumSq += features[i] * features[i];
    }
    const leverage = featureSumSq / sampleCount;

    // SE = √(σ² × (1 + 1/n + h))
    const variance = residualVariance * (1 + 1 / sampleCount + leverage);
    const se = Math.sqrt(variance);

    // Return finite value
    if (!isFinite(se) || se < 1e-10) {
      return Math.sqrt(residualVariance);
    }
    return se;
  }

  /**
   * Reset prediction engine state.
   */
  reset(): void {
    // No state to reset for this implementation
  }
}

// ============================================================================
// CONFIGURATION BUILDER
// ============================================================================

/**
 * Builder class for MultivariatePolynomialRegression configuration.
 * Implements the Builder pattern for fluent, validated configuration.
 *
 * @class ConfigurationBuilder
 *
 * @example
 * const config = new ConfigurationBuilder()
 *   .withPolynomialDegree(3)
 *   .withLearningRate(0.005)
 *   .withMomentum(0.95)
 *   .withNormalizationMethod('z-score')
 *   .build();
 *
 * const model = new MultivariatePolynomialRegression(config);
 */
class ConfigurationBuilder {
  /** Internal configuration being built */
  private readonly config: IConfiguration;

  /**
   * Creates a new ConfigurationBuilder with default values.
   */
  constructor() {
    this.config = {
      polynomialDegree: 2,
      enableNormalization: true,
      normalizationMethod: "min-max",
      learningRate: 0.01,
      learningRateDecay: 0.999,
      momentum: 0.9,
      regularization: 1e-6,
      gradientClipValue: 1.0,
      confidenceLevel: 0.95,
      batchSize: 1,
    };
  }

  /**
   * Set polynomial degree for feature expansion.
   *
   * @param degree - Polynomial degree (minimum: 1)
   * @returns this builder for chaining
   * @throws Error if degree < 1
   *
   * @example
   * builder.withPolynomialDegree(3); // Use cubic features
   */
  withPolynomialDegree(degree: number): ConfigurationBuilder {
    if (degree < 1) {
      throw new Error("Polynomial degree must be at least 1");
    }
    this.config.polynomialDegree = Math.floor(degree);
    return this;
  }

  /**
   * Enable or disable input normalization.
   *
   * @param enabled - Whether to normalize inputs
   * @returns this builder for chaining
   */
  withNormalization(enabled: boolean): ConfigurationBuilder {
    this.config.enableNormalization = enabled;
    return this;
  }

  /**
   * Set normalization method.
   *
   * @param method - 'none', 'min-max', or 'z-score'
   * @returns this builder for chaining
   */
  withNormalizationMethod(
    method: "none" | "min-max" | "z-score",
  ): ConfigurationBuilder {
    this.config.normalizationMethod = method;
    return this;
  }

  /**
   * Set initial learning rate for SGD.
   *
   * @param rate - Learning rate (range: (0, 1])
   * @returns this builder for chaining
   * @throws Error if rate not in valid range
   */
  withLearningRate(rate: number): ConfigurationBuilder {
    if (rate <= 0 || rate > 1) {
      throw new Error("Learning rate must be in range (0, 1]");
    }
    this.config.learningRate = rate;
    return this;
  }

  /**
   * Set learning rate decay factor.
   * Applied after each sample: η = η × decay
   *
   * @param decay - Decay factor (range: (0, 1])
   * @returns this builder for chaining
   * @throws Error if decay not in valid range
   */
  withLearningRateDecay(decay: number): ConfigurationBuilder {
    if (decay <= 0 || decay > 1) {
      throw new Error("Learning rate decay must be in range (0, 1]");
    }
    this.config.learningRateDecay = decay;
    return this;
  }

  /**
   * Set momentum coefficient.
   * v = momentum × v + η × g
   *
   * @param momentum - Momentum value (range: [0, 1))
   * @returns this builder for chaining
   * @throws Error if momentum not in valid range
   */
  withMomentum(momentum: number): ConfigurationBuilder {
    if (momentum < 0 || momentum >= 1) {
      throw new Error("Momentum must be in range [0, 1)");
    }
    this.config.momentum = momentum;
    return this;
  }

  /**
   * Set L2 regularization coefficient.
   * Adds λ × w to gradient to prevent overfitting.
   *
   * @param regularization - Regularization coefficient (≥ 0)
   * @returns this builder for chaining
   * @throws Error if regularization < 0
   */
  withRegularization(regularization: number): ConfigurationBuilder {
    if (regularization < 0) {
      throw new Error("Regularization must be non-negative");
    }
    this.config.regularization = regularization;
    return this;
  }

  /**
   * Set gradient clipping threshold.
   * Gradients are clipped to [-clipValue, clipValue].
   *
   * @param clipValue - Maximum gradient magnitude (> 0)
   * @returns this builder for chaining
   * @throws Error if clipValue ≤ 0
   */
  withGradientClipValue(clipValue: number): ConfigurationBuilder {
    if (clipValue <= 0) {
      throw new Error("Gradient clip value must be positive");
    }
    this.config.gradientClipValue = clipValue;
    return this;
  }

  /**
   * Set confidence level for prediction intervals.
   *
   * @param level - Confidence level (range: (0, 1))
   * @returns this builder for chaining
   * @throws Error if level not in valid range
   */
  withConfidenceLevel(level: number): ConfigurationBuilder {
    if (level <= 0 || level >= 1) {
      throw new Error("Confidence level must be in range (0, 1)");
    }
    this.config.confidenceLevel = level;
    return this;
  }

  /**
   * Set batch size for mini-batch SGD.
   *
   * @param size - Batch size (minimum: 1)
   * @returns this builder for chaining
   * @throws Error if size < 1
   */
  withBatchSize(size: number): ConfigurationBuilder {
    if (size < 1) {
      throw new Error("Batch size must be at least 1");
    }
    this.config.batchSize = Math.floor(size);
    return this;
  }

  /**
   * Build and return the configuration object.
   *
   * @returns Immutable configuration object
   */
  build(): Readonly<IConfiguration> {
    return Object.freeze({ ...this.config });
  }
}

// ============================================================================
// MAIN CLASS: MULTIVARIATE POLYNOMIAL REGRESSION
// ============================================================================

/**
 * Multivariate Polynomial Regression with Online Learning using SGD.
 *
 * This class implements multivariate polynomial regression with incremental
 * online learning using Stochastic Gradient Descent (SGD) with momentum.
 *
 * ## Features:
 * - **Incremental online learning**: Process data point-by-point without storing all data
 * - **Configurable polynomial degree**: Expand features to capture non-linear relationships
 * - **Multiple normalization strategies**: min-max, z-score, or none
 * - **Momentum-based SGD**: Faster convergence with velocity accumulation
 * - **Learning rate decay**: Automatic learning rate annealing
 * - **L2 regularization**: Prevent overfitting with weight decay
 * - **Gradient clipping**: Numerical stability against exploding gradients
 * - **Confidence intervals**: Prediction uncertainty quantification
 *
 * ## SGD Algorithm:
 * For each new sample (x, y):
 * 1. Generate polynomial features: φ = polynomial_features(normalize(x))
 * 2. Compute prediction: ŷ = wᵀ·φ
 * 3. Compute prediction error: e = y - ŷ
 * 4. Compute gradient with L2 regularization: g = -e·φ + λ·w
 * 5. Apply gradient clipping: g = clip(g, -clipValue, clipValue)
 * 6. Update velocity with momentum: v = μ·v + η·g
 * 7. Update weights: w = w - v
 * 8. Decay learning rate: η = η × decay
 *
 * @class MultivariatePolynomialRegression
 *
 * @example
 * // Basic usage with default configuration
 * const model = new MultivariatePolynomialRegression();
 *
 * // Train incrementally
 * model.fitOnline({
 *   xCoordinates: [[1, 2], [2, 3], [3, 4], [4, 5]],
 *   yCoordinates: [[5], [11], [19], [29]]
 * });
 *
 * // Make predictions
 * const result = model.predict({ futureSteps: 2 });
 * console.log(result.predictions);
 *
 * @example
 * // Custom configuration using builder
 * const config = new ConfigurationBuilder()
 *   .withPolynomialDegree(3)
 *   .withLearningRate(0.005)
 *   .withMomentum(0.95)
 *   .withNormalizationMethod('z-score')
 *   .withConfidenceLevel(0.99)
 *   .build();
 *
 * const model = new MultivariatePolynomialRegression(config);
 *
 * @example
 * // Predict at specific input points
 * const result = model.predict({
 *   futureSteps: 0,
 *   inputPoints: [[5, 6], [6, 7], [7, 8]]
 * });
 */
class MultivariatePolynomialRegression {
  // ========== Configuration ==========
  /** Frozen configuration object */
  private readonly config: Readonly<IConfiguration>;

  // ========== Component Managers (Dependency Injection) ==========
  /** Shared matrix operations utility */
  private readonly matrixOps: MatrixOperations;
  /** Input normalizer */
  private readonly normalizer: INormalizer;
  /** Polynomial feature generator */
  private readonly featureGenerator: IPolynomialFeatureGenerator;
  /** Weight storage and access */
  private readonly weightManager: IWeightManager;
  /** Gradient computation and momentum */
  private readonly gradientManager: IGradientManager;
  /** Prediction and confidence intervals */
  private readonly predictionEngine: IPredictionEngine;
  /** Model statistics tracking */
  private readonly statisticsTracker: IStatisticsTracker;

  // ========== State ==========
  /** Number of input features */
  private inputDimension: number = 0;
  /** Number of output targets */
  private outputDimension: number = 0;
  /** Current (decayed) learning rate */
  private currentLearningRate: number;
  /** Whether model has been initialized with data */
  private isInitialized: boolean = false;

  // ========== Preallocated Buffers (Hot Path Optimization) ==========
  /** Buffer for raw input */
  private inputBuffer: Float64Array | null = null;
  /** Buffer for normalized input */
  private normalizedBuffer: Float64Array | null = null;
  /** Buffer for polynomial features */
  private featureBuffer: Float64Array | null = null;
  /** Buffer for predictions */
  private predictionBuffer: Float64Array | null = null;
  /** Buffer for target values */
  private targetBuffer: Float64Array | null = null;
  /** Buffer for weight access */
  private weightBuffer: Float64Array | null = null;
  /** Ring buffer of recent inputs for extrapolation */
  private lastInputs: Float64Array[] = [];
  /** Maximum number of inputs to retain */
  private readonly maxRetainedInputs: number = 10;

  /**
   * Creates a new MultivariatePolynomialRegression instance.
   *
   * @param config - Configuration object (partial allowed, defaults applied)
   *
   * Time complexity: O(1) - lazy initialization
   * Space complexity: O(1) - buffers allocated on first fitOnline
   *
   * @example
   * // Default configuration
   * const model = new MultivariatePolynomialRegression();
   *
   * @example
   * // Partial configuration (unspecified values use defaults)
   * const model = new MultivariatePolynomialRegression({
   *   polynomialDegree: 3,
   *   learningRate: 0.005
   * });
   *
   * @example
   * // Full configuration via builder
   * const config = new ConfigurationBuilder()
   *   .withPolynomialDegree(2)
   *   .withLearningRate(0.01)
   *   .build();
   * const model = new MultivariatePolynomialRegression(config);
   */
  constructor(config?: Partial<IConfiguration>) {
    // Merge provided config with defaults
    this.config = Object.freeze({
      polynomialDegree: config?.polynomialDegree ?? 2,
      enableNormalization: config?.enableNormalization ?? true,
      normalizationMethod: config?.normalizationMethod ?? "min-max",
      learningRate: config?.learningRate ?? 0.01,
      learningRateDecay: config?.learningRateDecay ?? 0.999,
      momentum: config?.momentum ?? 0.9,
      regularization: config?.regularization ?? 1e-6,
      gradientClipValue: config?.gradientClipValue ?? 1.0,
      confidenceLevel: config?.confidenceLevel ?? 0.95,
      batchSize: config?.batchSize ?? 1,
    });

    this.currentLearningRate = this.config.learningRate;

    // Initialize components (Dependency Injection pattern)
    this.matrixOps = new MatrixOperations();
    this.normalizer = new Normalizer(
      this.config.enableNormalization,
      this.config.normalizationMethod,
    );
    this.featureGenerator = new PolynomialFeatureGenerator(
      this.config.polynomialDegree,
    );
    this.weightManager = new WeightManager();
    this.gradientManager = new GradientManager(
      this.config.momentum,
      this.config.gradientClipValue,
    );
    this.predictionEngine = new PredictionEngine();
    this.statisticsTracker = new StatisticsTracker();
  }

  /**
   * Initialize model for specific input/output dimensions.
   * Called automatically on first fitOnline call.
   *
   * @param inputDim - Number of input features
   * @param outputDim - Number of output targets
   *
   * Time complexity: O(featureCount × inputDim) for exponent generation
   * Space complexity: O(featureCount × (inputDim + outputDim))
   */
  private initialize(inputDim: number, outputDim: number): void {
    this.inputDimension = inputDim;
    this.outputDimension = outputDim;

    // Initialize feature generator first (needed for feature count)
    this.featureGenerator.initialize(inputDim);
    const featureCount = this.featureGenerator.getFeatureCount();

    // Initialize all components
    this.normalizer.initialize(inputDim);
    this.weightManager.initialize(featureCount, outputDim);
    this.gradientManager.initialize(featureCount, outputDim);
    this.predictionEngine.initialize(featureCount, outputDim);
    this.statisticsTracker.initialize(outputDim);

    // Preallocate buffers for hot paths
    this.inputBuffer = new Float64Array(inputDim);
    this.normalizedBuffer = new Float64Array(inputDim);
    this.featureBuffer = new Float64Array(featureCount);
    this.predictionBuffer = new Float64Array(outputDim);
    this.targetBuffer = new Float64Array(outputDim);
    this.weightBuffer = new Float64Array(featureCount);

    this.isInitialized = true;
  }

  /**
   * Train the model incrementally using online SGD.
   *
   * Processes each sample through the SGD algorithm:
   * 1. Normalize input and generate polynomial features
   * 2. Compute prediction and error
   * 3. Calculate gradient with regularization
   * 4. Update weights using momentum
   * 5. Decay learning rate
   *
   * @param params - Training data with xCoordinates and yCoordinates
   * @param params.xCoordinates - Input samples, shape [n_samples][n_features]
   * @param params.yCoordinates - Target values, shape [n_samples][n_outputs]
   * @throws Error if xCoordinates and yCoordinates have different lengths
   * @throws Error if dimensions change after initialization
   *
   * Time complexity: O(n_samples × n_outputs × n_features²) worst case
   * Space complexity: O(1) - uses preallocated buffers
   *
   * @example
   * model.fitOnline({
   *   xCoordinates: [[1, 2], [2, 3], [3, 4]],
   *   yCoordinates: [[5], [10], [17]]
   * });
   *
   * @example
   * // Incremental updates (can be called multiple times)
   * model.fitOnline({ xCoordinates: [[4, 5]], yCoordinates: [[26]] });
   * model.fitOnline({ xCoordinates: [[5, 6]], yCoordinates: [[37]] });
   */
  fitOnline(
    params: { xCoordinates: number[][]; yCoordinates: number[][] },
  ): void {
    const { xCoordinates, yCoordinates } = params;

    // Early exit for empty data
    if (xCoordinates.length === 0 || yCoordinates.length === 0) {
      return;
    }

    // Validate input lengths match
    if (xCoordinates.length !== yCoordinates.length) {
      throw new Error(
        "xCoordinates and yCoordinates must have the same number of samples",
      );
    }

    const inputDim = xCoordinates[0].length;
    const outputDim = yCoordinates[0].length;

    // Lazy initialization on first call
    if (!this.isInitialized) {
      this.initialize(inputDim, outputDim);
    }

    // Validate dimensions remain consistent
    if (
      inputDim !== this.inputDimension || outputDim !== this.outputDimension
    ) {
      throw new Error(
        `Dimension mismatch: expected (${this.inputDimension}, ${this.outputDimension}), ` +
          `got (${inputDim}, ${outputDim})`,
      );
    }

    const numSamples = xCoordinates.length;
    const featureCount = this.featureGenerator.getFeatureCount();

    // Process each sample
    for (let s = 0; s < numSamples; s++) {
      const x = xCoordinates[s];
      const y = yCoordinates[s];

      // Copy input to typed array buffer
      for (let i = 0; i < inputDim; i++) {
        this.inputBuffer![i] = x[i];
      }

      // Update normalization statistics BEFORE normalizing
      // This ensures statistics reflect all seen data
      this.normalizer.updateStatistics(this.inputBuffer!);

      // STEP 1: Normalize input
      this.normalizer.normalizeInPlace(
        this.inputBuffer!,
        this.normalizedBuffer!,
      );

      // STEP 1 (cont.): Generate polynomial features
      // φ = [1, x₁, x₂, x₁², x₁x₂, x₂², ...]
      this.featureGenerator.generateFeaturesInPlace(
        this.normalizedBuffer!,
        this.featureBuffer!,
      );

      // STEP 2: Compute prediction ŷ = wᵀφ
      this.predictionEngine.predict(
        this.featureBuffer!,
        this.weightManager,
        this.predictionBuffer!,
      );

      // Copy target to typed array buffer
      for (let i = 0; i < outputDim; i++) {
        this.targetBuffer![i] = y[i];
      }

      // Update statistics tracker
      this.statisticsTracker.update(this.targetBuffer!, this.predictionBuffer!);

      // Process each output dimension
      for (let o = 0; o < outputDim; o++) {
        // STEP 3: Compute prediction error e = y - ŷ
        const error = y[o] - this.predictionBuffer![o];

        // Get current weights for this output
        this.weightManager.getWeightsForOutput(o, this.weightBuffer!);

        // STEP 4-5: Compute gradient with regularization and clipping
        // g = -e·φ + λ·w, then clip to [-clipValue, clipValue]
        this.gradientManager.computeGradient(
          this.featureBuffer!,
          error,
          this.weightBuffer!,
          this.config.regularization,
          o,
        );

        // STEP 6: Update velocity with momentum v = μv + ηg
        const deltas = this.gradientManager.updateVelocityAndGetDelta(
          this.currentLearningRate,
          o,
        );

        // STEP 7: Update weights w = w - v
        for (let f = 0; f < featureCount; f++) {
          this.weightManager.updateWeight(o, f, deltas[f]);
        }
      }

      // Store input for extrapolation (ring buffer)
      this.storeLastInput();

      // STEP 8: Decay learning rate η = η × decay
      this.currentLearningRate *= this.config.learningRateDecay;
    }
  }

  /**
   * Store current input in ring buffer for extrapolation.
   * Reuses arrays to minimize allocations.
   */
  private storeLastInput(): void {
    if (this.lastInputs.length >= this.maxRetainedInputs) {
      // Reuse oldest array
      const reused = this.lastInputs.shift()!;
      this.matrixOps.copyTo(this.inputBuffer!, reused);
      this.lastInputs.push(reused);
    } else {
      // Allocate new array
      const copy = new Float64Array(this.inputDimension);
      this.matrixOps.copyTo(this.inputBuffer!, copy);
      this.lastInputs.push(copy);
    }
  }

  /**
   * Make predictions with confidence intervals.
   *
   * Supports two modes:
   * 1. **Extrapolation**: Predict future steps based on trend in recent inputs
   * 2. **Specific points**: Predict at provided input coordinates
   *
   * Confidence intervals use t-distribution for small samples (n ≤ 30)
   * and z-distribution for larger samples.
   *
   * @param params - Prediction parameters
   * @param params.futureSteps - Number of future points to extrapolate
   * @param params.inputPoints - Specific input coordinates to predict (optional)
   * @returns PredictionResult with predictions and model metrics
   *
   * Time complexity: O(n_points × n_outputs × n_features)
   * Space complexity: O(n_points × n_outputs) for results
   *
   * @example
   * // Extrapolate 5 future steps
   * const result = model.predict({ futureSteps: 5 });
   * console.log(result.predictions[0].predicted);  // First prediction
   * console.log(result.predictions[0].lowerBound); // Lower confidence bound
   * console.log(result.predictions[0].upperBound); // Upper confidence bound
   *
   * @example
   * // Predict at specific input points
   * const result = model.predict({
   *   futureSteps: 0,
   *   inputPoints: [[5, 6], [6, 7]]
   * });
   *
   * @example
   * // Check model readiness
   * const result = model.predict({ futureSteps: 3 });
   * if (!result.isModelReady) {
   *   console.log('Need more training data');
   * }
   */
  predict(
    params: { futureSteps: number; inputPoints?: number[][] },
  ): PredictionResult {
    const { futureSteps, inputPoints } = params;

    // Build base result
    const result: PredictionResult = {
      predictions: [],
      confidenceLevel: this.config.confidenceLevel,
      rSquared: this.statisticsTracker.getRSquared(),
      rmse: this.statisticsTracker.getRMSE(),
      sampleCount: this.statisticsTracker.getSampleCount(),
      isModelReady: this.isInitialized &&
        this.statisticsTracker.getSampleCount() >= 2,
    };

    // Return empty result if not initialized
    if (!this.isInitialized) {
      return result;
    }

    // Determine points to predict
    const pointsToPredict = this.buildPredictionPoints(
      futureSteps,
      inputPoints,
    );

    // Calculate variance and critical value for confidence intervals
    const residualVariance = this.statisticsTracker.getResidualVariance(
      this.featureGenerator.getFeatureCount(),
    );
    const criticalValue = this.predictionEngine.getCriticalValue(
      this.statisticsTracker.getSampleCount(),
      this.config.confidenceLevel,
    );

    // Generate predictions
    for (let p = 0; p < pointsToPredict.length; p++) {
      const point = pointsToPredict[p];
      const prediction = this.predictSinglePoint(
        point,
        residualVariance,
        criticalValue,
      );
      result.predictions.push(prediction);
    }

    return result;
  }

  /**
   * Build list of points to predict based on parameters.
   *
   * @param futureSteps - Number of extrapolation steps
   * @param inputPoints - Specific points to predict
   * @returns Array of input points
   */
  private buildPredictionPoints(
    futureSteps: number,
    inputPoints?: number[][],
  ): number[][] {
    const points: number[][] = [];

    // Add specific input points if provided
    if (inputPoints !== undefined && inputPoints.length > 0) {
      for (let i = 0; i < inputPoints.length; i++) {
        points.push(inputPoints[i]);
      }
    }

    // Add extrapolated points if requested
    if (futureSteps > 0 && this.lastInputs.length > 0) {
      const lastInput = this.lastInputs[this.lastInputs.length - 1];

      // Calculate trend from recent inputs
      const trend = new Float64Array(this.inputDimension);
      if (this.lastInputs.length >= 2) {
        const prevInput = this.lastInputs[this.lastInputs.length - 2];
        for (let i = 0; i < this.inputDimension; i++) {
          trend[i] = lastInput[i] - prevInput[i];
        }
      } else {
        // Default increment if only one point
        for (let i = 0; i < this.inputDimension; i++) {
          trend[i] = 1;
        }
      }

      // Generate extrapolated points
      for (let step = 1; step <= futureSteps; step++) {
        const futurePoint: number[] = [];
        for (let i = 0; i < this.inputDimension; i++) {
          futurePoint.push(lastInput[i] + trend[i] * step);
        }
        points.push(futurePoint);
      }
    }

    return points;
  }

  /**
   * Predict a single point with confidence interval.
   *
   * @param point - Input coordinates
   * @param residualVariance - Estimated residual variance
   * @param criticalValue - Critical value for confidence interval
   * @returns SinglePrediction object
   */
  private predictSinglePoint(
    point: number[],
    residualVariance: number,
    criticalValue: number,
  ): SinglePrediction {
    // Copy to input buffer
    for (let i = 0; i < this.inputDimension; i++) {
      this.inputBuffer![i] = point[i];
    }

    // Normalize
    this.normalizer.normalizeInPlace(this.inputBuffer!, this.normalizedBuffer!);

    // Generate features
    this.featureGenerator.generateFeaturesInPlace(
      this.normalizedBuffer!,
      this.featureBuffer!,
    );

    // Compute prediction
    this.predictionEngine.predict(
      this.featureBuffer!,
      this.weightManager,
      this.predictionBuffer!,
    );

    // Calculate standard error
    const standardError = this.predictionEngine.calculateStandardError(
      this.featureBuffer!,
      residualVariance,
      this.statisticsTracker.getSampleCount(),
    );

    // Build result arrays
    const predicted: number[] = [];
    const lowerBound: number[] = [];
    const upperBound: number[] = [];
    const standardErrors: number[] = [];
    const margin = criticalValue * standardError;

    for (let o = 0; o < this.outputDimension; o++) {
      const pred = this.predictionBuffer![o];
      predicted.push(pred);
      lowerBound.push(pred - margin);
      upperBound.push(pred + margin);
      standardErrors.push(standardError);
    }

    return {
      predicted,
      lowerBound,
      upperBound,
      standardError: standardErrors,
    };
  }

  /**
   * Get a summary of the current model state.
   *
   * @returns ModelSummary object with model statistics
   *
   * Time complexity: O(1)
   * Space complexity: O(1)
   *
   * @example
   * const summary = model.getModelSummary();
   * console.log(`Initialized: ${summary.isInitialized}`);
   * console.log(`Samples: ${summary.sampleCount}`);
   * console.log(`R²: ${summary.rSquared.toFixed(4)}`);
   * console.log(`RMSE: ${summary.rmse.toFixed(4)}`);
   * console.log(`Features: ${summary.polynomialFeatureCount}`);
   */
  getModelSummary(): ModelSummary {
    return {
      isInitialized: this.isInitialized,
      inputDimension: this.inputDimension,
      outputDimension: this.outputDimension,
      polynomialDegree: this.config.polynomialDegree,
      polynomialFeatureCount: this.isInitialized
        ? this.featureGenerator.getFeatureCount()
        : 0,
      sampleCount: this.statisticsTracker.getSampleCount(),
      rSquared: this.statisticsTracker.getRSquared(),
      rmse: this.statisticsTracker.getRMSE(),
      normalizationEnabled: this.config.enableNormalization,
      normalizationMethod: this.config.normalizationMethod,
    };
  }

  /**
   * Get current model weights.
   *
   * @returns 2D array of weights, shape [outputDimension][featureCount]
   *
   * Time complexity: O(outputDimension × featureCount)
   * Space complexity: O(outputDimension × featureCount)
   *
   * @example
   * const weights = model.getWeights();
   * // weights[0][0] is the bias for output 0
   * // weights[0][1] is the coefficient for x₁ in output 0
   */
  getWeights(): number[][] {
    return this.weightManager.getWeights();
  }

  /**
   * Get current normalization statistics.
   *
   * @returns NormalizationStats with min, max, mean, std per input dimension
   *
   * Time complexity: O(inputDimension)
   * Space complexity: O(inputDimension)
   *
   * @example
   * const stats = model.getNormalizationStats();
   * console.log('Min values:', stats.min);
   * console.log('Max values:', stats.max);
   * console.log('Mean values:', stats.mean);
   * console.log('Std values:', stats.std);
   * console.log('Sample count:', stats.count);
   */
  getNormalizationStats(): NormalizationStats {
    return this.normalizer.getStats();
  }

  /**
   * Reset the model to initial state.
   * Clears all learned weights, statistics, and normalization data.
   * Configuration is preserved.
   *
   * Time complexity: O(total parameters)
   * Space complexity: O(1) - frees existing allocations
   *
   * @example
   * model.reset();
   * // Model is now ready for fresh training
   * // Configuration remains unchanged
   */
  reset(): void {
    this.isInitialized = false;
    this.inputDimension = 0;
    this.outputDimension = 0;
    this.currentLearningRate = this.config.learningRate;

    // Reset all components
    this.weightManager.reset();
    this.gradientManager.reset();
    this.predictionEngine.reset();
    this.statisticsTracker.reset();
    this.normalizer.reset();

    // Release buffers
    this.inputBuffer = null;
    this.normalizedBuffer = null;
    this.featureBuffer = null;
    this.predictionBuffer = null;
    this.targetBuffer = null;
    this.weightBuffer = null;
    this.lastInputs = [];

    // Clear array pool
    this.matrixOps.clearPool();
  }
}

// ============================================================================
// EXPORTS
// ============================================================================

export {
  // Configuration
  ConfigurationBuilder,
  GradientManager,
  IConfiguration,
  IGradientManager,
  // Component interfaces (for testing/extension)
  IMatrixOperations,
  INormalizationStrategy,
  INormalizer,
  IPolynomialFeatureGenerator,
  IPredictionEngine,
  IStatisticsTracker,
  IWeightManager,
  // Concrete implementations (for advanced usage)
  MatrixOperations,
  MinMaxNormalizationStrategy,
  ModelSummary,
  // Main class
  MultivariatePolynomialRegression,
  // Normalization strategies
  NoNormalizationStrategy,
  NormalizationStats,
  Normalizer,
  PolynomialFeatureGenerator,
  PredictionEngine,
  // Result types
  PredictionResult,
  SinglePrediction,
  StatisticsTracker,
  WeightManager,
  ZScoreNormalizationStrategy,
};
