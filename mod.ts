// ==================== INTERFACES ====================

/**
 * Configuration options for the MultivariatePolynomialRegression model.
 * @interface IMultivariatePolynomialRegressionConfig
 */
export interface IMultivariatePolynomialRegressionConfig {
  /** Polynomial degree for feature generation (default: 2, minimum: 1) */
  polynomialDegree?: number;
  /** Enable/disable input normalization (default: true) */
  enableNormalization?: boolean;
  /** Normalization method to use (default: 'min-max') */
  normalizationMethod?: "none" | "min-max" | "z-score";
  /** Forgetting factor λ for RLS algorithm (default: 0.99, range: (0, 1]) */
  forgettingFactor?: number;
  /** Initial covariance matrix diagonal value (default: 1000) */
  initialCovariance?: number;
  /** Regularization term for numerical stability (default: 1e-6) */
  regularization?: number;
  /** Confidence level for prediction intervals (default: 0.95, range: (0, 1)) */
  confidenceLevel?: number;
}

/**
 * Result for a single prediction point.
 * @interface ISinglePrediction
 */
export interface ISinglePrediction {
  /** Predicted values for each output dimension */
  predicted: number[];
  /** Lower bound of confidence interval for each output dimension */
  lowerBound: number[];
  /** Upper bound of confidence interval for each output dimension */
  upperBound: number[];
  /** Standard error for each output dimension */
  standardError: number[];
}

/**
 * Complete prediction result including model statistics.
 * @interface IPredictionResult
 */
export interface IPredictionResult {
  /** Array of predictions for each input point */
  predictions: ISinglePrediction[];
  /** Confidence level used for intervals */
  confidenceLevel: number;
  /** R-squared coefficient of determination */
  rSquared: number;
  /** Root mean squared error */
  rmse: number;
  /** Number of samples used for training */
  sampleCount: number;
  /** Whether the model has been initialized with data */
  isModelReady: boolean;
}

/**
 * Summary of the model state and configuration.
 * @interface IModelSummary
 */
export interface IModelSummary {
  /** Whether the model has been initialized with data */
  isInitialized: boolean;
  /** Number of input dimensions */
  inputDimension: number;
  /** Number of output dimensions */
  outputDimension: number;
  /** Polynomial degree used for feature generation */
  polynomialDegree: number;
  /** Total number of polynomial features */
  polynomialFeatureCount: number;
  /** Number of training samples seen */
  sampleCount: number;
  /** R-squared coefficient of determination */
  rSquared: number;
  /** Root mean squared error */
  rmse: number;
  /** Whether normalization is enabled */
  normalizationEnabled: boolean;
  /** Normalization method being used */
  normalizationMethod: "none" | "min-max" | "z-score";
}

/**
 * Statistics used for normalization.
 * @interface INormalizationStats
 */
export interface INormalizationStats {
  /** Minimum values for each input dimension */
  min: number[];
  /** Maximum values for each input dimension */
  max: number[];
  /** Mean values for each input dimension */
  mean: number[];
  /** Standard deviation for each input dimension */
  std: number[];
  /** Number of samples used to compute statistics */
  count: number;
}

/**
 * Interface for normalization strategy (Strategy Pattern).
 * @interface INormalizationStrategy
 */
interface INormalizationStrategy {
  /** Normalizes a single value given statistics */
  normalize(
    value: number,
    min: number,
    max: number,
    mean: number,
    std: number,
  ): number;
  /** Denormalizes a single value given statistics */
  denormalize(
    value: number,
    min: number,
    max: number,
    mean: number,
    std: number,
  ): number;
  /** Name identifier for the strategy */
  readonly name: "none" | "min-max" | "z-score";
}

/**
 * Interface for matrix operations.
 * @interface IMatrixOperations
 */
interface IMatrixOperations {
  /** Creates a new Float64Array filled with zeros */
  createZeroVector(size: number): Float64Array;
  /** Creates a diagonal matrix as flat array */
  createIdentityDiagonal(size: number, value: number): Float64Array;
  /** Computes dot product of two vectors */
  dotProduct(a: Float64Array, b: Float64Array, length: number): number;
  /** Scales a vector in place */
  scaleInPlace(vec: Float64Array, scalar: number, length: number): void;
  /** Adds scaled vector to another in place */
  addScaledInPlace(
    result: Float64Array,
    b: Float64Array,
    scalar: number,
    length: number,
  ): void;
  /** Matrix-vector multiplication for symmetric matrix */
  symmetricMatrixVectorMultiply(
    matrix: Float64Array,
    vector: Float64Array,
    result: Float64Array,
    size: number,
  ): void;
  /** Symmetric rank-1 update: P = (P - k·(Pφ)ᵀ) / λ */
  rankOneUpdateSymmetric(
    matrix: Float64Array,
    k: Float64Array,
    Pphi: Float64Array,
    size: number,
    lambda: number,
  ): void;
}

/**
 * Interface for covariance management.
 * @interface ICovarianceManager
 */
interface ICovarianceManager {
  initialize(size: number, initialValue: number): void;
  computeGain(phi: Float64Array, lambda: number): Float64Array;
  update(k: Float64Array, lambda: number): void;
  applyRegularization(regularization: number): void;
  computePhiTPphi(phi: Float64Array): number;
  isInitialized(): boolean;
  reset(): void;
}

/**
 * Interface for weight management.
 * @interface IWeightManager
 */
interface IWeightManager {
  initialize(featureCount: number, outputDimension: number): void;
  predict(phi: Float64Array, output?: Float64Array): Float64Array;
  update(k: Float64Array, error: Float64Array): void;
  getWeights(): number[][];
  getOutputDimension(): number;
  isInitialized(): boolean;
  reset(): void;
}

/**
 * Interface for statistics tracking.
 * @interface IStatisticsTracker
 */
interface IStatisticsTracker {
  initialize(outputDimension: number): void;
  update(actual: Float64Array, predicted: Float64Array): void;
  getRSquared(): number;
  getRMSE(): number;
  getCount(): number;
  reset(): void;
}

/**
 * Interface for polynomial feature generation.
 * @interface IPolynomialFeatureGenerator
 */
interface IPolynomialFeatureGenerator {
  initialize(dimension: number): void;
  generate(input: Float64Array, output?: Float64Array): Float64Array;
  getFeatureCount(): number;
  getDegree(): number;
  isInitialized(): boolean;
  reset(): void;
}

// ==================== MATRIX OPERATIONS CLASS ====================

/**
 * Provides optimized matrix and vector operations using Float64Array.
 * All operations minimize memory allocations and work in-place where possible.
 *
 * @example
 * ```typescript
 * const matrixOps = new MatrixOperations();
 * const vec = matrixOps.createZeroVector(3);
 * const dot = matrixOps.dotProduct(vec1, vec2, 3);
 * ```
 *
 * @class MatrixOperations
 * @implements {IMatrixOperations}
 */
class MatrixOperations implements IMatrixOperations {
  /**
   * Creates a new zero-filled Float64Array of the specified size.
   *
   * Time complexity: O(n)
   * Space complexity: O(n)
   *
   * @param {number} size - The size of the vector
   * @returns {Float64Array} A new zero-filled array
   */
  public createZeroVector(size: number): Float64Array {
    return new Float64Array(size);
  }

  /**
   * Creates a diagonal matrix representation as a flat array in row-major order.
   * P = value × I (identity matrix scaled by value)
   *
   * Time complexity: O(n²)
   * Space complexity: O(n²)
   *
   * @param {number} size - The dimension of the square matrix
   * @param {number} value - The value for diagonal elements
   * @returns {Float64Array} The matrix as a flat array
   */
  public createIdentityDiagonal(size: number, value: number): Float64Array {
    const matrix = new Float64Array(size * size);
    // Set diagonal elements: matrix[i,i] = value
    for (let i = 0; i < size; i++) {
      matrix[i * size + i] = value;
    }
    return matrix;
  }

  /**
   * Computes the dot product of two vectors.
   * Mathematical formula: result = Σ(a[i] × b[i]) for i = 0 to length-1
   *
   * Time complexity: O(n)
   * Space complexity: O(1)
   *
   * @param {Float64Array} a - First vector
   * @param {Float64Array} b - Second vector
   * @param {number} length - Number of elements to process
   * @returns {number} The dot product a·b
   */
  public dotProduct(a: Float64Array, b: Float64Array, length: number): number {
    let sum = 0;
    // Unrolled loop for better performance
    const unrolledLength = length - (length % 4);
    let i = 0;

    for (; i < unrolledLength; i += 4) {
      sum += a[i] * b[i] + a[i + 1] * b[i + 1] + a[i + 2] * b[i + 2] +
        a[i + 3] * b[i + 3];
    }

    for (; i < length; i++) {
      sum += a[i] * b[i];
    }

    return sum;
  }

  /**
   * Scales a vector in place by a scalar value.
   * Mathematical formula: vec[i] = vec[i] × scalar for all i
   *
   * Time complexity: O(n)
   * Space complexity: O(1)
   *
   * @param {Float64Array} vec - The vector to scale (modified in place)
   * @param {number} scalar - The scalar multiplier
   * @param {number} length - Number of elements to process
   */
  public scaleInPlace(vec: Float64Array, scalar: number, length: number): void {
    for (let i = 0; i < length; i++) {
      vec[i] *= scalar;
    }
  }

  /**
   * Adds a scaled vector to another vector in place.
   * Mathematical formula: result[i] = result[i] + scalar × b[i]
   *
   * Time complexity: O(n)
   * Space complexity: O(1)
   *
   * @param {Float64Array} result - The target vector (modified in place)
   * @param {Float64Array} b - The vector to add
   * @param {number} scalar - The scalar multiplier for b
   * @param {number} length - Number of elements to process
   */
  public addScaledInPlace(
    result: Float64Array,
    b: Float64Array,
    scalar: number,
    length: number,
  ): void {
    for (let i = 0; i < length; i++) {
      result[i] += scalar * b[i];
    }
  }

  /**
   * Multiplies a symmetric matrix by a vector.
   * The matrix is stored as a flat array in row-major order.
   * Mathematical formula: result = M × vector
   *
   * Time complexity: O(n²)
   * Space complexity: O(1) (result is preallocated)
   *
   * @param {Float64Array} matrix - The symmetric matrix as flat array
   * @param {Float64Array} vector - The input vector
   * @param {Float64Array} result - Preallocated result vector
   * @param {number} size - The dimension of the matrix
   */
  public symmetricMatrixVectorMultiply(
    matrix: Float64Array,
    vector: Float64Array,
    result: Float64Array,
    size: number,
  ): void {
    for (let i = 0; i < size; i++) {
      let sum = 0;
      const rowOffset = i * size;

      // Compute row i of matrix times vector
      for (let j = 0; j < size; j++) {
        sum += matrix[rowOffset + j] * vector[j];
      }

      result[i] = sum;
    }
  }

  /**
   * Performs a symmetric rank-1 update on the covariance matrix.
   * This implements the RLS covariance update step.
   *
   * Mathematical formula: P = (P - k × (Pφ)ᵀ) / λ
   *
   * Time complexity: O(n²)
   * Space complexity: O(1)
   *
   * @param {Float64Array} matrix - The matrix to update (modified in place)
   * @param {Float64Array} k - The gain vector
   * @param {Float64Array} Pphi - The P×φ vector (precomputed)
   * @param {number} size - The dimension of the matrix
   * @param {number} lambda - The forgetting factor
   */
  public rankOneUpdateSymmetric(
    matrix: Float64Array,
    k: Float64Array,
    Pphi: Float64Array,
    size: number,
    lambda: number,
  ): void {
    // Precompute 1/λ to avoid division in inner loop
    const invLambda = 1.0 / lambda;

    for (let i = 0; i < size; i++) {
      const ki = k[i];
      const rowOffset = i * size;

      for (let j = 0; j < size; j++) {
        // P[i,j] = (P[i,j] - k[i] × Pphi[j]) / λ
        matrix[rowOffset + j] = (matrix[rowOffset + j] - ki * Pphi[j]) *
          invLambda;
      }
    }
  }
}

// ==================== NORMALIZATION STRATEGIES ====================

/**
 * No normalization strategy - passes values through unchanged.
 * Implements the Strategy pattern for normalization.
 *
 * @class NoNormalizationStrategy
 * @implements {INormalizationStrategy}
 */
class NoNormalizationStrategy implements INormalizationStrategy {
  /** @inheritdoc */
  public readonly name: "none" = "none";

  /**
   * Returns the value unchanged.
   *
   * @param {number} value - The input value
   * @returns {number} The same value unchanged
   */
  public normalize(value: number): number {
    return value;
  }

  /**
   * Returns the value unchanged.
   *
   * @param {number} value - The normalized value
   * @returns {number} The same value unchanged
   */
  public denormalize(value: number): number {
    return value;
  }
}

/**
 * Min-max normalization strategy.
 * Scales values to the range [0, 1].
 *
 * Mathematical formula: normalized = (x - min) / (max - min)
 *
 * @class MinMaxNormalizationStrategy
 * @implements {INormalizationStrategy}
 */
class MinMaxNormalizationStrategy implements INormalizationStrategy {
  /** @inheritdoc */
  public readonly name: "min-max" = "min-max";

  /** Epsilon for numerical stability */
  private readonly epsilon: number = 1e-10;

  /**
   * Normalizes a value to [0, 1] range using min-max scaling.
   *
   * Mathematical formula: (x - min) / (max - min)
   *
   * @param {number} value - The input value
   * @param {number} min - The minimum observed value
   * @param {number} max - The maximum observed value
   * @returns {number} The normalized value in [0, 1], clamped to prevent outliers
   */
  public normalize(value: number, min: number, max: number): number {
    const range = max - min;

    // Handle zero variance case
    if (range < this.epsilon) {
      return 0.5;
    }

    const normalized = (value - min) / range;

    // Clamp to [0, 1] to handle outliers
    return Math.max(0, Math.min(1, normalized));
  }

  /**
   * Denormalizes a value from [0, 1] to original scale.
   *
   * Mathematical formula: value × (max - min) + min
   *
   * @param {number} value - The normalized value
   * @param {number} min - The minimum observed value
   * @param {number} max - The maximum observed value
   * @returns {number} The denormalized value
   */
  public denormalize(value: number, min: number, max: number): number {
    return value * (max - min) + min;
  }
}

/**
 * Z-score normalization strategy.
 * Standardizes values to zero mean and unit variance.
 *
 * Mathematical formula: normalized = (x - μ) / σ
 *
 * @class ZScoreNormalizationStrategy
 * @implements {INormalizationStrategy}
 */
class ZScoreNormalizationStrategy implements INormalizationStrategy {
  /** @inheritdoc */
  public readonly name: "z-score" = "z-score";

  /** Epsilon for numerical stability */
  private readonly epsilon: number = 1e-10;

  /** Maximum allowed z-score to prevent extreme outliers */
  private readonly maxZScore: number = 5;

  /**
   * Standardizes a value using z-score normalization.
   *
   * Mathematical formula: (x - μ) / σ
   *
   * @param {number} value - The input value
   * @param {number} _min - Unused (for interface compatibility)
   * @param {number} _max - Unused (for interface compatibility)
   * @param {number} mean - The mean of observed values
   * @param {number} std - The standard deviation of observed values
   * @returns {number} The standardized value, clamped to ±5 standard deviations
   */
  public normalize(
    value: number,
    _min: number,
    _max: number,
    mean: number,
    std: number,
  ): number {
    // Handle zero variance case
    if (std < this.epsilon) {
      return 0;
    }

    const normalized = (value - mean) / std;

    // Clamp to prevent extreme outliers
    return Math.max(-this.maxZScore, Math.min(this.maxZScore, normalized));
  }

  /**
   * Denormalizes a z-score to original scale.
   *
   * Mathematical formula: z × σ + μ
   *
   * @param {number} value - The standardized value (z-score)
   * @param {number} _min - Unused (for interface compatibility)
   * @param {number} _max - Unused (for interface compatibility)
   * @param {number} mean - The mean of observed values
   * @param {number} std - The standard deviation of observed values
   * @returns {number} The denormalized value
   */
  public denormalize(
    value: number,
    _min: number,
    _max: number,
    mean: number,
    std: number,
  ): number {
    return value * std + mean;
  }
}

// ==================== NORMALIZATION MANAGER ====================

/**
 * Manages incremental normalization statistics and normalization operations.
 * Uses Welford's algorithm for numerically stable online variance computation.
 *
 * Welford's Algorithm for incremental mean and variance:
 * - δ = x - μₙ₋₁
 * - μₙ = μₙ₋₁ + δ/n
 * - M₂ₙ = M₂ₙ₋₁ + δ(x - μₙ)
 * - σ² = M₂/(n-1) (sample variance)
 *
 * @example
 * ```typescript
 * const strategy = new MinMaxNormalizationStrategy();
 * const normalizer = new NormalizationManager(strategy);
 * normalizer.initialize(3);
 * normalizer.updateStatistics(inputVector);
 * normalizer.normalizeInPlace(inputVector, outputVector);
 * ```
 *
 * @class NormalizationManager
 */
class NormalizationManager {
  /** The normalization strategy (Strategy Pattern) */
  private readonly strategy: INormalizationStrategy;

  /** Minimum values for each dimension */
  private min: Float64Array;

  /** Maximum values for each dimension */
  private max: Float64Array;

  /** Running mean for each dimension (Welford's algorithm) */
  private mean: Float64Array;

  /** Sum of squared differences M₂ (Welford's algorithm) */
  private m2: Float64Array;

  /** Sample count */
  private count: number = 0;

  /** Input dimension */
  private dimension: number = 0;

  /** Initialization flag */
  private initialized: boolean = false;

  /** Preallocated buffer for standard deviation computation */
  private stdBuffer: Float64Array;

  /**
   * Creates a new NormalizationManager with dependency injection.
   *
   * @param {INormalizationStrategy} strategy - The normalization strategy to use
   * @param {number} [initialDimension=0] - Initial dimension (lazy initialization if 0)
   */
  constructor(strategy: INormalizationStrategy, initialDimension: number = 0) {
    this.strategy = strategy;
    this.dimension = initialDimension;
    this.min = new Float64Array(initialDimension);
    this.max = new Float64Array(initialDimension);
    this.mean = new Float64Array(initialDimension);
    this.m2 = new Float64Array(initialDimension);
    this.stdBuffer = new Float64Array(initialDimension);

    if (initialDimension > 0) {
      this.min.fill(Infinity);
      this.max.fill(-Infinity);
    }
  }

  /**
   * Initializes or reinitializes the manager for a specific dimension.
   *
   * Time complexity: O(d) where d is dimension
   * Space complexity: O(d)
   *
   * @param {number} dimension - The input dimension
   */
  public initialize(dimension: number): void {
    if (this.dimension !== dimension || !this.initialized) {
      this.dimension = dimension;
      this.min = new Float64Array(dimension);
      this.max = new Float64Array(dimension);
      this.mean = new Float64Array(dimension);
      this.m2 = new Float64Array(dimension);
      this.stdBuffer = new Float64Array(dimension);
      this.min.fill(Infinity);
      this.max.fill(-Infinity);
      this.count = 0;
      this.initialized = true;
    }
  }

  /**
   * Updates running statistics incrementally using Welford's algorithm.
   * This allows tracking mean and variance without storing all data points.
   *
   * Welford's algorithm steps:
   * 1. δ = x - μ (delta from current mean)
   * 2. μ = μ + δ/n (update mean)
   * 3. δ₂ = x - μ (delta from new mean)
   * 4. M₂ = M₂ + δ × δ₂ (update sum of squared differences)
   *
   * Time complexity: O(d) where d is input dimension
   * Space complexity: O(1)
   *
   * @param {Float64Array} input - The new input sample
   */
  public updateStatistics(input: Float64Array): void {
    if (!this.initialized) {
      this.initialize(input.length);
    }

    this.count++;
    const n = this.count;

    for (let i = 0; i < this.dimension; i++) {
      const x = input[i];

      // Update min/max
      if (x < this.min[i]) this.min[i] = x;
      if (x > this.max[i]) this.max[i] = x;

      // Welford's online algorithm for mean and variance
      const delta = x - this.mean[i];
      this.mean[i] += delta / n;
      const delta2 = x - this.mean[i];
      this.m2[i] += delta * delta2;
    }
  }

  /**
   * Computes standard deviation from the running M₂ values.
   * Uses sample variance (n-1 denominator) for unbiased estimation.
   *
   * Mathematical formula: σ = √(M₂/(n-1))
   *
   * @returns {Float64Array} The standard deviation for each dimension (internal buffer)
   * @private
   */
  private computeStd(): Float64Array {
    const n = this.count;

    for (let i = 0; i < this.dimension; i++) {
      if (n > 1) {
        this.stdBuffer[i] = Math.sqrt(this.m2[i] / (n - 1));
      } else {
        this.stdBuffer[i] = 1; // Default to 1 for single sample to avoid division by zero
      }
    }

    return this.stdBuffer;
  }

  /**
   * Normalizes an input vector in place using the configured strategy.
   *
   * Time complexity: O(d) where d is input dimension
   * Space complexity: O(1)
   *
   * @param {Float64Array} input - The input vector
   * @param {Float64Array} output - The output vector (can be same as input for in-place)
   */
  public normalizeInPlace(input: Float64Array, output: Float64Array): void {
    if (!this.initialized || this.count === 0) {
      // No statistics yet, copy input to output
      for (let i = 0; i < input.length; i++) {
        output[i] = input[i];
      }
      return;
    }

    const std = this.computeStd();

    for (let i = 0; i < this.dimension; i++) {
      output[i] = this.strategy.normalize(
        input[i],
        this.min[i],
        this.max[i],
        this.mean[i],
        std[i],
      );
    }
  }

  /**
   * Returns the current normalization statistics.
   * Creates new arrays to avoid external modification of internal state.
   *
   * Time complexity: O(d)
   * Space complexity: O(d)
   *
   * @returns {INormalizationStats} The current statistics
   */
  public getStats(): INormalizationStats {
    const std = this.computeStd();

    return {
      min: Array.from(this.min),
      max: Array.from(this.max),
      mean: Array.from(this.mean),
      std: Array.from(std),
      count: this.count,
    };
  }

  /**
   * Resets all statistics to initial state.
   *
   * Time complexity: O(d)
   * Space complexity: O(1)
   */
  public reset(): void {
    if (this.dimension > 0) {
      this.min.fill(Infinity);
      this.max.fill(-Infinity);
      this.mean.fill(0);
      this.m2.fill(0);
    }
    this.count = 0;
    this.initialized = false;
  }

  /**
   * Gets the strategy name.
   * @returns {'none' | 'min-max' | 'z-score'} The strategy name
   */
  public getStrategyName(): "none" | "min-max" | "z-score" {
    return this.strategy.name;
  }

  /**
   * Gets the count of samples processed.
   * @returns {number} The sample count
   */
  public getCount(): number {
    return this.count;
  }

  /**
   * Checks if the manager is initialized.
   * @returns {boolean} True if initialized
   */
  public isInitialized(): boolean {
    return this.initialized;
  }
}

// ==================== POLYNOMIAL FEATURE GENERATOR ====================

/**
 * Generates polynomial features from input vectors up to a specified degree.
 * Uses precomputed exponent mappings for efficient feature generation.
 *
 * For degree d and input dimension n, generates all monomials where the
 * sum of exponents is at most d. The total number of features is C(n+d, d).
 *
 * Example for degree 2, input [x₁, x₂]:
 * Features: [1, x₁, x₂, x₁², x₁x₂, x₂²]
 *
 * @example
 * ```typescript
 * const generator = new PolynomialFeatureGenerator(2);
 * generator.initialize(2); // 2 input dimensions
 * const features = generator.generate(inputVector, outputBuffer);
 * // features = [1, x1, x2, x1², x1*x2, x2²]
 * ```
 *
 * @class PolynomialFeatureGenerator
 * @implements {IPolynomialFeatureGenerator}
 */
class PolynomialFeatureGenerator implements IPolynomialFeatureGenerator {
  /** Maximum polynomial degree */
  private readonly degree: number;

  /** Input dimension */
  private inputDimension: number = 0;

  /** Total number of polynomial features */
  private featureCount: number = 0;

  /** Initialization flag */
  private initialized: boolean = false;

  /**
   * Precomputed exponent combinations for each feature.
   * exponents[featureIndex][inputIndex] = exponent for that input dimension
   */
  private exponents: Uint8Array[] = [];

  /** Preallocated buffer for feature generation */
  private featureBuffer: Float64Array;

  /**
   * Creates a new PolynomialFeatureGenerator.
   *
   * @param {number} degree - The maximum polynomial degree (minimum: 1)
   * @param {number} [inputDimension=0] - Initial input dimension (lazy init if 0)
   * @throws {Error} If degree is less than 1
   */
  constructor(degree: number, inputDimension: number = 0) {
    if (degree < 1) {
      throw new Error("Polynomial degree must be at least 1");
    }

    this.degree = degree;
    this.featureBuffer = new Float64Array(0);

    if (inputDimension > 0) {
      this.initialize(inputDimension);
    }
  }

  /**
   * Initializes the generator for a specific input dimension.
   * Precomputes all exponent combinations using an iterative algorithm.
   *
   * The number of features is the multiset coefficient C(n+d, d) where:
   * - n is the input dimension
   * - d is the polynomial degree
   *
   * Time complexity: O(C(n+d, d) × n)
   * Space complexity: O(C(n+d, d) × n)
   *
   * @param {number} dimension - The input dimension
   */
  public initialize(dimension: number): void {
    if (this.inputDimension === dimension && this.initialized) {
      return;
    }

    this.inputDimension = dimension;
    this.exponents = [];
    this.generateExponentCombinations(dimension, this.degree);
    this.featureCount = this.exponents.length;
    this.featureBuffer = new Float64Array(this.featureCount);
    this.initialized = true;
  }

  /**
   * Generates all exponent combinations iteratively.
   * Uses a stack-based approach to avoid recursion overhead.
   *
   * Algorithm:
   * - Start with all zeros (constant term)
   * - Enumerate all multi-indices [e₁, e₂, ..., eₙ] where Σeᵢ ≤ d
   *
   * @param {number} dimension - Input dimension
   * @param {number} maxDegree - Maximum total degree
   * @private
   */
  private generateExponentCombinations(
    dimension: number,
    maxDegree: number,
  ): void {
    // Use iterative approach with stack to avoid recursion
    const stack: {
      dim: number;
      remainingDegree: number;
      exponents: number[];
    }[] = [];
    stack.push({
      dim: 0,
      remainingDegree: maxDegree,
      exponents: new Array(dimension).fill(0),
    });

    while (stack.length > 0) {
      const current = stack.pop()!;

      if (current.dim === dimension) {
        // Complete combination found - store as Uint8Array for memory efficiency
        this.exponents.push(new Uint8Array(current.exponents));
        continue;
      }

      // Generate all possible exponents for current dimension
      for (let exp = current.remainingDegree; exp >= 0; exp--) {
        const newExponents = current.exponents.slice();
        newExponents[current.dim] = exp;
        stack.push({
          dim: current.dim + 1,
          remainingDegree: current.remainingDegree - exp,
          exponents: newExponents,
        });
      }
    }

    // Sort for consistent ordering: by total degree, then lexicographically
    this.exponents.sort((a, b) => {
      let sumA = 0, sumB = 0;
      for (let i = 0; i < a.length; i++) {
        sumA += a[i];
        sumB += b[i];
      }

      if (sumA !== sumB) return sumA - sumB;

      // Lexicographic comparison
      for (let i = 0; i < a.length; i++) {
        if (a[i] !== b[i]) return a[i] - b[i];
      }

      return 0;
    });
  }

  /**
   * Generates polynomial features from an input vector.
   *
   * For each feature f with exponents [e₁, e₂, ..., eₙ]:
   * feature_f = x₁^e₁ × x₂^e₂ × ... × xₙ^eₙ
   *
   * Optimizations:
   * - Skip multiplication for zero exponents
   * - Use precomputed exponent arrays
   * - Support in-place output buffer
   *
   * Time complexity: O(f × d) where f is feature count, d is max degree
   * Space complexity: O(1) when using preallocated buffer
   *
   * @param {Float64Array} input - The normalized input vector
   * @param {Float64Array} [output] - Optional preallocated output buffer
   * @returns {Float64Array} The polynomial features
   */
  public generate(input: Float64Array, output?: Float64Array): Float64Array {
    if (!this.initialized) {
      this.initialize(input.length);
    }

    const result = output || this.featureBuffer;

    for (let i = 0; i < this.featureCount; i++) {
      const exp = this.exponents[i];
      let feature = 1.0;

      // Compute product: x[0]^exp[0] × x[1]^exp[1] × ... × x[n-1]^exp[n-1]
      for (let j = 0; j < this.inputDimension; j++) {
        const e = exp[j];

        if (e > 0) {
          // Optimization: use multiplication for small exponents, pow for larger
          if (e === 1) {
            feature *= input[j];
          } else if (e === 2) {
            feature *= input[j] * input[j];
          } else {
            feature *= Math.pow(input[j], e);
          }
        }
      }

      result[i] = feature;
    }

    return result;
  }

  /**
   * Returns the number of polynomial features.
   *
   * @returns {number} The feature count (C(n+d, d))
   */
  public getFeatureCount(): number {
    return this.featureCount;
  }

  /**
   * Returns the polynomial degree.
   *
   * @returns {number} The maximum polynomial degree
   */
  public getDegree(): number {
    return this.degree;
  }

  /**
   * Checks if the generator is initialized.
   *
   * @returns {boolean} True if initialized with a dimension
   */
  public isInitialized(): boolean {
    return this.initialized;
  }

  /**
   * Resets the generator state for reuse.
   */
  public reset(): void {
    this.inputDimension = 0;
    this.featureCount = 0;
    this.exponents = [];
    this.initialized = false;
    this.featureBuffer = new Float64Array(0);
  }
}

// ==================== COVARIANCE MANAGER ====================

/**
 * Manages the covariance matrix P for the RLS algorithm.
 * Handles initialization, gain computation, updates, and regularization.
 *
 * The covariance matrix represents the uncertainty in the weight estimates.
 * It is stored as a flat Float64Array in row-major order for efficiency.
 *
 * RLS Covariance Update:
 * - Gain: k = P·φ / (λ + φᵀ·P·φ)
 * - Update: P = (P - k·(P·φ)ᵀ) / λ
 *
 * @example
 * ```typescript
 * const covManager = new CovarianceManager(matrixOps);
 * covManager.initialize(featureCount, 1000);
 * const k = covManager.computeGain(phi, lambda);
 * covManager.update(k, lambda);
 * ```
 *
 * @class CovarianceManager
 * @implements {ICovarianceManager}
 */
class CovarianceManager implements ICovarianceManager {
  /** Matrix operations dependency */
  private readonly matrixOps: IMatrixOperations;

  /** Covariance matrix as flat array (row-major) */
  private covariance: Float64Array;

  /** Matrix dimension (number of features) */
  private size: number = 0;

  /** Initialization flag */
  private initialized: boolean = false;

  /** Preallocated buffer for P·φ computation */
  private Pphi: Float64Array;

  /** Preallocated buffer for gain vector k */
  private gainBuffer: Float64Array;

  /**
   * Creates a new CovarianceManager.
   *
   * @param {IMatrixOperations} matrixOps - Matrix operations implementation (DI)
   */
  constructor(matrixOps: IMatrixOperations) {
    this.matrixOps = matrixOps;
    this.covariance = new Float64Array(0);
    this.Pphi = new Float64Array(0);
    this.gainBuffer = new Float64Array(0);
  }

  /**
   * Initializes the covariance matrix as a scaled identity matrix.
   * P = initialValue × I
   *
   * A large initial value indicates high uncertainty in initial weights.
   *
   * Time complexity: O(n²)
   * Space complexity: O(n²)
   *
   * @param {number} size - The size of the matrix (number of features)
   * @param {number} initialValue - The initial diagonal value (e.g., 1000)
   */
  public initialize(size: number, initialValue: number): void {
    this.size = size;
    this.covariance = this.matrixOps.createIdentityDiagonal(size, initialValue);
    this.Pphi = new Float64Array(size);
    this.gainBuffer = new Float64Array(size);
    this.initialized = true;
  }

  /**
   * Computes the RLS gain vector.
   *
   * Mathematical formula: k = P·φ / (λ + φᵀ·P·φ)
   *
   * The gain vector determines how much to update weights based on
   * the prediction error. Higher gain = more weight update.
   *
   * Time complexity: O(n²) for matrix-vector multiply
   * Space complexity: O(1) using preallocated buffers
   *
   * @param {Float64Array} phi - The feature vector φ
   * @param {number} lambda - The forgetting factor λ
   * @returns {Float64Array} The gain vector k
   */
  public computeGain(phi: Float64Array, lambda: number): Float64Array {
    // Step 1: Compute P·φ
    this.matrixOps.symmetricMatrixVectorMultiply(
      this.covariance,
      phi,
      this.Pphi,
      this.size,
    );

    // Step 2: Compute φᵀ·P·φ (scalar)
    const phiTPphi = this.matrixOps.dotProduct(phi, this.Pphi, this.size);

    // Step 3: Compute k = P·φ / (λ + φᵀ·P·φ)
    const denominator = lambda + phiTPphi;

    for (let i = 0; i < this.size; i++) {
      this.gainBuffer[i] = this.Pphi[i] / denominator;
    }

    return this.gainBuffer;
  }

  /**
   * Updates the covariance matrix using the RLS formula.
   *
   * Mathematical formula: P = (P - k·(P·φ)ᵀ) / λ
   *
   * This reduces uncertainty in directions where we've seen data,
   * while the forgetting factor λ < 1 slowly increases uncertainty
   * to allow adaptation to changing data.
   *
   * Time complexity: O(n²)
   * Space complexity: O(1)
   *
   * @param {Float64Array} k - The gain vector (from computeGain)
   * @param {number} lambda - The forgetting factor λ
   */
  public update(k: Float64Array, lambda: number): void {
    // Pphi was already computed in computeGain
    this.matrixOps.rankOneUpdateSymmetric(
      this.covariance,
      k,
      this.Pphi,
      this.size,
      lambda,
    );
  }

  /**
   * Applies regularization to the diagonal of the covariance matrix.
   * Ensures numerical stability by preventing diagonal values from becoming too small.
   *
   * This prevents the covariance matrix from becoming singular and
   * maintains the model's ability to adapt to new data.
   *
   * Time complexity: O(n)
   * Space complexity: O(1)
   *
   * @param {number} regularization - The minimum diagonal value
   */
  public applyRegularization(regularization: number): void {
    for (let i = 0; i < this.size; i++) {
      const idx = i * this.size + i;

      if (this.covariance[idx] < regularization) {
        this.covariance[idx] = regularization;
      }
    }
  }

  /**
   * Computes φᵀ·P·φ for confidence interval calculation.
   * This scalar represents the prediction variance.
   *
   * Time complexity: O(n²)
   * Space complexity: O(1)
   *
   * @param {Float64Array} phi - The feature vector
   * @returns {number} The scalar φᵀ·P·φ (prediction variance)
   */
  public computePhiTPphi(phi: Float64Array): number {
    this.matrixOps.symmetricMatrixVectorMultiply(
      this.covariance,
      phi,
      this.Pphi,
      this.size,
    );

    return this.matrixOps.dotProduct(phi, this.Pphi, this.size);
  }

  /**
   * Checks if the manager is initialized.
   *
   * @returns {boolean} True if initialized
   */
  public isInitialized(): boolean {
    return this.initialized;
  }

  /**
   * Resets the covariance matrix state.
   */
  public reset(): void {
    this.covariance = new Float64Array(0);
    this.Pphi = new Float64Array(0);
    this.gainBuffer = new Float64Array(0);
    this.size = 0;
    this.initialized = false;
  }
}

// ==================== WEIGHT MANAGER ====================

/**
 * Manages the weight matrix W for the RLS algorithm.
 * Supports multiple output dimensions with efficient storage.
 *
 * Weights are organized as W[outputDim][featureIndex] using Float64Array
 * for each output dimension.
 *
 * RLS Weight Update: w = w + k·e
 *
 * @example
 * ```typescript
 * const weightManager = new WeightManager(matrixOps);
 * weightManager.initialize(featureCount, outputDim);
 * const predicted = weightManager.predict(phi);
 * weightManager.update(k, error);
 * ```
 *
 * @class WeightManager
 * @implements {IWeightManager}
 */
class WeightManager implements IWeightManager {
  /** Matrix operations dependency */
  private readonly matrixOps: IMatrixOperations;

  /** Weight vectors for each output dimension */
  private weights: Float64Array[];

  /** Number of polynomial features */
  private featureCount: number = 0;

  /** Number of output dimensions */
  private outputDimension: number = 0;

  /** Initialization flag */
  private initialized: boolean = false;

  /** Preallocated buffer for predictions */
  private predictionBuffer: Float64Array;

  /**
   * Creates a new WeightManager.
   *
   * @param {IMatrixOperations} matrixOps - Matrix operations implementation (DI)
   */
  constructor(matrixOps: IMatrixOperations) {
    this.matrixOps = matrixOps;
    this.weights = [];
    this.predictionBuffer = new Float64Array(0);
  }

  /**
   * Initializes the weight matrix with zeros.
   *
   * Time complexity: O(f × o) where f is feature count, o is output dimension
   * Space complexity: O(f × o)
   *
   * @param {number} featureCount - Number of polynomial features
   * @param {number} outputDimension - Number of output dimensions
   */
  public initialize(featureCount: number, outputDimension: number): void {
    this.featureCount = featureCount;
    this.outputDimension = outputDimension;
    this.weights = new Array(outputDimension);

    for (let i = 0; i < outputDimension; i++) {
      this.weights[i] = new Float64Array(featureCount);
    }

    this.predictionBuffer = new Float64Array(outputDimension);
    this.initialized = true;
  }

  /**
   * Computes prediction: ŷ = φᵀ·w for all output dimensions.
   *
   * Mathematical formula: ŷᵢ = Σⱼ(wᵢⱼ × φⱼ) for each output i
   *
   * Time complexity: O(f × o)
   * Space complexity: O(1) using preallocated buffer
   *
   * @param {Float64Array} phi - The feature vector φ
   * @param {Float64Array} [output] - Optional output buffer
   * @returns {Float64Array} The predicted values for each output dimension
   */
  public predict(phi: Float64Array, output?: Float64Array): Float64Array {
    const result = output || this.predictionBuffer;

    for (let i = 0; i < this.outputDimension; i++) {
      result[i] = this.matrixOps.dotProduct(
        this.weights[i],
        phi,
        this.featureCount,
      );
    }

    return result;
  }

  /**
   * Updates weights using the RLS update rule.
   *
   * Mathematical formula: wᵢ = wᵢ + k × eᵢ for each output dimension i
   *
   * Time complexity: O(f × o)
   * Space complexity: O(1)
   *
   * @param {Float64Array} k - The gain vector
   * @param {Float64Array} error - The prediction error for each output dimension
   */
  public update(k: Float64Array, error: Float64Array): void {
    for (let i = 0; i < this.outputDimension; i++) {
      this.matrixOps.addScaledInPlace(
        this.weights[i],
        k,
        error[i],
        this.featureCount,
      );
    }
  }

  /**
   * Returns the current weights as a 2D array.
   * Creates new arrays to protect internal state.
   *
   * Time complexity: O(f × o)
   * Space complexity: O(f × o)
   *
   * @returns {number[][]} The weight matrix [outputDim][featureIndex]
   */
  public getWeights(): number[][] {
    const result: number[][] = new Array(this.outputDimension);

    for (let i = 0; i < this.outputDimension; i++) {
      result[i] = Array.from(this.weights[i]);
    }

    return result;
  }

  /**
   * Checks if the manager is initialized.
   *
   * @returns {boolean} True if initialized
   */
  public isInitialized(): boolean {
    return this.initialized;
  }

  /**
   * Gets the output dimension.
   *
   * @returns {number} The output dimension
   */
  public getOutputDimension(): number {
    return this.outputDimension;
  }

  /**
   * Resets the weight manager state.
   */
  public reset(): void {
    this.weights = [];
    this.predictionBuffer = new Float64Array(0);
    this.featureCount = 0;
    this.outputDimension = 0;
    this.initialized = false;
  }
}

// ==================== STATISTICS TRACKER ====================

/**
 * Tracks model statistics incrementally including R-squared and RMSE.
 * Uses Welford's algorithm for numerically stable incremental variance computation.
 *
 * R-squared (Coefficient of Determination):
 * R² = 1 - SS_res / SS_tot
 * where SS_res = Σ(yᵢ - ŷᵢ)² and SS_tot = Σ(yᵢ - ȳ)²
 *
 * RMSE (Root Mean Squared Error):
 * RMSE = √(SS_res / n)
 *
 * @example
 * ```typescript
 * const stats = new StatisticsTracker();
 * stats.initialize(outputDim);
 * stats.update(actualValues, predictedValues);
 * console.log(`R²: ${stats.getRSquared()}, RMSE: ${stats.getRMSE()}`);
 * ```
 *
 * @class StatisticsTracker
 * @implements {IStatisticsTracker}
 */
class StatisticsTracker implements IStatisticsTracker {
  /** Sum of squared residuals for each output */
  private ssRes: Float64Array;

  /** Running mean of y for each output (Welford's) */
  private meanY: Float64Array;

  /** Welford's M₂ for variance of y (= SS_tot when done) */
  private m2Y: Float64Array;

  /** Sample count */
  private count: number = 0;

  /** Output dimension */
  private outputDimension: number = 0;

  /** Initialization flag */
  private initialized: boolean = false;

  /**
   * Creates a new StatisticsTracker.
   */
  constructor() {
    this.ssRes = new Float64Array(0);
    this.meanY = new Float64Array(0);
    this.m2Y = new Float64Array(0);
  }

  /**
   * Initializes the tracker for a specific output dimension.
   *
   * Time complexity: O(o)
   * Space complexity: O(o)
   *
   * @param {number} outputDimension - Number of output dimensions
   */
  public initialize(outputDimension: number): void {
    this.outputDimension = outputDimension;
    this.ssRes = new Float64Array(outputDimension);
    this.meanY = new Float64Array(outputDimension);
    this.m2Y = new Float64Array(outputDimension);
    this.count = 0;
    this.initialized = true;
  }

  /**
   * Updates statistics with a new sample.
   * Uses Welford's algorithm for stable online variance computation.
   *
   * Algorithm:
   * 1. SS_res += (y - ŷ)²
   * 2. δ = y - ȳ
   * 3. ȳ = ȳ + δ/n
   * 4. M₂ = M₂ + δ(y - ȳ)
   *
   * Time complexity: O(o)
   * Space complexity: O(1)
   *
   * @param {Float64Array} actual - The actual y values
   * @param {Float64Array} predicted - The predicted ŷ values
   */
  public update(actual: Float64Array, predicted: Float64Array): void {
    if (!this.initialized) {
      this.initialize(actual.length);
    }

    this.count++;
    const n = this.count;

    for (let i = 0; i < this.outputDimension; i++) {
      const y = actual[i];
      const yHat = predicted[i];
      const residual = y - yHat;

      // Update SS_res (sum of squared residuals)
      this.ssRes[i] += residual * residual;

      // Welford's algorithm for mean and M2
      const delta = y - this.meanY[i];
      this.meanY[i] += delta / n;
      const delta2 = y - this.meanY[i];
      this.m2Y[i] += delta * delta2;
    }
  }

  /**
   * Computes R-squared (coefficient of determination).
   *
   * R² = 1 - SS_res / SS_tot
   *
   * Returns the average R² across all output dimensions.
   * R² ranges from 0 to 1, where 1 indicates perfect fit.
   *
   * Time complexity: O(o)
   * Space complexity: O(1)
   *
   * @returns {number} The average R-squared across all output dimensions
   */
  public getRSquared(): number {
    if (this.count < 2) {
      return 0;
    }

    let totalRSquared = 0;

    for (let i = 0; i < this.outputDimension; i++) {
      // SS_tot = M₂ (from Welford's algorithm)
      const ssTot = this.m2Y[i];

      if (ssTot < 1e-10) {
        // Constant target: R² = 1 if no residuals, 0 otherwise
        totalRSquared += this.ssRes[i] < 1e-10 ? 1 : 0;
      } else {
        const rSquared = 1 - this.ssRes[i] / ssTot;
        // Clamp to [0, 1] to handle numerical issues
        totalRSquared += Math.max(0, Math.min(1, rSquared));
      }
    }

    return totalRSquared / this.outputDimension;
  }

  /**
   * Computes RMSE (Root Mean Squared Error).
   *
   * RMSE = √(SS_res / n)
   *
   * Returns the average RMSE across all output dimensions.
   *
   * Time complexity: O(o)
   * Space complexity: O(1)
   *
   * @returns {number} The average RMSE across all output dimensions
   */
  public getRMSE(): number {
    if (this.count === 0) {
      return 0;
    }

    let totalRMSE = 0;

    for (let i = 0; i < this.outputDimension; i++) {
      const mse = this.ssRes[i] / this.count;
      totalRMSE += Math.sqrt(mse);
    }

    return totalRMSE / this.outputDimension;
  }

  /**
   * Gets the sample count.
   *
   * @returns {number} The number of samples processed
   */
  public getCount(): number {
    return this.count;
  }

  /**
   * Checks if the tracker is initialized.
   *
   * @returns {boolean} True if initialized
   */
  public isInitialized(): boolean {
    return this.initialized;
  }

  /**
   * Resets all statistics.
   */
  public reset(): void {
    this.ssRes = new Float64Array(0);
    this.meanY = new Float64Array(0);
    this.m2Y = new Float64Array(0);
    this.count = 0;
    this.outputDimension = 0;
    this.initialized = false;
  }
}

// ==================== PREDICTION ENGINE ====================

/**
 * Handles prediction calculations including confidence intervals.
 * Uses the covariance matrix for uncertainty estimation.
 *
 * Confidence intervals are computed as:
 * CI = ŷ ± t_{α/2,df} × SE
 * where SE = √(φᵀ·P·φ) is the standard error
 *
 * Uses t-distribution for small samples (n ≤ 30) and z-distribution for large samples.
 *
 * @example
 * ```typescript
 * const engine = new PredictionEngine(covManager, weightManager);
 * engine.initialize(outputDim);
 * const result = engine.predict(features, 0.95, sampleCount);
 * ```
 *
 * @class PredictionEngine
 */
class PredictionEngine {
  /** Covariance manager dependency */
  private readonly covarianceManager: ICovarianceManager;

  /** Weight manager dependency */
  private readonly weightManager: IWeightManager;

  /** T-distribution critical values: Map<df, Map<confidence, critical>> */
  private readonly tCriticalValues: Map<number, Map<number, number>>;

  /** Z-distribution critical values: Map<confidence, critical> */
  private readonly zCriticalValues: Map<number, number>;

  /** Preallocated buffers */
  private predictedBuffer: Float64Array;
  private lowerBoundBuffer: Float64Array;
  private upperBoundBuffer: Float64Array;
  private stdErrorBuffer: Float64Array;

  /**
   * Creates a new PredictionEngine.
   *
   * @param {ICovarianceManager} covarianceManager - Covariance manager (DI)
   * @param {IWeightManager} weightManager - Weight manager (DI)
   */
  constructor(
    covarianceManager: ICovarianceManager,
    weightManager: IWeightManager,
  ) {
    this.covarianceManager = covarianceManager;
    this.weightManager = weightManager;

    // Initialize critical values lookup tables
    this.tCriticalValues = new Map();
    this.zCriticalValues = new Map([
      [0.90, 1.645],
      [0.95, 1.960],
      [0.99, 2.576],
    ]);

    this.initializeTCriticalValues();

    this.predictedBuffer = new Float64Array(0);
    this.lowerBoundBuffer = new Float64Array(0);
    this.upperBoundBuffer = new Float64Array(0);
    this.stdErrorBuffer = new Float64Array(0);
  }

  /**
   * Initializes t-distribution critical values for small samples.
   * Two-tailed critical values for selected degrees of freedom.
   *
   * @private
   */
  private initializeTCriticalValues(): void {
    const criticals: Record<number, Record<number, number>> = {
      1: { 0.90: 6.314, 0.95: 12.706, 0.99: 63.657 },
      2: { 0.90: 2.920, 0.95: 4.303, 0.99: 9.925 },
      3: { 0.90: 2.353, 0.95: 3.182, 0.99: 5.841 },
      4: { 0.90: 2.132, 0.95: 2.776, 0.99: 4.604 },
      5: { 0.90: 2.015, 0.95: 2.571, 0.99: 4.032 },
      6: { 0.90: 1.943, 0.95: 2.447, 0.99: 3.707 },
      7: { 0.90: 1.895, 0.95: 2.365, 0.99: 3.499 },
      8: { 0.90: 1.860, 0.95: 2.306, 0.99: 3.355 },
      9: { 0.90: 1.833, 0.95: 2.262, 0.99: 3.250 },
      10: { 0.90: 1.812, 0.95: 2.228, 0.99: 3.169 },
      15: { 0.90: 1.753, 0.95: 2.131, 0.99: 2.947 },
      20: { 0.90: 1.725, 0.95: 2.086, 0.99: 2.845 },
      25: { 0.90: 1.708, 0.95: 2.060, 0.99: 2.787 },
      30: { 0.90: 1.697, 0.95: 2.042, 0.99: 2.750 },
    };

    for (const df of Object.keys(criticals)) {
      const dfNum = parseInt(df);
      const innerMap = new Map<number, number>();

      for (const conf of Object.keys(criticals[dfNum])) {
        innerMap.set(parseFloat(conf), criticals[dfNum][parseFloat(conf)]);
      }

      this.tCriticalValues.set(dfNum, innerMap);
    }
  }

  /**
   * Gets the critical value for confidence interval calculation.
   * Uses t-distribution for n ≤ 30, z-distribution for n > 30.
   *
   * @param {number} sampleCount - Number of samples
   * @param {number} confidenceLevel - Confidence level (e.g., 0.95)
   * @returns {number} The critical value
   * @private
   */
  private getCriticalValue(
    sampleCount: number,
    confidenceLevel: number,
  ): number {
    // Use z-distribution for large samples
    if (sampleCount > 30) {
      return this.zCriticalValues.get(confidenceLevel) ||
        this.interpolateCritical(confidenceLevel, true);
    }

    // Use t-distribution for small samples
    const df = Math.max(1, sampleCount - 1);
    const availableDfs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30];

    // Find closest available degrees of freedom
    let closestDf = 1;
    for (const d of availableDfs) {
      if (d <= df) closestDf = d;
    }

    const dfMap = this.tCriticalValues.get(closestDf);
    if (dfMap) {
      return dfMap.get(confidenceLevel) ||
        this.interpolateCritical(confidenceLevel, false, dfMap);
    }

    return 1.96; // Fallback
  }

  /**
   * Interpolates critical value for non-standard confidence levels.
   *
   * @param {number} confidenceLevel - The confidence level
   * @param {boolean} useZ - Whether to use z-distribution
   * @param {Map<number, number>} [tMap] - T-distribution map for specific df
   * @returns {number} Interpolated critical value
   * @private
   */
  private interpolateCritical(
    confidenceLevel: number,
    useZ: boolean,
    tMap?: Map<number, number>,
  ): number {
    const levels = [0.90, 0.95, 0.99];
    const values = useZ
      ? levels.map((l) => this.zCriticalValues.get(l) || 2)
      : levels.map((l) => tMap?.get(l) || 2);

    // Linear interpolation between known values
    for (let i = 0; i < levels.length - 1; i++) {
      if (confidenceLevel >= levels[i] && confidenceLevel <= levels[i + 1]) {
        const t = (confidenceLevel - levels[i]) / (levels[i + 1] - levels[i]);
        return values[i] + t * (values[i + 1] - values[i]);
      }
    }

    if (confidenceLevel < levels[0]) {
      return values[0] * (confidenceLevel / levels[0]);
    }

    return values[values.length - 1];
  }

  /**
   * Initializes prediction buffers for the given output dimension.
   *
   * @param {number} outputDimension - Number of output dimensions
   */
  public initialize(outputDimension: number): void {
    this.predictedBuffer = new Float64Array(outputDimension);
    this.lowerBoundBuffer = new Float64Array(outputDimension);
    this.upperBoundBuffer = new Float64Array(outputDimension);
    this.stdErrorBuffer = new Float64Array(outputDimension);
  }

  /**
   * Makes a prediction with confidence intervals.
   *
   * Computes:
   * 1. Point prediction: ŷ = φᵀ·w
   * 2. Standard error: SE = √(φᵀ·P·φ)
   * 3. Confidence interval: ŷ ± t_{α/2} × SE
   *
   * Time complexity: O(f² + f×o) where f is feature count, o is output dimension
   * Space complexity: O(o)
   *
   * @param {Float64Array} phi - The polynomial feature vector
   * @param {number} confidenceLevel - The confidence level (e.g., 0.95)
   * @param {number} sampleCount - Number of training samples
   * @returns {ISinglePrediction} The prediction with confidence bounds
   */
  public predict(
    phi: Float64Array,
    confidenceLevel: number,
    sampleCount: number,
  ): ISinglePrediction {
    // Compute point prediction: ŷ = φᵀ·w
    this.weightManager.predict(phi, this.predictedBuffer);

    // Compute standard error: SE = √(φᵀ·P·φ)
    const phiTPphi = this.covarianceManager.computePhiTPphi(phi);
    const standardError = Math.sqrt(Math.max(0, phiTPphi));

    // Get critical value for confidence interval
    const criticalValue = this.getCriticalValue(sampleCount, confidenceLevel);
    const margin = criticalValue * standardError;

    // Compute bounds for each output dimension
    const outputDim = this.weightManager.getOutputDimension();

    for (let i = 0; i < outputDim; i++) {
      this.stdErrorBuffer[i] = standardError;
      this.lowerBoundBuffer[i] = this.predictedBuffer[i] - margin;
      this.upperBoundBuffer[i] = this.predictedBuffer[i] + margin;
    }

    return {
      predicted: Array.from(this.predictedBuffer),
      lowerBound: Array.from(this.lowerBoundBuffer),
      upperBound: Array.from(this.upperBoundBuffer),
      standardError: Array.from(this.stdErrorBuffer),
    };
  }
}

// ==================== CONFIGURATION BUILDER ====================

/**
 * Builder class for creating configuration objects.
 * Implements the Builder pattern for fluent, type-safe configuration.
 *
 * @example
 * ```typescript
 * const config = new ConfigurationBuilder()
 *   .setPolynomialDegree(3)
 *   .setNormalizationMethod('z-score')
 *   .setForgettingFactor(0.98)
 *   .setConfidenceLevel(0.99)
 *   .build();
 *
 * const model = new MultivariatePolynomialRegression(config);
 * ```
 *
 * @class ConfigurationBuilder
 */
export class ConfigurationBuilder {
  /** Internal configuration being built */
  private config: Required<IMultivariatePolynomialRegressionConfig>;

  /**
   * Creates a new ConfigurationBuilder with default values.
   */
  constructor() {
    this.config = {
      polynomialDegree: 2,
      enableNormalization: true,
      normalizationMethod: "min-max",
      forgettingFactor: 0.99,
      initialCovariance: 1000,
      regularization: 1e-6,
      confidenceLevel: 0.95,
    };
  }

  /**
   * Sets the polynomial degree for feature generation.
   *
   * @param {number} degree - The polynomial degree (minimum: 1)
   * @returns {ConfigurationBuilder} This builder for chaining
   * @throws {Error} If degree is less than 1
   */
  public setPolynomialDegree(degree: number): ConfigurationBuilder {
    if (degree < 1) {
      throw new Error("Polynomial degree must be at least 1");
    }
    this.config.polynomialDegree = degree;
    return this;
  }

  /**
   * Enables or disables input normalization.
   *
   * @param {boolean} enable - Whether to enable normalization
   * @returns {ConfigurationBuilder} This builder for chaining
   */
  public setEnableNormalization(enable: boolean): ConfigurationBuilder {
    this.config.enableNormalization = enable;
    return this;
  }

  /**
   * Sets the normalization method.
   *
   * @param {'none' | 'min-max' | 'z-score'} method - The normalization method
   * @returns {ConfigurationBuilder} This builder for chaining
   */
  public setNormalizationMethod(
    method: "none" | "min-max" | "z-score",
  ): ConfigurationBuilder {
    this.config.normalizationMethod = method;
    return this;
  }

  /**
   * Sets the forgetting factor λ for RLS algorithm.
   * Lower values give more weight to recent samples.
   *
   * @param {number} factor - The forgetting factor (range: (0, 1])
   * @returns {ConfigurationBuilder} This builder for chaining
   * @throws {Error} If factor is outside valid range
   */
  public setForgettingFactor(factor: number): ConfigurationBuilder {
    if (factor <= 0 || factor > 1) {
      throw new Error("Forgetting factor must be in range (0, 1]");
    }
    this.config.forgettingFactor = factor;
    return this;
  }

  /**
   * Sets the initial covariance diagonal value.
   * Larger values indicate higher initial uncertainty.
   *
   * @param {number} value - The initial diagonal value for covariance matrix
   * @returns {ConfigurationBuilder} This builder for chaining
   */
  public setInitialCovariance(value: number): ConfigurationBuilder {
    this.config.initialCovariance = value;
    return this;
  }

  /**
   * Sets the regularization parameter for numerical stability.
   *
   * @param {number} value - The regularization value
   * @returns {ConfigurationBuilder} This builder for chaining
   */
  public setRegularization(value: number): ConfigurationBuilder {
    this.config.regularization = value;
    return this;
  }

  /**
   * Sets the confidence level for prediction intervals.
   *
   * @param {number} level - The confidence level (range: (0, 1))
   * @returns {ConfigurationBuilder} This builder for chaining
   * @throws {Error} If level is outside valid range
   */
  public setConfidenceLevel(level: number): ConfigurationBuilder {
    if (level <= 0 || level >= 1) {
      throw new Error("Confidence level must be in range (0, 1)");
    }
    this.config.confidenceLevel = level;
    return this;
  }

  /**
   * Builds and returns the configuration object.
   *
   * @returns {Required<IMultivariatePolynomialRegressionConfig>} The completed configuration
   */
  public build(): Required<IMultivariatePolynomialRegressionConfig> {
    return { ...this.config };
  }
}

// ==================== MAIN CLASS ====================

/**
 * Multivariate Polynomial Regression with incremental online learning
 * using the Recursive Least Squares (RLS) algorithm.
 *
 * This class provides efficient online learning for polynomial regression
 * with multiple input and output dimensions. It uses optimized matrix
 * operations with Float64Array and minimizes memory allocations.
 *
 * ## RLS Algorithm Overview
 *
 * The Recursive Least Squares algorithm updates model parameters incrementally
 * without storing historical data. For each new sample (x, y):
 *
 * 1. **Feature Generation**: φ = polynomial_features(normalize(x))
 *    - Generates polynomial terms up to specified degree
 *
 * 2. **Gain Computation**: k = P·φ / (λ + φᵀ·P·φ)
 *    - k determines how much to update weights
 *    - λ is the forgetting factor (0 < λ ≤ 1)
 *
 * 3. **Error Computation**: e = y - φᵀ·w
 *    - Difference between actual and predicted output
 *
 * 4. **Weight Update**: w = w + k·e
 *    - Adjust weights in direction of error
 *
 * 5. **Covariance Update**: P = (P - k·φᵀ·P) / λ
 *    - Update uncertainty estimate
 *
 * 6. **Regularization**: Periodically add regularization to P diagonal
 *    - Ensures numerical stability
 *
 * ## Features
 *
 * - Incremental online learning (no need to store historical data)
 * - Multiple normalization strategies (min-max, z-score)
 * - Confidence intervals using t/z-distribution
 * - R-squared and RMSE metrics tracked incrementally
 * - Optimized for performance with typed arrays
 *
 * @example
 * ```typescript
 * // Basic usage
 * const model = new MultivariatePolynomialRegression();
 *
 * // Train incrementally
 * model.fitOnline({
 *   xCoordinates: [[1, 2], [2, 3], [3, 4]],
 *   yCoordinates: [[3], [5], [7]]
 * });
 *
 * // Make predictions with confidence intervals
 * const result = model.predict({
 *   futureSteps: 0,
 *   inputPoints: [[4, 5], [5, 6]]
 * });
 *
 * console.log(result.predictions[0].predicted); // Point predictions
 * console.log(result.predictions[0].lowerBound); // Lower CI bound
 * console.log(result.predictions[0].upperBound); // Upper CI bound
 * console.log(result.rSquared); // Model R²
 * ```
 *
 * @example
 * ```typescript
 * // Advanced configuration using builder
 * const config = MultivariatePolynomialRegression.createConfigBuilder()
 *   .setPolynomialDegree(3)
 *   .setNormalizationMethod('z-score')
 *   .setForgettingFactor(0.98)
 *   .setConfidenceLevel(0.99)
 *   .build();
 *
 * const model = new MultivariatePolynomialRegression(config);
 * ```
 *
 * @class MultivariatePolynomialRegression
 */
export class MultivariatePolynomialRegression {
  // ==================== CONFIGURATION ====================

  /** Complete configuration with all defaults applied */
  private readonly config: Required<IMultivariatePolynomialRegressionConfig>;

  // ==================== DEPENDENCIES (DI) ====================

  /** Matrix operations utility */
  private readonly matrixOps: MatrixOperations;

  /** Normalization statistics and operations manager */
  private readonly normalizationManager: NormalizationManager;

  /** Polynomial feature generator */
  private readonly featureGenerator: PolynomialFeatureGenerator;

  /** RLS covariance matrix manager */
  private readonly covarianceManager: CovarianceManager;

  /** Weight matrix manager */
  private readonly weightManager: WeightManager;

  /** Model statistics tracker */
  private readonly statisticsTracker: StatisticsTracker;

  /** Prediction engine with confidence intervals */
  private readonly predictionEngine: PredictionEngine;

  // ==================== STATE ====================

  /** Input dimension (set on first data) */
  private inputDimension: number = 0;

  /** Output dimension (set on first data) */
  private outputDimension: number = 0;

  /** Total samples processed */
  private sampleCount: number = 0;

  /** Initialization flag */
  private initialized: boolean = false;

  // ==================== PREALLOCATED BUFFERS ====================

  /** Buffer for input data */
  private inputBuffer: Float64Array;

  /** Buffer for normalized input */
  private normalizedBuffer: Float64Array;

  /** Buffer for polynomial features */
  private featureBuffer: Float64Array;

  /** Buffer for prediction errors */
  private errorBuffer: Float64Array;

  /** Buffer for actual output values */
  private actualBuffer: Float64Array;

  /** Buffer for predicted values */
  private predictedBuffer: Float64Array;

  // ==================== REGULARIZATION CONTROL ====================

  /** Counter for periodic regularization */
  private updateCounter: number = 0;

  /** Interval for applying regularization */
  private readonly regularizationInterval: number = 100;

  /**
   * Creates a new MultivariatePolynomialRegression instance.
   *
   * @param {IMultivariatePolynomialRegressionConfig} [config] - Configuration options
   *
   * @example
   * ```typescript
   * // With default configuration
   * const model1 = new MultivariatePolynomialRegression();
   *
   * // With custom configuration object
   * const model2 = new MultivariatePolynomialRegression({
   *   polynomialDegree: 3,
   *   forgettingFactor: 0.98,
   *   normalizationMethod: 'z-score'
   * });
   *
   * // Using builder pattern
   * const config = MultivariatePolynomialRegression.createConfigBuilder()
   *   .setPolynomialDegree(3)
   *   .setForgettingFactor(0.98)
   *   .build();
   * const model3 = new MultivariatePolynomialRegression(config);
   * ```
   */
  constructor(config?: IMultivariatePolynomialRegressionConfig) {
    // Apply defaults to configuration
    this.config = {
      polynomialDegree: config?.polynomialDegree ?? 2,
      enableNormalization: config?.enableNormalization ?? true,
      normalizationMethod: config?.normalizationMethod ?? "min-max",
      forgettingFactor: config?.forgettingFactor ?? 0.99,
      initialCovariance: config?.initialCovariance ?? 1000,
      regularization: config?.regularization ?? 1e-6,
      confidenceLevel: config?.confidenceLevel ?? 0.95,
    };

    // Validate configuration
    this.validateConfig();

    // Initialize dependencies with dependency injection
    this.matrixOps = new MatrixOperations();

    // Create normalization strategy based on config (Strategy Pattern)
    const normStrategy = this.createNormalizationStrategy();
    this.normalizationManager = new NormalizationManager(normStrategy);

    this.featureGenerator = new PolynomialFeatureGenerator(
      this.config.polynomialDegree,
    );
    this.covarianceManager = new CovarianceManager(this.matrixOps);
    this.weightManager = new WeightManager(this.matrixOps);
    this.statisticsTracker = new StatisticsTracker();
    this.predictionEngine = new PredictionEngine(
      this.covarianceManager,
      this.weightManager,
    );

    // Initialize empty buffers (will be sized on first data - lazy initialization)
    this.inputBuffer = new Float64Array(0);
    this.normalizedBuffer = new Float64Array(0);
    this.featureBuffer = new Float64Array(0);
    this.errorBuffer = new Float64Array(0);
    this.actualBuffer = new Float64Array(0);
    this.predictedBuffer = new Float64Array(0);
  }

  /**
   * Validates the configuration parameters.
   *
   * @throws {Error} If any configuration parameter is invalid
   * @private
   */
  private validateConfig(): void {
    if (this.config.polynomialDegree < 1) {
      throw new Error("Polynomial degree must be at least 1");
    }

    if (this.config.forgettingFactor <= 0 || this.config.forgettingFactor > 1) {
      throw new Error("Forgetting factor must be in range (0, 1]");
    }

    if (this.config.confidenceLevel <= 0 || this.config.confidenceLevel >= 1) {
      throw new Error("Confidence level must be in range (0, 1)");
    }
  }

  /**
   * Creates the appropriate normalization strategy based on configuration.
   * Implements the Strategy Pattern.
   *
   * @returns {INormalizationStrategy} The normalization strategy
   * @private
   */
  private createNormalizationStrategy(): INormalizationStrategy {
    if (
      !this.config.enableNormalization ||
      this.config.normalizationMethod === "none"
    ) {
      return new NoNormalizationStrategy();
    }

    switch (this.config.normalizationMethod) {
      case "z-score":
        return new ZScoreNormalizationStrategy();
      case "min-max":
      default:
        return new MinMaxNormalizationStrategy();
    }
  }

  /**
   * Initializes internal structures for the given dimensions.
   * Called lazily on first data arrival.
   *
   * Time complexity: O(f²) where f is feature count
   * Space complexity: O(f² + f×o) where o is output dimension
   *
   * @param {number} inputDim - Number of input dimensions
   * @param {number} outputDim - Number of output dimensions
   * @private
   */
  private initializeForDimensions(inputDim: number, outputDim: number): void {
    this.inputDimension = inputDim;
    this.outputDimension = outputDim;

    // Initialize feature generator to get feature count
    this.featureGenerator.initialize(inputDim);
    const featureCount = this.featureGenerator.getFeatureCount();

    // Initialize all managers
    this.normalizationManager.initialize(inputDim);
    this.covarianceManager.initialize(
      featureCount,
      this.config.initialCovariance,
    );
    this.weightManager.initialize(featureCount, outputDim);
    this.statisticsTracker.initialize(outputDim);
    this.predictionEngine.initialize(outputDim);

    // Allocate preallocated buffers
    this.inputBuffer = new Float64Array(inputDim);
    this.normalizedBuffer = new Float64Array(inputDim);
    this.featureBuffer = new Float64Array(featureCount);
    this.errorBuffer = new Float64Array(outputDim);
    this.actualBuffer = new Float64Array(outputDim);
    this.predictedBuffer = new Float64Array(outputDim);

    this.initialized = true;
  }

  /**
   * Incrementally trains the model using the RLS algorithm.
   *
   * ## Algorithm Steps
   *
   * For each sample (x, y):
   *
   * 1. **Normalize input**: x_norm = normalize(x)
   *    - Updates running statistics (min, max, mean, std)
   *    - Applies configured normalization method
   *
   * 2. **Generate features**: φ = [1, x₁, x₂, ..., x₁², x₁x₂, ...]
   *    - Creates polynomial terms up to configured degree
   *
   * 3. **Compute gain**: k = P·φ / (λ + φᵀ·P·φ)
   *    - Determines update magnitude
   *
   * 4. **Compute prediction**: ŷ = φᵀ·w
   *
   * 5. **Compute error**: e = y - ŷ
   *
   * 6. **Update weights**: w = w + k·e
   *
   * 7. **Update covariance**: P = (P - k·(P·φ)ᵀ) / λ
   *
   * 8. **Track statistics**: Update R², RMSE
   *
   * Time complexity: O(n × f²) where n is sample count, f is feature count
   * Space complexity: O(1) additional (uses preallocated buffers)
   *
   * @param {Object} params - Training parameters
   * @param {number[][]} params.xCoordinates - Input coordinates (n samples × d dimensions)
   * @param {number[][]} params.yCoordinates - Output coordinates (n samples × o dimensions)
   *
   * @throws {Error} If array lengths don't match
   * @throws {Error} If dimensions don't match previously seen data
   *
   * @example
   * ```typescript
   * const model = new MultivariatePolynomialRegression();
   *
   * // Initial training
   * model.fitOnline({
   *   xCoordinates: [[1, 2], [2, 3], [3, 4]],
   *   yCoordinates: [[3], [5], [7]]
   * });
   *
   * // Continue training online (no need to retrain from scratch)
   * model.fitOnline({
   *   xCoordinates: [[4, 5], [5, 6]],
   *   yCoordinates: [[9], [11]]
   * });
   *
   * // Single sample update
   * model.fitOnline({
   *   xCoordinates: [[6, 7]],
   *   yCoordinates: [[13]]
   * });
   * ```
   */
  public fitOnline(
    params: { xCoordinates: number[][]; yCoordinates: number[][] },
  ): void {
    const { xCoordinates, yCoordinates } = params;

    // Handle empty input
    if (xCoordinates.length === 0 || yCoordinates.length === 0) {
      return;
    }

    // Validate input arrays have matching lengths
    if (xCoordinates.length !== yCoordinates.length) {
      throw new Error("Number of x and y samples must match");
    }

    const inputDim = xCoordinates[0].length;
    const outputDim = yCoordinates[0].length;

    // Lazy initialization on first data
    if (!this.initialized) {
      this.initializeForDimensions(inputDim, outputDim);
    } else {
      // Validate dimensions match
      if (inputDim !== this.inputDimension) {
        throw new Error(
          `Input dimension mismatch: expected ${this.inputDimension}, got ${inputDim}`,
        );
      }
      if (outputDim !== this.outputDimension) {
        throw new Error(
          `Output dimension mismatch: expected ${this.outputDimension}, got ${outputDim}`,
        );
      }
    }

    const lambda = this.config.forgettingFactor;

    // Process each sample using optimized loop (no forEach/map)
    for (let sampleIdx = 0; sampleIdx < xCoordinates.length; sampleIdx++) {
      const x = xCoordinates[sampleIdx];
      const y = yCoordinates[sampleIdx];

      // Copy input to buffer (avoid allocation)
      for (let j = 0; j < inputDim; j++) {
        this.inputBuffer[j] = x[j];
      }

      // Copy output to actual buffer
      for (let j = 0; j < outputDim; j++) {
        this.actualBuffer[j] = y[j];
      }

      // ============ RLS ALGORITHM STEPS ============

      // Step 1: Update normalization statistics and normalize input
      this.normalizationManager.updateStatistics(this.inputBuffer);
      this.normalizationManager.normalizeInPlace(
        this.inputBuffer,
        this.normalizedBuffer,
      );

      // Step 2: Generate polynomial features
      // φ = [1, x₁, x₂, ..., x₁², x₁·x₂, x₂², ...]
      this.featureGenerator.generate(this.normalizedBuffer, this.featureBuffer);

      // Step 3: Compute gain vector
      // k = P·φ / (λ + φᵀ·P·φ)
      const k = this.covarianceManager.computeGain(this.featureBuffer, lambda);

      // Step 4: Compute prediction
      // ŷ = φᵀ·w
      this.weightManager.predict(this.featureBuffer, this.predictedBuffer);

      // Step 5: Compute prediction error
      // e = y - ŷ
      for (let j = 0; j < outputDim; j++) {
        this.errorBuffer[j] = this.actualBuffer[j] - this.predictedBuffer[j];
      }

      // Step 6: Update weights
      // w = w + k·e
      this.weightManager.update(k, this.errorBuffer);

      // Step 7: Update covariance matrix
      // P = (P - k·φᵀ·P) / λ
      this.covarianceManager.update(k, lambda);

      // Step 8: Update statistics (R², RMSE)
      this.statisticsTracker.update(this.actualBuffer, this.predictedBuffer);

      this.sampleCount++;
      this.updateCounter++;

      // Apply periodic regularization for numerical stability
      if (this.updateCounter >= this.regularizationInterval) {
        this.covarianceManager.applyRegularization(this.config.regularization);
        this.updateCounter = 0;
      }
    }
  }

  /**
   * Makes predictions with confidence intervals.
   *
   * Supports two modes:
   * 1. **Specific points**: Predict for given input points
   * 2. **Extrapolation**: Not directly supported (requires inputPoints)
   *
   * ## Confidence Interval Calculation
   *
   * - Standard Error: SE = √(φᵀ·P·φ)
   * - Critical Value: t_{α/2,df} for n ≤ 30, z_{α/2} for n > 30
   * - Interval: ŷ ± critical_value × SE
   *
   * Time complexity: O(m × f²) where m is number of points, f is feature count
   * Space complexity: O(m × o) for output
   *
   * @param {Object} params - Prediction parameters
   * @param {number} params.futureSteps - Number of future steps (for compatibility)
   * @param {number[][]} [params.inputPoints] - Specific input points to predict
   *
   * @returns {IPredictionResult} Predictions with confidence intervals and metrics
   *
   * @example
   * ```typescript
   * // Predict for specific points
   * const result = model.predict({
   *   futureSteps: 0,
   *   inputPoints: [[4, 5], [5, 6], [6, 7]]
   * });
   *
   * // Process results
   * result.predictions.forEach((pred, idx) => {
   *   console.log(`Point ${idx}:`);
   *   console.log(`  Predicted: ${pred.predicted}`);
   *   console.log(`  95% CI: [${pred.lowerBound}, ${pred.upperBound}]`);
   *   console.log(`  Std Error: ${pred.standardError}`);
   * });
   *
   * // Model statistics
   * console.log(`R²: ${result.rSquared.toFixed(4)}`);
   * console.log(`RMSE: ${result.rmse.toFixed(4)}`);
   * console.log(`Samples: ${result.sampleCount}`);
   * ```
   */
  public predict(
    params: { futureSteps: number; inputPoints?: number[][] },
  ): IPredictionResult {
    const { futureSteps, inputPoints } = params;

    // Initialize result with default values
    const result: IPredictionResult = {
      predictions: [],
      confidenceLevel: this.config.confidenceLevel,
      rSquared: this.statisticsTracker.getRSquared(),
      rmse: this.statisticsTracker.getRMSE(),
      sampleCount: this.sampleCount,
      isModelReady: this.initialized && this.sampleCount > 0,
    };

    // Return empty result if model not ready
    if (!this.initialized || this.sampleCount === 0) {
      return result;
    }

    // Determine points to predict
    let pointsToPredict: number[][] = [];

    if (inputPoints && inputPoints.length > 0) {
      pointsToPredict = inputPoints;
    } else if (futureSteps > 0) {
      // For time-series extrapolation without explicit points,
      // we cannot meaningfully generate future inputs
      // Return empty predictions in this case
      return result;
    }

    // Generate predictions for each point
    for (let i = 0; i < pointsToPredict.length; i++) {
      const point = pointsToPredict[i];

      // Copy input to buffer
      for (let j = 0; j < this.inputDimension; j++) {
        this.inputBuffer[j] = point[j] ?? 0;
      }

      // Normalize input
      this.normalizationManager.normalizeInPlace(
        this.inputBuffer,
        this.normalizedBuffer,
      );

      // Generate polynomial features
      this.featureGenerator.generate(this.normalizedBuffer, this.featureBuffer);

      // Make prediction with confidence intervals
      const prediction = this.predictionEngine.predict(
        this.featureBuffer,
        this.config.confidenceLevel,
        this.sampleCount,
      );

      result.predictions.push(prediction);
    }

    return result;
  }

  /**
   * Returns a comprehensive summary of the model state and configuration.
   *
   * Time complexity: O(1)
   * Space complexity: O(1)
   *
   * @returns {IModelSummary} The model summary
   *
   * @example
   * ```typescript
   * const summary = model.getModelSummary();
   *
   * console.log('Model Summary:');
   * console.log(`  Initialized: ${summary.isInitialized}`);
   * console.log(`  Input Dim: ${summary.inputDimension}`);
   * console.log(`  Output Dim: ${summary.outputDimension}`);
   * console.log(`  Degree: ${summary.polynomialDegree}`);
   * console.log(`  Features: ${summary.polynomialFeatureCount}`);
   * console.log(`  Samples: ${summary.sampleCount}`);
   * console.log(`  R²: ${summary.rSquared.toFixed(4)}`);
   * console.log(`  RMSE: ${summary.rmse.toFixed(4)}`);
   * console.log(`  Normalization: ${summary.normalizationMethod}`);
   * ```
   */
  public getModelSummary(): IModelSummary {
    return {
      isInitialized: this.initialized,
      inputDimension: this.inputDimension,
      outputDimension: this.outputDimension,
      polynomialDegree: this.config.polynomialDegree,
      polynomialFeatureCount: this.featureGenerator.getFeatureCount(),
      sampleCount: this.sampleCount,
      rSquared: this.statisticsTracker.getRSquared(),
      rmse: this.statisticsTracker.getRMSE(),
      normalizationEnabled: this.config.enableNormalization,
      normalizationMethod: this.config.normalizationMethod,
    };
  }

  /**
   * Returns the current weight matrix.
   * Weights are organized as W[outputDim][featureIndex].
   *
   * Time complexity: O(f × o) where f is feature count, o is output dimension
   * Space complexity: O(f × o)
   *
   * @returns {number[][]} The weight matrix (output_dim × feature_count)
   *
   * @example
   * ```typescript
   * const weights = model.getWeights();
   *
   * // For single output dimension
   * console.log('Weights:', weights[0]);
   *
   * // Intercept (constant term) is typically weights[0][0]
   * console.log('Intercept:', weights[0][0]);
   * ```
   */
  public getWeights(): number[][] {
    if (!this.initialized) {
      return [];
    }
    return this.weightManager.getWeights();
  }

  /**
   * Returns the current normalization statistics.
   *
   * Time complexity: O(d) where d is input dimension
   * Space complexity: O(d)
   *
   * @returns {INormalizationStats} The normalization statistics
   *
   * @example
   * ```typescript
   * const stats = model.getNormalizationStats();
   *
   * console.log('Normalization Statistics:');
   * console.log(`  Count: ${stats.count}`);
   * for (let i = 0; i < stats.min.length; i++) {
   *   console.log(`  Dim ${i}: min=${stats.min[i]}, max=${stats.max[i]}, mean=${stats.mean[i]}, std=${stats.std[i]}`);
   * }
   * ```
   */
  public getNormalizationStats(): INormalizationStats {
    return this.normalizationManager.getStats();
  }

  /**
   * Resets the model to its initial state.
   * Clears all learned weights, covariance matrix, statistics, and buffers.
   * The model can be retrained from scratch after reset.
   *
   * Time complexity: O(f²) for covariance matrix clearing
   * Space complexity: O(1)
   *
   * @example
   * ```typescript
   * // Train model
   * model.fitOnline({ xCoordinates: [...], yCoordinates: [...] });
   *
   * // Reset to initial state
   * model.reset();
   *
   * // Retrain with different data
   * model.fitOnline({ xCoordinates: [...], yCoordinates: [...] });
   * ```
   */
  public reset(): void {
    // Reset all managers
    this.normalizationManager.reset();
    this.featureGenerator.reset();
    this.covarianceManager.reset();
    this.weightManager.reset();
    this.statisticsTracker.reset();

    // Reset state
    this.inputDimension = 0;
    this.outputDimension = 0;
    this.sampleCount = 0;
    this.initialized = false;
    this.updateCounter = 0;

    // Clear buffers (release memory)
    this.inputBuffer = new Float64Array(0);
    this.normalizedBuffer = new Float64Array(0);
    this.featureBuffer = new Float64Array(0);
    this.errorBuffer = new Float64Array(0);
    this.actualBuffer = new Float64Array(0);
    this.predictedBuffer = new Float64Array(0);
  }

  /**
   * Creates a configuration builder for fluent configuration.
   * Static factory method for the Builder pattern.
   *
   * @returns {ConfigurationBuilder} A new configuration builder
   *
   * @example
   * ```typescript
   * const config = MultivariatePolynomialRegression.createConfigBuilder()
   *   .setPolynomialDegree(3)
   *   .setNormalizationMethod('z-score')
   *   .setForgettingFactor(0.98)
   *   .setConfidenceLevel(0.99)
   *   .setInitialCovariance(500)
   *   .setRegularization(1e-5)
   *   .build();
   *
   * const model = new MultivariatePolynomialRegression(config);
   * ```
   */
  public static createConfigBuilder(): ConfigurationBuilder {
    return new ConfigurationBuilder();
  }
}
