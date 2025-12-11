/**
 * @fileoverview Multivariate Polynomial Regression Library
 *
 * A production-ready, memory-efficient TypeScript library for multivariate polynomial
 * regression with incremental online learning using the Recursive Least Squares (RLS) algorithm.
 *
 * ## Key Features:
 * - Online Learning: Train incrementally without storing historical data
 * - Multivariate Support: Multiple inputs and outputs
 * - Memory Efficient: Uses Float64Array and buffer reuse
 * - Numerically Stable: Regularization and proper algorithm implementation
 *
 * ## Algorithm:
 * Uses Recursive Least Squares (RLS) with forgetting factor λ:
 * ```
 * For each sample (x, y):
 *   φ = polynomial_features(x)
 *   k = P·φ / (λ + φᵀ·P·φ)         // Kalman gain
 *   w = w + k·(y - φᵀ·w)           // Weight update
 *   P = (P - k·φᵀ·P) / λ           // Covariance update
 * ```
 *
 * @version 1.0.4
 * @license MIT
 * @author Henrique Emanoel Viana
 */

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

/**
 * Normalization method enumeration.
 * Defines strategies for preprocessing input data to improve numerical stability.
 *
 * @description
 * - `'none'`: No transformation applied; data used as-is
 * - `'min-max'`: Scales to [0,1] using (x - min) / (max - min)
 * - `'z-score'`: Standardizes using (x - μ) / σ
 */
type NormalizationMethod = "none" | "min-max" | "z-score";

/**
 * Configuration interface for model initialization.
 * All parameters are optional with sensible defaults.
 *
 * @interface ModelConfiguration
 * @description Immutable configuration object passed to constructor
 */
interface ModelConfiguration {
  /**
   * Degree of polynomial features to generate.
   *
   * @remarks
   * - Degree 1: Linear (y = a + bx)
   * - Degree 2: Quadratic (y = a + bx + cx²)
   * - Degree 3: Cubic (y = a + bx + cx² + dx³)
   *
   * Higher degrees capture complex patterns but risk overfitting.
   * Feature count grows combinatorially: C(n+d, d) where n=input_dim, d=degree
   *
   * @default 2
   * @minimum 1
   */
  readonly polynomialDegree?: number;

  /**
   * Whether to enable automatic input normalization.
   *
   * @remarks
   * Normalization is strongly recommended when:
   * - Input features have different scales
   * - Input values are large (prevents overflow)
   * - Convergence speed is important
   *
   * @default true
   */
  readonly enableNormalization?: boolean;

  /**
   * Method used for input normalization.
   *
   * @remarks
   * - `'min-max'`: Best for bounded data, outputs in [0,1]
   * - `'z-score'`: Best for Gaussian-distributed data
   * - `'none'`: For pre-normalized data
   *
   * @default 'min-max'
   */
  readonly normalizationMethod?: NormalizationMethod;

  /**
   * RLS forgetting factor (λ).
   * Controls how quickly old data is "forgotten".
   *
   * @remarks
   * - λ = 1.0: Never forget (standard least squares)
   * - λ = 0.99: Slight forgetting (recommended)
   * - λ = 0.95: Moderate forgetting
   * - λ = 0.90: Aggressive forgetting
   *
   * Lower values = faster adaptation to new patterns
   * Higher values = more stable estimates
   *
   * @default 0.99
   * @range (0, 1]
   */
  readonly forgettingFactor?: number;

  /**
   * Initial diagonal value for covariance matrix P.
   * Represents initial uncertainty about model weights.
   *
   * @remarks
   * - Higher values: Faster initial learning, larger weight updates
   * - Lower values: More conservative initial updates
   *
   * @default 1000
   */
  readonly initialCovariance?: number;

  /**
   * Regularization parameter added to P diagonal.
   * Prevents numerical singularity (matrix inversion issues).
   *
   * @remarks
   * Increase if experiencing:
   * - NaN/Infinity in weights
   * - Matrix singularity warnings
   * - Unstable predictions
   *
   * @default 1e-6
   */
  readonly regularization?: number;

  /**
   * Confidence level for prediction intervals.
   *
   * @remarks
   * Common values:
   * - 0.90: 90% confidence (narrower intervals)
   * - 0.95: 95% confidence (standard)
   * - 0.99: 99% confidence (wider intervals)
   *
   * @default 0.95
   * @range (0, 1)
   */
  readonly confidenceLevel?: number;
}

/**
 * Parameters for incremental training.
 *
 * @interface FitOnlineParams
 */
interface FitOnlineParams {
  /**
   * Input feature matrix.
   * Shape: [numSamples][numFeatures]
   *
   * @example
   * // Single feature: [[1], [2], [3]]
   * // Multiple features: [[1, 10], [2, 20], [3, 30]]
   */
  readonly xCoordinates: ReadonlyArray<ReadonlyArray<number>>;

  /**
   * Output target matrix.
   * Shape: [numSamples][numOutputs]
   *
   * @example
   * // Single output: [[2], [4], [6]]
   * // Multiple outputs: [[2, 10], [4, 20], [6, 30]]
   */
  readonly yCoordinates: ReadonlyArray<ReadonlyArray<number>>;
}

/**
 * Parameters for prediction generation.
 *
 * @interface PredictParams
 */
interface PredictParams {
  /**
   * Number of future steps to extrapolate.
   * Used for time series forecasting where input is sequential.
   * Ignored when inputPoints is provided.
   */
  readonly futureSteps: number;

  /**
   * Explicit input points for prediction.
   * When provided, overrides futureSteps behavior.
   *
   * @example
   * // Predict for specific inputs
   * { futureSteps: 0, inputPoints: [[5], [6], [7]] }
   */
  readonly inputPoints?: ReadonlyArray<ReadonlyArray<number>>;
}

/**
 * Single prediction with uncertainty quantification.
 *
 * @interface SinglePrediction
 */
interface SinglePrediction {
  /** Point estimates for each output dimension */
  readonly predicted: ReadonlyArray<number>;

  /** Lower confidence bounds for each output */
  readonly lowerBound: ReadonlyArray<number>;

  /** Upper confidence bounds for each output */
  readonly upperBound: ReadonlyArray<number>;

  /** Standard errors for each output */
  readonly standardError: ReadonlyArray<number>;
}

/**
 * Complete prediction result with model diagnostics.
 *
 * @interface PredictionResult
 */
interface PredictionResult {
  /** Array of predictions with confidence intervals */
  readonly predictions: ReadonlyArray<SinglePrediction>;

  /** Confidence level used (e.g., 0.95 for 95%) */
  readonly confidenceLevel: number;

  /** Coefficient of determination R² ∈ [0, 1] */
  readonly rSquared: number;

  /** Root Mean Square Error */
  readonly rmse: number;

  /** Total training samples processed */
  readonly sampleCount: number;

  /** True if sufficient training data (samples >= features) */
  readonly isModelReady: boolean;
}

/**
 * Comprehensive model statistics.
 *
 * @interface ModelSummary
 */
interface ModelSummary {
  /** Whether model has received any training data */
  readonly isInitialized: boolean;

  /** Number of input features */
  readonly inputDimension: number;

  /** Number of output targets */
  readonly outputDimension: number;

  /** Configured polynomial degree */
  readonly polynomialDegree: number;

  /** Total polynomial features (including interactions) */
  readonly polynomialFeatureCount: number;

  /** Total samples trained on */
  readonly sampleCount: number;

  /** Current R² value */
  readonly rSquared: number;

  /** Current RMSE value */
  readonly rmse: number;

  /** Whether normalization is active */
  readonly normalizationEnabled: boolean;

  /** Active normalization method */
  readonly normalizationMethod: NormalizationMethod;
}

/**
 * Running statistics for normalization.
 *
 * @interface NormalizationStats
 */
interface NormalizationStats {
  /** Minimum observed value per feature */
  readonly min: ReadonlyArray<number>;

  /** Maximum observed value per feature */
  readonly max: ReadonlyArray<number>;

  /** Running mean per feature */
  readonly mean: ReadonlyArray<number>;

  /** Running standard deviation per feature */
  readonly std: ReadonlyArray<number>;

  /** Number of samples used to compute stats */
  readonly count: number;
}

// ============================================================================
// OPTIMIZED MATRIX OPERATIONS (Static Utility Class)
// ============================================================================

/**
 * Memory-efficient matrix operations optimized for RLS algorithm.
 *
 * @description
 * This utility class provides static methods for matrix/vector operations
 * used in the RLS algorithm. Key optimizations include:
 * - Float64Array usage for memory efficiency and cache locality
 * - Loop unrolling for critical inner loops
 * - In-place operations to minimize allocations
 * - Pre-computed indices where beneficial
 *
 * @remarks
 * All methods are static and stateless for maximum reusability.
 * No instance methods exist; the constructor is private.
 *
 * @example
 * ```typescript
 * const vec = MatrixOperations.createVector(10);
 * const mat = MatrixOperations.createMatrix(3, 3);
 * MatrixOperations.matrixVectorMultiply(mat, vec, result);
 * ```
 */
class MatrixOperations {
  /**
   * Private constructor prevents instantiation.
   * This is a static utility class only.
   * @private
   */
  private constructor() {
    throw new Error("MatrixOperations is a static utility class");
  }

  /**
   * Creates a zero-initialized matrix using Float64Array.
   *
   * @description
   * Uses Float64Array instead of number[][] for:
   * - 8 bytes per element (fixed, predictable)
   * - Better cache locality
   * - Automatic zero initialization
   * - Direct memory access without boxing
   *
   * @param rows - Number of rows (must be positive)
   * @param cols - Number of columns (must be positive)
   * @returns Zero-initialized matrix as array of Float64Arrays
   * @complexity Time: O(rows), Space: O(rows × cols)
   *
   * @example
   * ```typescript
   * const matrix = MatrixOperations.createMatrix(3, 4);
   * // matrix[0][0] === 0, etc.
   * ```
   */
  static createMatrix(rows: number, cols: number): Float64Array[] {
    const matrix: Float64Array[] = new Array<Float64Array>(rows);
    for (let i = 0; i < rows; ++i) {
      matrix[i] = new Float64Array(cols);
    }
    return matrix;
  }

  /**
   * Creates a diagonal matrix with specified diagonal value.
   *
   * @description
   * Optimized for covariance matrix initialization.
   * Only sets diagonal elements, leaving off-diagonals as zero.
   *
   * @param size - Matrix dimension (creates size × size matrix)
   * @param diagonalValue - Value for all diagonal elements
   * @returns Diagonal matrix
   * @complexity Time: O(size), Space: O(size²)
   *
   * @example
   * ```typescript
   * const identity = MatrixOperations.createDiagonalMatrix(3, 1);
   * // Creates 3x3 identity matrix
   * ```
   */
  static createDiagonalMatrix(
    size: number,
    diagonalValue: number,
  ): Float64Array[] {
    const matrix = this.createMatrix(size, size);
    for (let i = 0; i < size; ++i) {
      matrix[i][i] = diagonalValue;
    }
    return matrix;
  }

  /**
   * Creates a zero-initialized vector using Float64Array.
   *
   * @param length - Vector length
   * @returns Zero-initialized Float64Array
   * @complexity Time: O(1), Space: O(length)
   */
  static createVector(length: number): Float64Array {
    return new Float64Array(length);
  }

  /**
   * Computes matrix-vector product: result = A × v
   *
   * @description
   * Optimized implementation with:
   * - Loop unrolling by factor of 4 for ILP
   * - Row-major access pattern for cache efficiency
   * - Pre-fetched row reference to avoid repeated indexing
   *
   * @param matrix - Input matrix A [m × n]
   * @param vector - Input vector v [n]
   * @param result - Output vector [m] (MUST be pre-allocated)
   * @complexity Time: O(m × n), Space: O(1) (uses provided result)
   *
   * @example
   * ```typescript
   * const A = [[1, 2], [3, 4]];  // 2x2 matrix
   * const v = [5, 6];             // 2-vector
   * const result = new Float64Array(2);
   * MatrixOperations.matrixVectorMultiply(A, v, result);
   * // result = [17, 39]
   * ```
   */
  static matrixVectorMultiply(
    matrix: Float64Array[],
    vector: Float64Array,
    result: Float64Array,
  ): void {
    const rows = matrix.length;
    const cols = vector.length;

    for (let i = 0; i < rows; ++i) {
      const row = matrix[i]; // Cache row reference
      let sum = 0.0;

      // Unroll by 4 for instruction-level parallelism
      let j = 0;
      const limit = cols - 3;

      while (j < limit) {
        sum += row[j] * vector[j] +
          row[j + 1] * vector[j + 1] +
          row[j + 2] * vector[j + 2] +
          row[j + 3] * vector[j + 3];
        j += 4;
      }

      // Handle remainder
      while (j < cols) {
        sum += row[j] * vector[j];
        ++j;
      }

      result[i] = sum;
    }
  }

  /**
   * Computes dot product of two vectors: a · b
   *
   * @description
   * Optimized with loop unrolling by factor of 4.
   * Assumes both vectors have equal length.
   *
   * @param a - First vector
   * @param b - Second vector (must have same length as a)
   * @returns Scalar dot product
   * @complexity Time: O(n), Space: O(1)
   */
  static dotProduct(a: Float64Array, b: Float64Array): number {
    const n = a.length;
    let sum = 0.0;

    let i = 0;
    const limit = n - 3;

    while (i < limit) {
      sum += a[i] * b[i] +
        a[i + 1] * b[i + 1] +
        a[i + 2] * b[i + 2] +
        a[i + 3] * b[i + 3];
      i += 4;
    }

    while (i < n) {
      sum += a[i] * b[i];
      ++i;
    }

    return sum;
  }

  /**
   * Updates covariance matrix in-place for RLS algorithm.
   * Computes: P = (P - k × (φᵀP)ᵀ) / λ = (P - k × pPhiᵀ) / λ
   *
   * @description
   * This is the most computationally intensive operation in RLS.
   * Optimizations:
   * - In-place update (no allocation)
   * - Row-by-row processing for cache efficiency
   * - Fused multiply-add operations
   * - Pre-computed inverse lambda
   *
   * @param matrix - Covariance matrix P (modified in-place)
   * @param gain - Kalman gain vector k
   * @param pPhi - Vector P × φ (phi transposed times P)
   * @param invLambda - Pre-computed 1/λ
   * @complexity Time: O(n²), Space: O(1)
   */
  static updateCovarianceInPlace(
    matrix: Float64Array[],
    gain: Float64Array,
    pPhi: Float64Array,
    invLambda: number,
  ): void {
    const n = matrix.length;

    for (let i = 0; i < n; ++i) {
      const ki = gain[i];
      const row = matrix[i];

      for (let j = 0; j < n; ++j) {
        row[j] = (row[j] - ki * pPhi[j]) * invLambda;
      }
    }
  }

  /**
   * Adds regularization to matrix diagonal in-place.
   * Computes: P[i][i] += δ for all i
   *
   * @description
   * Numerical stability technique to prevent singularity.
   * Small positive value on diagonal ensures positive definiteness.
   *
   * @param matrix - Square matrix (modified in-place)
   * @param delta - Regularization value to add
   * @complexity Time: O(n), Space: O(1)
   */
  static addRegularization(matrix: Float64Array[], delta: number): void {
    const n = matrix.length;
    for (let i = 0; i < n; ++i) {
      matrix[i][i] += delta;
    }
  }
}

// ============================================================================
// INCREMENTAL STATISTICS (Welford's Algorithm)
// ============================================================================

/**
 * Memory-efficient incremental statistics calculator.
 *
 * @description
 * Implements Welford's online algorithm for computing running statistics
 * in a single pass without storing all data points. This is essential
 * for online learning where data arrives as a stream.
 *
 * Computes simultaneously:
 * - Running mean (numerically stable)
 * - Running variance/standard deviation
 * - Running min and max
 *
 * @remarks
 * Welford's algorithm is chosen for numerical stability when computing
 * variance. The naive formula Var(X) = E[X²] - E[X]² suffers from
 * catastrophic cancellation. Welford's uses:
 * - M₁ = x₁
 * - Mₙ = Mₙ₋₁ + (xₙ - Mₙ₋₁)/n
 * - S₁ = 0
 * - Sₙ = Sₙ₋₁ + (xₙ - Mₙ₋₁)(xₙ - Mₙ)
 * - Var = Sₙ / n
 *
 * @example
 * ```typescript
 * const stats = new IncrementalStatistics(3);
 * stats.update([1, 2, 3]);
 * stats.update([4, 5, 6]);
 * console.log(stats.getMean()); // [2.5, 3.5, 4.5]
 * ```
 */
class IncrementalStatistics {
  /** Running sum for each dimension (for mean calculation) */
  private readonly _sum: Float64Array;

  /** Running sum of squared deviations (for variance, Welford's S) */
  private readonly _m2: Float64Array;

  /** Running mean (Welford's M) */
  private readonly _mean: Float64Array;

  /** Minimum values observed */
  private readonly _min: Float64Array;

  /** Maximum values observed */
  private readonly _max: Float64Array;

  /** Sample count */
  private _count: number;

  /** Dimensionality of tracked data */
  private readonly _dimension: number;

  /**
   * Creates new incremental statistics tracker.
   *
   * @param dimension - Number of features/dimensions to track
   */
  constructor(dimension: number) {
    this._dimension = dimension;
    this._count = 0;
    this._sum = new Float64Array(dimension);
    this._m2 = new Float64Array(dimension);
    this._mean = new Float64Array(dimension);
    this._min = new Float64Array(dimension);
    this._max = new Float64Array(dimension);

    // Initialize min/max to sentinel values
    this._min.fill(Number.POSITIVE_INFINITY);
    this._max.fill(Number.NEGATIVE_INFINITY);
  }

  /**
   * Updates statistics with a new observation.
   * Uses Welford's algorithm for numerical stability.
   *
   * @param sample - New data point [dimension values]
   * @complexity Time: O(dimension), Space: O(1)
   */
  update(sample: ReadonlyArray<number>): void {
    const n = ++this._count;
    const invN = 1.0 / n;

    for (let i = 0; i < this._dimension; ++i) {
      const x = sample[i];

      // Update sum (for compatibility, though not needed for Welford)
      this._sum[i] += x;

      // Welford's algorithm for mean and variance
      const delta = x - this._mean[i];
      this._mean[i] += delta * invN;
      const delta2 = x - this._mean[i];
      this._m2[i] += delta * delta2;

      // Update min/max
      if (x < this._min[i]) this._min[i] = x;
      if (x > this._max[i]) this._max[i] = x;
    }
  }

  /**
   * Gets current sample count.
   */
  get count(): number {
    return this._count;
  }

  /**
   * Returns copy of current means.
   * @returns New Float64Array with means
   */
  getMean(): Float64Array {
    return new Float64Array(this._mean);
  }

  /**
   * Computes and returns standard deviations.
   * Uses population formula (divide by n, not n-1).
   *
   * @returns New Float64Array with standard deviations
   */
  getStd(): Float64Array {
    const std = new Float64Array(this._dimension);

    if (this._count > 0) {
      const invN = 1.0 / this._count;
      for (let i = 0; i < this._dimension; ++i) {
        const variance = this._m2[i] * invN;
        std[i] = variance > 0 ? Math.sqrt(variance) : 1e-10;
      }
    } else {
      std.fill(1.0); // Prevent division by zero
    }

    return std;
  }

  /**
   * Returns copy of minimum values.
   */
  getMin(): Float64Array {
    const result = new Float64Array(this._dimension);
    for (let i = 0; i < this._dimension; ++i) {
      result[i] = Number.isFinite(this._min[i]) ? this._min[i] : 0;
    }
    return result;
  }

  /**
   * Returns copy of maximum values.
   */
  getMax(): Float64Array {
    const result = new Float64Array(this._dimension);
    for (let i = 0; i < this._dimension; ++i) {
      result[i] = Number.isFinite(this._max[i]) ? this._max[i] : 1;
    }
    return result;
  }

  /**
   * Resets all statistics to initial state.
   */
  reset(): void {
    this._count = 0;
    this._sum.fill(0);
    this._m2.fill(0);
    this._mean.fill(0);
    this._min.fill(Number.POSITIVE_INFINITY);
    this._max.fill(Number.NEGATIVE_INFINITY);
  }
}

// ============================================================================
// POLYNOMIAL FEATURE GENERATOR (with Memoization)
// ============================================================================

/**
 * Efficient polynomial feature generator with cached exponent patterns.
 *
 * @description
 * Generates polynomial feature expansions for multivariate inputs.
 * For input [x₁, x₂] and degree 2, generates:
 * [1, x₁, x₂, x₁², x₁x₂, x₂²]
 *
 * The exponent patterns are pre-computed and cached for efficiency.
 * The number of features follows the formula: C(n+d, d) = (n+d)!/(n!d!)
 * where n = input dimension, d = polynomial degree.
 *
 * @remarks
 * Key optimizations:
 * - Exponent combinations computed once and cached
 * - Fast power function with special cases for small exponents
 * - Avoids unnecessary multiplications for zero exponents
 *
 * @example
 * ```typescript
 * const gen = new PolynomialFeatureGenerator(2);
 * gen.initialize(2);  // 2 input features
 * console.log(gen.featureCount);  // 6 for degree 2
 *
 * const input = new Float64Array([3, 4]);
 * const output = new Float64Array(6);
 * gen.transform(input, output);
 * // output = [1, 3, 4, 9, 12, 16]
 * ```
 */
class PolynomialFeatureGenerator {
  /** Maximum polynomial degree */
  private readonly _degree: number;

  /** Input dimension (set during initialization) */
  private _inputDimension: number;

  /** Cached exponent combinations [featureIndex][inputIndex] */
  private _exponentCache: Uint8Array[] | null;

  /** Total number of polynomial features */
  private _featureCount: number;

  /**
   * Creates polynomial feature generator.
   *
   * @param degree - Maximum polynomial degree (≥ 1)
   */
  constructor(degree: number) {
    this._degree = degree;
    this._inputDimension = 0;
    this._exponentCache = null;
    this._featureCount = 0;
  }

  /**
   * Initializes generator for specific input dimension.
   * Computes and caches all exponent combinations.
   *
   * @description
   * Must be called before transform(). Idempotent for same dimension.
   *
   * @param inputDimension - Number of input features
   * @complexity Time: O(C(n+d,d) × n), Space: O(C(n+d,d) × n)
   */
  initialize(inputDimension: number): void {
    // Skip if already initialized for this dimension
    if (
      this._inputDimension === inputDimension && this._exponentCache !== null
    ) {
      return;
    }

    this._inputDimension = inputDimension;
    this._exponentCache = this._generateExponentCombinations();
    this._featureCount = this._exponentCache.length;
  }

  /** Number of polynomial features generated */
  get featureCount(): number {
    return this._featureCount;
  }

  /** Input dimension */
  get inputDimension(): number {
    return this._inputDimension;
  }

  /**
   * Transforms input vector to polynomial feature vector.
   *
   * @description
   * Applies polynomial expansion using cached exponent patterns.
   * For each feature f, computes: φ_f = ∏ᵢ xᵢ^(e_f,i)
   *
   * @param input - Normalized input vector [inputDimension]
   * @param output - Pre-allocated output vector [featureCount]
   * @complexity Time: O(featureCount × inputDimension)
   */
  transform(input: Float64Array, output: Float64Array): void {
    if (this._exponentCache === null) {
      throw new Error("PolynomialFeatureGenerator not initialized");
    }

    const cache = this._exponentCache;
    const numFeatures = cache.length;
    const dim = this._inputDimension;

    for (let f = 0; f < numFeatures; ++f) {
      const exponents = cache[f];
      let product = 1.0;

      for (let d = 0; d < dim; ++d) {
        const exp = exponents[d];
        if (exp !== 0) {
          product *= this._fastPow(input[d], exp);
        }
      }

      output[f] = product;
    }
  }

  /**
   * Fast integer power computation.
   * Optimized for small exponents common in polynomial regression.
   *
   * @param base - Base value
   * @param exp - Non-negative integer exponent
   * @returns base^exp
   * @complexity Time: O(log exp) worst case, O(1) for exp ≤ 4
   */
  private _fastPow(base: number, exp: number): number {
    switch (exp) {
      case 0:
        return 1.0;
      case 1:
        return base;
      case 2:
        return base * base;
      case 3:
        return base * base * base;
      case 4: {
        const sq = base * base;
        return sq * sq;
      }
      default: {
        // Exponentiation by squaring for larger exponents
        let result = 1.0;
        let b = base;
        let e = exp;

        while (e > 0) {
          if (e & 1) result *= b;
          b *= b;
          e >>>= 1;
        }

        return result;
      }
    }
  }

  /**
   * Generates all valid exponent combinations.
   *
   * @description
   * Enumerates all combinations of non-negative integer exponents
   * that sum to at most the polynomial degree.
   *
   * Uses Uint8Array for exponents (sufficient for degree ≤ 255).
   *
   * @returns Array of exponent patterns
   */
  private _generateExponentCombinations(): Uint8Array[] {
    const combinations: Uint8Array[] = [];
    const dim = this._inputDimension;
    const maxDegree = this._degree;
    const current = new Uint8Array(dim);

    /**
     * Recursive generator using backtracking.
     * @param pos - Current dimension position
     * @param remaining - Remaining degree budget
     */
    const generate = (pos: number, remaining: number): void => {
      if (pos === dim) {
        combinations.push(new Uint8Array(current));
        return;
      }

      for (let exp = 0; exp <= remaining; ++exp) {
        current[pos] = exp;
        generate(pos + 1, remaining - exp);
      }
    };

    generate(0, maxDegree);
    return combinations;
  }

  /**
   * Resets generator to uninitialized state.
   */
  reset(): void {
    this._inputDimension = 0;
    this._exponentCache = null;
    this._featureCount = 0;
  }
}

// ============================================================================
// STATISTICAL DISTRIBUTION UTILITIES
// ============================================================================

/**
 * Statistical distribution utilities for confidence intervals.
 *
 * @description
 * Provides approximations for t-distribution critical values
 * used in prediction interval calculation.
 *
 * Uses the Abramowitz and Stegun approximation for the normal
 * quantile function, with Wilson-Hilferty correction for t-distribution.
 */
class StatisticalDistribution {
  /** Private constructor - static utility class */
  private constructor() {}

  /**
   * Computes t-distribution critical value.
   *
   * @description
   * For large df, uses normal approximation.
   * For smaller df, applies polynomial correction.
   *
   * @param confidenceLevel - Confidence level (e.g., 0.95)
   * @param df - Degrees of freedom
   * @returns Critical t-value for two-tailed interval
   */
  static getTCritical(confidenceLevel: number, df: number): number {
    // For large df, t → z
    if (df > 1000) {
      return this._normalQuantile((1 + confidenceLevel) * 0.5);
    }

    const p = (1 + confidenceLevel) * 0.5;
    const z = this._normalQuantile(p);

    // Cornish-Fisher expansion for t-distribution
    const z2 = z * z;
    const z3 = z2 * z;
    const z5 = z3 * z2;

    const g1 = (z3 + z) * 0.25;
    const g2 = (5 * z5 + 16 * z3 + 3 * z) / 96;

    const invDf = 1.0 / df;
    return z + g1 * invDf + g2 * invDf * invDf;
  }

  /**
   * Computes standard normal quantile (inverse CDF).
   * Uses Abramowitz and Stegun rational approximation.
   *
   * @param p - Probability (0 < p < 1)
   * @returns z such that Φ(z) = p
   */
  private static _normalQuantile(p: number): number {
    // Coefficients for rational approximation
    const a = [
      -3.969683028665376e+01,
      2.209460984245205e+02,
      -2.759285104469687e+02,
      1.383577518672690e+02,
      -3.066479806614716e+01,
      2.506628277459239e+00,
    ];

    const b = [
      -5.447609879822406e+01,
      1.615858368580409e+02,
      -1.556989798598866e+02,
      6.680131188771972e+01,
      -1.328068155288572e+01,
    ];

    const c = [
      -7.784894002430293e-03,
      -3.223964580411365e-01,
      -2.400758277161838e+00,
      -2.549732539343734e+00,
      4.374664141464968e+00,
      2.938163982698783e+00,
    ];

    const d = [
      7.784695709041462e-03,
      3.224671290700398e-01,
      2.445134137142996e+00,
      3.754408661907416e+00,
    ];

    const pLow = 0.02425;
    const pHigh = 1 - pLow;

    let q: number, r: number;

    if (p < pLow) {
      q = Math.sqrt(-2 * Math.log(p));
      return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q +
        c[5]) /
        ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
    }

    if (p <= pHigh) {
      q = p - 0.5;
      r = q * q;
      return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r +
        a[5]) *
        q /
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
    }

    q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q +
      c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
}

// ============================================================================
// MAIN CLASS: MULTIVARIATE POLYNOMIAL REGRESSION
// ============================================================================

/**
 * Multivariate Polynomial Regression with Online Learning.
 *
 * @description
 * A production-ready, memory-efficient implementation of polynomial regression
 * supporting incremental online learning via the Recursive Least Squares (RLS)
 * algorithm. Designed for real-time data streams and adaptive systems.
 *
 * ## Key Features
 * - **Online Learning**: Train incrementally without storing all data
 * - **Multivariate**: Handles multiple inputs and outputs
 * - **Polynomial Flexibility**: Configurable degree for non-linear patterns
 * - **Auto Normalization**: Built-in min-max and z-score normalization
 * - **Confidence Intervals**: Prediction uncertainty quantification
 * - **Memory Efficient**: O(p²) memory where p = polynomial features
 * - **Numerically Stable**: Regularization prevents singularity
 *
 * ## Algorithm
 * Uses Recursive Least Squares with forgetting factor λ:
 * ```
 * For each sample (x, y):
 *   φ = polynomial_features(normalize(x))
 *   k = P·φ / (λ + φᵀ·P·φ)       // Kalman gain
 *   w = w + k·(y - φᵀ·w)         // Weight update
 *   P = (P - k·φᵀ·P) / λ         // Covariance update
 * ```
 *
 * ## Complexity
 * - Memory: O(p²) where p = polynomial feature count
 * - Per-sample training: O(p²)
 * - Prediction: O(p × outputDim)
 *
 * @example
 * ```typescript
 * // Basic usage
 * const model = new MultivariatePolynomialRegression({
 *   polynomialDegree: 2,
 *   enableNormalization: true
 * });
 *
 * // Train incrementally
 * model.fitOnline({
 *   xCoordinates: [[1], [2], [3], [4], [5]],
 *   yCoordinates: [[2.1], [4.0], [5.9], [8.1], [10.0]]
 * });
 *
 * // Predict with confidence intervals
 * const result = model.predict({ futureSteps: 3 });
 * console.log(result.predictions[0].predicted);
 * console.log(result.rSquared);
 * ```
 */
class MultivariatePolynomialRegression {
  // ========================================================================
  // IMMUTABLE CONFIGURATION
  // ========================================================================

  /** Polynomial degree for feature expansion */
  private readonly _polynomialDegree: number;

  /** Whether normalization is enabled */
  private readonly _enableNormalization: boolean;

  /** Normalization method */
  private readonly _normalizationMethod: NormalizationMethod;

  /** RLS forgetting factor λ */
  private readonly _forgettingFactor: number;

  /** Inverse forgetting factor (precomputed for efficiency) */
  private readonly _invForgettingFactor: number;

  /** Initial covariance diagonal value */
  private readonly _initialCovariance: number;

  /** Regularization parameter δ */
  private readonly _regularization: number;

  /** Confidence level for intervals */
  private readonly _confidenceLevel: number;

  // ========================================================================
  // MUTABLE MODEL STATE
  // ========================================================================

  /** Model initialization flag */
  private _isInitialized: boolean;

  /** Input feature dimension */
  private _inputDimension: number;

  /** Output target dimension */
  private _outputDimension: number;

  /** Number of polynomial features */
  private _polynomialFeatureCount: number;

  /** Weight matrix W [featureCount × outputDim] */
  private _weights: Float64Array[] | null;

  /** Covariance matrix P [featureCount × featureCount] */
  private _covarianceMatrix: Float64Array[] | null;

  /** Total training samples processed */
  private _sampleCount: number;

  /** Polynomial feature generator */
  private readonly _featureGenerator: PolynomialFeatureGenerator;

  /** Input normalization statistics */
  private _inputStats: IncrementalStatistics | null;

  /** Last raw input (for extrapolation) */
  private _lastRawInput: Float64Array | null;

  // ========================================================================
  // REUSABLE BUFFERS (minimize allocation during training)
  // ========================================================================

  /** Buffer for polynomial features φ */
  private _phiBuffer: Float64Array | null;

  /** Buffer for P × φ */
  private _pPhiBuffer: Float64Array | null;

  /** Buffer for Kalman gain k */
  private _gainBuffer: Float64Array | null;

  /** Buffer for prediction error */
  private _errorBuffer: Float64Array | null;

  /** Buffer for normalized input */
  private _normalizedBuffer: Float64Array | null;

  // ========================================================================
  // PERFORMANCE METRICS
  // ========================================================================

  /** Sum of squared residuals (for RMSE) */
  private _ssResiduals: number;

  /** Sum of squared total (for R²) */
  private _ssTotal: number;

  /** Running output mean (for R²) */
  private _outputMean: Float64Array | null;

  // ========================================================================
  // CONSTRUCTOR
  // ========================================================================

  /**
   * Creates a new MultivariatePolynomialRegression model.
   *
   * @param config - Optional configuration parameters
   * @throws {Error} If any configuration parameter is invalid
   *
   * @example
   * ```typescript
   * // Default configuration
   * const model1 = new MultivariatePolynomialRegression();
   *
   * // Custom configuration
   * const model2 = new MultivariatePolynomialRegression({
   *   polynomialDegree: 3,
   *   forgettingFactor: 0.95,
   *   enableNormalization: true,
   *   normalizationMethod: 'z-score',
   *   confidenceLevel: 0.99
   * });
   * ```
   */
  constructor(config?: ModelConfiguration) {
    // Validate and assign configuration
    this._polynomialDegree = this._validatePositiveInteger(
      config?.polynomialDegree ?? 2,
      "polynomialDegree",
      1,
    );

    this._enableNormalization = config?.enableNormalization ?? true;

    this._normalizationMethod = this._validateEnum(
      config?.normalizationMethod ?? "min-max",
      ["none", "min-max", "z-score"],
      "normalizationMethod",
    ) as NormalizationMethod;

    this._forgettingFactor = this._validateRange(
      config?.forgettingFactor ?? 0.99,
      "forgettingFactor",
      1e-10,
      1.0,
      false,
      true,
    );
    this._invForgettingFactor = 1.0 / this._forgettingFactor;

    this._initialCovariance = config?.initialCovariance ?? 1000;

    this._regularization = config?.regularization ?? 1e-6;

    this._confidenceLevel = this._validateRange(
      config?.confidenceLevel ?? 0.95,
      "confidenceLevel",
      0,
      1,
      false,
      false,
    );

    // Initialize empty state
    this._isInitialized = false;
    this._inputDimension = 0;
    this._outputDimension = 0;
    this._polynomialFeatureCount = 0;
    this._weights = null;
    this._covarianceMatrix = null;
    this._sampleCount = 0;
    this._featureGenerator = new PolynomialFeatureGenerator(
      this._polynomialDegree,
    );
    this._inputStats = null;
    this._lastRawInput = null;

    // Buffers initialized lazily
    this._phiBuffer = null;
    this._pPhiBuffer = null;
    this._gainBuffer = null;
    this._errorBuffer = null;
    this._normalizedBuffer = null;

    // Metrics
    this._ssResiduals = 0;
    this._ssTotal = 0;
    this._outputMean = null;
  }

  // ========================================================================
  // PUBLIC API: TRAINING
  // ========================================================================

  /**
   * Trains the model incrementally with new data.
   *
   * @description
   * Performs online learning using the Recursive Least Squares algorithm.
   * Each call processes new samples and updates the model weights without
   * needing to retrain from scratch.
   *
   * The RLS update equations are:
   * 1. φ = polynomial_features(normalize(x))
   * 2. k = P·φ / (λ + φᵀ·P·φ)
   * 3. W = W + k·(y - φᵀ·W)ᵀ
   * 4. P = (P - k·φᵀ·P) / λ
   *
   * @param params - Training data {xCoordinates, yCoordinates}
   * @throws {Error} If data dimensions are inconsistent
   *
   * @example
   * ```typescript
   * // Single variable
   * model.fitOnline({
   *   xCoordinates: [[1], [2], [3]],
   *   yCoordinates: [[2], [4], [6]]
   * });
   *
   * // Multivariate (house price from sqft, bedrooms)
   * model.fitOnline({
   *   xCoordinates: [[1500, 3], [2000, 4], [1200, 2]],
   *   yCoordinates: [[300000], [450000], [250000]]
   * });
   *
   * // Multiple outputs
   * model.fitOnline({
   *   xCoordinates: [[1], [2]],
   *   yCoordinates: [[2, 1], [4, 4]]
   * });
   * ```
   */
  fitOnline(params: FitOnlineParams): void {
    const { xCoordinates, yCoordinates } = params;

    // Validate input
    this._validateTrainingData(xCoordinates, yCoordinates);

    const numSamples = xCoordinates.length;
    if (numSamples === 0) return;

    // Lazy initialization on first data
    if (!this._isInitialized) {
      this._initializeModel(xCoordinates[0].length, yCoordinates[0].length);
    }

    // Process each sample
    for (let i = 0; i < numSamples; ++i) {
      this._processSample(xCoordinates[i], yCoordinates[i]);
    }
  }

  // ========================================================================
  // PUBLIC API: PREDICTION
  // ========================================================================

  /**
   * Generates predictions with confidence intervals.
   *
   * @description
   * Produces point predictions along with uncertainty estimates.
   * Can either extrapolate future time steps or predict for specific inputs.
   *
   * Confidence intervals use the t-distribution with df = n - p where
   * n = sample count and p = polynomial feature count.
   *
   * @param params - Prediction parameters
   * @returns PredictionResult with predictions and model diagnostics
   *
   * @example
   * ```typescript
   * // Extrapolate 5 future steps (time series)
   * const result1 = model.predict({ futureSteps: 5 });
   *
   * // Predict for specific inputs
   * const result2 = model.predict({
   *   futureSteps: 0,
   *   inputPoints: [[1800, 3], [2200, 5]]
   * });
   *
   * // Access predictions
   * result1.predictions.forEach((pred, i) => {
   *   console.log(`Prediction ${i}: ${pred.predicted[0]}`);
   *   console.log(`  95% CI: [${pred.lowerBound[0]}, ${pred.upperBound[0]}]`);
   * });
   * ```
   */
  predict(params: PredictParams): PredictionResult {
    const { futureSteps, inputPoints } = params;

    // Generate input points
    const points = inputPoints
      ? this._prepareInputPoints(inputPoints)
      : this._generateFuturePoints(futureSteps);

    // Generate predictions
    const predictions = this._computePredictions(points);

    // Return result with diagnostics
    return {
      predictions,
      confidenceLevel: this._confidenceLevel,
      rSquared: this._computeRSquared(),
      rmse: this._computeRMSE(),
      sampleCount: this._sampleCount,
      isModelReady: this._sampleCount >= this._polynomialFeatureCount,
    };
  }

  // ========================================================================
  // PUBLIC API: MODEL INSPECTION
  // ========================================================================

  /**
   * Returns comprehensive model statistics.
   *
   * @returns ModelSummary with all model parameters and metrics
   *
   * @example
   * ```typescript
   * const summary = model.getModelSummary();
   * console.log(`Initialized: ${summary.isInitialized}`);
   * console.log(`R²: ${summary.rSquared.toFixed(4)}`);
   * console.log(`RMSE: ${summary.rmse.toFixed(4)}`);
   * console.log(`Samples: ${summary.sampleCount}`);
   * ```
   */
  getModelSummary(): ModelSummary {
    return {
      isInitialized: this._isInitialized,
      inputDimension: this._inputDimension,
      outputDimension: this._outputDimension,
      polynomialDegree: this._polynomialDegree,
      polynomialFeatureCount: this._polynomialFeatureCount,
      sampleCount: this._sampleCount,
      rSquared: this._computeRSquared(),
      rmse: this._computeRMSE(),
      normalizationEnabled: this._enableNormalization,
      normalizationMethod: this._normalizationMethod,
    };
  }

  /**
   * Returns the current model weight matrix.
   *
   * @description
   * Weights map polynomial features to outputs.
   * Shape: [polynomialFeatureCount][outputDimension]
   *
   * @returns Weight matrix as nested arrays, or null if not initialized
   *
   * @example
   * ```typescript
   * const weights = model.getWeights();
   * if (weights) {
   *   console.log(`Intercept: ${weights[0][0]}`);
   *   console.log(`Linear coefficient: ${weights[1][0]}`);
   * }
   * ```
   */
  getWeights(): ReadonlyArray<ReadonlyArray<number>> | null {
    if (!this._weights) return null;
    return this._weights.map((row) => Array.from(row));
  }

  /**
   * Returns normalization statistics.
   *
   * @returns NormalizationStats or null if normalization disabled
   *
   * @example
   * ```typescript
   * const stats = model.getNormalizationStats();
   * if (stats) {
   *   console.log(`Min: ${stats.min}`);
   *   console.log(`Max: ${stats.max}`);
   *   console.log(`Mean: ${stats.mean}`);
   *   console.log(`Std: ${stats.std}`);
   * }
   * ```
   */
  getNormalizationStats(): NormalizationStats | null {
    if (!this._enableNormalization || !this._inputStats) {
      return null;
    }

    return {
      min: Array.from(this._inputStats.getMin()),
      max: Array.from(this._inputStats.getMax()),
      mean: Array.from(this._inputStats.getMean()),
      std: Array.from(this._inputStats.getStd()),
      count: this._inputStats.count,
    };
  }

  /**
   * Resets model to initial (untrained) state.
   *
   * @description
   * Clears all weights, covariance, statistics, and buffers.
   * After reset, the model can be trained on new data.
   *
   * @example
   * ```typescript
   * model.reset();
   * // Model is now empty, ready for fresh training
   * ```
   */
  reset(): void {
    this._isInitialized = false;
    this._inputDimension = 0;
    this._outputDimension = 0;
    this._polynomialFeatureCount = 0;
    this._weights = null;
    this._covarianceMatrix = null;
    this._sampleCount = 0;
    this._featureGenerator.reset();
    this._inputStats = null;
    this._lastRawInput = null;

    // Clear buffers
    this._phiBuffer = null;
    this._pPhiBuffer = null;
    this._gainBuffer = null;
    this._errorBuffer = null;
    this._normalizedBuffer = null;

    // Reset metrics
    this._ssResiduals = 0;
    this._ssTotal = 0;
    this._outputMean = null;
  }

  // ========================================================================
  // PRIVATE: VALIDATION HELPERS
  // ========================================================================

  /**
   * Validates positive integer parameter.
   * @private
   */
  private _validatePositiveInteger(
    value: number,
    name: string,
    minimum: number,
  ): number {
    if (!Number.isInteger(value) || value < minimum) {
      throw new Error(
        `${name} must be an integer >= ${minimum}, got: ${value}`,
      );
    }
    return value;
  }

  /**
   * Validates enumerated string parameter.
   * @private
   */
  private _validateEnum<T extends string>(
    value: T,
    allowed: T[],
    name: string,
  ): T {
    if (!allowed.includes(value)) {
      throw new Error(
        `${name} must be one of [${allowed.join(", ")}], got: ${value}`,
      );
    }
    return value;
  }

  /**
   * Validates numeric range parameter.
   * @private
   */
  private _validateRange(
    value: number,
    name: string,
    min: number,
    max: number,
    minInclusive: boolean,
    maxInclusive: boolean,
  ): number {
    const minOk = minInclusive ? value >= min : value > min;
    const maxOk = maxInclusive ? value <= max : value < max;

    if (!minOk || !maxOk) {
      const minBracket = minInclusive ? "[" : "(";
      const maxBracket = maxInclusive ? "]" : ")";
      throw new Error(
        `${name} must be in range ${minBracket}${min}, ${max}${maxBracket}, got: ${value}`,
      );
    }
    return value;
  }

  /**
   * Validates training data consistency.
   * @private
   */
  private _validateTrainingData(
    xCoordinates: ReadonlyArray<ReadonlyArray<number>>,
    yCoordinates: ReadonlyArray<ReadonlyArray<number>>,
  ): void {
    if (xCoordinates.length !== yCoordinates.length) {
      throw new Error(
        `Sample count mismatch: xCoordinates has ${xCoordinates.length}, ` +
          `yCoordinates has ${yCoordinates.length}`,
      );
    }

    if (xCoordinates.length === 0) return;

    const inputDim = xCoordinates[0].length;
    const outputDim = yCoordinates[0].length;

    if (inputDim === 0) {
      throw new Error("Input dimension must be >= 1");
    }
    if (outputDim === 0) {
      throw new Error("Output dimension must be >= 1");
    }

    // Check consistency with existing model
    if (this._isInitialized) {
      if (inputDim !== this._inputDimension) {
        throw new Error(
          `Input dimension mismatch: expected ${this._inputDimension}, got ${inputDim}`,
        );
      }
      if (outputDim !== this._outputDimension) {
        throw new Error(
          `Output dimension mismatch: expected ${this._outputDimension}, got ${outputDim}`,
        );
      }
    }

    // Validate all samples
    for (let i = 0; i < xCoordinates.length; ++i) {
      if (xCoordinates[i].length !== inputDim) {
        throw new Error(`Inconsistent input dimension at sample ${i}`);
      }
      if (yCoordinates[i].length !== outputDim) {
        throw new Error(`Inconsistent output dimension at sample ${i}`);
      }
    }
  }

  // ========================================================================
  // PRIVATE: INITIALIZATION
  // ========================================================================

  /**
   * Initializes model structures for given dimensions.
   * @private
   */
  private _initializeModel(inputDim: number, outputDim: number): void {
    this._inputDimension = inputDim;
    this._outputDimension = outputDim;

    // Initialize feature generator
    this._featureGenerator.initialize(inputDim);
    this._polynomialFeatureCount = this._featureGenerator.featureCount;

    const pfc = this._polynomialFeatureCount;

    // Initialize weight matrix W to zeros
    this._weights = MatrixOperations.createMatrix(pfc, outputDim);

    // Initialize covariance matrix P to diagonal
    this._covarianceMatrix = MatrixOperations.createDiagonalMatrix(
      pfc,
      this._initialCovariance,
    );

    // Initialize statistics tracker
    if (this._enableNormalization) {
      this._inputStats = new IncrementalStatistics(inputDim);
    }

    // Allocate reusable buffers
    this._phiBuffer = new Float64Array(pfc);
    this._pPhiBuffer = new Float64Array(pfc);
    this._gainBuffer = new Float64Array(pfc);
    this._errorBuffer = new Float64Array(outputDim);
    this._normalizedBuffer = new Float64Array(inputDim);
    this._lastRawInput = new Float64Array(inputDim);
    this._outputMean = new Float64Array(outputDim);

    this._isInitialized = true;
  }

  // ========================================================================
  // PRIVATE: TRAINING CORE
  // ========================================================================

  /**
   * Processes a single training sample.
   * @private
   */
  private _processSample(
    x: ReadonlyArray<number>,
    y: ReadonlyArray<number>,
  ): void {
    // Update normalization statistics
    if (this._enableNormalization && this._inputStats) {
      this._inputStats.update(x);
    }

    // Store raw input for extrapolation
    for (let i = 0; i < x.length; ++i) {
      this._lastRawInput![i] = x[i];
    }

    // Normalize input
    this._normalizeInput(x, this._normalizedBuffer!);

    // Generate polynomial features
    this._featureGenerator.transform(this._normalizedBuffer!, this._phiBuffer!);

    // Perform RLS update
    this._rlsUpdate(this._phiBuffer!, y);

    ++this._sampleCount;
  }

  /**
   * Normalizes input using configured method.
   * @private
   */
  private _normalizeInput(
    input: ReadonlyArray<number>,
    output: Float64Array,
  ): void {
    const dim = input.length;

    if (!this._enableNormalization || this._normalizationMethod === "none") {
      for (let i = 0; i < dim; ++i) {
        output[i] = input[i];
      }
      return;
    }

    const stats = this._inputStats!;

    if (this._normalizationMethod === "min-max") {
      const min = stats.getMin();
      const max = stats.getMax();

      for (let i = 0; i < dim; ++i) {
        const range = max[i] - min[i];
        output[i] = range > 1e-10 ? (input[i] - min[i]) / range : 0.5;
      }
    } else {
      // z-score
      const mean = stats.getMean();
      const std = stats.getStd();

      for (let i = 0; i < dim; ++i) {
        output[i] = std[i] > 1e-10 ? (input[i] - mean[i]) / std[i] : 0;
      }
    }
  }

  /**
   * Performs RLS (Recursive Least Squares) weight update.
   *
   * @description
   * Implements the standard RLS algorithm:
   * 1. k = P·φ / (λ + φᵀ·P·φ)
   * 2. W = W + k·(y - Wᵀ·φ)ᵀ
   * 3. P = (P - k·φᵀ·P) / λ
   *
   * @private
   */
  private _rlsUpdate(phi: Float64Array, y: ReadonlyArray<number>): void {
    const P = this._covarianceMatrix!;
    const W = this._weights!;
    const pPhi = this._pPhiBuffer!;
    const k = this._gainBuffer!;
    const error = this._errorBuffer!;

    const pfc = this._polynomialFeatureCount;
    const outDim = this._outputDimension;

    // Step 1: Compute P × φ
    MatrixOperations.matrixVectorMultiply(P, phi, pPhi);

    // Step 2: Compute denominator: λ + φᵀ × P × φ
    const denom = this._forgettingFactor +
      MatrixOperations.dotProduct(phi, pPhi);
    const invDenom = 1.0 / denom;

    // Step 3: Compute Kalman gain: k = (P × φ) / denom
    for (let i = 0; i < pfc; ++i) {
      k[i] = pPhi[i] * invDenom;
    }

    // Step 4: Compute prediction and error: e = y - Wᵀ × φ
    for (let j = 0; j < outDim; ++j) {
      let pred = 0.0;
      for (let i = 0; i < pfc; ++i) {
        pred += phi[i] * W[i][j];
      }
      error[j] = y[j] - pred;
    }

    // Update metrics (using prior error)
    this._updateMetrics(y, error);

    // Step 5: Update weights: W = W + k × eᵀ
    for (let i = 0; i < pfc; ++i) {
      const ki = k[i];
      const wRow = W[i];
      for (let j = 0; j < outDim; ++j) {
        wRow[j] += ki * error[j];
      }
    }

    // Step 6: Update covariance: P = (P - k × (φᵀ × P)ᵀ) / λ
    MatrixOperations.updateCovarianceInPlace(
      P,
      k,
      pPhi,
      this._invForgettingFactor,
    );

    // Add regularization for numerical stability
    MatrixOperations.addRegularization(P, this._regularization);
  }

  /**
   * Updates running metrics for R² and RMSE.
   * @private
   */
  private _updateMetrics(
    y: ReadonlyArray<number>,
    error: Float64Array,
  ): void {
    const outDim = this._outputDimension;
    const n = this._sampleCount + 1;
    const invN = 1.0 / n;

    // Update sum of squared residuals
    for (let j = 0; j < outDim; ++j) {
      this._ssResiduals += error[j] * error[j];
    }

    // Update sum of squared total (Welford's algorithm)
    for (let j = 0; j < outDim; ++j) {
      const delta = y[j] - this._outputMean![j];
      this._outputMean![j] += delta * invN;
      const delta2 = y[j] - this._outputMean![j];
      this._ssTotal += delta * delta2;
    }
  }

  // ========================================================================
  // PRIVATE: PREDICTION CORE
  // ========================================================================

  /**
   * Prepares input points for prediction (normalizes them).
   * @private
   */
  private _prepareInputPoints(
    inputPoints: ReadonlyArray<ReadonlyArray<number>>,
  ): Float64Array[] {
    return inputPoints.map((point) => {
      const normalized = new Float64Array(point.length);
      this._normalizeInput(point, normalized);
      return normalized;
    });
  }

  /**
   * Generates extrapolation points for time series.
   * @private
   */
  private _generateFuturePoints(steps: number): Float64Array[] {
    if (!this._isInitialized || !this._lastRawInput || steps <= 0) {
      return [];
    }

    const points: Float64Array[] = [];
    const currentRaw = new Float64Array(this._lastRawInput);
    const normalized = new Float64Array(this._inputDimension);

    for (let i = 0; i < steps; ++i) {
      // Increment first dimension (assumed time index)
      currentRaw[0] += 1;

      // Normalize and copy
      this._normalizeInput(currentRaw as any, normalized);
      points.push(new Float64Array(normalized));
    }

    return points;
  }

  /**
   * Computes predictions with confidence intervals.
   * @private
   */
  private _computePredictions(points: Float64Array[]): SinglePrediction[] {
    if (!this._isInitialized || !this._weights || points.length === 0) {
      return [];
    }

    const predictions: SinglePrediction[] = [];
    const phi = this._phiBuffer!;
    const pPhi = this._pPhiBuffer!;

    const pfc = this._polynomialFeatureCount;
    const outDim = this._outputDimension;

    // Compute t-critical value
    const df = Math.max(1, this._sampleCount - pfc);
    const tCrit = StatisticalDistribution.getTCritical(
      this._confidenceLevel,
      df,
    );

    // Estimate sigma² from residuals
    const mse = df > 0 ? this._ssResiduals / df : 0;

    for (const point of points) {
      // Generate polynomial features
      this._featureGenerator.transform(point, phi);

      // Compute φᵀ × P × φ for standard error
      MatrixOperations.matrixVectorMultiply(this._covarianceMatrix!, phi, pPhi);
      const varFactor = MatrixOperations.dotProduct(phi, pPhi);
      const se = Math.sqrt(Math.max(0, mse * varFactor));

      // Compute predictions
      const predicted: number[] = [];
      const lowerBound: number[] = [];
      const upperBound: number[] = [];
      const standardError: number[] = [];

      for (let j = 0; j < outDim; ++j) {
        let pred = 0.0;
        for (let i = 0; i < pfc; ++i) {
          pred += phi[i] * this._weights![i][j];
        }

        const margin = tCrit * se;

        predicted.push(pred);
        lowerBound.push(pred - margin);
        upperBound.push(pred + margin);
        standardError.push(se);
      }

      predictions.push({
        predicted,
        lowerBound,
        upperBound,
        standardError,
      });
    }

    return predictions;
  }

  /**
   * Computes R-squared (coefficient of determination).
   * @private
   */
  private _computeRSquared(): number {
    if (this._sampleCount < 2 || this._ssTotal <= 1e-10) {
      return 0;
    }

    const r2 = 1 - (this._ssResiduals / this._ssTotal);
    return Math.max(0, Math.min(1, r2));
  }

  /**
   * Computes RMSE (Root Mean Square Error).
   * @private
   */
  private _computeRMSE(): number {
    if (this._sampleCount === 0) {
      return 0;
    }

    return Math.sqrt(this._ssResiduals / this._sampleCount);
  }
}

// ============================================================================
// MODULE EXPORTS
// ============================================================================

export {
  type FitOnlineParams,
  type ModelConfiguration,
  type ModelSummary,
  MultivariatePolynomialRegression,
  type NormalizationMethod,
  type NormalizationStats,
  type PredictionResult,
  type PredictParams,
  type SinglePrediction,
};
