// =============================================================================
// INTERFACES
// =============================================================================

/**
 * Configuration options for the MultivariatePolynomialRegression model.
 * @interface
 */
interface IMultivariatePolynomialRegressionConfig {
  /** Degree of polynomial features (default: 2, minimum: 1) */
  polynomialDegree: number;
  /** Whether to enable input normalization (default: true) */
  enableNormalization: boolean;
  /** Method used for normalization (default: 'min-max') */
  normalizationMethod: "none" | "min-max" | "z-score";
  /** Forgetting factor λ for RLS (default: 0.99, range: (0, 1]) */
  forgettingFactor: number;
  /** Initial covariance matrix diagonal value (default: 1000) */
  initialCovariance: number;
  /** Regularization parameter for numerical stability (default: 1e-6) */
  regularization: number;
  /** Confidence level for prediction intervals (default: 0.95, range: (0, 1)) */
  confidenceLevel: number;
}

/**
 * Result of a single prediction including confidence intervals.
 * @interface
 */
interface SinglePrediction {
  /** Predicted output values for each output dimension */
  predicted: number[];
  /** Lower bound of confidence interval for each output dimension */
  lowerBound: number[];
  /** Upper bound of confidence interval for each output dimension */
  upperBound: number[];
  /** Standard error of predictions for each output dimension */
  standardError: number[];
}

/**
 * Complete prediction result including model metrics and all predictions.
 * @interface
 */
interface PredictionResult {
  /** Array of individual predictions with confidence intervals */
  predictions: SinglePrediction[];
  /** Confidence level used for interval calculation */
  confidenceLevel: number;
  /** R-squared goodness of fit metric (coefficient of determination) */
  rSquared: number;
  /** Root Mean Squared Error */
  rmse: number;
  /** Total number of training samples processed */
  sampleCount: number;
  /** Whether the model has sufficient data for reliable predictions */
  isModelReady: boolean;
}

/**
 * Comprehensive model summary statistics.
 * @interface
 */
interface ModelSummary {
  /** Whether the model has been initialized with training data */
  isInitialized: boolean;
  /** Number of input features/dimensions */
  inputDimension: number;
  /** Number of output features/dimensions */
  outputDimension: number;
  /** Polynomial degree used for feature expansion */
  polynomialDegree: number;
  /** Total number of polynomial features generated */
  polynomialFeatureCount: number;
  /** Number of training samples processed */
  sampleCount: number;
  /** R-squared metric (0 to 1, higher is better) */
  rSquared: number;
  /** Root Mean Squared Error (lower is better) */
  rmse: number;
  /** Whether input normalization is enabled */
  normalizationEnabled: boolean;
  /** Normalization method being used */
  normalizationMethod: string;
}

/**
 * Statistics collected during normalization.
 * @interface
 */
interface NormalizationStats {
  /** Minimum values observed for each input dimension */
  min: number[];
  /** Maximum values observed for each input dimension */
  max: number[];
  /** Running mean for each input dimension */
  mean: number[];
  /** Running standard deviation for each input dimension */
  std: number[];
  /** Number of samples processed for statistics */
  count: number;
}

/**
 * Interface for matrix operations with typed arrays.
 * All operations are designed to minimize allocations.
 * @interface
 */
interface IMatrixOperations {
  /** Creates a new zero-filled matrix as Float64Array */
  create(rows: number, cols: number): Float64Array;
  /** Gets element at (row, col) position */
  get(matrix: Float64Array, cols: number, row: number, col: number): number;
  /** Sets element at (row, col) position */
  set(
    matrix: Float64Array,
    cols: number,
    row: number,
    col: number,
    value: number,
  ): void;
  /** In-place matrix-vector multiplication: result = matrix · vec */
  matVecMulInPlace(
    result: Float64Array,
    matrix: Float64Array,
    rows: number,
    cols: number,
    vec: Float64Array,
  ): void;
  /** In-place outer product subtraction: matrix -= scale * vec1 ⊗ vec2ᵀ */
  outerProductSubInPlace(
    matrix: Float64Array,
    cols: number,
    vec1: Float64Array,
    vec2: Float64Array,
    scale: number,
  ): void;
  /** Computes vector dot product */
  dot(vec1: Float64Array, vec2: Float64Array): number;
  /** In-place scalar multiplication: matrix *= scale */
  scaleInPlace(matrix: Float64Array, scale: number): void;
  /** Adds value to matrix diagonal: P[i,i] += value */
  addDiagonalRegularization(
    matrix: Float64Array,
    cols: number,
    value: number,
  ): void;
  /** Copies source array to destination */
  copy(dest: Float64Array, src: Float64Array): void;
}

/**
 * Interface for normalization strategies (Strategy Pattern).
 * @interface
 */
interface INormalizationStrategy {
  /** Normalizes input values, storing result in output array */
  normalizeInPlace(
    output: Float64Array,
    input: Float64Array,
    stats: NormalizationStats,
  ): void;
  /** Denormalizes values, storing result in output array */
  denormalizeInPlace(
    output: Float64Array,
    input: Float64Array,
    stats: NormalizationStats,
  ): void;
  /** Updates running statistics with new data point */
  updateStats(stats: NormalizationStats, input: Float64Array): void;
  /** Resets statistics to initial state */
  resetStats(stats: NormalizationStats): void;
}

/**
 * Interface for polynomial feature generation.
 * @interface
 */
interface IPolynomialFeatureGenerator {
  /** Calculates total number of polynomial features */
  getFeatureCount(inputDim: number, degree: number): number;
  /** Generates polynomial features in-place */
  generateFeaturesInPlace(
    output: Float64Array,
    input: Float64Array,
    degree: number,
  ): void;
  /** Initializes generator for given dimensions */
  initialize(inputDim: number, degree: number): void;
}

/**
 * Interface for covariance matrix management in RLS.
 * @interface
 */
interface ICovarianceManager {
  /** Initializes covariance matrix as scaled identity */
  initialize(featureCount: number, initialValue: number): void;
  /** Gets current covariance matrix P */
  getCovarianceMatrix(): Float64Array;
  /** Computes RLS gain vector: k = P·φ / (λ + φᵀ·P·φ) */
  computeGainVector(
    features: Float64Array,
    forgettingFactor: number,
    gainOut: Float64Array,
  ): number;
  /** Updates covariance: P = (P - k·φᵀ·P) / λ */
  updateCovariance(
    gain: Float64Array,
    features: Float64Array,
    forgettingFactor: number,
  ): void;
  /** Applies regularization to diagonal for numerical stability */
  applyRegularization(regularization: number): void;
  /** Resets covariance to initial state */
  reset(): void;
}

/**
 * Interface for weight matrix management.
 * @interface
 */
interface IWeightManager {
  /** Initializes weight matrix to zeros */
  initialize(featureCount: number, outputDim: number): void;
  /** Gets current weight matrix */
  getWeights(): Float64Array;
  /** Updates weights: W = W + k·eᵀ */
  updateWeights(gain: Float64Array, error: Float64Array): void;
  /** Computes prediction: y = Wᵀ·φ */
  predict(features: Float64Array, output: Float64Array): void;
  /** Resets weights to zeros */
  reset(): void;
  /** Gets weights as 2D array for external use */
  getWeightsAs2D(): number[][];
}

/**
 * Interface for prediction engine with confidence intervals.
 * @interface
 */
interface IPredictionEngine {
  /** Makes prediction with confidence intervals using covariance */
  predictWithConfidence(
    features: Float64Array,
    covariance: Float64Array,
    weights: Float64Array,
    featureCount: number,
    outputDim: number,
    confidenceLevel: number,
    sampleCount: number,
    result: SinglePrediction,
  ): void;
}

/**
 * Interface for tracking model performance statistics.
 * @interface
 */
interface IStatisticsTracker {
  /** Updates statistics with new actual vs predicted values */
  update(actual: Float64Array, predicted: Float64Array): void;
  /** Gets current R-squared value */
  getRSquared(): number;
  /** Gets current RMSE value */
  getRMSE(): number;
  /** Gets total sample count */
  getSampleCount(): number;
  /** Resets all statistics */
  reset(): void;
}

// =============================================================================
// MATRIX OPERATIONS
// =============================================================================

/**
 * Efficient matrix operations using Float64Array with in-place modifications.
 *
 * @remarks
 * Matrices are stored in row-major order in a single Float64Array.
 * Element at (i, j) is located at index: i * cols + j.
 *
 * All methods are optimized to minimize memory allocations and use
 * loop unrolling for better performance on modern JavaScript engines.
 *
 * @example
 * ```typescript
 * const ops = new MatrixOperations();
 * const matrix = ops.create(3, 3); // 3x3 zero matrix
 * ops.set(matrix, 3, 1, 2, 5.0);   // Set element at row 1, col 2
 * const val = ops.get(matrix, 3, 1, 2); // Get element: 5.0
 * ```
 *
 * @implements {IMatrixOperations}
 */
class MatrixOperations implements IMatrixOperations {
  /** Reusable temporary vector for intermediate calculations */
  private tempVec: Float64Array | null = null;

  /**
   * Creates a new matrix initialized with zeros.
   *
   * @param rows - Number of rows in the matrix
   * @param cols - Number of columns in the matrix
   * @returns A new Float64Array of size rows × cols, initialized to zero
   *
   * @remarks
   * Time complexity: O(rows × cols) for zero initialization
   * Space complexity: O(rows × cols)
   */
  public create(rows: number, cols: number): Float64Array {
    return new Float64Array(rows * cols);
  }

  /**
   * Gets the element at the specified row and column.
   *
   * @param matrix - The matrix array in row-major order
   * @param cols - Number of columns in the matrix
   * @param row - Row index (0-based)
   * @param col - Column index (0-based)
   * @returns The value at position (row, col)
   *
   * @remarks
   * Time complexity: O(1)
   * Space complexity: O(1)
   */
  public get(
    matrix: Float64Array,
    cols: number,
    row: number,
    col: number,
  ): number {
    return matrix[row * cols + col];
  }

  /**
   * Sets the element at the specified row and column.
   *
   * @param matrix - The matrix array in row-major order
   * @param cols - Number of columns in the matrix
   * @param row - Row index (0-based)
   * @param col - Column index (0-based)
   * @param value - The value to set
   *
   * @remarks
   * Time complexity: O(1)
   * Space complexity: O(1)
   */
  public set(
    matrix: Float64Array,
    cols: number,
    row: number,
    col: number,
    value: number,
  ): void {
    matrix[row * cols + col] = value;
  }

  /**
   * In-place matrix-vector multiplication: result = matrix · vec
   *
   * @param result - Output vector (must be preallocated with length = rows)
   * @param matrix - Input matrix in row-major order
   * @param rows - Number of rows in matrix
   * @param cols - Number of columns in matrix
   * @param vec - Input vector (length must equal cols)
   *
   * @remarks
   * Mathematical formula: result[i] = Σⱼ matrix[i,j] × vec[j]
   *
   * Uses 4-way loop unrolling for better CPU pipeline utilization.
   *
   * Time complexity: O(rows × cols)
   * Space complexity: O(1) - no allocations, uses preallocated result
   */
  public matVecMulInPlace(
    result: Float64Array,
    matrix: Float64Array,
    rows: number,
    cols: number,
    vec: Float64Array,
  ): void {
    for (let i = 0; i < rows; i++) {
      let sum = 0;
      const rowOffset = i * cols;

      // 4-way unrolled loop for better performance
      let j = 0;
      const unrollLimit = cols - 3;
      for (; j < unrollLimit; j += 4) {
        sum += matrix[rowOffset + j] * vec[j] +
          matrix[rowOffset + j + 1] * vec[j + 1] +
          matrix[rowOffset + j + 2] * vec[j + 2] +
          matrix[rowOffset + j + 3] * vec[j + 3];
      }
      // Handle remaining elements
      for (; j < cols; j++) {
        sum += matrix[rowOffset + j] * vec[j];
      }
      result[i] = sum;
    }
  }

  /**
   * In-place outer product subtraction: matrix -= scale × vec1 ⊗ vec2ᵀ
   *
   * @param matrix - Matrix to update in-place
   * @param cols - Number of columns in matrix
   * @param vec1 - First vector (typically gain vector k)
   * @param vec2 - Second vector (typically φᵀP result)
   * @param scale - Scaling factor
   *
   * @remarks
   * Used in RLS covariance update: P = P - k ⊗ (φᵀP)
   * Mathematical formula: matrix[i,j] -= scale × vec1[i] × vec2[j]
   *
   * Time complexity: O(vec1.length × vec2.length)
   * Space complexity: O(1) - in-place operation
   */
  public outerProductSubInPlace(
    matrix: Float64Array,
    cols: number,
    vec1: Float64Array,
    vec2: Float64Array,
    scale: number,
  ): void {
    const rows = vec1.length;
    for (let i = 0; i < rows; i++) {
      const rowOffset = i * cols;
      const scaledV1i = vec1[i] * scale;
      for (let j = 0; j < cols; j++) {
        matrix[rowOffset + j] -= scaledV1i * vec2[j];
      }
    }
  }

  /**
   * Computes the dot product of two vectors: vec1 · vec2
   *
   * @param vec1 - First vector
   * @param vec2 - Second vector (must have same length as vec1)
   * @returns The scalar dot product
   *
   * @remarks
   * Mathematical formula: result = Σᵢ vec1[i] × vec2[i]
   *
   * Uses 4-way loop unrolling for performance optimization.
   *
   * Time complexity: O(n) where n = vector length
   * Space complexity: O(1)
   */
  public dot(vec1: Float64Array, vec2: Float64Array): number {
    let sum = 0;
    const len = vec1.length;

    // 4-way unrolled loop
    let i = 0;
    const unrollLimit = len - 3;
    for (; i < unrollLimit; i += 4) {
      sum += vec1[i] * vec2[i] +
        vec1[i + 1] * vec2[i + 1] +
        vec1[i + 2] * vec2[i + 2] +
        vec1[i + 3] * vec2[i + 3];
    }
    // Handle remaining elements
    for (; i < len; i++) {
      sum += vec1[i] * vec2[i];
    }
    return sum;
  }

  /**
   * In-place scalar multiplication: matrix *= scale
   *
   * @param matrix - Matrix/vector to scale
   * @param scale - Scaling factor
   *
   * @remarks
   * Time complexity: O(n) where n = array length
   * Space complexity: O(1)
   */
  public scaleInPlace(matrix: Float64Array, scale: number): void {
    const len = matrix.length;
    for (let i = 0; i < len; i++) {
      matrix[i] *= scale;
    }
  }

  /**
   * Adds regularization value to the diagonal: P[i,i] += value
   *
   * @param matrix - Square matrix to regularize
   * @param cols - Number of columns (= rows for square matrix)
   * @param value - Regularization value to add to diagonal
   *
   * @remarks
   * Prevents numerical instability from small eigenvalues in covariance matrix.
   *
   * Time complexity: O(cols)
   * Space complexity: O(1)
   */
  public addDiagonalRegularization(
    matrix: Float64Array,
    cols: number,
    value: number,
  ): void {
    for (let i = 0; i < cols; i++) {
      matrix[i * cols + i] += value;
    }
  }

  /**
   * Copies source array to destination array.
   *
   * @param dest - Destination array
   * @param src - Source array
   *
   * @remarks
   * Time complexity: O(n)
   * Space complexity: O(1)
   */
  public copy(dest: Float64Array, src: Float64Array): void {
    dest.set(src);
  }

  /**
   * Ensures temporary vector is allocated with sufficient size.
   * Reuses existing buffer if large enough.
   *
   * @param size - Required minimum size
   * @returns The temporary vector
   *
   * @internal
   */
  public ensureTempVec(size: number): Float64Array {
    if (this.tempVec === null || this.tempVec.length < size) {
      this.tempVec = new Float64Array(size);
    }
    return this.tempVec;
  }
}

// =============================================================================
// NORMALIZATION STRATEGIES
// =============================================================================

/**
 * No-operation normalization strategy.
 * Passes through values unchanged - used when normalization is disabled.
 *
 * @implements {INormalizationStrategy}
 */
class NoNormalizationStrategy implements INormalizationStrategy {
  /**
   * Copies input to output without any transformation.
   *
   * @param output - Output array
   * @param input - Input array
   * @param _stats - Unused normalization statistics
   */
  public normalizeInPlace(
    output: Float64Array,
    input: Float64Array,
    _stats: NormalizationStats,
  ): void {
    output.set(input);
  }

  /**
   * Copies input to output without any transformation.
   *
   * @param output - Output array
   * @param input - Input array
   * @param _stats - Unused normalization statistics
   */
  public denormalizeInPlace(
    output: Float64Array,
    input: Float64Array,
    _stats: NormalizationStats,
  ): void {
    output.set(input);
  }

  /**
   * Updates only the sample count.
   *
   * @param stats - Statistics to update
   * @param _input - Unused input data
   */
  public updateStats(stats: NormalizationStats, _input: Float64Array): void {
    stats.count++;
  }

  /**
   * Resets the sample count.
   *
   * @param stats - Statistics to reset
   */
  public resetStats(stats: NormalizationStats): void {
    stats.count = 0;
  }
}

/**
 * Min-Max normalization strategy.
 * Scales values to approximately [0, 1] range using: (x - min) / (max - min)
 *
 * @remarks
 * - Tracks running min and max values incrementally
 * - Handles zero-range case (max = min) by returning 0.5
 * - Clamps normalized values to [-10, 10] to prevent extreme outliers
 *
 * @example
 * ```typescript
 * const strategy = new MinMaxNormalizationStrategy();
 * const stats = { min: [0], max: [100], mean: [], std: [], count: 10 };
 * const input = new Float64Array([50]);
 * const output = new Float64Array(1);
 * strategy.normalizeInPlace(output, input, stats);
 * // output[0] = 0.5
 * ```
 *
 * @implements {INormalizationStrategy}
 */
class MinMaxNormalizationStrategy implements INormalizationStrategy {
  /** Minimum acceptable range to prevent division by near-zero */
  private static readonly MIN_RANGE: number = 1e-10;
  /** Lower clamp bound for normalized values */
  private static readonly CLAMP_MIN: number = -10;
  /** Upper clamp bound for normalized values */
  private static readonly CLAMP_MAX: number = 10;

  /**
   * Normalizes input using min-max scaling.
   *
   * Formula: normalized = (x - min) / (max - min)
   *
   * @param output - Output array for normalized values
   * @param input - Input array to normalize
   * @param stats - Current normalization statistics with min/max values
   *
   * @remarks
   * Time complexity: O(n) where n = input length
   * Space complexity: O(1)
   */
  public normalizeInPlace(
    output: Float64Array,
    input: Float64Array,
    stats: NormalizationStats,
  ): void {
    const len = input.length;
    for (let i = 0; i < len; i++) {
      const range = stats.max[i] - stats.min[i];
      if (range < MinMaxNormalizationStrategy.MIN_RANGE) {
        // Zero or near-zero range: output midpoint value
        output[i] = 0.5;
      } else {
        const normalized = (input[i] - stats.min[i]) / range;
        // Clamp to prevent extreme outliers from causing numerical issues
        output[i] = Math.max(
          MinMaxNormalizationStrategy.CLAMP_MIN,
          Math.min(MinMaxNormalizationStrategy.CLAMP_MAX, normalized),
        );
      }
    }
  }

  /**
   * Denormalizes values back to original scale.
   *
   * Formula: x = normalized × (max - min) + min
   *
   * @param output - Output array for denormalized values
   * @param input - Input array of normalized values
   * @param stats - Normalization statistics with min/max values
   *
   * @remarks
   * Time complexity: O(n)
   * Space complexity: O(1)
   */
  public denormalizeInPlace(
    output: Float64Array,
    input: Float64Array,
    stats: NormalizationStats,
  ): void {
    const len = input.length;
    for (let i = 0; i < len; i++) {
      const range = stats.max[i] - stats.min[i];
      output[i] = input[i] * range + stats.min[i];
    }
  }

  /**
   * Updates running min/max statistics with a new data point.
   *
   * @param stats - Statistics to update
   * @param input - New data point
   *
   * @remarks
   * Time complexity: O(n)
   * Space complexity: O(1)
   */
  public updateStats(stats: NormalizationStats, input: Float64Array): void {
    const len = input.length;
    if (stats.count === 0) {
      // First sample: initialize min/max to this value
      for (let i = 0; i < len; i++) {
        stats.min[i] = input[i];
        stats.max[i] = input[i];
      }
    } else {
      // Update running min/max
      for (let i = 0; i < len; i++) {
        if (input[i] < stats.min[i]) stats.min[i] = input[i];
        if (input[i] > stats.max[i]) stats.max[i] = input[i];
      }
    }
    stats.count++;
  }

  /**
   * Resets statistics to initial state (min=∞, max=-∞).
   *
   * @param stats - Statistics to reset
   */
  public resetStats(stats: NormalizationStats): void {
    const len = stats.min.length;
    for (let i = 0; i < len; i++) {
      stats.min[i] = Infinity;
      stats.max[i] = -Infinity;
    }
    stats.count = 0;
  }
}

/**
 * Z-score (standard score) normalization strategy.
 * Scales values using: (x - mean) / std
 *
 * @remarks
 * - Uses Welford's online algorithm for numerically stable running statistics
 * - Handles zero-variance case by returning 0
 * - Clamps normalized values to [-10, 10] to prevent extreme outliers
 *
 * Welford's algorithm formula:
 * - delta = x - mean
 * - mean_new = mean + delta / n
 * - M2_new = M2 + delta × (x - mean_new)
 * - variance = M2 / (n - 1) [Bessel's correction]
 *
 * @example
 * ```typescript
 * const strategy = new ZScoreNormalizationStrategy();
 * const stats = { min: [], max: [], mean: [50], std: [10], count: 100 };
 * const input = new Float64Array([60]);
 * const output = new Float64Array(1);
 * strategy.normalizeInPlace(output, input, stats);
 * // output[0] = 1.0 (one standard deviation above mean)
 * ```
 *
 * @implements {INormalizationStrategy}
 */
class ZScoreNormalizationStrategy implements INormalizationStrategy {
  /** Minimum acceptable standard deviation */
  private static readonly MIN_STD: number = 1e-10;
  /** Lower clamp bound for normalized values */
  private static readonly CLAMP_MIN: number = -10;
  /** Upper clamp bound for normalized values */
  private static readonly CLAMP_MAX: number = 10;

  /** Running sum of squared differences for Welford's algorithm */
  private m2: Float64Array | null = null;

  /**
   * Normalizes input using z-score scaling.
   *
   * Formula: z = (x - μ) / σ
   *
   * @param output - Output array for normalized values
   * @param input - Input array to normalize
   * @param stats - Current statistics with mean and std
   *
   * @remarks
   * Time complexity: O(n)
   * Space complexity: O(1)
   */
  public normalizeInPlace(
    output: Float64Array,
    input: Float64Array,
    stats: NormalizationStats,
  ): void {
    const len = input.length;
    for (let i = 0; i < len; i++) {
      const std = stats.std[i];
      if (std < ZScoreNormalizationStrategy.MIN_STD) {
        // Zero variance: all values are the same, normalize to 0
        output[i] = 0;
      } else {
        const normalized = (input[i] - stats.mean[i]) / std;
        output[i] = Math.max(
          ZScoreNormalizationStrategy.CLAMP_MIN,
          Math.min(ZScoreNormalizationStrategy.CLAMP_MAX, normalized),
        );
      }
    }
  }

  /**
   * Denormalizes z-scores back to original scale.
   *
   * Formula: x = z × σ + μ
   *
   * @param output - Output array for denormalized values
   * @param input - Input array of z-scores
   * @param stats - Statistics with mean and std
   *
   * @remarks
   * Time complexity: O(n)
   * Space complexity: O(1)
   */
  public denormalizeInPlace(
    output: Float64Array,
    input: Float64Array,
    stats: NormalizationStats,
  ): void {
    const len = input.length;
    for (let i = 0; i < len; i++) {
      output[i] = input[i] * stats.std[i] + stats.mean[i];
    }
  }

  /**
   * Updates running mean and variance using Welford's online algorithm.
   *
   * @param stats - Statistics to update
   * @param input - New data point
   *
   * @remarks
   * Welford's algorithm provides numerically stable computation of variance.
   *
   * Time complexity: O(n)
   * Space complexity: O(n) for M2 array (allocated lazily)
   */
  public updateStats(stats: NormalizationStats, input: Float64Array): void {
    const len = input.length;

    // Lazy initialization of M2 array
    if (this.m2 === null || this.m2.length !== len) {
      this.m2 = new Float64Array(len);
    }

    stats.count++;
    const n = stats.count;

    for (let i = 0; i < len; i++) {
      // Welford's online algorithm
      const delta = input[i] - stats.mean[i];
      stats.mean[i] += delta / n;
      const delta2 = input[i] - stats.mean[i];
      this.m2[i] += delta * delta2;

      // Update standard deviation with Bessel's correction
      if (n > 1) {
        stats.std[i] = Math.sqrt(this.m2[i] / (n - 1));
      } else {
        stats.std[i] = 0;
      }
    }
  }

  /**
   * Resets statistics and internal M2 array.
   *
   * @param stats - Statistics to reset
   */
  public resetStats(stats: NormalizationStats): void {
    const len = stats.mean.length;
    for (let i = 0; i < len; i++) {
      stats.mean[i] = 0;
      stats.std[i] = 0;
    }
    if (this.m2 !== null) {
      this.m2.fill(0);
    }
    stats.count = 0;
  }
}

// =============================================================================
// POLYNOMIAL FEATURE GENERATOR
// =============================================================================

/**
 * Generates polynomial features from input vectors for regression.
 *
 * @remarks
 * For polynomial degree d and input dimension n, generates all monomial
 * combinations up to total degree d.
 *
 * The number of features is C(n + d, d) = (n + d)! / (n! × d!)
 *
 * Example for degree 2, input [x₁, x₂]:
 * Features: [1, x₁, x₁², x₁x₂, x₂, x₂²]
 * (Note: order may vary based on enumeration algorithm)
 *
 * Uses precomputed index patterns for efficient feature generation.
 *
 * @example
 * ```typescript
 * const generator = new PolynomialFeatureGenerator();
 * generator.initialize(2, 2); // 2D input, degree 2
 * const input = new Float64Array([2.0, 3.0]);
 * const features = new Float64Array(6);
 * generator.generateFeaturesInPlace(features, input, 2);
 * // features = [1, 2, 4, 6, 3, 9] (1, x₁, x₁², x₁x₂, x₂, x₂²)
 * ```
 *
 * @implements {IPolynomialFeatureGenerator}
 */
class PolynomialFeatureGenerator implements IPolynomialFeatureGenerator {
  /**
   * Cached feature indices - each inner array contains the input indices
   * to multiply together for that feature.
   * e.g., [0, 1] means x₁ × x₂, [] means constant (1)
   */
  private featureIndices: Int32Array[] = [];
  /** Cached total number of features */
  private cachedFeatureCount: number = 0;
  /** Input dimension for current configuration */
  private inputDim: number = 0;
  /** Polynomial degree for current configuration */
  private degree: number = 0;
  /** Whether generator has been initialized */
  private initialized: boolean = false;

  /**
   * Calculates the number of polynomial features for given parameters.
   *
   * Formula: C(n + d, d) = (n + d)! / (n! × d!)
   *
   * @param inputDim - Number of input dimensions (n)
   * @param degree - Polynomial degree (d)
   * @returns Total number of polynomial features
   *
   * @remarks
   * Uses multiplicative formula to avoid factorial overflow.
   *
   * Time complexity: O(min(n, d))
   * Space complexity: O(1)
   *
   * @example
   * ```typescript
   * const gen = new PolynomialFeatureGenerator();
   * gen.getFeatureCount(2, 2); // Returns 6: [1, x1, x2, x1², x1x2, x2²]
   * gen.getFeatureCount(3, 2); // Returns 10
   * ```
   */
  public getFeatureCount(inputDim: number, degree: number): number {
    // Binomial coefficient C(n + d, d) using multiplicative formula
    // C(n+d, d) = Π(i=0 to d-1) [(n+d-i) / (i+1)]
    let result = 1;
    for (let i = 0; i < degree; i++) {
      result = result * (inputDim + degree - i) / (i + 1);
    }
    return Math.round(result);
  }

  /**
   * Initializes the generator for given input dimension and degree.
   *
   * @param inputDim - Number of input dimensions
   * @param degree - Polynomial degree
   *
   * @remarks
   * Precomputes all feature index patterns for efficient generation.
   * Skips reinitialization if called with same parameters.
   *
   * Time complexity: O(featureCount × degree)
   * Space complexity: O(featureCount × degree) for cached indices
   */
  public initialize(inputDim: number, degree: number): void {
    // Skip if already initialized with same parameters
    if (
      this.initialized && this.inputDim === inputDim && this.degree === degree
    ) {
      return;
    }

    this.inputDim = inputDim;
    this.degree = degree;
    this.cachedFeatureCount = this.getFeatureCount(inputDim, degree);
    this.featureIndices = [];

    // Generate all monomial combinations using recursive enumeration
    this.generateCombinations(inputDim, degree);
    this.initialized = true;
  }

  /**
   * Recursively generates all polynomial term combinations.
   * Uses a stars-and-bars style enumeration.
   *
   * @param inputDim - Number of input dimensions
   * @param maxDegree - Maximum total degree
   *
   * @internal
   */
  private generateCombinations(inputDim: number, maxDegree: number): void {
    const tempIndices: number[] = [];

    /**
     * Recursive helper to generate combinations.
     * @param startIdx - Minimum variable index to use (prevents duplicates)
     * @param remainingDegree - Remaining degree budget
     */
    const generateRecursive = (
      startIdx: number,
      remainingDegree: number,
    ): void => {
      // Add current combination as Int32Array for efficient storage
      this.featureIndices.push(new Int32Array(tempIndices));

      // Generate higher degree terms if budget remains
      if (remainingDegree > 0) {
        for (let i = startIdx; i < inputDim; i++) {
          tempIndices.push(i);
          generateRecursive(i, remainingDegree - 1);
          tempIndices.pop();
        }
      }
    };

    generateRecursive(0, maxDegree);
  }

  /**
   * Generates polynomial features in-place from input vector.
   *
   * @param output - Output array for features (must be preallocated with correct size)
   * @param input - Input vector
   * @param degree - Polynomial degree (must match initialized degree)
   *
   * @throws {Error} If generator not initialized or dimension mismatch
   *
   * @remarks
   * Each feature is computed as the product of input values at the indices
   * stored in featureIndices. An empty index array produces 1 (constant term).
   *
   * Time complexity: O(featureCount × averageTermDegree)
   * Space complexity: O(1) - no allocations
   *
   * @example
   * ```typescript
   * const features = new Float64Array(generator.getFeatureCount(2, 2));
   * generator.generateFeaturesInPlace(features, input, 2);
   * ```
   */
  public generateFeaturesInPlace(
    output: Float64Array,
    input: Float64Array,
    degree: number,
  ): void {
    if (!this.initialized || degree !== this.degree) {
      throw new Error("PolynomialFeatureGenerator not properly initialized");
    }
    if (input.length !== this.inputDim) {
      throw new Error(
        `Input dimension mismatch: expected ${this.inputDim}, got ${input.length}`,
      );
    }

    const numFeatures = this.featureIndices.length;
    for (let f = 0; f < numFeatures; f++) {
      const indices = this.featureIndices[f];
      let value = 1.0;

      // Multiply together all input values at specified indices
      // e.g., indices = [0, 1] computes x₁ × x₂
      const termLength = indices.length;
      for (let t = 0; t < termLength; t++) {
        value *= input[indices[t]];
      }

      output[f] = value;
    }
  }

  /**
   * Checks if the generator has been initialized.
   *
   * @returns true if initialized, false otherwise
   */
  public isInitialized(): boolean {
    return this.initialized;
  }

  /**
   * Resets the generator to uninitialized state.
   */
  public reset(): void {
    this.featureIndices = [];
    this.cachedFeatureCount = 0;
    this.inputDim = 0;
    this.degree = 0;
    this.initialized = false;
  }
}

// =============================================================================
// COVARIANCE MANAGER
// =============================================================================

/**
 * Manages the covariance matrix P for the Recursive Least Squares algorithm.
 *
 * @remarks
 * The covariance matrix represents the uncertainty in the weight estimates.
 * It is initialized as P = initialValue × I (scaled identity matrix) and
 * updated with each new sample using:
 *
 * 1. Gain vector: k = P·φ / (λ + φᵀ·P·φ)
 * 2. Covariance update: P = (P - k·φᵀ·P) / λ
 *
 * Where:
 * - φ is the feature vector
 * - λ is the forgetting factor (0 < λ ≤ 1)
 * - k is the Kalman gain vector
 *
 * The matrix is stored in row-major order for efficient access.
 *
 * @example
 * ```typescript
 * const matOps = new MatrixOperations();
 * const covManager = new CovarianceManager(matOps);
 * covManager.initialize(6, 1000); // 6 features, initial covariance 1000
 *
 * const gain = new Float64Array(6);
 * covManager.computeGainVector(features, 0.99, gain);
 * covManager.updateCovariance(gain, features, 0.99);
 * ```
 *
 * @implements {ICovarianceManager}
 */
class CovarianceManager implements ICovarianceManager {
  /** Matrix operations helper */
  private readonly matrixOps: IMatrixOperations;
  /** Current covariance matrix P (featureCount × featureCount) */
  private covarianceMatrix: Float64Array | null = null;
  /** Number of features (matrix dimension) */
  private featureCount: number = 0;
  /** Initial diagonal value for reset */
  private initialValue: number = 1000;
  /** Temporary buffer for P·φ computation */
  private tempPphi: Float64Array | null = null;
  /** Temporary buffer for φᵀP computation */
  private tempPhiP: Float64Array | null = null;

  /**
   * Creates a new CovarianceManager with injected matrix operations.
   *
   * @param matrixOps - Matrix operations implementation for dependency injection
   */
  constructor(matrixOps: IMatrixOperations) {
    this.matrixOps = matrixOps;
  }

  /**
   * Initializes the covariance matrix as a scaled identity matrix.
   *
   * P = initialValue × I
   *
   * @param featureCount - Number of polynomial features
   * @param initialValue - Initial diagonal value (larger = more uncertain)
   *
   * @remarks
   * Time complexity: O(featureCount²) for matrix creation and initialization
   * Space complexity: O(featureCount²) for covariance matrix storage
   */
  public initialize(featureCount: number, initialValue: number): void {
    this.featureCount = featureCount;
    this.initialValue = initialValue;

    // Allocate covariance matrix (initialized to zeros)
    this.covarianceMatrix = this.matrixOps.create(featureCount, featureCount);

    // Set diagonal elements to initialValue (identity × initialValue)
    for (let i = 0; i < featureCount; i++) {
      this.matrixOps.set(
        this.covarianceMatrix,
        featureCount,
        i,
        i,
        initialValue,
      );
    }

    // Allocate temporary buffers
    this.tempPphi = new Float64Array(featureCount);
    this.tempPhiP = new Float64Array(featureCount);
  }

  /**
   * Gets the current covariance matrix.
   *
   * @returns The covariance matrix as Float64Array in row-major order
   * @throws {Error} If manager not initialized
   */
  public getCovarianceMatrix(): Float64Array {
    if (this.covarianceMatrix === null) {
      throw new Error("CovarianceManager not initialized");
    }
    return this.covarianceMatrix;
  }

  /**
   * Computes the gain vector for RLS update.
   *
   * Formula: k = P·φ / (λ + φᵀ·P·φ)
   *
   * @param features - Polynomial feature vector φ
   * @param forgettingFactor - Forgetting factor λ ∈ (0, 1]
   * @param gainOut - Output array for gain vector k (must be preallocated)
   * @returns The denominator (λ + φᵀ·P·φ) for debugging/monitoring
   *
   * @remarks
   * The gain vector determines how much the weights should be adjusted
   * for the current prediction error. It balances:
   * - Current estimate uncertainty (from P)
   * - Feature relevance (from φ)
   * - Forgetting factor (λ, controls adaptation rate)
   *
   * Time complexity: O(featureCount²)
   * Space complexity: O(1) - uses preallocated buffers
   */
  public computeGainVector(
    features: Float64Array,
    forgettingFactor: number,
    gainOut: Float64Array,
  ): number {
    if (this.covarianceMatrix === null || this.tempPphi === null) {
      throw new Error("CovarianceManager not initialized");
    }

    const n = this.featureCount;

    // Step 1: Compute P·φ (matrix-vector multiplication)
    this.matrixOps.matVecMulInPlace(
      this.tempPphi,
      this.covarianceMatrix,
      n,
      n,
      features,
    );

    // Step 2: Compute φᵀ·P·φ (dot product of features with P·φ)
    const phiTPphi = this.matrixOps.dot(features, this.tempPphi);

    // Step 3: Compute denominator: λ + φᵀ·P·φ
    const denom = forgettingFactor + phiTPphi;

    // Step 4: Compute gain: k = (P·φ) / denom
    const invDenom = 1.0 / denom;
    for (let i = 0; i < n; i++) {
      gainOut[i] = this.tempPphi[i] * invDenom;
    }

    return denom;
  }

  /**
   * Updates the covariance matrix using the RLS update rule.
   *
   * Formula: P = (P - k·φᵀ·P) / λ
   *
   * This can be rewritten as:
   * P_new = (P - k ⊗ (φᵀ·P)) / λ
   * where ⊗ denotes outer product
   *
   * @param gain - Gain vector k from computeGainVector
   * @param features - Feature vector φ
   * @param forgettingFactor - Forgetting factor λ
   *
   * @remarks
   * The covariance update reduces uncertainty in directions where
   * we've observed data, while the 1/λ factor slightly increases
   * uncertainty over time (for λ < 1), allowing adaptation to changes.
   *
   * Time complexity: O(featureCount²)
   * Space complexity: O(1) - in-place update
   */
  public updateCovariance(
    gain: Float64Array,
    features: Float64Array,
    forgettingFactor: number,
  ): void {
    if (this.covarianceMatrix === null || this.tempPhiP === null) {
      throw new Error("CovarianceManager not initialized");
    }

    const n = this.featureCount;

    // Step 1: Compute φᵀ·P (row vector × matrix = row vector)
    // (φᵀ·P)_j = Σᵢ features[i] × P[i,j]
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let i = 0; i < n; i++) {
        sum += features[i] * this.matrixOps.get(this.covarianceMatrix, n, i, j);
      }
      this.tempPhiP[j] = sum;
    }

    // Step 2: Update P in-place: P = (P - k ⊗ φᵀP) / λ
    // P[i,j] = (P[i,j] - gain[i] × tempPhiP[j]) / λ
    const invLambda = 1.0 / forgettingFactor;
    for (let i = 0; i < n; i++) {
      const rowOffset = i * n;
      const gi = gain[i];
      for (let j = 0; j < n; j++) {
        this.covarianceMatrix[rowOffset + j] =
          (this.covarianceMatrix[rowOffset + j] - gi * this.tempPhiP[j]) *
          invLambda;
      }
    }
  }

  /**
   * Applies regularization to the diagonal of P for numerical stability.
   *
   * P[i,i] += regularization
   *
   * @param regularization - Value to add to diagonal elements
   *
   * @remarks
   * Prevents the covariance matrix from becoming singular or
   * having very small eigenvalues, which would cause numerical issues.
   * Should be called periodically during training.
   *
   * Time complexity: O(featureCount)
   * Space complexity: O(1)
   */
  public applyRegularization(regularization: number): void {
    if (this.covarianceMatrix === null) {
      throw new Error("CovarianceManager not initialized");
    }
    this.matrixOps.addDiagonalRegularization(
      this.covarianceMatrix,
      this.featureCount,
      regularization,
    );
  }

  /**
   * Resets the covariance matrix to initial scaled identity state.
   */
  public reset(): void {
    if (this.covarianceMatrix !== null) {
      const n = this.featureCount;
      // Clear entire matrix
      this.covarianceMatrix.fill(0);
      // Reinitialize diagonal
      for (let i = 0; i < n; i++) {
        this.covarianceMatrix[i * n + i] = this.initialValue;
      }
    }
  }
}

// =============================================================================
// WEIGHT MANAGER
// =============================================================================

/**
 * Manages the weight matrix W for the RLS algorithm.
 *
 * @remarks
 * The weight matrix maps polynomial features to output dimensions.
 * Shape: (featureCount × outputDim)
 *
 * Prediction: y = Wᵀ·φ
 * Update: W = W + k·eᵀ (where k is gain, e is error)
 *
 * @example
 * ```typescript
 * const weightManager = new WeightManager();
 * weightManager.initialize(6, 2); // 6 features, 2 output dimensions
 *
 * // Make prediction
 * const prediction = new Float64Array(2);
 * weightManager.predict(features, prediction);
 *
 * // Update weights
 * weightManager.updateWeights(gain, error);
 * ```
 *
 * @implements {IWeightManager}
 */
class WeightManager implements IWeightManager {
  /** Weight matrix (featureCount × outputDim) in row-major order */
  private weights: Float64Array | null = null;
  /** Number of polynomial features */
  private featureCount: number = 0;
  /** Number of output dimensions */
  private outputDim: number = 0;

  /**
   * Initializes the weight matrix to zeros.
   *
   * @param featureCount - Number of polynomial features (rows)
   * @param outputDim - Number of output dimensions (columns)
   *
   * @remarks
   * Time complexity: O(featureCount × outputDim)
   * Space complexity: O(featureCount × outputDim)
   */
  public initialize(featureCount: number, outputDim: number): void {
    this.featureCount = featureCount;
    this.outputDim = outputDim;
    this.weights = new Float64Array(featureCount * outputDim);
  }

  /**
   * Gets the current weight matrix as a flat Float64Array.
   *
   * @returns Weight matrix in row-major order
   * @throws {Error} If not initialized
   */
  public getWeights(): Float64Array {
    if (this.weights === null) {
      throw new Error("WeightManager not initialized");
    }
    return this.weights;
  }

  /**
   * Updates weights using the RLS update rule.
   *
   * Formula: W = W + k·eᵀ
   *
   * For each output dimension j:
   *   W[:, j] = W[:, j] + k × error[j]
   *
   * @param gain - Gain vector k (length = featureCount)
   * @param error - Prediction error e (length = outputDim)
   *
   * @remarks
   * The update is essentially a rank-1 update to the weight matrix,
   * adjusting weights proportionally to both the gain and the error.
   *
   * Time complexity: O(featureCount × outputDim)
   * Space complexity: O(1) - in-place update
   */
  public updateWeights(gain: Float64Array, error: Float64Array): void {
    if (this.weights === null) {
      throw new Error("WeightManager not initialized");
    }

    // W[i, j] += gain[i] × error[j]
    for (let i = 0; i < this.featureCount; i++) {
      const rowOffset = i * this.outputDim;
      const gi = gain[i];
      for (let j = 0; j < this.outputDim; j++) {
        this.weights[rowOffset + j] += gi * error[j];
      }
    }
  }

  /**
   * Computes prediction using current weights.
   *
   * Formula: y = Wᵀ·φ = φᵀ·W (equivalent for prediction)
   *
   * @param features - Polynomial feature vector φ (length = featureCount)
   * @param output - Output array for predictions (length = outputDim)
   *
   * @remarks
   * For each output j: y[j] = Σᵢ features[i] × W[i, j]
   *
   * Time complexity: O(featureCount × outputDim)
   * Space complexity: O(1) - uses preallocated output
   */
  public predict(features: Float64Array, output: Float64Array): void {
    if (this.weights === null) {
      throw new Error("WeightManager not initialized");
    }

    // For each output dimension
    for (let j = 0; j < this.outputDim; j++) {
      let sum = 0;
      for (let i = 0; i < this.featureCount; i++) {
        sum += features[i] * this.weights[i * this.outputDim + j];
      }
      output[j] = sum;
    }
  }

  /**
   * Resets all weights to zero.
   */
  public reset(): void {
    if (this.weights !== null) {
      this.weights.fill(0);
    }
  }

  /**
   * Gets the weight matrix as a 2D JavaScript array.
   *
   * @returns Weight matrix as number[][] (featureCount × outputDim)
   *
   * @remarks
   * Creates new arrays - use for external inspection only, not in hot paths.
   *
   * Time complexity: O(featureCount × outputDim)
   * Space complexity: O(featureCount × outputDim) for new arrays
   */
  public getWeightsAs2D(): number[][] {
    if (this.weights === null) {
      return [];
    }

    const result: number[][] = [];
    for (let i = 0; i < this.featureCount; i++) {
      const row: number[] = [];
      const rowOffset = i * this.outputDim;
      for (let j = 0; j < this.outputDim; j++) {
        row.push(this.weights[rowOffset + j]);
      }
      result.push(row);
    }
    return result;
  }
}

// =============================================================================
// PREDICTION ENGINE
// =============================================================================

/**
 * Handles prediction with confidence intervals using the covariance matrix.
 *
 * @remarks
 * Calculates:
 * - Point predictions: y = Wᵀ·φ
 * - Standard errors: SE = √(φᵀ·P·φ)
 * - Confidence intervals: y ± t_α × SE
 *
 * Uses t-distribution for small samples (n ≤ 30) and z-distribution
 * for large samples, with pre-computed critical values for efficiency.
 *
 * @example
 * ```typescript
 * const engine = new PredictionEngine(matrixOps);
 * const result = {
 *   predicted: [0], lowerBound: [0], upperBound: [0], standardError: [0]
 * };
 * engine.predictWithConfidence(features, covariance, weights, 6, 1, 0.95, 100, result);
 * ```
 *
 * @implements {IPredictionEngine}
 */
class PredictionEngine implements IPredictionEngine {
  /** Matrix operations helper */
  private readonly matrixOps: IMatrixOperations;
  /** Temporary buffer for P·φ computation */
  private tempPphi: Float64Array | null = null;

  /**
   * Pre-computed t-distribution critical values for 95% confidence.
   * Index corresponds to degrees of freedom minus 1 (df 1-30).
   */
  private static readonly T_VALUES_95: readonly number[] = [
    12.706,
    4.303,
    3.182,
    2.776,
    2.571,
    2.447,
    2.365,
    2.306,
    2.262,
    2.228,
    2.201,
    2.179,
    2.160,
    2.145,
    2.131,
    2.120,
    2.110,
    2.101,
    2.093,
    2.086,
    2.080,
    2.074,
    2.069,
    2.064,
    2.060,
    2.056,
    2.052,
    2.048,
    2.045,
    2.042,
  ];

  /**
   * Pre-computed t-distribution critical values for 99% confidence.
   * Index corresponds to degrees of freedom minus 1 (df 1-30).
   */
  private static readonly T_VALUES_99: readonly number[] = [
    63.657,
    9.925,
    5.841,
    4.604,
    4.032,
    3.707,
    3.499,
    3.355,
    3.250,
    3.169,
    3.106,
    3.055,
    3.012,
    2.977,
    2.947,
    2.921,
    2.898,
    2.878,
    2.861,
    2.845,
    2.831,
    2.819,
    2.807,
    2.797,
    2.787,
    2.779,
    2.771,
    2.763,
    2.756,
    2.750,
  ];

  /** Z-value for 95% confidence (large samples) */
  private static readonly Z_VALUE_95: number = 1.96;
  /** Z-value for 99% confidence (large samples) */
  private static readonly Z_VALUE_99: number = 2.576;

  /**
   * Creates a new PredictionEngine with injected matrix operations.
   *
   * @param matrixOps - Matrix operations implementation
   */
  constructor(matrixOps: IMatrixOperations) {
    this.matrixOps = matrixOps;
  }

  /**
   * Makes prediction with confidence intervals.
   *
   * @param features - Polynomial feature vector φ
   * @param covariance - Covariance matrix P
   * @param weights - Weight matrix W
   * @param featureCount - Number of features
   * @param outputDim - Number of output dimensions
   * @param confidenceLevel - Confidence level (e.g., 0.95 for 95%)
   * @param sampleCount - Total training samples (for t-distribution df)
   * @param result - Output object to populate with results
   *
   * @remarks
   * Standard error formula: SE = √(φᵀ·P·φ)
   * This represents the uncertainty in the prediction based on
   * the model's current uncertainty (covariance) and the features.
   *
   * Time complexity: O(featureCount² + featureCount × outputDim)
   * Space complexity: O(featureCount) for temporary buffer
   */
  public predictWithConfidence(
    features: Float64Array,
    covariance: Float64Array,
    weights: Float64Array,
    featureCount: number,
    outputDim: number,
    confidenceLevel: number,
    sampleCount: number,
    result: SinglePrediction,
  ): void {
    // Ensure temporary buffer is allocated
    if (this.tempPphi === null || this.tempPphi.length < featureCount) {
      this.tempPphi = new Float64Array(featureCount);
    }

    // Step 1: Compute P·φ
    this.matrixOps.matVecMulInPlace(
      this.tempPphi,
      covariance,
      featureCount,
      featureCount,
      features,
    );

    // Step 2: Compute variance = φᵀ·P·φ and standard error = √variance
    const variance = this.matrixOps.dot(features, this.tempPphi);
    const standardError = Math.sqrt(Math.max(0, variance));

    // Step 3: Get critical value based on sample size and confidence level
    const degreesOfFreedom = Math.max(1, sampleCount - featureCount);
    const criticalValue = this.getCriticalValue(
      confidenceLevel,
      degreesOfFreedom,
    );

    // Step 4: Compute predictions and confidence intervals for each output
    for (let j = 0; j < outputDim; j++) {
      // Compute point prediction: y[j] = Σᵢ features[i] × W[i, j]
      let prediction = 0;
      for (let i = 0; i < featureCount; i++) {
        prediction += features[i] * weights[i * outputDim + j];
      }

      result.predicted[j] = prediction;
      result.standardError[j] = standardError;
      result.lowerBound[j] = prediction - criticalValue * standardError;
      result.upperBound[j] = prediction + criticalValue * standardError;
    }
  }

  /**
   * Gets the critical value for confidence interval calculation.
   *
   * @param confidenceLevel - Confidence level (0.95 or 0.99 primarily supported)
   * @param degreesOfFreedom - n - p (sample size minus feature count)
   * @returns Critical value for the given confidence level
   *
   * @remarks
   * Uses t-distribution for df ≤ 30, z-distribution for larger samples.
   * Interpolates between 95% and 99% for other confidence levels.
   *
   * Time complexity: O(1)
   * Space complexity: O(1)
   */
  private getCriticalValue(
    confidenceLevel: number,
    degreesOfFreedom: number,
  ): number {
    // For large samples (df > 30), use z-distribution
    if (degreesOfFreedom > 30) {
      if (confidenceLevel >= 0.99) {
        return PredictionEngine.Z_VALUE_99;
      } else if (confidenceLevel >= 0.95) {
        // Linear interpolation between 95% and 99%
        const t = (confidenceLevel - 0.95) / 0.04;
        return PredictionEngine.Z_VALUE_95 +
          t * (PredictionEngine.Z_VALUE_99 - PredictionEngine.Z_VALUE_95);
      } else {
        // For lower confidence, extrapolate down from 95%
        return PredictionEngine.Z_VALUE_95 * (confidenceLevel / 0.95);
      }
    }

    // For small samples, use t-distribution lookup
    const df = Math.max(1, Math.min(30, Math.floor(degreesOfFreedom)));
    const t95 = PredictionEngine.T_VALUES_95[df - 1];
    const t99 = PredictionEngine.T_VALUES_99[df - 1];

    if (confidenceLevel >= 0.99) {
      return t99;
    } else if (confidenceLevel >= 0.95) {
      // Linear interpolation
      const t = (confidenceLevel - 0.95) / 0.04;
      return t95 + t * (t99 - t95);
    } else {
      // Extrapolate for lower confidence
      return t95 * (confidenceLevel / 0.95);
    }
  }
}

// =============================================================================
// STATISTICS TRACKER
// =============================================================================

/**
 * Tracks running statistics for model evaluation metrics.
 *
 * @remarks
 * Computes incrementally without storing all historical data:
 * - R-squared (coefficient of determination): R² = 1 - (SS_res / SS_tot)
 * - RMSE (root mean squared error): RMSE = √(SS_res / n)
 *
 * Uses Welford's online algorithm for numerically stable variance computation.
 *
 * @example
 * ```typescript
 * const tracker = new StatisticsTracker();
 * tracker.update(actualValues, predictedValues);
 * console.log(`R² = ${tracker.getRSquared()}`);
 * console.log(`RMSE = ${tracker.getRMSE()}`);
 * ```
 *
 * @implements {IStatisticsTracker}
 */
class StatisticsTracker implements IStatisticsTracker {
  /** Total number of individual values processed */
  private sampleCount: number = 0;
  /** Sum of squared residuals: Σ(yᵢ - ŷᵢ)² */
  private sumSquaredResiduals: number = 0;
  /** Running mean of actual values (for total variance) */
  private meanY: number = 0;
  /** M2 accumulator for Welford's algorithm */
  private m2: number = 0;

  /**
   * Updates statistics with new actual vs predicted values.
   *
   * @param actual - Actual output values (Float64Array)
   * @param predicted - Model predictions (Float64Array, same length)
   *
   * @remarks
   * Processes all output dimensions, treating each as a separate observation.
   * Uses Welford's algorithm for stable variance computation.
   *
   * Time complexity: O(outputDim)
   * Space complexity: O(1)
   */
  public update(actual: Float64Array, predicted: Float64Array): void {
    const len = actual.length;

    for (let i = 0; i < len; i++) {
      this.sampleCount++;
      const y = actual[i];
      const yHat = predicted[i];

      // Update residual sum of squares: SS_res
      const residual = y - yHat;
      this.sumSquaredResiduals += residual * residual;

      // Update total sum of squares using Welford's algorithm
      // This computes Σ(yᵢ - mean(y))² incrementally
      const delta = y - this.meanY;
      this.meanY += delta / this.sampleCount;
      const delta2 = y - this.meanY;
      this.m2 += delta * delta2;
    }
  }

  /**
   * Gets the current R-squared (coefficient of determination) value.
   *
   * Formula: R² = 1 - (SS_res / SS_tot)
   *
   * @returns R-squared value in range [0, 1], or 0 if insufficient data
   *
   * @remarks
   * - R² = 1: Perfect fit (all predictions match actual values)
   * - R² = 0: Model is no better than predicting the mean
   * - R² can technically be negative for very bad fits, but is clamped to [0, 1]
   *
   * Time complexity: O(1)
   * Space complexity: O(1)
   */
  public getRSquared(): number {
    // Need at least 2 samples for meaningful R²
    if (this.sampleCount < 2) {
      return 0;
    }

    // SS_tot = m2 (total variance × (n-1))
    const sumSquaredTotal = this.m2;

    // Avoid division by zero (constant target variable)
    if (sumSquaredTotal < 1e-10) {
      return this.sumSquaredResiduals < 1e-10 ? 1 : 0;
    }

    const rSquared = 1 - (this.sumSquaredResiduals / sumSquaredTotal);
    // Clamp to valid range (can be negative for very bad fits)
    return Math.max(0, Math.min(1, rSquared));
  }

  /**
   * Gets the current Root Mean Squared Error.
   *
   * Formula: RMSE = √(SS_res / n)
   *
   * @returns RMSE value (≥ 0), or 0 if no data
   *
   * @remarks
   * RMSE is in the same units as the target variable.
   * Lower is better.
   *
   * Time complexity: O(1)
   * Space complexity: O(1)
   */
  public getRMSE(): number {
    if (this.sampleCount === 0) {
      return 0;
    }
    return Math.sqrt(this.sumSquaredResiduals / this.sampleCount);
  }

  /**
   * Gets the total number of values processed.
   *
   * @returns Sample count
   */
  public getSampleCount(): number {
    return this.sampleCount;
  }

  /**
   * Resets all statistics to initial state.
   */
  public reset(): void {
    this.sampleCount = 0;
    this.sumSquaredResiduals = 0;
    this.meanY = 0;
    this.m2 = 0;
  }
}

// =============================================================================
// CONFIGURATION BUILDER
// =============================================================================

/**
 * Builder class for MultivariatePolynomialRegression configuration.
 *
 * @remarks
 * Implements the Builder pattern for fluent, type-safe configuration.
 * All parameters are validated during building.
 *
 * @example
 * ```typescript
 * const config = new ConfigurationBuilder()
 *   .setPolynomialDegree(3)
 *   .setForgettingFactor(0.95)
 *   .setNormalizationMethod('z-score')
 *   .setConfidenceLevel(0.99)
 *   .build();
 *
 * const model = new MultivariatePolynomialRegression(config);
 * ```
 */
class ConfigurationBuilder {
  /** Configuration being built */
  private config: IMultivariatePolynomialRegressionConfig;

  /**
   * Creates a new ConfigurationBuilder with default values.
   *
   * Default configuration:
   * - polynomialDegree: 2
   * - enableNormalization: true
   * - normalizationMethod: 'min-max'
   * - forgettingFactor: 0.99
   * - initialCovariance: 1000
   * - regularization: 1e-6
   * - confidenceLevel: 0.95
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
   * Sets the polynomial degree for feature expansion.
   *
   * @param degree - Polynomial degree (minimum: 1)
   * @returns This builder for method chaining
   * @throws {Error} If degree < 1
   *
   * @example
   * ```typescript
   * builder.setPolynomialDegree(3); // Cubic features
   * ```
   */
  public setPolynomialDegree(degree: number): ConfigurationBuilder {
    if (degree < 1) {
      throw new Error("Polynomial degree must be at least 1");
    }
    if (!Number.isInteger(degree)) {
      throw new Error("Polynomial degree must be an integer");
    }
    this.config.polynomialDegree = degree;
    return this;
  }

  /**
   * Enables or disables input normalization.
   *
   * @param enable - Whether to enable normalization
   * @returns This builder for method chaining
   */
  public setEnableNormalization(enable: boolean): ConfigurationBuilder {
    this.config.enableNormalization = enable;
    return this;
  }

  /**
   * Sets the normalization method.
   *
   * @param method - 'none', 'min-max', or 'z-score'
   * @returns This builder for method chaining
   *
   * @remarks
   * Setting method to 'none' automatically disables normalization.
   */
  public setNormalizationMethod(
    method: "none" | "min-max" | "z-score",
  ): ConfigurationBuilder {
    this.config.normalizationMethod = method;
    if (method === "none") {
      this.config.enableNormalization = false;
    }
    return this;
  }

  /**
   * Sets the forgetting factor λ for RLS.
   *
   * @param factor - Forgetting factor in range (0, 1]
   * @returns This builder for method chaining
   * @throws {Error} If factor not in valid range
   *
   * @remarks
   * - λ = 1: No forgetting (standard recursive least squares)
   * - λ < 1: Exponentially down-weights older samples (adaptive RLS)
   * - Common values: 0.95 to 0.99
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
   *
   * @param value - Initial covariance (positive)
   * @returns This builder for method chaining
   * @throws {Error} If value not positive
   *
   * @remarks
   * Larger values indicate more initial uncertainty, leading to
   * faster initial learning but potentially more initial noise.
   */
  public setInitialCovariance(value: number): ConfigurationBuilder {
    if (value <= 0) {
      throw new Error("Initial covariance must be positive");
    }
    this.config.initialCovariance = value;
    return this;
  }

  /**
   * Sets the regularization parameter for numerical stability.
   *
   * @param value - Regularization value (non-negative)
   * @returns This builder for method chaining
   * @throws {Error} If value negative
   *
   * @remarks
   * Small positive values (e.g., 1e-6) prevent singular covariance matrices.
   */
  public setRegularization(value: number): ConfigurationBuilder {
    if (value < 0) {
      throw new Error("Regularization must be non-negative");
    }
    this.config.regularization = value;
    return this;
  }

  /**
   * Sets the confidence level for prediction intervals.
   *
   * @param level - Confidence level in range (0, 1)
   * @returns This builder for method chaining
   * @throws {Error} If level not in valid range
   *
   * @example
   * ```typescript
   * builder.setConfidenceLevel(0.95); // 95% confidence intervals
   * builder.setConfidenceLevel(0.99); // 99% confidence intervals
   * ```
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
   * @returns A copy of the configuration (to prevent modification)
   */
  public build(): IMultivariatePolynomialRegressionConfig {
    return { ...this.config };
  }
}

// =============================================================================
// MAIN CLASS: MultivariatePolynomialRegression
// =============================================================================

/**
 * Multivariate Polynomial Regression with Recursive Least Squares (RLS) online learning.
 *
 * @remarks
 * This class implements incremental online learning using the RLS algorithm,
 * allowing the model to be updated efficiently with new data points without
 * retraining on the entire dataset.
 *
 * ## RLS Algorithm Overview
 *
 * For each new sample (x, y):
 * 1. **Normalize**: x_norm = normalize(x)
 * 2. **Feature Generation**: φ = polynomial_features(x_norm)
 * 3. **Gain Computation**: k = P·φ / (λ + φᵀ·P·φ)
 * 4. **Error Computation**: e = y - Wᵀ·φ
 * 5. **Weight Update**: W = W + k·eᵀ
 * 6. **Covariance Update**: P = (P - k·φᵀ·P) / λ
 * 7. **Regularization**: P[i,i] += ε (periodically)
 *
 * Where:
 * - P: Covariance matrix (uncertainty in weights)
 * - W: Weight matrix
 * - φ: Polynomial feature vector
 * - λ: Forgetting factor (0 < λ ≤ 1)
 * - k: Kalman gain vector
 * - ε: Regularization parameter
 *
 * ## Key Features
 *
 * - **Online Learning**: Update model incrementally without batch retraining
 * - **Polynomial Features**: Automatic expansion up to specified degree
 * - **Configurable Normalization**: Min-max or z-score scaling
 * - **Confidence Intervals**: Statistical intervals for predictions
 * - **Running Metrics**: R² and RMSE computed incrementally
 * - **Memory Efficient**: Uses Float64Array and preallocated buffers
 *
 * ## Computational Complexity
 *
 * Let:
 * - n = input dimension
 * - d = polynomial degree
 * - p = number of polynomial features = C(n+d, d)
 * - m = output dimension
 * - N = number of samples
 *
 * - fitOnline: O(N × p²) time, O(p²) space
 * - predict: O(p² + p × m) time per prediction
 *
 * @example
 * ```typescript
 * // Create model with custom configuration
 * const model = new MultivariatePolynomialRegression({
 *   polynomialDegree: 2,
 *   forgettingFactor: 0.99,
 *   normalizationMethod: 'z-score',
 *   confidenceLevel: 0.95
 * });
 *
 * // Train incrementally with batches
 * model.fitOnline({
 *   xCoordinates: [[1, 2], [2, 3], [3, 4]],
 *   yCoordinates: [[5], [8], [13]]
 * });
 *
 * // Add more data later
 * model.fitOnline({
 *   xCoordinates: [[4, 5]],
 *   yCoordinates: [[20]]
 * });
 *
 * // Make predictions with confidence intervals
 * const result = model.predict({
 *   inputPoints: [[5, 6], [6, 7]]
 * });
 *
 * result.predictions.forEach((pred, i) => {
 *   console.log(`Point ${i}: ${pred.predicted[0]} ± ${pred.standardError[0]}`);
 *   console.log(`  95% CI: [${pred.lowerBound[0]}, ${pred.upperBound[0]}]`);
 * });
 *
 * // Get model summary
 * const summary = model.getModelSummary();
 * console.log(`R² = ${summary.rSquared.toFixed(4)}`);
 * console.log(`RMSE = ${summary.rmse.toFixed(4)}`);
 * ```
 */
class MultivariatePolynomialRegression {
  // ===========================================================================
  // PRIVATE FIELDS
  // ===========================================================================

  /** Model configuration (immutable after construction) */
  private readonly config: IMultivariatePolynomialRegressionConfig;

  /** Matrix operations helper (dependency injected) */
  private readonly matrixOps: MatrixOperations;

  /** Polynomial feature generator */
  private readonly featureGenerator: PolynomialFeatureGenerator;

  /** Covariance matrix manager for RLS */
  private readonly covarianceManager: CovarianceManager;

  /** Weight matrix manager */
  private readonly weightManager: WeightManager;

  /** Prediction engine with confidence intervals */
  private readonly predictionEngine: PredictionEngine;

  /** Statistics tracker for R² and RMSE */
  private readonly statisticsTracker: StatisticsTracker;

  /** Current normalization strategy (Strategy pattern) */
  private normalizationStrategy: INormalizationStrategy;

  /** Current normalization statistics */
  private normalizationStats: NormalizationStats;

  /** Whether model has been initialized with first sample */
  private initialized: boolean = false;

  /** Input dimension (number of features per sample) */
  private inputDim: number = 0;

  /** Output dimension (number of targets per sample) */
  private outputDim: number = 0;

  /** Number of polynomial features */
  private featureCount: number = 0;

  /** Total number of training samples processed */
  private sampleCount: number = 0;

  // ===========================================================================
  // PREALLOCATED BUFFERS (for performance optimization)
  // ===========================================================================

  /** Buffer for normalized input values */
  private normalizedInput: Float64Array | null = null;

  /** Buffer for polynomial features */
  private featureBuffer: Float64Array | null = null;

  /** Buffer for RLS gain vector */
  private gainBuffer: Float64Array | null = null;

  /** Buffer for prediction error */
  private errorBuffer: Float64Array | null = null;

  /** Buffer for prediction output */
  private predictionBuffer: Float64Array | null = null;

  /** Buffer for target values (avoids allocation in hot path) */
  private targetBuffer: Float64Array | null = null;

  /** Last input point seen (for extrapolation) */
  private lastInput: Float64Array | null = null;

  /** Object pool for SinglePrediction results (reduces GC pressure) */
  private predictionPool: SinglePrediction[] = [];

  /** Counter for periodic regularization application */
  private regularizationCounter: number = 0;

  /** How often to apply regularization (every N samples) */
  private static readonly REGULARIZATION_FREQUENCY: number = 10;

  // ===========================================================================
  // CONSTRUCTOR
  // ===========================================================================

  /**
   * Creates a new MultivariatePolynomialRegression model.
   *
   * @param config - Partial configuration (missing values use defaults)
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
   *   normalizationMethod: 'z-score'
   * });
   *
   * // Using builder
   * const config = new ConfigurationBuilder()
   *   .setPolynomialDegree(3)
   *   .setConfidenceLevel(0.99)
   *   .build();
   * const model3 = new MultivariatePolynomialRegression(config);
   * ```
   */
  constructor(config?: Partial<IMultivariatePolynomialRegressionConfig>) {
    // Build configuration with defaults and validation
    const builder = new ConfigurationBuilder();
    if (config) {
      if (config.polynomialDegree !== undefined) {
        builder.setPolynomialDegree(config.polynomialDegree);
      }
      if (config.enableNormalization !== undefined) {
        builder.setEnableNormalization(config.enableNormalization);
      }
      if (config.normalizationMethod !== undefined) {
        builder.setNormalizationMethod(config.normalizationMethod);
      }
      if (config.forgettingFactor !== undefined) {
        builder.setForgettingFactor(config.forgettingFactor);
      }
      if (config.initialCovariance !== undefined) {
        builder.setInitialCovariance(config.initialCovariance);
      }
      if (config.regularization !== undefined) {
        builder.setRegularization(config.regularization);
      }
      if (config.confidenceLevel !== undefined) {
        builder.setConfidenceLevel(config.confidenceLevel);
      }
    }
    this.config = builder.build();

    // Initialize components with dependency injection
    this.matrixOps = new MatrixOperations();
    this.featureGenerator = new PolynomialFeatureGenerator();
    this.covarianceManager = new CovarianceManager(this.matrixOps);
    this.weightManager = new WeightManager();
    this.predictionEngine = new PredictionEngine(this.matrixOps);
    this.statisticsTracker = new StatisticsTracker();

    // Initialize normalization based on configuration
    this.normalizationStrategy = this.createNormalizationStrategy();

    // Initialize empty normalization stats
    this.normalizationStats = {
      min: [],
      max: [],
      mean: [],
      std: [],
      count: 0,
    };
  }

  /**
   * Creates the appropriate normalization strategy based on configuration.
   *
   * @returns Normalization strategy instance
   * @internal
   */
  private createNormalizationStrategy(): INormalizationStrategy {
    if (
      !this.config.enableNormalization ||
      this.config.normalizationMethod === "none"
    ) {
      return new NoNormalizationStrategy();
    }
    if (this.config.normalizationMethod === "z-score") {
      return new ZScoreNormalizationStrategy();
    }
    return new MinMaxNormalizationStrategy();
  }

  // ===========================================================================
  // PUBLIC METHODS
  // ===========================================================================

  /**
   * Incrementally trains the model using the RLS algorithm.
   *
   * @param params - Training data
   * @param params.xCoordinates - Input samples, shape [nSamples][inputDim]
   * @param params.yCoordinates - Target values, shape [nSamples][outputDim]
   *
   * @throws {Error} If xCoordinates and yCoordinates have different lengths
   * @throws {Error} If input/output dimensions are inconsistent with prior data
   *
   * @remarks
   * The model is lazily initialized on the first call based on the dimensions
   * of the input data. Subsequent calls must have matching dimensions.
   *
   * Each sample is processed sequentially using the RLS update rules,
   * making this suitable for streaming data applications.
   *
   * Time complexity: O(nSamples × featureCount²)
   * Space complexity: O(featureCount²) for covariance matrix (one-time)
   *
   * @example
   * ```typescript
   * const model = new MultivariatePolynomialRegression();
   *
   * // Initial training
   * model.fitOnline({
   *   xCoordinates: [[1, 2], [2, 3], [3, 4]],
   *   yCoordinates: [[5], [8], [13]]
   * });
   *
   * // Incremental update with new data
   * model.fitOnline({
   *   xCoordinates: [[4, 5]],
   *   yCoordinates: [[20]]
   * });
   * ```
   */
  public fitOnline(
    params: { xCoordinates: number[][]; yCoordinates: number[][] },
  ): void {
    const { xCoordinates, yCoordinates } = params;

    // Validate input arrays
    if (xCoordinates.length !== yCoordinates.length) {
      throw new Error(
        `xCoordinates length (${xCoordinates.length}) must match ` +
          `yCoordinates length (${yCoordinates.length})`,
      );
    }
    if (xCoordinates.length === 0) {
      return; // Nothing to process
    }

    // Lazy initialization on first data
    if (!this.initialized) {
      this.initializeModel(xCoordinates[0].length, yCoordinates[0].length);
    }

    // Validate dimensions match
    for (let i = 0; i < xCoordinates.length; i++) {
      if (xCoordinates[i].length !== this.inputDim) {
        throw new Error(
          `Input dimension mismatch at index ${i}: ` +
            `expected ${this.inputDim}, got ${xCoordinates[i].length}`,
        );
      }
      if (yCoordinates[i].length !== this.outputDim) {
        throw new Error(
          `Output dimension mismatch at index ${i}: ` +
            `expected ${this.outputDim}, got ${yCoordinates[i].length}`,
        );
      }
    }

    // Process each sample using RLS
    const numSamples = xCoordinates.length;
    for (let s = 0; s < numSamples; s++) {
      this.processOneSample(xCoordinates[s], yCoordinates[s]);
    }
  }

  /**
   * Makes predictions with confidence intervals.
   *
   * @param params - Prediction parameters
   * @param params.futureSteps - Number of steps to extrapolate beyond last input (optional)
   * @param params.inputPoints - Specific input points to predict (optional)
   * @returns Complete prediction results with confidence intervals and metrics
   *
   * @remarks
   * Either futureSteps or inputPoints should be provided (or both).
   *
   * If inputPoints is provided, predictions are made at those exact points.
   * If only futureSteps is provided, simple linear extrapolation from the
   * last training input is used (incrementing each dimension by 1 per step).
   *
   * The confidence intervals use t-distribution for small samples (n ≤ 30)
   * and z-distribution for larger samples.
   *
   * Time complexity: O(nPoints × featureCount²) for predictions
   * Space complexity: O(nPoints) for results
   *
   * @example
   * ```typescript
   * // Predict at specific points
   * const result1 = model.predict({
   *   inputPoints: [[5, 6], [6, 7], [7, 8]]
   * });
   *
   * // Extrapolate 5 steps from last training point
   * const result2 = model.predict({ futureSteps: 5 });
   *
   * // Access results
   * result1.predictions.forEach((pred) => {
   *   console.log(`Predicted: ${pred.predicted}`);
   *   console.log(`95% CI: [${pred.lowerBound}, ${pred.upperBound}]`);
   *   console.log(`SE: ${pred.standardError}`);
   * });
   * ```
   */
  public predict(
    params: { futureSteps?: number; inputPoints?: number[][] },
  ): PredictionResult {
    const { futureSteps, inputPoints } = params;

    // Initialize result with current model state
    const result: PredictionResult = {
      predictions: [],
      confidenceLevel: this.config.confidenceLevel,
      rSquared: this.statisticsTracker.getRSquared(),
      rmse: this.statisticsTracker.getRMSE(),
      sampleCount: this.sampleCount,
      isModelReady: this.initialized && this.sampleCount >= this.featureCount,
    };

    // Cannot predict without initialization
    if (!this.initialized) {
      return result;
    }

    // Determine points to predict
    let pointsToPredict: number[][];

    if (inputPoints && inputPoints.length > 0) {
      // Validate input dimensions
      for (let i = 0; i < inputPoints.length; i++) {
        if (inputPoints[i].length !== this.inputDim) {
          throw new Error(
            `Input point ${i} dimension mismatch: ` +
              `expected ${this.inputDim}, got ${inputPoints[i].length}`,
          );
        }
      }
      pointsToPredict = inputPoints;
    } else if (
      futureSteps !== undefined && futureSteps > 0 && this.lastInput !== null
    ) {
      // Generate extrapolation points
      pointsToPredict = this.generateExtrapolationPoints(futureSteps);
    } else {
      return result; // Nothing to predict
    }

    // Make predictions for each point
    const numPoints = pointsToPredict.length;
    for (let i = 0; i < numPoints; i++) {
      const prediction = this.predictSinglePoint(pointsToPredict[i]);
      result.predictions.push(prediction);
    }

    return result;
  }

  /**
   * Returns a comprehensive summary of the model state.
   *
   * @returns Model summary object
   *
   * @remarks
   * Time complexity: O(1)
   * Space complexity: O(1)
   *
   * @example
   * ```typescript
   * const summary = model.getModelSummary();
   * console.log(`Initialized: ${summary.isInitialized}`);
   * console.log(`Input dim: ${summary.inputDimension}`);
   * console.log(`Output dim: ${summary.outputDimension}`);
   * console.log(`Polynomial degree: ${summary.polynomialDegree}`);
   * console.log(`Feature count: ${summary.polynomialFeatureCount}`);
   * console.log(`Sample count: ${summary.sampleCount}`);
   * console.log(`R²: ${summary.rSquared.toFixed(4)}`);
   * console.log(`RMSE: ${summary.rmse.toFixed(4)}`);
   * ```
   */
  public getModelSummary(): ModelSummary {
    return {
      isInitialized: this.initialized,
      inputDimension: this.inputDim,
      outputDimension: this.outputDim,
      polynomialDegree: this.config.polynomialDegree,
      polynomialFeatureCount: this.featureCount,
      sampleCount: this.sampleCount,
      rSquared: this.statisticsTracker.getRSquared(),
      rmse: this.statisticsTracker.getRMSE(),
      normalizationEnabled: this.config.enableNormalization,
      normalizationMethod: this.config.normalizationMethod,
    };
  }

  /**
   * Returns the current weight matrix.
   *
   * @returns Weight matrix as 2D array [featureCount][outputDim]
   *
   * @remarks
   * Returns empty array if model not initialized.
   * Creates new arrays - not suitable for hot paths.
   *
   * Time complexity: O(featureCount × outputDim)
   * Space complexity: O(featureCount × outputDim)
   *
   * @example
   * ```typescript
   * const weights = model.getWeights();
   * console.log(`Number of polynomial features: ${weights.length}`);
   * weights.forEach((row, i) => {
   *   console.log(`Feature ${i} weights: ${row}`);
   * });
   * ```
   */
  public getWeights(): number[][] {
    if (!this.initialized) {
      return [];
    }
    return this.weightManager.getWeightsAs2D();
  }

  /**
   * Returns the current normalization statistics.
   *
   * @returns Copy of normalization statistics
   *
   * @remarks
   * Returns defensive copy to prevent external modification.
   *
   * Time complexity: O(inputDim)
   * Space complexity: O(inputDim)
   *
   * @example
   * ```typescript
   * const stats = model.getNormalizationStats();
   * console.log(`Data min: ${stats.min}`);
   * console.log(`Data max: ${stats.max}`);
   * console.log(`Data mean: ${stats.mean}`);
   * console.log(`Data std: ${stats.std}`);
   * console.log(`Samples: ${stats.count}`);
   * ```
   */
  public getNormalizationStats(): NormalizationStats {
    return {
      min: [...this.normalizationStats.min],
      max: [...this.normalizationStats.max],
      mean: [...this.normalizationStats.mean],
      std: [...this.normalizationStats.std],
      count: this.normalizationStats.count,
    };
  }

  /**
   * Resets the model to initial uninitialized state.
   *
   * @remarks
   * Clears all learned weights, covariance matrix, and statistics.
   * The model can be retrained from scratch after reset.
   *
   * Time complexity: O(featureCount²)
   * Space complexity: O(1)
   *
   * @example
   * ```typescript
   * model.fitOnline({ xCoordinates: [...], yCoordinates: [...] });
   * console.log(model.getModelSummary().sampleCount); // > 0
   *
   * model.reset();
   * console.log(model.getModelSummary().sampleCount); // 0
   * console.log(model.getModelSummary().isInitialized); // false
   * ```
   */
  public reset(): void {
    // Reset state flags
    this.initialized = false;
    this.inputDim = 0;
    this.outputDim = 0;
    this.featureCount = 0;
    this.sampleCount = 0;
    this.regularizationCounter = 0;

    // Reset all managers
    this.featureGenerator.reset();
    this.covarianceManager.reset();
    this.weightManager.reset();
    this.statisticsTracker.reset();

    // Reinitialize normalization strategy
    this.normalizationStrategy = this.createNormalizationStrategy();
    this.normalizationStats = {
      min: [],
      max: [],
      mean: [],
      std: [],
      count: 0,
    };

    // Clear preallocated buffers (will be reallocated on next fit)
    this.normalizedInput = null;
    this.featureBuffer = null;
    this.gainBuffer = null;
    this.errorBuffer = null;
    this.predictionBuffer = null;
    this.targetBuffer = null;
    this.lastInput = null;

    // Clear object pool
    this.predictionPool.length = 0;
  }

  // ===========================================================================
  // PRIVATE METHODS
  // ===========================================================================

  /**
   * Initializes model dimensions and allocates buffers.
   * Called lazily on first fitOnline call.
   *
   * @param inputDim - Number of input dimensions
   * @param outputDim - Number of output dimensions
   * @internal
   */
  private initializeModel(inputDim: number, outputDim: number): void {
    this.inputDim = inputDim;
    this.outputDim = outputDim;

    // Initialize polynomial feature generator
    this.featureGenerator.initialize(inputDim, this.config.polynomialDegree);
    this.featureCount = this.featureGenerator.getFeatureCount(
      inputDim,
      this.config.polynomialDegree,
    );

    // Initialize RLS components
    this.covarianceManager.initialize(
      this.featureCount,
      this.config.initialCovariance,
    );
    this.weightManager.initialize(this.featureCount, outputDim);

    // Initialize normalization stats arrays
    this.normalizationStats = {
      min: new Array<number>(inputDim).fill(Infinity),
      max: new Array<number>(inputDim).fill(-Infinity),
      mean: new Array<number>(inputDim).fill(0),
      std: new Array<number>(inputDim).fill(0),
      count: 0,
    };

    // Preallocate all buffers
    this.normalizedInput = new Float64Array(inputDim);
    this.featureBuffer = new Float64Array(this.featureCount);
    this.gainBuffer = new Float64Array(this.featureCount);
    this.errorBuffer = new Float64Array(outputDim);
    this.predictionBuffer = new Float64Array(outputDim);
    this.targetBuffer = new Float64Array(outputDim);
    this.lastInput = new Float64Array(inputDim);

    this.initialized = true;
  }

  /**
   * Processes a single training sample using the RLS algorithm.
   *
   * Algorithm steps:
   * 1. Update normalization statistics
   * 2. Normalize input
   * 3. Generate polynomial features
   * 4. Compute current prediction
   * 5. Compute prediction error
   * 6. Compute gain vector
   * 7. Update weights
   * 8. Update covariance
   * 9. Apply regularization (periodically)
   * 10. Update performance statistics
   *
   * @param x - Input vector as number[]
   * @param y - Target vector as number[]
   * @internal
   */
  private processOneSample(x: number[], y: number[]): void {
    // Copy input to typed array buffer (avoids allocation)
    for (let i = 0; i < this.inputDim; i++) {
      this.normalizedInput![i] = x[i];
      this.lastInput![i] = x[i];
    }

    // Copy target to typed array buffer
    for (let i = 0; i < this.outputDim; i++) {
      this.targetBuffer![i] = y[i];
    }

    // Step 1: Update normalization statistics with raw input
    this.normalizationStrategy.updateStats(
      this.normalizationStats,
      this.normalizedInput!,
    );

    // Step 2: Normalize input in-place
    this.normalizationStrategy.normalizeInPlace(
      this.normalizedInput!,
      this.normalizedInput!,
      this.normalizationStats,
    );

    // Step 3: Generate polynomial features φ = poly(normalize(x))
    this.featureGenerator.generateFeaturesInPlace(
      this.featureBuffer!,
      this.normalizedInput!,
      this.config.polynomialDegree,
    );

    // Step 4: Compute current prediction ŷ = Wᵀ·φ
    this.weightManager.predict(this.featureBuffer!, this.predictionBuffer!);

    // Step 5: Compute prediction error e = y - ŷ
    for (let i = 0; i < this.outputDim; i++) {
      this.errorBuffer![i] = this.targetBuffer![i] - this.predictionBuffer![i];
    }

    // Step 6: Compute gain vector k = P·φ / (λ + φᵀ·P·φ)
    this.covarianceManager.computeGainVector(
      this.featureBuffer!,
      this.config.forgettingFactor,
      this.gainBuffer!,
    );

    // Step 7: Update weights W = W + k·eᵀ
    this.weightManager.updateWeights(this.gainBuffer!, this.errorBuffer!);

    // Step 8: Update covariance P = (P - k·φᵀ·P) / λ
    this.covarianceManager.updateCovariance(
      this.gainBuffer!,
      this.featureBuffer!,
      this.config.forgettingFactor,
    );

    // Step 9: Apply regularization periodically for numerical stability
    this.regularizationCounter++;
    if (
      this.regularizationCounter >=
        MultivariatePolynomialRegression.REGULARIZATION_FREQUENCY
    ) {
      this.covarianceManager.applyRegularization(this.config.regularization);
      this.regularizationCounter = 0;
    }

    // Step 10: Update statistics with NEW prediction (after weight update)
    this.weightManager.predict(this.featureBuffer!, this.predictionBuffer!);
    this.statisticsTracker.update(this.targetBuffer!, this.predictionBuffer!);

    this.sampleCount++;
  }

  /**
   * Generates extrapolation points by incrementing each input dimension.
   *
   * @param steps - Number of future steps to generate
   * @returns Array of input points
   * @internal
   */
  private generateExtrapolationPoints(steps: number): number[][] {
    const points: number[][] = [];

    // Simple extrapolation: increment each dimension by step index
    // Note: This is a naive approach. For better extrapolation,
    // users should provide specific inputPoints.
    for (let s = 1; s <= steps; s++) {
      const point: number[] = new Array(this.inputDim);
      for (let i = 0; i < this.inputDim; i++) {
        point[i] = this.lastInput![i] + s;
      }
      points.push(point);
    }

    return points;
  }

  /**
   * Predicts output for a single input point with confidence intervals.
   *
   * @param x - Input point as number[]
   * @returns SinglePrediction with confidence intervals
   * @internal
   */
  private predictSinglePoint(x: number[]): SinglePrediction {
    // Get or create prediction object from pool
    const prediction = this.getOrCreatePrediction();

    // Copy input to buffer
    for (let i = 0; i < this.inputDim; i++) {
      this.normalizedInput![i] = x[i];
    }

    // Normalize input
    this.normalizationStrategy.normalizeInPlace(
      this.normalizedInput!,
      this.normalizedInput!,
      this.normalizationStats,
    );

    // Generate polynomial features
    this.featureGenerator.generateFeaturesInPlace(
      this.featureBuffer!,
      this.normalizedInput!,
      this.config.polynomialDegree,
    );

    // Make prediction with confidence intervals
    this.predictionEngine.predictWithConfidence(
      this.featureBuffer!,
      this.covarianceManager.getCovarianceMatrix(),
      this.weightManager.getWeights(),
      this.featureCount,
      this.outputDim,
      this.config.confidenceLevel,
      this.sampleCount,
      prediction,
    );

    return prediction;
  }

  /**
   * Gets a SinglePrediction from pool or creates a new one.
   * Implements object pooling for reduced GC pressure.
   *
   * @returns SinglePrediction object with preallocated arrays
   * @internal
   */
  private getOrCreatePrediction(): SinglePrediction {
    if (this.predictionPool.length > 0) {
      const prediction = this.predictionPool.pop()!;
      // Ensure arrays are correct size
      if (prediction.predicted.length !== this.outputDim) {
        prediction.predicted = new Array<number>(this.outputDim).fill(0);
        prediction.lowerBound = new Array<number>(this.outputDim).fill(0);
        prediction.upperBound = new Array<number>(this.outputDim).fill(0);
        prediction.standardError = new Array<number>(this.outputDim).fill(0);
      }
      return prediction;
    }

    // Create new prediction object with correct size arrays
    return {
      predicted: new Array<number>(this.outputDim).fill(0),
      lowerBound: new Array<number>(this.outputDim).fill(0),
      upperBound: new Array<number>(this.outputDim).fill(0),
      standardError: new Array<number>(this.outputDim).fill(0),
    };
  }

  /**
   * Returns a prediction object to the pool for reuse.
   *
   * @param prediction - Prediction to return to pool
   * @internal
   */
  private recyclePrediction(prediction: SinglePrediction): void {
    // Limit pool size to prevent memory buildup
    if (this.predictionPool.length < 100) {
      this.predictionPool.push(prediction);
    }
  }
}

// =============================================================================
// EXPORTS
// =============================================================================

export {
  // Builder
  ConfigurationBuilder,
  CovarianceManager,
  type ICovarianceManager,
  // Internal interfaces (for testing/extension)
  type IMatrixOperations,
  // Interfaces
  type IMultivariatePolynomialRegressionConfig,
  type INormalizationStrategy,
  type IPolynomialFeatureGenerator,
  type IPredictionEngine,
  type IStatisticsTracker,
  type IWeightManager,
  // Component classes (for advanced usage)
  MatrixOperations,
  MinMaxNormalizationStrategy,
  type ModelSummary,
  // Main class
  MultivariatePolynomialRegression,
  NoNormalizationStrategy,
  type NormalizationStats,
  PolynomialFeatureGenerator,
  PredictionEngine,
  type PredictionResult,
  type SinglePrediction,
  StatisticsTracker,
  WeightManager,
  ZScoreNormalizationStrategy,
};
