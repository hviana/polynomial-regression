/**
 * Multivariate Polynomial Regression with Online Training
 *
 * This module provides a comprehensive implementation of multivariate polynomial
 * regression with support for online (incremental) learning using the Recursive
 * Least Squares (RLS) algorithm.
 *
 * Key Features:
 * - Online/incremental training via RLS algorithm
 * - Configurable polynomial degree
 * - Optional automatic data normalization (min-max or z-score)
 * - Confidence interval estimation
 * - Multi-dimensional input and output support
 *
 * FIXES APPLIED IN THIS VERSION (v1.0.2):
 * 1. Fixed confidence interval calculation - was returning 0 due to incorrect
 *    TSS tracking and residual variance estimation
 * 2. Fixed excessive predictions by properly bounding covariance matrix updates
 *    and improving numerical stability throughout
 * 3. Completely rewrote polynomial term generation to use true iterative approach
 *    without intermediate array allocations - prevents memory overflow
 * 4. Optimized all matrix operations to minimize allocations and use in-place
 *    updates where safe
 * 5. Added proper effective sample counting for exponentially weighted statistics
 * 6. Fixed normalization to handle extrapolation gracefully
 * 7. Added covariance matrix conditioning to prevent ill-conditioning
 *
 * @module MultivariatePolynomialRegression
 * @version 1.0.2
 * @author Henrique Emanoel Viana
 */

// ============================================================================
// TYPE DEFINITIONS AND INTERFACES
// ============================================================================

/**
 * Supported normalization methods for input data preprocessing
 * - 'none': No normalization applied
 * - 'min-max': Scales values to [0, 1] range based on observed min/max
 * - 'z-score': Standardizes values to have mean 0 and std dev 1
 */
export type NormalizationMethod = "none" | "min-max" | "z-score";

/**
 * Configuration options for the polynomial regression model
 */
export interface PolynomialRegressionConfig {
  /** Degree of the polynomial (default: 2). Must be >= 1 */
  polynomialDegree?: number;

  /** Whether to normalize input data (default: true) */
  enableNormalization?: boolean;

  /** Method used for normalization (default: 'min-max') */
  normalizationMethod?: NormalizationMethod;

  /**
   * Forgetting factor for RLS algorithm (0 < λ ≤ 1, default: 0.99)
   * Values closer to 1 give more weight to historical data
   * Values closer to 0 adapt faster to recent data
   */
  forgettingFactor?: number;

  /**
   * Initial covariance matrix diagonal value (default: 100)
   * Higher values allow faster initial learning but may cause instability
   * CHANGED: Reduced from 1000 to 100 for better numerical stability
   */
  initialCovariance?: number;

  /**
   * Regularization parameter to prevent numerical instability (default: 1e-4)
   * Added to covariance matrix diagonal after each update
   * CHANGED: Increased from 1e-6 to 1e-4 for better conditioning
   */
  regularization?: number;

  /**
   * Confidence level for prediction intervals (0-1, default: 0.95)
   * E.g., 0.95 gives 95% confidence intervals
   */
  confidenceLevel?: number;
}

/**
 * Parameters for the fitOnline method
 */
export interface FitOnlineParams {
  /**
   * Input feature vectors where each element is a vector of numbers
   * Shape: [n_samples][n_features]
   * Each inner array represents one sample's features
   */
  xCoordinates: number[][];

  /**
   * Output target vectors where each element is a vector of numbers
   * Shape: [n_samples][n_outputs]
   * Each inner array represents one sample's target values
   */
  yCoordinates: number[][];
}

/**
 * Parameters for the predict method
 */
export interface PredictParams {
  /**
   * Number of future time steps to predict.
   * The model will extrapolate from the last seen data point.
   * Used when inputPoints is not provided.
   */
  futureSteps: number;

  /**
   * Optional: Specific input points to predict for (overrides futureSteps extrapolation)
   * Shape: [n_points][n_features]
   * If provided, predictions are made for these exact points
   */
  inputPoints?: number[][];
}

/**
 * Single prediction result for one time step or input point
 */
export interface SinglePrediction {
  /** Predicted y values for each output dimension */
  predicted: number[];

  /** Lower bound of confidence interval for each output */
  lowerBound: number[];

  /** Upper bound of confidence interval for each output */
  upperBound: number[];

  /** Standard error of the prediction for each output */
  standardError: number[];
}

/**
 * Complete result from the predict method
 */
export interface PredictionResult {
  /** Array of predictions for each future step or input point */
  predictions: SinglePrediction[];

  /** Confidence level used for intervals (0-1) */
  confidenceLevel: number;

  /**
   * Model's current R-squared value (coefficient of determination)
   * Range: [0, 1], where 1 indicates perfect fit
   */
  rSquared: number;

  /** Root Mean Square Error of the model (lower is better) */
  rmse: number;

  /** Number of samples the model has been trained on */
  sampleCount: number;

  /**
   * Whether the model is sufficiently trained for reliable predictions
   * True when sample count >= 2 * polynomial feature count
   */
  isModelReady: boolean;
}

/**
 * Statistics for normalization - tracks running statistics for data scaling
 */
interface NormalizationStats {
  /** Minimum values observed for each feature (used for min-max normalization) */
  min: number[];

  /** Maximum values observed for each feature (used for min-max normalization) */
  max: number[];

  /** Running mean values for each feature (used for z-score normalization) */
  mean: number[];

  /**
   * Running M2 accumulator for Welford's algorithm (sum of squared deviations)
   * Used to compute variance incrementally: variance = M2 / (n - 1)
   */
  m2: number[];

  /** Number of samples used to compute statistics */
  count: number;
}

/**
 * Internal model state - maintains all learned parameters and statistics
 */
interface ModelState {
  /**
   * Weight coefficients matrix [n_polynomial_features][n_outputs]
   * Each row corresponds to a polynomial feature, each column to an output dimension
   */
  weights: number[][];

  /**
   * Covariance matrix for RLS [n_polynomial_features][n_polynomial_features]
   * Represents the uncertainty in weight estimates
   */
  covarianceMatrix: number[][];

  /**
   * Sum of squared residuals for each output dimension (unweighted)
   * Used for RMSE calculation
   */
  residualSumSquares: number[];

  /**
   * Effective sum of squared residuals with forgetting factor applied
   * Used for variance estimation in confidence intervals
   */
  weightedResidualSS: number[];

  /**
   * Effective sample count accounting for forgetting factor
   * Converges to 1/(1-λ) as samples increase
   */
  effectiveSampleCount: number;

  /** Running mean of y values for each output dimension */
  yMean: number[];

  /**
   * Running M2 accumulator for y variance (Welford's algorithm)
   * Used for TSS calculation: TSS = M2
   */
  yM2: number[];

  /** Number of samples processed */
  sampleCount: number;

  /** Last input point seen (for extrapolation in prediction) */
  lastInputPoint: number[] | null;

  /** Time index for sequential data tracking */
  timeIndex: number;
}

// ============================================================================
// UTILITY CLASSES
// ============================================================================

/**
 * Matrix operations utility class
 * Provides essential linear algebra operations for the regression algorithm
 *
 * OPTIMIZATION NOTES:
 * - Methods are designed to minimize memory allocations
 * - Where possible, operations modify arrays in place
 * - Pre-allocation is used for predictable output sizes
 * - Loop unrolling is avoided in favor of clarity (JIT handles this)
 */
class MatrixOperations {
  /**
   * Creates an identity matrix of size n x n
   * The identity matrix has 1s on the diagonal and 0s elsewhere
   *
   * @param n - Size of the matrix
   * @returns n x n identity matrix
   */
  static identity(n: number): number[][] {
    // Pre-allocate the result array for better performance
    const result: number[][] = new Array(n);
    for (let i = 0; i < n; i++) {
      // Create each row with zeros, then set diagonal
      result[i] = new Array(n).fill(0);
      result[i][i] = 1;
    }
    return result;
  }

  /**
   * Creates a matrix filled with zeros
   * Uses a single loop with fill() for efficiency
   *
   * @param rows - Number of rows
   * @param cols - Number of columns
   * @returns rows x cols matrix of zeros
   */
  static zeros(rows: number, cols: number): number[][] {
    const result: number[][] = new Array(rows);
    for (let i = 0; i < rows; i++) {
      result[i] = new Array(cols).fill(0);
    }
    return result;
  }

  /**
   * Multiplies a matrix by a scalar value IN PLACE
   * Each element is multiplied by the scalar
   * WARNING: This modifies the input matrix!
   *
   * @param matrix - Input matrix (modified in place)
   * @param scalar - Scalar multiplier
   */
  static scalarMultiplyInPlace(matrix: number[][], scalar: number): void {
    const rows = matrix.length;
    for (let i = 0; i < rows; i++) {
      const row = matrix[i];
      const cols = row.length;
      for (let j = 0; j < cols; j++) {
        row[j] *= scalar;
      }
    }
  }

  /**
   * Multiplies a matrix by a scalar value, returning a new matrix
   * Use this when the original must be preserved
   *
   * @param matrix - Input matrix
   * @param scalar - Scalar multiplier
   * @returns New matrix with scaled values
   */
  static scalarMultiply(matrix: number[][], scalar: number): number[][] {
    const rows = matrix.length;
    const result: number[][] = new Array(rows);
    for (let i = 0; i < rows; i++) {
      const row = matrix[i];
      const cols = row.length;
      const newRow = new Array(cols);
      for (let j = 0; j < cols; j++) {
        newRow[j] = row[j] * scalar;
      }
      result[i] = newRow;
    }
    return result;
  }

  /**
   * Adds two matrices element-wise, storing result in first matrix IN PLACE
   * Both matrices must have the same dimensions
   * WARNING: This modifies matrix a!
   *
   * @param a - First matrix (modified in place to hold result)
   * @param b - Second matrix
   */
  static addInPlace(a: number[][], b: number[][]): void {
    const rows = a.length;
    for (let i = 0; i < rows; i++) {
      const rowA = a[i];
      const rowB = b[i];
      const cols = rowA.length;
      for (let j = 0; j < cols; j++) {
        rowA[j] += rowB[j];
      }
    }
  }

  /**
   * Subtracts matrix b from matrix a element-wise IN PLACE
   * Both matrices must have the same dimensions
   * WARNING: This modifies matrix a!
   *
   * @param a - Matrix to subtract from (modified in place)
   * @param b - Matrix to subtract
   */
  static subtractInPlace(a: number[][], b: number[][]): void {
    const rows = a.length;
    for (let i = 0; i < rows; i++) {
      const rowA = a[i];
      const rowB = b[i];
      const cols = rowA.length;
      for (let j = 0; j < cols; j++) {
        rowA[j] -= rowB[j];
      }
    }
  }

  /**
   * Multiplies a matrix by a column vector
   * The vector length must equal the number of columns in the matrix
   *
   * OPTIMIZATION: Uses direct indexing and accumulation for cache efficiency
   *
   * @param matrix - Input matrix (n x m)
   * @param vector - Input vector (length m)
   * @returns Result vector (length n) where result[i] = sum(matrix[i][j] * vector[j])
   */
  static multiplyVector(matrix: number[][], vector: number[]): number[] {
    const rows = matrix.length;
    const result = new Array(rows);

    for (let i = 0; i < rows; i++) {
      const row = matrix[i];
      const cols = row.length;
      let sum = 0;
      // Direct accumulation loop - very cache friendly
      for (let j = 0; j < cols; j++) {
        sum += row[j] * vector[j];
      }
      result[i] = sum;
    }
    return result;
  }

  /**
   * Computes the outer product of two vectors and SUBTRACTS scaled result from matrix
   * This is optimized for the RLS covariance update: P = P - k * (P * phi)'
   * Combines outer product and subtraction to avoid intermediate allocation
   *
   * @param matrix - Matrix to update IN PLACE
   * @param a - First vector (length n)
   * @param b - Second vector (length m)
   * @param scale - Scale factor for the outer product (default 1)
   */
  static subtractScaledOuterProductInPlace(
    matrix: number[][],
    a: number[],
    b: number[],
    scale: number = 1,
  ): void {
    const n = a.length;
    const m = b.length;
    for (let i = 0; i < n; i++) {
      const ai = a[i] * scale;
      const row = matrix[i];
      for (let j = 0; j < m; j++) {
        row[j] -= ai * b[j];
      }
    }
  }

  /**
   * Computes the dot product (inner product) of two vectors
   * Both vectors must have the same length
   *
   * @param a - First vector
   * @param b - Second vector
   * @returns Scalar result = sum(a[i] * b[i])
   */
  static dotProduct(a: number[], b: number[]): number {
    const len = a.length;
    let sum = 0;
    for (let i = 0; i < len; i++) {
      sum += a[i] * b[i];
    }
    return sum;
  }

  /**
   * Adds a scalar to the diagonal elements of a square matrix IN PLACE
   * Used for regularization: P = P + λI
   *
   * @param matrix - Square matrix to modify
   * @param scalar - Value to add to diagonal elements
   */
  static addToDiagonalInPlace(matrix: number[][], scalar: number): void {
    const n = matrix.length;
    for (let i = 0; i < n; i++) {
      matrix[i][i] += scalar;
    }
  }

  /**
   * Creates a deep copy of a matrix
   *
   * @param matrix - Input matrix
   * @returns New matrix with copied values
   */
  static clone(matrix: number[][]): number[][] {
    const rows = matrix.length;
    const result: number[][] = new Array(rows);
    for (let i = 0; i < rows; i++) {
      // Use slice() for efficient array copying
      result[i] = matrix[i].slice();
    }
    return result;
  }

  /**
   * Computes the maximum absolute value of diagonal elements
   * Used for matrix conditioning checks
   *
   * @param matrix - Square matrix
   * @returns Maximum absolute diagonal value
   */
  static maxDiagonal(matrix: number[][]): number {
    const n = matrix.length;
    let maxVal = 0;
    for (let i = 0; i < n; i++) {
      const absVal = Math.abs(matrix[i][i]);
      if (absVal > maxVal) {
        maxVal = absVal;
      }
    }
    return maxVal;
  }

  /**
   * Computes the minimum absolute value of diagonal elements
   * Used for matrix conditioning checks
   *
   * @param matrix - Square matrix
   * @returns Minimum absolute diagonal value
   */
  static minDiagonal(matrix: number[][]): number {
    const n = matrix.length;
    let minVal = Infinity;
    for (let i = 0; i < n; i++) {
      const absVal = Math.abs(matrix[i][i]);
      if (absVal < minVal) {
        minVal = absVal;
      }
    }
    return minVal;
  }
}

/**
 * Statistical utility functions
 * Provides statistical calculations needed for confidence intervals
 */
class StatisticsUtils {
  /**
   * Computes the t-distribution critical value approximation
   * Uses the Cornish-Fisher expansion for better accuracy with small df
   *
   * @param confidenceLevel - Desired confidence level (e.g., 0.95 for 95%)
   * @param degreesOfFreedom - Degrees of freedom for the t-distribution
   * @returns Two-tailed critical value for the given confidence level
   */
  static tCriticalValue(
    confidenceLevel: number,
    degreesOfFreedom: number,
  ): number {
    // For confidence intervals, we need the two-tailed critical value
    // alpha is the total probability in both tails
    const alpha = 1 - confidenceLevel;

    // Clamp degrees of freedom to at least 1 for numerical stability
    const df = Math.max(1, degreesOfFreedom);

    // For large degrees of freedom, t-distribution approaches normal
    if (df > 30) {
      return this.normalInverseCDF(1 - alpha / 2);
    }

    // For smaller df, use Cornish-Fisher expansion for better accuracy
    const zAlpha = this.normalInverseCDF(1 - alpha / 2);

    // Cornish-Fisher correction terms
    const g1 = (zAlpha ** 3 + zAlpha) / 4;
    const g2 = (5 * zAlpha ** 5 + 16 * zAlpha ** 3 + 3 * zAlpha) / 96;

    return zAlpha + g1 / df + g2 / (df * df);
  }

  /**
   * Approximates the inverse CDF of the standard normal distribution
   * Uses the Abramowitz and Stegun rational approximation (formula 26.2.23)
   *
   * This is highly accurate for p in [0.0000001, 0.9999999]
   *
   * @param p - Probability (must be between 0 and 1 exclusive)
   * @returns z-value such that P(Z < z) = p for standard normal Z
   */
  static normalInverseCDF(p: number): number {
    // Clamp p to valid range to prevent NaN/Infinity
    const clampedP = Math.max(1e-10, Math.min(1 - 1e-10, p));

    // Coefficients for the rational approximation
    const a1 = -3.969683028665376e1;
    const a2 = 2.209460984245205e2;
    const a3 = -2.759285104469687e2;
    const a4 = 1.383577518672690e2;
    const a5 = -3.066479806614716e1;
    const a6 = 2.506628277459239e0;

    const b1 = -5.447609879822406e1;
    const b2 = 1.615858368580409e2;
    const b3 = -1.556989798598866e2;
    const b4 = 6.680131188771972e1;
    const b5 = -1.328068155288572e1;

    const c1 = -7.784894002430293e-3;
    const c2 = -3.223964580411365e-1;
    const c3 = -2.400758277161838e0;
    const c4 = -2.549732539343734e0;
    const c5 = 4.374664141464968e0;
    const c6 = 2.938163982698783e0;

    const d1 = 7.784695709041462e-3;
    const d2 = 3.224671290700398e-1;
    const d3 = 2.445134137142996e0;
    const d4 = 3.754408661907416e0;

    // Threshold values for switching between approximation regions
    const pLow = 0.02425;
    const pHigh = 1 - pLow;

    let q: number, r: number;

    if (clampedP < pLow) {
      // Lower tail approximation
      q = Math.sqrt(-2 * Math.log(clampedP));
      return (
        (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
        ((((d1 * q + d2) * q + d3) * q + d4) * q + 1)
      );
    } else if (clampedP <= pHigh) {
      // Central region approximation (most common case)
      q = clampedP - 0.5;
      r = q * q;
      return (
        ((((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q) /
        (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1)
      );
    } else {
      // Upper tail approximation (symmetric to lower tail)
      q = Math.sqrt(-2 * Math.log(1 - clampedP));
      return (
        -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
        ((((d1 * q + d2) * q + d3) * q + d4) * q + 1)
      );
    }
  }
}

// ============================================================================
// MAIN CLASS IMPLEMENTATION
// ============================================================================

/**
 * Multivariate Polynomial Regression with Online Training
 *
 * This class implements a polynomial regression model that supports:
 * - Multiple input features (multivariate)
 * - Multiple output targets
 * - Online (incremental) learning using Recursive Least Squares
 * - Automatic data normalization
 * - Confidence interval estimation
 *
 * The Recursive Least Squares (RLS) algorithm allows the model to be updated
 * incrementally as new data arrives, without needing to retrain from scratch.
 * This is ideal for streaming data applications.
 *
 * @example
 * ```typescript
 * const model = new MultivariatePolynomialRegression({
 *     polynomialDegree: 2,
 *     enableNormalization: true,
 *     normalizationMethod: 'min-max'
 * });
 *
 * // Train incrementally
 * model.fitOnline({
 *     xCoordinates: [[1, 2], [2, 3], [3, 4]],
 *     yCoordinates: [[5], [8], [13]]
 * });
 *
 * // Predict future values
 * const result = model.predict({ futureSteps: 5 });
 * ```
 */
export class MultivariatePolynomialRegression {
  // ============================================================================
  // PRIVATE PROPERTIES
  // ============================================================================

  /**
   * Configuration with all required fields filled in with defaults
   * Made readonly to prevent accidental modification after construction
   */
  private readonly config: Required<PolynomialRegressionConfig>;

  /**
   * Internal model state containing weights, covariance, and statistics
   */
  private state: ModelState;

  /**
   * Statistics for input normalization
   */
  private normStats: NormalizationStats;

  /**
   * Number of input features (set on first training call)
   */
  private inputDimension: number = 0;

  /**
   * Number of output targets (set on first training call)
   */
  private outputDimension: number = 0;

  /**
   * Total number of polynomial features including constant term
   * For degree d and n inputs, this is C(n+d, d) = (n+d)! / (n! * d!)
   */
  private polynomialFeatureCount: number = 0;

  /**
   * Flag indicating if model has been initialized with first data batch
   */
  private isInitialized: boolean = false;

  /**
   * Cache of polynomial term exponent patterns for efficient feature generation
   * Each entry is an array of exponents for each input variable
   * E.g., for 2 inputs degree 2: [[0,0], [1,0], [0,1], [2,0], [1,1], [0,2]]
   *
   * OPTIMIZATION: This is a flat Int8Array for memory efficiency
   * Access pattern: patterns[termIndex * inputDim + featureIndex]
   */
  private polynomialTermPatterns: Int8Array = new Int8Array(0);

  /**
   * Reusable buffer for polynomial feature computation
   * Avoids allocating new arrays on every prediction
   */
  private featureBuffer: Float64Array = new Float64Array(0);

  /**
   * Reusable buffer for P * phi computation in RLS update
   */
  private pPhiBuffer: Float64Array = new Float64Array(0);

  /**
   * Reusable buffer for gain vector k in RLS update
   */
  private gainBuffer: Float64Array = new Float64Array(0);

  // ============================================================================
  // CONSTRUCTOR
  // ============================================================================

  /**
   * Creates a new MultivariatePolynomialRegression instance
   *
   * @param config - Configuration options for the model
   * @throws Error if configuration values are invalid
   */
  constructor(config: PolynomialRegressionConfig = {}) {
    // Set default configuration values using nullish coalescing
    // NOTE: Changed defaults for better numerical stability
    this.config = {
      polynomialDegree: config.polynomialDegree ?? 2,
      enableNormalization: config.enableNormalization ?? true,
      normalizationMethod: config.normalizationMethod ?? "min-max",
      forgettingFactor: config.forgettingFactor ?? 0.99,
      // CHANGED: Reduced from 1000 to 100 for stability
      initialCovariance: config.initialCovariance ?? 100,
      // CHANGED: Increased from 1e-6 to 1e-4 for better conditioning
      regularization: config.regularization ?? 1e-4,
      confidenceLevel: config.confidenceLevel ?? 0.95,
    };

    // Validate configuration parameters
    this.validateConfig();

    // Initialize empty state structures
    this.state = this.createEmptyState();
    this.normStats = this.createEmptyNormStats();
  }

  // ============================================================================
  // PRIVATE INITIALIZATION METHODS
  // ============================================================================

  /**
   * Validates the configuration parameters
   * Called during construction to catch invalid configurations early
   *
   * @throws Error if any configuration parameter is invalid
   */
  private validateConfig(): void {
    if (this.config.polynomialDegree < 1) {
      throw new Error("Polynomial degree must be at least 1");
    }
    // Limit polynomial degree to prevent combinatorial explosion
    if (this.config.polynomialDegree > 10) {
      throw new Error("Polynomial degree must not exceed 10");
    }
    if (this.config.forgettingFactor <= 0 || this.config.forgettingFactor > 1) {
      throw new Error("Forgetting factor must be in range (0, 1]");
    }
    if (this.config.confidenceLevel <= 0 || this.config.confidenceLevel >= 1) {
      throw new Error("Confidence level must be in range (0, 1)");
    }
    if (this.config.initialCovariance <= 0) {
      throw new Error("Initial covariance must be positive");
    }
    if (this.config.regularization < 0) {
      throw new Error("Regularization must be non-negative");
    }
  }

  /**
   * Creates an empty model state with default/zero values
   * Used during initialization and reset
   *
   * @returns Fresh ModelState object
   */
  private createEmptyState(): ModelState {
    return {
      weights: [],
      covarianceMatrix: [],
      residualSumSquares: [],
      weightedResidualSS: [],
      effectiveSampleCount: 0,
      yMean: [],
      yM2: [],
      sampleCount: 0,
      lastInputPoint: null,
      timeIndex: 0,
    };
  }

  /**
   * Creates empty normalization statistics structure
   * Used during initialization and reset
   *
   * @returns Fresh NormalizationStats object
   */
  private createEmptyNormStats(): NormalizationStats {
    return {
      min: [],
      max: [],
      mean: [],
      m2: [],
      count: 0,
    };
  }

  /**
   * Initializes the model with dimensions from the first data batch
   * This sets up all internal structures based on the actual data dimensions
   *
   * @param inputDim - Number of input features
   * @param outputDim - Number of output targets
   */
  private initializeModel(inputDim: number, outputDim: number): void {
    this.inputDimension = inputDim;
    this.outputDimension = outputDim;

    // Calculate polynomial feature count first (needed for buffer allocation)
    // Formula: C(inputDim + degree, degree) = (inputDim + degree)! / (inputDim! * degree!)
    this.polynomialFeatureCount = this.computePolynomialFeatureCount(
      inputDim,
      this.config.polynomialDegree,
    );

    // Pre-compute polynomial term patterns (MEMORY-OPTIMIZED ITERATIVE)
    this.polynomialTermPatterns = this.generatePolynomialTermPatterns(inputDim);

    // Allocate reusable buffers for computation
    this.featureBuffer = new Float64Array(this.polynomialFeatureCount);
    this.pPhiBuffer = new Float64Array(this.polynomialFeatureCount);
    this.gainBuffer = new Float64Array(this.polynomialFeatureCount);

    // Initialize weights matrix to zeros
    // Shape: [polynomialFeatureCount][outputDim]
    this.state.weights = MatrixOperations.zeros(
      this.polynomialFeatureCount,
      outputDim,
    );

    // Initialize covariance matrix as scaled identity matrix
    // The initial value represents our prior uncertainty about the weights
    this.state.covarianceMatrix = MatrixOperations.scalarMultiply(
      MatrixOperations.identity(this.polynomialFeatureCount),
      this.config.initialCovariance,
    );

    // Initialize output statistics arrays
    this.state.residualSumSquares = new Array(outputDim).fill(0);
    this.state.weightedResidualSS = new Array(outputDim).fill(0);
    this.state.yMean = new Array(outputDim).fill(0);
    this.state.yM2 = new Array(outputDim).fill(0);
    this.state.effectiveSampleCount = 0;

    // Initialize normalization statistics arrays
    this.normStats.min = new Array(inputDim).fill(Infinity);
    this.normStats.max = new Array(inputDim).fill(-Infinity);
    this.normStats.mean = new Array(inputDim).fill(0);
    this.normStats.m2 = new Array(inputDim).fill(0);

    this.isInitialized = true;
  }

  /**
   * Computes the number of polynomial features for given input dimension and degree
   * Uses the formula: C(n + d, d) where n = inputDim, d = degree
   * This equals the number of monomials of degree <= d in n variables
   *
   * @param inputDim - Number of input variables
   * @param degree - Maximum polynomial degree
   * @returns Total number of polynomial features
   */
  private computePolynomialFeatureCount(
    inputDim: number,
    degree: number,
  ): number {
    // C(n + d, d) = (n + d)! / (n! * d!)
    // Compute iteratively to avoid large factorials
    let result = 1;
    for (let i = 1; i <= degree; i++) {
      result = (result * (inputDim + i)) / i;
    }
    return Math.round(result); // Round to handle floating point errors
  }

  // ============================================================================
  // POLYNOMIAL FEATURE GENERATION (MEMORY-OPTIMIZED ITERATIVE)
  // ============================================================================

  /**
   * Generates all polynomial term patterns up to the configured degree
   *
   * MEMORY OPTIMIZATION: Uses Int8Array instead of nested arrays
   * This dramatically reduces memory usage and GC pressure
   *
   * Each pattern is stored as a contiguous block in the flat array:
   * - Term i's exponents are at indices [i * inputDim, (i+1) * inputDim)
   *
   * For example, with 2 inputs (x, y) and degree 2:
   * Flat array: [0,0, 1,0, 0,1, 2,0, 1,1, 0,2]
   * Which represents: 1, x, y, x², xy, y²
   *
   * @param inputDim - Number of input features
   * @returns Flat Int8Array containing all exponent patterns
   */
  private generatePolynomialTermPatterns(inputDim: number): Int8Array {
    const degree = this.config.polynomialDegree;
    const totalTerms = this.polynomialFeatureCount;

    // Allocate flat array: totalTerms patterns, each with inputDim exponents
    const patterns = new Int8Array(totalTerms * inputDim);

    // Pattern 0 is all zeros (constant term) - already initialized

    let patternIndex = 1; // Start after constant term

    // Generate terms for each degree from 1 to max degree
    // Using iterative enumeration with a single working array
    const currentExponents = new Int8Array(inputDim);

    for (let d = 1; d <= degree; d++) {
      // Reset working array for this degree
      currentExponents.fill(0);
      currentExponents[0] = d; // Start with all exponent on first variable

      // Copy first pattern of this degree
      const baseIdx = patternIndex * inputDim;
      for (let i = 0; i < inputDim; i++) {
        patterns[baseIdx + i] = currentExponents[i];
      }
      patternIndex++;

      // Generate remaining patterns for this degree using next-pattern iteration
      while (this.nextExponentPattern(currentExponents, d)) {
        const idx = patternIndex * inputDim;
        for (let i = 0; i < inputDim; i++) {
          patterns[idx + i] = currentExponents[i];
        }
        patternIndex++;
      }
    }

    return patterns;
  }

  /**
   * Advances to the next exponent pattern of a given total degree
   * Uses a "carry" algorithm similar to incrementing a number in a mixed-radix system
   *
   * The algorithm:
   * 1. Find the rightmost non-zero exponent (excluding the last position)
   * 2. Decrement it and add 1 to the position to its right
   * 3. Move any remaining exponent from later positions to the position after the decremented one
   *
   * This generates patterns in graded lexicographic order.
   *
   * @param exponents - Current exponent pattern (modified in place)
   * @param totalDegree - The sum that all exponents must equal
   * @returns true if advanced to a new pattern, false if no more patterns
   */
  private nextExponentPattern(
    exponents: Int8Array,
    totalDegree: number,
  ): boolean {
    const n = exponents.length;

    // Find rightmost position (excluding last) with non-zero exponent
    let pos = n - 2;
    while (pos >= 0 && exponents[pos] === 0) {
      pos--;
    }

    // If no such position exists, we've enumerated all patterns
    if (pos < 0) {
      return false;
    }

    // Decrement exponent at pos
    exponents[pos]--;

    // Calculate sum of exponents from pos+1 to end (will be redistributed)
    // After decrementing, we need to add 1 to maintain total degree
    let sumToRedistribute = 1; // The 1 we removed from pos
    for (let i = pos + 1; i < n; i++) {
      sumToRedistribute += exponents[i];
      exponents[i] = 0; // Clear these positions
    }

    // Put all the sum at position pos+1 (maintaining graded lex order)
    exponents[pos + 1] = sumToRedistribute;

    return true;
  }

  /**
   * Generates polynomial features from an input vector using pre-computed patterns
   * Uses the reusable featureBuffer to avoid allocations
   *
   * @param input - Input feature vector (normalized)
   * @returns The featureBuffer filled with polynomial features
   */
  private generatePolynomialFeatures(input: number[]): Float64Array {
    const numTerms = this.polynomialFeatureCount;
    const inputDim = this.inputDimension;
    const patterns = this.polynomialTermPatterns;
    const features = this.featureBuffer;

    // Compute each polynomial term using the pre-computed patterns
    for (let termIdx = 0; termIdx < numTerms; termIdx++) {
      let term = 1;
      const patternBase = termIdx * inputDim;

      // Multiply input values raised to their respective exponents
      for (let varIdx = 0; varIdx < inputDim; varIdx++) {
        const exp = patterns[patternBase + varIdx];
        if (exp !== 0) {
          // Optimize common exponent cases
          const x = input[varIdx];
          if (exp === 1) {
            term *= x;
          } else if (exp === 2) {
            term *= x * x;
          } else if (exp === 3) {
            term *= x * x * x;
          } else {
            term *= Math.pow(x, exp);
          }
        }
      }

      features[termIdx] = term;
    }

    return features;
  }

  // ============================================================================
  // NORMALIZATION METHODS
  // ============================================================================

  /**
   * Updates normalization statistics with new data points
   * Uses Welford's online algorithm for numerically stable mean and variance
   *
   * Welford's algorithm:
   * - mean_new = mean_old + (x - mean_old) / n
   * - M2_new = M2_old + (x - mean_old) * (x - mean_new)
   * - variance = M2 / (n - 1)
   *
   * @param xCoordinates - Array of input data points
   */
  private updateNormalizationStats(xCoordinates: number[][]): void {
    if (!this.config.enableNormalization) return;

    for (const x of xCoordinates) {
      this.normStats.count++;
      const n = this.normStats.count;

      for (let i = 0; i < x.length; i++) {
        const value = x[i];

        // Update min/max for min-max normalization
        if (value < this.normStats.min[i]) {
          this.normStats.min[i] = value;
        }
        if (value > this.normStats.max[i]) {
          this.normStats.max[i] = value;
        }

        // Welford's online algorithm for mean and M2
        const delta = value - this.normStats.mean[i];
        this.normStats.mean[i] += delta / n;
        const delta2 = value - this.normStats.mean[i];
        this.normStats.m2[i] += delta * delta2;
      }
    }
  }

  /**
   * Gets the standard deviation for a feature from the M2 accumulator
   *
   * @param featureIndex - Index of the feature
   * @returns Standard deviation, or 1 if insufficient data
   */
  private getFeatureStd(featureIndex: number): number {
    if (this.normStats.count < 2) return 1;
    const variance = this.normStats.m2[featureIndex] /
      (this.normStats.count - 1);
    return Math.sqrt(Math.max(0, variance));
  }

  /**
   * Normalizes an input vector based on current statistics
   * Applied before polynomial feature generation
   *
   * IMPROVEMENT: Now handles extrapolation gracefully by soft-clamping
   * values that fall outside the observed range
   *
   * @param input - Raw input vector
   * @returns Normalized input vector
   */
  private normalizeInput(input: number[]): number[] {
    // Skip normalization if disabled or insufficient data
    if (!this.config.enableNormalization || this.normStats.count < 2) {
      // Return copy to avoid mutation issues
      return input.slice();
    }

    const normalized = new Array(input.length);

    switch (this.config.normalizationMethod) {
      case "min-max": {
        for (let i = 0; i < input.length; i++) {
          const min = this.normStats.min[i];
          const max = this.normStats.max[i];
          const range = max - min;

          if (range < 1e-10) {
            // Constant feature - normalize to 0.5
            normalized[i] = 0.5;
          } else {
            // Standard min-max normalization
            let val = (input[i] - min) / range;

            // IMPROVEMENT: Soft-clamp for extrapolation to prevent extreme values
            // Use sigmoid-like compression outside [0, 1] range
            if (val < 0) {
              // Map negative values to (0, 0) asymptotically
              // Using: 0.5 * (1 + tanh(val))
              val = 0.5 * (1 + Math.tanh(val));
            } else if (val > 1) {
              // Map values > 1 to (1, 1.5) asymptotically
              val = 1 + 0.5 * Math.tanh(val - 1);
            }

            normalized[i] = val;
          }
        }
        break;
      }

      case "z-score": {
        for (let i = 0; i < input.length; i++) {
          const std = this.getFeatureStd(i);

          if (std < 1e-10) {
            // Constant feature
            normalized[i] = 0;
          } else {
            let val = (input[i] - this.normStats.mean[i]) / std;

            // IMPROVEMENT: Soft-clamp extreme z-scores to prevent instability
            // Compress values outside [-3, 3] range
            if (Math.abs(val) > 3) {
              val = 3 * Math.tanh(val / 3);
            }

            normalized[i] = val;
          }
        }
        break;
      }

      default:
        // No normalization
        return input.slice();
    }

    return normalized;
  }

  // ============================================================================
  // PUBLIC TRAINING METHOD
  // ============================================================================

  /**
   * Performs online training using Recursive Least Squares algorithm
   *
   * The RLS algorithm updates the model incrementally for each new sample:
   * 1. Compute gain vector: k = P * φ / (λ + φ' * P * φ)
   * 2. Update weights: w = w + k * (y - φ' * w)
   * 3. Update covariance: P = (P - k * φ' * P) / λ
   *
   * Where:
   * - P is the covariance matrix (uncertainty in weights)
   * - φ is the polynomial feature vector
   * - λ is the forgetting factor
   * - k is the gain vector (how much to update based on new data)
   *
   * @param params - Training parameters containing x and y coordinates
   * @throws Error if input dimensions are inconsistent
   */
  public fitOnline(params: FitOnlineParams): void {
    const { xCoordinates, yCoordinates } = params;

    // Validate input data before processing
    this.validateTrainingInput(xCoordinates, yCoordinates);

    // Initialize model on first training call
    if (!this.isInitialized) {
      this.initializeModel(xCoordinates[0].length, yCoordinates[0].length);
    }

    // Update normalization statistics before processing
    this.updateNormalizationStats(xCoordinates);

    // Process each sample through the RLS update
    for (let i = 0; i < xCoordinates.length; i++) {
      this.processOneSample(xCoordinates[i], yCoordinates[i]);
    }
  }

  /**
   * Validates training input data for consistency
   *
   * @param xCoordinates - Input coordinates
   * @param yCoordinates - Output coordinates
   * @throws Error if validation fails
   */
  private validateTrainingInput(
    xCoordinates: number[][],
    yCoordinates: number[][],
  ): void {
    if (xCoordinates.length === 0) {
      throw new Error("xCoordinates cannot be empty");
    }
    if (yCoordinates.length === 0) {
      throw new Error("yCoordinates cannot be empty");
    }
    if (xCoordinates.length !== yCoordinates.length) {
      throw new Error(
        "xCoordinates and yCoordinates must have the same length",
      );
    }

    const xDim = xCoordinates[0].length;
    const yDim = yCoordinates[0].length;

    if (xDim === 0) {
      throw new Error("Input dimension cannot be zero");
    }
    if (yDim === 0) {
      throw new Error("Output dimension cannot be zero");
    }

    // Check dimension consistency within the batch
    for (let i = 0; i < xCoordinates.length; i++) {
      if (xCoordinates[i].length !== xDim) {
        throw new Error(`Inconsistent x dimension at index ${i}`);
      }
      if (yCoordinates[i].length !== yDim) {
        throw new Error(`Inconsistent y dimension at index ${i}`);
      }

      // Check for NaN/Infinity in inputs
      for (let j = 0; j < xDim; j++) {
        if (!Number.isFinite(xCoordinates[i][j])) {
          throw new Error(`Invalid value in xCoordinates at [${i}][${j}]`);
        }
      }
      for (let j = 0; j < yDim; j++) {
        if (!Number.isFinite(yCoordinates[i][j])) {
          throw new Error(`Invalid value in yCoordinates at [${i}][${j}]`);
        }
      }
    }

    // Check dimension consistency with existing model
    if (this.isInitialized) {
      if (xDim !== this.inputDimension) {
        throw new Error(
          `Input dimension mismatch: expected ${this.inputDimension}, got ${xDim}`,
        );
      }
      if (yDim !== this.outputDimension) {
        throw new Error(
          `Output dimension mismatch: expected ${this.outputDimension}, got ${yDim}`,
        );
      }
    }
  }

  /**
   * Processes a single training sample using RLS update equations
   *
   * OPTIMIZATIONS:
   * - Uses pre-allocated buffers to avoid allocations
   * - Performs covariance update in-place
   * - Includes numerical stability safeguards
   *
   * @param x - Input vector (raw, unnormalized)
   * @param y - Output vector (target values)
   */
  private processOneSample(x: number[], y: number[]): void {
    // Step 1: Normalize input and generate polynomial features
    const normalizedX = this.normalizeInput(x);
    const phi = this.generatePolynomialFeatures(normalizedX);

    // Step 2: Compute RLS gain vector
    const lambda = this.config.forgettingFactor;
    const P = this.state.covarianceMatrix;
    const numFeatures = this.polynomialFeatureCount;

    // P * φ - store in pre-allocated buffer
    const Pphi = this.pPhiBuffer;
    for (let i = 0; i < numFeatures; i++) {
      let sum = 0;
      const row = P[i];
      for (let j = 0; j < numFeatures; j++) {
        sum += row[j] * phi[j];
      }
      Pphi[i] = sum;
    }

    // Denominator: λ + φ' * P * φ (scalar)
    let phiTPphi = 0;
    for (let i = 0; i < numFeatures; i++) {
      phiTPphi += phi[i] * Pphi[i];
    }
    const denominator = lambda + phiTPphi;

    // NUMERICAL STABILITY: Check for near-zero denominator
    if (Math.abs(denominator) < 1e-10) {
      // Skip this update - data point is nearly linearly dependent
      console.warn("RLS update skipped: near-singular condition detected");
      return;
    }

    // Gain vector: k = P * φ / denominator
    const k = this.gainBuffer;
    const invDenom = 1 / denominator;
    for (let i = 0; i < numFeatures; i++) {
      k[i] = Pphi[i] * invDenom;
    }

    // Step 3: Compute predictions and update weights for each output dimension
    for (let j = 0; j < this.outputDimension; j++) {
      // Current prediction with existing weights
      let prediction = 0;
      for (let i = 0; i < numFeatures; i++) {
        prediction += phi[i] * this.state.weights[i][j];
      }

      // Prediction error
      const error = y[j] - prediction;

      // Update weights: w = w + k * error
      for (let i = 0; i < numFeatures; i++) {
        this.state.weights[i][j] += k[i] * error;
      }

      // Update error statistics for this output dimension
      this.updateErrorStatistics(j, y[j], error);
    }

    // Step 4: Update covariance matrix IN PLACE
    // P_new = (P - k * Pphi') / λ
    // This is equivalent to: P_new = (P - k ⊗ Pphi) / λ
    // where ⊗ is outer product

    // First: P = P - k * Pphi' (subtract scaled outer product)
    MatrixOperations.subtractScaledOuterProductInPlace(P, k, Pphi, 1);

    // Second: P = P / λ (scale by inverse of forgetting factor)
    const invLambda = 1 / lambda;
    MatrixOperations.scalarMultiplyInPlace(P, invLambda);

    // Third: Add regularization to diagonal for numerical stability
    MatrixOperations.addToDiagonalInPlace(P, this.config.regularization);

    // NUMERICAL STABILITY: Ensure covariance matrix stays well-conditioned
    this.conditionCovarianceMatrix();

    // Step 5: Update tracking variables
    this.state.lastInputPoint = x.slice(); // Copy to avoid reference issues
    this.state.timeIndex++;
    this.state.sampleCount++;

    // Update effective sample count with forgetting factor
    // This converges to 1/(1-λ) as samples increase
    this.state.effectiveSampleCount = lambda * this.state.effectiveSampleCount +
      1;
  }

  /**
   * Ensures the covariance matrix remains numerically well-conditioned
   * This prevents the matrix from becoming near-singular or having
   * exploding values, which can cause prediction instability
   */
  private conditionCovarianceMatrix(): void {
    const P = this.state.covarianceMatrix;
    const n = P.length;

    // Check condition number via diagonal elements
    const maxDiag = MatrixOperations.maxDiagonal(P);
    const minDiag = MatrixOperations.minDiagonal(P);

    // If condition number is too high, add regularization
    // Condition number ~ maxDiag / minDiag
    if (maxDiag / (minDiag + 1e-10) > 1e8) {
      // Add scaled identity to improve conditioning
      const boost = maxDiag * 1e-6;
      MatrixOperations.addToDiagonalInPlace(P, boost);
    }

    // Ensure all diagonal elements are positive and bounded
    const maxAllowed = this.config.initialCovariance * 10;
    const minAllowed = this.config.regularization;

    for (let i = 0; i < n; i++) {
      // Clamp diagonal elements
      if (P[i][i] < minAllowed) {
        P[i][i] = minAllowed;
      } else if (P[i][i] > maxAllowed) {
        P[i][i] = maxAllowed;
      }

      // Ensure symmetry (fix floating point drift)
      for (let j = i + 1; j < n; j++) {
        const avg = (P[i][j] + P[j][i]) / 2;
        P[i][j] = avg;
        P[j][i] = avg;
      }
    }
  }

  /**
   * Updates error statistics for R² and RMSE calculation
   *
   * FIXED: Uses Welford's algorithm for stable TSS tracking
   * and properly tracks both weighted and unweighted residuals
   *
   * @param outputIdx - Index of the output dimension
   * @param actual - Actual y value
   * @param error - Prediction error (actual - predicted)
   */
  private updateErrorStatistics(
    outputIdx: number,
    actual: number,
    error: number,
  ): void {
    const n = this.state.sampleCount + 1; // +1 because we haven't incremented yet
    const lambda = this.config.forgettingFactor;

    // Update unweighted residual sum of squares (for RMSE)
    const residualSquared = error * error;
    this.state.residualSumSquares[outputIdx] += residualSquared;

    // Update weighted residual sum of squares (for confidence intervals)
    // This gives more weight to recent residuals
    this.state.weightedResidualSS[outputIdx] =
      lambda * this.state.weightedResidualSS[outputIdx] + residualSquared;

    // Update running statistics for y using Welford's algorithm
    // This is used for TSS (Total Sum of Squares) calculation
    const prevMean = this.state.yMean[outputIdx];
    const delta = actual - prevMean;
    const newMean = prevMean + delta / n;
    const delta2 = actual - newMean;

    this.state.yMean[outputIdx] = newMean;
    // M2 accumulates sum of squared deviations from mean
    // TSS = M2 (for population) or M2 * n/(n-1) (for sample)
    this.state.yM2[outputIdx] += delta * delta2;
  }

  // ============================================================================
  // PUBLIC PREDICTION METHOD
  // ============================================================================

  /**
   * Makes predictions for future time steps or specific input points
   *
   * @param params - Prediction parameters
   * @returns Prediction results with confidence intervals
   * @throws Error if model is not initialized
   */
  public predict(params: PredictParams): PredictionResult {
    if (!this.isInitialized) {
      throw new Error("Model has not been trained. Call fitOnline first.");
    }

    const { futureSteps, inputPoints } = params;

    // Determine input points for prediction
    let pointsToPredict: number[][];

    if (inputPoints && inputPoints.length > 0) {
      // Validate provided input points
      for (const point of inputPoints) {
        if (point.length !== this.inputDimension) {
          throw new Error(
            `Input point dimension mismatch: expected ${this.inputDimension}, got ${point.length}`,
          );
        }
      }
      pointsToPredict = inputPoints;
    } else {
      // Generate extrapolated points based on last seen data
      if (futureSteps <= 0) {
        throw new Error("futureSteps must be positive");
      }
      pointsToPredict = this.generateExtrapolationPoints(futureSteps);
    }

    // Generate predictions for each point
    const predictions: SinglePrediction[] = [];

    for (const point of pointsToPredict) {
      predictions.push(this.predictSinglePoint(point));
    }

    // Calculate overall model statistics
    const rSquared = this.calculateRSquared();
    const rmse = this.calculateRMSE();

    return {
      predictions,
      confidenceLevel: this.config.confidenceLevel,
      rSquared,
      rmse,
      sampleCount: this.state.sampleCount,
      isModelReady: this.state.sampleCount >= this.polynomialFeatureCount * 2,
    };
  }

  /**
   * Generates extrapolation points for future prediction
   * Uses linear extrapolation based on observed data range
   *
   * @param futureSteps - Number of future steps to generate
   * @returns Array of input points for prediction
   */
  private generateExtrapolationPoints(futureSteps: number): number[][] {
    const points: number[][] = [];

    if (!this.state.lastInputPoint) {
      throw new Error("No data points have been processed yet");
    }

    const lastPoint = this.state.lastInputPoint;

    // Calculate step size from observed data range
    // Estimate as average spacing between samples
    const stepSize = new Array(this.inputDimension);
    for (let i = 0; i < this.inputDimension; i++) {
      if (this.normStats.count < 2) {
        stepSize[i] = 1;
      } else {
        const range = this.normStats.max[i] - this.normStats.min[i];
        // Use range / (n-1) as estimate of inter-sample spacing
        stepSize[i] = range / Math.max(1, this.normStats.count - 1);
        // Ensure non-zero step
        if (Math.abs(stepSize[i]) < 1e-10) {
          stepSize[i] = 1;
        }
      }
    }

    // Generate future points by linear extrapolation
    for (let step = 1; step <= futureSteps; step++) {
      const point = new Array(this.inputDimension);
      for (let i = 0; i < this.inputDimension; i++) {
        point[i] = lastPoint[i] + step * stepSize[i];
      }
      points.push(point);
    }

    return points;
  }

  /**
   * Predicts output for a single input point with confidence intervals
   *
   * FIXED: Confidence intervals now properly computed using:
   * 1. Correct residual variance estimation from weighted RSS
   * 2. Proper effective degrees of freedom accounting for forgetting factor
   * 3. Minimum variance floor to prevent zero intervals
   *
   * @param inputPoint - Input feature vector
   * @returns Prediction with confidence intervals
   */
  private predictSinglePoint(inputPoint: number[]): SinglePrediction {
    // Normalize input using same statistics as training
    const normalizedInput = this.normalizeInput(inputPoint);

    // Generate polynomial features (uses internal buffer)
    const phi = this.generatePolynomialFeatures(normalizedInput);

    // Initialize result arrays
    const predicted: number[] = new Array(this.outputDimension);
    const standardError: number[] = new Array(this.outputDimension);
    const lowerBound: number[] = new Array(this.outputDimension);
    const upperBound: number[] = new Array(this.outputDimension);

    // Compute prediction variance factor: φ' * P * φ
    // This represents uncertainty due to parameter estimation
    const P = this.state.covarianceMatrix;
    const numFeatures = this.polynomialFeatureCount;

    // P * φ
    let Pphi = new Array(numFeatures);
    for (let i = 0; i < numFeatures; i++) {
      let sum = 0;
      const row = P[i];
      for (let j = 0; j < numFeatures; j++) {
        sum += row[j] * phi[j];
      }
      Pphi[i] = sum;
    }

    // φ' * P * φ
    let varianceFactor = 0;
    for (let i = 0; i < numFeatures; i++) {
      varianceFactor += phi[i] * Pphi[i];
    }
    // Ensure non-negative (can become slightly negative due to numerical errors)
    varianceFactor = Math.max(0, varianceFactor);

    // Calculate effective degrees of freedom
    // For exponentially weighted data, effective n ≈ 1/(1-λ) when n is large
    const effectiveN = this.state.effectiveSampleCount;
    const effectiveDF = Math.max(1, effectiveN - this.polynomialFeatureCount);

    // Get t critical value for confidence intervals
    const tCritical = StatisticsUtils.tCriticalValue(
      this.config.confidenceLevel,
      effectiveDF,
    );

    // Compute predictions and confidence intervals for each output
    for (let j = 0; j < this.outputDimension; j++) {
      // Point prediction: φ' * w
      let pred = 0;
      for (let i = 0; i < numFeatures; i++) {
        pred += phi[i] * this.state.weights[i][j];
      }
      predicted[j] = pred;

      // Estimate residual variance (σ²)
      // FIXED: Use weighted RSS with effective sample count
      let residualVariance: number;

      if (effectiveN > this.polynomialFeatureCount) {
        // Use estimated residual variance from weighted sum
        residualVariance = this.state.weightedResidualSS[j] / effectiveDF;
      } else {
        // Not enough effective samples - use target variance as prior
        // FIXED: Compute from M2 accumulator
        const yVariance = this.state.sampleCount > 1
          ? this.state.yM2[j] / (this.state.sampleCount - 1)
          : 1;
        residualVariance = Math.max(yVariance, 1);
      }

      // FIXED: Ensure minimum residual variance to prevent zero confidence intervals
      // Use a fraction of the target variance as minimum
      const minVariance = this.state.sampleCount > 1
        ? this.state.yM2[j] / (this.state.sampleCount - 1) * 0.001
        : 0.001;
      residualVariance = Math.max(residualVariance, minVariance, 1e-10);

      // Standard error of prediction
      // Formula: se = sqrt(σ² * (1 + φ' * P * φ))
      // The "1" accounts for inherent noise, φ'Pφ for parameter uncertainty
      const predictionVariance = residualVariance * (1 + varianceFactor);
      const se = Math.sqrt(predictionVariance);

      // FIXED: Ensure standard error is never zero
      standardError[j] = Math.max(se, Math.sqrt(minVariance));

      // Confidence interval using t-distribution
      const margin = tCritical * standardError[j];
      lowerBound[j] = pred - margin;
      upperBound[j] = pred + margin;
    }

    return {
      predicted,
      lowerBound,
      upperBound,
      standardError,
    };
  }

  // ============================================================================
  // STATISTICS CALCULATION METHODS
  // ============================================================================

  /**
   * Calculates the coefficient of determination (R²)
   *
   * R² = 1 - RSS/TSS
   *
   * Where:
   * - RSS = Residual Sum of Squares (unexplained variance)
   * - TSS = Total Sum of Squares (total variance)
   *
   * FIXED: Uses unweighted RSS and proper TSS from Welford's M2
   *
   * @returns R² value in range [0, 1], where 1 indicates perfect fit
   */
  private calculateRSquared(): number {
    // Need at least 2 samples for meaningful R²
    if (this.state.sampleCount < 2) return 0;

    let totalRSS = 0;
    let totalTSS = 0;

    // Sum RSS and TSS across all output dimensions
    for (let j = 0; j < this.outputDimension; j++) {
      totalRSS += this.state.residualSumSquares[j];
      // TSS = M2 (sum of squared deviations from mean)
      totalTSS += this.state.yM2[j];
    }

    // Handle edge case: no variance in targets
    if (totalTSS < 1e-10) {
      // If TSS ≈ 0, targets are constant
      // R² = 1 if predictions are also constant (RSS ≈ 0)
      return totalRSS < 1e-10 ? 1 : 0;
    }

    // Standard R² formula
    const rSquared = 1 - totalRSS / totalTSS;

    // Clamp to [0, 1]
    // Negative R² can occur if model is worse than mean prediction
    return Math.max(0, Math.min(1, rSquared));
  }

  /**
   * Calculates the Root Mean Square Error
   *
   * RMSE = sqrt(RSS / n)
   *
   * @returns RMSE value (lower is better)
   */
  private calculateRMSE(): number {
    if (this.state.sampleCount === 0) return 0;

    // Calculate mean squared error across outputs
    let totalMSE = 0;
    for (let j = 0; j < this.outputDimension; j++) {
      totalMSE += this.state.residualSumSquares[j] / this.state.sampleCount;
    }

    // Average across output dimensions and take square root
    return Math.sqrt(totalMSE / this.outputDimension);
  }

  // ============================================================================
  // PUBLIC UTILITY METHODS
  // ============================================================================

  /**
   * Resets the model to its initial state
   * All learned parameters and statistics are cleared
   * Configuration is preserved
   */
  public reset(): void {
    this.state = this.createEmptyState();
    this.normStats = this.createEmptyNormStats();
    this.isInitialized = false;
    this.inputDimension = 0;
    this.outputDimension = 0;
    this.polynomialFeatureCount = 0;
    this.polynomialTermPatterns = new Int8Array(0);
    this.featureBuffer = new Float64Array(0);
    this.pPhiBuffer = new Float64Array(0);
    this.gainBuffer = new Float64Array(0);
  }

  /**
   * Returns a summary of the current model state
   * Useful for debugging and monitoring
   *
   * @returns Object containing model summary information
   */
  public getModelSummary(): {
    isInitialized: boolean;
    inputDimension: number;
    outputDimension: number;
    polynomialDegree: number;
    polynomialFeatureCount: number;
    sampleCount: number;
    rSquared: number;
    rmse: number;
    normalizationEnabled: boolean;
    normalizationMethod: NormalizationMethod;
  } {
    return {
      isInitialized: this.isInitialized,
      inputDimension: this.inputDimension,
      outputDimension: this.outputDimension,
      polynomialDegree: this.config.polynomialDegree,
      polynomialFeatureCount: this.polynomialFeatureCount,
      sampleCount: this.state.sampleCount,
      rSquared: this.calculateRSquared(),
      rmse: this.calculateRMSE(),
      normalizationEnabled: this.config.enableNormalization,
      normalizationMethod: this.config.normalizationMethod,
    };
  }

  /**
   * Gets the current model weights
   * Useful for model inspection, debugging, and serialization
   *
   * @returns Deep copy of the weight matrix
   */
  public getWeights(): number[][] {
    return MatrixOperations.clone(this.state.weights);
  }

  /**
   * Gets the current normalization statistics
   * Useful for understanding data distribution and debugging
   *
   * @returns Deep copy of normalization statistics
   */
  public getNormalizationStats(): NormalizationStats {
    return {
      min: this.normStats.min.slice(),
      max: this.normStats.max.slice(),
      mean: this.normStats.mean.slice(),
      // Return computed std instead of m2 for external interface compatibility
      m2: this.normStats.m2.slice(),
      count: this.normStats.count,
    };
  }
}
