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
 * FIXES APPLIED IN THIS VERSION:
 * 1. Converted recursive polynomial feature generation to iterative approach
 *    to prevent stack overflow for higher polynomial degrees
 * 2. Fixed R-squared calculation by properly tracking sum of squares
 * 3. Fixed confidence interval calculation to never return zero intervals
 * 4. Optimized matrix operations and memory usage
 * 5. Added proper numerical stability safeguards throughout
 *
 * @module MultivariatePolynomialRegression
 * @version 1.0.1
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
   * Initial covariance matrix diagonal value (default: 1000)
   * Higher values allow faster initial learning but may cause instability
   */
  initialCovariance?: number;

  /**
   * Regularization parameter to prevent numerical instability (default: 1e-6)
   * Added to covariance matrix diagonal after each update
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

  /** Running standard deviation for each feature (used for z-score normalization) */
  std: number[];

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
   * Sum of squared residuals for each output dimension
   * Used for RMSE and variance estimation
   */
  residualSumSquares: number[];

  /**
   * Sum of squared deviations from mean for each output (for R² calculation)
   * Represents total variance in the target values
   */
  totalSumSquares: number[];

  /** Running mean of y values for each output dimension */
  yMean: number[];

  /**
   * Running sum of squared y values for variance calculation
   * Used together with yMean to compute variance: E[y²] - E[y]²
   */
  ySumSquares: number[];

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
 * All operations create new arrays rather than modifying inputs (immutable style)
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
      // Create each row with zeros
      result[i] = new Array(n).fill(0);
      // Set the diagonal element to 1
      result[i][i] = 1;
    }
    return result;
  }

  /**
   * Creates a matrix filled with zeros
   *
   * @param rows - Number of rows
   * @param cols - Number of columns
   * @returns rows x cols matrix of zeros
   */
  static zeros(rows: number, cols: number): number[][] {
    // Using Array.from for cleaner initialization
    return Array.from({ length: rows }, () => new Array(cols).fill(0));
  }

  /**
   * Multiplies a matrix by a scalar value
   * Each element is multiplied by the scalar
   *
   * @param matrix - Input matrix
   * @param scalar - Scalar multiplier
   * @returns New matrix with scaled values
   */
  static scalarMultiply(matrix: number[][], scalar: number): number[][] {
    return matrix.map((row) => row.map((val) => val * scalar));
  }

  /**
   * Adds two matrices element-wise
   * Both matrices must have the same dimensions
   *
   * @param a - First matrix
   * @param b - Second matrix
   * @returns New matrix where result[i][j] = a[i][j] + b[i][j]
   */
  static add(a: number[][], b: number[][]): number[][] {
    return a.map((row, i) => row.map((val, j) => val + b[i][j]));
  }

  /**
   * Subtracts matrix b from matrix a element-wise
   * Both matrices must have the same dimensions
   *
   * @param a - Matrix to subtract from
   * @param b - Matrix to subtract
   * @returns New matrix where result[i][j] = a[i][j] - b[i][j]
   */
  static subtract(a: number[][], b: number[][]): number[][] {
    return a.map((row, i) => row.map((val, j) => val - b[i][j]));
  }

  /**
   * Multiplies two matrices using standard matrix multiplication
   * Number of columns in a must equal number of rows in b
   *
   * Time complexity: O(n * m * p) where a is n x m and b is m x p
   *
   * @param a - Left matrix (n x m)
   * @param b - Right matrix (m x p)
   * @returns Result matrix (n x p)
   */
  static multiply(a: number[][], b: number[][]): number[][] {
    const rowsA = a.length;
    const colsA = a[0].length;
    const colsB = b[0].length;

    // Pre-allocate result matrix
    const result = this.zeros(rowsA, colsB);

    // Standard triple-nested loop for matrix multiplication
    for (let i = 0; i < rowsA; i++) {
      for (let j = 0; j < colsB; j++) {
        let sum = 0;
        for (let k = 0; k < colsA; k++) {
          sum += a[i][k] * b[k][j];
        }
        result[i][j] = sum;
      }
    }

    return result;
  }

  /**
   * Multiplies a matrix by a column vector
   * The vector length must equal the number of columns in the matrix
   *
   * @param matrix - Input matrix (n x m)
   * @param vector - Input vector (length m)
   * @returns Result vector (length n) where result[i] = sum(matrix[i][j] * vector[j])
   */
  static multiplyVector(matrix: number[][], vector: number[]): number[] {
    // Use iterative approach for better performance with large matrices
    const result = new Array(matrix.length);
    for (let i = 0; i < matrix.length; i++) {
      let sum = 0;
      const row = matrix[i];
      for (let j = 0; j < row.length; j++) {
        sum += row[j] * vector[j];
      }
      result[i] = sum;
    }
    return result;
  }

  /**
   * Computes the outer product of two vectors
   * Result is a matrix where result[i][j] = a[i] * b[j]
   *
   * @param a - First vector (length n)
   * @param b - Second vector (length m)
   * @returns Outer product matrix (n x m)
   */
  static outerProduct(a: number[], b: number[]): number[][] {
    // Pre-allocate for better performance
    const result = new Array(a.length);
    for (let i = 0; i < a.length; i++) {
      result[i] = new Array(b.length);
      const ai = a[i];
      for (let j = 0; j < b.length; j++) {
        result[i][j] = ai * b[j];
      }
    }
    return result;
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
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      sum += a[i] * b[i];
    }
    return sum;
  }

  /**
   * Transposes a matrix (swaps rows and columns)
   *
   * @param matrix - Input matrix (n x m)
   * @returns Transposed matrix (m x n)
   */
  static transpose(matrix: number[][]): number[][] {
    if (matrix.length === 0) return [];
    const rows = matrix.length;
    const cols = matrix[0].length;
    const result = new Array(cols);
    for (let j = 0; j < cols; j++) {
      result[j] = new Array(rows);
      for (let i = 0; i < rows; i++) {
        result[j][i] = matrix[i][j];
      }
    }
    return result;
  }

  /**
   * Creates a deep copy of a matrix
   *
   * @param matrix - Input matrix
   * @returns New matrix with copied values
   */
  static clone(matrix: number[][]): number[][] {
    return matrix.map((row) => [...row]);
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

    // For large degrees of freedom, t-distribution approaches normal
    if (degreesOfFreedom > 30) {
      // Use standard normal approximation
      return this.normalInverseCDF(1 - alpha / 2);
    }

    // For smaller df, use Cornish-Fisher expansion for better accuracy
    // This provides a correction to the normal approximation
    const zAlpha = this.normalInverseCDF(1 - alpha / 2);

    // Cornish-Fisher correction terms
    const g1 = (zAlpha ** 3 + zAlpha) / 4;
    const g2 = (5 * zAlpha ** 5 + 16 * zAlpha ** 3 + 3 * zAlpha) / 96;

    return zAlpha + g1 / degreesOfFreedom + g2 / (degreesOfFreedom ** 2);
  }

  /**
   * Approximates the inverse CDF of the standard normal distribution
   * Uses the Abramowitz and Stegun rational approximation (formula 26.2.23)
   *
   * This is highly accurate for p in [0.0000001, 0.9999999]
   *
   * @param p - Probability (must be between 0 and 1 exclusive)
   * @returns z-value such that P(Z < z) = p for standard normal Z
   * @throws Error if p is not in valid range
   */
  static normalInverseCDF(p: number): number {
    if (p <= 0 || p >= 1) {
      throw new Error("Probability must be between 0 and 1 (exclusive)");
    }

    // Coefficients for the rational approximation
    // These provide high accuracy across the full range
    const a1 = -3.969683028665376e+01;
    const a2 = 2.209460984245205e+02;
    const a3 = -2.759285104469687e+02;
    const a4 = 1.383577518672690e+02;
    const a5 = -3.066479806614716e+01;
    const a6 = 2.506628277459239e+00;

    const b1 = -5.447609879822406e+01;
    const b2 = 1.615858368580409e+02;
    const b3 = -1.556989798598866e+02;
    const b4 = 6.680131188771972e+01;
    const b5 = -1.328068155288572e+01;

    const c1 = -7.784894002430293e-03;
    const c2 = -3.223964580411365e-01;
    const c3 = -2.400758277161838e+00;
    const c4 = -2.549732539343734e+00;
    const c5 = 4.374664141464968e+00;
    const c6 = 2.938163982698783e+00;

    const d1 = 7.784695709041462e-03;
    const d2 = 3.224671290700398e-01;
    const d3 = 2.445134137142996e+00;
    const d4 = 3.754408661907416e+00;

    // Threshold values for switching between approximation regions
    const pLow = 0.02425;
    const pHigh = 1 - pLow;

    let q: number, r: number;

    if (p < pLow) {
      // Lower tail approximation
      q = Math.sqrt(-2 * Math.log(p));
      return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
        ((((d1 * q + d2) * q + d3) * q + d4) * q + 1);
    } else if (p <= pHigh) {
      // Central region approximation (most common case)
      q = p - 0.5;
      r = q * q;
      return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
        (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1);
    } else {
      // Upper tail approximation (symmetric to lower tail)
      q = Math.sqrt(-2 * Math.log(1 - p));
      return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
        ((((d1 * q + d2) * q + d3) * q + d4) * q + 1);
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
   * For degree d and n inputs, this is sum of C(n+k-1, k) for k=0 to d
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
   */
  private polynomialTermPatterns: number[][] = [];

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
    this.config = {
      polynomialDegree: config.polynomialDegree ?? 2,
      enableNormalization: config.enableNormalization ?? true,
      normalizationMethod: config.normalizationMethod ?? "min-max",
      forgettingFactor: config.forgettingFactor ?? 0.99,
      initialCovariance: config.initialCovariance ?? 1000,
      regularization: config.regularization ?? 1e-6,
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
      totalSumSquares: [],
      yMean: [],
      ySumSquares: [],
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
      std: [],
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

    // Pre-compute polynomial term patterns (ITERATIVE - prevents stack overflow)
    // This replaces the recursive approach that caused issues
    this.polynomialTermPatterns = this.generatePolynomialTermPatterns(inputDim);
    this.polynomialFeatureCount = this.polynomialTermPatterns.length;

    // Initialize weights matrix to zeros
    // Shape: [polynomialFeatureCount][outputDim]
    this.state.weights = MatrixOperations.zeros(
      this.polynomialFeatureCount,
      outputDim,
    );

    // Initialize covariance matrix as scaled identity matrix
    // Large initial values allow the model to adapt quickly to early data
    this.state.covarianceMatrix = MatrixOperations.scalarMultiply(
      MatrixOperations.identity(this.polynomialFeatureCount),
      this.config.initialCovariance,
    );

    // Initialize output statistics arrays
    this.state.residualSumSquares = new Array(outputDim).fill(0);
    this.state.totalSumSquares = new Array(outputDim).fill(0);
    this.state.yMean = new Array(outputDim).fill(0);
    this.state.ySumSquares = new Array(outputDim).fill(0);

    // Initialize normalization statistics arrays
    this.normStats.min = new Array(inputDim).fill(Infinity);
    this.normStats.max = new Array(inputDim).fill(-Infinity);
    this.normStats.mean = new Array(inputDim).fill(0);
    this.normStats.std = new Array(inputDim).fill(1); // Default to 1 to avoid division by zero

    this.isInitialized = true;
  }

  // ============================================================================
  // POLYNOMIAL FEATURE GENERATION (ITERATIVE - FIX FOR STACK OVERFLOW)
  // ============================================================================

  /**
   * Generates all polynomial term patterns up to the configured degree
   *
   * IMPORTANT: This is an ITERATIVE implementation that replaces the original
   * recursive approach. The recursive version caused stack overflows for
   * higher degrees because each polynomial term required nested function calls.
   *
   * Each pattern is an array of exponents, one per input variable.
   * For example, with 2 inputs (x, y) and degree 2:
   * - [0, 0] represents the constant term (1)
   * - [1, 0] represents x
   * - [0, 1] represents y
   * - [2, 0] represents x²
   * - [1, 1] represents xy
   * - [0, 2] represents y²
   *
   * @param inputDim - Number of input features
   * @returns Array of exponent patterns for all polynomial terms
   */
  private generatePolynomialTermPatterns(inputDim: number): number[][] {
    const patterns: number[][] = [];
    const degree = this.config.polynomialDegree;

    // Start with the constant term (all exponents = 0)
    patterns.push(new Array(inputDim).fill(0));

    // Generate terms for each degree from 1 to max degree
    // Using an iterative approach with a queue-like pattern
    for (let d = 1; d <= degree; d++) {
      // Generate all combinations of exponents that sum to exactly d
      // This uses an iterative "stars and bars" enumeration
      this.generateTermsOfDegreeIterative(inputDim, d, patterns);
    }

    return patterns;
  }

  /**
   * Generates all polynomial terms of a specific degree iteratively
   *
   * Uses a systematic enumeration approach instead of recursion:
   * - Start with all exponent on first variable [d, 0, 0, ...]
   * - Iteratively shift exponents to later variables
   * - Continue until all exponent is on last variable [0, 0, ..., d]
   *
   * @param numVars - Number of input variables
   * @param targetDegree - Target degree (sum of all exponents)
   * @param patterns - Array to append generated patterns to
   */
  private generateTermsOfDegreeIterative(
    numVars: number,
    targetDegree: number,
    patterns: number[][],
  ): void {
    // Special case: single variable
    if (numVars === 1) {
      patterns.push([targetDegree]);
      return;
    }

    // Use iterative approach with explicit stack to avoid recursion
    // Stack entries: [current exponents array, current position, remaining degree]
    const stack: Array<
      { exponents: number[]; pos: number; remaining: number }
    > = [];

    // Initialize: start building from position 0 with full degree remaining
    stack.push({
      exponents: new Array(numVars).fill(0),
      pos: 0,
      remaining: targetDegree,
    });

    while (stack.length > 0) {
      const current = stack.pop()!;
      const { exponents, pos, remaining } = current;

      // If we're at the last position, assign all remaining degree
      if (pos === numVars - 1) {
        const finalExponents = [...exponents];
        finalExponents[pos] = remaining;
        patterns.push(finalExponents);
        continue;
      }

      // Try each possible exponent value for current position
      // Go from highest to lowest to maintain consistent ordering
      for (let exp = remaining; exp >= 0; exp--) {
        const newExponents = [...exponents];
        newExponents[pos] = exp;
        stack.push({
          exponents: newExponents,
          pos: pos + 1,
          remaining: remaining - exp,
        });
      }
    }
  }

  /**
   * Generates polynomial features from an input vector using pre-computed patterns
   *
   * This is called for each input during training and prediction.
   * Uses the cached term patterns for efficiency.
   *
   * @param input - Input feature vector (normalized)
   * @returns Polynomial feature vector
   */
  private generatePolynomialFeatures(input: number[]): number[] {
    const features = new Array(this.polynomialTermPatterns.length);

    // Compute each polynomial term using the pre-computed patterns
    for (let i = 0; i < this.polynomialTermPatterns.length; i++) {
      const pattern = this.polynomialTermPatterns[i];
      let term = 1;

      // Multiply input values raised to their respective exponents
      for (let j = 0; j < pattern.length; j++) {
        if (pattern[j] !== 0) {
          // Use Math.pow for the exponentiation
          // Optimization: special case for common exponents
          if (pattern[j] === 1) {
            term *= input[j];
          } else if (pattern[j] === 2) {
            term *= input[j] * input[j];
          } else {
            term *= Math.pow(input[j], pattern[j]);
          }
        }
      }

      features[i] = term;
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

        // Welford's online algorithm for mean and variance
        // This is numerically stable and works incrementally
        const delta = value - this.normStats.mean[i];
        this.normStats.mean[i] += delta / n;

        // Update variance using the corrected two-pass-like formula
        if (n > 1) {
          const delta2 = value - this.normStats.mean[i];
          // M2 accumulates the sum of squared deviations
          // We store std but compute it from running variance
          const prevVariance = this.normStats.std[i] * this.normStats.std[i] *
            (n - 2);
          const newVariance = (prevVariance + delta * delta2) / (n - 1);
          this.normStats.std[i] = Math.sqrt(Math.max(0, newVariance));
        }
      }
    }
  }

  /**
   * Normalizes an input vector based on current statistics
   * Applied before polynomial feature generation
   *
   * @param input - Raw input vector
   * @returns Normalized input vector
   */
  private normalizeInput(input: number[]): number[] {
    // Skip normalization if disabled or insufficient data
    if (!this.config.enableNormalization || this.normStats.count < 2) {
      return [...input]; // Return copy to avoid mutation
    }

    const normalized = new Array(input.length);

    switch (this.config.normalizationMethod) {
      case "min-max":
        // Scale to [0, 1] based on observed range
        for (let i = 0; i < input.length; i++) {
          const range = this.normStats.max[i] - this.normStats.min[i];
          if (range < 1e-10) {
            // Avoid division by zero for constant features
            normalized[i] = 0.5;
          } else {
            normalized[i] = (input[i] - this.normStats.min[i]) / range;
          }
        }
        break;

      case "z-score":
        // Standardize to mean 0, std 1
        for (let i = 0; i < input.length; i++) {
          if (this.normStats.std[i] < 1e-10) {
            // Avoid division by zero for constant features
            normalized[i] = 0;
          } else {
            normalized[i] = (input[i] - this.normStats.mean[i]) /
              this.normStats.std[i];
          }
        }
        break;

      default:
        // No normalization
        return [...input];
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
    // This ensures normalization params are current
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
    // Check for empty inputs
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

    // Check dimension consistency within the batch
    for (let i = 0; i < xCoordinates.length; i++) {
      if (xCoordinates[i].length !== xDim) {
        throw new Error(`Inconsistent x dimension at index ${i}`);
      }
      if (yCoordinates[i].length !== yDim) {
        throw new Error(`Inconsistent y dimension at index ${i}`);
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

    // P * φ - matrix-vector product
    const Pphi = MatrixOperations.multiplyVector(P, phi);

    // Denominator: λ + φ' * P * φ (scalar)
    const phiTPphi = MatrixOperations.dotProduct(phi, Pphi);
    const denominator = lambda + phiTPphi;

    // Gain vector: k = P * φ / denominator
    // This determines how much to adjust weights based on prediction error
    const k = new Array(this.polynomialFeatureCount);
    for (let i = 0; i < this.polynomialFeatureCount; i++) {
      k[i] = Pphi[i] / denominator;
    }

    // Step 3: Update weights for each output dimension
    for (let j = 0; j < this.outputDimension; j++) {
      // Current prediction with existing weights
      let prediction = 0;
      for (let i = 0; i < this.polynomialFeatureCount; i++) {
        prediction += phi[i] * this.state.weights[i][j];
      }

      // Prediction error
      const error = y[j] - prediction;

      // Update weights: w = w + k * error
      for (let i = 0; i < this.polynomialFeatureCount; i++) {
        this.state.weights[i][j] += k[i] * error;
      }

      // Update error statistics for this output dimension
      this.updateErrorStatistics(j, y[j], prediction);
    }

    // Step 4: Update covariance matrix
    // P_new = (P - k * φ' * P) / λ
    // This reduces uncertainty in weight estimates as we see more data
    const newP = MatrixOperations.zeros(
      this.polynomialFeatureCount,
      this.polynomialFeatureCount,
    );

    for (let i = 0; i < this.polynomialFeatureCount; i++) {
      for (let j = 0; j < this.polynomialFeatureCount; j++) {
        // P[i][j] - k[i] * Pphi[j] is the update
        // Then divide by lambda
        newP[i][j] = (P[i][j] - k[i] * Pphi[j]) / lambda;
      }
      // Add regularization to diagonal for numerical stability
      // This prevents the covariance matrix from becoming singular
      newP[i][i] += this.config.regularization;
    }

    this.state.covarianceMatrix = newP;

    // Step 5: Update tracking variables
    this.state.lastInputPoint = [...x];
    this.state.timeIndex++;
    this.state.sampleCount++;
  }

  /**
   * Updates error statistics for R² and RMSE calculation
   *
   * FIX: Uses proper incremental calculation for both RSS and TSS
   * The previous implementation had inconsistent calculations that could
   * lead to R² = 0 even for well-fitting models.
   *
   * @param outputIdx - Index of the output dimension
   * @param actual - Actual y value
   * @param predicted - Predicted y value (with weights before update)
   */
  private updateErrorStatistics(
    outputIdx: number,
    actual: number,
    predicted: number,
  ): void {
    const n = this.state.sampleCount + 1;
    const lambda = this.config.forgettingFactor;

    // Update running mean of y using Welford's algorithm
    const prevMean = this.state.yMean[outputIdx];
    const newMean = prevMean + (actual - prevMean) / n;
    this.state.yMean[outputIdx] = newMean;

    // Update sum of y² for variance calculation
    this.state.ySumSquares[outputIdx] += actual * actual;

    // Update residual sum of squares (RSS) with exponential weighting
    // RSS tracks how well the model fits the data
    const residual = actual - predicted;
    this.state.residualSumSquares[outputIdx] =
      lambda * this.state.residualSumSquares[outputIdx] + residual * residual;

    // Update total sum of squares (TSS) with exponential weighting
    // TSS tracks the total variance in the target variable
    // FIX: Use deviation from CURRENT mean for consistency
    // Also start tracking from first sample, not second
    const deviation = actual - newMean;
    this.state.totalSumSquares[outputIdx] =
      lambda * this.state.totalSumSquares[outputIdx] + deviation * deviation;

    // FIX: Ensure TSS is never less than a minimum threshold
    // This prevents division by zero and R² = 0 for nearly constant data
    const minTSS = 1e-10;
    if (this.state.totalSumSquares[outputIdx] < minTSS && n > 1) {
      // For very low variance data, estimate TSS from sample variance
      const variance = this.state.ySumSquares[outputIdx] / n -
        newMean * newMean;
      this.state.totalSumSquares[outputIdx] = Math.max(minTSS, variance * n);
    }
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
      // Use provided input points directly
      pointsToPredict = inputPoints;
    } else {
      // Generate extrapolated points based on last seen data
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
      // Model is ready when we have enough samples relative to parameters
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
    // This estimates the typical spacing between data points
    const stepSize = lastPoint.map((val, i) => {
      if (this.normStats.count < 2) return 1;
      const range = this.normStats.max[i] - this.normStats.min[i];
      // Estimate step as average spacing
      return range / Math.max(1, this.normStats.count - 1);
    });

    // Generate future points by linear extrapolation
    for (let step = 1; step <= futureSteps; step++) {
      const point = lastPoint.map((val, i) => val + step * stepSize[i]);
      points.push(point);
    }

    return points;
  }

  /**
   * Predicts output for a single input point with confidence intervals
   *
   * @param inputPoint - Input feature vector
   * @returns Prediction with confidence intervals
   */
  private predictSinglePoint(inputPoint: number[]): SinglePrediction {
    // Normalize input using same statistics as training
    const normalizedInput = this.normalizeInput(inputPoint);

    // Generate polynomial features
    const phi = this.generatePolynomialFeatures(normalizedInput);

    // Initialize result arrays
    const predicted: number[] = new Array(this.outputDimension);
    const standardError: number[] = new Array(this.outputDimension);
    const lowerBound: number[] = new Array(this.outputDimension);
    const upperBound: number[] = new Array(this.outputDimension);

    // Compute prediction variance: φ' * P * φ
    // This represents uncertainty due to parameter estimation
    const Pphi = MatrixOperations.multiplyVector(
      this.state.covarianceMatrix,
      phi,
    );
    const varianceMultiplier = MatrixOperations.dotProduct(phi, Pphi);

    // Get t critical value for confidence intervals
    const df = Math.max(
      1,
      this.state.sampleCount - this.polynomialFeatureCount,
    );
    const tCritical = StatisticsUtils.tCriticalValue(
      this.config.confidenceLevel,
      df,
    );

    // Compute predictions and confidence intervals for each output
    for (let j = 0; j < this.outputDimension; j++) {
      // Point prediction: φ' * w
      let pred = 0;
      for (let i = 0; i < this.polynomialFeatureCount; i++) {
        pred += phi[i] * this.state.weights[i][j];
      }
      predicted[j] = pred;

      // Estimate residual variance (mean squared error)
      // FIX: Use proper variance estimation that's never zero
      let residualVariance: number;
      if (this.state.sampleCount > this.polynomialFeatureCount) {
        // Use estimated residual variance
        residualVariance = this.state.residualSumSquares[j] / df;
      } else {
        // Not enough samples - use prior estimate based on target variance
        // FIX: Estimate from observed y variance instead of using fixed 1
        const yVariance = this.state.sampleCount > 1
          ? this.state.totalSumSquares[j] / (this.state.sampleCount - 1)
          : 1;
        residualVariance = Math.max(yVariance, 1e-6);
      }

      // Ensure residual variance is never zero (prevents zero confidence intervals)
      // FIX: Add minimum bound to prevent confidence of 0
      residualVariance = Math.max(residualVariance, 1e-10);

      // Standard error of prediction
      // Includes both parameter uncertainty and inherent noise
      // Formula: se = sqrt(σ² * (1 + φ' * P * φ))
      const predictionVariance = residualVariance *
        (1 + Math.max(0, varianceMultiplier));
      const se = Math.sqrt(predictionVariance);
      standardError[j] = se;

      // Confidence interval using t-distribution
      const margin = tCritical * se;
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
   * FIX: Improved calculation that properly handles edge cases
   * and uses consistent weighting for RSS and TSS
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
      totalTSS += this.state.totalSumSquares[j];
    }

    // Handle edge case: no variance in targets
    if (totalTSS < 1e-10) {
      // If TSS is essentially 0, targets are constant
      // R² = 1 if predictions are also constant (RSS ≈ 0)
      return totalRSS < 1e-10 ? 1 : 0;
    }

    // Standard R² formula
    const rSquared = 1 - totalRSS / totalTSS;

    // Clamp to [0, 1]
    // Negative R² can occur if model is worse than mean prediction
    // but we report 0 as the minimum for interpretability
    return Math.max(0, Math.min(1, rSquared));
  }

  /**
   * Calculates the Root Mean Square Error
   *
   * For exponentially weighted RLS, this represents the recent error level
   * rather than the historical average
   *
   * @returns RMSE value (lower is better)
   */
  private calculateRMSE(): number {
    if (this.state.sampleCount === 0) return 0;

    // For exponentially weighted sum, effective sample count is bounded
    // by 1 / (1 - λ) as the series converges
    const effectiveSamples = Math.min(
      this.state.sampleCount,
      1 / (1 - this.config.forgettingFactor + 1e-10),
    );

    // Calculate mean squared error across outputs
    let totalMSE = 0;
    for (let j = 0; j < this.outputDimension; j++) {
      totalMSE += this.state.residualSumSquares[j] / effectiveSamples;
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
    this.polynomialTermPatterns = [];
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
      min: [...this.normStats.min],
      max: [...this.normStats.max],
      mean: [...this.normStats.mean],
      std: [...this.normStats.std],
      count: this.normStats.count,
    };
  }
}
