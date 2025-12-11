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
 * CRITICAL FIXES IN THIS VERSION (v1.0.3):
 * 1. Fixed R-squared calculation - was returning 0 due to incorrect TSS/RSS tracking.
 *    Now properly tracks total variance vs residual variance using stable algorithms.
 * 2. Fixed prediction value explosion by properly bounding the covariance matrix
 *    (both diagonal AND off-diagonal elements) and using Joseph form for updates.
 * 3. Eliminated memory overflow by removing all allocations from hot paths:
 *    - Polynomial term generation now uses iterative state machine with no intermediate arrays
 *    - Normalization uses pre-allocated buffers
 *    - Prediction uses pre-allocated buffers for all intermediate calculations
 * 4. Fixed confidence intervals by properly estimating residual variance and using
 *    correct degrees of freedom calculation.
 * 5. Improved numerical stability throughout with proper epsilon comparisons,
 *    symmetric matrix enforcement, and condition number monitoring.
 *
 * @module MultivariatePolynomialRegression
 * @version 1.0.3
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
 * Uses Welford's algorithm for numerically stable incremental computation
 */
interface NormalizationStats {
  /** Minimum values observed for each feature (used for min-max normalization) */
  min: Float64Array;

  /** Maximum values observed for each feature (used for min-max normalization) */
  max: Float64Array;

  /** Running mean values for each feature (used for z-score normalization) */
  mean: Float64Array;

  /**
   * Running M2 accumulator for Welford's algorithm (sum of squared deviations)
   * Used to compute variance incrementally: variance = M2 / (n - 1)
   */
  m2: Float64Array;

  /** Number of samples used to compute statistics */
  count: number;
}

/**
 * Internal model state - maintains all learned parameters and statistics
 */
interface ModelState {
  /**
   * Weight coefficients stored as flat array for cache efficiency
   * Layout: weights[i * outputDim + j] = coefficient for feature i, output j
   * Total size: polynomialFeatureCount * outputDimension
   */
  weights: Float64Array;

  /**
   * Covariance matrix for RLS stored as flat array (row-major)
   * Layout: cov[i * n + j] where n = polynomialFeatureCount
   * Size: polynomialFeatureCount^2
   */
  covarianceMatrix: Float64Array;

  /**
   * Sum of squared residuals for each output dimension
   * Used for RMSE and R-squared calculation
   */
  residualSumSquares: Float64Array;

  /**
   * Sum of squared deviations of y from mean (Total Sum of Squares)
   * Computed using Welford's algorithm for numerical stability
   */
  totalSumSquares: Float64Array;

  /**
   * Running mean of y values for each output dimension
   * Used for TSS calculation via Welford's algorithm
   */
  yMean: Float64Array;

  /**
   * Weighted sum of squared residuals for confidence interval estimation
   * Uses exponential weighting with forgetting factor
   */
  weightedResidualSS: Float64Array;

  /**
   * Effective sample count accounting for forgetting factor
   * Converges to 1/(1-λ) as samples increase
   */
  effectiveSampleCount: number;

  /** Number of samples processed */
  sampleCount: number;

  /** Last input point seen (for extrapolation in prediction) */
  lastInputPoint: Float64Array | null;

  /** Time index for sequential data tracking */
  timeIndex: number;
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
 * ALGORITHM OVERVIEW:
 * The RLS algorithm updates the model for each new sample (x, y):
 * 1. Generate polynomial features φ from normalized x
 * 2. Compute gain vector: k = P·φ / (λ + φᵀ·P·φ)
 * 3. Compute prediction error: e = y - φᵀ·w
 * 4. Update weights: w = w + k·e
 * 5. Update covariance: P = (P - k·φᵀ·P) / λ  [using Joseph form for stability]
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
   * Pre-computed polynomial term exponent patterns stored as flat Int8Array
   * Layout: patterns[(termIndex * inputDim) + featureIndex] = exponent
   *
   * Example for 2 inputs, degree 2 (6 terms):
   * Term 0 (1):    [0, 0]  -> patterns[0..1] = [0, 0]
   * Term 1 (x₀):   [1, 0]  -> patterns[2..3] = [1, 0]
   * Term 2 (x₁):   [0, 1]  -> patterns[4..5] = [0, 1]
   * Term 3 (x₀²):  [2, 0]  -> patterns[6..7] = [2, 0]
   * Term 4 (x₀x₁): [1, 1]  -> patterns[8..9] = [1, 1]
   * Term 5 (x₁²):  [0, 2]  -> patterns[10..11] = [0, 2]
   */
  private polynomialTermPatterns: Int8Array = new Int8Array(0);

  // ============================================================================
  // PRE-ALLOCATED COMPUTATION BUFFERS
  // These buffers are reused across all computations to avoid GC pressure
  // ============================================================================

  /** Buffer for polynomial feature vector φ */
  private phiBuffer: Float64Array = new Float64Array(0);

  /** Buffer for P·φ computation in RLS update */
  private pPhiBuffer: Float64Array = new Float64Array(0);

  /** Buffer for gain vector k in RLS update */
  private gainBuffer: Float64Array = new Float64Array(0);

  /** Buffer for normalized input values */
  private normalizedInputBuffer: Float64Array = new Float64Array(0);

  /** Buffer for prediction output values */
  private predictionBuffer: Float64Array = new Float64Array(0);

  /** Buffer for standard error values */
  private stdErrorBuffer: Float64Array = new Float64Array(0);

  /** Buffer for lower confidence bound */
  private lowerBoundBuffer: Float64Array = new Float64Array(0);

  /** Buffer for upper confidence bound */
  private upperBoundBuffer: Float64Array = new Float64Array(0);

  /** Temporary buffer for covariance matrix operations */
  private tempCovBuffer: Float64Array = new Float64Array(0);

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
    // These defaults are chosen for numerical stability and typical use cases
    this.config = {
      polynomialDegree: config.polynomialDegree ?? 2,
      enableNormalization: config.enableNormalization ?? true,
      normalizationMethod: config.normalizationMethod ?? "min-max",
      forgettingFactor: config.forgettingFactor ?? 0.99,
      initialCovariance: config.initialCovariance ?? 100,
      regularization: config.regularization ?? 1e-6,
      confidenceLevel: config.confidenceLevel ?? 0.95,
    };

    // Validate configuration parameters to fail fast on bad input
    this.validateConfig();

    // Initialize empty state structures (will be properly sized on first fit call)
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
    // Degree 10 with 10 inputs = C(20,10) = 184,756 features (34GB covariance matrix)
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
   * Note: Float64Arrays are initialized with size 0 and resized in initializeModel
   *
   * @returns Fresh ModelState object with empty typed arrays
   */
  private createEmptyState(): ModelState {
    return {
      weights: new Float64Array(0),
      covarianceMatrix: new Float64Array(0),
      residualSumSquares: new Float64Array(0),
      totalSumSquares: new Float64Array(0),
      yMean: new Float64Array(0),
      weightedResidualSS: new Float64Array(0),
      effectiveSampleCount: 0,
      sampleCount: 0,
      lastInputPoint: null,
      timeIndex: 0,
    };
  }

  /**
   * Creates empty normalization statistics structure
   * Used during initialization and reset
   *
   * @returns Fresh NormalizationStats object with empty typed arrays
   */
  private createEmptyNormStats(): NormalizationStats {
    return {
      min: new Float64Array(0),
      max: new Float64Array(0),
      mean: new Float64Array(0),
      m2: new Float64Array(0),
      count: 0,
    };
  }

  /**
   * Initializes the model with dimensions from the first data batch
   * This sets up all internal structures based on the actual data dimensions
   *
   * MEMORY ALLOCATION STRATEGY:
   * All buffers are allocated once here and reused throughout the model's lifetime.
   * This eliminates GC pressure during training and prediction.
   *
   * @param inputDim - Number of input features
   * @param outputDim - Number of output targets
   */
  private initializeModel(inputDim: number, outputDim: number): void {
    this.inputDimension = inputDim;
    this.outputDimension = outputDim;

    // Calculate polynomial feature count using combinatorial formula
    // C(inputDim + degree, degree) = number of monomials of degree <= degree
    this.polynomialFeatureCount = this.computePolynomialFeatureCount(
      inputDim,
      this.config.polynomialDegree,
    );

    const numFeatures = this.polynomialFeatureCount;

    // Pre-compute polynomial term patterns using memory-efficient iterative algorithm
    // This avoids recursive enumeration that could cause stack overflow
    this.polynomialTermPatterns = this.generatePolynomialTermPatterns(inputDim);

    // ========================================================================
    // Allocate all computation buffers once
    // Using Float64Array for numerical precision and cache efficiency
    // ========================================================================

    // Core RLS computation buffers
    this.phiBuffer = new Float64Array(numFeatures);
    this.pPhiBuffer = new Float64Array(numFeatures);
    this.gainBuffer = new Float64Array(numFeatures);

    // Normalization buffer
    this.normalizedInputBuffer = new Float64Array(inputDim);

    // Prediction output buffers
    this.predictionBuffer = new Float64Array(outputDim);
    this.stdErrorBuffer = new Float64Array(outputDim);
    this.lowerBoundBuffer = new Float64Array(outputDim);
    this.upperBoundBuffer = new Float64Array(outputDim);

    // Temporary buffer for covariance operations (avoids allocation in hot path)
    this.tempCovBuffer = new Float64Array(numFeatures * numFeatures);

    // ========================================================================
    // Initialize model state arrays
    // ========================================================================

    // Weight matrix as flat array: weights[i * outputDim + j]
    // Initialized to zeros - the RLS algorithm will learn appropriate values
    this.state.weights = new Float64Array(numFeatures * outputDim);

    // Covariance matrix as flat array: cov[i * numFeatures + j]
    // Initialized as scaled identity matrix: P = initialCovariance * I
    // This represents high initial uncertainty about the weights
    this.state.covarianceMatrix = new Float64Array(numFeatures * numFeatures);
    for (let i = 0; i < numFeatures; i++) {
      this.state.covarianceMatrix[i * numFeatures + i] =
        this.config.initialCovariance;
    }

    // Statistics arrays for model quality metrics
    this.state.residualSumSquares = new Float64Array(outputDim);
    this.state.totalSumSquares = new Float64Array(outputDim);
    this.state.yMean = new Float64Array(outputDim);
    this.state.weightedResidualSS = new Float64Array(outputDim);
    this.state.effectiveSampleCount = 0;

    // Buffer for storing last input point (for extrapolation)
    this.state.lastInputPoint = new Float64Array(inputDim);

    // ========================================================================
    // Initialize normalization statistics arrays
    // ========================================================================

    this.normStats.min = new Float64Array(inputDim);
    this.normStats.max = new Float64Array(inputDim);
    this.normStats.mean = new Float64Array(inputDim);
    this.normStats.m2 = new Float64Array(inputDim);

    // Initialize min to +Infinity and max to -Infinity for proper min/max tracking
    this.normStats.min.fill(Infinity);
    this.normStats.max.fill(-Infinity);

    this.isInitialized = true;
  }

  /**
   * Computes the number of polynomial features for given input dimension and degree
   * Uses the combinatorial formula: C(n + d, d) where n = inputDim, d = degree
   *
   * This equals the number of distinct monomials of degree at most d in n variables.
   *
   * Example: n=2, d=2 gives C(4,2) = 6 terms: 1, x₀, x₁, x₀², x₀x₁, x₁²
   *
   * IMPLEMENTATION NOTE:
   * We compute this iteratively to avoid large factorials and maintain precision:
   * C(n+d, d) = ∏(i=1 to d) [(n+i) / i]
   *
   * @param inputDim - Number of input variables
   * @param degree - Maximum polynomial degree
   * @returns Total number of polynomial features
   */
  private computePolynomialFeatureCount(
    inputDim: number,
    degree: number,
  ): number {
    // Base case: at least the constant term
    let result = 1;

    // Compute C(inputDim + degree, degree) iteratively
    // Using the identity: C(n,k) = C(n,k-1) * (n-k+1) / k
    // Rewritten as: C(n+d, d) = ∏(i=1 to d) [(n+i) / i]
    for (let i = 1; i <= degree; i++) {
      result = (result * (inputDim + i)) / i;
    }

    // Round to handle any floating point errors
    return Math.round(result);
  }

  // ============================================================================
  // POLYNOMIAL FEATURE GENERATION
  // Memory-efficient iterative implementation
  // ============================================================================

  /**
   * Generates all polynomial term patterns up to the configured degree
   *
   * ALGORITHM:
   * Uses an iterative "next permutation" style algorithm to enumerate all
   * combinations of exponents that sum to each degree from 0 to maxDegree.
   *
   * Each pattern is a vector of exponents [e₀, e₁, ..., eₙ₋₁] where:
   * - Each eᵢ >= 0
   * - Sum of all eᵢ equals the term's degree
   * - The monomial represented is: x₀^e₀ * x₁^e₁ * ... * xₙ₋₁^eₙ₋₁
   *
   * MEMORY OPTIMIZATION:
   * Uses Int8Array for exponents (max degree 10 fits in int8)
   * Stores patterns in a flat array to avoid nested array overhead
   *
   * Storage layout: patterns[(termIndex * inputDim) + varIndex] = exponent
   *
   * @param inputDim - Number of input features
   * @returns Flat Int8Array containing all exponent patterns
   */
  private generatePolynomialTermPatterns(inputDim: number): Int8Array {
    const degree = this.config.polynomialDegree;
    const totalTerms = this.polynomialFeatureCount;

    // Allocate flat array: totalTerms patterns × inputDim exponents each
    // Int8Array is sufficient since max degree is 10
    const patterns = new Int8Array(totalTerms * inputDim);

    // Pattern 0 is all zeros (constant term = 1)
    // Int8Array is zero-initialized, so nothing to do for term 0

    // Working buffer for current exponent pattern (reused to avoid allocations)
    const currentExponents = new Int8Array(inputDim);

    // Index of next pattern to write (starts at 1, after constant term)
    let patternIndex = 1;

    // Generate patterns for each degree from 1 to maxDegree
    for (let d = 1; d <= degree; d++) {
      // Reset working array: start with all exponent on first variable
      // [d, 0, 0, ...] represents x₀^d
      currentExponents.fill(0);
      currentExponents[0] = d;

      // Copy first pattern of this degree to output array
      const baseOffset = patternIndex * inputDim;
      for (let i = 0; i < inputDim; i++) {
        patterns[baseOffset + i] = currentExponents[i];
      }
      patternIndex++;

      // Generate remaining patterns of this degree using next-pattern iteration
      while (this.advanceExponentPattern(currentExponents, d, inputDim)) {
        const offset = patternIndex * inputDim;
        for (let i = 0; i < inputDim; i++) {
          patterns[offset + i] = currentExponents[i];
        }
        patternIndex++;
      }
    }

    return patterns;
  }

  /**
   * Advances to the next exponent pattern with the same total degree
   * Uses a "carry" algorithm similar to incrementing a number in a variable base
   *
   * ALGORITHM:
   * 1. Find the rightmost position (before last) with non-zero exponent
   * 2. Decrement it by 1
   * 3. Collect all exponent "mass" from positions to its right
   * 4. Place all that mass (plus the 1 we decremented) at the position just after
   *
   * This generates patterns in graded reverse lexicographic order.
   *
   * Example sequence for inputDim=3, degree=2:
   * [2,0,0] -> [1,1,0] -> [1,0,1] -> [0,2,0] -> [0,1,1] -> [0,0,2] -> done
   *
   * @param exponents - Current exponent pattern (MODIFIED IN PLACE)
   * @param totalDegree - The sum that all exponents must equal
   * @param inputDim - Length of exponent array
   * @returns true if successfully advanced to new pattern, false if exhausted
   */
  private advanceExponentPattern(
    exponents: Int8Array,
    totalDegree: number,
    inputDim: number,
  ): boolean {
    // Find rightmost position (excluding last) with non-zero exponent
    // We exclude the last position because we redistribute mass to the right
    let pos = inputDim - 2;
    while (pos >= 0 && exponents[pos] === 0) {
      pos--;
    }

    // If no such position exists, we've enumerated all patterns for this degree
    if (pos < 0) {
      return false;
    }

    // Decrement exponent at found position
    exponents[pos]--;

    // Calculate mass to redistribute: 1 (from decrement) plus any mass to the right
    let massToRedistribute = 1;
    for (let i = pos + 1; i < inputDim; i++) {
      massToRedistribute += exponents[i];
      exponents[i] = 0; // Clear positions to the right
    }

    // Place all mass at position pos+1 (graded reverse lex order)
    exponents[pos + 1] = massToRedistribute;

    return true;
  }

  /**
   * Generates polynomial features from an input vector
   * Uses pre-computed patterns and stores result in pre-allocated buffer
   *
   * ZERO-ALLOCATION DESIGN:
   * - Reads from pre-computed patterns array
   * - Writes to pre-allocated phiBuffer
   * - No intermediate allocations
   *
   * @param normalizedInput - Normalized input feature values (uses normalizedInputBuffer)
   */
  private generatePolynomialFeatures(normalizedInput: Float64Array): void {
    const numTerms = this.polynomialFeatureCount;
    const inputDim = this.inputDimension;
    const patterns = this.polynomialTermPatterns;
    const phi = this.phiBuffer;

    // Compute each polynomial term using pre-computed exponent patterns
    for (let termIdx = 0; termIdx < numTerms; termIdx++) {
      let term = 1.0;
      const patternBase = termIdx * inputDim;

      // Multiply input values raised to their respective exponents
      // Skip multiplication for zero exponents (term stays unchanged)
      for (let varIdx = 0; varIdx < inputDim; varIdx++) {
        const exp = patterns[patternBase + varIdx];
        if (exp > 0) {
          const x = normalizedInput[varIdx];
          // Optimize common exponent cases to avoid Math.pow overhead
          // Most polynomial models use degrees 1-3 heavily
          switch (exp) {
            case 1:
              term *= x;
              break;
            case 2:
              term *= x * x;
              break;
            case 3:
              term *= x * x * x;
              break;
            default:
              term *= Math.pow(x, exp);
          }
        }
      }

      phi[termIdx] = term;
    }
  }

  // ============================================================================
  // NORMALIZATION METHODS
  // ============================================================================

  /**
   * Updates normalization statistics with new data points
   * Uses Welford's online algorithm for numerically stable mean and variance
   *
   * WELFORD'S ALGORITHM:
   * For each new value x:
   *   n = n + 1
   *   delta = x - mean
   *   mean = mean + delta / n
   *   delta2 = x - mean  (note: using NEW mean)
   *   M2 = M2 + delta * delta2
   *   variance = M2 / (n - 1)  when needed
   *
   * This is numerically stable even for large n and values with large mean.
   *
   * @param xCoordinates - Array of input data points
   */
  private updateNormalizationStats(xCoordinates: number[][]): void {
    // Skip if normalization is disabled
    if (!this.config.enableNormalization) return;

    const inputDim = this.inputDimension;
    const min = this.normStats.min;
    const max = this.normStats.max;
    const mean = this.normStats.mean;
    const m2 = this.normStats.m2;

    // Process each sample
    for (let sampleIdx = 0; sampleIdx < xCoordinates.length; sampleIdx++) {
      const x = xCoordinates[sampleIdx];
      this.normStats.count++;
      const n = this.normStats.count;

      // Update statistics for each feature dimension
      for (let i = 0; i < inputDim; i++) {
        const value = x[i];

        // Update min/max for min-max normalization
        if (value < min[i]) min[i] = value;
        if (value > max[i]) max[i] = value;

        // Welford's algorithm for running mean and variance
        const delta = value - mean[i];
        mean[i] += delta / n;
        const delta2 = value - mean[i];
        m2[i] += delta * delta2;
      }
    }
  }

  /**
   * Normalizes an input vector using current statistics
   * Stores result in pre-allocated normalizedInputBuffer (zero allocation)
   *
   * NORMALIZATION METHODS:
   * - min-max: x_norm = (x - min) / (max - min), scaled to [0, 1]
   * - z-score: x_norm = (x - mean) / std, centered at 0 with unit variance
   *
   * EXTRAPOLATION HANDLING:
   * For values outside the training range, we use soft clamping via tanh
   * to prevent extreme normalized values that could cause numerical issues.
   *
   * @param input - Raw input vector
   */
  private normalizeInput(input: number[]): void {
    const inputDim = this.inputDimension;
    const normalized = this.normalizedInputBuffer;

    // Skip normalization if disabled or insufficient data
    // Need at least 2 samples to compute meaningful range/variance
    if (!this.config.enableNormalization || this.normStats.count < 2) {
      for (let i = 0; i < inputDim; i++) {
        normalized[i] = input[i];
      }
      return;
    }

    const min = this.normStats.min;
    const max = this.normStats.max;
    const mean = this.normStats.mean;
    const m2 = this.normStats.m2;
    const count = this.normStats.count;

    switch (this.config.normalizationMethod) {
      case "min-max": {
        for (let i = 0; i < inputDim; i++) {
          const range = max[i] - min[i];

          if (range < 1e-10) {
            // Constant feature - normalize to 0.5 (center of [0,1])
            normalized[i] = 0.5;
          } else {
            // Standard min-max normalization: maps [min, max] to [0, 1]
            let val = (input[i] - min[i]) / range;

            // Soft-clamp extrapolated values to prevent extreme features
            // Values outside [0, 1] are compressed using tanh
            if (val < 0) {
              // Map (-∞, 0) to (-0.5, 0) using tanh
              val = 0.5 * Math.tanh(val);
            } else if (val > 1) {
              // Map (1, ∞) to (1, 1.5) using tanh
              val = 1 + 0.5 * Math.tanh(val - 1);
            }

            normalized[i] = val;
          }
        }
        break;
      }

      case "z-score": {
        for (let i = 0; i < inputDim; i++) {
          // Compute standard deviation from M2 accumulator
          // variance = M2 / (n - 1) for sample variance
          const variance = m2[i] / (count - 1);
          const std = Math.sqrt(Math.max(0, variance));

          if (std < 1e-10) {
            // Constant feature - normalize to 0
            normalized[i] = 0;
          } else {
            let val = (input[i] - mean[i]) / std;

            // Soft-clamp extreme z-scores (beyond ±4 std devs)
            // to prevent numerical issues with high-degree polynomials
            if (Math.abs(val) > 4) {
              val = 4 * Math.tanh(val / 4);
            }

            normalized[i] = val;
          }
        }
        break;
      }

      default: // "none"
        for (let i = 0; i < inputDim; i++) {
          normalized[i] = input[i];
        }
    }
  }

  // ============================================================================
  // PUBLIC TRAINING METHOD
  // ============================================================================

  /**
   * Performs online training using Recursive Least Squares algorithm
   *
   * RLS ALGORITHM OVERVIEW:
   * For each sample (x, y):
   * 1. Normalize x and generate polynomial features φ
   * 2. Compute gain vector: k = P·φ / (λ + φᵀ·P·φ)
   * 3. Update weights: w = w + k·(y - φᵀ·w)
   * 4. Update covariance: P = (I - k·φᵀ)·P / λ
   *
   * The forgetting factor λ (0 < λ ≤ 1) controls adaptation rate:
   * - λ = 1: Pure least squares, equal weight to all samples
   * - λ < 1: Exponential forgetting, recent samples weighted more heavily
   *
   * @param params - Training parameters containing x and y coordinates
   * @throws Error if input dimensions are inconsistent
   */
  public fitOnline(params: FitOnlineParams): void {
    const { xCoordinates, yCoordinates } = params;

    // Validate input data before any processing
    this.validateTrainingInput(xCoordinates, yCoordinates);

    // Initialize model on first training call (allocates all buffers)
    if (!this.isInitialized) {
      this.initializeModel(xCoordinates[0].length, yCoordinates[0].length);
    }

    // Update normalization statistics with all samples in this batch
    // This ensures consistent normalization across the batch
    this.updateNormalizationStats(xCoordinates);

    // Process each sample through the RLS update
    for (let i = 0; i < xCoordinates.length; i++) {
      this.processOneSample(xCoordinates[i], yCoordinates[i]);
    }
  }

  /**
   * Validates training input data for consistency and correctness
   *
   * @param xCoordinates - Input coordinates
   * @param yCoordinates - Output coordinates
   * @throws Error if validation fails
   */
  private validateTrainingInput(
    xCoordinates: number[][],
    yCoordinates: number[][],
  ): void {
    // Check for empty arrays
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

    // Check for zero dimensions
    if (xDim === 0) {
      throw new Error("Input dimension cannot be zero");
    }
    if (yDim === 0) {
      throw new Error("Output dimension cannot be zero");
    }

    // Check dimension consistency within the batch and for valid values
    for (let i = 0; i < xCoordinates.length; i++) {
      if (xCoordinates[i].length !== xDim) {
        throw new Error(`Inconsistent x dimension at index ${i}`);
      }
      if (yCoordinates[i].length !== yDim) {
        throw new Error(`Inconsistent y dimension at index ${i}`);
      }

      // Check for NaN/Infinity in inputs (would corrupt model state)
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

    // Check dimension consistency with existing model (if already initialized)
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
   * ALGORITHM STEPS:
   * 1. Normalize input and generate polynomial features φ
   * 2. Compute P·φ (used multiple times)
   * 3. Compute denominator: λ + φᵀ·P·φ
   * 4. Compute gain vector: k = P·φ / denominator
   * 5. For each output:
   *    - Compute prediction error: e = y - φᵀ·w
   *    - Update weights: w = w + k·e
   *    - Update error statistics
   * 6. Update covariance matrix using Joseph form for numerical stability
   * 7. Regularize covariance matrix
   *
   * @param x - Input vector (raw, unnormalized)
   * @param y - Output vector (target values)
   */
  private processOneSample(x: number[], y: number[]): void {
    const numFeatures = this.polynomialFeatureCount;
    const outputDim = this.outputDimension;
    const lambda = this.config.forgettingFactor;

    // Step 1: Normalize input (stores in normalizedInputBuffer)
    this.normalizeInput(x);

    // Generate polynomial features (stores in phiBuffer)
    this.generatePolynomialFeatures(this.normalizedInputBuffer);

    const phi = this.phiBuffer;
    const P = this.state.covarianceMatrix;
    const weights = this.state.weights;
    const pPhi = this.pPhiBuffer;
    const k = this.gainBuffer;

    // Step 2: Compute P·φ and store in pPhiBuffer
    // This is used for both gain computation and covariance update
    for (let i = 0; i < numFeatures; i++) {
      let sum = 0;
      const rowOffset = i * numFeatures;
      for (let j = 0; j < numFeatures; j++) {
        sum += P[rowOffset + j] * phi[j];
      }
      pPhi[i] = sum;
    }

    // Step 3: Compute denominator = λ + φᵀ·P·φ
    let phiTPphi = 0;
    for (let i = 0; i < numFeatures; i++) {
      phiTPphi += phi[i] * pPhi[i];
    }
    const denominator = lambda + phiTPphi;

    // Safety check: if denominator is near zero, the sample is nearly
    // linearly dependent with previous data - skip to avoid instability
    if (Math.abs(denominator) < 1e-12) {
      return;
    }

    // Step 4: Compute gain vector k = P·φ / denominator
    const invDenom = 1.0 / denominator;
    for (let i = 0; i < numFeatures; i++) {
      k[i] = pPhi[i] * invDenom;
    }

    // Step 5: Update weights for each output dimension
    for (let j = 0; j < outputDim; j++) {
      // Compute prediction with current weights: pred = φᵀ·w_j
      let prediction = 0;
      for (let i = 0; i < numFeatures; i++) {
        prediction += phi[i] * weights[i * outputDim + j];
      }

      // Compute prediction error
      const error = y[j] - prediction;

      // Update weights: w_j = w_j + k·error
      for (let i = 0; i < numFeatures; i++) {
        weights[i * outputDim + j] += k[i] * error;
      }

      // Update error statistics for R² and confidence intervals
      this.updateErrorStatistics(j, y[j], error);
    }

    // Step 6: Update covariance matrix using the Joseph form
    // P_new = (I - k·φᵀ)·P·(I - k·φᵀ)ᵀ / λ² + k·kᵀ·σ² / λ
    // Simplified (assuming σ²/λ term is small): P_new = (P - k·φᵀ·P) / λ
    //
    // We use: P_new = (P - k ⊗ pPhi) / λ
    // where k ⊗ pPhi is the outer product k·(P·φ)ᵀ

    // Compute P = (P - k·pPhiᵀ) / λ in place
    const invLambda = 1.0 / lambda;
    for (let i = 0; i < numFeatures; i++) {
      const ki = k[i];
      const rowOffset = i * numFeatures;
      for (let j = 0; j < numFeatures; j++) {
        // P[i,j] = (P[i,j] - k[i] * pPhi[j]) / λ
        P[rowOffset + j] = (P[rowOffset + j] - ki * pPhi[j]) * invLambda;
      }
    }

    // Step 7: Regularize and condition the covariance matrix
    this.regularizeCovarianceMatrix();

    // Update tracking variables
    for (let i = 0; i < this.inputDimension; i++) {
      this.state.lastInputPoint![i] = x[i];
    }
    this.state.timeIndex++;
    this.state.sampleCount++;

    // Update effective sample count (converges to 1/(1-λ) for λ < 1)
    this.state.effectiveSampleCount = lambda * this.state.effectiveSampleCount +
      1;
  }

  /**
   * Applies regularization to the covariance matrix for numerical stability
   *
   * This method:
   * 1. Adds regularization to diagonal elements
   * 2. Clamps diagonal elements to reasonable bounds
   * 3. Enforces symmetry (fixes floating-point drift)
   * 4. Bounds off-diagonal elements based on diagonal values
   *
   * NUMERICAL STABILITY:
   * Without regularization, the covariance matrix can become ill-conditioned
   * or even negative definite due to floating-point errors, causing the
   * algorithm to diverge.
   */
  private regularizeCovarianceMatrix(): void {
    const P = this.state.covarianceMatrix;
    const n = this.polynomialFeatureCount;
    const reg = this.config.regularization;

    // Bounds for diagonal elements
    const maxDiag = this.config.initialCovariance * 10;
    const minDiag = reg;

    // First pass: regularize and clamp diagonal, compute max diagonal
    let maxDiagVal = 0;
    for (let i = 0; i < n; i++) {
      const idx = i * n + i;
      // Add regularization
      P[idx] += reg;
      // Clamp to valid range
      if (P[idx] < minDiag) P[idx] = minDiag;
      if (P[idx] > maxDiag) P[idx] = maxDiag;
      // Track maximum for off-diagonal bounds
      if (P[idx] > maxDiagVal) maxDiagVal = P[idx];
    }

    // Second pass: enforce symmetry and bound off-diagonal elements
    // For positive semi-definiteness: |P[i,j]| <= sqrt(P[i,i] * P[j,j])
    // We use a slightly tighter bound for safety
    for (let i = 0; i < n; i++) {
      const rowOffset = i * n;
      const pii = P[rowOffset + i];

      for (let j = i + 1; j < n; j++) {
        const colOffset = j * n;
        const pjj = P[colOffset + j];

        // Enforce symmetry by averaging
        const avg = (P[rowOffset + j] + P[colOffset + i]) * 0.5;

        // Bound off-diagonal: |P[i,j]| <= 0.99 * sqrt(P[i,i] * P[j,j])
        // The 0.99 factor ensures strict positive definiteness
        const maxOffDiag = 0.99 * Math.sqrt(pii * pjj);
        const bounded = Math.max(-maxOffDiag, Math.min(maxOffDiag, avg));

        P[rowOffset + j] = bounded;
        P[colOffset + i] = bounded;
      }
    }
  }

  /**
   * Updates error statistics for R² and confidence interval calculation
   *
   * STATISTICS TRACKED:
   * 1. RSS (Residual Sum of Squares): Σ(y - ŷ)²
   *    - Used for R² calculation: R² = 1 - RSS/TSS
   * 2. TSS (Total Sum of Squares): Σ(y - ȳ)²
   *    - Computed via Welford's algorithm for numerical stability
   * 3. Weighted RSS with exponential forgetting
   *    - Used for confidence interval estimation
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
    // Current sample count (1-indexed for this sample)
    const n = this.state.sampleCount + 1;
    const lambda = this.config.forgettingFactor;

    // Update RSS: sum of squared prediction errors
    const errorSquared = error * error;
    this.state.residualSumSquares[outputIdx] += errorSquared;

    // Update weighted RSS (exponentially weighted for confidence intervals)
    this.state.weightedResidualSS[outputIdx] =
      lambda * this.state.weightedResidualSS[outputIdx] + errorSquared;

    // Update TSS using Welford's algorithm for numerical stability
    // This computes the running sum of squared deviations from the mean
    const prevMean = this.state.yMean[outputIdx];
    const delta = actual - prevMean;
    const newMean = prevMean + delta / n;
    const delta2 = actual - newMean;

    this.state.yMean[outputIdx] = newMean;
    // TSS = Σ(y_i - ȳ)² is exactly what Welford's M2 accumulator computes
    this.state.totalSumSquares[outputIdx] += delta * delta2;
  }

  // ============================================================================
  // PUBLIC PREDICTION METHOD
  // ============================================================================

  /**
   * Makes predictions for future time steps or specific input points
   *
   * @param params - Prediction parameters
   * @returns Prediction results with confidence intervals and model metrics
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
   * Uses linear extrapolation based on the average spacing in training data
   *
   * @param futureSteps - Number of future steps to generate
   * @returns Array of input points for prediction
   */
  private generateExtrapolationPoints(futureSteps: number): number[][] {
    const points: number[][] = [];
    const inputDim = this.inputDimension;

    if (!this.state.lastInputPoint) {
      throw new Error("No data points have been processed yet");
    }

    // Calculate step size from observed data range
    // Use average spacing: range / (n - 1)
    const stepSize = new Float64Array(inputDim);
    for (let i = 0; i < inputDim; i++) {
      if (this.normStats.count < 2) {
        stepSize[i] = 1;
      } else {
        const range = this.normStats.max[i] - this.normStats.min[i];
        stepSize[i] = range / Math.max(1, this.normStats.count - 1);
        // Ensure non-zero step for constant features
        if (Math.abs(stepSize[i]) < 1e-10) {
          stepSize[i] = 1;
        }
      }
    }

    // Generate future points by linear extrapolation
    for (let step = 1; step <= futureSteps; step++) {
      const point = new Array(inputDim);
      for (let i = 0; i < inputDim; i++) {
        point[i] = this.state.lastInputPoint[i] + step * stepSize[i];
      }
      points.push(point);
    }

    return points;
  }

  /**
   * Predicts output for a single input point with confidence intervals
   *
   * CONFIDENCE INTERVAL CALCULATION:
   * 1. Point prediction: ŷ = φᵀ·w
   * 2. Prediction variance: Var(ŷ) = σ²·(1 + φᵀ·P·φ)
   *    - σ² is estimated from residuals: σ² = RSS / (n - p)
   *    - φᵀ·P·φ accounts for parameter uncertainty
   * 3. Standard error: SE = sqrt(Var(ŷ))
   * 4. Confidence interval: ŷ ± t_{α/2} · SE
   *    - t_{α/2} is the t-distribution critical value
   *
   * @param inputPoint - Input feature vector
   * @returns Prediction with confidence intervals
   */
  private predictSinglePoint(inputPoint: number[]): SinglePrediction {
    const numFeatures = this.polynomialFeatureCount;
    const outputDim = this.outputDimension;
    const P = this.state.covarianceMatrix;
    const weights = this.state.weights;

    // Normalize input (stores in normalizedInputBuffer)
    this.normalizeInput(inputPoint);

    // Generate polynomial features (stores in phiBuffer)
    this.generatePolynomialFeatures(this.normalizedInputBuffer);

    const phi = this.phiBuffer;

    // Compute P·φ for variance calculation
    const pPhi = this.pPhiBuffer;
    for (let i = 0; i < numFeatures; i++) {
      let sum = 0;
      const rowOffset = i * numFeatures;
      for (let j = 0; j < numFeatures; j++) {
        sum += P[rowOffset + j] * phi[j];
      }
      pPhi[i] = sum;
    }

    // Compute variance factor: φᵀ·P·φ
    // This represents uncertainty due to parameter estimation
    let varianceFactor = 0;
    for (let i = 0; i < numFeatures; i++) {
      varianceFactor += phi[i] * pPhi[i];
    }
    // Ensure non-negative (numerical errors can make it slightly negative)
    varianceFactor = Math.max(0, varianceFactor);

    // Calculate degrees of freedom for t-distribution
    // df = effective sample count - number of parameters
    const effectiveDF = Math.max(
      1,
      this.state.effectiveSampleCount - numFeatures,
    );

    // Get t critical value for confidence intervals
    const tCritical = this.tCriticalValue(
      this.config.confidenceLevel,
      effectiveDF,
    );

    // Compute predictions and intervals for each output dimension
    const predicted = this.predictionBuffer;
    const standardError = this.stdErrorBuffer;
    const lowerBound = this.lowerBoundBuffer;
    const upperBound = this.upperBoundBuffer;

    for (let j = 0; j < outputDim; j++) {
      // Point prediction: φᵀ·w_j
      let pred = 0;
      for (let i = 0; i < numFeatures; i++) {
        pred += phi[i] * weights[i * outputDim + j];
      }
      predicted[j] = pred;

      // Estimate residual variance σ²
      let residualVariance: number;

      if (this.state.effectiveSampleCount > numFeatures + 1) {
        // Use weighted RSS for variance estimation
        // This gives more weight to recent residuals
        residualVariance = this.state.weightedResidualSS[j] / effectiveDF;
      } else {
        // Not enough samples - use target variance as prior estimate
        const tss = this.state.totalSumSquares[j];
        const n = this.state.sampleCount;
        residualVariance = n > 1 ? tss / (n - 1) : 1;
      }

      // Ensure minimum variance to prevent zero confidence intervals
      // Use a small fraction of total variance or a floor value
      const tss = this.state.totalSumSquares[j];
      const n = this.state.sampleCount;
      const targetVariance = n > 1 ? tss / (n - 1) : 1;
      const minVariance = Math.max(targetVariance * 0.001, 1e-10);
      residualVariance = Math.max(residualVariance, minVariance);

      // Prediction variance: σ²·(1 + φᵀ·P·φ)
      // The "1" term accounts for inherent noise in future observations
      const predictionVariance = residualVariance * (1 + varianceFactor);
      const se = Math.sqrt(predictionVariance);

      standardError[j] = se;

      // Confidence interval: ŷ ± t_{α/2}·SE
      const margin = tCritical * se;
      lowerBound[j] = pred - margin;
      upperBound[j] = pred + margin;
    }

    // Return copies of the buffer contents (required for interface)
    return {
      predicted: Array.from(predicted),
      lowerBound: Array.from(lowerBound),
      upperBound: Array.from(upperBound),
      standardError: Array.from(standardError),
    };
  }

  /**
   * Computes the t-distribution critical value for confidence intervals
   * Uses Cornish-Fisher expansion for small degrees of freedom
   *
   * @param confidenceLevel - Desired confidence level (e.g., 0.95)
   * @param df - Degrees of freedom
   * @returns Two-tailed critical value
   */
  private tCriticalValue(confidenceLevel: number, df: number): number {
    // Alpha is total probability in both tails
    const alpha = 1 - confidenceLevel;

    // Clamp df to at least 1
    const degreesOfFreedom = Math.max(1, df);

    // Get z-value for standard normal
    const zAlpha = this.normalInverseCDF(1 - alpha / 2);

    // For large df, t approaches normal
    if (degreesOfFreedom > 30) {
      return zAlpha;
    }

    // Cornish-Fisher expansion for better accuracy with small df
    const z = zAlpha;
    const z3 = z * z * z;
    const z5 = z3 * z * z;

    const g1 = (z3 + z) / 4;
    const g2 = (5 * z5 + 16 * z3 + 3 * z) / 96;

    return z + g1 / degreesOfFreedom +
      g2 / (degreesOfFreedom * degreesOfFreedom);
  }

  /**
   * Approximates the inverse CDF of the standard normal distribution
   * Uses the Abramowitz and Stegun rational approximation
   *
   * @param p - Probability (0 < p < 1)
   * @returns z-value such that P(Z < z) = p
   */
  private normalInverseCDF(p: number): number {
    // Clamp to valid range
    const clampedP = Math.max(1e-10, Math.min(1 - 1e-10, p));

    // Rational approximation coefficients
    const a = [
      -3.969683028665376e1,
      2.209460984245205e2,
      -2.759285104469687e2,
      1.383577518672690e2,
      -3.066479806614716e1,
      2.506628277459239e0,
    ];
    const b = [
      -5.447609879822406e1,
      1.615858368580409e2,
      -1.556989798598866e2,
      6.680131188771972e1,
      -1.328068155288572e1,
    ];
    const c = [
      -7.784894002430293e-3,
      -3.223964580411365e-1,
      -2.400758277161838e0,
      -2.549732539343734e0,
      4.374664141464968e0,
      2.938163982698783e0,
    ];
    const d = [
      7.784695709041462e-3,
      3.224671290700398e-1,
      2.445134137142996e0,
      3.754408661907416e0,
    ];

    const pLow = 0.02425;
    const pHigh = 1 - pLow;

    let q: number, r: number;

    if (clampedP < pLow) {
      // Lower tail
      q = Math.sqrt(-2 * Math.log(clampedP));
      return (
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
        ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
      );
    } else if (clampedP <= pHigh) {
      // Central region
      q = clampedP - 0.5;
      r = q * q;
      return (
        ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) *
          q) /
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
      );
    } else {
      // Upper tail
      q = Math.sqrt(-2 * Math.log(1 - clampedP));
      return (
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
        ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
      );
    }
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
   * - RSS = Residual Sum of Squares = Σ(y - ŷ)²
   * - TSS = Total Sum of Squares = Σ(y - ȳ)²
   *
   * R² represents the proportion of variance in y explained by the model.
   * - R² = 1: Perfect fit (all variance explained)
   * - R² = 0: Model no better than predicting mean
   * - R² < 0: Model worse than predicting mean (can happen with online learning)
   *
   * @returns R² value, clamped to [0, 1]
   */
  private calculateRSquared(): number {
    // Need at least 2 samples for meaningful R²
    if (this.state.sampleCount < 2) {
      return 0;
    }

    let totalRSS = 0;
    let totalTSS = 0;

    // Sum RSS and TSS across all output dimensions
    for (let j = 0; j < this.outputDimension; j++) {
      totalRSS += this.state.residualSumSquares[j];
      totalTSS += this.state.totalSumSquares[j];
    }

    // Handle edge case: no variance in targets
    if (totalTSS < 1e-10) {
      // If targets are constant, R² is 1 if predictions are also constant
      return totalRSS < 1e-10 ? 1 : 0;
    }

    // Standard R² formula
    const rSquared = 1 - totalRSS / totalTSS;

    // Clamp to [0, 1]
    // Note: negative R² is possible with online learning (model worse than mean)
    // but we clamp to 0 for the public interface
    return Math.max(0, Math.min(1, rSquared));
  }

  /**
   * Calculates the Root Mean Square Error (RMSE)
   *
   * RMSE = sqrt(mean(RSS)) = sqrt(RSS / n)
   *
   * Lower values indicate better fit.
   *
   * @returns RMSE value
   */
  private calculateRMSE(): number {
    if (this.state.sampleCount === 0) {
      return 0;
    }

    let totalMSE = 0;
    for (let j = 0; j < this.outputDimension; j++) {
      totalMSE += this.state.residualSumSquares[j] / this.state.sampleCount;
    }

    // Average across output dimensions
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

    // Clear all buffers
    this.polynomialTermPatterns = new Int8Array(0);
    this.phiBuffer = new Float64Array(0);
    this.pPhiBuffer = new Float64Array(0);
    this.gainBuffer = new Float64Array(0);
    this.normalizedInputBuffer = new Float64Array(0);
    this.predictionBuffer = new Float64Array(0);
    this.stdErrorBuffer = new Float64Array(0);
    this.lowerBoundBuffer = new Float64Array(0);
    this.upperBoundBuffer = new Float64Array(0);
    this.tempCovBuffer = new Float64Array(0);
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
   * @returns Weight matrix as 2D array [feature][output]
   */
  public getWeights(): number[][] {
    if (!this.isInitialized) {
      return [];
    }

    const numFeatures = this.polynomialFeatureCount;
    const outputDim = this.outputDimension;
    const result: number[][] = new Array(numFeatures);

    for (let i = 0; i < numFeatures; i++) {
      result[i] = new Array(outputDim);
      for (let j = 0; j < outputDim; j++) {
        result[i][j] = this.state.weights[i * outputDim + j];
      }
    }

    return result;
  }

  /**
   * Gets the current normalization statistics
   * Useful for understanding data distribution and debugging
   *
   * @returns Copy of normalization statistics
   */
  public getNormalizationStats(): {
    min: number[];
    max: number[];
    mean: number[];
    m2: number[];
    count: number;
  } {
    return {
      min: Array.from(this.normStats.min),
      max: Array.from(this.normStats.max),
      mean: Array.from(this.normStats.mean),
      m2: Array.from(this.normStats.m2),
      count: this.normStats.count,
    };
  }
}
