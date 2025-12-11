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
 * @module MultivariatePolynomialRegression
 * @version 1.0.0
 * @author Henrique Emanoel Viana
 */

// ============================================================================
// TYPE DEFINITIONS AND INTERFACES
// ============================================================================

/**
 * Supported normalization methods for input data preprocessing
 */
export type NormalizationMethod = "none" | "min-max" | "z-score";

/**
 * Configuration options for the polynomial regression model
 */
export interface PolynomialRegressionConfig {
  /** Degree of the polynomial (default: 2) */
  polynomialDegree?: number;

  /** Whether to normalize input data (default: true) */
  enableNormalization?: boolean;

  /** Method used for normalization (default: 'min-max') */
  normalizationMethod?: NormalizationMethod;

  /** Forgetting factor for RLS algorithm (0 < λ ≤ 1, default: 0.99) */
  forgettingFactor?: number;

  /** Initial covariance matrix diagonal value (default: 1000) */
  initialCovariance?: number;

  /** Regularization parameter to prevent numerical instability (default: 1e-6) */
  regularization?: number;

  /** Confidence level for prediction intervals (0-1, default: 0.95) */
  confidenceLevel?: number;
}

/**
 * Parameters for the fitOnline method
 */
export interface FitOnlineParams {
  /**
   * Input feature vectors where each element is a vector of numbers
   * Shape: [n_samples][n_features]
   */
  xCoordinates: number[][];

  /**
   * Output target vectors where each element is a vector of numbers
   * Shape: [n_samples][n_outputs]
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
   */
  futureSteps: number;

  /**
   * Optional: Specific input points to predict for (overrides futureSteps extrapolation)
   * Shape: [n_points][n_features]
   */
  inputPoints?: number[][];
}

/**
 * Single prediction result for one time step or input point
 */
export interface SinglePrediction {
  /** Predicted y values */
  predicted: number[];

  /** Lower bound of confidence interval */
  lowerBound: number[];

  /** Upper bound of confidence interval */
  upperBound: number[];

  /** Standard error of the prediction */
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

  /** Model's current R-squared value (coefficient of determination) */
  rSquared: number;

  /** Root Mean Square Error of the model */
  rmse: number;

  /** Number of samples the model has been trained on */
  sampleCount: number;

  /** Whether the model is sufficiently trained for reliable predictions */
  isModelReady: boolean;
}

/**
 * Statistics for normalization
 */
interface NormalizationStats {
  /** Minimum values for each feature (min-max normalization) */
  min: number[];

  /** Maximum values for each feature (min-max normalization) */
  max: number[];

  /** Mean values for each feature (z-score normalization) */
  mean: number[];

  /** Standard deviation for each feature (z-score normalization) */
  std: number[];

  /** Number of samples used to compute statistics */
  count: number;
}

/**
 * Internal model state
 */
interface ModelState {
  /** Weight coefficients matrix [n_polynomial_features][n_outputs] */
  weights: number[][];

  /** Covariance matrix for RLS [n_polynomial_features][n_polynomial_features] */
  covarianceMatrix: number[][];

  /** Sum of squared residuals for each output */
  residualSumSquares: number[];

  /** Sum of squared totals for each output (for R² calculation) */
  totalSumSquares: number[];

  /** Running mean of y values for each output */
  yMean: number[];

  /** Number of samples processed */
  sampleCount: number;

  /** Last input point seen (for extrapolation) */
  lastInputPoint: number[] | null;

  /** Time index for sequential data */
  timeIndex: number;
}

// ============================================================================
// UTILITY CLASSES
// ============================================================================

/**
 * Matrix operations utility class
 * Provides essential linear algebra operations for the regression algorithm
 */
class MatrixOperations {
  /**
   * Creates an identity matrix of size n x n
   */
  static identity(n: number): number[][] {
    const result: number[][] = [];
    for (let i = 0; i < n; i++) {
      result[i] = new Array(n).fill(0);
      result[i][i] = 1;
    }
    return result;
  }

  /**
   * Creates a matrix filled with zeros
   */
  static zeros(rows: number, cols: number): number[][] {
    return Array.from({ length: rows }, () => new Array(cols).fill(0));
  }

  /**
   * Multiplies a matrix by a scalar
   */
  static scalarMultiply(matrix: number[][], scalar: number): number[][] {
    return matrix.map((row) => row.map((val) => val * scalar));
  }

  /**
   * Adds two matrices element-wise
   */
  static add(a: number[][], b: number[][]): number[][] {
    return a.map((row, i) => row.map((val, j) => val + b[i][j]));
  }

  /**
   * Subtracts matrix b from matrix a element-wise
   */
  static subtract(a: number[][], b: number[][]): number[][] {
    return a.map((row, i) => row.map((val, j) => val - b[i][j]));
  }

  /**
   * Multiplies two matrices
   */
  static multiply(a: number[][], b: number[][]): number[][] {
    const rowsA = a.length;
    const colsA = a[0].length;
    const colsB = b[0].length;

    const result = this.zeros(rowsA, colsB);

    for (let i = 0; i < rowsA; i++) {
      for (let j = 0; j < colsB; j++) {
        for (let k = 0; k < colsA; k++) {
          result[i][j] += a[i][k] * b[k][j];
        }
      }
    }

    return result;
  }

  /**
   * Multiplies a matrix by a vector
   */
  static multiplyVector(matrix: number[][], vector: number[]): number[] {
    return matrix.map((row) =>
      row.reduce((sum, val, i) => sum + val * vector[i], 0)
    );
  }

  /**
   * Computes the outer product of two vectors
   */
  static outerProduct(a: number[], b: number[]): number[][] {
    return a.map((ai) => b.map((bi) => ai * bi));
  }

  /**
   * Computes the dot product of two vectors
   */
  static dotProduct(a: number[], b: number[]): number {
    return a.reduce((sum, val, i) => sum + val * b[i], 0);
  }

  /**
   * Transposes a matrix
   */
  static transpose(matrix: number[][]): number[][] {
    if (matrix.length === 0) return [];
    return matrix[0].map((_, colIndex) => matrix.map((row) => row[colIndex]));
  }

  /**
   * Clones a matrix (deep copy)
   */
  static clone(matrix: number[][]): number[][] {
    return matrix.map((row) => [...row]);
  }
}

/**
 * Statistical utility functions
 */
class StatisticsUtils {
  /**
   * Computes the t-distribution critical value approximation
   * Uses the approximation for large degrees of freedom
   */
  static tCriticalValue(
    confidenceLevel: number,
    degreesOfFreedom: number,
  ): number {
    // For confidence intervals, we need the two-tailed critical value
    const alpha = 1 - confidenceLevel;

    // Approximation using normal distribution for large df
    if (degreesOfFreedom > 30) {
      // Standard normal approximation
      const z = this.normalInverseCDF(1 - alpha / 2);
      return z;
    }

    // For smaller df, use a lookup table approximation
    const zAlpha = this.normalInverseCDF(1 - alpha / 2);
    const g1 = (zAlpha ** 3 + zAlpha) / 4;
    const g2 = (5 * zAlpha ** 5 + 16 * zAlpha ** 3 + 3 * zAlpha) / 96;

    return zAlpha + g1 / degreesOfFreedom + g2 / (degreesOfFreedom ** 2);
  }

  /**
   * Approximates the inverse CDF of the standard normal distribution
   * Uses the Abramowitz and Stegun approximation
   */
  static normalInverseCDF(p: number): number {
    if (p <= 0 || p >= 1) {
      throw new Error("Probability must be between 0 and 1 (exclusive)");
    }

    // Coefficients for the approximation
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

    const pLow = 0.02425;
    const pHigh = 1 - pLow;

    let q: number, r: number;

    if (p < pLow) {
      q = Math.sqrt(-2 * Math.log(p));
      return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
        ((((d1 * q + d2) * q + d3) * q + d4) * q + 1);
    } else if (p <= pHigh) {
      q = p - 0.5;
      r = q * q;
      return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
        (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1);
    } else {
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
  // Configuration
  private readonly config: Required<PolynomialRegressionConfig>;

  // Model state
  private state: ModelState;

  // Normalization statistics
  private normStats: NormalizationStats;

  // Feature dimensions
  private inputDimension: number = 0;
  private outputDimension: number = 0;
  private polynomialFeatureCount: number = 0;

  // Flag indicating if model has been initialized with first data
  private isInitialized: boolean = false;

  /**
   * Creates a new MultivariatePolynomialRegression instance
   *
   * @param config - Configuration options for the model
   */
  constructor(config: PolynomialRegressionConfig = {}) {
    // Set default configuration values
    this.config = {
      polynomialDegree: config.polynomialDegree ?? 2,
      enableNormalization: config.enableNormalization ?? true,
      normalizationMethod: config.normalizationMethod ?? "min-max",
      forgettingFactor: config.forgettingFactor ?? 0.99,
      initialCovariance: config.initialCovariance ?? 1000,
      regularization: config.regularization ?? 1e-6,
      confidenceLevel: config.confidenceLevel ?? 0.95,
    };

    // Validate configuration
    this.validateConfig();

    // Initialize empty state
    this.state = this.createEmptyState();
    this.normStats = this.createEmptyNormStats();
  }

  /**
   * Validates the configuration parameters
   * @throws Error if any configuration is invalid
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
   * Creates an empty model state
   */
  private createEmptyState(): ModelState {
    return {
      weights: [],
      covarianceMatrix: [],
      residualSumSquares: [],
      totalSumSquares: [],
      yMean: [],
      sampleCount: 0,
      lastInputPoint: null,
      timeIndex: 0,
    };
  }

  /**
   * Creates empty normalization statistics
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
   * Initializes the model with the dimensions from the first data batch
   *
   * @param inputDim - Number of input features
   * @param outputDim - Number of output targets
   */
  private initializeModel(inputDim: number, outputDim: number): void {
    this.inputDimension = inputDim;
    this.outputDimension = outputDim;

    // Calculate the number of polynomial features
    this.polynomialFeatureCount = this.calculatePolynomialFeatureCount(
      inputDim,
    );

    // Initialize weights to zeros
    this.state.weights = MatrixOperations.zeros(
      this.polynomialFeatureCount,
      outputDim,
    );

    // Initialize covariance matrix as scaled identity
    this.state.covarianceMatrix = MatrixOperations.scalarMultiply(
      MatrixOperations.identity(this.polynomialFeatureCount),
      this.config.initialCovariance,
    );

    // Initialize statistics
    this.state.residualSumSquares = new Array(outputDim).fill(0);
    this.state.totalSumSquares = new Array(outputDim).fill(0);
    this.state.yMean = new Array(outputDim).fill(0);

    // Initialize normalization stats
    this.normStats.min = new Array(inputDim).fill(Infinity);
    this.normStats.max = new Array(inputDim).fill(-Infinity);
    this.normStats.mean = new Array(inputDim).fill(0);
    this.normStats.std = new Array(inputDim).fill(0);

    this.isInitialized = true;
  }

  /**
   * Calculates the number of polynomial features for a given input dimension
   * Uses the formula for combinations with repetition: C(n + d, d)
   * where n is the number of features and d is the polynomial degree
   *
   * @param inputDim - Number of input features
   * @returns Number of polynomial features
   */
  private calculatePolynomialFeatureCount(inputDim: number): number {
    const degree = this.config.polynomialDegree;

    // Generate all polynomial terms and count them
    // This includes constant term (1), linear terms, and all interaction terms
    let count = 1; // Start with constant term

    // For each degree from 1 to max degree
    for (let d = 1; d <= degree; d++) {
      count += this.countTermsOfDegree(inputDim, d);
    }

    return count;
  }

  /**
   * Counts the number of polynomial terms of a specific degree
   * Uses stars and bars combinatorics: C(n + d - 1, d)
   *
   * @param n - Number of variables
   * @param d - Degree
   * @returns Number of terms
   */
  private countTermsOfDegree(n: number, d: number): number {
    // C(n + d - 1, d) = (n + d - 1)! / (d! * (n - 1)!)
    return this.binomial(n + d - 1, d);
  }

  /**
   * Computes binomial coefficient C(n, k)
   */
  private binomial(n: number, k: number): number {
    if (k > n) return 0;
    if (k === 0 || k === n) return 1;

    let result = 1;
    for (let i = 0; i < k; i++) {
      result = result * (n - i) / (i + 1);
    }
    return Math.round(result);
  }

  /**
   * Generates polynomial features from input vector
   *
   * @param input - Input feature vector
   * @returns Polynomial feature vector
   */
  private generatePolynomialFeatures(input: number[]): number[] {
    const features: number[] = [1]; // Constant term
    const degree = this.config.polynomialDegree;

    // Generate all polynomial terms using recursive enumeration
    const generateTerms = (
      currentDegree: number,
      startIdx: number,
      currentProduct: number,
    ): void => {
      if (currentDegree === 0) {
        features.push(currentProduct);
        return;
      }

      for (let i = startIdx; i < input.length; i++) {
        generateTerms(
          currentDegree - 1,
          i, // Allow repeated use of same variable
          currentProduct * input[i],
        );
      }
    };

    // Generate terms for each degree from 1 to max degree
    for (let d = 1; d <= degree; d++) {
      generateTerms(d, 0, 1);
    }

    return features;
  }

  /**
   * Updates normalization statistics with new data points
   *
   * @param xCoordinates - Input data points
   */
  private updateNormalizationStats(xCoordinates: number[][]): void {
    if (!this.config.enableNormalization) return;

    for (const x of xCoordinates) {
      const prevCount = this.normStats.count;
      this.normStats.count++;

      for (let i = 0; i < x.length; i++) {
        // Update min/max
        this.normStats.min[i] = Math.min(this.normStats.min[i], x[i]);
        this.normStats.max[i] = Math.max(this.normStats.max[i], x[i]);

        // Update running mean and variance (Welford's algorithm)
        const delta = x[i] - this.normStats.mean[i];
        this.normStats.mean[i] += delta / this.normStats.count;

        if (prevCount > 0) {
          const delta2 = x[i] - this.normStats.mean[i];
          // Running variance * count
          this.normStats.std[i] = (
            (prevCount - 1) * this.normStats.std[i] ** 2 +
            delta * delta2
          ) / prevCount;
          this.normStats.std[i] = Math.sqrt(this.normStats.std[i]);
        }
      }
    }
  }

  /**
   * Normalizes an input vector based on current statistics
   *
   * @param input - Raw input vector
   * @returns Normalized input vector
   */
  private normalizeInput(input: number[]): number[] {
    if (!this.config.enableNormalization || this.normStats.count < 2) {
      return input;
    }

    switch (this.config.normalizationMethod) {
      case "min-max":
        return input.map((val, i) => {
          const range = this.normStats.max[i] - this.normStats.min[i];
          if (range === 0) return 0;
          return (val - this.normStats.min[i]) / range;
        });

      case "z-score":
        return input.map((val, i) => {
          if (this.normStats.std[i] === 0) return 0;
          return (val - this.normStats.mean[i]) / this.normStats.std[i];
        });

      default:
        return input;
    }
  }

  /**
   * Performs online training using Recursive Least Squares algorithm
   *
   * The RLS algorithm updates the model incrementally for each new sample:
   * 1. Compute gain vector: k = P * x / (λ + x' * P * x)
   * 2. Update weights: w = w + k * (y - x' * w)
   * 3. Update covariance: P = (P - k * x' * P) / λ
   *
   * @param params - Training parameters containing x and y coordinates
   * @throws Error if input dimensions are inconsistent
   */
  public fitOnline(params: FitOnlineParams): void {
    const { xCoordinates, yCoordinates } = params;

    // Validate input
    this.validateTrainingInput(xCoordinates, yCoordinates);

    // Initialize model on first call
    if (!this.isInitialized) {
      this.initializeModel(xCoordinates[0].length, yCoordinates[0].length);
    }

    // Update normalization statistics
    this.updateNormalizationStats(xCoordinates);

    // Process each sample
    for (let i = 0; i < xCoordinates.length; i++) {
      this.processOneSample(xCoordinates[i], yCoordinates[i]);
    }
  }

  /**
   * Validates training input data
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

    // Check consistency
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
   * Processes a single training sample using RLS update
   *
   * @param x - Input vector
   * @param y - Output vector
   */
  private processOneSample(x: number[], y: number[]): void {
    // Normalize input
    const normalizedX = this.normalizeInput(x);

    // Generate polynomial features
    const phi = this.generatePolynomialFeatures(normalizedX);

    // RLS Update
    const lambda = this.config.forgettingFactor;
    const P = this.state.covarianceMatrix;

    // Compute P * phi
    const Pphi = MatrixOperations.multiplyVector(P, phi);

    // Compute denominator: lambda + phi' * P * phi
    const denominator = lambda + MatrixOperations.dotProduct(phi, Pphi);

    // Compute gain vector: k = P * phi / denominator
    const k = Pphi.map((val) => val / denominator);

    // Update weights for each output dimension
    for (let j = 0; j < this.outputDimension; j++) {
      // Prediction error: y - phi' * w
      const prediction = MatrixOperations.dotProduct(
        phi,
        this.state.weights.map((row) => row[j]),
      );
      const error = y[j] - prediction;

      // Update weights: w = w + k * error
      for (let i = 0; i < this.polynomialFeatureCount; i++) {
        this.state.weights[i][j] += k[i] * error;
      }

      // Update error statistics
      this.updateErrorStatistics(j, y[j], prediction);
    }

    // Update covariance matrix: P = (P - k * phi' * P) / lambda
    const kPhiP = MatrixOperations.outerProduct(k, Pphi);
    this.state.covarianceMatrix = MatrixOperations.scalarMultiply(
      MatrixOperations.subtract(P, kPhiP),
      1 / lambda,
    );

    // Add regularization to prevent numerical instability
    for (let i = 0; i < this.polynomialFeatureCount; i++) {
      this.state.covarianceMatrix[i][i] += this.config.regularization;
    }

    // Update state
    this.state.lastInputPoint = [...x];
    this.state.timeIndex++;
    this.state.sampleCount++;
  }

  /**
   * Updates error statistics for R² and RMSE calculation
   * Uses exponential weighting to give more importance to recent predictions
   *
   * @param outputIdx - Index of the output dimension
   * @param actual - Actual y value
   * @param predicted - Predicted y value
   */
  private updateErrorStatistics(
    outputIdx: number,
    actual: number,
    predicted: number,
  ): void {
    const n = this.state.sampleCount + 1;
    const lambda = this.config.forgettingFactor;

    // Update running mean of y (for TSS calculation)
    const prevMean = this.state.yMean[outputIdx];
    const newMean = prevMean + (actual - prevMean) / n;
    this.state.yMean[outputIdx] = newMean;

    // Update residual sum of squares with exponential weighting
    const residual = actual - predicted;
    this.state.residualSumSquares[outputIdx] =
      lambda * this.state.residualSumSquares[outputIdx] + residual ** 2;

    // Update total sum of squares using the deviation from previous mean
    // This gives a better online estimate of variance
    const deviation = actual - prevMean;
    if (n > 1) {
      this.state.totalSumSquares[outputIdx] =
        lambda * this.state.totalSumSquares[outputIdx] + deviation ** 2;
    }
  }

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
      // Use provided input points
      pointsToPredict = inputPoints;
    } else {
      // Generate extrapolated points
      pointsToPredict = this.generateExtrapolationPoints(futureSteps);
    }

    // Generate predictions
    const predictions: SinglePrediction[] = [];

    for (const point of pointsToPredict) {
      predictions.push(this.predictSinglePoint(point));
    }

    // Calculate model statistics
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
   *
   * @param futureSteps - Number of future steps to generate
   * @returns Array of input points for prediction
   */
  private generateExtrapolationPoints(futureSteps: number): number[][] {
    const points: number[][] = [];

    if (!this.state.lastInputPoint) {
      throw new Error("No data points have been processed yet");
    }

    // For time series extrapolation, we increment each dimension linearly
    // This assumes the input represents time or sequential data
    const lastPoint = this.state.lastInputPoint;

    // Calculate average step size from normalization stats
    const stepSize = lastPoint.map((val, i) => {
      if (this.normStats.count < 2) return 1;
      const range = this.normStats.max[i] - this.normStats.min[i];
      // Assume linear spacing, estimate step as range / (count - 1)
      return range / Math.max(1, this.normStats.count - 1);
    });

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
    // Normalize input
    const normalizedInput = this.normalizeInput(inputPoint);

    // Generate polynomial features
    const phi = this.generatePolynomialFeatures(normalizedInput);

    // Compute predictions
    const predicted: number[] = [];
    const standardError: number[] = [];
    const lowerBound: number[] = [];
    const upperBound: number[] = [];

    // Compute prediction variance: sigma^2 * phi' * P * phi
    const Pphi = MatrixOperations.multiplyVector(
      this.state.covarianceMatrix,
      phi,
    );
    const varianceMultiplier = MatrixOperations.dotProduct(phi, Pphi);

    // Get t critical value
    const df = Math.max(
      1,
      this.state.sampleCount - this.polynomialFeatureCount,
    );
    const tCritical = StatisticsUtils.tCriticalValue(
      this.config.confidenceLevel,
      df,
    );

    for (let j = 0; j < this.outputDimension; j++) {
      // Point prediction: phi' * w
      const pred = MatrixOperations.dotProduct(
        phi,
        this.state.weights.map((row) => row[j]),
      );
      predicted.push(pred);

      // Estimate residual variance
      const residualVariance =
        this.state.sampleCount > this.polynomialFeatureCount
          ? this.state.residualSumSquares[j] / df
          : 1;

      // Standard error of prediction
      const se = Math.sqrt(residualVariance * (1 + varianceMultiplier));
      standardError.push(se);

      // Confidence interval
      const margin = tCritical * se;
      lowerBound.push(pred - margin);
      upperBound.push(pred + margin);
    }

    return {
      predicted,
      lowerBound,
      upperBound,
      standardError,
    };
  }

  /**
   * Calculates the coefficient of determination (R²)
   * Uses exponentially weighted sums for online learning context
   *
   * @returns R² value (0-1, higher is better)
   */
  private calculateRSquared(): number {
    if (this.state.sampleCount < 3) return 0;

    let totalRSS = 0;
    let totalTSS = 0;

    for (let j = 0; j < this.outputDimension; j++) {
      totalRSS += this.state.residualSumSquares[j];
      totalTSS += this.state.totalSumSquares[j];
    }

    // Handle edge cases
    if (totalTSS < 1e-10) {
      // No variance in y - if predictions are also perfect, R² = 1
      return totalRSS < 1e-10 ? 1 : 0;
    }

    const rSquared = 1 - totalRSS / totalTSS;

    // Clamp to [0, 1] - negative R² can happen with very poor models
    return Math.max(0, Math.min(1, rSquared));
  }

  /**
   * Calculates the Root Mean Square Error
   * Uses exponentially weighted sum, so represents recent error level
   *
   * @returns RMSE value (lower is better)
   */
  private calculateRMSE(): number {
    if (this.state.sampleCount === 0) return 0;

    let totalMSE = 0;

    // The exponentially weighted RSS approximates the weighted average error
    // Normalize by effective sample count (1 / (1 - lambda))
    const effectiveSamples = Math.min(
      this.state.sampleCount,
      1 / (1 - this.config.forgettingFactor + 1e-10),
    );

    for (let j = 0; j < this.outputDimension; j++) {
      totalMSE += this.state.residualSumSquares[j] / effectiveSamples;
    }

    return Math.sqrt(totalMSE / this.outputDimension);
  }

  /**
   * Resets the model to its initial state
   * Useful for retraining from scratch
   */
  public reset(): void {
    this.state = this.createEmptyState();
    this.normStats = this.createEmptyNormStats();
    this.isInitialized = false;
    this.inputDimension = 0;
    this.outputDimension = 0;
    this.polynomialFeatureCount = 0;
  }

  /**
   * Returns a summary of the current model state
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
   * Useful for model inspection and debugging
   *
   * @returns Copy of the weight matrix
   */
  public getWeights(): number[][] {
    return MatrixOperations.clone(this.state.weights);
  }

  /**
   * Gets the current normalization statistics
   *
   * @returns Copy of normalization statistics
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
