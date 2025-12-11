/**
 * MultivariatePolynomialRegression Library
 *
 * High-performance TypeScript library for multivariate polynomial regression
 * with incremental online learning, Adam optimizer, and z-score normalization.
 *
 * @module MultivariatePolynomialRegression
 */

// ============================================================================
// INTERFACES AND TYPES
// ============================================================================

/**
 * Configuration options for the regression model
 */
export interface MultivariatePolynomialRegressionConfig {
  /** Polynomial degree for feature expansion (1-10, default: 2) */
  polynomialDegree?: number;
  /** Base learning rate for Adam optimizer (default: 0.001) */
  learningRate?: number;
  /** Number of warmup steps for learning rate schedule (default: 100) */
  warmupSteps?: number;
  /** Total steps for cosine decay schedule (default: 10000) */
  totalSteps?: number;
  /** Adam β₁ parameter for first moment estimate (default: 0.9) */
  beta1?: number;
  /** Adam β₂ parameter for second moment estimate (default: 0.999) */
  beta2?: number;
  /** Numerical stability constant (default: 1e-8) */
  epsilon?: number;
  /** L2 regularization strength λ (default: 1e-4) */
  regularizationStrength?: number;
  /** Mini-batch size for batch training (default: 32) */
  batchSize?: number;
  /** Convergence threshold for early stopping (default: 1e-6) */
  convergenceThreshold?: number;
  /** Z-score threshold for outlier detection (default: 3.0) */
  outlierThreshold?: number;
  /** ADWIN delta parameter for drift detection (default: 0.002) */
  adwinDelta?: number;
}

/**
 * Input data for fitting operations
 */
export interface FitInput {
  /** Input coordinates: array of samples, each sample is array of features */
  xCoordinates: number[][];
  /** Output coordinates: array of samples, each sample is array of outputs */
  yCoordinates: number[][];
}

/**
 * Result from online fitting operation
 */
export interface FitResult {
  /** Current loss value (MSE) */
  loss: number;
  /** L2 norm of the gradient */
  gradientNorm: number;
  /** Current effective learning rate after warmup/decay */
  effectiveLearningRate: number;
  /** Whether any sample was detected as an outlier */
  isOutlier: boolean;
  /** Whether the model has converged */
  converged: boolean;
  /** Total samples processed so far */
  sampleIndex: number;
  /** Whether concept drift was detected */
  driftDetected: boolean;
}

/**
 * Input for batch fitting
 */
export interface BatchFitInput extends FitInput {
  /** Number of training epochs (default: 100) */
  epochs?: number;
}

/**
 * Result from batch fitting operation
 */
export interface BatchFitResult {
  /** Final loss after training */
  finalLoss: number;
  /** History of losses per epoch */
  lossHistory: number[];
  /** Whether training converged */
  converged: boolean;
  /** Number of epochs completed */
  epochsCompleted: number;
  /** Total samples processed */
  totalSamplesProcessed: number;
}

/**
 * Single prediction with uncertainty estimates
 */
export interface SinglePrediction {
  /** Predicted output values */
  predicted: number[];
  /** Lower confidence bound (95%) */
  lowerBound: number[];
  /** Upper confidence bound (95%) */
  upperBound: number[];
  /** Standard error for each output dimension */
  standardError: number[];
}

/**
 * Result from prediction operation
 */
export interface PredictionResult {
  /** Array of predictions */
  predictions: SinglePrediction[];
  /** Model accuracy metric: 1/(1 + L̄) */
  accuracy: number;
  /** Number of samples used for training */
  sampleCount: number;
  /** Whether the model is ready for predictions */
  isModelReady: boolean;
}

/**
 * Weight information from the model
 */
export interface WeightInfo {
  /** Current weight matrix W[outputDim][featureCount] */
  weights: number[][];
  /** First moment estimates m[outputDim][featureCount] */
  firstMoment: number[][];
  /** Second moment estimates v[outputDim][featureCount] */
  secondMoment: number[][];
  /** Number of weight updates performed */
  updateCount: number;
}

/**
 * Normalization statistics from Welford's algorithm
 */
export interface NormalizationStats {
  /** Running mean of inputs */
  inputMean: number[];
  /** Running standard deviation of inputs */
  inputStd: number[];
  /** Running mean of outputs */
  outputMean: number[];
  /** Running standard deviation of outputs */
  outputStd: number[];
  /** Number of samples seen */
  count: number;
}

/**
 * Overall model summary
 */
export interface ModelSummary {
  /** Whether the model has been initialized */
  isInitialized: boolean;
  /** Input dimension (number of features) */
  inputDimension: number;
  /** Output dimension */
  outputDimension: number;
  /** Polynomial degree used */
  polynomialDegree: number;
  /** Number of polynomial features C(n+d,d) */
  polynomialFeatureCount: number;
  /** Number of samples processed */
  sampleCount: number;
  /** Current accuracy metric */
  accuracy: number;
  /** Whether model has converged */
  converged: boolean;
  /** Current effective learning rate */
  effectiveLearningRate: number;
  /** Number of drift events detected */
  driftCount: number;
}

// ============================================================================
// INTERNAL INTERFACES
// ============================================================================

/**
 * Internal state for Welford's online statistics
 * Tracks running mean and M₂ for variance computation
 */
interface WelfordState {
  count: number;
  mean: Float64Array;
  m2: Float64Array;
}

/**
 * ADWIN bucket for adaptive windowing drift detection
 */
interface ADWINBucket {
  total: number;
  variance: number;
  count: number;
}

// ============================================================================
// OBJECT POOL FOR MEMORY REUSE
// ============================================================================

/**
 * Object pool for Float64Array instances
 * Minimizes garbage collection pressure by reusing arrays
 */
class Float64ArrayPool {
  private readonly pools: Map<number, Float64Array[]> = new Map();
  private static readonly MAX_POOL_SIZE = 64;

  /**
   * Acquires a Float64Array of the specified size
   * @param size - Required array length
   * @returns Float64Array (reused or newly allocated)
   */
  acquire(size: number): Float64Array {
    const pool = this.pools.get(size);
    if (pool !== undefined && pool.length > 0) {
      return pool.pop()!;
    }
    return new Float64Array(size);
  }

  /**
   * Releases a Float64Array back to the pool
   * @param arr - Array to release
   */
  release(arr: Float64Array): void {
    const size = arr.length;
    let pool = this.pools.get(size);
    if (pool === undefined) {
      pool = [];
      this.pools.set(size, pool);
    }
    if (pool.length < Float64ArrayPool.MAX_POOL_SIZE) {
      // Zero out for security and fresh state
      for (let i = 0; i < size; i++) {
        arr[i] = 0;
      }
      pool.push(arr);
    }
  }

  /**
   * Clears all pooled arrays
   */
  clear(): void {
    this.pools.clear();
  }
}

// ============================================================================
// MAIN CLASS
// ============================================================================

/**
 * MultivariatePolynomialRegression
 *
 * High-performance multivariate polynomial regression with:
 * - Incremental online learning via Adam optimizer
 * - Z-score normalization using Welford's algorithm
 * - L2 regularization for weight decay
 * - Outlier detection and downweighting
 * - ADWIN-based concept drift detection
 * - Cosine warmup learning rate schedule
 *
 * Mathematical Background:
 * - Polynomial features: φ(x) = all monomials of degree ≤ d
 * - Feature count: C(n+d, d) where n = input dimension
 * - Model: ŷ = W·φ(x) where W ∈ ℝ^(m×k), m = output dim, k = feature count
 *
 * @example
 * ```typescript
 * const model = new MultivariatePolynomialRegression({ polynomialDegree: 2 });
 *
 * // Online learning
 * const result = model.fitOnline({
 *   xCoordinates: [[1, 2], [3, 4]],
 *   yCoordinates: [[5], [6]]
 * });
 *
 * // Batch learning
 * const batchResult = model.fitBatch({
 *   xCoordinates: data.x,
 *   yCoordinates: data.y,
 *   epochs: 100
 * });
 *
 * // Prediction
 * const predictions = model.predict(10);
 * ```
 */
export class MultivariatePolynomialRegression {
  // =========================================================================
  // CONFIGURATION (readonly after construction)
  // =========================================================================

  private readonly _polynomialDegree: number;
  private readonly _learningRate: number;
  private readonly _warmupSteps: number;
  private readonly _totalSteps: number;
  private readonly _beta1: number;
  private readonly _beta2: number;
  private readonly _epsilon: number;
  private readonly _regularizationStrength: number;
  private readonly _batchSize: number;
  private readonly _convergenceThreshold: number;
  private readonly _outlierThreshold: number;
  private readonly _adwinDelta: number;

  // =========================================================================
  // MODEL DIMENSIONS (lazy initialized)
  // =========================================================================

  private _inputDimension: number = 0;
  private _outputDimension: number = 0;
  private _featureCount: number = 0;
  private _isInitialized: boolean = false;

  // =========================================================================
  // WEIGHTS AND ADAM STATE (Float64Arrays for performance)
  // =========================================================================

  /** Weight matrix W, flattened: [outputDim * featureCount] */
  private _weights: Float64Array | null = null;
  /** Adam first moment m */
  private _firstMoment: Float64Array | null = null;
  /** Adam second moment v */
  private _secondMoment: Float64Array | null = null;

  // =========================================================================
  // NORMALIZATION STATE (Welford's algorithm)
  // =========================================================================

  private _inputStats: WelfordState | null = null;
  private _outputStats: WelfordState | null = null;
  private _residualStats: WelfordState | null = null;

  // =========================================================================
  // TRAINING STATE
  // =========================================================================

  private _updateCount: number = 0;
  private _sampleCount: number = 0;
  private _runningLossSum: number = 0;
  private _runningLossCount: number = 0;
  private _converged: boolean = false;
  private _previousLoss: number = Infinity;
  private _driftCount: number = 0;

  // =========================================================================
  // ADWIN DRIFT DETECTION STATE
  // =========================================================================

  private _adwinBuckets: ADWINBucket[] = [];
  private _adwinTotal: number = 0;
  private _adwinCount: number = 0;

  // =========================================================================
  // PREALLOCATED BUFFERS (hot path optimization)
  // =========================================================================

  private _featureBuffer: Float64Array | null = null;
  private _gradientBuffer: Float64Array | null = null;
  private _errorBuffer: Float64Array | null = null;
  private _normalizedInputBuffer: Float64Array | null = null;
  private _normalizedOutputBuffer: Float64Array | null = null;
  private _predictionBuffer: Float64Array | null = null;
  private _tempBuffer: Float64Array | null = null;

  // =========================================================================
  // POLYNOMIAL FEATURE CACHE
  // =========================================================================

  /** Cached exponent patterns for polynomial features */
  private _exponentCache: Uint8Array[] | null = null;

  // =========================================================================
  // OBJECT POOL
  // =========================================================================

  private readonly _pool: Float64ArrayPool = new Float64ArrayPool();

  // =========================================================================
  // RECENT SAMPLES BUFFER (for prediction)
  // =========================================================================

  private _recentX: Float64Array[] = [];
  private _recentY: Float64Array[] = [];
  private _recentIndex: number = 0;
  private readonly _maxRecentSamples: number = 128;

  // =========================================================================
  // RANDOM NUMBER GENERATOR STATE (Box-Muller)
  // =========================================================================

  private _hasSpare: boolean = false;
  private _spare: number = 0;

  /**
   * Creates a new MultivariatePolynomialRegression instance
   *
   * @param config - Configuration options
   * @throws Error if polynomialDegree is outside valid range [1, 10]
   *
   * @example
   * ```typescript
   * const model = new MultivariatePolynomialRegression({
   *   polynomialDegree: 3,
   *   learningRate: 0.01,
   *   regularizationStrength: 0.001
   * });
   * ```
   */
  constructor(config: MultivariatePolynomialRegressionConfig = {}) {
    // Validate and set polynomial degree
    const degree = config.polynomialDegree ?? 2;
    if (degree < 1 || degree > 10 || !Number.isInteger(degree)) {
      throw new Error(
        `polynomialDegree must be an integer between 1 and 10, got: ${degree}`,
      );
    }
    this._polynomialDegree = degree;

    // Set learning parameters with defaults
    this._learningRate = this._validatePositive(
      config.learningRate ?? 0.001,
      "learningRate",
    );
    this._warmupSteps = this._validateNonNegativeInt(
      config.warmupSteps ?? 100,
      "warmupSteps",
    );
    this._totalSteps = this._validatePositiveInt(
      config.totalSteps ?? 10000,
      "totalSteps",
    );

    // Adam parameters
    this._beta1 = this._validateRange(config.beta1 ?? 0.9, 0, 1, "beta1");
    this._beta2 = this._validateRange(config.beta2 ?? 0.999, 0, 1, "beta2");
    this._epsilon = this._validatePositive(config.epsilon ?? 1e-8, "epsilon");

    // Regularization and batch settings
    this._regularizationStrength = this._validateNonNegative(
      config.regularizationStrength ?? 1e-4,
      "regularizationStrength",
    );
    this._batchSize = this._validatePositiveInt(
      config.batchSize ?? 32,
      "batchSize",
    );

    // Convergence and outlier detection
    this._convergenceThreshold = this._validatePositive(
      config.convergenceThreshold ?? 1e-6,
      "convergenceThreshold",
    );
    this._outlierThreshold = this._validatePositive(
      config.outlierThreshold ?? 3.0,
      "outlierThreshold",
    );

    // ADWIN drift detection
    this._adwinDelta = this._validateRange(
      config.adwinDelta ?? 0.002,
      0,
      1,
      "adwinDelta",
    );
  }

  // =========================================================================
  // VALIDATION HELPERS
  // =========================================================================

  private _validatePositive(value: number, name: string): number {
    if (typeof value !== "number" || !isFinite(value) || value <= 0) {
      throw new Error(
        `${name} must be a positive finite number, got: ${value}`,
      );
    }
    return value;
  }

  private _validateNonNegative(value: number, name: string): number {
    if (typeof value !== "number" || !isFinite(value) || value < 0) {
      throw new Error(
        `${name} must be a non-negative finite number, got: ${value}`,
      );
    }
    return value;
  }

  private _validatePositiveInt(value: number, name: string): number {
    if (!Number.isInteger(value) || value <= 0) {
      throw new Error(`${name} must be a positive integer, got: ${value}`);
    }
    return value;
  }

  private _validateNonNegativeInt(value: number, name: string): number {
    if (!Number.isInteger(value) || value < 0) {
      throw new Error(`${name} must be a non-negative integer, got: ${value}`);
    }
    return value;
  }

  private _validateRange(
    value: number,
    min: number,
    max: number,
    name: string,
  ): number {
    if (
      typeof value !== "number" || !isFinite(value) || value < min ||
      value > max
    ) {
      throw new Error(
        `${name} must be between ${min} and ${max}, got: ${value}`,
      );
    }
    return value;
  }

  // =========================================================================
  // PUBLIC API
  // =========================================================================

  /**
   * Performs incremental online learning on the provided samples
   *
   * Uses Adam optimizer with:
   * - Cosine warmup learning rate schedule
   * - Welford's algorithm for z-score normalization
   * - L2 regularization
   * - Outlier downweighting based on z-score of residuals
   * - ADWIN-based concept drift detection
   *
   * Mathematical formulation:
   * ```
   * Normalize:    x̃ = (x - μ)/(σ + ε)
   * Features:     φ(x) = [1, x₁, x₂, ..., x₁², x₁x₂, ...]  (monomials ≤ degree d)
   * Forward:      ŷ = W·φ(x)
   * Loss:         L = (1/2n)Σ‖y - ŷ‖² + (λ/2)‖W‖²
   * Gradient:     g = -(1/n)Σ(e ⊗ φ) + λW
   * Adam:         m = β₁m + (1-β₁)g
   *               v = β₂v + (1-β₂)g²
   * Update:       W -= η·(m/(1-β₁ᵗ))/(√(v/(1-β₂ᵗ)) + ε)
   * ```
   *
   * @param input - Input coordinates and output coordinates
   * @returns FitResult with loss, gradient norm, and training state
   *
   * @example
   * ```typescript
   * const result = model.fitOnline({
   *   xCoordinates: [[1, 2], [3, 4]],
   *   yCoordinates: [[5], [6]]
   * });
   * console.log(`Loss: ${result.loss}, Converged: ${result.converged}`);
   * ```
   */
  fitOnline(input: FitInput): FitResult {
    const { xCoordinates, yCoordinates } = input;

    // Input validation
    if (!Array.isArray(xCoordinates) || !Array.isArray(yCoordinates)) {
      throw new Error("xCoordinates and yCoordinates must be arrays");
    }
    if (xCoordinates.length === 0 || yCoordinates.length === 0) {
      throw new Error("Input arrays must not be empty");
    }
    if (xCoordinates.length !== yCoordinates.length) {
      throw new Error(
        `xCoordinates length (${xCoordinates.length}) must equal yCoordinates length (${yCoordinates.length})`,
      );
    }

    const n = xCoordinates.length;

    // Validate first sample dimensions
    if (!Array.isArray(xCoordinates[0]) || xCoordinates[0].length === 0) {
      throw new Error("Each x sample must be a non-empty array");
    }
    if (!Array.isArray(yCoordinates[0]) || yCoordinates[0].length === 0) {
      throw new Error("Each y sample must be a non-empty array");
    }

    // Lazy initialization on first data
    if (!this._isInitialized) {
      this._initializeModel(xCoordinates[0].length, yCoordinates[0].length);
    }

    // Validate dimension consistency
    const expectedInputDim = this._inputDimension;
    const expectedOutputDim = this._outputDimension;

    let totalLoss = 0;
    let totalGradientNorm = 0;
    let anyOutlier = false;
    let driftDetected = false;

    // Process each sample incrementally
    for (let i = 0; i < n; i++) {
      const x = xCoordinates[i];
      const y = yCoordinates[i];

      // Dimension check (fast path: only check first dimension)
      if (x.length !== expectedInputDim) {
        throw new Error(
          `Sample ${i}: x dimension (${x.length}) doesn't match expected (${expectedInputDim})`,
        );
      }
      if (y.length !== expectedOutputDim) {
        throw new Error(
          `Sample ${i}: y dimension (${y.length}) doesn't match expected (${expectedOutputDim})`,
        );
      }

      // Update normalization statistics using Welford's algorithm
      // Welford: δ = x - μ, μ += δ/n, M₂ += δ(x - μ)
      this._updateWelfordStats(this._inputStats!, x);
      this._updateWelfordStats(this._outputStats!, y);

      // Normalize input and output: x̃ = (x - μ)/(σ + ε)
      this._normalizeVector(x, this._inputStats!, this._normalizedInputBuffer!);
      this._normalizeVector(
        y,
        this._outputStats!,
        this._normalizedOutputBuffer!,
      );

      // Generate polynomial features φ(x)
      // φ(x) = [1, x₁, x₂, ..., x₁², x₁x₂, ...] (graded lex order)
      this._generatePolynomialFeatures(
        this._normalizedInputBuffer!,
        this._featureBuffer!,
      );

      // Forward pass: ŷ = W·φ(x)
      this._forward(this._featureBuffer!, this._predictionBuffer!);

      // Compute error: e = y - ŷ
      this._computeError(
        this._normalizedOutputBuffer!,
        this._predictionBuffer!,
        this._errorBuffer!,
      );

      // Compute loss: L = (1/2)‖e‖²
      const sampleLoss = this._computeMSELoss(this._errorBuffer!);

      // Check for outlier based on residual z-score
      // r = (y - ŷ)/σ_residual; outlier if |r| > threshold
      const isCurrentOutlier = this._detectOutlier(this._errorBuffer!);
      if (isCurrentOutlier) {
        anyOutlier = true;
      }

      // Outlier downweighting factor: 0.1 for outliers, 1.0 otherwise
      const sampleWeight = isCurrentOutlier ? 0.1 : 1.0;

      // Compute gradient: g = -(e ⊗ φ) + λW
      const gradNorm = this._computeGradient(
        this._errorBuffer!,
        this._featureBuffer!,
        sampleWeight,
      );
      totalGradientNorm += gradNorm;

      // Get effective learning rate (warmup + cosine decay)
      // Warmup: η_t = η * t / warmup_steps
      // Cosine: η_t = η * 0.5 * (1 + cos(π * progress))
      this._updateCount++;
      const effectiveLR = this._getEffectiveLearningRate();

      // Adam update
      // m = β₁m + (1-β₁)g, v = β₂v + (1-β₂)g²
      // W -= η·m̂/(√v̂ + ε)
      this._adamUpdate(effectiveLR);

      // Update running loss for accuracy tracking
      this._runningLossSum += sampleLoss;
      this._runningLossCount++;
      totalLoss += sampleLoss;

      // Update residual statistics for confidence intervals
      this._updateWelfordStatsTyped(this._residualStats!, this._errorBuffer!);

      // ADWIN drift detection
      // Detect: |μ₀ - μ₁| ≥ √((2/m)·ln(2/δ))
      if (this._detectDrift(sampleLoss)) {
        driftDetected = true;
        this._handleDrift();
      }

      // Store recent sample for prediction
      this._storeRecentSample(x, y);

      this._sampleCount++;
    }

    // Check convergence based on loss change
    const avgLoss = totalLoss / n;
    const lossDelta = Math.abs(this._previousLoss - avgLoss);
    if (
      lossDelta < this._convergenceThreshold &&
      this._sampleCount > this._warmupSteps
    ) {
      this._converged = true;
    }
    this._previousLoss = avgLoss;

    return {
      loss: avgLoss,
      gradientNorm: totalGradientNorm / n,
      effectiveLearningRate: this._getEffectiveLearningRate(),
      isOutlier: anyOutlier,
      converged: this._converged,
      sampleIndex: this._sampleCount,
      driftDetected,
    };
  }

  /**
   * Performs batch training with mini-batch gradient descent
   *
   * Features:
   * - Mini-batch gradient descent with configurable batch size
   * - Fisher-Yates shuffling between epochs
   * - Early stopping based on convergence threshold
   * - Gradient accumulation for memory efficiency
   *
   * @param input - Training data and optional number of epochs
   * @returns BatchFitResult with final loss and training history
   *
   * @example
   * ```typescript
   * const result = model.fitBatch({
   *   xCoordinates: trainingX,
   *   yCoordinates: trainingY,
   *   epochs: 200
   * });
   * console.log(`Final loss: ${result.finalLoss}, Epochs: ${result.epochsCompleted}`);
   * ```
   */
  fitBatch(input: BatchFitInput): BatchFitResult {
    const { xCoordinates, yCoordinates, epochs = 100 } = input;

    // Input validation
    if (!Array.isArray(xCoordinates) || !Array.isArray(yCoordinates)) {
      throw new Error("xCoordinates and yCoordinates must be arrays");
    }
    if (xCoordinates.length === 0 || yCoordinates.length === 0) {
      throw new Error("Input arrays must not be empty");
    }
    if (xCoordinates.length !== yCoordinates.length) {
      throw new Error(
        `xCoordinates length (${xCoordinates.length}) must equal yCoordinates length (${yCoordinates.length})`,
      );
    }

    const n = xCoordinates.length;

    // Lazy initialization
    if (!this._isInitialized) {
      this._initializeModel(xCoordinates[0].length, yCoordinates[0].length);
    }

    // First pass: compute normalization statistics for all data
    for (let i = 0; i < n; i++) {
      this._updateWelfordStats(this._inputStats!, xCoordinates[i]);
      this._updateWelfordStats(this._outputStats!, yCoordinates[i]);
    }

    const lossHistory: number[] = [];
    let epochsCompleted = 0;
    let totalSamplesProcessed = 0;
    let converged = false;

    // Create index array for shuffling (Uint32Array for memory efficiency)
    const indices = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
      indices[i] = i;
    }

    // Batch gradient accumulator (reused across batches)
    const batchGradient = this._pool.acquire(
      this._outputDimension * this._featureCount,
    );

    // Patience counter for early stopping
    let patienceCounter = 0;
    const patience = 10;
    let bestLoss = Infinity;

    for (let epoch = 0; epoch < epochs; epoch++) {
      // Shuffle indices using Fisher-Yates
      this._shuffleIndices(indices);

      let epochLoss = 0;

      // Process mini-batches
      for (let batchStart = 0; batchStart < n; batchStart += this._batchSize) {
        const batchEnd = Math.min(batchStart + this._batchSize, n);
        const currentBatchSize = batchEnd - batchStart;

        // Reset batch gradient accumulator
        for (let j = 0; j < batchGradient.length; j++) {
          batchGradient[j] = 0;
        }

        let batchLoss = 0;

        // Accumulate gradients over mini-batch
        for (let b = batchStart; b < batchEnd; b++) {
          const idx = indices[b];
          const x = xCoordinates[idx];
          const y = yCoordinates[idx];

          // Normalize
          this._normalizeVector(
            x,
            this._inputStats!,
            this._normalizedInputBuffer!,
          );
          this._normalizeVector(
            y,
            this._outputStats!,
            this._normalizedOutputBuffer!,
          );

          // Generate features
          this._generatePolynomialFeatures(
            this._normalizedInputBuffer!,
            this._featureBuffer!,
          );

          // Forward pass
          this._forward(this._featureBuffer!, this._predictionBuffer!);

          // Compute error
          this._computeError(
            this._normalizedOutputBuffer!,
            this._predictionBuffer!,
            this._errorBuffer!,
          );

          // Accumulate loss
          batchLoss += this._computeMSELoss(this._errorBuffer!);

          // Accumulate gradient: g += -(e ⊗ φ)
          this._accumulateGradient(
            this._errorBuffer!,
            this._featureBuffer!,
            batchGradient,
          );

          totalSamplesProcessed++;
        }

        // Average gradient and add L2 regularization: g = g/batch_size + λW
        const scale = 1.0 / currentBatchSize;
        const weights = this._weights!;
        const lambda = this._regularizationStrength;

        for (let j = 0; j < batchGradient.length; j++) {
          batchGradient[j] = batchGradient[j] * scale + lambda * weights[j];
        }

        // Get effective learning rate
        this._updateCount++;
        const effectiveLR = this._getEffectiveLearningRate();

        // Adam update with accumulated gradient
        this._adamUpdateWithGradient(effectiveLR, batchGradient);

        epochLoss += batchLoss;
      }

      const avgEpochLoss = epochLoss / n;
      lossHistory.push(avgEpochLoss);
      epochsCompleted++;

      // Update running loss tracking
      this._runningLossSum += epochLoss;
      this._runningLossCount += n;

      // Check for early stopping convergence
      if (avgEpochLoss < bestLoss - this._convergenceThreshold) {
        bestLoss = avgEpochLoss;
        patienceCounter = 0;
      } else {
        patienceCounter++;
      }

      if (patienceCounter >= patience) {
        converged = true;
        this._converged = true;
        break;
      }

      // Also check direct convergence
      if (lossHistory.length >= 2) {
        const prevLoss = lossHistory[lossHistory.length - 2];
        if (Math.abs(prevLoss - avgEpochLoss) < this._convergenceThreshold) {
          converged = true;
          this._converged = true;
          break;
        }
      }
    }

    // Release pooled gradient buffer
    this._pool.release(batchGradient);

    // Store samples for prediction (up to max recent)
    const samplesToStore = Math.min(n, this._maxRecentSamples);
    for (let i = 0; i < samplesToStore; i++) {
      this._storeRecentSample(xCoordinates[i], yCoordinates[i]);
    }
    this._sampleCount += n;

    return {
      finalLoss: lossHistory.length > 0
        ? lossHistory[lossHistory.length - 1]
        : 0,
      lossHistory,
      converged,
      epochsCompleted,
      totalSamplesProcessed,
    };
  }

  /**
   * Generates predictions for future steps
   *
   * Uses the trained model to predict outputs with uncertainty estimates.
   * Confidence intervals are computed using residual standard error with
   * 95% coverage (±1.96 standard errors).
   *
   * @param futureSteps - Number of predictions to generate
   * @returns PredictionResult with predictions and model state
   *
   * @example
   * ```typescript
   * const predictions = model.predict(5);
   * predictions.predictions.forEach((p, i) => {
   *   console.log(`Step ${i}: ${p.predicted} ± ${p.standardError}`);
   * });
   * ```
   */
  predict(futureSteps: number): PredictionResult {
    // Validate input
    if (!Number.isInteger(futureSteps) || futureSteps < 0) {
      throw new Error(
        `futureSteps must be a non-negative integer, got: ${futureSteps}`,
      );
    }

    // Check if model is ready
    if (!this._isInitialized || this._sampleCount === 0) {
      return {
        predictions: [],
        accuracy: 0,
        sampleCount: 0,
        isModelReady: false,
      };
    }

    const predictions: SinglePrediction[] = [];
    const accuracy = this._getAccuracy();

    // Get residual standard deviation for confidence intervals
    const residualStd = this._getResidualStd();

    // Generate predictions
    for (let step = 0; step < futureSteps; step++) {
      // Get input for prediction
      const inputX = this._getInputForPrediction(step);

      if (inputX === null) {
        break;
      }

      // Normalize input: x̃ = (x - μ)/(σ + ε)
      this._normalizeTypedArray(
        inputX,
        this._inputStats!,
        this._normalizedInputBuffer!,
      );

      // Generate polynomial features φ(x)
      this._generatePolynomialFeatures(
        this._normalizedInputBuffer!,
        this._featureBuffer!,
      );

      // Forward pass: ŷ = W·φ(x)
      this._forward(this._featureBuffer!, this._predictionBuffer!);

      // Denormalize output and compute confidence intervals
      const predicted: number[] = new Array(this._outputDimension);
      const lowerBound: number[] = new Array(this._outputDimension);
      const upperBound: number[] = new Array(this._outputDimension);
      const standardError: number[] = new Array(this._outputDimension);

      const outputStats = this._outputStats!;
      const predBuffer = this._predictionBuffer!;

      for (let j = 0; j < this._outputDimension; j++) {
        // Compute output standard deviation from Welford state
        const outputCount = outputStats.count;
        let outputStd: number;
        if (outputCount <= 1) {
          outputStd = 1.0;
        } else {
          outputStd = Math.sqrt(outputStats.m2[j] / (outputCount - 1));
          if (outputStd < this._epsilon) {
            outputStd = 1.0;
          }
        }

        // Denormalize: y = ŷ·σ + μ
        const normalizedPred = predBuffer[j];
        predicted[j] = normalizedPred * outputStd + outputStats.mean[j];

        // Standard error increases with prediction step (uncertainty grows)
        // SE = σ_residual * σ_output * √(1 + step * 0.1)
        const se = residualStd[j] * outputStd * Math.sqrt(1.0 + step * 0.1);
        standardError[j] = se;

        // 95% confidence interval: ±1.96 standard errors
        const margin = 1.96 * se;
        lowerBound[j] = predicted[j] - margin;
        upperBound[j] = predicted[j] + margin;
      }

      predictions.push({
        predicted,
        lowerBound,
        upperBound,
        standardError,
      });
    }

    return {
      predictions,
      accuracy,
      sampleCount: this._sampleCount,
      isModelReady: true,
    };
  }

  /**
   * Returns a summary of the model state
   *
   * @returns ModelSummary with dimensions, training state, and accuracy
   *
   * @example
   * ```typescript
   * const summary = model.getModelSummary();
   * console.log(`Features: ${summary.polynomialFeatureCount}, Accuracy: ${summary.accuracy}`);
   * ```
   */
  getModelSummary(): ModelSummary {
    return {
      isInitialized: this._isInitialized,
      inputDimension: this._inputDimension,
      outputDimension: this._outputDimension,
      polynomialDegree: this._polynomialDegree,
      polynomialFeatureCount: this._featureCount,
      sampleCount: this._sampleCount,
      accuracy: this._getAccuracy(),
      converged: this._converged,
      effectiveLearningRate: this._isInitialized
        ? this._getEffectiveLearningRate()
        : this._learningRate,
      driftCount: this._driftCount,
    };
  }

  /**
   * Returns the current weight information
   *
   * @returns WeightInfo with weights, Adam moments, and update count
   *
   * @example
   * ```typescript
   * const weights = model.getWeights();
   * console.log(`Updates: ${weights.updateCount}`);
   * ```
   */
  getWeights(): WeightInfo {
    if (!this._isInitialized || this._weights === null) {
      return {
        weights: [],
        firstMoment: [],
        secondMoment: [],
        updateCount: 0,
      };
    }

    // Convert flat arrays to 2D for API compatibility
    const weights: number[][] = [];
    const firstMoment: number[][] = [];
    const secondMoment: number[][] = [];

    for (let i = 0; i < this._outputDimension; i++) {
      const wRow: number[] = new Array(this._featureCount);
      const mRow: number[] = new Array(this._featureCount);
      const vRow: number[] = new Array(this._featureCount);

      const offset = i * this._featureCount;
      for (let j = 0; j < this._featureCount; j++) {
        wRow[j] = this._weights![offset + j];
        mRow[j] = this._firstMoment![offset + j];
        vRow[j] = this._secondMoment![offset + j];
      }

      weights.push(wRow);
      firstMoment.push(mRow);
      secondMoment.push(vRow);
    }

    return {
      weights,
      firstMoment,
      secondMoment,
      updateCount: this._updateCount,
    };
  }

  /**
   * Returns the normalization statistics
   *
   * @returns NormalizationStats with mean, std, and count for inputs/outputs
   *
   * @example
   * ```typescript
   * const stats = model.getNormalizationStats();
   * console.log(`Input means: ${stats.inputMean}`);
   * ```
   */
  getNormalizationStats(): NormalizationStats {
    if (
      !this._isInitialized || this._inputStats === null ||
      this._outputStats === null
    ) {
      return {
        inputMean: [],
        inputStd: [],
        outputMean: [],
        outputStd: [],
        count: 0,
      };
    }

    // Convert typed arrays to regular arrays and compute std from M2
    const inputMean: number[] = new Array(this._inputDimension);
    const inputStd: number[] = new Array(this._inputDimension);
    const outputMean: number[] = new Array(this._outputDimension);
    const outputStd: number[] = new Array(this._outputDimension);

    const inputCount = this._inputStats.count;
    const outputCount = this._outputStats.count;

    for (let i = 0; i < this._inputDimension; i++) {
      inputMean[i] = this._inputStats.mean[i];
      if (inputCount <= 1) {
        inputStd[i] = 0;
      } else {
        inputStd[i] = Math.sqrt(this._inputStats.m2[i] / (inputCount - 1));
      }
    }

    for (let i = 0; i < this._outputDimension; i++) {
      outputMean[i] = this._outputStats.mean[i];
      if (outputCount <= 1) {
        outputStd[i] = 0;
      } else {
        outputStd[i] = Math.sqrt(this._outputStats.m2[i] / (outputCount - 1));
      }
    }

    return {
      inputMean,
      inputStd,
      outputMean,
      outputStd,
      count: inputCount,
    };
  }

  /**
   * Resets the model to its initial state
   *
   * Clears all weights, statistics, and training state.
   * Configuration is preserved.
   *
   * @example
   * ```typescript
   * model.reset();
   * // Model is now ready for fresh training
   * ```
   */
  reset(): void {
    // Reset dimensions
    this._inputDimension = 0;
    this._outputDimension = 0;
    this._featureCount = 0;
    this._isInitialized = false;

    // Reset weights and Adam state
    this._weights = null;
    this._firstMoment = null;
    this._secondMoment = null;

    // Reset normalization state
    this._inputStats = null;
    this._outputStats = null;
    this._residualStats = null;

    // Reset training state
    this._updateCount = 0;
    this._sampleCount = 0;
    this._runningLossSum = 0;
    this._runningLossCount = 0;
    this._converged = false;
    this._previousLoss = Infinity;
    this._driftCount = 0;

    // Reset ADWIN state
    this._adwinBuckets = [];
    this._adwinTotal = 0;
    this._adwinCount = 0;

    // Reset buffers
    this._featureBuffer = null;
    this._gradientBuffer = null;
    this._errorBuffer = null;
    this._normalizedInputBuffer = null;
    this._normalizedOutputBuffer = null;
    this._predictionBuffer = null;
    this._tempBuffer = null;

    // Reset polynomial cache
    this._exponentCache = null;

    // Clear object pool
    this._pool.clear();

    // Reset recent samples
    this._recentX = [];
    this._recentY = [];
    this._recentIndex = 0;

    // Reset RNG state
    this._hasSpare = false;
    this._spare = 0;
  }

  // =========================================================================
  // PRIVATE METHODS - INITIALIZATION
  // =========================================================================

  /**
   * Initializes the model with given dimensions
   *
   * Computes polynomial feature count using binomial coefficient:
   * C(n+d, d) = (n+d)! / (n! * d!)
   *
   * This gives the count of all monomials with total degree ≤ d in n variables.
   *
   * @param inputDim - Number of input features
   * @param outputDim - Number of output dimensions
   */
  private _initializeModel(inputDim: number, outputDim: number): void {
    this._inputDimension = inputDim;
    this._outputDimension = outputDim;

    // Compute number of polynomial features: C(n+d, d)
    // This counts all monomials x₁^e₁ · x₂^e₂ · ... · x_n^e_n where Σe_i ≤ d
    this._featureCount = this._computeBinomial(
      inputDim + this._polynomialDegree,
      this._polynomialDegree,
    );

    const totalWeights = outputDim * this._featureCount;

    // Initialize weights with Xavier/Glorot initialization
    // σ = √(2 / (fan_in + fan_out)) for better gradient flow
    this._weights = new Float64Array(totalWeights);
    const scale = Math.sqrt(2.0 / (this._featureCount + outputDim));
    for (let i = 0; i < totalWeights; i++) {
      this._weights[i] = this._randomNormal() * scale;
    }

    // Initialize Adam moments to zero
    this._firstMoment = new Float64Array(totalWeights);
    this._secondMoment = new Float64Array(totalWeights);

    // Initialize normalization statistics (Welford's algorithm)
    this._inputStats = {
      count: 0,
      mean: new Float64Array(inputDim),
      m2: new Float64Array(inputDim),
    };

    this._outputStats = {
      count: 0,
      mean: new Float64Array(outputDim),
      m2: new Float64Array(outputDim),
    };

    this._residualStats = {
      count: 0,
      mean: new Float64Array(outputDim),
      m2: new Float64Array(outputDim),
    };

    // Preallocate buffers for hot paths
    this._featureBuffer = new Float64Array(this._featureCount);
    this._gradientBuffer = new Float64Array(totalWeights);
    this._errorBuffer = new Float64Array(outputDim);
    this._normalizedInputBuffer = new Float64Array(inputDim);
    this._normalizedOutputBuffer = new Float64Array(outputDim);
    this._predictionBuffer = new Float64Array(outputDim);
    this._tempBuffer = new Float64Array(
      Math.max(inputDim, outputDim, this._featureCount),
    );

    // Generate polynomial exponent cache for efficient feature computation
    this._generateExponentCache();

    this._isInitialized = true;
  }

  /**
   * Computes binomial coefficient C(n, k) iteratively
   *
   * Formula: C(n,k) = n! / (k! * (n-k)!) = ∏(i=0 to k-1) (n-i)/(i+1)
   *
   * Uses iterative multiplication to avoid overflow for large values.
   *
   * @param n - Total items
   * @param k - Items to choose
   * @returns Binomial coefficient
   */
  private _computeBinomial(n: number, k: number): number {
    if (k > n - k) {
      k = n - k;
    }
    let result = 1;
    for (let i = 0; i < k; i++) {
      result = (result * (n - i)) / (i + 1);
    }
    return Math.round(result);
  }

  /**
   * Generates exponent cache for polynomial features in graded lexicographic order
   *
   * Each entry is a Uint8Array of exponents [e₁, e₂, ..., e_n] where Σe_i ≤ d
   * Graded lex order: first by total degree, then lexicographically
   *
   * Example for n=2, d=2:
   * [0,0], [1,0], [0,1], [2,0], [1,1], [0,2]
   *  deg0   deg1   deg1   deg2   deg2   deg2
   */
  private _generateExponentCache(): void {
    this._exponentCache = [];
    const n = this._inputDimension;
    const d = this._polynomialDegree;

    // Stack-based iterative generation to avoid recursion overhead
    const current = new Uint8Array(n);

    // Degree 0: constant term [0, 0, ..., 0]
    this._exponentCache.push(new Uint8Array(current));

    // Generate monomials for each total degree from 1 to d
    for (let totalDeg = 1; totalDeg <= d; totalDeg++) {
      this._generateMonomialsOfDegree(n, totalDeg);
    }
  }

  /**
   * Generates all monomials of exactly the given total degree
   * Uses iterative approach with stack simulation
   *
   * @param n - Number of variables
   * @param degree - Exact total degree to generate
   */
  private _generateMonomialsOfDegree(n: number, degree: number): void {
    // Stack-based implementation for efficiency
    const stack: Array<
      { pos: number; remaining: number; exponents: Uint8Array }
    > = [];
    stack.push({ pos: 0, remaining: degree, exponents: new Uint8Array(n) });

    while (stack.length > 0) {
      const { pos, remaining, exponents } = stack.pop()!;

      if (pos === n - 1) {
        // Last position: assign remaining degree
        exponents[pos] = remaining;
        this._exponentCache!.push(new Uint8Array(exponents));
        exponents[pos] = 0;
        continue;
      }

      // Try all possible exponents for current position (descending for graded lex)
      for (let exp = remaining; exp >= 0; exp--) {
        const newExponents = new Uint8Array(exponents);
        newExponents[pos] = exp;
        stack.push({
          pos: pos + 1,
          remaining: remaining - exp,
          exponents: newExponents,
        });
      }
    }
  }

  // =========================================================================
  // PRIVATE METHODS - WELFORD'S ALGORITHM FOR ONLINE STATISTICS
  // =========================================================================

  /**
   * Updates Welford's online statistics with a new sample
   *
   * Welford's algorithm for numerically stable running mean and variance:
   * δ = x - μ
   * μ += δ/n
   * M₂ += δ(x - μ)
   * σ² = M₂/(n-1)
   *
   * @param state - Welford state to update
   * @param sample - New sample values (regular array)
   */
  private _updateWelfordStats(state: WelfordState, sample: number[]): void {
    state.count++;
    const n = state.count;
    const mean = state.mean;
    const m2 = state.m2;

    // Unrolled loop for common small dimensions
    const len = sample.length;
    let i = 0;

    // Process 4 elements at a time when possible
    const len4 = len - (len % 4);
    for (; i < len4; i += 4) {
      const x0 = sample[i];
      const x1 = sample[i + 1];
      const x2 = sample[i + 2];
      const x3 = sample[i + 3];

      const delta0 = x0 - mean[i];
      const delta1 = x1 - mean[i + 1];
      const delta2 = x2 - mean[i + 2];
      const delta3 = x3 - mean[i + 3];

      mean[i] += delta0 / n;
      mean[i + 1] += delta1 / n;
      mean[i + 2] += delta2 / n;
      mean[i + 3] += delta3 / n;

      m2[i] += delta0 * (x0 - mean[i]);
      m2[i + 1] += delta1 * (x1 - mean[i + 1]);
      m2[i + 2] += delta2 * (x2 - mean[i + 2]);
      m2[i + 3] += delta3 * (x3 - mean[i + 3]);
    }

    // Handle remaining elements
    for (; i < len; i++) {
      const x = sample[i];
      const delta = x - mean[i];
      mean[i] += delta / n;
      m2[i] += delta * (x - mean[i]);
    }
  }

  /**
   * Updates Welford's statistics with a typed array sample
   *
   * @param state - Welford state to update
   * @param sample - New sample values (Float64Array)
   */
  private _updateWelfordStatsTyped(
    state: WelfordState,
    sample: Float64Array,
  ): void {
    state.count++;
    const n = state.count;
    const mean = state.mean;
    const m2 = state.m2;
    const len = sample.length;

    for (let i = 0; i < len; i++) {
      const x = sample[i];
      const delta = x - mean[i];
      mean[i] += delta / n;
      m2[i] += delta * (x - mean[i]);
    }
  }

  // =========================================================================
  // PRIVATE METHODS - NORMALIZATION
  // =========================================================================

  /**
   * Normalizes a vector using z-score normalization
   *
   * Formula: x̃ = (x - μ)/(σ + ε)
   *
   * Uses sample standard deviation: σ = √(M₂/(n-1))
   * Falls back to σ=1 when insufficient samples or near-zero variance.
   *
   * @param input - Input values (regular array)
   * @param stats - Welford statistics for mean and variance
   * @param output - Output buffer for normalized values
   */
  private _normalizeVector(
    input: number[],
    stats: WelfordState,
    output: Float64Array,
  ): void {
    const n = stats.count;
    const mean = stats.mean;
    const m2 = stats.m2;
    const eps = this._epsilon;
    const len = input.length;

    if (n <= 1) {
      // Not enough samples: center only
      for (let i = 0; i < len; i++) {
        output[i] = input[i] - mean[i];
      }
      return;
    }

    const varianceDenom = n - 1;

    for (let i = 0; i < len; i++) {
      // σ = √(M₂/(n-1))
      let std = Math.sqrt(m2[i] / varianceDenom);
      if (std < eps) {
        std = 1.0; // Avoid division by near-zero
      }
      output[i] = (input[i] - mean[i]) / (std + eps);
    }
  }

  /**
   * Normalizes a typed array using z-score normalization
   *
   * @param input - Input values (Float64Array)
   * @param stats - Welford statistics
   * @param output - Output buffer
   */
  private _normalizeTypedArray(
    input: Float64Array,
    stats: WelfordState,
    output: Float64Array,
  ): void {
    const n = stats.count;
    const mean = stats.mean;
    const m2 = stats.m2;
    const eps = this._epsilon;
    const len = input.length;

    if (n <= 1) {
      for (let i = 0; i < len; i++) {
        output[i] = input[i] - mean[i];
      }
      return;
    }

    const varianceDenom = n - 1;

    for (let i = 0; i < len; i++) {
      let std = Math.sqrt(m2[i] / varianceDenom);
      if (std < eps) {
        std = 1.0;
      }
      output[i] = (input[i] - mean[i]) / (std + eps);
    }
  }

  // =========================================================================
  // PRIVATE METHODS - POLYNOMIAL FEATURES
  // =========================================================================

  /**
   * Generates polynomial features φ(x) for normalized input
   *
   * Computes all monomials of degree ≤ d:
   * φ(x) = [1, x₁, x₂, ..., x₁², x₁x₂, ..., x_n^d]
   *
   * Each monomial x₁^e₁ · x₂^e₂ · ... · x_n^e_n where Σe_i ≤ d
   *
   * Uses cached exponents and optimized power computation for small exponents.
   *
   * @param input - Normalized input vector
   * @param output - Output buffer for polynomial features
   */
  private _generatePolynomialFeatures(
    input: Float64Array,
    output: Float64Array,
  ): void {
    const exponents = this._exponentCache!;
    const inputDim = this._inputDimension;
    const featureCount = this._featureCount;

    for (let f = 0; f < featureCount; f++) {
      const exp = exponents[f];
      let value = 1.0;

      // Compute product x₁^e₁ · x₂^e₂ · ... · x_n^e_n
      for (let i = 0; i < inputDim; i++) {
        const e = exp[i];
        if (e === 0) continue;

        const x = input[i];
        // Optimized power computation for common small exponents
        switch (e) {
          case 1:
            value *= x;
            break;
          case 2:
            value *= x * x;
            break;
          case 3:
            value *= x * x * x;
            break;
          case 4:
            const x2 = x * x;
            value *= x2 * x2;
            break;
          default:
            // Fall back to Math.pow for larger exponents
            value *= Math.pow(x, e);
        }
      }

      // Clamp to prevent numerical overflow
      if (!isFinite(value)) {
        value = value > 0 ? 1e10 : -1e10;
      }

      output[f] = value;
    }
  }

  // =========================================================================
  // PRIVATE METHODS - FORWARD PASS
  // =========================================================================

  /**
   * Forward pass: ŷ = W·φ(x)
   *
   * Computes matrix-vector product with flattened weight matrix.
   * W is stored as [outputDim × featureCount] in row-major order.
   *
   * @param features - Polynomial feature vector φ(x)
   * @param output - Output buffer for predictions ŷ
   */
  private _forward(features: Float64Array, output: Float64Array): void {
    const weights = this._weights!;
    const outputDim = this._outputDimension;
    const featureCount = this._featureCount;

    for (let i = 0; i < outputDim; i++) {
      let sum = 0;
      const offset = i * featureCount;

      // Unrolled inner loop for better performance
      const fc4 = featureCount - (featureCount % 4);
      let j = 0;

      for (; j < fc4; j += 4) {
        sum += weights[offset + j] * features[j] +
          weights[offset + j + 1] * features[j + 1] +
          weights[offset + j + 2] * features[j + 2] +
          weights[offset + j + 3] * features[j + 3];
      }

      for (; j < featureCount; j++) {
        sum += weights[offset + j] * features[j];
      }

      output[i] = sum;
    }
  }

  // =========================================================================
  // PRIVATE METHODS - LOSS AND ERROR
  // =========================================================================

  /**
   * Computes prediction error: e = y - ŷ
   *
   * @param target - Target values y
   * @param prediction - Predicted values ŷ
   * @param error - Output buffer for error e
   */
  private _computeError(
    target: Float64Array,
    prediction: Float64Array,
    error: Float64Array,
  ): void {
    const len = target.length;
    for (let i = 0; i < len; i++) {
      error[i] = target[i] - prediction[i];
    }
  }

  /**
   * Computes mean squared error loss: L = (1/2)‖e‖²
   *
   * @param error - Error vector
   * @returns MSE loss value
   */
  private _computeMSELoss(error: Float64Array): number {
    const len = error.length;
    let sum = 0;

    // Unrolled loop for performance
    const len4 = len - (len % 4);
    let i = 0;

    for (; i < len4; i += 4) {
      const e0 = error[i];
      const e1 = error[i + 1];
      const e2 = error[i + 2];
      const e3 = error[i + 3];
      sum += e0 * e0 + e1 * e1 + e2 * e2 + e3 * e3;
    }

    for (; i < len; i++) {
      sum += error[i] * error[i];
    }

    return sum * 0.5;
  }

  // =========================================================================
  // PRIVATE METHODS - GRADIENT COMPUTATION
  // =========================================================================

  /**
   * Computes gradient and stores in gradient buffer
   *
   * Gradient formula: g = -(e ⊗ φ) + λW
   *
   * Where:
   * - e is the error vector
   * - φ is the feature vector
   * - λ is the regularization strength
   * - ⊗ is the outer product
   *
   * @param error - Error vector e
   * @param features - Feature vector φ
   * @param weight - Sample weight (for outlier downweighting)
   * @returns L2 norm of gradient
   */
  private _computeGradient(
    error: Float64Array,
    features: Float64Array,
    weight: number,
  ): number {
    const gradient = this._gradientBuffer!;
    const weights = this._weights!;
    const lambda = this._regularizationStrength;
    const outputDim = this._outputDimension;
    const featureCount = this._featureCount;

    let normSum = 0;

    for (let i = 0; i < outputDim; i++) {
      const offset = i * featureCount;
      const e = -error[i] * weight; // Negative because gradient of (y - ŷ)² is -2(y - ŷ)

      for (let j = 0; j < featureCount; j++) {
        // g_ij = -e_i * φ_j + λ * W_ij
        const grad = e * features[j] + lambda * weights[offset + j];
        gradient[offset + j] = grad;
        normSum += grad * grad;
      }
    }

    return Math.sqrt(normSum);
  }

  /**
   * Accumulates gradient into buffer (for mini-batch)
   *
   * Accumulates: g += -(e ⊗ φ)
   *
   * @param error - Error vector
   * @param features - Feature vector
   * @param accumulator - Gradient accumulator buffer
   */
  private _accumulateGradient(
    error: Float64Array,
    features: Float64Array,
    accumulator: Float64Array,
  ): void {
    const outputDim = this._outputDimension;
    const featureCount = this._featureCount;

    for (let i = 0; i < outputDim; i++) {
      const offset = i * featureCount;
      const e = -error[i];

      for (let j = 0; j < featureCount; j++) {
        accumulator[offset + j] += e * features[j];
      }
    }
  }

  // =========================================================================
  // PRIVATE METHODS - ADAM OPTIMIZER
  // =========================================================================

  /**
   * Gets effective learning rate with warmup and cosine decay
   *
   * Learning rate schedule:
   * - Warmup phase (t < warmup_steps): η_t = η * t / warmup_steps
   * - Decay phase: η_t = η * 0.5 * (1 + cos(π * progress))
   *
   * @returns Current effective learning rate
   */
  private _getEffectiveLearningRate(): number {
    const t = this._updateCount;

    if (t < this._warmupSteps) {
      // Linear warmup: η_t = η * (t+1) / warmup_steps
      return this._learningRate * (t + 1) / this._warmupSteps;
    }

    // Cosine decay after warmup
    const warmupSteps = this._warmupSteps;
    const totalSteps = this._totalSteps;
    const decaySteps = totalSteps - warmupSteps;

    if (decaySteps <= 0) {
      return this._learningRate;
    }

    const progress = Math.min(1.0, (t - warmupSteps) / decaySteps);
    // η_t = η * 0.5 * (1 + cos(π * progress))
    const cosineDecay = 0.5 * (1.0 + Math.cos(Math.PI * progress));

    return this._learningRate * cosineDecay;
  }

  /**
   * Performs Adam optimizer update step
   *
   * Adam algorithm:
   * m = β₁m + (1-β₁)g         (first moment estimate)
   * v = β₂v + (1-β₂)g²        (second moment estimate)
   * m̂ = m / (1 - β₁ᵗ)        (bias correction)
   * v̂ = v / (1 - β₂ᵗ)        (bias correction)
   * W = W - η * m̂ / (√v̂ + ε)  (parameter update)
   *
   * @param lr - Current learning rate
   */
  private _adamUpdate(lr: number): void {
    const t = this._updateCount;
    const beta1 = this._beta1;
    const beta2 = this._beta2;
    const eps = this._epsilon;

    // Compute bias correction factors
    // (1 - β₁ᵗ) and (1 - β₂ᵗ)
    const beta1Power = Math.pow(beta1, t);
    const beta2Power = Math.pow(beta2, t);
    const mCorrection = 1.0 / (1.0 - beta1Power);
    const vCorrection = 1.0 / (1.0 - beta2Power);

    const gradient = this._gradientBuffer!;
    const weights = this._weights!;
    const firstMoment = this._firstMoment!;
    const secondMoment = this._secondMoment!;
    const totalWeights = weights.length;

    for (let i = 0; i < totalWeights; i++) {
      const g = gradient[i];

      // Update biased first moment: m = β₁m + (1-β₁)g
      const m = beta1 * firstMoment[i] + (1.0 - beta1) * g;
      firstMoment[i] = m;

      // Update biased second moment: v = β₂v + (1-β₂)g²
      const v = beta2 * secondMoment[i] + (1.0 - beta2) * g * g;
      secondMoment[i] = v;

      // Compute bias-corrected estimates
      const mHat = m * mCorrection;
      const vHat = v * vCorrection;

      // Update weights: W -= η * m̂ / (√v̂ + ε)
      weights[i] -= lr * mHat / (Math.sqrt(vHat) + eps);
    }
  }

  /**
   * Performs Adam update with precomputed gradient
   *
   * @param lr - Current learning rate
   * @param gradient - Precomputed gradient
   */
  private _adamUpdateWithGradient(lr: number, gradient: Float64Array): void {
    const t = this._updateCount;
    const beta1 = this._beta1;
    const beta2 = this._beta2;
    const eps = this._epsilon;

    const beta1Power = Math.pow(beta1, t);
    const beta2Power = Math.pow(beta2, t);
    const mCorrection = 1.0 / (1.0 - beta1Power);
    const vCorrection = 1.0 / (1.0 - beta2Power);

    const weights = this._weights!;
    const firstMoment = this._firstMoment!;
    const secondMoment = this._secondMoment!;
    const totalWeights = weights.length;

    for (let i = 0; i < totalWeights; i++) {
      const g = gradient[i];

      const m = beta1 * firstMoment[i] + (1.0 - beta1) * g;
      firstMoment[i] = m;

      const v = beta2 * secondMoment[i] + (1.0 - beta2) * g * g;
      secondMoment[i] = v;

      const mHat = m * mCorrection;
      const vHat = v * vCorrection;

      weights[i] -= lr * mHat / (Math.sqrt(vHat) + eps);
    }
  }

  // =========================================================================
  // PRIVATE METHODS - OUTLIER DETECTION
  // =========================================================================

  /**
   * Detects if the current residual is an outlier
   *
   * Uses z-score of residual: r = (e - μ_e) / σ_e
   * Sample is outlier if |r| > outlierThreshold for any dimension
   *
   * @param error - Current prediction error
   * @returns true if outlier detected
   */
  private _detectOutlier(error: Float64Array): boolean {
    const residualStats = this._residualStats!;

    // Need sufficient samples for reliable outlier detection
    if (residualStats.count < 20) {
      return false;
    }

    const n = residualStats.count;
    const mean = residualStats.mean;
    const m2 = residualStats.m2;
    const threshold = this._outlierThreshold;
    const eps = this._epsilon;
    const outputDim = this._outputDimension;

    for (let i = 0; i < outputDim; i++) {
      const std = Math.sqrt(m2[i] / (n - 1));
      if (std < eps) continue;

      const zScore = Math.abs(error[i] - mean[i]) / std;
      if (zScore > threshold) {
        return true;
      }
    }

    return false;
  }

  // =========================================================================
  // PRIVATE METHODS - ADWIN DRIFT DETECTION
  // =========================================================================

  /**
   * Detects concept drift using ADWIN algorithm
   *
   * ADWIN (ADaptive WINdowing) maintains a window of recent losses and
   * detects drift when the means of two subwindows differ significantly.
   *
   * Detection condition: |μ₀ - μ₁| ≥ ε_cut(δ)
   * where ε_cut = √((2/m)·ln(2/δ)) and m = harmonic mean of subwindow sizes
   *
   * @param loss - Current sample loss
   * @returns true if drift detected
   */
  private _detectDrift(loss: number): boolean {
    // Add loss to ADWIN window
    this._adwinBuckets.push({
      total: loss,
      variance: 0,
      count: 1,
    });
    this._adwinTotal += loss;
    this._adwinCount++;

    // Compress buckets to maintain logarithmic memory
    this._compressADWINBuckets();

    // Need minimum samples for drift detection
    if (this._adwinCount < 30) {
      return false;
    }

    // Check for drift by testing different cut points
    let leftTotal = 0;
    let leftCount = 0;
    const buckets = this._adwinBuckets;
    const totalCount = this._adwinCount;
    const total = this._adwinTotal;
    const delta = this._adwinDelta;

    for (let i = 0; i < buckets.length - 1; i++) {
      const bucket = buckets[i];
      leftTotal += bucket.total;
      leftCount += bucket.count;

      const rightTotal = total - leftTotal;
      const rightCount = totalCount - leftCount;

      if (leftCount < 5 || rightCount < 5) continue;

      const leftMean = leftTotal / leftCount;
      const rightMean = rightTotal / rightCount;

      // Harmonic mean of window sizes
      const m = (leftCount * rightCount) / (leftCount + rightCount);

      // Hoeffding bound: ε_cut = √((2/m)·ln(2/δ))
      const epsilonCut = Math.sqrt((2.0 / m) * Math.log(2.0 / delta));

      if (Math.abs(leftMean - rightMean) >= epsilonCut) {
        return true;
      }
    }

    return false;
  }

  /**
   * Compresses ADWIN buckets using exponential histogram technique
   * Maintains O(log(W)) buckets for window of size W
   */
  private _compressADWINBuckets(): void {
    const maxBuckets = 64;

    while (this._adwinBuckets.length > maxBuckets) {
      // Merge two oldest buckets
      const b1 = this._adwinBuckets.shift()!;
      if (this._adwinBuckets.length > 0) {
        const b2 = this._adwinBuckets[0];
        b2.total += b1.total;
        b2.count += b1.count;
      }
    }
  }

  /**
   * Handles detected drift by partially resetting model state
   *
   * Strategy:
   * - Shrink ADWIN window to recent data
   * - Decay Adam momentum (keep some learning)
   * - Partially reset loss tracking
   */
  private _handleDrift(): void {
    this._driftCount++;

    // Shrink ADWIN window to focus on recent data
    while (this._adwinBuckets.length > 10) {
      const removed = this._adwinBuckets.shift()!;
      this._adwinTotal -= removed.total;
      this._adwinCount -= removed.count;
    }

    // Decay Adam momentum to allow faster adaptation
    const decayFactor = 0.5;
    const firstMoment = this._firstMoment!;
    const secondMoment = this._secondMoment!;
    const len = firstMoment.length;

    for (let i = 0; i < len; i++) {
      firstMoment[i] *= decayFactor;
      secondMoment[i] *= decayFactor;
    }

    // Partially reset loss tracking
    this._runningLossSum *= 0.5;
    this._runningLossCount = Math.max(
      1,
      Math.floor(this._runningLossCount * 0.5),
    );

    // Reset convergence flag
    this._converged = false;
  }

  // =========================================================================
  // PRIVATE METHODS - PREDICTION HELPERS
  // =========================================================================

  /**
   * Computes model accuracy metric: 1/(1 + L̄)
   *
   * where L̄ = Σ Loss / n is the running average loss
   *
   * @returns Accuracy in range (0, 1]
   */
  private _getAccuracy(): number {
    if (this._runningLossCount === 0) {
      return 0;
    }
    const avgLoss = this._runningLossSum / this._runningLossCount;
    return 1.0 / (1.0 + avgLoss);
  }

  /**
   * Gets residual standard deviation for confidence intervals
   *
   * @returns Array of std dev for each output dimension
   */
  private _getResidualStd(): number[] {
    const residualStats = this._residualStats!;
    const outputDim = this._outputDimension;
    const std: number[] = new Array(outputDim);

    const n = residualStats.count;
    const m2 = residualStats.m2;
    const eps = this._epsilon;

    if (n <= 1) {
      for (let i = 0; i < outputDim; i++) {
        std[i] = 1.0;
      }
      return std;
    }

    for (let i = 0; i < outputDim; i++) {
      const variance = m2[i] / (n - 1);
      std[i] = Math.sqrt(variance);
      if (std[i] < eps) {
        std[i] = eps;
      }
    }

    return std;
  }

  /**
   * Stores a recent sample for prediction
   * Uses circular buffer to maintain fixed memory
   *
   * @param x - Input sample
   * @param y - Output sample
   */
  private _storeRecentSample(x: number[], y: number[]): void {
    const xCopy = new Float64Array(x.length);
    const yCopy = new Float64Array(y.length);

    for (let i = 0; i < x.length; i++) {
      xCopy[i] = x[i];
    }
    for (let i = 0; i < y.length; i++) {
      yCopy[i] = y[i];
    }

    if (this._recentX.length < this._maxRecentSamples) {
      this._recentX.push(xCopy);
      this._recentY.push(yCopy);
    } else {
      this._recentX[this._recentIndex] = xCopy;
      this._recentY[this._recentIndex] = yCopy;
      this._recentIndex = (this._recentIndex + 1) % this._maxRecentSamples;
    }
  }

  /**
   * Gets input for prediction at given step
   *
   * For general regression, uses the most recent input.
   * For time series, could extrapolate (not implemented here).
   *
   * @param step - Prediction step (0 = next)
   * @returns Input for prediction or null if unavailable
   */
  private _getInputForPrediction(step: number): Float64Array | null {
    if (this._recentX.length === 0) {
      return null;
    }

    // Get most recent sample index
    const lastIdx = this._recentX.length < this._maxRecentSamples
      ? this._recentX.length - 1
      : (this._recentIndex - 1 + this._maxRecentSamples) %
        this._maxRecentSamples;

    return this._recentX[lastIdx];
  }

  // =========================================================================
  // PRIVATE METHODS - UTILITIES
  // =========================================================================

  /**
   * Generates a random number from standard normal distribution
   * Uses Box-Muller transform with cached spare value
   *
   * @returns Random sample from N(0, 1)
   */
  private _randomNormal(): number {
    if (this._hasSpare) {
      this._hasSpare = false;
      return this._spare;
    }

    let u: number, v: number, s: number;

    do {
      u = Math.random() * 2.0 - 1.0;
      v = Math.random() * 2.0 - 1.0;
      s = u * u + v * v;
    } while (s >= 1.0 || s === 0.0);

    const mul = Math.sqrt(-2.0 * Math.log(s) / s);
    this._spare = v * mul;
    this._hasSpare = true;

    return u * mul;
  }

  /**
   * Fisher-Yates shuffle for index array
   * In-place, O(n) time complexity
   *
   * @param indices - Index array to shuffle
   */
  private _shuffleIndices(indices: Uint32Array): void {
    const n = indices.length;

    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      // Swap
      const temp = indices[i];
      indices[i] = indices[j];
      indices[j] = temp;
    }
  }
}

// ============================================================================
// EXPORTS
// ============================================================================

export default MultivariatePolynomialRegression;
