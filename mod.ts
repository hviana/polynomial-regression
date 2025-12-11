/* ============================================================================
 * Multivariate Polynomial Regression with Online RLS Learning
 * ============================================================================
 * - Incremental Recursive Least Squares (RLS)
 * - Multivariate inputs & outputs
 * - Polynomial feature expansion
 * - Optional normalization (min-max / z-score)
 * - Confidence intervals with configurable confidence level
 * - Extremely memory- and CPU-conscious implementation
 * ========================================================================== */

export type NormalizationMethod = "none" | "min-max" | "z-score";

export interface MultivariatePolynomialRegressionConfig {
  polynomialDegree?: number;
  enableNormalization?: boolean;
  normalizationMethod?: NormalizationMethod;
  forgettingFactor?: number;
  initialCovariance?: number;
  regularization?: number;
  confidenceLevel?: number;
}

/** Parameters for incremental training. */
export interface FitOnlineParams {
  /** Input features: [samples][features]. */
  xCoordinates: number[][];
  /** Output targets: [samples][outputs]. */
  yCoordinates: number[][];
}

/** Parameters for prediction. */
export interface PredictParams {
  /**
   * Number of extrapolated predictions.
   * Used when `inputPoints` is not provided.
   */
  futureSteps: number;
  /**
   * Optional specific points to predict:
   * - If provided and non-empty, `futureSteps` is ignored.
   * - Shape: [samples][features]
   */
  inputPoints?: number[][];
}

/** Single prediction with confidence interval. */
export interface SinglePrediction {
  predicted: number[];
  lowerBound: number[];
  upperBound: number[];
  standardError: number[];
}

/** Global prediction result and model diagnostics. */
export interface PredictionResult {
  predictions: SinglePrediction[];
  confidenceLevel: number;
  rSquared: number;
  rmse: number;
  sampleCount: number;
  isModelReady: boolean;
}

/** Normalization statistics snapshot. */
export interface NormalizationStats {
  min: number[];
  max: number[];
  mean: number[];
  std: number[];
  count: number;
}

/** Model summary for inspection/debugging. */
export interface ModelSummary {
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
}

/* ============================================================================
 * Internal Helpers
 * ========================================================================== */

/**
 * Generates polynomial feature exponents and transforms normalized inputs into
 * polynomial feature vectors.
 *
 * Features include all monomials up to a given total degree:
 *   degree 0..D of variables x₁..x_d
 */
class PolynomialFeatureGenerator {
  public readonly inputDimension: number;
  public readonly degree: number;
  public readonly featureCount: number;

  // Flattened exponent matrix: [featureIndex * inputDimension + dimIndex]
  private readonly exponents: Int16Array;

  constructor(inputDimension: number, degree: number) {
    if (inputDimension <= 0) {
      throw new Error("inputDimension must be positive.");
    }
    if (degree < 1) {
      throw new Error("polynomialDegree must be ≥ 1.");
    }

    this.inputDimension = inputDimension;
    this.degree = degree;
    this.featureCount = PolynomialFeatureGenerator.computeFeatureCount(
      inputDimension,
      degree,
    );

    this.exponents = new Int16Array(this.featureCount * this.inputDimension);
    this.populateExponents();
  }

  /**
   * Number of polynomial features including constant term:
   * sum_{k=0}^D C(d + k - 1, k)
   */
  public static computeFeatureCount(d: number, degree: number): number {
    let total = 0;
    for (let k = 0; k <= degree; k++) {
      total += PolynomialFeatureGenerator.combination(d + k - 1, k);
    }
    return total;
  }

  private static combination(n: number, k: number): number {
    if (k < 0 || k > n) return 0;
    if (k === 0 || k === n) return 1;
    k = Math.min(k, n - k);
    let result = 1;
    for (let i = 1; i <= k; i++) {
      result = (result * (n - k + i)) / i;
    }
    return Math.round(result);
  }

  /**
   * Populate the exponents table for all monomials of degree ≤ D.
   * Uses a DFS over integer compositions.
   */
  private populateExponents(): void {
    const d = this.inputDimension;
    const degree = this.degree;
    const current = new Int16Array(d);
    let featureIndex = 0;

    const exps = this.exponents;

    const recurse = (dimIndex: number, remaining: number): void => {
      if (dimIndex === d - 1) {
        // Last dimension: assign exponents from 0..remaining
        for (let e = 0; e <= remaining; e++) {
          current[dimIndex] = e as number;
          let offset = featureIndex * d;
          for (let j = 0; j < d; j++) {
            exps[offset + j] = current[j];
          }
          featureIndex++;
        }
        return;
      }

      for (let e = 0; e <= remaining; e++) {
        current[dimIndex] = e as number;
        recurse(dimIndex + 1, remaining - e);
      }
    };

    recurse(0, degree);

    if (featureIndex !== this.featureCount) {
      // This should never happen; guard for logical errors.
      throw new Error(
        `Internal error: expected ${this.featureCount} features, got ${featureIndex}.`,
      );
    }
  }

  /**
   * Transform a normalized input vector into its polynomial feature vector.
   *
   * @param normalizedInput - Input vector of length `inputDimension`.
   * @param outFeatures - Output buffer of length `featureCount`.
   */
  public transform(
    normalizedInput: Float64Array,
    outFeatures: Float64Array,
  ): void {
    const d = this.inputDimension;
    const fc = this.featureCount;
    const exps = this.exponents;

    if (normalizedInput.length !== d) {
      throw new Error("Normalized input dimension mismatch.");
    }
    if (outFeatures.length !== fc) {
      throw new Error("Feature buffer length mismatch.");
    }

    let expOffset = 0;
    for (let f = 0; f < fc; f++) {
      let value = 1.0;
      for (let dim = 0; dim < d; dim++) {
        const exponent = exps[expOffset++];
        if (exponent > 0) {
          value *= PolynomialFeatureGenerator.powInt(
            normalizedInput[dim],
            exponent,
          );
        }
      }
      outFeatures[f] = value;
    }
  }

  /** Small integer power for performance (degree is usually tiny). */
  private static powInt(base: number, exp: number): number {
    if (exp === 0) return 1.0;
    let result = 1.0;
    for (let i = 0; i < exp; i++) {
      result *= base;
    }
    return result;
  }
}

/**
 * Online normalizer for min-max or z-score scaling.
 * Uses Welford's algorithm for numerically stable mean/std computation.
 */
class OnlineNormalizer {
  public readonly dimension: number;
  public readonly method: NormalizationMethod;

  private readonly min: Float64Array;
  private readonly max: Float64Array;
  private readonly mean: Float64Array;
  private readonly m2: Float64Array;
  private count: number;

  constructor(dimension: number, method: NormalizationMethod) {
    this.dimension = dimension;
    this.method = method;
    this.min = new Float64Array(dimension);
    this.max = new Float64Array(dimension);
    this.mean = new Float64Array(dimension);
    this.m2 = new Float64Array(dimension);
    this.count = 0;
  }

  /** Update statistics with a new raw sample (does NOT transform it). */
  public update(sample: number[]): void {
    const d = this.dimension;
    if (sample.length !== d) {
      throw new Error("Normalizer.update: dimension mismatch.");
    }

    const newCount = this.count + 1;

    if (this.count === 0) {
      for (let i = 0; i < d; i++) {
        const x = sample[i];
        this.min[i] = x;
        this.max[i] = x;
        this.mean[i] = x;
        this.m2[i] = 0.0;
      }
      this.count = newCount;
      return;
    }

    for (let i = 0; i < d; i++) {
      const x = sample[i];

      if (x < this.min[i]) this.min[i] = x;
      if (x > this.max[i]) this.max[i] = x;

      const oldMean = this.mean[i];
      const delta = x - oldMean;
      const newMean = oldMean + delta / newCount;
      const delta2 = x - newMean;
      this.mean[i] = newMean;
      this.m2[i] += delta * delta2;
    }

    this.count = newCount;
  }

  /**
   * Apply normalization to a sample using current stats.
   * Does NOT update statistics; safe to use during prediction.
   */
  public transform(sample: number[], out: Float64Array): void {
    const d = this.dimension;
    if (sample.length !== d) {
      throw new Error("Normalizer.transform: dimension mismatch.");
    }
    if (out.length !== d) {
      throw new Error("Normalizer.transform: output buffer mismatch.");
    }

    if (this.method === "none" || this.count === 0) {
      for (let i = 0; i < d; i++) {
        out[i] = sample[i];
      }
      return;
    }

    if (this.method === "min-max") {
      for (let i = 0; i < d; i++) {
        const mn = this.min[i];
        const mx = this.max[i];
        const denom = mx - mn;
        if (denom > 0) {
          out[i] = (sample[i] - mn) / denom;
        } else {
          out[i] = 0.0;
        }
      }
      return;
    }

    // z-score
    const n = this.count;
    for (let i = 0; i < d; i++) {
      let std = 0.0;
      if (n > 1) {
        const variance = this.m2[i] / (n - 1);
        std = variance > 0 ? Math.sqrt(variance) : 0;
      }
      if (std > 0) {
        out[i] = (sample[i] - this.mean[i]) / std;
      } else {
        out[i] = 0.0;
      }
    }
  }

  /** Snapshot of current normalization statistics. */
  public getStats(): NormalizationStats {
    const d = this.dimension;
    const min = new Array<number>(d);
    const max = new Array<number>(d);
    const mean = new Array<number>(d);
    const std = new Array<number>(d);

    const n = this.count;
    for (let i = 0; i < d; i++) {
      min[i] = this.min[i];
      max[i] = this.max[i];
      mean[i] = this.mean[i];
      if (n > 1) {
        const variance = this.m2[i] / (n - 1);
        std[i] = variance > 0 ? Math.sqrt(variance) : 0.0;
      } else {
        std[i] = 0.0;
      }
    }

    return {
      min,
      max,
      mean,
      std,
      count: this.count,
    };
  }

  /** Internal: number of samples used for stats. */
  public getCount(): number {
    return this.count;
  }
}

/**
 * Internal state for Recursive Least Squares with multiple outputs.
 * Maintains:
 * - Weight matrix W: [features][outputs]
 * - Covariance matrix P: [features][features]
 * - Residual statistics for RMSE and R²
 */
class RLSModelState {
  public readonly featureCount: number;
  public readonly outputDimension: number;

  private readonly forgettingFactor: number;
  private readonly regularization: number;

  // Flattened matrices for cache-friendly access.
  private readonly weights: Float64Array; // size: featureCount * outputDimension
  private readonly covariance: Float64Array; // size: featureCount * featureCount

  // Scratch buffers reused across updates and predictions.
  private readonly gainVector: Float64Array; // size: featureCount
  private readonly tmpVector: Float64Array; // size: featureCount
  private readonly predictionBuffer: Float64Array; // size: outputDimension
  private readonly errorBuffer: Float64Array; // size: outputDimension

  // Streaming residual statistics.
  private sampleCount: number;
  private readonly residualSSE: Float64Array; // per output
  private readonly yMean: Float64Array; // per output
  private readonly yM2: Float64Array; // per output

  constructor(
    featureCount: number,
    outputDimension: number,
    initialCovariance: number,
    regularization: number,
    forgettingFactor: number,
  ) {
    this.featureCount = featureCount;
    this.outputDimension = outputDimension;
    this.forgettingFactor = forgettingFactor;
    this.regularization = regularization;

    this.weights = new Float64Array(featureCount * outputDimension);
    this.covariance = new Float64Array(featureCount * featureCount);

    this.gainVector = new Float64Array(featureCount);
    this.tmpVector = new Float64Array(featureCount);
    this.predictionBuffer = new Float64Array(outputDimension);
    this.errorBuffer = new Float64Array(outputDimension);

    this.sampleCount = 0;
    this.residualSSE = new Float64Array(outputDimension);
    this.yMean = new Float64Array(outputDimension);
    this.yM2 = new Float64Array(outputDimension);

    // Initialize covariance to scaled identity.
    const P = this.covariance;
    const scaling = initialCovariance + regularization;
    for (let i = 0; i < featureCount; i++) {
      const base = i * featureCount;
      for (let j = 0; j < featureCount; j++) {
        P[base + j] = i === j ? scaling : 0.0;
      }
    }
  }

  /** Current number of samples processed. */
  public getSampleCount(): number {
    return this.sampleCount;
  }

  /** Predict output given a feature vector φ. */
  public predict(features: Float64Array, out: Float64Array): void {
    const p = this.featureCount;
    const m = this.outputDimension;
    if (features.length !== p) {
      throw new Error("RLSModelState.predict: feature length mismatch.");
    }
    if (out.length !== m) {
      throw new Error("RLSModelState.predict: output buffer mismatch.");
    }

    const W = this.weights;
    for (let j = 0; j < m; j++) {
      let sum = 0.0;
      let idx = j;
      for (let i = 0; i < p; i++) {
        sum += W[idx] * features[i];
        idx += m;
      }
      out[j] = sum;
    }
  }

  /**
   * RLS update with a new sample:
   * - φ: feature vector
   * - y: target vector
   */
  public update(features: Float64Array, target: number[]): void {
    const p = this.featureCount;
    const m = this.outputDimension;
    if (features.length !== p) {
      throw new Error("RLSModelState.update: feature length mismatch.");
    }
    if (target.length !== m) {
      throw new Error("RLSModelState.update: target length mismatch.");
    }

    const W = this.weights;
    const P = this.covariance;
    const v = this.tmpVector;
    const k = this.gainVector;
    const yHat = this.predictionBuffer;
    const err = this.errorBuffer;

    // 1. Predict with current weights: ŷ = Wᵀ φ
    for (let j = 0; j < m; j++) {
      let sum = 0.0;
      let idx = j;
      for (let i = 0; i < p; i++) {
        sum += W[idx] * features[i];
        idx += m;
      }
      yHat[j] = sum;
    }

    // 2. v = P φ
    for (let i = 0; i < p; i++) {
      let sum = 0.0;
      const rowStart = i * p;
      for (let j = 0; j < p; j++) {
        sum += P[rowStart + j] * features[j];
      }
      v[i] = sum;
    }

    // 3. s = λ + φᵀ v + regularization
    let s = this.forgettingFactor;
    for (let i = 0; i < p; i++) {
      s += features[i] * v[i];
    }
    s += this.regularization;
    if (s <= 0) {
      s = this.regularization;
    }
    const invS = 1.0 / s;

    // 4. Gain vector k = v / s
    for (let i = 0; i < p; i++) {
      k[i] = v[i] * invS;
    }

    // 5. Error vector e = y - ŷ & update statistics
    const prevCount = this.sampleCount;
    this.sampleCount = prevCount + 1;
    const n = this.sampleCount;

    for (let j = 0; j < m; j++) {
      const yVal = target[j];
      const residual = yVal - yHat[j];
      err[j] = residual;

      // Residual SSE
      this.residualSSE[j] += residual * residual;

      // Update running mean & M2 of targets
      const oldMean = this.yMean[j];
      const delta = yVal - oldMean;
      const newMean = oldMean + delta / n;
      const delta2 = yVal - newMean;
      this.yMean[j] = newMean;
      this.yM2[j] += delta * delta2;
    }

    // 6. Weight update: W = W + k eᵀ
    for (let i = 0; i < p; i++) {
      const ki = k[i];
      if (ki === 0) continue;
      let idx = i * m;
      for (let j = 0; j < m; j++) {
        W[idx++] += ki * err[j];
      }
    }

    // 7. Covariance update: P = (P - k vᵀ) / λ
    const lambda = this.forgettingFactor;
    const invLambda = 1.0 / lambda;
    for (let i = 0; i < p; i++) {
      const ki = k[i];
      const rowStart = i * p;
      for (let j = 0; j < p; j++) {
        P[rowStart + j] = (P[rowStart + j] - ki * v[j]) * invLambda;
      }
    }
  }

  /**
   * Quadratic form φᵀ P φ used for prediction variance.
   */
  public phiQuadraticForm(features: Float64Array): number {
    const p = this.featureCount;
    if (features.length !== p) {
      throw new Error(
        "RLSModelState.phiQuadraticForm: feature length mismatch.",
      );
    }

    const P = this.covariance;
    let result = 0.0;

    // Compute t = P φ, then φᵀ t.
    for (let i = 0; i < p; i++) {
      let sum = 0.0;
      const rowStart = i * p;
      for (let j = 0; j < p; j++) {
        sum += P[rowStart + j] * features[j];
      }
      result += features[i] * sum;
    }

    return result;
  }

  /** Residual variance estimate for a given output dimension. */
  public getResidualVariance(outputIndex: number): number {
    const n = this.sampleCount;
    if (n === 0) return 0.0;

    const sse = this.residualSSE[outputIndex];
    // Use n - p as degrees of freedom when possible.
    const dof = Math.max(1, n - this.featureCount);
    return sse / dof;
  }

  /** Aggregate RMSE across all outputs. */
  public computeRmse(): number {
    const n = this.sampleCount;
    if (n === 0) return 0.0;

    const m = this.outputDimension;
    let totalSSE = 0.0;
    for (let j = 0; j < m; j++) {
      totalSSE += this.residualSSE[j];
    }
    const denom = n * m;
    return denom > 0 ? Math.sqrt(totalSSE / denom) : 0.0;
  }

  /**
   * Aggregate R² coefficient across all outputs.
   * R²_j = 1 - SSE_j / M2_y_j
   */
  public computeRSquared(): number {
    const n = this.sampleCount;
    if (n === 0) return 0.0;

    const m = this.outputDimension;
    let sumR2 = 0.0;
    let countValid = 0;

    for (let j = 0; j < m; j++) {
      const m2 = this.yM2[j];
      if (m2 <= 0) {
        continue;
      }
      const sse = this.residualSSE[j];
      const r2 = 1.0 - sse / m2;
      sumR2 += r2;
      countValid++;
    }

    if (countValid === 0) return 0.0;
    return sumR2 / countValid;
  }

  /** User-friendly 2D view of the weight matrix [features][outputs]. */
  public getWeightsMatrix(): number[][] {
    const p = this.featureCount;
    const m = this.outputDimension;
    const W = this.weights;
    const result: number[][] = new Array(p);

    for (let i = 0; i < p; i++) {
      const row: number[] = new Array(m);
      let idx = i * m;
      for (let j = 0; j < m; j++) {
        row[j] = W[idx++];
      }
      result[i] = row;
    }

    return result;
  }
}

/* ============================================================================
 * Public API: MultivariatePolynomialRegression
 * ========================================================================== */

/**
 * Production-ready multivariate polynomial regression with online RLS learning.
 *
 * - Train incrementally with `fitOnline`
 * - Predict with confidence intervals using `predict`
 * - Inspect training state via `getModelSummary`, `getWeights`, `getNormalizationStats`
 */
export class MultivariatePolynomialRegression {
  private readonly config: Required<MultivariatePolynomialRegressionConfig>;

  private inputDimension: number | null = null;
  private outputDimension: number | null = null;

  private featureGenerator: PolynomialFeatureGenerator | null = null;
  private normalizer: OnlineNormalizer | null = null;
  private rls: RLSModelState | null = null;

  // Reusable buffers to avoid allocations in hot paths.
  private normalizedInputBuffer: Float64Array | null = null;
  private featureBuffer: Float64Array | null = null;
  private predictionBuffer: Float64Array | null = null;
  private standardErrorBuffer: Float64Array | null = null;
  private lowerBuffer: Float64Array | null = null;
  private upperBuffer: Float64Array | null = null;

  // Simple time-step tracking for extrapolation when futureSteps > 0.
  private lastInputSample: Float64Array | null = null;
  private prevInputSample: Float64Array | null = null;
  private lastStepSize: number = 1.0;

  constructor(config: MultivariatePolynomialRegressionConfig = {}) {
    const polynomialDegree =
      config.polynomialDegree !== undefined && config.polynomialDegree >= 1
        ? Math.floor(config.polynomialDegree)
        : 2;

    const forgettingFactor = config.forgettingFactor !== undefined
      ? Math.min(Math.max(config.forgettingFactor, 1e-6), 1.0)
      : 0.99;

    const initialCovariance =
      config.initialCovariance !== undefined && config.initialCovariance > 0
        ? config.initialCovariance
        : 1000.0;

    const regularization =
      config.regularization !== undefined && config.regularization > 0
        ? config.regularization
        : 1e-6;

    const confidenceLevel = config.confidenceLevel !== undefined
      ? Math.min(Math.max(config.confidenceLevel, 0.5), 0.999)
      : 0.95;

    const enableNormalization = config.enableNormalization !== undefined
      ? config.enableNormalization
      : true;

    const normalizationMethod: NormalizationMethod =
      config.normalizationMethod ?? "min-max";

    this.config = {
      polynomialDegree,
      forgettingFactor,
      initialCovariance,
      regularization,
      confidenceLevel,
      enableNormalization,
      normalizationMethod,
    };
  }

  /**
   * Incrementally train the model with new data.
   *
   * @param params.xCoordinates - Input features [samples][features]
   * @param params.yCoordinates - Output targets [samples][outputs]
   */
  public fitOnline(params: FitOnlineParams): void {
    const { xCoordinates, yCoordinates } = params;

    if (xCoordinates.length === 0 || yCoordinates.length === 0) {
      return;
    }
    if (xCoordinates.length !== yCoordinates.length) {
      throw new Error(
        "fitOnline: xCoordinates and yCoordinates must match in length.",
      );
    }

    const firstX = xCoordinates[0];
    const firstY = yCoordinates[0];

    if (!firstX || !firstY) {
      return;
    }

    const inputDim = firstX.length;
    const outputDim = firstY.length;

    if (inputDim === 0 || outputDim === 0) {
      throw new Error("fitOnline: Input and output dimensions must be > 0.");
    }

    this.ensureModelInitialized(inputDim, outputDim);

    const normalizer = this.normalizer;
    const featureGen = this.featureGenerator!;
    const rls = this.rls!;
    const normBuf = this.normalizedInputBuffer!;
    const featBuf = this.featureBuffer!;
    const inputDimChecked = this.inputDimension!;
    const outputDimChecked = this.outputDimension!;

    // Process each sample sequentially.
    for (let i = 0; i < xCoordinates.length; i++) {
      const x = xCoordinates[i];
      const y = yCoordinates[i];

      if (x.length !== inputDimChecked || y.length !== outputDimChecked) {
        throw new Error("fitOnline: Inconsistent sample dimensions.");
      }

      // Update time-step tracking for extrapolation.
      this.updateTimeTracking(x);

      // Update normalization statistics if enabled.
      if (normalizer) {
        normalizer.update(x);
        normalizer.transform(x, normBuf);
      } else {
        for (let d = 0; d < inputDimChecked; d++) {
          normBuf[d] = x[d];
        }
      }

      // Convert to polynomial features.
      featureGen.transform(normBuf, featBuf);

      // RLS update.
      rls.update(featBuf, y);
    }
  }

  /**
   * Predicts outputs and confidence intervals.
   *
   * - If `inputPoints` is provided and non-empty, predicts for those points.
   * - Otherwise, extrapolates `futureSteps` steps ahead using a simple
   *   time-based heuristic (primarily for 1D time-series).
   */
  public predict(params: PredictParams): PredictionResult {
    if (!this.isInitialized()) {
      return {
        predictions: [],
        confidenceLevel: this.config.confidenceLevel,
        rSquared: 0.0,
        rmse: 0.0,
        sampleCount: 0,
        isModelReady: false,
      };
    }

    const rls = this.rls!;
    const featureGen = this.featureGenerator!;
    const normalizer = this.normalizer;
    const normBuf = this.normalizedInputBuffer!;
    const featBuf = this.featureBuffer!;
    const predBuf = this.predictionBuffer!;
    const seBuf = this.standardErrorBuffer!;
    const lowerBuf = this.lowerBuffer!;
    const upperBuf = this.upperBuffer!;

    const inputDim = this.inputDimension!;
    const outputDim = this.outputDimension!;

    const zScore = MultivariatePolynomialRegression.getZScore(
      this.config.confidenceLevel,
    );
    const predictions: SinglePrediction[] = [];

    const inputPoints = params.inputPoints ?? [];

    const predictForX = (x: number[]): void => {
      if (x.length !== inputDim) {
        throw new Error("predict: input point dimension mismatch.");
      }

      // Apply normalization (read-only).
      if (normalizer) {
        normalizer.transform(x, normBuf);
      } else {
        for (let d = 0; d < inputDim; d++) {
          normBuf[d] = x[d];
        }
      }

      // Polynomial features.
      featureGen.transform(normBuf, featBuf);

      // Point prediction.
      rls.predict(featBuf, predBuf);

      // Predictive variance factor (shared across outputs).
      const phiPphi = rls.phiQuadraticForm(featBuf);

      for (let j = 0; j < outputDim; j++) {
        const residualVar = rls.getResidualVariance(j);
        let se = 0.0;
        if (residualVar > 0 && phiPphi > 0) {
          se = Math.sqrt(residualVar * phiPphi);
        }
        seBuf[j] = se;
        const margin = zScore * se;
        const yhat = predBuf[j];
        lowerBuf[j] = yhat - margin;
        upperBuf[j] = yhat + margin;
      }

      // Copy from buffers into plain arrays for user-facing result.
      const predicted: number[] = new Array(outputDim);
      const lowerBound: number[] = new Array(outputDim);
      const upperBound: number[] = new Array(outputDim);
      const standardError: number[] = new Array(outputDim);

      for (let j = 0; j < outputDim; j++) {
        predicted[j] = predBuf[j];
        lowerBound[j] = lowerBuf[j];
        upperBound[j] = upperBuf[j];
        standardError[j] = seBuf[j];
      }

      predictions.push({
        predicted,
        lowerBound,
        upperBound,
        standardError,
      });
    };

    if (inputPoints.length > 0) {
      // Direct prediction for specific inputs.
      for (let i = 0; i < inputPoints.length; i++) {
        predictForX(inputPoints[i]);
      }
    } else {
      // Extrapolation along the first dimension (time index).
      const steps = Math.max(0, Math.floor(params.futureSteps));
      if (steps > 0 && this.lastInputSample) {
        const base = this.lastInputSample[0];
        const stepSize = this.lastStepSize || 1.0;

        for (let s = 1; s <= steps; s++) {
          const xNew = new Array<number>(inputDim);
          xNew[0] = base + stepSize * s;
          for (let d = 1; d < inputDim; d++) {
            xNew[d] = this.lastInputSample[d];
          }
          predictForX(xNew);
        }
      }
    }

    const sampleCount = rls.getSampleCount();
    const rSquared = rls.computeRSquared();
    const rmse = rls.computeRmse();
    const featureCount = this.featureGenerator!.featureCount;

    const isModelReady = sampleCount > featureCount;

    return {
      predictions,
      confidenceLevel: this.config.confidenceLevel,
      rSquared,
      rmse,
      sampleCount,
      isModelReady,
    };
  }

  /**
   * Returns a comprehensive model summary, including dimensions, statistics,
   * and normalization configuration.
   */
  public getModelSummary(): ModelSummary {
    const initialized = this.isInitialized();
    const rls = this.rls;
    const featureGen = this.featureGenerator;

    const sampleCount = rls ? rls.getSampleCount() : 0;
    const rSquared = rls ? rls.computeRSquared() : 0.0;
    const rmse = rls ? rls.computeRmse() : 0.0;
    const polyFeatures = featureGen ? featureGen.featureCount : 0;

    return {
      isInitialized: initialized,
      inputDimension: this.inputDimension ?? 0,
      outputDimension: this.outputDimension ?? 0,
      polynomialDegree: this.config.polynomialDegree,
      polynomialFeatureCount: polyFeatures,
      sampleCount,
      rSquared,
      rmse,
      normalizationEnabled: !!this.normalizer,
      normalizationMethod: this.normalizer
        ? this.config.normalizationMethod
        : "none",
    };
  }

  /**
   * Returns the current weight matrix:
   * shape [polynomialFeatureCount][outputDimension].
   */
  public getWeights(): number[][] {
    if (!this.rls) return [];
    return this.rls.getWeightsMatrix();
  }

  /**
   * Returns the normalization statistics (if normalization is enabled).
   * If disabled or not yet initialized, returns `null`.
   */
  public getNormalizationStats(): NormalizationStats | null {
    if (!this.normalizer) return null;
    return this.normalizer.getStats();
  }

  /**
   * Clears all training state but preserves configuration.
   */
  public reset(): void {
    this.inputDimension = null;
    this.outputDimension = null;
    this.featureGenerator = null;
    this.normalizer = null;
    this.rls = null;

    this.normalizedInputBuffer = null;
    this.featureBuffer = null;
    this.predictionBuffer = null;
    this.standardErrorBuffer = null;
    this.lowerBuffer = null;
    this.upperBuffer = null;

    this.lastInputSample = null;
    this.prevInputSample = null;
    this.lastStepSize = 1.0;
  }

  /* ------------------------------------------------------------------------ */
  /* Internal helpers                                                         */
  /* ------------------------------------------------------------------------ */

  private isInitialized(): boolean {
    return this.inputDimension !== null && this.outputDimension !== null &&
      !!this.rls;
  }

  /** Lazy model initialization on first training call. */
  private ensureModelInitialized(inputDim: number, outputDim: number): void {
    if (this.isInitialized()) {
      if (
        this.inputDimension !== inputDim || this.outputDimension !== outputDim
      ) {
        throw new Error(
          "Model dimension mismatch: once initialized, input/output dimensions must remain constant.",
        );
      }
      return;
    }

    this.inputDimension = inputDim;
    this.outputDimension = outputDim;

    const featureGen = new PolynomialFeatureGenerator(
      inputDim,
      this.config.polynomialDegree,
    );
    this.featureGenerator = featureGen;

    const featureCount = featureGen.featureCount;

    // Normalizer created only when enabled.
    if (
      this.config.enableNormalization &&
      this.config.normalizationMethod !== "none"
    ) {
      this.normalizer = new OnlineNormalizer(
        inputDim,
        this.config.normalizationMethod,
      );
    } else {
      this.normalizer = null;
    }

    this.rls = new RLSModelState(
      featureCount,
      outputDim,
      this.config.initialCovariance,
      this.config.regularization,
      this.config.forgettingFactor,
    );

    // Allocate reusable buffers.
    this.normalizedInputBuffer = new Float64Array(inputDim);
    this.featureBuffer = new Float64Array(featureCount);
    this.predictionBuffer = new Float64Array(outputDim);
    this.standardErrorBuffer = new Float64Array(outputDim);
    this.lowerBuffer = new Float64Array(outputDim);
    this.upperBuffer = new Float64Array(outputDim);
  }

  /** Track last and previous input samples to estimate time step size. */
  private updateTimeTracking(x: number[]): void {
    const inputDim = this.inputDimension;
    if (inputDim === null || inputDim === 0) return;

    if (!this.lastInputSample) {
      this.lastInputSample = new Float64Array(inputDim);
      for (let i = 0; i < inputDim; i++) {
        this.lastInputSample[i] = x[i];
      }
      return;
    }

    if (!this.prevInputSample) {
      this.prevInputSample = new Float64Array(inputDim);
      for (let i = 0; i < inputDim; i++) {
        this.prevInputSample[i] = this.lastInputSample[i];
      }
    } else {
      // Shift last → prev
      for (let i = 0; i < inputDim; i++) {
        this.prevInputSample[i] = this.lastInputSample[i];
      }
    }

    // Update last
    for (let i = 0; i < inputDim; i++) {
      this.lastInputSample[i] = x[i];
    }

    // Estimate step size along the first dimension, if monotonic.
    const prev = this.prevInputSample[0];
    const last = this.lastInputSample[0];
    const delta = last - prev;
    if (delta !== 0) {
      this.lastStepSize = delta;
    }
  }

  /**
   * Approximate z-score for a given confidence level.
   * Uses a tiny lookup table; supports standard levels very accurately and
   * approximates others by nearest neighbor.
   */
  private static getZScore(confidenceLevel: number): number {
    const clamped = Math.min(Math.max(confidenceLevel, 0.5), 0.999);

    // Common confidence levels.
    const table = [
      { level: 0.80, z: 1.28155 },
      { level: 0.90, z: 1.64485 },
      { level: 0.95, z: 1.96 },
      { level: 0.98, z: 2.32635 },
      { level: 0.99, z: 2.57583 },
    ];

    let best = table[2].z; // Default ~1.96
    let bestDiff = Number.POSITIVE_INFINITY;

    for (let i = 0; i < table.length; i++) {
      const diff = Math.abs(table[i].level - clamped);
      if (diff < bestDiff) {
        bestDiff = diff;
        best = table[i].z;
      }
    }

    return best;
  }
}
