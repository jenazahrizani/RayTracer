/*
 * RAYTRC.AI.NAV / TRAINING WORKER
 *
 * Browser-side background trainer for Rust + WASM DQN navigation.
 *
 * Contract:
 *   START     -> train steps / episodes / continuous
 *   EVALUATE  -> deterministic evaluation using the current model
 *   STOP      -> cancel the active job between WASM chunks
 *
 * Checkpoint V2 notes:
 *   - export_model/import_model may contain the FULL training checkpoint.
 *   - A successfully imported checkpoint keeps its environment state.
 *   - Current UI trainer hyperparameters are applied after import.
 *   - Environment configuration is only applied to a fresh model unless
 *     forceEnvironmentConfig is explicitly requested.
 */

export type TrainingWorkerConfig = {
  difficulty: number;
  obstacles: number;
  episodeSteps: number;
  learningRate: number;
  gamma: number;
  batchSize: number;
  epsilon: number;
  epsilonMin: number;
  epsilonDecay: number;
  targetUpdate: number;
};

type TrainingMode = "steps" | "episodes" | "continuous";

type Engine = {
  version(): string;
  backend(): string;

  current_seed(): number;
  current_step(): number;
  current_episode(): number | bigint;
  training_step_count?: () => number | bigint;

  is_done(): boolean;
  is_success(): boolean;
  is_collision(): boolean;
  last_terminal_done?: () => boolean;
  last_terminal_success?: () => boolean;
  last_terminal_collision?: () => boolean;
  last_terminal_distance?: () => number;

  epsilon(): number;
  learning_rate?: () => number;
  gamma?: () => number;
  last_loss(): number;
  last_reward(): number;
  last_episode_reward(): number;
  last_episode_steps(): number;
  replay_size(): number;

  sensor_count(): number;
  observation_size(): number;
  action_count(): number;
  model_parameter_count(): number;

  set_difficulty(value: number): void;
  set_obstacle_count(value: number): void;
  set_episode_steps(value: number): void;
  set_learning_rate(value: number): void;
  set_gamma(value: number): void;
  set_batch_size(value: number): void;
  set_epsilon(value: number): void;
  set_epsilon_min(value: number): void;
  set_epsilon_decay(value: number): void;
  set_target_update(value: number): void;

  reset_environment(seed: number): void;

  train_steps(count: number): number[] | Float32Array;
  train_episode(): number[] | Float32Array;

  evaluate_episodes(
    count: number,
    seed: number,
  ): number[] | Float32Array;

  metrics(): number[] | Float32Array;

  export_model(): string;
  import_model(data: string): boolean;
  export_policy(): string;
  import_policy(data: string): boolean;
};

type WasmModule = {
  default(input?: unknown): Promise<unknown>;
  RayTracerEngine: new () => Engine;
};

type WorkerMessage = Record<string, unknown>;

const BASE = (() => {
  const env =
    (import.meta as ImportMeta & {
      env?: { BASE_URL?: string };
    }).env;

  const raw = env?.BASE_URL || "/";

  return raw.endsWith("/") ? raw : `${raw}/`;
})();

const WASM_JS_URL = new URL(
  "wasm/raytracer.js",
  new URL(BASE, self.location.href),
).href;

const DEFAULTS: TrainingWorkerConfig = {
  difficulty: 3,
  obstacles: 12,
  episodeSteps: 1000,
  learningRate: 0.0005,
  gamma: 0.99,
  batchSize: 64,
  epsilon: 1.0,
  epsilonMin: 0.05,
  epsilonDecay: 0.99998,
  targetUpdate: 128,
};

const MIN_CHUNK = 128;
const MAX_CHUNK = 2048;
const DEFAULT_CHUNK = 512;
const PREVIEW_SYNC_STEPS = 1024;
const PREVIEW_SYNC_EPISODES = 1;

let engine: Engine | null = null;
let activeJobId = -1;
let active = false;
let generation = 0;

function finite(value: unknown, fallback = 0): number {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

function integer(value: unknown, fallback = 0): number {
  return Math.floor(finite(value, fallback));
}

function clamp(value: unknown, min: number, max: number, fallback: number): number {
  return Math.min(max, Math.max(min, finite(value, fallback)));
}

function clampInt(value: unknown, min: number, max: number, fallback: number): number {
  return Math.min(max, Math.max(min, integer(value, fallback)));
}

function safeBuffer(value: unknown): number[] {
  if (value === null || value === undefined) {
    return [];
  }

  if (Array.isArray(value)) {
    return value.map((item) => finite(item));
  }

  if (ArrayBuffer.isView(value)) {
    const view = value as unknown as ArrayLike<unknown>;

    return Array.from(
      { length: view.length },
      (_, index) => finite(view[index]),
    );
  }

  if (
    typeof value === "object" &&
    value !== null &&
    Symbol.iterator in value
  ) {
    try {
      return Array.from(
        value as Iterable<unknown>,
        (item) => finite(item),
      );
    } catch {
      return [];
    }
  }

  return [];
}

async function importPublicWasmModule(url: string): Promise<WasmModule> {
  const nativeImport = new Function(
    "url",
    "return import(url);",
  ) as (url: string) => Promise<WasmModule>;

  return nativeImport(url);
}

async function loadEngine(): Promise<Engine> {
  if (engine) {
    return engine;
  }

  const response = await fetch(WASM_JS_URL, {
    cache: "no-cache",
  });

  if (!response.ok) {
    throw new Error(`WASM loader HTTP ${response.status}`);
  }

  const contentType = (
    response.headers.get("content-type") || ""
  ).toLowerCase();

  if (contentType.includes("text/html")) {
    throw new Error(
      "WASM loader resolved to HTML instead of JavaScript",
    );
  }

  const imported = await importPublicWasmModule(WASM_JS_URL);

  if (typeof imported.default !== "function") {
    throw new Error("WASM DEFAULT INITIALIZER NOT FOUND");
  }

  if (typeof imported.RayTracerEngine !== "function") {
    throw new Error("RayTracerEngine EXPORT NOT FOUND");
  }

  await imported.default();

  engine = new imported.RayTracerEngine();
  return engine;
}

function normalizeConfig(
  incoming: Partial<TrainingWorkerConfig> | TrainingWorkerConfig,
): TrainingWorkerConfig {
  return {
    difficulty: clampInt(
      incoming.difficulty,
      1,
      10,
      DEFAULTS.difficulty,
    ),

    obstacles: clampInt(
      incoming.obstacles,
      1,
      64,
      DEFAULTS.obstacles,
    ),

    episodeSteps: clampInt(
      incoming.episodeSteps,
      50,
      5000,
      DEFAULTS.episodeSteps,
    ),

    learningRate: clamp(
      incoming.learningRate,
      0.00001,
      0.1,
      DEFAULTS.learningRate,
    ),

    gamma: clamp(
      incoming.gamma,
      0.80,
      0.9999,
      DEFAULTS.gamma,
    ),

    batchSize: clampInt(
      incoming.batchSize,
      1,
      256,
      DEFAULTS.batchSize,
    ),

    epsilon: clamp(
      incoming.epsilon,
      0,
      1,
      DEFAULTS.epsilon,
    ),

    epsilonMin: clamp(
      incoming.epsilonMin,
      0,
      1,
      DEFAULTS.epsilonMin,
    ),

    epsilonDecay: clamp(
      incoming.epsilonDecay,
      0.9,
      0.999999,
      DEFAULTS.epsilonDecay,
    ),

    targetUpdate: clampInt(
      incoming.targetUpdate,
      1,
      100000,
      DEFAULTS.targetUpdate,
    ),
  };
}

function applyEnvironmentConfig(
  target: Engine,
  config: TrainingWorkerConfig,
): void {
  target.set_difficulty(config.difficulty);
  target.set_obstacle_count(config.obstacles);
  target.set_episode_steps(config.episodeSteps);
}

function applyTrainingConfig(
  target: Engine,
  config: TrainingWorkerConfig,
): void {
  target.set_learning_rate(config.learningRate);
  target.set_gamma(config.gamma);
  target.set_batch_size(config.batchSize);
  target.set_epsilon(config.epsilon);
  target.set_epsilon_min(config.epsilonMin);
  target.set_epsilon_decay(config.epsilonDecay);
  target.set_target_update(config.targetUpdate);
}

function applyConfig(
  target: Engine,
  config: TrainingWorkerConfig,
  preserveCheckpointEnvironment = false,
): void {
  if (!preserveCheckpointEnvironment) {
    applyEnvironmentConfig(target, config);
  }

  applyTrainingConfig(target, config);
}

function metricVector(target: Engine): number[] {
  try {
    return safeBuffer(target.metrics());
  } catch {
    return [];
  }
}

function trainingStepCount(target: Engine): number {
  if (typeof target.training_step_count === "function") {
    return finite(target.training_step_count());
  }

  // Compatibility fallback: metrics()[1] is training_steps in the Rust contract.
  return finite(metricVector(target)[1]);
}

function terminalFlag(
  target: Engine,
  kind: "done" | "success" | "collision",
): boolean {
  if (
    kind === "done" &&
    typeof target.last_terminal_done === "function"
  ) {
    return target.last_terminal_done();
  }

  if (
    kind === "success" &&
    typeof target.last_terminal_success === "function"
  ) {
    return target.last_terminal_success();
  }

  if (
    kind === "collision" &&
    typeof target.last_terminal_collision === "function"
  ) {
    return target.last_terminal_collision();
  }

  if (kind === "done") return target.is_done();
  if (kind === "success") return target.is_success();
  return target.is_collision();
}

function terminalDistance(target: Engine): number {
  if (typeof target.last_terminal_distance === "function") {
    return finite(target.last_terminal_distance());
  }

  return 0;
}

function telemetry(target: Engine): Record<string, unknown> {
  const result: Record<string, unknown> = {
    backend: target.backend(),
    version: target.version(),

    seed: finite(target.current_seed()),
    step: finite(target.current_step()),
    trainingSteps: finite(trainingStepCount(target)),
    episode: finite(target.current_episode()),
    episodeSteps: finite(target.last_episode_steps()),

    reward: finite(target.last_reward()),
    episodeReward: finite(target.last_episode_reward()),
    loss: finite(target.last_loss()),
    epsilon: finite(target.epsilon()),
    replay: finite(target.replay_size()),

    success: target.is_success(),
    collision: target.is_collision(),
    done: target.is_done(),

    lastTerminalDone: terminalFlag(target, "done"),
    lastTerminalSuccess: terminalFlag(target, "success"),
    lastTerminalCollision: terminalFlag(target, "collision"),
    lastTerminalDistance: terminalDistance(target),

    sensorCount: finite(target.sensor_count()),
    observationSize: finite(target.observation_size()),
    actionCount: finite(target.action_count()),
    parameterCount: finite(target.model_parameter_count()),
  };

  if (typeof target.learning_rate === "function") {
    result.learningRate = finite(target.learning_rate());
  }

  if (typeof target.gamma === "function") {
    result.gamma = finite(target.gamma());
  }

  return result;
}

function modelKind(model: string): "checkpoint" | "legacy" | "unknown" {
  const trimmed = model.trim();

  if (
    trimmed.startsWith("RAYTRC_NAV_CHECKPOINT_V2")
  ) {
    return "checkpoint";
  }

  if (
    trimmed.startsWith("RAYTRC_NAV_MODEL_V1")
  ) {
    return "legacy";
  }

  return "unknown";
}

function post(
  type: string,
  payload: Record<string, unknown> = {},
): void {
  self.postMessage({
    type,
    jobId: activeJobId,
    ...payload,
  });
}

function isJobActive(jobToken: number): boolean {
  return active && jobToken === generation;
}

function requestYield(
  callback: () => void,
  jobToken: number,
): void {
  if (!isJobActive(jobToken)) {
    return;
  }

  setTimeout(() => {
    if (isJobActive(jobToken)) {
      callback();
    }
  }, 0);
}

function sendProgress(
  mode: TrainingMode,
  requested: number,
  completed: number,
  target: Engine,
  includeModel: boolean,
): void {
  const progressPayload: Record<string, unknown> = {
    mode: mode === "episodes" ? "TRAIN-EPISODES" : "TRAIN",
    state: "RUNNING",
    status:
      requested > 0
        ? `TRAINING ${completed}/${requested}`
        : `TRAINING ${completed}`,
    completed,
    requested,
    message:
      requested > 0
        ? `TRAINING ${completed}/${requested}`
        : `CONTINUOUS TRAINING ${completed}`,
    ...telemetry(target),
  };

  if (includeModel) {
    progressPayload.previewPolicy = target.export_policy();
  }

  post("PROGRESS", progressPayload);
}

function runTrainingChunk(
  mode: TrainingMode,
  requested: number,
  chunk: number,
  completed: number,
  jobToken: number,
): void {
  if (!engine || !isJobActive(jobToken)) {
    return;
  }

  try {
    let nextCompleted = completed;

    if (mode === "episodes") {
      engine.train_episode();
      nextCompleted += 1;
    } else {
      const remaining = Math.max(0, requested - completed);
      const amount =
        mode === "continuous"
          ? chunk
          : Math.min(chunk, remaining);

      if (amount > 0) {
        engine.train_steps(amount);
        nextCompleted += amount;
      }
    }

    const finished =
      mode !== "continuous" &&
      nextCompleted >= requested;

    const syncStride =
      mode === "episodes"
        ? PREVIEW_SYNC_EPISODES
        : PREVIEW_SYNC_STEPS;

    const shouldSyncModel =
      finished ||
      (nextCompleted > 0 &&
        nextCompleted % syncStride === 0);

    sendProgress(
      mode,
      requested,
      nextCompleted,
      engine,
      shouldSyncModel,
    );

    if (finished) {
      const model = engine.export_model();
      const enginePolicy = engine.export_policy();

      post("COMPLETE", {
        mode: mode === "episodes" ? "TRAIN-EPISODES" : "TRAIN",
        state: "COMPLETE",
        status: "TRAINING COMPLETE / MODEL EXPORTED",
        completed: nextCompleted,
        requested,
        checkpointType: modelKind(model),
        model,
        policyModel: enginePolicy,
        message: "TRAINING COMPLETE / POLICY UPDATED",
        ...telemetry(engine),
      });

      active = false;
      return;
    }

    requestYield(
      () => {
        runTrainingChunk(
          mode,
          requested,
          chunk,
          nextCompleted,
          jobToken,
        );
      },
      jobToken,
    );
  } catch (error) {
    active = false;

    post("ERROR", {
      message:
        error instanceof Error
          ? error.message
          : String(error),
    });
  }
}

async function startTraining(message: WorkerMessage): Promise<void> {
  generation += 1;
  const jobToken = generation;

  active = true;
  activeJobId = integer(message.jobId, -1);

  const target = await loadEngine();

  if (!isJobActive(jobToken)) {
    return;
  }

  const config = normalizeConfig(
    (message.config ?? {}) as Partial<TrainingWorkerConfig>,
  );

  const model = String(message.model ?? "").trim();

  let importedCheckpoint = false;

  if (model) {
    const imported = target.import_model(model);

    if (!imported) {
      throw new Error(
        `training worker could not import ${modelKind(model) === "checkpoint" ? "checkpoint" : "model"}`,
      );
    }

    importedCheckpoint = modelKind(model) === "checkpoint";
  }

  /*
   * Training hyperparameters always follow the current UI configuration.
   * A V2 checkpoint keeps its complete environment/replay/RNG state;
   * environment configuration is not silently reset after import.
   */
  applyConfig(target, config, importedCheckpoint);

  if (!model || !importedCheckpoint) {
    const seed = finite(
      message.seed,
      Date.now() >>> 0,
    ) >>> 0;

    target.reset_environment(seed);
  }

  const mode = String(
    message.mode ?? "steps",
  ).toLowerCase() as TrainingMode;

  if (
    mode !== "steps" &&
    mode !== "episodes" &&
    mode !== "continuous"
  ) {
    throw new Error(`unsupported training mode: ${mode}`);
  }

  const requested = Math.max(
    0,
    integer(message.requested, 0),
  );

  const chunk = clampInt(
    message.chunk,
    MIN_CHUNK,
    MAX_CHUNK,
    DEFAULT_CHUNK,
  );

  if (
    !isJobActive(jobToken)
  ) {
    return;
  }

  post("PROGRESS", {
    mode: mode === "episodes" ? "TRAIN-EPISODES" : "TRAIN",
    state: "RUNNING",
    status: importedCheckpoint
      ? "CHECKPOINT RESTORED / TRAINING WORKER READY"
      : "TRAINING WORKER READY",
    completed: 0,
    requested,
    checkpointType: importedCheckpoint ? "RAYTRC_NAV_CHECKPOINT_V2" : "NEW",
    checkpointRestored: importedCheckpoint,
    ...telemetry(target),
  });

  runTrainingChunk(
    mode,
    requested,
    chunk,
    0,
    jobToken,
  );
}

async function evaluate(message: WorkerMessage): Promise<void> {
  generation += 1;
  const jobToken = generation;

  active = true;
  activeJobId = integer(message.jobId, -1);

  const target = await loadEngine();

  if (!isJobActive(jobToken)) {
    return;
  }

  const config = normalizeConfig(
    (message.config ?? {}) as Partial<TrainingWorkerConfig>,
  );

  const model = String(message.model ?? "").trim();
  let importedCheckpoint = false;

  if (model) {
    const imported = target.import_model(model);

    if (!imported) {
      throw new Error(
        `evaluation worker could not import ${modelKind(model) === "checkpoint" ? "checkpoint" : "model"}`,
      );
    }

    importedCheckpoint = modelKind(model) === "checkpoint";
  }

  /*
   * Preserve checkpoint environment for exact evaluation. For a legacy/new
   * model, apply the current UI environment configuration and start at the
   * requested deterministic base seed.
   */
  applyConfig(target, config, importedCheckpoint);

  const requested = clampInt(
    message.requested,
    1,
    10000,
    1,
  );

  const baseSeed =
    finite(
      message.baseSeed,
      Date.now() >>> 0,
    ) >>> 0;

  if (!importedCheckpoint) {
    target.reset_environment(baseSeed);
  }

  const result = safeBuffer(
    target.evaluate_episodes(
      requested,
      baseSeed,
    ),
  );

  if (!isJobActive(jobToken)) {
    return;
  }

  const average =
    result.length > 0
      ? result.reduce(
          (total, value) => total + value,
          0,
        ) / result.length
      : 0;

  post("EVAL_COMPLETE", {
    requested,
    baseSeed,
    checkpointRestored: importedCheckpoint,
    results: result,
    average,
    ...telemetry(target),
  });

  active = false;
}

function stopActiveJob(): void {
  active = false;
  generation += 1;
  activeJobId = -1;
}

self.addEventListener("message", (event: MessageEvent) => {
  const message = (event.data ?? {}) as WorkerMessage;
  const type = String(message.type ?? "").toUpperCase();

  if (type === "STOP") {
    stopActiveJob();
    return;
  }

  if (type === "START") {
    /*
     * A newer START supersedes the previous job. Timers from the previous
     * generation become no-ops through isJobActive().
     */
    active = false;
    generation += 1;

    startTraining(message).catch((error) => {
      active = false;
      activeJobId = integer(message.jobId, -1);

      post("ERROR", {
        message:
          error instanceof Error
            ? error.message
            : String(error),
      });
    });

    return;
  }

  if (type === "EVALUATE") {
    active = false;
    generation += 1;

    evaluate(message).catch((error) => {
      active = false;
      activeJobId = integer(message.jobId, -1);

      post("ERROR", {
        message:
          error instanceof Error
            ? error.message
            : String(error),
      });
    });
  }
});
