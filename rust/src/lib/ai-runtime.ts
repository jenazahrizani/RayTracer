/*
 * RAYTRC.AI.RUNTIME
 * ---------------------------------------------------------------------------
 * Browser-side runtime bridge for the Rust/WASM DQN engine.
 *
 * Responsibilities:
 *   - Resolve and load the public wasm-bindgen glue safely.
 *   - Own one RayTracerEngine instance per page.
 *   - Normalize training/environment configuration.
 *   - Detect V2 full checkpoints vs legacy V1 models.
 *   - Persist full checkpoints in IndexedDB (not localStorage).
 *   - Keep a legacy localStorage fallback for older installations.
 *   - Import/export/download checkpoint text.
 *   - Normalize engine telemetry into a stable TypeScript shape.
 *   - Manage the training Web Worker lifecycle and job generations.
 *   - Provide small DOM/event helpers used by Astro components.
 *
 * No external dependencies.
 * GitHub Pages / static-host friendly.
 */

export const ENGINE_VERSION = "RAYTRC.AI.RUNTIME/1.0.0";
export const CHECKPOINT_MAGIC = "RAYTRC_NAV_CHECKPOINT_V2";
export const LEGACY_MODEL_MAGIC = "RAYTRC_NAV_MODEL_V1";

export const CHECKPOINT_DB = "RAYTRC.AI.CHECKPOINTS";
export const CHECKPOINT_STORE = "state";
export const CHECKPOINT_KEY = "current";
export const LEGACY_LOCAL_STORAGE_KEY = "raytrc.ai.model";

export type TrainingWorkerMode =
  | "steps"
  | "episodes"
  | "continuous";

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

export type EngineArray = number[] | Float32Array;

export type RayTracerEngine = {
  version(): string;
  backend(): string;
  sensor_count(): number;
  observation_size(): number;
  action_count(): number;
  model_parameter_count(): number;

  reset_ai(): void;
  reset_environment(seed: number): void;
  recover_from_collision(): void;
  hard_reset(): void;

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

  current_seed(): number;
  difficulty?: () => number;
  configured_obstacle_count?: () => number;
  episode_limit?: () => number;
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

  obstacle_count(): number;

  get_observation(): EngineArray;
  get_sensor_buffer(): EngineArray;
  get_agent_buffer(): EngineArray;
  get_target_buffer(): EngineArray;
  get_obstacle_buffer(): EngineArray;
  get_path_buffer(): EngineArray;

  predict_action(): number;
  q_values(): EngineArray;

  step(action: number): EngineArray;
  train_step(): EngineArray;
  train_steps(count: number): EngineArray;
  train_episode(): EngineArray;
  train_episodes(count: number): EngineArray;

  evaluate_episodes(
    count: number,
    base_seed: number,
  ): EngineArray;

  metrics(): EngineArray;
  status_json(): string;

  export_model(): string;
  import_model(data: string): boolean;
  save_model(): string;
  load_model(data: string): boolean;
};

export type WasmModule = {
  default(input?: unknown): Promise<unknown>;
  RayTracerEngine: new () => RayTracerEngine;
};

export type RuntimeTelemetry = {
  backend: string;
  version: string;

  seed: number;
  step: number;
  trainingSteps: number;
  episode: number;
  episodeSteps: number;

  reward: number;
  episodeReward: number;
  loss: number;
  epsilon: number;
  learningRate: number;
  gamma: number;
  replay: number;

  success: boolean;
  collision: boolean;
  done: boolean;

  lastTerminalDone: boolean;
  lastTerminalSuccess: boolean;
  lastTerminalCollision: boolean;
  lastTerminalDistance: number;

  sensorCount: number;
  observationSize: number;
  actionCount: number;
  parameterCount: number;
  obstacleCount: number;
};

export type CheckpointRecord = {
  format: typeof CHECKPOINT_MAGIC;
  savedAt: number;
  bytes: number;
  data: string;
};

export type RuntimeJob = {
  id: number;
  mode: TrainingWorkerMode | "evaluate";
  requested: number;
  startedAt: number;
  active: boolean;
};

export type WorkerProgress = {
  type: "PROGRESS";
  jobId: number;
  completed: number;
  requested: number;
  mode: string;
  state: string;
  status: string;
  message?: string;
  previewModel?: string;
  checkpointType?: string;
  checkpointRestored?: boolean;
  [key: string]: unknown;
};

export type WorkerComplete = {
  type: "COMPLETE" | "EVAL_COMPLETE";
  jobId: number;
  [key: string]: unknown;
};

export type WorkerFailure = {
  type: "ERROR";
  jobId: number;
  message: string;
};

export type RuntimeEventMap = {
  ready: {
    engine: RayTracerEngine;
    wasm: WasmModule;
  };
  error: {
    error: Error;
  };
  checkpointSaved: CheckpointRecord;
  checkpointLoaded: {
    data: string;
    kind: "checkpoint" | "legacy" | "unknown";
  };
  checkpointCleared: void;
  workerProgress: WorkerProgress;
  workerComplete: WorkerComplete;
  workerError: WorkerFailure;
};

type RuntimeListener<K extends keyof RuntimeEventMap> =
  (payload: RuntimeEventMap[K]) => void;

type RuntimeWorkerListener =
  (message: WorkerProgress | WorkerComplete | WorkerFailure) => void;

export const DEFAULT_TRAINING_CONFIG: TrainingWorkerConfig = {
  difficulty: 3,
  obstacles: 12,
  episodeSteps: 1000,
  learningRate: 0.0005,
  gamma: 0.99,
  batchSize: 64,
  epsilon: 1.0,
  epsilonMin: 0.05,
  epsilonDecay: 0.9985,
  targetUpdate: 250,
};

export const TRAINING_LIMITS = {
  difficulty: [1, 10] as const,
  obstacles: [1, 64] as const,
  episodeSteps: [50, 5000] as const,
  learningRate: [0.00001, 0.1] as const,
  gamma: [0.8, 0.9999] as const,
  batchSize: [1, 256] as const,
  epsilon: [0, 1] as const,
  epsilonMin: [0, 1] as const,
  epsilonDecay: [0.9, 0.999999] as const,
  targetUpdate: [1, 100000] as const,
};

const MIN_WORKER_CHUNK = 128;
const MAX_WORKER_CHUNK = 2048;
const DEFAULT_WORKER_CHUNK = 512;

const BASE_URL = resolveBaseUrl();
const WASM_JS_URL = new URL(
  "wasm/raytracer.js",
  new URL(BASE_URL, getBaseLocation()),
).href;

function getBaseLocation(): string {
  if (typeof self !== "undefined" && self.location?.href) {
    return self.location.href;
  }

  if (typeof window !== "undefined" && window.location?.href) {
    return window.location.href;
  }

  return "http://localhost/";
}

function resolveBaseUrl(): string {
  const base =
    typeof import.meta !== "undefined" &&
    (import.meta as ImportMeta & {
      env?: { BASE_URL?: string };
    }).env?.BASE_URL;

  const raw =
    typeof base === "string" && base.length > 0
      ? base
      : "/";

  return raw.endsWith("/")
    ? raw
    : `${raw}/`;
}

export function finite(
  value: unknown,
  fallback = 0,
): number {
  const numeric = Number(value);
  return Number.isFinite(numeric)
    ? numeric
    : fallback;
}

export function integer(
  value: unknown,
  fallback = 0,
): number {
  return Math.floor(finite(value, fallback));
}

export function clamp(
  value: unknown,
  min: number,
  max: number,
  fallback: number,
): number {
  return Math.min(
    max,
    Math.max(
      min,
      finite(value, fallback),
    ),
  );
}

export function clampInt(
  value: unknown,
  min: number,
  max: number,
  fallback: number,
): number {
  return Math.min(
    max,
    Math.max(
      min,
      integer(value, fallback),
    ),
  );
}

export function safeBuffer(
  value: unknown,
): number[] {
  if (
    value === null ||
    value === undefined
  ) {
    return [];
  }

  if (Array.isArray(value)) {
    return value.map((item) => finite(item));
  }

  if (ArrayBuffer.isView(value)) {
    const view =
      value as unknown as ArrayLike<unknown>;

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

export function normalizeTrainingConfig(
  input: Partial<TrainingWorkerConfig> = {},
): TrainingWorkerConfig {
  return {
    difficulty: clampInt(
      input.difficulty,
      ...TRAINING_LIMITS.difficulty,
      DEFAULT_TRAINING_CONFIG.difficulty,
    ),
    obstacles: clampInt(
      input.obstacles,
      ...TRAINING_LIMITS.obstacles,
      DEFAULT_TRAINING_CONFIG.obstacles,
    ),
    episodeSteps: clampInt(
      input.episodeSteps,
      ...TRAINING_LIMITS.episodeSteps,
      DEFAULT_TRAINING_CONFIG.episodeSteps,
    ),
    learningRate: clamp(
      input.learningRate,
      ...TRAINING_LIMITS.learningRate,
      DEFAULT_TRAINING_CONFIG.learningRate,
    ),
    gamma: clamp(
      input.gamma,
      ...TRAINING_LIMITS.gamma,
      DEFAULT_TRAINING_CONFIG.gamma,
    ),
    batchSize: clampInt(
      input.batchSize,
      ...TRAINING_LIMITS.batchSize,
      DEFAULT_TRAINING_CONFIG.batchSize,
    ),
    epsilon: clamp(
      input.epsilon,
      ...TRAINING_LIMITS.epsilon,
      DEFAULT_TRAINING_CONFIG.epsilon,
    ),
    epsilonMin: clamp(
      input.epsilonMin,
      ...TRAINING_LIMITS.epsilonMin,
      DEFAULT_TRAINING_CONFIG.epsilonMin,
    ),
    epsilonDecay: clamp(
      input.epsilonDecay,
      ...TRAINING_LIMITS.epsilonDecay,
      DEFAULT_TRAINING_CONFIG.epsilonDecay,
    ),
    targetUpdate: clampInt(
      input.targetUpdate,
      ...TRAINING_LIMITS.targetUpdate,
      DEFAULT_TRAINING_CONFIG.targetUpdate,
    ),
  };
}

export function normalizeWorkerChunk(
  value: unknown,
): number {
  return clampInt(
    value,
    MIN_WORKER_CHUNK,
    MAX_WORKER_CHUNK,
    DEFAULT_WORKER_CHUNK,
  );
}

export function modelKind(
  data: string,
): "checkpoint" | "legacy" | "unknown" {
  const trimmed = String(data ?? "").trim();

  if (trimmed.startsWith(CHECKPOINT_MAGIC)) {
    return "checkpoint";
  }

  if (trimmed.startsWith(LEGACY_MODEL_MAGIC)) {
    return "legacy";
  }

  return "unknown";
}

export function isCheckpoint(
  data: string,
): boolean {
  return modelKind(data) === "checkpoint";
}

export function isLegacyModel(
  data: string,
): boolean {
  return modelKind(data) === "legacy";
}

export function actionName(
  action: unknown,
): string {
  const names = [
    "FORWARD",
    "LEFT",
    "RIGHT",
    "BRAKE",
    "REVERSE",
  ];

  const index = clampInt(action, 0, names.length - 1, 0);
  return names[index] ?? "UNKNOWN";
}

function terminalFlag(
  target: RayTracerEngine,
  kind: "done" | "success" | "collision",
): boolean {
  if (
    kind === "done" &&
    typeof target.last_terminal_done === "function"
  ) {
    return Boolean(target.last_terminal_done());
  }

  if (
    kind === "success" &&
    typeof target.last_terminal_success === "function"
  ) {
    return Boolean(target.last_terminal_success());
  }

  if (
    kind === "collision" &&
    typeof target.last_terminal_collision === "function"
  ) {
    return Boolean(target.last_terminal_collision());
  }

  if (kind === "done") {
    return Boolean(target.is_done());
  }

  if (kind === "success") {
    return Boolean(target.is_success());
  }

  return Boolean(target.is_collision());
}

function terminalDistance(
  target: RayTracerEngine,
): number {
  if (
    typeof target.last_terminal_distance === "function"
  ) {
    return finite(target.last_terminal_distance());
  }

  return 0;
}

export function trainingStepCount(
  target: RayTracerEngine,
): number {
  if (
    typeof target.training_step_count === "function"
  ) {
    return finite(target.training_step_count());
  }

  try {
    return finite(safeBuffer(target.metrics())[1]);
  } catch {
    return 0;
  }
}

export function readTelemetry(
  target: RayTracerEngine,
): RuntimeTelemetry {
  const learningRate =
    typeof target.learning_rate === "function"
      ? finite(target.learning_rate())
      : DEFAULT_TRAINING_CONFIG.learningRate;

  const gamma =
    typeof target.gamma === "function"
      ? finite(target.gamma())
      : DEFAULT_TRAINING_CONFIG.gamma;

  return {
    backend: String(target.backend()),
    version: String(target.version()),

    seed: finite(target.current_seed()),
    step: finite(target.current_step()),
    trainingSteps: finite(trainingStepCount(target)),
    episode: finite(target.current_episode()),
    episodeSteps: finite(target.last_episode_steps()),

    reward: finite(target.last_reward()),
    episodeReward: finite(target.last_episode_reward()),
    loss: finite(target.last_loss()),
    epsilon: finite(target.epsilon()),
    learningRate,
    gamma,
    replay: finite(target.replay_size()),

    success: Boolean(target.is_success()),
    collision: Boolean(target.is_collision()),
    done: Boolean(target.is_done()),

    lastTerminalDone: terminalFlag(target, "done"),
    lastTerminalSuccess: terminalFlag(target, "success"),
    lastTerminalCollision: terminalFlag(target, "collision"),
    lastTerminalDistance: terminalDistance(target),

    sensorCount: finite(target.sensor_count()),
    observationSize: finite(target.observation_size()),
    actionCount: finite(target.action_count()),
    parameterCount: finite(target.model_parameter_count()),
    obstacleCount: finite(target.obstacle_count()),
  };
}

export function readQValues(
  target: RayTracerEngine,
): number[] {
  return safeBuffer(target.q_values());
}

export function readObservation(
  target: RayTracerEngine,
): number[] {
  return safeBuffer(target.get_observation());
}

export function readWorldBuffers(
  target: RayTracerEngine,
): {
  sensors: number[];
  agent: number[];
  target: number[];
  obstacles: number[];
  path: number[];
} {
  return {
    sensors: safeBuffer(target.get_sensor_buffer()),
    agent: safeBuffer(target.get_agent_buffer()),
    target: safeBuffer(target.get_target_buffer()),
    obstacles: safeBuffer(target.get_obstacle_buffer()),
    path: safeBuffer(target.get_path_buffer()),
  };
}

export function checkpointFilename(
  prefix = "raytrc-ai-checkpoint",
): string {
  return `${prefix}-v2.txt`;
}

export function downloadText(
  text: string,
  filename = checkpointFilename(),
): void {
  if (typeof document === "undefined") {
    throw new Error("DOWNLOAD REQUIRES A BROWSER DOCUMENT");
  }

  const blob = new Blob(
    [text],
    { type: "text/plain;charset=utf-8" },
  );

  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");

  anchor.href = url;
  anchor.download = filename;
  anchor.rel = "noopener";

  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();

  window.setTimeout(() => {
    URL.revokeObjectURL(url);
  }, 1000);
}

export function pickTextFile(
  accept = ".txt,.json,.model,.checkpoint,text/plain,application/json",
): Promise<string | null> {
  if (typeof document === "undefined") {
    return Promise.reject(
      new Error("FILE PICKER REQUIRES A BROWSER DOCUMENT"),
    );
  }

  return new Promise((resolve, reject) => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = accept;

    input.addEventListener("change", async () => {
      const file = input.files?.[0];

      if (!file) {
        resolve(null);
        return;
      }

      try {
        resolve(await file.text());
      } catch (error) {
        reject(
          error instanceof Error
            ? error
            : new Error(String(error)),
        );
      }
    }, { once: true });

    input.click();
  });
}

export function canUseIndexedDb(): boolean {
  return typeof indexedDB !== "undefined";
}

function openCheckpointDb(): Promise<IDBDatabase> {
  if (!canUseIndexedDb()) {
    return Promise.reject(
      new Error("INDEXEDDB UNAVAILABLE"),
    );
  }

  return new Promise((resolve, reject) => {
    const request = indexedDB.open(
      CHECKPOINT_DB,
      1,
    );

    request.onupgradeneeded = () => {
      const db = request.result;

      if (!db.objectStoreNames.contains(CHECKPOINT_STORE)) {
        db.createObjectStore(CHECKPOINT_STORE);
      }
    };

    request.onsuccess = () => {
      resolve(request.result);
    };

    request.onerror = () => {
      reject(
        request.error ??
          new Error("INDEXEDDB OPEN FAILED"),
      );
    };

    request.onblocked = () => {
      reject(
        new Error("INDEXEDDB OPEN BLOCKED"),
      );
    };
  });
}

export async function saveCheckpointData(
  data: string,
): Promise<CheckpointRecord> {
  if (!data.trim()) {
    throw new Error("EMPTY CHECKPOINT DATA");
  }

  if (modelKind(data) !== "checkpoint") {
    throw new Error("ONLY RAYTRC_NAV_CHECKPOINT_V2 CAN BE SAVED AS A CHECKPOINT");
  }

  const record: CheckpointRecord = {
    format: CHECKPOINT_MAGIC,
    savedAt: Date.now(),
    bytes: new TextEncoder().encode(data).byteLength,
    data,
  };

  const db = await openCheckpointDb();

  try {
    await new Promise<void>((resolve, reject) => {
      const transaction = db.transaction(
        CHECKPOINT_STORE,
        "readwrite",
      );

      transaction.objectStore(CHECKPOINT_STORE).put(
        record,
        CHECKPOINT_KEY,
      );

      transaction.oncomplete = () => resolve();
      transaction.onerror = () => {
        reject(
          transaction.error ??
            new Error("INDEXEDDB SAVE FAILED"),
        );
      };
      transaction.onabort = () => {
        reject(
          transaction.error ??
            new Error("INDEXEDDB SAVE ABORTED"),
        );
      };
    });
  } finally {
    db.close();
  }

  return record;
}

export async function loadCheckpointData(): Promise<string | null> {
  if (!canUseIndexedDb()) {
    return null;
  }

  const db = await openCheckpointDb();

  try {
    const record = await new Promise<unknown>((resolve, reject) => {
      const transaction = db.transaction(
        CHECKPOINT_STORE,
        "readonly",
      );

      const request = transaction
        .objectStore(CHECKPOINT_STORE)
        .get(CHECKPOINT_KEY);

      request.onsuccess = () => {
        resolve(request.result ?? null);
      };

      request.onerror = () => {
        reject(
          request.error ??
            new Error("INDEXEDDB LOAD FAILED"),
        );
      };
    });

    if (!record || typeof record !== "object") {
      return null;
    }

    const value = record as Partial<CheckpointRecord>;

    if (
      value.format !== CHECKPOINT_MAGIC ||
      typeof value.data !== "string"
    ) {
      return null;
    }

    return value.data;
  } finally {
    db.close();
  }
}

export async function clearCheckpointData(): Promise<void> {
  if (!canUseIndexedDb()) {
    return;
  }

  const db = await openCheckpointDb();

  try {
    await new Promise<void>((resolve, reject) => {
      const transaction = db.transaction(
        CHECKPOINT_STORE,
        "readwrite",
      );

      transaction.objectStore(CHECKPOINT_STORE).delete(
        CHECKPOINT_KEY,
      );

      transaction.oncomplete = () => resolve();
      transaction.onerror = () => {
        reject(
          transaction.error ??
            new Error("INDEXEDDB DELETE FAILED"),
        );
      };
      transaction.onabort = () => {
        reject(
          transaction.error ??
            new Error("INDEXEDDB DELETE ABORTED"),
        );
      };
    });
  } finally {
    db.close();
  }
}

export function loadLegacyLocalStorageModel(): string | null {
  if (typeof localStorage === "undefined") {
    return null;
  }

  try {
    const value = localStorage.getItem(
      LEGACY_LOCAL_STORAGE_KEY,
    );

    return value && value.trim()
      ? value
      : null;
  } catch {
    return null;
  }
}

export function saveLegacyLocalStorageModel(
  data: string,
): void {
  if (typeof localStorage === "undefined") {
    throw new Error("LOCALSTORAGE UNAVAILABLE");
  }

  localStorage.setItem(
    LEGACY_LOCAL_STORAGE_KEY,
    data,
  );
}

export function clearLegacyLocalStorageModel(): void {
  if (typeof localStorage === "undefined") {
    return;
  }

  try {
    localStorage.removeItem(
      LEGACY_LOCAL_STORAGE_KEY,
    );
  } catch {
    // Best effort only.
  }
}

export async function loadBestAvailableCheckpoint(): Promise<{
  data: string | null;
  source: "indexeddb" | "localstorage" | "none";
  kind: "checkpoint" | "legacy" | "unknown";
}> {
  const indexed = await loadCheckpointData().catch(() => null);

  if (indexed) {
    return {
      data: indexed,
      source: "indexeddb",
      kind: modelKind(indexed),
    };
  }

  const legacy = loadLegacyLocalStorageModel();

  if (legacy) {
    return {
      data: legacy,
      source: "localstorage",
      kind: modelKind(legacy),
    };
  }

  return {
    data: null,
    source: "none",
    kind: "unknown",
  };
}

export async function importBestAvailableCheckpoint(
  target: RayTracerEngine,
): Promise<{
  loaded: boolean;
  source: "indexeddb" | "localstorage" | "none";
  kind: "checkpoint" | "legacy" | "unknown";
  data: string | null;
}> {
  const available = await loadBestAvailableCheckpoint();

  if (!available.data) {
    return {
      loaded: false,
      ...available,
    };
  }

  const loaded = target.load_model(
    available.data,
  );

  return {
    loaded,
    ...available,
  };
}

async function importPublicWasmModule(
  url: string,
): Promise<WasmModule> {
  const nativeImport = new Function(
    "url",
    "return import(url);",
  ) as (
    url: string,
  ) => Promise<WasmModule>;

  return nativeImport(url);
}

export class AIRuntime {
  private engine: RayTracerEngine | null = null;
  private wasm: WasmModule | null = null;
  private loadPromise: Promise<RayTracerEngine> | null = null;

  private listeners: {
    [K in keyof RuntimeEventMap]: Set<RuntimeListener<K>>;
  } = {
    ready: new Set(),
    error: new Set(),
    checkpointSaved: new Set(),
    checkpointLoaded: new Set(),
    checkpointCleared: new Set(),
    workerProgress: new Set(),
    workerComplete: new Set(),
    workerError: new Set(),
  } as {
    [K in keyof RuntimeEventMap]: Set<RuntimeListener<K>>;
  };

  private worker: Worker | null = null;
  private workerGeneration = 0;
  private workerJobId = 0;
  private workerJob: RuntimeJob | null = null;
  private workerListeners = new Set<RuntimeWorkerListener>();

  public get wasmUrl(): string {
    return WASM_JS_URL;
  }

  public get baseUrl(): string {
    return BASE_URL;
  }

  public get currentEngine(): RayTracerEngine | null {
    return this.engine;
  }

  public get currentWasmModule(): WasmModule | null {
    return this.wasm;
  }

  public isReady(): boolean {
    return this.engine !== null;
  }

  public on<K extends keyof RuntimeEventMap>(
    event: K,
    listener: RuntimeListener<K>,
  ): () => void {
    this.listeners[event].add(
      listener as RuntimeListener<K>,
    );

    return () => {
      this.off(event, listener);
    };
  }

  public off<K extends keyof RuntimeEventMap>(
    event: K,
    listener: RuntimeListener<K>,
  ): void {
    this.listeners[event].delete(
      listener as RuntimeListener<K>,
    );
  }

  private emit<K extends keyof RuntimeEventMap>(
    event: K,
    payload: RuntimeEventMap[K],
  ): void {
    for (const listener of this.listeners[event]) {
      try {
        listener(payload);
      } catch {
        // A telemetry/UI listener must never break the runtime.
      }
    }
  }

  private fail(error: unknown): Error {
    const normalized =
      error instanceof Error
        ? error
        : new Error(String(error));

    this.emit("error", {
      error: normalized,
    });

    return normalized;
  }

  public async load(): Promise<RayTracerEngine> {
    if (this.engine) {
      return this.engine;
    }

    if (this.loadPromise) {
      return this.loadPromise;
    }

    this.loadPromise = (async () => {
      try {
        const response = await fetch(
          WASM_JS_URL,
          { cache: "no-store" },
        );

        if (!response.ok) {
          throw new Error(
            `WASM JS HTTP ${response.status}: ${WASM_JS_URL}`,
          );
        }

        const contentType =
          String(
            response.headers.get("content-type") ?? "",
          ).toLowerCase();

        if (contentType.includes("text/html")) {
          throw new Error(
            "WASM JS URL returned HTML instead of JavaScript",
          );
        }

        const wasm =
          await importPublicWasmModule(WASM_JS_URL);

        if (typeof wasm.default !== "function") {
          throw new Error(
            "WASM GLUE INVALID: DEFAULT INITIALIZER NOT FOUND",
          );
        }

        if (typeof wasm.RayTracerEngine !== "function") {
          throw new Error(
            "WASM GLUE INVALID: RayTracerEngine NOT FOUND",
          );
        }

        await wasm.default();

        const engine =
          new wasm.RayTracerEngine();

        this.wasm = wasm;
        this.engine = engine;

        this.emit("ready", {
          engine,
          wasm,
        });

        return engine;
      } catch (error) {
        throw this.fail(error);
      } finally {
        this.loadPromise = null;
      }
    })();

    return this.loadPromise;
  }

  public requireEngine(): RayTracerEngine {
    if (!this.engine) {
      throw new Error("RUNTIME NOT READY");
    }

    return this.engine;
  }

  public config(): TrainingWorkerConfig {
    const target = this.requireEngine();

    return {
      difficulty:
        typeof target.difficulty === "function"
          ? clampInt(
              target.difficulty(),
              ...TRAINING_LIMITS.difficulty,
              DEFAULT_TRAINING_CONFIG.difficulty,
            )
          : DEFAULT_TRAINING_CONFIG.difficulty,
      obstacles:
        typeof target.configured_obstacle_count === "function"
          ? clampInt(
              target.configured_obstacle_count(),
              ...TRAINING_LIMITS.obstacles,
              DEFAULT_TRAINING_CONFIG.obstacles,
            )
          : DEFAULT_TRAINING_CONFIG.obstacles,
      episodeSteps:
        typeof target.episode_limit === "function"
          ? clampInt(
              target.episode_limit(),
              ...TRAINING_LIMITS.episodeSteps,
              DEFAULT_TRAINING_CONFIG.episodeSteps,
            )
          : DEFAULT_TRAINING_CONFIG.episodeSteps,
      learningRate:
        typeof target.learning_rate === "function"
          ? clamp(
              target.learning_rate(),
              ...TRAINING_LIMITS.learningRate,
              DEFAULT_TRAINING_CONFIG.learningRate,
            )
          : DEFAULT_TRAINING_CONFIG.learningRate,
      gamma:
        typeof target.gamma === "function"
          ? clamp(
              target.gamma(),
              ...TRAINING_LIMITS.gamma,
              DEFAULT_TRAINING_CONFIG.gamma,
            )
          : DEFAULT_TRAINING_CONFIG.gamma,
      batchSize: DEFAULT_TRAINING_CONFIG.batchSize,
      epsilon: clamp(
        target.epsilon(),
        ...TRAINING_LIMITS.epsilon,
        DEFAULT_TRAINING_CONFIG.epsilon,
      ),
      epsilonMin: DEFAULT_TRAINING_CONFIG.epsilonMin,
      epsilonDecay: DEFAULT_TRAINING_CONFIG.epsilonDecay,
      targetUpdate: DEFAULT_TRAINING_CONFIG.targetUpdate,
    };
  }

  public applyTrainingConfig(
    input: Partial<TrainingWorkerConfig>,
  ): TrainingWorkerConfig {
    const target = this.requireEngine();
    const config = normalizeTrainingConfig(input);

    target.set_difficulty(config.difficulty);
    target.set_obstacle_count(config.obstacles);
    target.set_episode_steps(config.episodeSteps);
    target.set_learning_rate(config.learningRate);
    target.set_gamma(config.gamma);
    target.set_batch_size(config.batchSize);
    target.set_epsilon(config.epsilon);
    target.set_epsilon_min(config.epsilonMin);
    target.set_epsilon_decay(config.epsilonDecay);
    target.set_target_update(config.targetUpdate);

    return config;
  }

  public statusJson(): string {
    return this.requireEngine().status_json();
  }

  public telemetry(): RuntimeTelemetry {
    return readTelemetry(this.requireEngine());
  }

  public exportModel(): string {
    return this.requireEngine().export_model();
  }

  public saveModel(): string {
    return this.requireEngine().save_model();
  }

  public importModel(data: string): boolean {
    if (!data.trim()) {
      return false;
    }

    return this.requireEngine().import_model(data);
  }

  public loadModel(data: string): boolean {
    if (!data.trim()) {
      return false;
    }

    return this.requireEngine().load_model(data);
  }

  public async saveCheckpoint(): Promise<CheckpointRecord> {
    const data = this.saveModel();

    const record = await saveCheckpointData(data);
    this.emit("checkpointSaved", record);
    return record;
  }

  public async loadCheckpoint(): Promise<{
    loaded: boolean;
    source: "indexeddb" | "localstorage" | "none";
    kind: "checkpoint" | "legacy" | "unknown";
    data: string | null;
  }> {
    const target = this.requireEngine();
    const result =
      await importBestAvailableCheckpoint(target);

    if (result.loaded && result.data) {
      this.emit("checkpointLoaded", {
        data: result.data,
        kind: result.kind,
      });
    }

    return result;
  }

  public async clearCheckpoint(): Promise<void> {
    await clearCheckpointData();
    this.emit("checkpointCleared", undefined);
  }

  public async importCheckpointFile(): Promise<{
    imported: boolean;
    data: string | null;
    kind: "checkpoint" | "legacy" | "unknown";
  }> {
    const data = await pickTextFile();

    if (!data) {
      return {
        imported: false,
        data: null,
        kind: "unknown",
      };
    }

    const kind = modelKind(data);

    if (
      kind !== "checkpoint" &&
      kind !== "legacy"
    ) {
      return {
        imported: false,
        data,
        kind,
      };
    }

    const imported =
      this.requireEngine().import_model(data);

    if (imported) {
      this.emit("checkpointLoaded", {
        data,
        kind,
      });
    }

    return {
      imported,
      data,
      kind,
    };
  }

  public exportCheckpointFile(
    filename = checkpointFilename(),
  ): string {
    const data = this.saveModel();
    downloadText(data, filename);
    return data;
  }

  public async loadWorker(): Promise<Worker> {
    if (
      typeof Worker === "undefined"
    ) {
      throw new Error("WEB WORKER UNAVAILABLE");
    }

    return new Worker(
      new URL(
        "../workers/training.worker.ts",
        import.meta.url,
      ),
      { type: "module" },
    );
  }

  public onWorkerMessage(
    listener: RuntimeWorkerListener,
  ): () => void {
    this.workerListeners.add(listener);

    return () => {
      this.workerListeners.delete(listener);
    };
  }

  private broadcastWorkerMessage(
    message:
      | WorkerProgress
      | WorkerComplete
      | WorkerFailure,
  ): void {
    for (const listener of this.workerListeners) {
      try {
        listener(message);
      } catch {
        // UI callbacks are isolated from the worker runtime.
      }
    }

    if (message.type === "PROGRESS") {
      this.emit("workerProgress", message);
    } else if (message.type === "COMPLETE" || message.type === "EVAL_COMPLETE") {
      this.emit("workerComplete", message);
    } else if (message.type === "ERROR") {
      this.emit("workerError", message);
    }
  }

  public get activeWorkerJob(): RuntimeJob | null {
    return this.workerJob
      ? { ...this.workerJob }
      : null;
  }

  public async startWorkerJob(options: {
    mode: TrainingWorkerMode;
    requested: number;
    chunk?: number;
    seed?: number;
    config: Partial<TrainingWorkerConfig>;
    model?: string;
  }): Promise<RuntimeJob> {
    await this.load();

    this.stopWorker();

    const worker = await this.loadWorker();
    this.worker = worker;

    const jobId = ++this.workerJobId;
    const generation = ++this.workerGeneration;

    const job: RuntimeJob = {
      id: jobId,
      mode: options.mode,
      requested: Math.max(
        0,
        integer(options.requested, 0),
      ),
      startedAt: Date.now(),
      active: true,
    };

    this.workerJob = job;

    worker.addEventListener("message", (event: MessageEvent) => {
      const message =
        (event.data ?? {}) as WorkerProgress | WorkerComplete | WorkerFailure;

      if (generation !== this.workerGeneration) {
        return;
      }

      if (
        finite(message.jobId, -1) !== jobId
      ) {
        return;
      }

      if (
        message.type === "COMPLETE" ||
        message.type === "EVAL_COMPLETE" ||
        message.type === "ERROR"
      ) {
        job.active = false;
      }

      this.broadcastWorkerMessage(message);
    });

    worker.addEventListener("error", (event) => {
      if (generation !== this.workerGeneration) {
        return;
      }

      job.active = false;

      const failure: WorkerFailure = {
        type: "ERROR",
        jobId,
        message:
          event.error instanceof Error
            ? event.error.message
            : event.message || "TRAINING WORKER ERROR",
      };

      this.broadcastWorkerMessage(failure);
    });

    const model =
      typeof options.model === "string"
        ? options.model
        : this.safeExportModel();

    worker.postMessage({
      type: "START",
      jobId,
      mode: options.mode,
      requested: job.requested,
      chunk: normalizeWorkerChunk(options.chunk),
      seed:
        finite(
          options.seed,
          Date.now() >>> 0,
        ) >>> 0,
      model,
      config: normalizeTrainingConfig(options.config),
    });

    return { ...job };
  }

  public async startEvaluation(options: {
    count: number;
    baseSeed: number;
    config: Partial<TrainingWorkerConfig>;
    model?: string;
  }): Promise<RuntimeJob> {
    await this.load();

    this.stopWorker();

    const worker = await this.loadWorker();
    this.worker = worker;

    const jobId = ++this.workerJobId;
    const generation = ++this.workerGeneration;

    const job: RuntimeJob = {
      id: jobId,
      mode: "evaluate",
      requested: clampInt(
        options.count,
        1,
        10000,
        1,
      ),
      startedAt: Date.now(),
      active: true,
    };

    this.workerJob = job;

    worker.addEventListener("message", (event: MessageEvent) => {
      const message =
        (event.data ?? {}) as WorkerProgress | WorkerComplete | WorkerFailure;

      if (generation !== this.workerGeneration) {
        return;
      }

      if (
        finite(message.jobId, -1) !== jobId
      ) {
        return;
      }

      if (
        message.type === "COMPLETE" ||
        message.type === "EVAL_COMPLETE" ||
        message.type === "ERROR"
      ) {
        job.active = false;
      }

      this.broadcastWorkerMessage(message);
    });

    worker.addEventListener("error", (event) => {
      if (generation !== this.workerGeneration) {
        return;
      }

      job.active = false;

      const failure: WorkerFailure = {
        type: "ERROR",
        jobId,
        message:
          event.error instanceof Error
            ? event.error.message
            : event.message || "EVALUATION WORKER ERROR",
      };

      this.broadcastWorkerMessage(failure);
    });

    worker.postMessage({
      type: "EVALUATE",
      jobId,
      requested: job.requested,
      baseSeed:
        finite(
          options.baseSeed,
          Date.now() >>> 0,
        ) >>> 0,
      model:
        typeof options.model === "string"
          ? options.model
          : this.safeExportModel(),
      config: normalizeTrainingConfig(options.config),
    });

    return { ...job };
  }

  public stopWorker(): void {
    this.workerGeneration += 1;
    this.workerJob = null;

    if (this.worker) {
      try {
        this.worker.postMessage({
          type: "STOP",
        });
      } catch {
        // Worker may already be shutting down.
      }

      try {
        this.worker.terminate();
      } catch {
        // Best effort.
      }
    }

    this.worker = null;
  }

  public destroy(): void {
    this.stopWorker();
    this.engine = null;
    this.wasm = null;
    this.loadPromise = null;
  }

  private safeExportModel(): string {
    try {
      return this.exportModel();
    } catch {
      return "";
    }
  }
}

let singleton: AIRuntime | null = null;

export function getAIRuntime(): AIRuntime {
  if (!singleton) {
    singleton = new AIRuntime();
  }

  return singleton;
}

export async function getAIEngine(): Promise<RayTracerEngine> {
  return getAIRuntime().load();
}

export function resetAIRuntimeSingleton(): void {
  singleton?.destroy();
  singleton = null;
}
