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

  train_steps(
    count: number,
  ): number[] | Float32Array;

  train_episode():
    number[] | Float32Array;

  evaluate_episodes(
    count: number,
    seed: number,
  ): number[] | Float32Array;

  metrics(): number[] | Float32Array;
  export_model(): string;
  import_model(data: string): boolean;
};

type WasmModule = {
  default(
    input?: unknown,
  ): Promise<unknown>;

  RayTracerEngine:
    new () => Engine;
};

const BASE = (() => {
  const env =
    (import.meta as ImportMeta & {
      env?: { BASE_URL?: string };
    }).env;

  const raw =
    env?.BASE_URL ||
    "/";

  return raw.endsWith("/")
    ? raw
    : `${raw}/`;
})();

const WASM_JS_URL =
  new URL(
    "wasm/raytracer.js",
    new URL(
      BASE,
      self.location.href,
    ),
  ).href;

let engine:
  | Engine
  | null = null;

let activeJobId = -1;
let active = false;

function finite(
  value: unknown,
  fallback = 0,
): number {
  const numeric =
    Number(value);

  return Number.isFinite(
    numeric,
  )
    ? numeric
    : fallback;
}

function safeBuffer(
  value: unknown,
): number[] {
  if (
    value === null ||
    value === undefined
  ) {
    return [];
  }

  if (
    Array.isArray(value)
  ) {
    return value.map(
      (item) =>
        finite(item),
    );
  }

  if (
    ArrayBuffer.isView(value)
  ) {
    const view =
      value as unknown as
        ArrayLike<unknown>;

    return Array.from(
      {
        length:
          view.length,
      },
      (_, index) =>
        finite(
          view[index],
        ),
    );
  }

  return [];
}

async function
importPublicWasmModule(
  url: string,
): Promise<WasmModule> {
  const nativeImport =
    new Function(
      "url",
      "return import(url);",
    ) as (
      url: string,
    ) => Promise<WasmModule>;

  return nativeImport(
    url,
  );
}

async function loadEngine():
  Promise<Engine> {
  if (
    engine
  ) {
    return engine;
  }

  const response =
    await fetch(
      WASM_JS_URL,
      {
        cache:
          "no-cache",
      },
    );

  if (
    !response.ok
  ) {
    throw new Error(
      `WASM loader HTTP ${response.status}`,
    );
  }

  const contentType =
    response.headers.get(
      "content-type",
    ) ?? "";

  if (
    contentType.includes(
      "text/html",
    )
  ) {
    throw new Error(
      "WASM loader resolved to HTML instead of JavaScript",
    );
  }

  const imported =
    await importPublicWasmModule(
      WASM_JS_URL,
    );

  await imported.default();

  engine =
    new imported.RayTracerEngine();

  return engine;
}

function applyConfig(
  target: Engine,
  config:
    TrainingWorkerConfig,
): void {
  target.set_difficulty(
    Math.max(
      1,
      Math.min(
        10,
        Math.floor(
          config.difficulty,
        ),
      ),
    ),
  );

  target.set_obstacle_count(
    Math.max(
      1,
      Math.min(
        64,
        Math.floor(
          config.obstacles,
        ),
      ),
    ),
  );

  target.set_episode_steps(
    Math.max(
      50,
      Math.min(
        5000,
        Math.floor(
          config.episodeSteps,
        ),
      ),
    ),
  );

  target.set_learning_rate(
    Math.max(
      0.00001,
      Math.min(
        0.1,
        config.learningRate,
      ),
    ),
  );

  target.set_gamma(
    Math.max(
      0.80,
      Math.min(
        0.9999,
        config.gamma,
      ),
    ),
  );

  target.set_batch_size(
    Math.max(
      1,
      Math.min(
        256,
        Math.floor(
          config.batchSize,
        ),
      ),
    ),
  );

  target.set_epsilon(
    Math.max(
      0,
      Math.min(
        1,
        config.epsilon,
      ),
    ),
  );

  target.set_epsilon_min(
    Math.max(
      0,
      Math.min(
        1,
        config.epsilonMin,
      ),
    ),
  );

  target.set_epsilon_decay(
    Math.max(
      0.9,
      Math.min(
        0.999999,
        config.epsilonDecay,
      ),
    ),
  );

  target.set_target_update(
    Math.max(
      1,
      Math.min(
        100000,
        Math.floor(
          config.targetUpdate,
        ),
      ),
    ),
  );
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

  // Backward-compatible fallback for an older wasm-bindgen build.
  // metrics()[1] is training_steps in the Rust engine contract.
  return finite(metricVector(target)[1]);
}

function terminalFlag(
  target: Engine,
  kind: "done" | "success" | "collision",
): boolean {
  if (kind === "done" && typeof target.last_terminal_done === "function") {
    return target.last_terminal_done();
  }
  if (kind === "success" && typeof target.last_terminal_success === "function") {
    return target.last_terminal_success();
  }
  if (kind === "collision" && typeof target.last_terminal_collision === "function") {
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

function telemetry(
  target: Engine,
): Record<string, unknown> {
  return {
    backend:
      target.backend(),

    version:
      target.version(),

    seed:
      finite(
        target.current_seed(),
      ),

    step:
      finite(
        target.current_step(),
      ),

    trainingSteps:
      finite(
        trainingStepCount(target),
      ),

    episode:
      finite(
        target.current_episode(),
      ),

    episodeSteps:
      finite(
        target.last_episode_steps(),
      ),

    reward:
      finite(
        target.last_reward(),
      ),

    episodeReward:
      finite(
        target.last_episode_reward(),
      ),

    loss:
      finite(
        target.last_loss(),
      ),

    epsilon:
      finite(
        target.epsilon(),
      ),

    replay:
      finite(
        target.replay_size(),
      ),

    success:
      target.is_success(),

    collision:
      target.is_collision(),

    done:
      target.is_done(),

    lastTerminalDone:
      terminalFlag(target, "done"),

    lastTerminalSuccess:
      terminalFlag(target, "success"),

    lastTerminalCollision:
      terminalFlag(target, "collision"),

    lastTerminalDistance:
      finite(
        terminalDistance(target),
      ),

    sensorCount:
      finite(
        target.sensor_count(),
      ),

    observationSize:
      finite(
        target.observation_size(),
      ),

    actionCount:
      finite(
        target.action_count(),
      ),

    parameterCount:
      finite(
        target.model_parameter_count(),
      ),
  };
}

function post(
  type: string,
  payload:
    Record<string, unknown> = {},
): void {
  self.postMessage({
    type,
    jobId:
      activeJobId,
    ...payload,
  });
}

function runTrainingChunk(
  mode:
    | "steps"
    | "episodes"
    | "continuous",

  requested: number,

  chunk: number,

  completed: number,

): void {
  if (
    !active ||
    !engine
  ) {
    return;
  }

  try {
    const modelSyncStride =
      mode === "episodes"
        ? 4
        : 1024;

    let nextCompleted =
      completed;

    if (
      mode ===
      "episodes"
    ) {
      engine.train_episode();
      nextCompleted += 1;
    } else {
      const remaining =
        Math.max(
          0,
          requested -
            completed,
        );

      const amount =
        mode ===
          "continuous"
          ? chunk
          : Math.min(
              chunk,
              remaining,
            );

      if (
        amount > 0
      ) {
        engine.train_steps(
          amount,
        );

        nextCompleted +=
          amount;
      }
    }

    const finished =
      mode ===
        "continuous"
        ? false
        : nextCompleted >=
          requested;

    const syncModel =
      finished ||
      (
        nextCompleted > 0 &&
        (
          nextCompleted %
            modelSyncStride
        ) < 1
      );

    const progressPayload:
      Record<string, unknown> = {
        mode:
          mode ===
          "episodes"
            ? "TRAIN-EPISODES"
            : "TRAIN",

        state:
          "RUNNING",

        status:
          requested > 0
            ? `TRAINING ${nextCompleted}/${requested}`
            : `TRAINING ${nextCompleted}`,

        completed:
          nextCompleted,

        requested,

        message:
          requested > 0
            ? `TRAINING ${nextCompleted}/${requested}`
            : `CONTINUOUS TRAINING ${nextCompleted}`,

        ...telemetry(
          engine,
        ),
      };

    /*
     * Model snapshots are sent periodically, not on every small
     * worker chunk. This keeps the main thread smooth while still
     * allowing the live preview to follow learning.
     */
    if (syncModel) {
      progressPayload.previewModel =
        engine.export_model();
    }

    post(
      "PROGRESS",
      progressPayload,
    );

    if (
      finished
    ) {
      const model =
        engine.export_model();

      post(
        "COMPLETE",
        {
          mode:
            mode ===
            "episodes"
              ? "TRAIN-EPISODES"
              : "TRAIN",

          state:
            "COMPLETE",

          status:
            "TRAINING COMPLETE / MODEL EXPORTED",

          completed:
            nextCompleted,

          requested,

          model,

          message:
            "TRAINING COMPLETE / MODEL APPLIED TO PREVIEW",

          ...telemetry(
            engine,
          ),
        },
      );

      active =
        false;

      return;
    }

    /*
     * Yield to the worker event loop.
     * This makes cancellation responsive
     * between WASM chunks.
     */
    setTimeout(
      () =>
        runTrainingChunk(
          mode,
          requested,
          chunk,
          nextCompleted,
        ),
      0,
    );
  } catch (
    error
  ) {
    active =
      false;

    post(
      "ERROR",
      {
        message:
          error instanceof Error
            ? error.message
            : String(error),
      },
    );
  }
}

async function startTraining(
  message:
    Record<string, unknown>,
): Promise<void> {
  activeJobId =
    finite(
      message.jobId,
      -1,
    );

  active =
    true;

  const target =
    await loadEngine();

  applyConfig(
    target,
    (message.config ??
      {}) as TrainingWorkerConfig,
  );

  const model =
    String(
      message.model ??
        "",
    );

  if (
    model &&
    !target.import_model(
      model,
    )
  ) {
    throw new Error(
      "training worker could not import the current model",
    );
  }

  const seed =
    finite(
      message.seed,
      Date.now() >>> 0,
    );

  target.reset_environment(
    seed,
  );

  const mode =
    String(
      message.mode ??
        "steps",
    ) as
      | "steps"
      | "episodes"
      | "continuous";

  const requested =
    Math.max(
      0,
      Math.floor(
        finite(
          message.requested,
          0,
        ),
      ),
    );

  const chunk =
    Math.max(
      128,
      Math.min(
        2048,
        Math.floor(
          finite(
            message.chunk,
            1024,
          ),
        ),
      ),
    );

  post(
    "PROGRESS",
    {
      mode:
        mode ===
        "episodes"
          ? "TRAIN-EPISODES"
          : "TRAIN",

      state:
        "RUNNING",

      status:
        "TRAINING WORKER READY",

      completed:
        0,

      requested,

      ...telemetry(
        target,
      ),
    },
  );

  runTrainingChunk(
    mode,
    requested,
    chunk,
    0,
  );
}

async function evaluate(
  message:
    Record<string, unknown>,
): Promise<void> {
  activeJobId =
    finite(
      message.jobId,
      -1,
    );

  active =
    true;

  const target =
    await loadEngine();

  applyConfig(
    target,
    (message.config ??
      {}) as TrainingWorkerConfig,
  );

  const model =
    String(
      message.model ??
        "",
    );

  if (
    model &&
    !target.import_model(
      model,
    )
  ) {
    throw new Error(
      "evaluation worker could not import the current model",
    );
  }

  const requested =
    Math.max(
      1,
      Math.min(
        10000,
        Math.floor(
          finite(
            message.requested,
            1,
          ),
        ),
      ),
    );

  const baseSeed =
    finite(
      message.baseSeed,
      Date.now() >>> 0,
    );

  const result =
    safeBuffer(
      target.evaluate_episodes(
        requested,
        baseSeed,
      ),
    );

  const average =
    result.length > 0
      ? result.reduce(
          (
            total,
            value,
          ) =>
            total +
            value,
          0,
        ) /
        result.length
      : 0;

  post(
    "EVAL_COMPLETE",
    {
      requested,
      baseSeed,
      results:
        result,
      average,
      ...telemetry(
        target,
      ),
    },
  );

  active =
    false;
}

self.addEventListener(
  "message",
  (
    event:
      MessageEvent,
  ) => {
    const message =
      (event.data ??
        {}) as Record<
        string,
        unknown
      >;

    const type =
      String(
        message.type ??
          "",
      ).toUpperCase();

    if (
      type ===
      "STOP"
    ) {
      active =
        false;
      return;
    }

    if (
      type ===
      "START"
    ) {
      startTraining(
        message,
      ).catch(
        (
          error,
        ) => {
          active =
            false;

          activeJobId =
            finite(
              message.jobId,
              -1,
            );

          post(
            "ERROR",
            {
              message:
                error instanceof
                Error
                  ? error.message
                  : String(
                      error,
                    ),
            },
          );
        },
      );

      return;
    }

    if (
      type ===
      "EVALUATE"
    ) {
      evaluate(
        message,
      ).catch(
        (
          error,
        ) => {
          active =
            false;

          activeJobId =
            finite(
              message.jobId,
              -1,
            );

          post(
            "ERROR",
            {
              message:
                error instanceof
                Error
                  ? error.message
                  : String(
                      error,
                    ),
            },
          );
        },
      );
    }
  },
);
