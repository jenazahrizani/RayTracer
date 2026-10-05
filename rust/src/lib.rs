use std::fmt::Write as FmtWrite;

use wasm_bindgen::prelude::*;

mod ai;
mod checkpoint;
mod environment;

// ============================================================================
// RAYTRC.AI / NAVIGATION ENGINE
// ============================================================================
//
// This file is intentionally the small WASM-facing orchestration layer.
//
// Responsibilities kept here:
//   - WASM public API
//   - engine lifecycle
//   - compatibility with the existing Astro/Worker API
//   - terminal-event bookkeeping
//   - telemetry/status serialization
//
// Responsibilities moved into modules:
//   ai.rs
//       DQN network, replay, exploration, expert bootstrap, training.
//
//   environment.rs
//       world generation, agent movement, collisions, sensors, observations,
//       rewards, render buffers.
//
//   checkpoint.rs
//       model/checkpoint serialization and restoration.
//
// The public method names intentionally remain compatible with the current
// Renderer.astro and training.worker.ts contract so the frontend can be
// refactored independently from the Rust internals.
//
// Target runtime:
//   Rust -> WASM -> Astro -> Web Worker -> GitHub Pages
//
// No server is required.
// No image model is used.
// ============================================================================

// ============================================================================
// PUBLICLY SHARED ENGINE CONSTANTS
// ============================================================================

pub(crate) const OBS_SIZE: usize = 37;
pub(crate) const SENSOR_COUNT: usize = 32;
pub(crate) const ACTION_COUNT: usize = 5;

pub(crate) const HIDDEN_1: usize = 64;
pub(crate) const HIDDEN_2: usize = 64;

pub(crate) const WORLD_HALF: f32 = 10.0;
pub(crate) const WORLD_SIZE: f32 = WORLD_HALF * 2.0;

pub(crate) const AGENT_RADIUS: f32 = 0.30;
pub(crate) const TARGET_RADIUS: f32 = 0.70;
pub(crate) const MAX_SENSOR_DISTANCE: f32 = 14.5;

pub(crate) const MAX_SPEED: f32 = 0.16;
pub(crate) const TURN_RATE: f32 = 0.18;

pub(crate) const DEFAULT_MAX_STEPS: u32 = 500;
pub(crate) const DEFAULT_REPLAY_CAPACITY: usize = 20_000;
pub(crate) const DEFAULT_BATCH_SIZE: usize = 64;
pub(crate) const DEFAULT_TARGET_UPDATE: u64 = 128;

pub(crate) const DEFAULT_LEARNING_RATE: f32 = 0.0005;
pub(crate) const DEFAULT_GAMMA: f32 = 0.99;
pub(crate) const DEFAULT_EPSILON: f32 = 1.0;
pub(crate) const DEFAULT_EPSILON_MIN: f32 = 0.05;
pub(crate) const DEFAULT_EPSILON_DECAY: f32 = 0.99998;

pub(crate) const ENGINE_VERSION: &str = "RAYTRC.AI.NAV/1.4.0";
pub(crate) const ENGINE_BACKEND: &str = "RUST + WASM + DQN + 32-RAY SENSORS + RECOVERY SHIELD";

// ============================================================================
// SHARED LOW-LEVEL TYPES
// ============================================================================

/// Small 3D vector used by the environment module.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Vec3 {
    pub(crate) x: f32,
    pub(crate) y: f32,
    pub(crate) z: f32,
}

impl Vec3 {
    pub(crate) const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub(crate) fn distance_xz(self, other: Self) -> f32 {
        let dx = self.x - other.x;
        let dz = self.z - other.z;

        (dx * dx + dz * dz).sqrt()
    }
}

/// Axis-aligned X/Z obstacle bounds used by the environment.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Aabb {
    pub(crate) center: Vec3,
    pub(crate) half: Vec3,
}

impl Aabb {
    pub(crate) fn min_x(&self) -> f32 {
        self.center.x - self.half.x
    }

    pub(crate) fn max_x(&self) -> f32 {
        self.center.x + self.half.x
    }

    pub(crate) fn min_z(&self) -> f32 {
        self.center.z - self.half.z
    }

    pub(crate) fn max_z(&self) -> f32 {
        self.center.z + self.half.z
    }
}

/// Environment obstacle. Material is retained for renderer compatibility.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Obstacle {
    pub(crate) bounds: Aabb,
    pub(crate) material: u32,
}

/// Current agent state.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Agent {
    pub(crate) position: Vec3,
    pub(crate) heading: f32,
    pub(crate) speed: f32,
}

/// Result of a single environment transition.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct StepResult {
    pub(crate) reward: f32,
    pub(crate) done: bool,
    pub(crate) success: bool,
    pub(crate) collision: bool,
    pub(crate) distance_to_target: f32,
}

// ============================================================================
// ENGINE
// ============================================================================

#[wasm_bindgen]
pub struct RayTracerEngine {
    // These fields are crate-visible so checkpoint.rs can serialize the full
    // engine state without exposing implementation details to JavaScript.
    pub(crate) environment: environment::Environment,
    pub(crate) ai: ai::AiEngine,

    // Terminal state is intentionally stored separately from Environment.
    // The manual `step()` API keeps terminal maps sticky, while training
    // immediately advances into the next episode.
    last_terminal_done: bool,
    last_terminal_success: bool,
    last_terminal_collision: bool,
    last_terminal_distance: f32,
}

#[wasm_bindgen]
impl RayTracerEngine {
    // ========================================================================
    // CREATE
    // ========================================================================

    #[wasm_bindgen(constructor)]
    pub fn new() -> RayTracerEngine {
        let ai = ai::AiEngine::new(0x5241595452435F41);

        let environment = environment::Environment::new(1, 1, 4, DEFAULT_MAX_STEPS);

        let last_terminal_distance = environment.last_distance();

        Self {
            environment,
            ai,
            last_terminal_done: false,
            last_terminal_success: false,
            last_terminal_collision: false,
            last_terminal_distance,
        }
    }

    // ========================================================================
    // ENGINE INFO
    // ========================================================================

    pub fn version(&self) -> String {
        ENGINE_VERSION.to_string()
    }

    pub fn backend(&self) -> String {
        ENGINE_BACKEND.to_string()
    }

    pub fn sensor_count(&self) -> u32 {
        SENSOR_COUNT as u32
    }

    pub fn observation_size(&self) -> u32 {
        OBS_SIZE as u32
    }

    pub fn action_count(&self) -> u32 {
        ACTION_COUNT as u32
    }

    pub fn model_parameter_count(&self) -> u32 {
        ai::AiEngine::parameter_count() as u32
    }

    // ========================================================================
    // ACTION NAMES
    // ========================================================================

    pub fn action_name(&self, action: u32) -> String {
        ai::action_name(action).to_string()
    }

    // ========================================================================
    // RESET AI
    // ========================================================================

    pub fn reset_ai(&mut self) {
        self.ai.reset();

        self.clear_terminal_state();
    }

    // ========================================================================
    // RESET ENVIRONMENT
    // ========================================================================

    pub fn reset_environment(&mut self, seed: u32) {
        self.environment.reset(seed);

        self.clear_terminal_state();

        self.last_terminal_distance = self.environment.last_distance();
    }

    // ========================================================================
    // COLLISION RECOVERY / SAME MAP
    // ========================================================================

    pub fn recover_from_collision(&mut self) {
        if !self.environment.is_collision_terminal() {
            return;
        }

        self.environment.recover_from_collision();

        self.clear_terminal_state();

        self.last_terminal_distance = self.environment.last_distance();
    }

    // ========================================================================
    // HARD RESET
    // ========================================================================

    pub fn hard_reset(&mut self) {
        self.ai.reset();

        self.environment.set_difficulty(1);
        self.environment.set_obstacle_count(4);
        self.environment.set_episode_steps(DEFAULT_MAX_STEPS);

        let seed = self.ai.next_seed();

        self.environment.reset(seed);

        self.clear_terminal_state();

        self.last_terminal_distance = self.environment.last_distance();
    }

    // ========================================================================
    // ENVIRONMENT CONFIG
    // ========================================================================

    pub fn set_difficulty(&mut self, difficulty: u32) {
        self.environment.set_difficulty(difficulty);

        self.clear_terminal_state();
    }

    pub fn set_obstacle_count(&mut self, count: u32) {
        self.environment.set_obstacle_count(count);

        self.clear_terminal_state();
    }

    pub fn set_episode_steps(&mut self, steps: u32) {
        self.environment.set_episode_steps(steps);
    }

    pub fn difficulty(&self) -> u32 {
        self.environment.difficulty()
    }

    pub fn configured_obstacle_count(&self) -> u32 {
        self.environment.configured_obstacle_count()
    }

    pub fn episode_limit(&self) -> u32 {
        self.environment.episode_limit()
    }

    // ========================================================================
    // AI CONFIG
    // ========================================================================

    pub fn set_learning_rate(&mut self, value: f32) {
        self.ai.set_learning_rate(value);
    }

    pub fn set_gamma(&mut self, value: f32) {
        self.ai.set_gamma(value);
    }

    pub fn set_batch_size(&mut self, value: u32) {
        self.ai.set_batch_size(value);
    }

    pub fn set_epsilon(&mut self, value: f32) {
        self.ai.set_epsilon(value);
    }

    pub fn set_epsilon_min(&mut self, value: f32) {
        self.ai.set_epsilon_min(value);
    }

    pub fn set_epsilon_decay(&mut self, value: f32) {
        self.ai.set_epsilon_decay(value);
    }

    pub fn set_target_update(&mut self, value: u32) {
        self.ai.set_target_update(value);
    }

    // ========================================================================
    // CURRENT STATE
    // ========================================================================

    pub fn current_seed(&self) -> u32 {
        self.environment.seed()
    }

    pub fn current_step(&self) -> u32 {
        self.environment.current_step()
    }

    pub fn current_episode(&self) -> u64 {
        self.ai.completed_episodes()
    }

    pub fn training_step_count(&self) -> u64 {
        self.ai.training_steps()
    }

    pub fn is_done(&self) -> bool {
        self.environment.is_done()
    }

    pub fn is_success(&self) -> bool {
        self.environment.is_success()
    }

    pub fn is_collision(&self) -> bool {
        self.environment.is_collision()
    }

    pub fn last_terminal_done(&self) -> bool {
        self.last_terminal_done
    }

    pub fn last_terminal_success(&self) -> bool {
        self.last_terminal_success
    }

    pub fn last_terminal_collision(&self) -> bool {
        self.last_terminal_collision
    }

    pub fn last_terminal_distance(&self) -> f32 {
        self.last_terminal_distance
    }

    pub fn epsilon(&self) -> f32 {
        self.ai.epsilon()
    }

    pub fn learning_rate(&self) -> f32 {
        self.ai.learning_rate()
    }

    pub fn gamma(&self) -> f32 {
        self.ai.gamma()
    }

    pub fn last_loss(&self) -> f32 {
        self.ai.last_loss()
    }

    pub fn last_reward(&self) -> f32 {
        self.ai.last_reward()
    }

    pub fn last_episode_reward(&self) -> f32 {
        self.ai.last_episode_reward()
    }

    pub fn last_episode_steps(&self) -> u32 {
        self.ai.last_episode_steps()
    }

    pub fn replay_size(&self) -> u32 {
        self.ai.replay_size() as u32
    }

    // ========================================================================
    // RAW ENVIRONMENT BUFFERS
    // ========================================================================

    pub fn get_observation(&self) -> Vec<f32> {
        self.environment.observation().to_vec()
    }

    pub fn get_sensor_buffer(&self) -> Vec<f32> {
        self.environment.sensor_buffer().to_vec()
    }

    pub fn get_agent_buffer(&self) -> Vec<f32> {
        self.environment.agent_buffer()
    }

    pub fn get_target_buffer(&self) -> Vec<f32> {
        self.environment.target_buffer()
    }

    pub fn get_obstacle_buffer(&self) -> Vec<f32> {
        self.environment.obstacle_buffer()
    }

    pub fn get_path_buffer(&self) -> Vec<f32> {
        self.environment.path_buffer()
    }

    pub fn obstacle_count(&self) -> u32 {
        self.environment.obstacle_count() as u32
    }

    // ========================================================================
    // AI PREDICTION
    // ========================================================================

    pub fn predict_action(&self) -> u32 {
        let observation = self.environment.observation();

        self.ai.predict_action(&observation) as u32
    }

    pub fn q_values(&self) -> Vec<f32> {
        let observation = self.environment.observation();

        self.ai.q_values(&observation).to_vec()
    }

    // ========================================================================
    // MANUAL / PREVIEW STEP
    // ========================================================================
    //
    // Terminal states are sticky for this API.
    // A caller must explicitly reset the environment or recover from
    // collision. This is what lets RUN distinguish:
    //
    //   collision -> same map recovery
    //   success   -> new map
    //
    // Training uses the separate train_step() API and automatically advances
    // into the next episode.
    // ========================================================================

    pub fn step(&mut self, action: u32) -> Vec<f32> {
        let action = (action as usize).min(ACTION_COUNT - 1);

        if self.environment.is_done() {
            self.last_terminal_done = true;
            self.last_terminal_success = self.environment.is_success();
            self.last_terminal_collision = self.environment.is_collision();
            self.last_terminal_distance = self.environment.last_distance();

            self.ai.record_manual_action(action, 0.0);

            return vec![
                0.0,
                1.0,
                if self.last_terminal_success { 1.0 } else { 0.0 },
                if self.last_terminal_collision {
                    1.0
                } else {
                    0.0
                },
                self.last_terminal_distance,
                action as f32,
            ];
        }

        let result = self.environment.step(action);

        self.ai.record_manual_action(action, result.reward);

        if result.done {
            self.record_terminal_result(result);
            self.ai.record_episode_result(&self.environment, result);
        } else {
            self.clear_terminal_state();
        }

        vec![
            result.reward,
            if result.done { 1.0 } else { 0.0 },
            if result.success { 1.0 } else { 0.0 },
            if result.collision { 1.0 } else { 0.0 },
            result.distance_to_target,
            action as f32,
        ]
    }

    // ========================================================================
    // TRAIN ONE STEP
    // ========================================================================

    pub fn train_step(&mut self) -> Vec<f32> {
        let report = self.ai.train_step(&mut self.environment);

        self.last_terminal_done = report.result.done;
        self.last_terminal_success = report.result.success;
        self.last_terminal_collision = report.result.collision;
        self.last_terminal_distance = report.result.distance_to_target;

        vec![
            self.ai.training_steps() as f32,
            self.ai.completed_episodes() as f32,
            report.result.reward,
            self.ai.last_loss(),
            self.ai.epsilon(),
            report.action as f32,
            if report.result.done { 1.0 } else { 0.0 },
            if report.result.success { 1.0 } else { 0.0 },
            if report.result.collision { 1.0 } else { 0.0 },
        ]
    }

    // ========================================================================
    // TRAIN N STEPS
    // ========================================================================

    pub fn train_steps(&mut self, count: u32) -> Vec<f32> {
        let count = count.min(1_000_000);

        for _ in 0..count {
            self.train_step();
        }

        self.metrics()
    }

    // ========================================================================
    // TRAIN ONE EPISODE
    // ========================================================================

    pub fn train_episode(&mut self) -> Vec<f32> {
        let starting_episode = self.ai.completed_episodes();

        let mut guard = 0usize;

        while self.ai.completed_episodes() == starting_episode
            && guard < self.environment.episode_limit() as usize + 2
        {
            self.train_step();
            guard += 1;
        }

        self.metrics()
    }

    // ========================================================================
    // TRAIN MULTIPLE EPISODES
    // ========================================================================

    pub fn train_episodes(&mut self, count: u32) -> Vec<f32> {
        let count = count.min(10_000);
        let target = self.ai.completed_episodes() + count as u64;

        let max_work = count as usize * (self.environment.episode_limit() as usize + 2);

        let mut guard = 0usize;

        while self.ai.completed_episodes() < target && guard < max_work {
            self.train_step();
            guard += 1;
        }

        self.metrics()
    }

    // ========================================================================
    // EVALUATION
    // ========================================================================

    pub fn evaluate_episodes(&self, count: u32, base_seed: u32) -> Vec<f32> {
        self.ai
            .evaluate_episodes(&self.environment, count, base_seed)
    }

    // ========================================================================
    // METRICS
    // ========================================================================

    pub fn metrics(&self) -> Vec<f32> {
        self.ai.metrics(&self.environment)
    }

    // ========================================================================
    // STATUS JSON
    // ========================================================================

    pub fn status_json(&self) -> String {
        let completed = self.ai.completed_episodes().max(1) as f32;

        let status = if self.environment.is_done() {
            if self.environment.is_success() {
                "SUCCESS"
            } else if self.environment.is_collision() {
                "COLLISION"
            } else {
                "DONE"
            }
        } else {
            "RUNNING"
        };

        let action = self.action_name(self.ai.last_action());

        let average_steps = if self.ai.completed_episodes() == 0 {
            0.0
        } else {
            self.ai.last_episode_steps() as f32
        };

        let mut output = String::with_capacity(2048);

        output.push('{');

        push_json_string(&mut output, "version", &self.version());

        push_json_string(&mut output, "backend", &self.backend());

        push_json_string(&mut output, "status", status);

        push_json_u32(&mut output, "seed", self.environment.seed());

        push_json_u32(&mut output, "difficulty", self.environment.difficulty());

        push_json_u32(
            &mut output,
            "obstacles",
            self.environment.obstacle_count() as u32,
        );

        push_json_u32(
            &mut output,
            "configuredObstacles",
            self.environment.configured_obstacle_count(),
        );

        push_json_u64(&mut output, "episode", self.ai.completed_episodes());

        push_json_u32(&mut output, "step", self.environment.current_step());

        push_json_u64(&mut output, "trainingSteps", self.ai.training_steps());

        push_json_f32(&mut output, "reward", self.ai.last_reward());

        push_json_f32(&mut output, "episodeReward", self.ai.last_episode_reward());

        push_json_f32(&mut output, "loss", self.ai.last_loss());

        push_json_f32(&mut output, "epsilon", self.ai.epsilon());

        push_json_f32(
            &mut output,
            "successRate",
            self.ai.success_episodes() as f32 / completed,
        );

        push_json_f32(
            &mut output,
            "collisionRate",
            self.ai.collision_episodes() as f32 / completed,
        );

        push_json_f32(&mut output, "avgEpisodeSteps", average_steps);

        push_json_u64(&mut output, "replay", self.ai.replay_size() as u64);

        push_json_u64(
            &mut output,
            "modelParameters",
            self.model_parameter_count() as u64,
        );

        push_json_u64(&mut output, "sensorCount", SENSOR_COUNT as u64);

        push_json_u64(&mut output, "observationSize", OBS_SIZE as u64);

        push_json_u64(&mut output, "actionCount", ACTION_COUNT as u64);

        push_json_bool(&mut output, "done", self.environment.is_done());

        push_json_bool(&mut output, "success", self.environment.is_success());

        push_json_bool(&mut output, "collision", self.environment.is_collision());

        push_json_string(&mut output, "action", &action);

        push_json_f32(&mut output, "learningRate", self.ai.learning_rate());

        push_json_f32(&mut output, "gamma", self.ai.gamma());

        push_json_u64(&mut output, "batchSize", self.ai.batch_size() as u64);

        push_json_u64(&mut output, "targetUpdate", self.ai.target_update());

        output.push('}');
        output
    }

    // ========================================================================
    // CHECKPOINT / MODEL API
    // ========================================================================

    /// Export the current full training state.
    ///
    /// The implementation lives in checkpoint.rs. Keeping the public API here
    /// means the browser does not need to know the serialization format.
    pub fn export_model(&self) -> String {
        checkpoint::export(self)
    }

    /// Import a full training state.
    pub fn import_model(&mut self, data: String) -> bool {
        let loaded = checkpoint::import(self, &data);

        if loaded {
            self.clear_terminal_state();
            self.last_terminal_distance = self.environment.last_distance();
        }

        loaded
    }

    pub fn save_model(&self) -> String {
        self.export_model()
    }

    pub fn load_model(&mut self, data: String) -> bool {
        self.import_model(data)
    }

    /// Export only the learned inference policy.
    ///
    /// This intentionally excludes the live environment so a background
    /// trainer can update the foreground RUN policy without changing the
    /// currently visible map, target, agent, or path.
    pub fn export_policy(&self) -> String {
        checkpoint::export_policy(self)
    }

    /// Import only learned policy parameters and trainer counters.
    ///
    /// The current environment is intentionally preserved.
    pub fn import_policy(&mut self, data: String) -> bool {
        checkpoint::import_policy(self, &data)
    }

    // ========================================================================
    // INTERNAL HELPERS
    // ========================================================================

    fn clear_terminal_state(&mut self) {
        self.last_terminal_done = false;
        self.last_terminal_success = false;
        self.last_terminal_collision = false;
    }

    fn record_terminal_result(&mut self, result: StepResult) {
        self.last_terminal_done = result.done;
        self.last_terminal_success = result.success;
        self.last_terminal_collision = result.collision;
        self.last_terminal_distance = result.distance_to_target;
    }
}

// ============================================================================
// JSON HELPERS
// ============================================================================

fn push_json_key(output: &mut String, key: &str) {
    if output.ends_with('{') {
        output.push('"');
    } else {
        output.push_str(",\"");
    }

    json_escape_into(output, key);
    output.push_str("\":");
}

fn push_json_string(output: &mut String, key: &str, value: &str) {
    push_json_key(output, key);
    output.push('"');
    json_escape_into(output, value);
    output.push('"');
}

fn push_json_bool(output: &mut String, key: &str, value: bool) {
    push_json_key(output, key);

    output.push_str(if value { "true" } else { "false" });
}

fn push_json_u32(output: &mut String, key: &str, value: u32) {
    push_json_key(output, key);

    let _ = write!(output, "{}", value,);
}

fn push_json_u64(output: &mut String, key: &str, value: u64) {
    push_json_key(output, key);

    let _ = write!(output, "{}", value,);
}

fn push_json_f32(output: &mut String, key: &str, value: f32) {
    push_json_key(output, key);

    if value.is_finite() {
        let _ = write!(output, "{:.8}", value,);
    } else {
        output.push_str("0.0");
    }
}

fn json_escape_into(output: &mut String, input: &str) {
    for character in input.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            character if character.is_control() => {
                let _ = write!(output, "\\u{:04X}", character as u32,);
            }
            character => output.push(character),
        }
    }
}

// ============================================================================
// WASM START
// ============================================================================

#[wasm_bindgen(start)]
pub fn wasm_start() {
    // The engine is created explicitly from JavaScript:
    //
    //   const engine = new RayTracerEngine();
    //
    // Keeping startup empty makes the WASM module deterministic and lets the
    // worker own engine lifecycle.
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_match_frontend_contract() {
        assert_eq!(OBS_SIZE, 37);
        assert_eq!(SENSOR_COUNT, 32);
        assert_eq!(ACTION_COUNT, 5);
        assert_eq!(HIDDEN_1, 64);
        assert_eq!(HIDDEN_2, 64);
    }

    #[test]
    fn network_parameter_count_is_6917() {
        assert_eq!(ai::AiEngine::parameter_count(), 6917,);
    }

    #[test]
    fn engine_starts_with_valid_environment() {
        let engine = RayTracerEngine::new();

        assert_eq!(engine.observation_size(), OBS_SIZE as u32,);

        assert_eq!(engine.sensor_count(), SENSOR_COUNT as u32,);

        assert_eq!(engine.action_count(), ACTION_COUNT as u32,);

        assert_eq!(engine.get_observation().len(), OBS_SIZE,);
    }

    #[test]
    fn observation_values_are_finite() {
        let engine = RayTracerEngine::new();

        assert!(engine
            .get_observation()
            .iter()
            .all(|value| value.is_finite()));
    }

    #[test]
    fn sensors_are_normalized() {
        let engine = RayTracerEngine::new();

        assert!(engine
            .get_sensor_buffer()
            .iter()
            .all(|value| { *value >= 0.0 && *value <= 1.0 }));
    }

    #[test]
    fn status_json_has_expected_shape() {
        let engine = RayTracerEngine::new();

        let json = engine.status_json();

        assert!(json.starts_with('{'));

        assert!(json.ends_with('}'));

        for field in [
            "\"version\"",
            "\"backend\"",
            "\"status\"",
            "\"seed\"",
            "\"episode\"",
            "\"trainingSteps\"",
            "\"sensorCount\"",
            "\"observationSize\"",
            "\"actionCount\"",
        ] {
            assert!(json.contains(field), "missing field: {field}");
        }
    }

    #[test]
    fn action_names_remain_stable() {
        let engine = RayTracerEngine::new();

        assert_eq!(engine.action_name(0), "FORWARD");
        assert_eq!(engine.action_name(1), "LEFT");
        assert_eq!(engine.action_name(2), "RIGHT");
        assert_eq!(engine.action_name(3), "BRAKE");
        assert_eq!(engine.action_name(4), "REVERSE");
    }

    #[test]
    fn manual_terminal_state_is_sticky() {
        let mut engine = RayTracerEngine::new();

        // Force a terminal environment state through the environment module.
        engine
            .environment
            .force_terminal_for_test(true, false, 0.25);

        let episode_before = engine.current_episode();

        let result = engine.step(0);

        assert_eq!(result[0], 0.0);

        assert_eq!(result[1], 1.0);

        assert_eq!(result[2], 1.0);

        assert_eq!(engine.current_episode(), episode_before);
    }

    #[test]
    fn json_helpers_do_not_duplicate_commas() {
        let mut output = String::from("{");

        push_json_string(&mut output, "a", "b");

        push_json_u32(&mut output, "n", 3);

        output.push('}');

        assert_eq!(output, "{\"a\":\"b\",\"n\":3}");
    }
}
