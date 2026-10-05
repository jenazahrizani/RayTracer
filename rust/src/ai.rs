use crate::environment::Environment;
use crate::{
    StepResult, ACTION_COUNT, DEFAULT_BATCH_SIZE, DEFAULT_EPSILON, DEFAULT_EPSILON_DECAY,
    DEFAULT_EPSILON_MIN, DEFAULT_GAMMA, DEFAULT_LEARNING_RATE, DEFAULT_REPLAY_CAPACITY,
    DEFAULT_TARGET_UPDATE, HIDDEN_1, HIDDEN_2, OBS_SIZE, SENSOR_COUNT,
};

// ============================================================================
// RAYTRC.AI / AI ENGINE
// ============================================================================
//
// Responsibilities:
//   - DQN network: 37 -> 64 -> 64 -> 5
//   - online + target networks
//   - replay buffer
//   - deterministic RNG
//   - Double-DQN targets
//   - Huber loss
//   - epsilon-greedy exploration
//   - geometry-guided expert bootstrap during TRAIN
//   - training/evaluation metrics
//
// The module deliberately does not know anything about Astro, DOM, Canvas,
// WebGPU, or the training Worker. `lib.rs` is the WASM-facing orchestrator.
// ============================================================================

// ============================================================================
// ACTIONS
// ============================================================================

pub(crate) const FORWARD: usize = 0;
pub(crate) const LEFT: usize = 1;
pub(crate) const RIGHT: usize = 2;
pub(crate) const BRAKE: usize = 3;
pub(crate) const REVERSE: usize = 4;

pub(crate) fn action_name(action: u32) -> &'static str {
    match action {
        0 => "FORWARD",
        1 => "LEFT",
        2 => "RIGHT",
        3 => "BRAKE",
        4 => "REVERSE",
        _ => "UNKNOWN",
    }
}

// ============================================================================
// REPLAY TRANSITION
// ============================================================================

#[derive(Clone, Copy, Debug)]
pub(crate) struct Transition {
    pub(crate) state: [f32; OBS_SIZE],
    pub(crate) action: u8,
    pub(crate) reward: f32,
    pub(crate) next_state: [f32; OBS_SIZE],
    pub(crate) done: bool,
}

// ============================================================================
// REPLAY BUFFER
// ============================================================================

#[derive(Clone, Debug)]
pub(crate) struct ReplayBuffer {
    pub(crate) items: Vec<Transition>,
    pub(crate) capacity: usize,
    pub(crate) cursor: usize,
}

impl ReplayBuffer {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            items: Vec::with_capacity(capacity.max(1)),
            capacity: capacity.max(1),
            cursor: 0,
        }
    }

    pub(crate) fn clear(&mut self) {
        self.items.clear();
        self.cursor = 0;
    }

    pub(crate) fn len(&self) -> usize {
        self.items.len()
    }

    pub(crate) fn push(&mut self, item: Transition) {
        if self.items.len() < self.capacity {
            self.items.push(item);
            return;
        }

        self.items[self.cursor] = item;
        self.cursor = (self.cursor + 1) % self.capacity;
    }

    pub(crate) fn sample(&self, rng: &mut Lcg) -> Option<Transition> {
        if self.items.is_empty() {
            None
        } else {
            let index = rng.range_usize(self.items.len());

            Some(self.items[index])
        }
    }
}

// ============================================================================
// DETERMINISTIC RANDOM GENERATOR
// ============================================================================
//
// xorshift64* style generator.
// No external rand crate is required. The state is Clone/Copy so the full
// training process can later be included in checkpoint.rs.
// ============================================================================

#[derive(Clone, Copy, Debug)]
pub(crate) struct Lcg {
    pub(crate) state: u64,
}

impl Lcg {
    pub(crate) fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x9E3779B97F4A7C15 } else { seed },
        }
    }

    pub(crate) fn next_u32(&mut self) -> u32 {
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;

        let value = self.state.wrapping_mul(2685821657736338717);

        (value >> 32) as u32
    }

    pub(crate) fn next_f32(&mut self) -> f32 {
        self.next_u32() as f32 / 4294967296.0
    }

    pub(crate) fn range_f32(&mut self, min: f32, max: f32) -> f32 {
        min + (max - min) * self.next_f32()
    }

    pub(crate) fn range_usize(&mut self, max: usize) -> usize {
        if max == 0 {
            0
        } else {
            (self.next_u32() as usize) % max
        }
    }
}

// ============================================================================
// TRAINING REPORT
// ============================================================================

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct TrainingReport {
    pub(crate) result: StepResult,
    pub(crate) action: usize,
}

// ============================================================================
// NEURAL NETWORK
// ============================================================================
//
// 37 -> 64 -> 64 -> 5
//
// Output:
//   Q(FORWARD)
//   Q(LEFT)
//   Q(RIGHT)
//   Q(BRAKE)
//   Q(REVERSE)
// ============================================================================

#[derive(Clone, Debug)]
pub(crate) struct Network {
    pub(crate) w1: Vec<f32>,
    pub(crate) b1: Vec<f32>,
    pub(crate) w2: Vec<f32>,
    pub(crate) b2: Vec<f32>,
    pub(crate) w3: Vec<f32>,
    pub(crate) b3: Vec<f32>,
}

impl Network {
    pub(crate) fn new(rng: &mut Lcg) -> Self {
        let mut network = Self {
            w1: vec![0.0; OBS_SIZE * HIDDEN_1],
            b1: vec![0.0; HIDDEN_1],
            w2: vec![0.0; HIDDEN_1 * HIDDEN_2],
            b2: vec![0.0; HIDDEN_2],
            w3: vec![0.0; HIDDEN_2 * ACTION_COUNT],
            b3: vec![0.0; ACTION_COUNT],
        };

        // Xavier/Glorot-style uniform initialization.
        let limit_1 = (6.0 / (OBS_SIZE + HIDDEN_1) as f32).sqrt();

        let limit_2 = (6.0 / (HIDDEN_1 + HIDDEN_2) as f32).sqrt();

        let limit_3 = (6.0 / (HIDDEN_2 + ACTION_COUNT) as f32).sqrt();

        for weight in &mut network.w1 {
            *weight = rng.range_f32(-limit_1, limit_1);
        }

        for weight in &mut network.w2 {
            *weight = rng.range_f32(-limit_2, limit_2);
        }

        for weight in &mut network.w3 {
            *weight = rng.range_f32(-limit_3, limit_3);
        }

        network
    }

    pub(crate) fn parameter_count() -> usize {
        OBS_SIZE * HIDDEN_1
            + HIDDEN_1
            + HIDDEN_1 * HIDDEN_2
            + HIDDEN_2
            + HIDDEN_2 * ACTION_COUNT
            + ACTION_COUNT
    }

    pub(crate) fn forward(
        &self,
        input: &[f32; OBS_SIZE],
    ) -> ([f32; HIDDEN_1], [f32; HIDDEN_2], [f32; ACTION_COUNT]) {
        let mut hidden_1 = [0.0; HIDDEN_1];

        let mut hidden_2 = [0.0; HIDDEN_2];

        let mut q = [0.0; ACTION_COUNT];

        for output in 0..HIDDEN_1 {
            let mut value = self.b1[output];

            for input_index in 0..OBS_SIZE {
                value += input[input_index] * self.w1[input_index * HIDDEN_1 + output];
            }

            hidden_1[output] = value.max(0.0);
        }

        for output in 0..HIDDEN_2 {
            let mut value = self.b2[output];

            for input_index in 0..HIDDEN_1 {
                value += hidden_1[input_index] * self.w2[input_index * HIDDEN_2 + output];
            }

            hidden_2[output] = value.max(0.0);
        }

        for output in 0..ACTION_COUNT {
            let mut value = self.b3[output];

            for input_index in 0..HIDDEN_2 {
                value += hidden_2[input_index] * self.w3[input_index * ACTION_COUNT + output];
            }

            q[output] = value;
        }

        (hidden_1, hidden_2, q)
    }

    pub(crate) fn q_values(&self, input: &[f32; OBS_SIZE]) -> [f32; ACTION_COUNT] {
        self.forward(input).2
    }

    pub(crate) fn train_sample(
        &mut self,
        target_network: &Network,
        transition: &Transition,
        learning_rate: f32,
        gamma: f32,
    ) -> f32 {
        let (hidden_1, hidden_2, q_values) = self.forward(&transition.state);

        // Double-DQN:
        //   online network chooses the next action
        //   target network evaluates that chosen action
        let online_next = self.q_values(&transition.next_state);

        let next_action = argmax(&online_next);

        let target_q = target_network.q_values(&transition.next_state);

        let target = transition.reward
            + if transition.done {
                0.0
            } else {
                gamma * target_q[next_action]
            };

        let action = (transition.action as usize).min(ACTION_COUNT - 1);

        let error = q_values[action] - target;

        let mut delta_3 = [0.0; ACTION_COUNT];

        // Huber derivative.
        delta_3[action] = if error.abs() <= 1.0 {
            error
        } else {
            error.signum()
        };

        let mut delta_2 = [0.0; HIDDEN_2];

        for hidden in 0..HIDDEN_2 {
            let mut value = 0.0f32;

            for output in 0..ACTION_COUNT {
                value += self.w3[hidden * ACTION_COUNT + output] * delta_3[output];
            }

            delta_2[hidden] = if hidden_2[hidden] > 0.0 { value } else { 0.0 };
        }

        let mut delta_1 = [0.0; HIDDEN_1];

        for hidden in 0..HIDDEN_1 {
            let mut value = 0.0f32;

            for next_hidden in 0..HIDDEN_2 {
                value += self.w2[hidden * HIDDEN_2 + next_hidden] * delta_2[next_hidden];
            }

            delta_1[hidden] = if hidden_1[hidden] > 0.0 { value } else { 0.0 };
        }

        let step = learning_rate.clamp(1.0e-6, 1.0);

        for hidden in 0..HIDDEN_2 {
            for output in 0..ACTION_COUNT {
                let gradient = (hidden_2[hidden] * delta_3[output]).clamp(-5.0, 5.0);

                self.w3[hidden * ACTION_COUNT + output] -= step * gradient;
            }
        }

        for output in 0..ACTION_COUNT {
            self.b3[output] -= step * delta_3[output];
        }

        for hidden in 0..HIDDEN_1 {
            for next_hidden in 0..HIDDEN_2 {
                let gradient = (hidden_1[hidden] * delta_2[next_hidden]).clamp(-5.0, 5.0);

                self.w2[hidden * HIDDEN_2 + next_hidden] -= step * gradient;
            }
        }

        for hidden in 0..HIDDEN_2 {
            self.b2[hidden] -= step * delta_2[hidden];
        }

        for input_index in 0..OBS_SIZE {
            for hidden in 0..HIDDEN_1 {
                let gradient = (transition.state[input_index] * delta_1[hidden]).clamp(-5.0, 5.0);

                self.w1[input_index * HIDDEN_1 + hidden] -= step * gradient;
            }
        }

        for hidden in 0..HIDDEN_1 {
            self.b1[hidden] -= step * delta_1[hidden];
        }

        if error.abs() <= 1.0 {
            0.5 * error * error
        } else {
            error.abs() - 0.5
        }
    }

    pub(crate) fn parameters(&self) -> impl Iterator<Item = f32> + '_ {
        self.w1
            .iter()
            .chain(self.b1.iter())
            .chain(self.w2.iter())
            .chain(self.b2.iter())
            .chain(self.w3.iter())
            .chain(self.b3.iter())
            .copied()
    }

    pub(crate) fn load_parameters(&mut self, data: &[f32]) -> bool {
        if data.len() != Self::parameter_count() {
            return false;
        }

        let mut offset = 0usize;

        let copy_block = |target: &mut [f32], offset: &mut usize, data: &[f32]| {
            let end = *offset + target.len();

            target.copy_from_slice(&data[*offset..end]);

            *offset = end;
        };

        copy_block(&mut self.w1, &mut offset, data);
        copy_block(&mut self.b1, &mut offset, data);
        copy_block(&mut self.w2, &mut offset, data);
        copy_block(&mut self.b2, &mut offset, data);
        copy_block(&mut self.w3, &mut offset, data);
        copy_block(&mut self.b3, &mut offset, data);

        true
    }
}

// ============================================================================
// AI ENGINE
// ============================================================================

#[derive(Clone, Debug)]
pub(crate) struct AiEngine {
    // pub(crate) fields are intentional: checkpoint.rs needs to serialize the
    // full training state without exposing implementation details to JS.
    pub(crate) network: Network,
    pub(crate) target_network: Network,
    pub(crate) replay: ReplayBuffer,
    pub(crate) rng: Lcg,

    pub(crate) learning_rate: f32,
    pub(crate) gamma: f32,

    pub(crate) epsilon: f32,
    pub(crate) epsilon_min: f32,
    pub(crate) epsilon_decay: f32,

    pub(crate) batch_size: usize,
    pub(crate) target_update: u64,

    pub(crate) training_steps: u64,
    pub(crate) completed_episodes: u64,
    pub(crate) success_episodes: u64,
    pub(crate) collision_episodes: u64,

    pub(crate) last_episode_reward: f32,
    pub(crate) last_episode_steps: u32,
    pub(crate) last_loss: f32,
    pub(crate) last_reward: f32,
    pub(crate) last_action: u32,
}

impl AiEngine {
    pub(crate) fn new(seed: u64) -> Self {
        let mut rng = Lcg::new(seed);

        let network = Network::new(&mut rng);

        let target_network = network.clone();

        Self {
            network,
            target_network,
            replay: ReplayBuffer::new(DEFAULT_REPLAY_CAPACITY),
            rng,
            learning_rate: DEFAULT_LEARNING_RATE,
            gamma: DEFAULT_GAMMA,
            epsilon: DEFAULT_EPSILON,
            epsilon_min: DEFAULT_EPSILON_MIN,
            epsilon_decay: DEFAULT_EPSILON_DECAY,
            batch_size: DEFAULT_BATCH_SIZE,
            target_update: DEFAULT_TARGET_UPDATE,
            training_steps: 0,
            completed_episodes: 0,
            success_episodes: 0,
            collision_episodes: 0,
            last_episode_reward: 0.0,
            last_episode_steps: 0,
            last_loss: 0.0,
            last_reward: 0.0,
            last_action: 0,
        }
    }

    pub(crate) fn parameter_count() -> usize {
        Network::parameter_count()
    }

    pub(crate) fn reset(&mut self) {
        self.network = Network::new(&mut self.rng);

        self.target_network = self.network.clone();

        self.replay.clear();

        self.epsilon = DEFAULT_EPSILON;

        self.training_steps = 0;
        self.completed_episodes = 0;
        self.success_episodes = 0;
        self.collision_episodes = 0;

        self.last_episode_reward = 0.0;
        self.last_episode_steps = 0;
        self.last_loss = 0.0;
        self.last_reward = 0.0;
        self.last_action = FORWARD as u32;
    }

    pub(crate) fn next_seed(&mut self) -> u32 {
        self.rng.next_u32().max(1)
    }

    pub(crate) fn set_learning_rate(&mut self, value: f32) {
        self.learning_rate = value.clamp(0.00001, 0.1);
    }

    pub(crate) fn set_gamma(&mut self, value: f32) {
        self.gamma = value.clamp(0.80, 0.9999);
    }

    pub(crate) fn set_batch_size(&mut self, value: u32) {
        self.batch_size = (value as usize).clamp(1, 256);
    }

    pub(crate) fn set_epsilon(&mut self, value: f32) {
        self.epsilon = value.clamp(0.0, 1.0);

        self.epsilon = self.epsilon.max(self.epsilon_min);
    }

    pub(crate) fn set_epsilon_min(&mut self, value: f32) {
        self.epsilon_min = value.clamp(0.0, 1.0);

        if self.epsilon < self.epsilon_min {
            self.epsilon = self.epsilon_min;
        }
    }

    pub(crate) fn set_epsilon_decay(&mut self, value: f32) {
        self.epsilon_decay = value.clamp(0.9, 0.999999);
    }

    pub(crate) fn set_target_update(&mut self, value: u32) {
        self.target_update = (value as u64).clamp(1, 100_000);
    }

    pub(crate) fn completed_episodes(&self) -> u64 {
        self.completed_episodes
    }

    pub(crate) fn training_steps(&self) -> u64 {
        self.training_steps
    }

    pub(crate) fn success_episodes(&self) -> u64 {
        self.success_episodes
    }

    pub(crate) fn collision_episodes(&self) -> u64 {
        self.collision_episodes
    }

    pub(crate) fn epsilon(&self) -> f32 {
        self.epsilon
    }

    pub(crate) fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    pub(crate) fn gamma(&self) -> f32 {
        self.gamma
    }

    pub(crate) fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub(crate) fn target_update(&self) -> u64 {
        self.target_update
    }

    pub(crate) fn last_loss(&self) -> f32 {
        self.last_loss
    }

    pub(crate) fn last_reward(&self) -> f32 {
        self.last_reward
    }

    pub(crate) fn last_episode_reward(&self) -> f32 {
        self.last_episode_reward
    }

    pub(crate) fn last_episode_steps(&self) -> u32 {
        self.last_episode_steps
    }

    pub(crate) fn last_action(&self) -> u32 {
        self.last_action
    }

    pub(crate) fn replay_size(&self) -> usize {
        self.replay.len()
    }

    pub(crate) fn record_manual_action(&mut self, action: usize, reward: f32) {
        self.last_action = action.min(ACTION_COUNT - 1) as u32;

        self.last_reward = finite_or_zero(reward);
    }

    pub(crate) fn record_episode_result(&mut self, environment: &Environment, result: StepResult) {
        if !result.done {
            return;
        }

        self.completed_episodes += 1;

        self.last_episode_reward = finite_or_zero(environment.episode_reward());

        self.last_episode_steps = environment.current_step();

        if result.success {
            self.success_episodes += 1;
        }

        if result.collision {
            self.collision_episodes += 1;
        }
    }

    pub(crate) fn predict_action(&self, state: &[f32; OBS_SIZE]) -> usize {
        let q = self.network.q_values(state);
        self.safety_filter(state, argmax(&q))
    }

    pub(crate) fn q_values(&self, state: &[f32; OBS_SIZE]) -> [f32; ACTION_COUNT] {
        self.network.q_values(state)
    }

    pub(crate) fn train_step(&mut self, environment: &mut Environment) -> TrainingReport {
        if environment.is_done() {
            environment.reset(0);
        }

        let state = environment.observation();

        let action = self.select_training_action(&state);

        let result = environment.step(action);

        let next_state = environment.observation();

        self.replay.push(Transition {
            state,
            action: action as u8,
            reward: finite_or_zero(result.reward),
            next_state,
            done: result.done,
        });

        self.last_reward = finite_or_zero(result.reward);

        self.last_action = action as u32;

        self.training_steps += 1;

        // Exploration decays per environment step, not per episode.
        // Collision-heavy episodes can be only a few steps long; decaying per
        // episode made epsilon collapse almost immediately and froze a bad
        // policy before it had learned anything useful.
        self.epsilon = (self.epsilon * self.epsilon_decay).max(self.epsilon_min);

        // One optimizer update per environment step once replay has warmed up.
        if self.replay.len() >= self.batch_size {
            self.last_loss = self.train_replay_update();
        }

        if self.training_steps % self.target_update == 0 {
            self.target_network = self.network.clone();
        }

        if result.done {
            self.record_episode_result(environment, result);

            environment.reset(0);
        }

        TrainingReport { result, action }
    }

    pub(crate) fn evaluate_episodes(
        &self,
        source_environment: &Environment,
        count: u32,
        base_seed: u32,
    ) -> Vec<f32> {
        let total = count.max(1).min(10_000);

        let mut evaluation = source_environment.clone();

        let mut successes = 0u32;
        let mut collisions = 0u32;
        let mut total_steps = 0u64;
        let mut total_reward = 0.0f64;

        for episode in 0..total {
            let seed = base_seed.wrapping_add(episode.wrapping_mul(7919)).max(1);

            evaluation.reset(seed);

            let mut steps = 0u32;

            loop {
                let observation = evaluation.observation();

                let action = self.predict_action(&observation);

                let result = evaluation.step(action);

                steps += 1;
                total_reward += result.reward as f64;

                if result.done || steps >= evaluation.episode_limit() {
                    if result.success {
                        successes += 1;
                    }

                    if result.collision {
                        collisions += 1;
                    }

                    break;
                }
            }

            total_steps += steps as u64;
        }

        vec![
            successes as f32 / total as f32,
            collisions as f32 / total as f32,
            total_steps as f32 / total as f32,
            (total_reward / total as f64) as f32,
        ]
    }

    pub(crate) fn metrics(&self, environment: &Environment) -> Vec<f32> {
        let divisor = self.completed_episodes.max(1) as f32;

        vec![
            self.completed_episodes as f32,
            self.training_steps as f32,
            self.last_reward,
            self.last_loss,
            self.epsilon,
            self.success_episodes as f32 / divisor,
            self.collision_episodes as f32 / divisor,
            environment.episode_reward(),
            environment.current_step() as f32,
            environment.difficulty() as f32,
            environment.obstacle_count() as f32,
            environment.seed() as f32,
            self.replay.len() as f32,
            self.learning_rate,
            self.gamma,
            self.batch_size as f32,
            self.target_update as f32,
            self.last_action as f32,
            self.last_episode_reward,
            self.last_episode_steps as f32,
        ]
    }

    // ========================================================================
    // EXPERT BOOTSTRAP POLICY
    // ========================================================================

    fn expert_action(&self, state: &[f32; OBS_SIZE]) -> usize {
        let angle = state[33].atan2(state[34]);

        let front = state[0].clamp(0.0, 1.0);
        let front_left = (state[31] + state[30] + state[29]) / 3.0;
        let front_right = (state[1] + state[2] + state[3]) / 3.0;

        let left = (state[31] + state[30] + state[29] + state[28] + state[27]) / 5.0;
        let right = (state[1] + state[2] + state[3] + state[4] + state[5]) / 5.0;

        // Very close front obstacle: never keep driving forward. Prefer the
        // side with the larger forward-facing clearance.
        if front < 0.085 {
            if front_right > front_left + 0.015 {
                return RIGHT;
            }
            if front_left > front_right + 0.015 {
                return LEFT;
            }
            return if right >= left { RIGHT } else { LEFT };
        }

        // Near obstacle: turn early. This threshold is deliberately expressed
        // in normalized sensor space (~1.2 world units) so the teacher has
        // enough look-ahead to learn smooth avoidance rather than late braking.
        if front < 0.16 {
            if angle > 0.08 && right >= left * 0.80 {
                return RIGHT;
            }
            if angle < -0.08 && left >= right * 0.80 {
                return LEFT;
            }
            return if right >= left { RIGHT } else { LEFT };
        }

        // If the goal is mostly to one side, steer toward it only when that
        // side also has useful clearance. Otherwise stay on the open side.
        if angle > 0.20 {
            return if right >= 0.16 || right >= left {
                RIGHT
            } else {
                LEFT
            };
        }

        if angle < -0.20 {
            return if left >= 0.16 || left >= right {
                LEFT
            } else {
                RIGHT
            };
        }

        // Dead-ahead target: advance.
        FORWARD
    }

    fn select_action(&mut self, state: &[f32; OBS_SIZE]) -> usize {
        let moving_actions = [FORWARD, LEFT, RIGHT, REVERSE];

        let speed = state[35].clamp(0.0, 1.0);

        let raw_action = if self.rng.next_f32() < self.epsilon {
            if speed < 0.15 {
                moving_actions[self.rng.range_usize(moving_actions.len())]
            } else {
                self.rng.range_usize(ACTION_COUNT)
            }
        } else {
            let q = self.network.q_values(state);
            let mut action = argmax(&q);

            // Do not let a stationary agent learn to sit forever on BRAKE.
            if action == BRAKE && speed < 0.15 {
                action = moving_actions[0];
                let mut best = q[action];

                for candidate in moving_actions.iter().copied().skip(1) {
                    if q[candidate] > best {
                        best = q[candidate];
                        action = candidate;
                    }
                }
            }

            action
        };

        self.safety_filter(state, raw_action)
    }

    fn select_training_action(&mut self, state: &[f32; OBS_SIZE]) -> usize {
        // Keep the geometry teacher dominant until the learned policy proves
        // it can navigate. This prevents the early DQN from taking over with
        // a badly learned greedy policy and then filling replay with failures.
        let episodes = self.completed_episodes.max(1) as f32;
        let success_rate = self.success_episodes as f32 / episodes;

        let teacher_probability = if self.training_steps < 50_000 {
            0.95
        } else if success_rate < 0.20 {
            0.90
        } else if success_rate < 0.45 {
            0.75
        } else if success_rate < 0.70 {
            0.55
        } else {
            0.35
        };

        let action = if self.rng.next_f32() < teacher_probability {
            self.expert_action(state)
        } else {
            self.select_action(state)
        };

        self.safety_filter(state, action)
    }

    /// Collision shield used during both TRAIN and RUN.
    ///
    /// The DQN remains responsible for the final policy, but it is never
    /// allowed to deliberately drive straight into an obstacle when the ray
    /// sensors already say that impact is imminent. This dramatically reduces
    /// the useless repeated-collision trajectories that used to dominate the
    /// replay buffer.
    fn safety_filter(&self, state: &[f32; OBS_SIZE], candidate: usize) -> usize {
        let front = state[0].clamp(0.0, 1.0);

        // Approximate the clearance along the direction an action will point
        // toward. The 32 sensors are spaced at 11.25 degrees, close to the
        // current turn rate, so adjacent rays make a useful one-step lookahead.
        let left_turn = (state[31] + state[30] + state[29]) / 3.0;
        let right_turn = (state[1] + state[2] + state[3]) / 3.0;
        let reverse = state[16].clamp(0.0, 1.0);

        let target_angle = state[33].atan2(state[34]);

        // Hard shield: forward is not permitted when the next movement can
        // plausibly enter the obstacle envelope.
        if front < 0.085 {
            if right_turn > left_turn + 0.01 && right_turn >= 0.06 {
                return RIGHT;
            }
            if left_turn > right_turn + 0.01 && left_turn >= 0.06 {
                return LEFT;
            }
            if right_turn >= 0.055 || left_turn >= 0.055 {
                return if right_turn >= left_turn { RIGHT } else { LEFT };
            }
            return REVERSE;
        }

        // Prevent selecting a side that is already visibly blocked.
        if candidate == RIGHT && right_turn < 0.055 {
            return if left_turn >= 0.055 { LEFT } else { REVERSE };
        }

        if candidate == LEFT && left_turn < 0.055 {
            return if right_turn >= 0.055 { RIGHT } else { REVERSE };
        }

        // Forward becomes unsafe well before impact. Prefer the side that both
        // has clearance and agrees with the target direction.
        if front < 0.16 && candidate == FORWARD {
            if target_angle > 0.10 && right_turn >= left_turn * 0.80 {
                return RIGHT;
            }

            if target_angle < -0.10 && left_turn >= right_turn * 0.80 {
                return LEFT;
            }

            return if right_turn >= left_turn { RIGHT } else { LEFT };
        }

        // A candidate side action should be rejected if the ray in that
        // direction has markedly less clearance than the opposite side.
        if candidate == RIGHT && right_turn + 0.025 < left_turn && right_turn < 0.12 {
            return LEFT;
        }

        if candidate == LEFT && left_turn + 0.025 < right_turn && left_turn < 0.12 {
            return RIGHT;
        }

        // Reverse is a last-resort escape action only when the rear is actually
        // open. Otherwise continue with the safest turn.
        if candidate == REVERSE && reverse < 0.06 {
            return if right_turn >= left_turn { RIGHT } else { LEFT };
        }

        candidate.min(ACTION_COUNT - 1)
    }

    fn train_replay_update(&mut self) -> f32 {
        if self.replay.len() < self.batch_size {
            return self.last_loss;
        }

        // The previous implementation only trained on ONE sampled transition
        // despite exposing BATCH-SIZE=64. That made learning extremely noisy.
        // Perform a compact multi-sample update each environment step.
        let updates = (self.batch_size / 8).clamp(4, 16);
        let mut total_loss = 0.0f32;
        let mut completed = 0usize;

        for _ in 0..updates {
            if let Some(transition) = self.replay.sample(&mut self.rng) {
                total_loss += self.network.train_sample(
                    &self.target_network,
                    &transition,
                    self.learning_rate,
                    self.gamma,
                );
                completed += 1;
            }
        }

        if completed == 0 {
            self.last_loss
        } else {
            total_loss / completed as f32
        }
    }
}

// ============================================================================
// HELPERS
// ============================================================================

pub(crate) fn argmax(values: &[f32]) -> usize {
    if values.is_empty() {
        return 0;
    }

    let mut best = 0usize;

    for index in 1..values.len() {
        if values[index].is_finite() && (!values[best].is_finite() || values[index] > values[best])
        {
            best = index;
        }
    }

    best
}

fn finite_or_zero(value: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn network_parameter_count_is_correct() {
        assert_eq!(Network::parameter_count(), 6917,);
    }

    #[test]
    fn network_forward_has_five_actions() {
        let mut rng = Lcg::new(12345);

        let network = Network::new(&mut rng);

        let input = [0.0f32; OBS_SIZE];

        let q = network.q_values(&input);

        assert_eq!(q.len(), ACTION_COUNT,);

        assert!(q.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn rng_is_deterministic() {
        let mut a = Lcg::new(1234);
        let mut b = Lcg::new(1234);

        for _ in 0..32 {
            assert_eq!(a.next_u32(), b.next_u32(),);
        }
    }

    #[test]
    fn replay_capacity_is_bounded() {
        let mut replay = ReplayBuffer::new(3);

        let transition = Transition {
            state: [0.0; OBS_SIZE],
            action: FORWARD as u8,
            reward: 0.0,
            next_state: [0.0; OBS_SIZE],
            done: false,
        };

        for _ in 0..10 {
            replay.push(transition);
        }

        assert_eq!(replay.len(), 3,);
    }

    #[test]
    fn ai_reset_clears_training_state() {
        let mut ai = AiEngine::new(42);

        ai.training_steps = 99;
        ai.completed_episodes = 12;
        ai.success_episodes = 4;
        ai.collision_episodes = 5;
        ai.last_loss = 1.2;
        ai.last_reward = -0.4;
        ai.epsilon = 0.12;

        ai.reset();

        assert_eq!(ai.training_steps, 0,);
        assert_eq!(ai.completed_episodes, 0,);
        assert_eq!(ai.success_episodes, 0,);
        assert_eq!(ai.collision_episodes, 0,);
        assert_eq!(ai.last_loss, 0.0,);
        assert_eq!(ai.last_reward, 0.0,);
        assert!((ai.epsilon - DEFAULT_EPSILON).abs() < 1.0e-6);
        assert_eq!(ai.replay.len(), 0,);
    }
}
