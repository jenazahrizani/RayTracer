use crate::ai::{AiEngine, Lcg, Network, ReplayBuffer, Transition};
use crate::environment::Environment;
use crate::{Aabb, Agent, Obstacle, Vec3, ACTION_COUNT, OBS_SIZE, SENSOR_COUNT};

// ============================================================================
// RAYTRC.AI NAVIGATION CHECKPOINT
// ============================================================================
//
// V2 stores the COMPLETE deterministic training state:
//   - online + target networks
//   - replay buffer
//   - AI RNG
//   - optimizer/training configuration
//   - episode/training metrics
//   - environment RNG + generated map
//   - agent/start/target state
//   - sensors + path history
//
// The format deliberately uses plain ASCII text and hexadecimal IEEE-754
// bit patterns. No serde, JSON parser, compression crate, or browser API is
// required, so the WASM build remains self-contained and GitHub-Pages friendly.
//
// Top-level format:
//   RAYTRC_NAV_CHECKPOINT_V2
//   ENGINE <version>
//   SHAPES <obs> <hidden1> <hidden2> <actions> <sensors> <params>
//   AI <lr> <gamma> <epsilon> <epsilon_min> <epsilon_decay> <batch> <target_update>
//   AIMET <training_steps> <completed> <success> <collision> <last_episode_steps> <last_action> <last_episode_reward> <last_loss> <last_reward>
//   AIRNG <u64>
//   ENV <seed> <difficulty> <obstacle_count_config> <max_steps> <step> <last_distance> <episode_reward> <done> <success> <collision>
//   ENVRNG <u64>
//   AGENT <x> <y> <z> <heading> <speed>
//   START <x> <y> <z>
//   TARGET <x> <y> <z>
//   SENSORS <32 hex-f32 values>
//   OBSTACLES <count>
//   O <12 hex-f32 values + material>
//   PATH <count>
//   P <x> <y> <z> (hex-f32)
//   NETWORK <count>
//   N <hex-f32>...
//   TARGET_NETWORK <count>
//   T <hex-f32>...
//   REPLAY <count> <capacity> <cursor>
//   R <action> <reward> <done> <74 hex-f32 values>
//   END <fnv1a64>
//
// The import path is transactional: the current engine is only mutated after
// every record has been parsed and validated successfully.
// ============================================================================

const MAGIC_V2: &str = "RAYTRC_NAV_CHECKPOINT_V2";
const MAGIC_V1: &str = "RAYTRC_NAV_MODEL_V1";

const MAX_CHECKPOINT_REPLAY: usize = 100_000;
const MAX_CHECKPOINT_OBSTACLES: usize = 64;
const MAX_CHECKPOINT_PATH: usize = 16_384;
const MAX_CHECKPOINT_TEXT_BYTES: usize = 64 * 1024 * 1024;

// The current network shape is intentionally explicit in the checkpoint so a
// checkpoint from a future/incompatible model cannot be loaded silently.
const HIDDEN_1_CHECK: usize = crate::HIDDEN_1;
const HIDDEN_2_CHECK: usize = crate::HIDDEN_2;
const PARAMETER_COUNT_CHECK: usize = 6917;

#[derive(Clone, Debug)]
struct CheckpointState {
    // AI configuration.
    learning_rate: f32,
    gamma: f32,
    epsilon: f32,
    epsilon_min: f32,
    epsilon_decay: f32,
    batch_size: usize,
    target_update: u64,

    // AI metrics / state.
    training_steps: u64,
    completed_episodes: u64,
    success_episodes: u64,
    collision_episodes: u64,
    last_episode_reward: f32,
    last_episode_steps: u32,
    last_loss: f32,
    last_reward: f32,
    last_action: u32,
    ai_rng_state: u64,

    // Networks.
    network: Vec<f32>,
    target_network: Vec<f32>,

    // Replay.
    replay_capacity: usize,
    replay_cursor: usize,
    replay: Vec<Transition>,

    // Environment configuration/state.
    env_seed: u32,
    env_rng_state: u64,
    difficulty: u32,
    obstacle_count: u32,
    max_steps: u32,
    step: u32,
    last_distance: f32,
    episode_reward: f32,
    done: bool,
    success: bool,
    collision: bool,
    agent: Agent,
    start: Vec3,
    target: Vec3,
    sensors: [f32; SENSOR_COUNT],
    obstacles: Vec<Obstacle>,
    path: Vec<Vec3>,
}

// ============================================================================
// POLICY-ONLY SNAPSHOT API
// ============================================================================
//
// Unlike V2 checkpoints, these snapshots contain ONLY the learned policy and
// trainer counters. They deliberately do NOT contain the environment, target,
// obstacles, agent position, replay buffer, or environment RNG. This is used
// by the background worker to update the foreground RUN policy without
// replacing the map that the user is currently watching.
// ============================================================================

pub(crate) fn export_policy(engine: &crate::RayTracerEngine) -> String {
    let mut output = String::with_capacity(256 + Network::parameter_count() * 8);

    output.push_str(MAGIC_V1);
    output.push('|');
    output.push_str(&engine.ai.epsilon.to_string());
    output.push('|');
    output.push_str(&engine.ai.learning_rate.to_string());
    output.push('|');
    output.push_str(&engine.ai.gamma.to_string());
    output.push('|');
    output.push_str(&engine.ai.training_steps.to_string());
    output.push('|');
    output.push_str(&engine.ai.completed_episodes.to_string());
    output.push('|');
    output.push_str(&engine.environment.difficulty().to_string());
    output.push('|');
    output.push_str(&engine.environment.configured_obstacle_count().to_string());
    output.push('|');

    for value in engine.ai.network.parameters() {
        output.push_str(&hex_f32(value));
    }

    output
}

pub(crate) fn import_policy(engine: &mut crate::RayTracerEngine, data: &str) -> bool {
    let mut parts = data.trim().split('|');

    if parts.next() != Some(MAGIC_V1) {
        return false;
    }

    let epsilon = match parts.next().and_then(parse_decimal_f32) {
        Some(value) if finite(value) => value,
        _ => return false,
    };

    let learning_rate = match parts.next().and_then(parse_decimal_f32) {
        Some(value) if finite(value) => value,
        _ => return false,
    };

    let gamma = match parts.next().and_then(parse_decimal_f32) {
        Some(value) if finite(value) => value,
        _ => return false,
    };

    let training_steps = match parts.next().and_then(parse_decimal_u64) {
        Some(value) => value,
        None => return false,
    };

    let completed_episodes = match parts.next().and_then(parse_decimal_u64) {
        Some(value) => value,
        None => return false,
    };

    // These two values are metadata only. The foreground environment is not
    // changed by a policy import.
    let _difficulty = match parts.next().and_then(parse_decimal_u32) {
        Some(value) if (1..=10).contains(&value) => value,
        _ => return false,
    };

    let _obstacle_count = match parts.next().and_then(parse_decimal_u32) {
        Some(value) if (1..=64).contains(&value) => value,
        _ => return false,
    };

    let parameter_hex = match parts.next() {
        Some(value) => value.trim(),
        None => return false,
    };

    if parts.next().is_some() || parameter_hex.len() != Network::parameter_count() * 8 {
        return false;
    }

    let mut parameters = Vec::with_capacity(Network::parameter_count());

    for index in 0..Network::parameter_count() {
        let start = index * 8;
        let end = start + 8;
        let Some(value) = parse_hex_f32(&parameter_hex[start..end]) else {
            return false;
        };
        if !finite(value) {
            return false;
        }
        parameters.push(value);
    }

    let mut network = engine.ai.network.clone();
    if !network.load_parameters(&parameters) {
        return false;
    }

    engine.ai.network = network.clone();
    engine.ai.target_network = network;
    engine.ai.epsilon = epsilon.clamp(0.0, 1.0);
    engine.ai.learning_rate = learning_rate.max(0.000001);
    engine.ai.gamma = gamma.clamp(0.80, 0.99999);
    engine.ai.training_steps = training_steps;
    engine.ai.completed_episodes = completed_episodes;

    true
}

// ============================================================================
// PUBLIC API
// ============================================================================

pub(crate) fn export(engine: &crate::RayTracerEngine) -> String {
    // Exact preallocation is intentionally approximate. It prevents repeated
    // reallocations for ordinary checkpoints without trying to predict replay
    // contents perfectly.
    let estimated = 32_768usize.saturating_add(engine.ai.replay.items.len().saturating_mul(700));

    let mut output = String::with_capacity(estimated);

    push_line(&mut output, MAGIC_V2);
    push_line(&mut output, &format!("ENGINE {}", crate::ENGINE_VERSION));
    push_line(
        &mut output,
        &format!(
            "SHAPES {} {} {} {} {} {}",
            OBS_SIZE,
            HIDDEN_1_CHECK,
            HIDDEN_2_CHECK,
            ACTION_COUNT,
            SENSOR_COUNT,
            Network::parameter_count(),
        ),
    );

    push_line(
        &mut output,
        &format!(
            "AI {} {} {} {} {} {} {}",
            hex_f32(engine.ai.learning_rate),
            hex_f32(engine.ai.gamma),
            hex_f32(engine.ai.epsilon),
            hex_f32(engine.ai.epsilon_min),
            hex_f32(engine.ai.epsilon_decay),
            engine.ai.batch_size,
            engine.ai.target_update,
        ),
    );

    push_line(
        &mut output,
        &format!(
            "AIMET {} {} {} {} {} {} {} {} {}",
            engine.ai.training_steps,
            engine.ai.completed_episodes,
            engine.ai.success_episodes,
            engine.ai.collision_episodes,
            engine.ai.last_episode_steps,
            engine.ai.last_action,
            hex_f32(engine.ai.last_episode_reward),
            hex_f32(engine.ai.last_loss),
            hex_f32(engine.ai.last_reward),
        ),
    );

    push_line(&mut output, &format!("AIRNG {}", engine.ai.rng.state));

    push_line(
        &mut output,
        &format!(
            "ENV {} {} {} {} {} {} {} {} {} {}",
            engine.environment.seed,
            engine.environment.difficulty,
            engine.environment.obstacle_count,
            engine.environment.max_steps,
            engine.environment.step,
            hex_f32(engine.environment.last_distance),
            hex_f32(engine.environment.episode_reward),
            bool_number(engine.environment.done),
            bool_number(engine.environment.success),
            bool_number(engine.environment.collision),
        ),
    );

    push_line(
        &mut output,
        &format!("ENVRNG {}", engine.environment.rng.state),
    );

    push_line(
        &mut output,
        &format!(
            "AGENT {} {} {} {} {}",
            hex_f32(engine.environment.agent.position.x),
            hex_f32(engine.environment.agent.position.y),
            hex_f32(engine.environment.agent.position.z),
            hex_f32(engine.environment.agent.heading),
            hex_f32(engine.environment.agent.speed),
        ),
    );

    push_line(
        &mut output,
        &format!(
            "START {} {} {}",
            hex_f32(engine.environment.start.x),
            hex_f32(engine.environment.start.y),
            hex_f32(engine.environment.start.z),
        ),
    );

    push_line(
        &mut output,
        &format!(
            "TARGET {} {} {}",
            hex_f32(engine.environment.target.x),
            hex_f32(engine.environment.target.y),
            hex_f32(engine.environment.target.z),
        ),
    );

    output.push_str("SENSORS");
    for value in &engine.environment.sensors {
        output.push(' ');
        output.push_str(&hex_f32(*value));
    }
    output.push('\n');

    push_line(
        &mut output,
        &format!("OBSTACLES {}", engine.environment.obstacles.len(),),
    );

    for obstacle in &engine.environment.obstacles {
        push_line(
            &mut output,
            &format!(
                "O {} {} {} {} {} {} {} {} {} {} {} {} {}",
                hex_f32(obstacle.bounds.center.x),
                hex_f32(obstacle.bounds.center.y),
                hex_f32(obstacle.bounds.center.z),
                hex_f32(obstacle.bounds.half.x),
                hex_f32(obstacle.bounds.half.y),
                hex_f32(obstacle.bounds.half.z),
                obstacle.material,
                // Reserved duplicate shape words keep the record easy to
                // inspect and allow future material metadata additions without
                // changing the core obstacle geometry fields.
                0,
                0,
                0,
                0,
                0,
                0,
            ),
        );
    }

    push_line(
        &mut output,
        &format!("PATH {}", engine.environment.path.len()),
    );

    for point in &engine.environment.path {
        push_line(
            &mut output,
            &format!(
                "P {} {} {}",
                hex_f32(point.x),
                hex_f32(point.y),
                hex_f32(point.z),
            ),
        );
    }

    let network_parameters: Vec<f32> = engine.ai.network.parameters().collect();
    push_line(
        &mut output,
        &format!("NETWORK {}", network_parameters.len()),
    );
    push_network_records(&mut output, 'N', &network_parameters);

    let target_parameters: Vec<f32> = engine.ai.target_network.parameters().collect();
    push_line(
        &mut output,
        &format!("TARGET_NETWORK {}", target_parameters.len()),
    );
    push_network_records(&mut output, 'T', &target_parameters);

    push_line(
        &mut output,
        &format!(
            "REPLAY {} {} {}",
            engine.ai.replay.items.len(),
            engine.ai.replay.capacity,
            engine.ai.replay.cursor,
        ),
    );

    for transition in &engine.ai.replay.items {
        output.push_str("R ");
        output.push_str(&transition.action.to_string());
        output.push(' ');
        output.push_str(&hex_f32(transition.reward));
        output.push(' ');
        output.push_str(&bool_number(transition.done));

        for value in &transition.state {
            output.push(' ');
            output.push_str(&hex_f32(*value));
        }

        for value in &transition.next_state {
            output.push(' ');
            output.push_str(&hex_f32(*value));
        }

        output.push('\n');
    }

    // Compute the checksum over every record before END. This catches most
    // accidental truncation/copy corruption and gives import a cheap integrity
    // gate before touching engine state.
    let checksum = fnv1a64(output.as_bytes());
    push_line(&mut output, &format!("END {:016X}", checksum));

    output
}

pub(crate) fn import(engine: &mut crate::RayTracerEngine, data: &str) -> bool {
    if data.len() > MAX_CHECKPOINT_TEXT_BYTES {
        return false;
    }

    if data.starts_with(MAGIC_V1) {
        return import_v1(engine, data);
    }

    if !data.starts_with(MAGIC_V2) {
        return false;
    }

    let Some(checkpoint) = parse_v2(data) else {
        return false;
    };

    apply_checkpoint(engine, checkpoint)
}

// ============================================================================
// V2 PARSER
// ============================================================================

fn parse_v2(data: &str) -> Option<CheckpointState> {
    let mut lines = data.lines();

    expect_exact(&mut lines, MAGIC_V2)?;

    // ENGINE
    {
        let parts = next_parts(&mut lines, "ENGINE", 2)?;
        if parts[1] != crate::ENGINE_VERSION {
            return None;
        }
    }

    // SHAPES
    {
        let parts = next_parts(&mut lines, "SHAPES", 7)?;
        if parse_usize(parts[1])? != OBS_SIZE
            || parse_usize(parts[2])? != HIDDEN_1_CHECK
            || parse_usize(parts[3])? != HIDDEN_2_CHECK
            || parse_usize(parts[4])? != ACTION_COUNT
            || parse_usize(parts[5])? != SENSOR_COUNT
            || parse_usize(parts[6])? != PARAMETER_COUNT_CHECK
            || parse_usize(parts[6])? != Network::parameter_count()
        {
            return None;
        }
    }

    // AI
    let ai_parts = next_parts(&mut lines, "AI", 8)?;
    let learning_rate = parse_hex_f32(ai_parts[1])?;
    let gamma = parse_hex_f32(ai_parts[2])?;
    let epsilon = parse_hex_f32(ai_parts[3])?;
    let epsilon_min = parse_hex_f32(ai_parts[4])?;
    let epsilon_decay = parse_hex_f32(ai_parts[5])?;
    let batch_size = parse_usize(ai_parts[6])?;
    let target_update = parse_u64(ai_parts[7])?;

    validate_ai_config(
        learning_rate,
        gamma,
        epsilon,
        epsilon_min,
        epsilon_decay,
        batch_size,
        target_update,
    )?;

    // AIMET
    let metric_parts = next_parts(&mut lines, "AIMET", 10)?;
    let training_steps = parse_u64(metric_parts[1])?;
    let completed_episodes = parse_u64(metric_parts[2])?;
    let success_episodes = parse_u64(metric_parts[3])?;
    let collision_episodes = parse_u64(metric_parts[4])?;
    let last_episode_steps = parse_u32(metric_parts[5])?;
    let last_action = parse_u32(metric_parts[6])?;
    let last_episode_reward = parse_hex_f32(metric_parts[7])?;
    let last_loss = parse_hex_f32(metric_parts[8])?;
    let last_reward = parse_hex_f32(metric_parts[9])?;

    if !finite(last_episode_reward)
        || !finite(last_loss)
        || !finite(last_reward)
        || last_action as usize >= ACTION_COUNT
    {
        return None;
    }

    // AIRNG
    let ai_rng_parts = next_parts(&mut lines, "AIRNG", 2)?;
    let ai_rng_state = parse_u64(ai_rng_parts[1])?;
    if ai_rng_state == 0 {
        return None;
    }

    // ENV
    let env_parts = next_parts(&mut lines, "ENV", 11)?;
    let env_seed = parse_u32(env_parts[1])?;
    let difficulty = parse_u32(env_parts[2])?;
    let obstacle_count = parse_u32(env_parts[3])?;
    let max_steps = parse_u32(env_parts[4])?;
    let step = parse_u32(env_parts[5])?;
    let last_distance = parse_hex_f32(env_parts[6])?;
    let episode_reward = parse_hex_f32(env_parts[7])?;
    let done = parse_bool_number(env_parts[8])?;
    let success = parse_bool_number(env_parts[9])?;
    let collision = parse_bool_number(env_parts[10])?;

    if env_seed == 0
        || !(1..=10).contains(&difficulty)
        || !(1..=64).contains(&obstacle_count)
        || !(1..=5000).contains(&max_steps)
        || step > max_steps
        || !finite(last_distance)
        || !finite(episode_reward)
    {
        return None;
    }

    if success && !done {
        return None;
    }
    if collision && !done {
        return None;
    }
    if success && collision {
        return None;
    }

    // ENVRNG
    let env_rng_parts = next_parts(&mut lines, "ENVRNG", 2)?;
    let env_rng_state = parse_u64(env_rng_parts[1])?;
    if env_rng_state == 0 {
        return None;
    }

    // AGENT
    let agent_parts = next_parts(&mut lines, "AGENT", 6)?;
    let agent = Agent {
        position: Vec3::new(
            parse_hex_f32(agent_parts[1])?,
            parse_hex_f32(agent_parts[2])?,
            parse_hex_f32(agent_parts[3])?,
        ),
        heading: parse_hex_f32(agent_parts[4])?,
        speed: parse_hex_f32(agent_parts[5])?,
    };

    // START
    let start_parts = next_parts(&mut lines, "START", 4)?;
    let start = Vec3::new(
        parse_hex_f32(start_parts[1])?,
        parse_hex_f32(start_parts[2])?,
        parse_hex_f32(start_parts[3])?,
    );

    // TARGET
    let target_parts = next_parts(&mut lines, "TARGET", 4)?;
    let target = Vec3::new(
        parse_hex_f32(target_parts[1])?,
        parse_hex_f32(target_parts[2])?,
        parse_hex_f32(target_parts[3])?,
    );

    validate_vec3(agent.position)?;
    validate_f32(agent.heading)?;
    validate_f32(agent.speed)?;
    validate_vec3(start)?;
    validate_vec3(target)?;

    // SENSORS
    let sensor_parts = next_parts_at_least(&mut lines, "SENSORS", SENSOR_COUNT + 1)?;
    let mut sensors = [0.0f32; SENSOR_COUNT];
    for index in 0..SENSOR_COUNT {
        sensors[index] = parse_hex_f32(sensor_parts[index + 1])?;
        if !finite(sensors[index]) {
            return None;
        }
    }

    // OBSTACLES
    let obstacle_count_parts = next_parts(&mut lines, "OBSTACLES", 2)?;
    let serialized_obstacles = parse_usize(obstacle_count_parts[1])?;
    if serialized_obstacles > MAX_CHECKPOINT_OBSTACLES
        || serialized_obstacles != obstacle_count as usize
    {
        return None;
    }

    let mut obstacles = Vec::with_capacity(serialized_obstacles);
    for _ in 0..serialized_obstacles {
        let parts = next_parts(&mut lines, "O", 14)?;
        let obstacle = Obstacle {
            bounds: Aabb {
                center: Vec3::new(
                    parse_hex_f32(parts[1])?,
                    parse_hex_f32(parts[2])?,
                    parse_hex_f32(parts[3])?,
                ),
                half: Vec3::new(
                    parse_hex_f32(parts[4])?,
                    parse_hex_f32(parts[5])?,
                    parse_hex_f32(parts[6])?,
                ),
            },
            material: parse_u32(parts[7])?,
        };

        // Reserved words must currently be zero. This prevents accepting a
        // partially corrupted or future record with a subtly shifted schema.
        for reserved in &parts[8..14] {
            if parse_u32(reserved)? != 0 {
                return None;
            }
        }

        validate_vec3(obstacle.bounds.center)?;
        validate_vec3(obstacle.bounds.half)?;
        if obstacle.bounds.half.x <= 0.0
            || obstacle.bounds.half.y <= 0.0
            || obstacle.bounds.half.z <= 0.0
        {
            return None;
        }

        obstacles.push(obstacle);
    }

    // PATH
    let path_count_parts = next_parts(&mut lines, "PATH", 2)?;
    let path_count = parse_usize(path_count_parts[1])?;
    if path_count > MAX_CHECKPOINT_PATH {
        return None;
    }

    let mut path = Vec::with_capacity(path_count);
    for _ in 0..path_count {
        let parts = next_parts(&mut lines, "P", 4)?;
        let point = Vec3::new(
            parse_hex_f32(parts[1])?,
            parse_hex_f32(parts[2])?,
            parse_hex_f32(parts[3])?,
        );
        validate_vec3(point)?;
        path.push(point);
    }

    // NETWORK
    let network_count_parts = next_parts(&mut lines, "NETWORK", 2)?;
    let network_count = parse_usize(network_count_parts[1])?;
    if network_count != Network::parameter_count() {
        return None;
    }
    let network = parse_network_records(&mut lines, 'N', network_count)?;

    // TARGET NETWORK
    let target_count_parts = next_parts(&mut lines, "TARGET_NETWORK", 2)?;
    let target_count = parse_usize(target_count_parts[1])?;
    if target_count != Network::parameter_count() {
        return None;
    }
    let target_network = parse_network_records(&mut lines, 'T', target_count)?;

    // REPLAY
    let replay_parts = next_parts(&mut lines, "REPLAY", 4)?;
    let replay_count = parse_usize(replay_parts[1])?;
    let replay_capacity = parse_usize(replay_parts[2])?;
    let replay_cursor = parse_usize(replay_parts[3])?;

    if replay_count > MAX_CHECKPOINT_REPLAY
        || replay_capacity == 0
        || replay_capacity > MAX_CHECKPOINT_REPLAY
        || replay_count > replay_capacity
        || (replay_count < replay_capacity && replay_cursor != 0)
        || (replay_count == replay_capacity && replay_cursor >= replay_capacity)
    {
        return None;
    }

    let mut replay = Vec::with_capacity(replay_count);
    for _ in 0..replay_count {
        let parts = next_parts_at_least(&mut lines, "R", 4 + OBS_SIZE * 2)?;

        let action = parse_u32(parts[1])?;
        if action as usize >= ACTION_COUNT {
            return None;
        }

        let reward = parse_hex_f32(parts[2])?;
        if !finite(reward) {
            return None;
        }

        let transition_done = parse_bool_number(parts[3])?;
        let mut state = [0.0f32; OBS_SIZE];
        let mut next_state = [0.0f32; OBS_SIZE];

        let state_start = 4usize;
        let next_state_start = 4usize + OBS_SIZE;

        for index in 0..OBS_SIZE {
            state[index] = parse_hex_f32(parts[state_start + index])?;
            if !finite(state[index]) {
                return None;
            }
        }

        for index in 0..OBS_SIZE {
            next_state[index] = parse_hex_f32(parts[next_state_start + index])?;
            if !finite(next_state[index]) {
                return None;
            }
        }

        replay.push(Transition {
            state,
            action: action as u8,
            reward,
            next_state,
            done: transition_done,
        });
    }

    // There must be exactly one END record and nothing after it.
    let end_line = lines.next()?;
    let end_parts: Vec<&str> = end_line.split_whitespace().collect();
    if end_parts.len() != 2 || end_parts[0] != "END" {
        return None;
    }
    let expected_checksum = parse_hex_u64(end_parts[1])?;

    if lines.next().is_some() {
        return None;
    }

    // Checksum excludes the END line itself, exactly matching export().
    let end_offset = data.rfind("\nEND ")?;
    let checksum_source = &data[..end_offset + 1];
    let actual_checksum = fnv1a64(checksum_source.as_bytes());
    if actual_checksum != expected_checksum {
        return None;
    }

    Some(CheckpointState {
        learning_rate,
        gamma,
        epsilon,
        epsilon_min,
        epsilon_decay,
        batch_size,
        target_update,
        training_steps,
        completed_episodes,
        success_episodes,
        collision_episodes,
        last_episode_reward,
        last_episode_steps,
        last_loss,
        last_reward,
        last_action,
        ai_rng_state,
        network,
        target_network,
        replay_capacity,
        replay_cursor,
        replay,
        env_seed,
        env_rng_state,
        difficulty,
        obstacle_count,
        max_steps,
        step,
        last_distance,
        episode_reward,
        done,
        success,
        collision,
        agent,
        start,
        target,
        sensors,
        obstacles,
        path,
    })
}

// ============================================================================
// APPLY / TRANSACTION
// ============================================================================

fn apply_checkpoint(engine: &mut crate::RayTracerEngine, checkpoint: CheckpointState) -> bool {
    // A second validation pass is inexpensive and makes this function safe if
    // it is ever reused by a non-parser caller in the future.
    if checkpoint.network.len() != Network::parameter_count()
        || checkpoint.target_network.len() != Network::parameter_count()
        || checkpoint.replay.len() > checkpoint.replay_capacity
        || checkpoint.replay_capacity > MAX_CHECKPOINT_REPLAY
        || checkpoint.replay_cursor >= checkpoint.replay_capacity
        || checkpoint.last_action as usize >= ACTION_COUNT
    {
        return false;
    }

    // The network loaders perform exact-length checks. We create temporary
    // networks by cloning the current shape, then only swap after all loads
    // succeed.
    let mut network = engine.ai.network.clone();
    let mut target_network = engine.ai.target_network.clone();

    if !network.load_parameters(&checkpoint.network)
        || !target_network.load_parameters(&checkpoint.target_network)
    {
        return false;
    }

    let mut replay = ReplayBuffer::new(checkpoint.replay_capacity);
    replay.items = checkpoint.replay;
    replay.cursor = checkpoint.replay_cursor;

    // Keep the exact serialized environment state. Do not call reset(), because
    // reset() would regenerate the map and consume RNG state.
    let environment = Environment {
        seed: checkpoint.env_seed,
        rng: Lcg::new(checkpoint.env_rng_state),
        difficulty: checkpoint.difficulty,
        obstacle_count: checkpoint.obstacle_count,
        max_steps: checkpoint.max_steps,
        step: checkpoint.step,
        obstacles: checkpoint.obstacles,
        agent: checkpoint.agent,
        start: checkpoint.start,
        target: checkpoint.target,
        last_distance: checkpoint.last_distance,
        episode_reward: checkpoint.episode_reward,
        done: checkpoint.done,
        success: checkpoint.success,
        collision: checkpoint.collision,
        sensors: checkpoint.sensors,
        path: checkpoint.path,
    };

    // AI temporary state is not constructed through AiEngine::new(), because
    // doing so would consume the RNG and initialize a new model. Assigning the
    // complete fields preserves deterministic continuation.
    engine.ai.network = network;
    engine.ai.target_network = target_network;
    engine.ai.replay = replay;
    engine.ai.rng = Lcg::new(checkpoint.ai_rng_state);

    engine.ai.learning_rate = checkpoint.learning_rate;
    engine.ai.gamma = checkpoint.gamma;
    engine.ai.epsilon = checkpoint.epsilon;
    engine.ai.epsilon_min = checkpoint.epsilon_min;
    engine.ai.epsilon_decay = checkpoint.epsilon_decay;
    engine.ai.batch_size = checkpoint.batch_size;
    engine.ai.target_update = checkpoint.target_update;

    engine.ai.training_steps = checkpoint.training_steps;
    engine.ai.completed_episodes = checkpoint.completed_episodes;
    engine.ai.success_episodes = checkpoint.success_episodes;
    engine.ai.collision_episodes = checkpoint.collision_episodes;

    engine.ai.last_episode_reward = checkpoint.last_episode_reward;
    engine.ai.last_episode_steps = checkpoint.last_episode_steps;
    engine.ai.last_loss = checkpoint.last_loss;
    engine.ai.last_reward = checkpoint.last_reward;
    engine.ai.last_action = checkpoint.last_action;

    engine.environment = environment;

    true
}

// ============================================================================
// LEGACY V1 IMPORT
// ============================================================================
//
// Previous builds stored only the online model and a handful of scalars. That
// format cannot restore a replay buffer or target network, but importing it is
// still useful: the target network is cloned from the imported online model,
// replay is cleared, and the environment is reset using the stored seedless
// configuration. This is deliberately one-way migration into V2.

fn import_v1(engine: &mut crate::RayTracerEngine, data: &str) -> bool {
    let mut parts = data.trim().split('|');

    if parts.next() != Some(MAGIC_V1) {
        return false;
    }

    let epsilon = match parts.next().and_then(parse_decimal_f32) {
        Some(value) if finite(value) => value,
        _ => return false,
    };

    let learning_rate = match parts.next().and_then(parse_decimal_f32) {
        Some(value) if finite(value) => value,
        _ => return false,
    };

    let gamma = match parts.next().and_then(parse_decimal_f32) {
        Some(value) if finite(value) => value,
        _ => return false,
    };

    let training_steps = match parts.next().and_then(parse_decimal_u64) {
        Some(value) => value,
        None => return false,
    };

    let completed_episodes = match parts.next().and_then(parse_decimal_u64) {
        Some(value) => value,
        None => return false,
    };

    let difficulty = match parts.next().and_then(parse_decimal_u32) {
        Some(value) if (1..=10).contains(&value) => value,
        _ => return false,
    };

    let obstacle_count = match parts.next().and_then(parse_decimal_u32) {
        Some(value) if (1..=64).contains(&value) => value,
        _ => return false,
    };

    let parameter_hex = match parts.next() {
        Some(value) => value.trim(),
        None => return false,
    };

    if parts.next().is_some() || parameter_hex.len() != Network::parameter_count() * 8 {
        return false;
    }

    let mut parameters = Vec::with_capacity(Network::parameter_count());

    for index in 0..Network::parameter_count() {
        let start = index * 8;
        let end = start + 8;
        let Some(value) = parse_hex_f32(&parameter_hex[start..end]) else {
            return false;
        };
        if !finite(value) {
            return false;
        }
        parameters.push(value);
    }

    let mut network = engine.ai.network.clone();
    if !network.load_parameters(&parameters) {
        return false;
    }

    engine.ai.network = network.clone();
    engine.ai.target_network = network;
    engine.ai.replay.clear();
    engine.ai.training_steps = training_steps;
    engine.ai.completed_episodes = completed_episodes;
    engine.ai.success_episodes = 0;
    engine.ai.collision_episodes = 0;
    engine.ai.epsilon = epsilon.clamp(0.0, 1.0);
    engine.ai.learning_rate = learning_rate.clamp(0.00001, 0.1);
    engine.ai.gamma = gamma.clamp(0.80, 0.9999);
    engine.ai.last_episode_reward = 0.0;
    engine.ai.last_episode_steps = 0;
    engine.ai.last_loss = 0.0;
    engine.ai.last_reward = 0.0;
    engine.ai.last_action = 0;

    engine.environment.difficulty = difficulty;
    engine.environment.obstacle_count = obstacle_count;
    let seed = engine.environment.seed;
    engine.environment.reset(seed);

    true
}

// ============================================================================
// SERIALIZATION HELPERS
// ============================================================================

fn push_line(output: &mut String, line: &str) {
    output.push_str(line);
    output.push('\n');
}

fn push_network_records(output: &mut String, marker: char, parameters: &[f32]) {
    // Fixed-size chunks keep line length bounded while remaining simple to
    // inspect manually. Eight hex characters per f32 means lossless roundtrip.
    const CHUNK: usize = 32;

    for chunk in parameters.chunks(CHUNK) {
        output.push(marker);
        for value in chunk {
            output.push(' ');
            output.push_str(&hex_f32(*value));
        }
        output.push('\n');
    }
}

fn parse_network_records<'a, I>(
    lines: &mut I,
    marker: char,
    expected_count: usize,
) -> Option<Vec<f32>>
where
    I: Iterator<Item = &'a str>,
{
    const CHUNK: usize = 32;

    let mut values = Vec::with_capacity(expected_count);
    let expected_lines = (expected_count + CHUNK - 1) / CHUNK;

    for line_index in 0..expected_lines {
        let line = lines.next()?;
        let parts: Vec<&str> = line.split_whitespace().collect();

        if parts.is_empty()
            || parts[0].chars().next()? != marker
            || parts[0].len() != 1
            || parts.len() < 2
            || parts.len() > CHUNK + 1
        {
            return None;
        }

        let expected_on_line = if line_index + 1 == expected_lines {
            expected_count - line_index * CHUNK
        } else {
            CHUNK
        };

        if parts.len() != expected_on_line + 1 {
            return None;
        }

        for value in &parts[1..] {
            let parsed = parse_hex_f32(value)?;
            if !finite(parsed) {
                return None;
            }
            values.push(parsed);
        }
    }

    if values.len() != expected_count {
        return None;
    }

    Some(values)
}

fn next_parts_at_least<'a, I>(lines: &mut I, marker: &str, min_len: usize) -> Option<Vec<&'a str>>
where
    I: Iterator<Item = &'a str>,
{
    let line = lines.next()?;
    let parts: Vec<&str> = line.split_whitespace().collect();

    if parts.len() < min_len || parts.first().copied() != Some(marker) {
        None
    } else {
        Some(parts)
    }
}

fn next_parts<'a, I>(lines: &mut I, marker: &str, exact_len: usize) -> Option<Vec<&'a str>>
where
    I: Iterator<Item = &'a str>,
{
    let parts = next_parts_at_least(lines, marker, exact_len)?;

    if parts.len() != exact_len {
        None
    } else {
        Some(parts)
    }
}

fn expect_exact<'a, I>(lines: &mut I, expected: &str) -> Option<()>
where
    I: Iterator<Item = &'a str>,
{
    if lines.next()? == expected {
        Some(())
    } else {
        None
    }
}

fn validate_ai_config(
    learning_rate: f32,
    gamma: f32,
    epsilon: f32,
    epsilon_min: f32,
    epsilon_decay: f32,
    batch_size: usize,
    target_update: u64,
) -> Option<()> {
    if !finite(learning_rate)
        || !finite(gamma)
        || !finite(epsilon)
        || !finite(epsilon_min)
        || !finite(epsilon_decay)
        || !(0.00001..=0.1).contains(&learning_rate)
        || !(0.80..=0.9999).contains(&gamma)
        || !(0.0..=1.0).contains(&epsilon)
        || !(0.0..=1.0).contains(&epsilon_min)
        || !(0.0..=1.0).contains(&epsilon_decay)
        || batch_size == 0
        || batch_size > 256
        || target_update == 0
        || target_update > 100_000
    {
        return None;
    }

    if epsilon < epsilon_min {
        return None;
    }

    Some(())
}

fn validate_vec3(value: Vec3) -> Option<()> {
    if finite(value.x) && finite(value.y) && finite(value.z) {
        Some(())
    } else {
        None
    }
}

fn validate_f32(value: f32) -> Option<()> {
    if finite(value) {
        Some(())
    } else {
        None
    }
}

fn finite(value: f32) -> bool {
    value.is_finite()
}

fn bool_number(value: bool) -> &'static str {
    if value {
        "1"
    } else {
        "0"
    }
}

fn parse_bool_number(value: &str) -> Option<bool> {
    match value {
        "0" => Some(false),
        "1" => Some(true),
        _ => None,
    }
}

fn hex_f32(value: f32) -> String {
    format!("{:08X}", value.to_bits())
}

fn parse_hex_f32(value: &str) -> Option<f32> {
    if value.len() != 8 {
        return None;
    }

    let bits = u32::from_str_radix(value, 16).ok()?;
    let parsed = f32::from_bits(bits);

    if parsed.is_finite() {
        Some(parsed)
    } else {
        None
    }
}

fn parse_hex_u64(value: &str) -> Option<u64> {
    if value.is_empty() || value.len() > 16 {
        None
    } else {
        u64::from_str_radix(value, 16).ok()
    }
}

fn parse_u64(value: &str) -> Option<u64> {
    value.parse::<u64>().ok()
}

fn parse_u32(value: &str) -> Option<u32> {
    value.parse::<u32>().ok()
}

fn parse_usize(value: &str) -> Option<usize> {
    value.parse::<usize>().ok()
}

fn parse_decimal_f32(value: &str) -> Option<f32> {
    value.parse::<f32>().ok()
}

fn parse_decimal_u64(value: &str) -> Option<u64> {
    value.parse::<u64>().ok()
}

fn parse_decimal_u32(value: &str) -> Option<u32> {
    value.parse::<u32>().ok()
}

// ============================================================================
// FNV-1A 64 CHECKSUM
// ============================================================================

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xCBF29CE484222325u64;

    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001B3u64);
    }

    hash
}

// ============================================================================
// UNIT TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::AiEngine;

    #[test]
    fn hex_float_roundtrip_is_lossless_for_finite_values() {
        let values = [
            0.0f32,
            1.0,
            -1.25,
            0.0005,
            3.1415927,
            f32::MAX,
            f32::MIN_POSITIVE,
        ];

        for value in values {
            let encoded = hex_f32(value);
            let decoded = parse_hex_f32(&encoded).unwrap();
            assert_eq!(value.to_bits(), decoded.to_bits());
        }
    }

    #[test]
    fn fnv1a_is_deterministic() {
        let first = fnv1a64(b"RAYTRC");
        let second = fnv1a64(b"RAYTRC");
        assert_eq!(first, second);
        assert_ne!(first, fnv1a64(b"RAYTRC!"));
    }

    #[test]
    fn network_parameter_count_matches_checkpoint_shape() {
        assert_eq!(Network::parameter_count(), PARAMETER_COUNT_CHECK);
    }

    #[test]
    fn replay_bounds_reject_invalid_shape() {
        assert!((0usize <= MAX_CHECKPOINT_REPLAY));
        assert!(MAX_CHECKPOINT_REPLAY >= 20_000);
        assert!(MAX_CHECKPOINT_OBSTACLES >= 64);
    }

    #[test]
    fn legacy_parser_rejects_wrong_magic() {
        let mut engine = crate::RayTracerEngine::new(1);
        assert!(!import_v1(&mut engine, "NOPE|1|0.1|0.99|0|0|1|8|"));
    }

    #[test]
    fn legacy_parameter_hex_length_matches_network() {
        assert_eq!(Network::parameter_count() * 8, 6917 * 8,);
    }

    #[test]
    fn ai_rng_never_serializes_as_zero_for_valid_rng() {
        let rng = Lcg::new(123);
        assert_ne!(rng.state, 0);
    }

    #[test]
    fn bool_codec_is_strict() {
        assert_eq!(parse_bool_number("0"), Some(false));
        assert_eq!(parse_bool_number("1"), Some(true));
        assert_eq!(parse_bool_number("2"), None);
    }
}
