use std::collections::VecDeque;
use std::f32::consts::{PI, TAU};

use crate::ai::Lcg;
use crate::{
    Aabb, Agent, Obstacle, StepResult, Vec3, ACTION_COUNT, AGENT_RADIUS, MAX_SENSOR_DISTANCE,
    MAX_SPEED, OBS_SIZE, SENSOR_COUNT, TARGET_RADIUS, TURN_RATE, WORLD_HALF, WORLD_SIZE,
};

// ============================================================================
// ENVIRONMENT
// ============================================================================

#[derive(Clone, Debug)]
pub(crate) struct Environment {
    pub(crate) seed: u32,
    pub(crate) rng: Lcg,

    pub(crate) difficulty: u32,
    pub(crate) obstacle_count: u32,
    pub(crate) max_steps: u32,

    pub(crate) step: u32,

    pub(crate) obstacles: Vec<Obstacle>,

    pub(crate) agent: Agent,

    pub(crate) start: Vec3,
    pub(crate) target: Vec3,

    pub(crate) last_distance: f32,
    pub(crate) episode_reward: f32,

    pub(crate) done: bool,
    pub(crate) success: bool,
    pub(crate) collision: bool,

    pub(crate) sensors: [f32; SENSOR_COUNT],

    pub(crate) path: Vec<Vec3>,
}

impl Environment {
    pub(crate) fn new(seed: u32, difficulty: u32, obstacle_count: u32, max_steps: u32) -> Self {
        let mut environment = Self {
            seed: if seed == 0 { 1 } else { seed },

            rng: Lcg::new(seed as u64),

            difficulty: difficulty.clamp(1, 10),

            obstacle_count: obstacle_count.clamp(1, 64),

            max_steps: max_steps.max(1),

            step: 0,

            obstacles: Vec::new(),

            agent: Agent::default(),

            start: Vec3::default(),
            target: Vec3::default(),

            last_distance: 0.0,
            episode_reward: 0.0,

            done: false,
            success: false,
            collision: false,

            sensors: [1.0; SENSOR_COUNT],

            path: Vec::new(),
        };

        environment.reset(seed);

        environment
    }

    // =========================================================================
    // RESET ENVIRONMENT
    // =========================================================================

    pub(crate) fn reset(&mut self, requested_seed: u32) {
        let seed = if requested_seed == 0 {
            self.next_seed()
        } else {
            requested_seed
        };

        self.seed = seed;

        self.rng = Lcg::new(seed as u64);

        self.step = 0;
        self.episode_reward = 0.0;

        self.done = false;
        self.success = false;
        self.collision = false;

        self.path.clear();

        let desired_count = self.obstacle_count.clamp(1, 64) as usize;

        let mut found_valid_map = false;
        let mut attempts = 0usize;

        // ---------------------------------------------------------------------
        // MAP GENERATION
        // ---------------------------------------------------------------------

        while attempts < 120 && !found_valid_map {
            attempts += 1;

            self.obstacles.clear();

            self.start = self.random_point(-8.2, 8.2);

            self.target = self.random_point(-8.2, 8.2);

            if self.start.distance_xz(self.target) < 8.0 {
                continue;
            }

            let difficulty = self.difficulty as f32;

            let mut obstacle_attempts = 0usize;

            while self.obstacles.len() < desired_count
                && obstacle_attempts < desired_count * 60 + 120
            {
                obstacle_attempts += 1;

                let center = self.random_point(-8.5, 8.5);

                let scale = 0.75 + difficulty * 0.08;

                let max_size = (1.35 + scale).min(3.0);

                let half_x = self.rng.range_f32(0.45, max_size);

                let half_z = self.rng.range_f32(0.45, max_size);

                let candidate = Aabb {
                    center: Vec3::new(center.x, 0.5, center.z),

                    half: Vec3::new(half_x, 0.5, half_z),
                };

                // ----------------------------------------------------------------
                // KEEP START CLEAR
                // ----------------------------------------------------------------

                if point_aabb_distance_xz(self.start, &candidate) < AGENT_RADIUS + 0.9 {
                    continue;
                }

                // ----------------------------------------------------------------
                // KEEP TARGET CLEAR
                // ----------------------------------------------------------------

                if point_aabb_distance_xz(self.target, &candidate) < TARGET_RADIUS + 0.9 {
                    continue;
                }

                // ----------------------------------------------------------------
                // KEEP OBSTACLES SEPARATED
                // ----------------------------------------------------------------

                if self
                    .obstacles
                    .iter()
                    .any(|obstacle| aabb_overlap_expanded(&obstacle.bounds, &candidate, 0.35))
                {
                    continue;
                }

                let material = self.rng.next_u32() % 4;

                self.obstacles.push(Obstacle {
                    bounds: candidate,
                    material,
                });
            }

            // ------------------------------------------------------------------
            // PATH VALIDATION
            // ------------------------------------------------------------------

            found_valid_map = self.path_exists();
        }

        // ---------------------------------------------------------------------
        // FALLBACK
        // ---------------------------------------------------------------------

        if !found_valid_map {
            self.obstacles.truncate(desired_count.min(4));

            while !self.path_exists() && !self.obstacles.is_empty() {
                self.obstacles.pop();
            }
        }

        // ---------------------------------------------------------------------
        // INITIAL HEADING
        // ---------------------------------------------------------------------

        let dx = self.target.x - self.start.x;

        let dz = self.target.z - self.start.z;

        // Movement convention is direction=(sin(heading), cos(heading)),
        // therefore the world heading toward (dx,dz) is atan2(dx,dz).
        let target_heading = dx.atan2(dz);

        // Never start an episode facing directly into an obstacle. The old
        // implementation randomized the initial heading by +/-0.65 rad, which
        // could put the agent on a collision course immediately. Sample the
        // whole circle and choose the clearest useful direction while staying
        // reasonably aligned with the target.
        let safe_heading = self.choose_safe_heading(target_heading);

        self.agent = Agent {
            position: Vec3::new(self.start.x, 0.5, self.start.z),

            heading: safe_heading,

            speed: 0.0,
        };

        self.last_distance = self.agent.position.distance_xz(self.target);

        self.path.push(self.agent.position);

        self.update_sensors();
    }

    /// Choose a safe heading from the current start position. The score favors
    /// obstacle clearance first, then alignment with the target direction.
    fn choose_recovery_heading(&mut self, target_heading: f32) -> f32 {
        let mut top_headings = [target_heading; 8];
        let mut top_scores = [f32::NEG_INFINITY; 8];

        // Score the same 72 candidate headings used by the normal spawn
        // selector, but retain several good alternatives instead of always
        // returning the single best one.
        for index in 0..72usize {
            let heading = target_heading + (index as f32 / 72.0) * TAU;

            let direction = Vec3::new(heading.sin(), 0.0, heading.cos());

            let mut clearance = boundary_distance(self.start, direction).min(MAX_SENSOR_DISTANCE);

            for obstacle in &self.obstacles {
                if let Some(hit) = ray_aabb_distance_xz(self.start, direction, &obstacle.bounds) {
                    clearance = clearance.min(hit);
                }
            }

            if clearance < 1.0 {
                continue;
            }

            let alignment = (heading - target_heading).cos();

            let score = (clearance.min(6.0) / 6.0) * 2.5 + alignment * 1.0;

            for slot in 0..top_scores.len() {
                if score > top_scores[slot] {
                    for shift in (slot + 1..top_scores.len()).rev() {
                        top_scores[shift] = top_scores[shift - 1];
                        top_headings[shift] = top_headings[shift - 1];
                    }

                    top_scores[slot] = score;
                    top_headings[slot] = heading;
                    break;
                }
            }
        }

        let available = top_scores.iter().filter(|score| score.is_finite()).count();

        if available == 0 {
            return self.choose_safe_heading(target_heading);
        }

        let pick = self.rng.range_usize(available);

        top_headings[pick]
    }

    fn choose_safe_heading(&self, target_heading: f32) -> f32 {
        let mut best_heading = target_heading;
        let mut best_score = f32::NEG_INFINITY;

        for index in 0..72usize {
            let heading = target_heading + (index as f32 / 72.0) * TAU;

            let direction = Vec3::new(heading.sin(), 0.0, heading.cos());

            let mut clearance = boundary_distance(self.start, direction).min(MAX_SENSOR_DISTANCE);

            for obstacle in &self.obstacles {
                if let Some(hit) = ray_aabb_distance_xz(self.start, direction, &obstacle.bounds) {
                    clearance = clearance.min(hit);
                }
            }

            let alignment = (heading - target_heading).cos();

            // Reject headings that are too close to impact. Among safe
            // headings, prefer useful clearance while remaining goal-oriented.
            let safety_bonus = if clearance < 1.20 {
                -100.0 + clearance
            } else {
                (clearance.min(5.0) / 5.0) * 2.2
            };

            let score = safety_bonus + alignment * 1.25;

            if score > best_score {
                best_score = score;
                best_heading = heading;
            }
        }

        best_heading
    }

    fn next_seed(&mut self) -> u32 {
        self.rng.next_u32().max(1)
    }

    fn random_point(&mut self, min: f32, max: f32) -> Vec3 {
        Vec3::new(
            self.rng.range_f32(min, max),
            0.5,
            self.rng.range_f32(min, max),
        )
    }

    // =========================================================================
    // PATH VALIDATION
    // =========================================================================

    fn path_exists(&self) -> bool {
        const GRID: usize = 40;

        let mut blocked = vec![false; GRID * GRID];

        for z in 0..GRID {
            for x in 0..GRID {
                let point = grid_to_world(x, z, GRID);

                blocked[z * GRID + x] = self.obstacles.iter().any(|obstacle| {
                    point_aabb_distance_xz(point, &obstacle.bounds) < AGENT_RADIUS * 1.4
                });
            }
        }

        let start = world_to_grid(self.start, GRID);

        let target = world_to_grid(self.target, GRID);

        let start_index = start.1 * GRID + start.0;

        let target_index = target.1 * GRID + target.0;

        if blocked[start_index] || blocked[target_index] {
            return false;
        }

        let mut visited = vec![false; GRID * GRID];

        let mut queue = VecDeque::new();

        visited[start_index] = true;

        queue.push_back(start);

        let directions = [(1isize, 0isize), (-1, 0), (0, 1), (0, -1)];

        while let Some((x, z)) = queue.pop_front() {
            if (x, z) == target {
                return true;
            }

            for (dx, dz) in directions {
                let nx = x as isize + dx;

                let nz = z as isize + dz;

                if nx < 0 || nz < 0 || nx >= GRID as isize || nz >= GRID as isize {
                    continue;
                }

                let cell = (nx as usize, nz as usize);

                let index = cell.1 * GRID + cell.0;

                if blocked[index] || visited[index] {
                    continue;
                }

                visited[index] = true;

                queue.push_back(cell);
            }
        }

        false
    }

    // =========================================================================
    // ENVIRONMENT STEP
    // =========================================================================

    pub(crate) fn step(&mut self, action: usize) -> StepResult {
        if self.done {
            return StepResult {
                reward: 0.0,

                done: true,

                success: self.success,

                collision: self.collision,

                distance_to_target: self.last_distance,
            };
        }

        let previous_distance = self.agent.position.distance_xz(self.target);

        // ---------------------------------------------------------------------
        // ACTION
        // ---------------------------------------------------------------------

        match action.min(ACTION_COUNT - 1) {
            // FORWARD
            0 => {
                self.agent.speed = (self.agent.speed + 0.018).min(MAX_SPEED);
            }

            // LEFT
            1 => {
                self.agent.heading -= TURN_RATE;

                self.agent.speed = (self.agent.speed + 0.015).min(MAX_SPEED);
            }

            // RIGHT
            2 => {
                self.agent.heading += TURN_RATE;

                self.agent.speed = (self.agent.speed + 0.015).min(MAX_SPEED);
            }

            // BRAKE
            3 => {
                self.agent.speed *= 0.35;
            }

            // REVERSE
            4 => {
                self.agent.speed = -(self.agent.speed.abs().max(0.06)).min(MAX_SPEED * 0.7);
            }

            _ => {}
        }

        let direction = Vec3::new(self.agent.heading.sin(), 0.0, self.agent.heading.cos());

        let proposed = Vec3::new(
            self.agent.position.x + direction.x * self.agent.speed,
            0.5,
            self.agent.position.z + direction.z * self.agent.speed,
        );

        let mut reward = -0.002;

        let mut done = false;
        let mut success = false;
        let mut collision = false;

        // ---------------------------------------------------------------------
        // COLLISION
        // ---------------------------------------------------------------------

        if proposed.x.abs() > WORLD_HALF - AGENT_RADIUS
            || proposed.z.abs() > WORLD_HALF - AGENT_RADIUS
            || self.collides_circle(proposed, AGENT_RADIUS)
        {
            collision = true;

            done = true;

            reward = -2.0;

            self.collision = true;

            self.agent.speed = 0.0;
        } else {
            self.agent.position = proposed;

            let new_distance = self.agent.position.distance_xz(self.target);

            // -----------------------------------------------------------------
            // PROGRESS REWARD
            // -----------------------------------------------------------------

            let progress = previous_distance - new_distance;

            reward += progress * 0.35;

            // Dense directional signal. The network gets a small positive
            // value for facing the target and a negative value for turning
            // away, which makes credit assignment much easier than relying
            // on the terminal +1 alone.
            let target_heading = (self.target.x - self.agent.position.x)
                .atan2(self.target.z - self.agent.position.z);

            let heading_error = wrap_angle(target_heading - self.agent.heading);

            reward += heading_error.cos() * 0.010;

            // Penalize getting dangerously close to an obstacle. The sensor
            // value is normalized distance, so this is local and scale-free.
            let front_clearance = self.sensors[0];

            if front_clearance < 0.35 {
                reward -= (0.35 - front_clearance) * 0.035;
            }

            reward = reward.clamp(-0.20, 0.20);

            // -----------------------------------------------------------------
            // TARGET
            // -----------------------------------------------------------------

            if new_distance <= TARGET_RADIUS {
                done = true;

                success = true;

                reward += 2.0;

                self.success = true;
            } else if self.step + 1 >= self.max_steps {
                done = true;
            }
        }

        self.step += 1;

        self.episode_reward += reward;

        self.done = done;

        self.last_distance = self.agent.position.distance_xz(self.target);

        // ---------------------------------------------------------------------
        // PATH HISTORY
        // ---------------------------------------------------------------------

        if self.path.len() < 512 {
            self.path.push(self.agent.position);
        } else if self.step % 4 == 0 {
            let index = (self.step as usize / 4) % self.path.len();

            self.path[index] = self.agent.position;
        }

        self.update_sensors();

        StepResult {
            reward,

            done,

            success,

            collision,

            distance_to_target: self.last_distance,
        }
    }

    // =========================================================================
    // COLLISION TEST
    // =========================================================================

    fn collides_circle(&self, point: Vec3, radius: f32) -> bool {
        self.obstacles
            .iter()
            .any(|obstacle| point_aabb_distance_xz(point, &obstacle.bounds) < radius)
    }

    // =========================================================================
    // UPDATE SENSOR ARRAY
    // =========================================================================

    fn update_sensors(&mut self) {
        for index in 0..SENSOR_COUNT {
            let angle = self.agent.heading + TAU * (index as f32 / SENSOR_COUNT as f32);

            let direction = Vec3::new(angle.sin(), 0.0, angle.cos());

            let mut distance =
                boundary_distance(self.agent.position, direction).min(MAX_SENSOR_DISTANCE);

            for obstacle in &self.obstacles {
                if let Some(distance_hit) =
                    ray_aabb_distance_xz(self.agent.position, direction, &obstacle.bounds)
                {
                    distance = distance.min(distance_hit);
                }
            }

            self.sensors[index] = (distance / MAX_SENSOR_DISTANCE).clamp(0.0, 1.0);
        }
    }

    // =========================================================================
    // OBSERVATION
    // =========================================================================
    //
    // 0..31  = 32 ray sensors
    // 32     = target distance
    // 33     = target angle sin
    // 34     = target angle cos
    // 35     = speed
    // 36     = collision
    //
    // =========================================================================

    pub(crate) fn observation(&self) -> [f32; OBS_SIZE] {
        let mut observation = [0.0; OBS_SIZE];

        observation[..SENSOR_COUNT].copy_from_slice(&self.sensors);

        let dx = self.target.x - self.agent.position.x;

        let dz = self.target.z - self.agent.position.z;

        let distance = (dx * dx + dz * dz).sqrt();

        // Agent heading convention:
        //   heading = 0 means +Z
        //   direction = (sin(heading), 0, cos(heading))
        // Therefore target heading must be atan2(dx, dz).
        let target_heading = dx.atan2(dz);

        let angle = wrap_angle(target_heading - self.agent.heading);

        observation[32] = (distance / (WORLD_SIZE * 1.41421356)).clamp(0.0, 1.0);

        observation[33] = angle.sin();

        observation[34] = angle.cos();

        observation[35] = (self.agent.speed.abs() / MAX_SPEED).clamp(0.0, 1.0);

        observation[36] = if self.collision { 1.0 } else { 0.0 };

        observation
    }

    // =========================================================================
    // AGENT BUFFER
    // =========================================================================

    pub(crate) fn agent_buffer(&self) -> Vec<f32> {
        vec![
            self.agent.position.x,
            self.agent.position.y,
            self.agent.position.z,
            self.agent.heading,
            self.agent.speed,
            AGENT_RADIUS,
            if self.collision { 1.0 } else { 0.0 },
            if self.success { 1.0 } else { 0.0 },
        ]
    }

    // =========================================================================
    // TARGET BUFFER
    // =========================================================================

    pub(crate) fn target_buffer(&self) -> Vec<f32> {
        vec![self.target.x, self.target.y, self.target.z, TARGET_RADIUS]
    }

    // =========================================================================
    // OBSTACLE BUFFER
    // =========================================================================
    //
    // 8 floats/object:
    //
    // [0] center.x
    // [1] center.y
    // [2] center.z
    // [3] half.x
    // [4] half.y
    // [5] half.z
    // [6] material
    // [7] reserved
    //
    // =========================================================================

    pub(crate) fn obstacle_buffer(&self) -> Vec<f32> {
        let mut output = Vec::with_capacity(self.obstacles.len() * 8);

        for obstacle in &self.obstacles {
            output.extend_from_slice(&[
                obstacle.bounds.center.x,
                obstacle.bounds.center.y,
                obstacle.bounds.center.z,
                obstacle.bounds.half.x,
                obstacle.bounds.half.y,
                obstacle.bounds.half.z,
                obstacle.material as f32,
                0.0,
            ]);
        }

        output
    }

    // =========================================================================
    // PATH BUFFER
    // =========================================================================

    pub(crate) fn path_buffer(&self) -> Vec<f32> {
        let mut output = Vec::with_capacity(self.path.len() * 3);

        for point in &self.path {
            output.extend_from_slice(&[point.x, point.y, point.z]);
        }

        output
    }
}

// ============================================================================
// ANGLE NORMALIZATION
// ============================================================================

fn wrap_angle(mut angle: f32) -> f32 {
    while angle > PI {
        angle -= TAU;
    }

    while angle < -PI {
        angle += TAU;
    }

    angle
}

// ============================================================================
// POINT -> AABB DISTANCE
// ============================================================================

fn point_aabb_distance_xz(point: Vec3, aabb: &Aabb) -> f32 {
    let dx = (point.x - aabb.center.x).abs() - aabb.half.x;

    let dz = (point.z - aabb.center.z).abs() - aabb.half.z;

    let qx = dx.max(0.0);

    let qz = dz.max(0.0);

    if dx <= 0.0 && dz <= 0.0 {
        0.0
    } else {
        (qx * qx + qz * qz).sqrt()
    }
}

// ============================================================================
// AABB OVERLAP
// ============================================================================

fn aabb_overlap_expanded(a: &Aabb, b: &Aabb, margin: f32) -> bool {
    (a.center.x - b.center.x).abs() <= a.half.x + b.half.x + margin
        && (a.center.z - b.center.z).abs() <= a.half.z + b.half.z + margin
}

// ============================================================================
// RAY -> AABB
// ============================================================================

fn ray_aabb_distance_xz(origin: Vec3, direction: Vec3, aabb: &Aabb) -> Option<f32> {
    let min_x = aabb.min_x();

    let max_x = aabb.max_x();

    let min_z = aabb.min_z();

    let max_z = aabb.max_z();

    let mut t_min = 0.0f32;

    let mut t_max = MAX_SENSOR_DISTANCE;

    for (origin_component, direction_component, minimum, maximum) in [
        (origin.x, direction.x, min_x, max_x),
        (origin.z, direction.z, min_z, max_z),
    ] {
        if direction_component.abs() < 1.0e-8 {
            if origin_component < minimum || origin_component > maximum {
                return None;
            }
        } else {
            let inverse = 1.0 / direction_component;

            let mut t0 = (minimum - origin_component) * inverse;

            let mut t1 = (maximum - origin_component) * inverse;

            if t0 > t1 {
                std::mem::swap(&mut t0, &mut t1);
            }

            t_min = t_min.max(t0);

            t_max = t_max.min(t1);

            if t_max < t_min {
                return None;
            }
        }
    }

    if t_max < 0.0 {
        None
    } else {
        Some(t_min.max(0.0))
    }
}

// ============================================================================
// WORLD BOUNDARY SENSOR
// ============================================================================

fn boundary_distance(origin: Vec3, direction: Vec3) -> f32 {
    let mut best = MAX_SENSOR_DISTANCE;

    if direction.x > 1.0e-8 {
        best = best.min((WORLD_HALF - origin.x) / direction.x);
    }

    if direction.x < -1.0e-8 {
        best = best.min((-WORLD_HALF - origin.x) / direction.x);
    }

    if direction.z > 1.0e-8 {
        best = best.min((WORLD_HALF - origin.z) / direction.z);
    }

    if direction.z < -1.0e-8 {
        best = best.min((-WORLD_HALF - origin.z) / direction.z);
    }

    if best.is_finite() && best > 0.0 {
        best
    } else {
        MAX_SENSOR_DISTANCE
    }
}

// ============================================================================
// WORLD <-> GRID
// ============================================================================

fn world_to_grid(point: Vec3, grid: usize) -> (usize, usize) {
    let x = ((point.x + WORLD_HALF) / WORLD_SIZE * grid as f32)
        .floor()
        .clamp(0.0, (grid - 1) as f32) as usize;

    let z = ((point.z + WORLD_HALF) / WORLD_SIZE * grid as f32)
        .floor()
        .clamp(0.0, (grid - 1) as f32) as usize;

    (x, z)
}

fn grid_to_world(x: usize, z: usize, grid: usize) -> Vec3 {
    let fx = (x as f32 + 0.5) / grid as f32;

    let fz = (z as f32 + 0.5) / grid as f32;

    Vec3::new(
        fx * WORLD_SIZE - WORLD_HALF,
        0.5,
        fz * WORLD_SIZE - WORLD_HALF,
    )
}

impl Environment {
    // ========================================================================
    // RUNTIME / CHECKPOINT ACCESSORS
    // ========================================================================

    pub(crate) fn seed(&self) -> u32 {
        self.seed
    }

    pub(crate) fn last_distance(&self) -> f32 {
        self.last_distance
    }

    pub(crate) fn episode_reward(&self) -> f32 {
        self.episode_reward
    }

    pub(crate) fn current_step(&self) -> u32 {
        self.step
    }

    pub(crate) fn is_done(&self) -> bool {
        self.done
    }

    pub(crate) fn is_success(&self) -> bool {
        self.success
    }

    pub(crate) fn is_collision(&self) -> bool {
        self.collision
    }

    pub(crate) fn is_collision_terminal(&self) -> bool {
        self.done && self.collision
    }

    pub(crate) fn difficulty(&self) -> u32 {
        self.difficulty
    }

    pub(crate) fn obstacle_count(&self) -> usize {
        self.obstacles.len()
    }

    pub(crate) fn configured_obstacle_count(&self) -> u32 {
        self.obstacle_count
    }

    pub(crate) fn episode_limit(&self) -> u32 {
        self.max_steps
    }

    pub(crate) fn set_difficulty(&mut self, difficulty: u32) {
        self.difficulty = difficulty.clamp(1, 10);

        let seed = self.seed;
        self.reset(seed);
    }

    pub(crate) fn set_obstacle_count(&mut self, count: u32) {
        self.obstacle_count = count.clamp(1, 64);

        let seed = self.seed;
        self.reset(seed);
    }

    pub(crate) fn set_episode_steps(&mut self, steps: u32) {
        self.max_steps = steps.clamp(50, 5000);

        if self.step > self.max_steps {
            self.step = self.max_steps;
        }
    }

    pub(crate) fn sensor_buffer(&self) -> Vec<f32> {
        self.sensors.to_vec()
    }

    pub(crate) fn recover_from_collision(&mut self) {
        if !self.is_collision_terminal() {
            return;
        }

        let dx = self.target.x - self.start.x;
        let dz = self.target.z - self.start.z;
        let target_heading = dx.atan2(dz);

        // IMPORTANT: collision recovery must NOT recreate the exact same
        // initial state on every retry. That would make a deterministic
        // policy repeat the identical collision forever on the same map.
        // Pick one of the best safe headings using the environment RNG so
        // each recovery explores a different valid departure direction.
        let safe_heading = self.choose_recovery_heading(target_heading);

        self.step = 0;
        self.episode_reward = 0.0;
        self.done = false;
        self.success = false;
        self.collision = false;

        self.agent = Agent {
            position: Vec3::new(self.start.x, 0.5, self.start.z),
            heading: safe_heading,
            speed: 0.0,
        };

        self.last_distance = self.agent.position.distance_xz(self.target);

        self.path.clear();
        self.path.push(self.agent.position);
        self.update_sensors();
    }

    #[cfg(test)]
    pub(crate) fn force_terminal_for_test(
        &mut self,
        success: bool,
        collision: bool,
        distance: f32,
    ) {
        self.done = true;
        self.success = success;
        self.collision = collision;
        self.last_distance = distance;
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_reset_is_reproducible() {
        let first = Environment::new(1234, 3, 8, 500);
        let second = Environment::new(1234, 3, 8, 500);

        assert_eq!(first.seed, second.seed);
        assert_eq!(first.start.x, second.start.x);
        assert_eq!(first.start.z, second.start.z);
        assert_eq!(first.target.x, second.target.x);
        assert_eq!(first.target.z, second.target.z);
        assert_eq!(first.obstacles.len(), second.obstacles.len());
        assert_eq!(first.sensors, second.sensors);

        for (a, b) in first.obstacles.iter().zip(second.obstacles.iter()) {
            assert_eq!(a.bounds.center.x, b.bounds.center.x);
            assert_eq!(a.bounds.center.z, b.bounds.center.z);
            assert_eq!(a.bounds.half.x, b.bounds.half.x);
            assert_eq!(a.bounds.half.z, b.bounds.half.z);
            assert_eq!(a.material, b.material);
        }
    }

    #[test]
    fn observation_has_expected_shape_and_range() {
        let environment = Environment::new(9876, 4, 10, 500);

        let observation = environment.observation();

        assert_eq!(observation.len(), OBS_SIZE);
        assert!(observation.iter().all(|v| v.is_finite()));
        assert!(environment.sensors.iter().all(|v| *v >= 0.0 && *v <= 1.0));
        assert!(observation[32] >= 0.0 && observation[32] <= 1.0);
        assert!(observation[35] >= 0.0 && observation[35] <= 1.0);
        assert!(observation[36] == 0.0);
    }

    #[test]
    fn generated_map_has_valid_start_target_and_clearance() {
        let environment = Environment::new(4567, 5, 12, 500);

        assert!(environment.start.x.abs() <= WORLD_HALF);
        assert!(environment.start.z.abs() <= WORLD_HALF);
        assert!(environment.target.x.abs() <= WORLD_HALF);
        assert!(environment.target.z.abs() <= WORLD_HALF);
        assert!(environment.start.distance_xz(environment.target) >= 8.0);

        for obstacle in &environment.obstacles {
            assert!(
                point_aabb_distance_xz(environment.start, &obstacle.bounds) >= AGENT_RADIUS + 0.9
            );
            assert!(
                point_aabb_distance_xz(environment.target, &obstacle.bounds) >= TARGET_RADIUS + 0.9
            );
        }
    }

    #[test]
    fn terminal_state_is_sticky() {
        let mut environment = Environment::new(123, 2, 4, 500);

        environment.force_terminal_for_test(true, false, 0.2);
        let result = environment.step(0);

        assert!(result.done);
        assert!(result.success);
        assert!(!result.collision);
        assert_eq!(result.reward, 0.0);
    }

    #[test]
    fn collision_recovery_keeps_same_map() {
        let mut environment = Environment::new(2222, 4, 10, 500);

        let seed = environment.seed;
        let start = environment.start;
        let target = environment.target;
        let obstacle_count = environment.obstacles.len();

        environment.force_terminal_for_test(false, true, 3.0);
        environment.recover_from_collision();

        assert_eq!(environment.seed, seed);
        assert_eq!(environment.start.x, start.x);
        assert_eq!(environment.start.z, start.z);
        assert_eq!(environment.target.x, target.x);
        assert_eq!(environment.target.z, target.z);
        assert_eq!(environment.obstacles.len(), obstacle_count);
        assert!(!environment.done);
        assert!(!environment.collision);
        assert!(!environment.success);
        assert_eq!(environment.step, 0);
        assert_eq!(environment.agent.speed, 0.0);
    }

    #[test]
    fn configuration_resets_same_seed() {
        let mut environment = Environment::new(3333, 2, 4, 500);

        let seed = environment.seed;
        environment.set_difficulty(7);
        assert_eq!(environment.seed, seed);
        assert_eq!(environment.difficulty, 7);

        environment.set_obstacle_count(12);
        assert_eq!(environment.seed, seed);
        assert_eq!(environment.configured_obstacle_count(), 12);

        environment.set_episode_steps(750);
        assert_eq!(environment.episode_limit(), 750);
    }
}
