use std::collections::VecDeque;
use std::fmt::Write as FmtWrite;

use wasm_bindgen::prelude::*;


// ============================================================================
// RAYTRC.AI / NAVIGATION ENGINE
// ============================================================================
//
// PURPOSE
//
//   3D navigation simulator + ray sensors + DQN-lite.
//
// CORE LOOP
//
//   ENVIRONMENT
//       |
//       v
//   32 RAY SENSORS
//       |
//       v
//   37-D OBSERVATION
//       |
//       v
//   NEURAL NETWORK
//       |
//       v
//   ACTION
//       |
//       v
//   ENVIRONMENT STEP
//       |
//       +---- reward
//       +---- collision
//       +---- success
//       |
//       v
//   REPLAY BUFFER
//       |
//       v
//   DQN TRAINING
//
// NO IMAGE AI.
// NO IMAGE GENERATION.
// NO SERVER REQUIRED.
//
// Designed for:
//   Rust -> WASM -> Astro -> GitHub Pages
//
// ============================================================================


// ============================================================================
// CONSTANTS
// ============================================================================

const OBS_SIZE: usize = 37;
const SENSOR_COUNT: usize = 32;

const HIDDEN_1: usize = 64;
const HIDDEN_2: usize = 64;

const ACTION_COUNT: usize = 5;


// -----------------------------------------------------------------------------
// WORLD
// -----------------------------------------------------------------------------

const WORLD_HALF: f32 = 10.0;
const WORLD_SIZE: f32 = WORLD_HALF * 2.0;

const AGENT_RADIUS: f32 = 0.30;
const TARGET_RADIUS: f32 = 0.70;

const MAX_SENSOR_DISTANCE: f32 = 14.5;

const MAX_SPEED: f32 = 0.18;
const TURN_RATE: f32 = 0.26;

const DEFAULT_MAX_STEPS: u32 = 500;


// -----------------------------------------------------------------------------
// AI
// -----------------------------------------------------------------------------

const DEFAULT_REPLAY_CAPACITY: usize = 20_000;
const DEFAULT_BATCH_SIZE: usize = 64;
const DEFAULT_TARGET_UPDATE: u64 = 1_000;


// -----------------------------------------------------------------------------
// MATH
// -----------------------------------------------------------------------------

const PI: f32 = std::f32::consts::PI;
const TWO_PI: f32 = std::f32::consts::TAU;


// ============================================================================
// VECTOR
// ============================================================================

#[derive(Clone, Copy, Debug, Default)]
struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

impl Vec3 {
    fn new(
        x: f32,
        y: f32,
        z: f32,
    ) -> Self {
        Self { x, y, z }
    }

    fn distance_xz(
        self,
        other: Self,
    ) -> f32 {
        let dx = self.x - other.x;
        let dz = self.z - other.z;

        (dx * dx + dz * dz).sqrt()
    }
}


// ============================================================================
// AABB
// ============================================================================

#[derive(Clone, Copy, Debug)]
struct Aabb {
    center: Vec3,
    half: Vec3,
}

impl Aabb {
    fn min_x(
        &self,
    ) -> f32 {
        self.center.x - self.half.x
    }

    fn max_x(
        &self,
    ) -> f32 {
        self.center.x + self.half.x
    }

    fn min_z(
        &self,
    ) -> f32 {
        self.center.z - self.half.z
    }

    fn max_z(
        &self,
    ) -> f32 {
        self.center.z + self.half.z
    }
}


// ============================================================================
// OBSTACLE
// ============================================================================

#[derive(Clone, Copy, Debug)]
struct Obstacle {
    bounds: Aabb,
    material: u32,
}


// ============================================================================
// AGENT
// ============================================================================

#[derive(Clone, Copy, Debug, Default)]
struct Agent {
    position: Vec3,
    heading: f32,
    speed: f32,
}


// ============================================================================
// STEP RESULT
// ============================================================================

#[derive(Clone, Copy, Debug, Default)]
struct StepResult {
    reward: f32,
    done: bool,
    success: bool,
    collision: bool,
    distance_to_target: f32,
}


// ============================================================================
// REPLAY TRANSITION
// ============================================================================

#[derive(Clone, Copy)]
struct Transition {
    state: [f32; OBS_SIZE],
    action: u8,
    reward: f32,
    next_state: [f32; OBS_SIZE],
    done: bool,
}


// ============================================================================
// REPLAY BUFFER
// ============================================================================

#[derive(Clone)]
struct ReplayBuffer {
    items: Vec<Transition>,
    capacity: usize,
    cursor: usize,
}

impl ReplayBuffer {
    fn new(
        capacity: usize,
    ) -> Self {
        Self {
            items: Vec::with_capacity(capacity),
            capacity: capacity.max(1),
            cursor: 0,
        }
    }

    fn clear(
        &mut self,
    ) {
        self.items.clear();
        self.cursor = 0;
    }

    fn len(
        &self,
    ) -> usize {
        self.items.len()
    }

    fn push(
        &mut self,
        item: Transition,
    ) {
        if self.items.len() < self.capacity {
            self.items.push(item);
        } else {
            self.items[self.cursor] = item;

            self.cursor =
                (self.cursor + 1) % self.capacity;
        }
    }

    fn sample(
        &self,
        rng: &mut Lcg,
    ) -> Option<Transition> {
        if self.items.is_empty() {
            None
        } else {
            let index =
                rng.range_usize(
                    self.items.len()
                );

            Some(self.items[index])
        }
    }
}


// ============================================================================
// DETERMINISTIC RANDOM GENERATOR
// ============================================================================
//
// No external rand crate.
// Good for WASM and reproducible environments.
//
// ============================================================================

#[derive(Clone, Copy)]
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(
        seed: u64,
    ) -> Self {
        Self {
            state: if seed == 0 {
                0x9E3779B97F4A7C15
            } else {
                seed
            },
        }
    }

    fn next_u32(
        &mut self,
    ) -> u32 {
        self.state ^=
            self.state >> 12;

        self.state ^=
            self.state << 25;

        self.state ^=
            self.state >> 27;

        let value =
            self.state.wrapping_mul(
                2685821657736338717,
            );

        (value >> 32) as u32
    }

    fn next_f32(
        &mut self,
    ) -> f32 {
        self.next_u32() as f32 /
        4294967296.0
    }

    fn range_f32(
        &mut self,
        min: f32,
        max: f32,
    ) -> f32 {
        min +
        (max - min) *
        self.next_f32()
    }

    fn range_usize(
        &mut self,
        max: usize,
    ) -> usize {
        if max == 0 {
            0
        } else {
            (self.next_u32() as usize) % max
        }
    }

    fn range_u32(
        &mut self,
        min: u32,
        max_inclusive: u32,
    ) -> u32 {
        if max_inclusive <= min {
            min
        } else {
            min +
            (
                self.next_u32() %
                (
                    max_inclusive -
                    min +
                    1
                )
            )
        }
    }
}


// ============================================================================
// ENVIRONMENT
// ============================================================================

#[derive(Clone)]
struct Environment {
    seed: u32,
    rng: Lcg,

    difficulty: u32,
    obstacle_count: u32,
    max_steps: u32,

    step: u32,

    obstacles: Vec<Obstacle>,

    agent: Agent,

    start: Vec3,
    target: Vec3,

    last_distance: f32,
    episode_reward: f32,

    done: bool,
    success: bool,
    collision: bool,

    sensors: [f32; SENSOR_COUNT],

    path: Vec<Vec3>,
}

impl Environment {
    fn new(
        seed: u32,
        difficulty: u32,
        obstacle_count: u32,
        max_steps: u32,
    ) -> Self {
        let mut environment = Self {
            seed: if seed == 0 { 1 } else { seed },

            rng: Lcg::new(
                seed as u64
            ),

            difficulty: difficulty.clamp(
                1,
                10,
            ),

            obstacle_count: obstacle_count.clamp(
                1,
                64,
            ),

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

    fn reset(
        &mut self,
        requested_seed: u32,
    ) {
        let seed =
            if requested_seed == 0 {
                self.next_seed()
            } else {
                requested_seed
            };

        self.seed = seed;

        self.rng =
            Lcg::new(
                seed as u64
            );

        self.step = 0;
        self.episode_reward = 0.0;

        self.done = false;
        self.success = false;
        self.collision = false;

        self.path.clear();

        let desired_count =
            self.obstacle_count
                .clamp(
                    1,
                    64,
                ) as usize;

        let mut found_valid_map = false;
        let mut attempts = 0usize;


        // ---------------------------------------------------------------------
        // MAP GENERATION
        // ---------------------------------------------------------------------

        while attempts < 120 &&
              !found_valid_map
        {
            attempts += 1;

            self.obstacles.clear();

            self.start =
                self.random_point(
                    -8.2,
                    8.2,
                );

            self.target =
                self.random_point(
                    -8.2,
                    8.2,
                );

            if self.start.distance_xz(
                self.target
            ) < 8.0 {
                continue;
            }

            let difficulty =
                self.difficulty as f32;

            let mut obstacle_attempts = 0usize;

            while self.obstacles.len() < desired_count &&
                  obstacle_attempts <
                  desired_count * 60 + 120
            {
                obstacle_attempts += 1;

                let center =
                    self.random_point(
                        -8.5,
                        8.5,
                    );

                let scale =
                    0.75 +
                    difficulty * 0.08;

                let max_size =
                    (1.35 + scale).min(3.0);

                let half_x =
                    self.rng.range_f32(
                        0.45,
                        max_size,
                    );

                let half_z =
                    self.rng.range_f32(
                        0.45,
                        max_size,
                    );

                let candidate =
                    Aabb {
                        center: Vec3::new(
                            center.x,
                            0.5,
                            center.z,
                        ),

                        half: Vec3::new(
                            half_x,
                            0.5,
                            half_z,
                        ),
                    };


                // ----------------------------------------------------------------
                // KEEP START CLEAR
                // ----------------------------------------------------------------

                if point_aabb_distance_xz(
                    self.start,
                    &candidate,
                ) <
                    AGENT_RADIUS + 0.9
                {
                    continue;
                }


                // ----------------------------------------------------------------
                // KEEP TARGET CLEAR
                // ----------------------------------------------------------------

                if point_aabb_distance_xz(
                    self.target,
                    &candidate,
                ) <
                    TARGET_RADIUS + 0.9
                {
                    continue;
                }


                // ----------------------------------------------------------------
                // KEEP OBSTACLES SEPARATED
                // ----------------------------------------------------------------

                if self.obstacles.iter().any(
                    |obstacle| {
                        aabb_overlap_expanded(
                            &obstacle.bounds,
                            &candidate,
                            0.35,
                        )
                    },
                ) {
                    continue;
                }

                let material =
                    self.rng.range_u32(
                        0,
                        3,
                    );

                self.obstacles.push(
                    Obstacle {
                        bounds: candidate,
                        material,
                    },
                );
            }


            // ------------------------------------------------------------------
            // PATH VALIDATION
            // ------------------------------------------------------------------

            found_valid_map =
                self.path_exists();
        }


        // ---------------------------------------------------------------------
        // FALLBACK
        // ---------------------------------------------------------------------

        if !found_valid_map {
            self.obstacles.truncate(
                desired_count.min(4)
            );

            while !self.path_exists() &&
                  !self.obstacles.is_empty()
            {
                self.obstacles.pop();
            }
        }


        // ---------------------------------------------------------------------
        // INITIAL HEADING
        // ---------------------------------------------------------------------

        let dx =
            self.target.x -
            self.start.x;

        let dz =
            self.target.z -
            self.start.z;

        let target_heading =
            dz.atan2(dx);

        self.agent =
            Agent {
                position: Vec3::new(
                    self.start.x,
                    0.5,
                    self.start.z,
                ),

                heading:
                    target_heading -
                    PI * 0.5 +
                    self.rng.range_f32(
                        -0.4,
                        0.4,
                    ),

                speed: 0.0,
            };

        self.last_distance =
            self.agent.position
                .distance_xz(
                    self.target,
                );

        self.path.push(
            self.agent.position
        );

        self.update_sensors();
    }

    fn next_seed(
        &mut self,
    ) -> u32 {
        self.rng
            .next_u32()
            .max(1)
    }

    fn random_point(
        &mut self,
        min: f32,
        max: f32,
    ) -> Vec3 {
        Vec3::new(
            self.rng.range_f32(
                min,
                max,
            ),

            0.5,

            self.rng.range_f32(
                min,
                max,
            ),
        )
    }


    // =========================================================================
    // PATH VALIDATION
    // =========================================================================

    fn path_exists(
        &self,
    ) -> bool {
        const GRID: usize = 40;

        let mut blocked =
            vec![
                false;
                GRID * GRID
            ];

        for z in 0..GRID {
            for x in 0..GRID {
                let point =
                    grid_to_world(
                        x,
                        z,
                        GRID,
                    );

                blocked[
                    z * GRID + x
                ] =
                    self.obstacles.iter().any(
                        |obstacle| {
                            point_aabb_distance_xz(
                                point,
                                &obstacle.bounds,
                            ) <
                                AGENT_RADIUS * 1.4
                        },
                    );
            }
        }

        let start =
            world_to_grid(
                self.start,
                GRID,
            );

        let target =
            world_to_grid(
                self.target,
                GRID,
            );

        let start_index =
            start.1 * GRID + start.0;

        let target_index =
            target.1 * GRID + target.0;

        if blocked[start_index] ||
           blocked[target_index]
        {
            return false;
        }

        let mut visited =
            vec![
                false;
                GRID * GRID
            ];

        let mut queue =
            VecDeque::new();

        visited[start_index] = true;

        queue.push_back(start);

        let directions =
            [
                (1isize, 0isize),
                (-1, 0),
                (0, 1),
                (0, -1),
            ];

        while let Some((x, z)) =
            queue.pop_front()
        {
            if (x, z) == target {
                return true;
            }

            for (dx, dz) in directions {
                let nx =
                    x as isize + dx;

                let nz =
                    z as isize + dz;

                if nx < 0 ||
                   nz < 0 ||
                   nx >= GRID as isize ||
                   nz >= GRID as isize
                {
                    continue;
                }

                let cell =
                    (
                        nx as usize,
                        nz as usize,
                    );

                let index =
                    cell.1 * GRID + cell.0;

                if blocked[index] ||
                   visited[index]
                {
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

    fn step(
        &mut self,
        action: usize,
    ) -> StepResult {
        if self.done {
            return StepResult {
                reward: 0.0,

                done: true,

                success:
                    self.success,

                collision:
                    self.collision,

                distance_to_target:
                    self.last_distance,
            };
        }

        let previous_distance =
            self.agent.position
                .distance_xz(
                    self.target,
                );


        // ---------------------------------------------------------------------
        // ACTION
        // ---------------------------------------------------------------------

        match action.min(
            ACTION_COUNT - 1,
        ) {
            // FORWARD
            0 => {
                self.agent.speed =
                    (
                        self.agent.speed +
                        0.018
                    )
                    .min(
                        MAX_SPEED
                    );
            }

            // LEFT
            1 => {
                self.agent.heading -=
                    TURN_RATE;

                self.agent.speed =
                    (
                        self.agent.speed +
                        0.015
                    )
                    .min(
                        MAX_SPEED
                    );
            }

            // RIGHT
            2 => {
                self.agent.heading +=
                    TURN_RATE;

                self.agent.speed =
                    (
                        self.agent.speed +
                        0.015
                    )
                    .min(
                        MAX_SPEED
                    );
            }

            // BRAKE
            3 => {
                self.agent.speed *=
                    0.35;
            }

            // REVERSE
            4 => {
                self.agent.speed =
                    -(
                        self.agent.speed
                            .abs()
                            .max(0.06)
                    )
                    .min(
                        MAX_SPEED * 0.7
                    );
            }

            _ => {}
        }


        let direction =
            Vec3::new(
                self.agent.heading.sin(),
                0.0,
                self.agent.heading.cos(),
            );

        let proposed =
            Vec3::new(
                self.agent.position.x +
                    direction.x *
                    self.agent.speed,

                0.5,

                self.agent.position.z +
                    direction.z *
                    self.agent.speed,
            );

        let mut reward =
            -0.002;

        let mut done = false;
        let mut success = false;
        let mut collision = false;


        // ---------------------------------------------------------------------
        // COLLISION
        // ---------------------------------------------------------------------

        if proposed.x.abs() >
                WORLD_HALF - AGENT_RADIUS ||
            proposed.z.abs() >
                WORLD_HALF - AGENT_RADIUS ||
            self.collides_circle(
                proposed,
                AGENT_RADIUS,
            )
        {
            collision = true;

            done = true;

            reward = -1.0;

            self.collision = true;

            self.agent.speed = 0.0;
        } else {
            self.agent.position =
                proposed;

            let new_distance =
                self.agent.position
                    .distance_xz(
                        self.target,
                    );


            // -----------------------------------------------------------------
            // PROGRESS REWARD
            // -----------------------------------------------------------------

            reward +=
                (
                    previous_distance -
                    new_distance
                )
                * 0.10;

            reward =
                reward.clamp(
                    -0.05,
                    0.05,
                );


            // -----------------------------------------------------------------
            // TARGET
            // -----------------------------------------------------------------

            if new_distance <=
                TARGET_RADIUS
            {
                done = true;

                success = true;

                reward += 1.0;

                self.success = true;
            } else if
                self.step + 1 >=
                self.max_steps
            {
                done = true;
            }
        }

        self.step += 1;

        self.episode_reward +=
            reward;

        self.done = done;

        self.last_distance =
            self.agent.position
                .distance_xz(
                    self.target,
                );


        // ---------------------------------------------------------------------
        // PATH HISTORY
        // ---------------------------------------------------------------------

        if self.path.len() < 512 {
            self.path.push(
                self.agent.position
            );
        } else if self.step % 4 == 0 {
            let index =
                (
                    self.step as usize /
                    4
                ) %
                self.path.len();

            self.path[index] =
                self.agent.position;
        }

        self.update_sensors();

        StepResult {
            reward,

            done,

            success,

            collision,

            distance_to_target:
                self.last_distance,
        }
    }


    // =========================================================================
    // COLLISION TEST
    // =========================================================================

    fn collides_circle(
        &self,
        point: Vec3,
        radius: f32,
    ) -> bool {
        self.obstacles.iter().any(
            |obstacle| {
                point_aabb_distance_xz(
                    point,
                    &obstacle.bounds,
                ) < radius
            },
        )
    }


    // =========================================================================
    // UPDATE SENSOR ARRAY
    // =========================================================================

    fn update_sensors(
        &mut self,
    ) {
        for index in 0..SENSOR_COUNT {
            let angle =
                self.agent.heading +
                TWO_PI *
                (
                    index as f32 /
                    SENSOR_COUNT as f32
                );

            let direction =
                Vec3::new(
                    angle.sin(),
                    0.0,
                    angle.cos(),
                );

            let mut distance =
                boundary_distance(
                    self.agent.position,
                    direction,
                )
                .min(
                    MAX_SENSOR_DISTANCE
                );

            for obstacle in &self.obstacles {
                if let Some(
                    distance_hit
                ) =
                    ray_aabb_distance_xz(
                        self.agent.position,
                        direction,
                        &obstacle.bounds,
                    )
                {
                    distance =
                        distance.min(
                            distance_hit
                        );
                }
            }

            self.sensors[index] =
                (
                    distance /
                    MAX_SENSOR_DISTANCE
                )
                .clamp(
                    0.0,
                    1.0,
                );
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

    fn observation(
        &self,
    ) -> [f32; OBS_SIZE] {
        let mut observation =
            [0.0; OBS_SIZE];

        observation[..SENSOR_COUNT]
            .copy_from_slice(
                &self.sensors
            );

        let dx =
            self.target.x -
            self.agent.position.x;

        let dz =
            self.target.z -
            self.agent.position.z;

        let distance =
            (
                dx * dx +
                dz * dz
            )
            .sqrt();

        let target_heading =
            dz.atan2(
                dx
            );

        let angle =
            wrap_angle(
                target_heading -
                (
                    self.agent.heading -
                    PI * 0.5
                ),
            );

        observation[32] =
            (
                distance /
                (
                    WORLD_SIZE *
                    1.41421356
                )
            )
            .clamp(
                0.0,
                1.0,
            );

        observation[33] =
            angle.sin();

        observation[34] =
            angle.cos();

        observation[35] =
            (
                self.agent.speed.abs() /
                MAX_SPEED
            )
            .clamp(
                0.0,
                1.0,
            );

        observation[36] =
            if self.collision {
                1.0
            } else {
                0.0
            };

        observation
    }


    // =========================================================================
    // AGENT BUFFER
    // =========================================================================

    fn agent_buffer(
        &self,
    ) -> Vec<f32> {
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

    fn target_buffer(
        &self,
    ) -> Vec<f32> {
        vec![
            self.target.x,
            self.target.y,
            self.target.z,
            TARGET_RADIUS,
        ]
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

    fn obstacle_buffer(
        &self,
    ) -> Vec<f32> {
        let mut output =
            Vec::with_capacity(
                self.obstacles.len() * 8
            );

        for obstacle in &self.obstacles {
            output.extend_from_slice(
                &[
                    obstacle.bounds.center.x,
                    obstacle.bounds.center.y,
                    obstacle.bounds.center.z,

                    obstacle.bounds.half.x,
                    obstacle.bounds.half.y,
                    obstacle.bounds.half.z,

                    obstacle.material as f32,
                    0.0,
                ],
            );
        }

        output
    }


    // =========================================================================
    // PATH BUFFER
    // =========================================================================

    fn path_buffer(
        &self,
    ) -> Vec<f32> {
        let mut output =
            Vec::with_capacity(
                self.path.len() * 3
            );

        for point in &self.path {
            output.extend_from_slice(
                &[
                    point.x,
                    point.y,
                    point.z,
                ],
            );
        }

        output
    }
}


// ============================================================================
// ANGLE NORMALIZATION
// ============================================================================

fn wrap_angle(
    mut angle: f32,
) -> f32 {
    while angle > PI {
        angle -= TWO_PI;
    }

    while angle < -PI {
        angle += TWO_PI;
    }

    angle
}


// ============================================================================
// POINT -> AABB DISTANCE
// ============================================================================

fn point_aabb_distance_xz(
    point: Vec3,
    aabb: &Aabb,
) -> f32 {
    let dx =
        (
            point.x -
            aabb.center.x
        )
        .abs()
        -
        aabb.half.x;

    let dz =
        (
            point.z -
            aabb.center.z
        )
        .abs()
        -
        aabb.half.z;

    let qx =
        dx.max(0.0);

    let qz =
        dz.max(0.0);

    if dx <= 0.0 &&
       dz <= 0.0
    {
        0.0
    } else {
        (
            qx * qx +
            qz * qz
        )
        .sqrt()
    }
}


// ============================================================================
// AABB OVERLAP
// ============================================================================

fn aabb_overlap_expanded(
    a: &Aabb,
    b: &Aabb,
    margin: f32,
) -> bool {
    (
        a.center.x -
        b.center.x
    )
    .abs()
        <=
    a.half.x +
    b.half.x +
    margin

    &&

    (
        a.center.z -
        b.center.z
    )
    .abs()
        <=
    a.half.z +
    b.half.z +
    margin
}


// ============================================================================
// RAY -> AABB
// ============================================================================

fn ray_aabb_distance_xz(
    origin: Vec3,
    direction: Vec3,
    aabb: &Aabb,
) -> Option<f32> {
    let min_x =
        aabb.min_x();

    let max_x =
        aabb.max_x();

    let min_z =
        aabb.min_z();

    let max_z =
        aabb.max_z();

    let mut t_min =
        0.0f32;

    let mut t_max =
        MAX_SENSOR_DISTANCE;

    for (
        origin_component,
        direction_component,
        minimum,
        maximum,
    ) in [
        (
            origin.x,
            direction.x,
            min_x,
            max_x,
        ),
        (
            origin.z,
            direction.z,
            min_z,
            max_z,
        ),
    ] {
        if direction_component.abs() <
           1.0e-8
        {
            if origin_component < minimum ||
               origin_component > maximum
            {
                return None;
            }
        } else {
            let inverse =
                1.0 /
                direction_component;

            let mut t0 =
                (
                    minimum -
                    origin_component
                )
                *
                inverse;

            let mut t1 =
                (
                    maximum -
                    origin_component
                )
                *
                inverse;

            if t0 > t1 {
                std::mem::swap(
                    &mut t0,
                    &mut t1,
                );
            }

            t_min =
                t_min.max(
                    t0
                );

            t_max =
                t_max.min(
                    t1
                );

            if t_max < t_min {
                return None;
            }
        }
    }

    if t_max < 0.0 {
        None
    } else {
        Some(
            t_min.max(0.0)
        )
    }
}


// ============================================================================
// WORLD BOUNDARY SENSOR
// ============================================================================

fn boundary_distance(
    origin: Vec3,
    direction: Vec3,
) -> f32 {
    let mut best =
        MAX_SENSOR_DISTANCE;

    if direction.x > 1.0e-8 {
        best =
            best.min(
                (
                    WORLD_HALF -
                    origin.x
                )
                /
                direction.x
            );
    }

    if direction.x < -1.0e-8 {
        best =
            best.min(
                (
                    -WORLD_HALF -
                    origin.x
                )
                /
                direction.x
            );
    }

    if direction.z > 1.0e-8 {
        best =
            best.min(
                (
                    WORLD_HALF -
                    origin.z
                )
                /
                direction.z
            );
    }

    if direction.z < -1.0e-8 {
        best =
            best.min(
                (
                    -WORLD_HALF -
                    origin.z
                )
                /
                direction.z
            );
    }

    if best.is_finite() &&
       best > 0.0
    {
        best
    } else {
        MAX_SENSOR_DISTANCE
    }
}


// ============================================================================
// WORLD <-> GRID
// ============================================================================

fn world_to_grid(
    point: Vec3,
    grid: usize,
) -> (usize, usize) {
    let x =
        (
            (
                point.x +
                WORLD_HALF
            )
            /
            WORLD_SIZE *
            grid as f32
        )
        .floor()
        .clamp(
            0.0,
            (
                grid - 1
            ) as f32,
        )
        as usize;

    let z =
        (
            (
                point.z +
                WORLD_HALF
            )
            /
            WORLD_SIZE *
            grid as f32
        )
        .floor()
        .clamp(
            0.0,
            (
                grid - 1
            ) as f32,
        )
        as usize;

    (
        x,
        z,
    )
}

fn grid_to_world(
    x: usize,
    z: usize,
    grid: usize,
) -> Vec3 {
    let fx =
        (
            x as f32 +
            0.5
        )
        /
        grid as f32;

    let fz =
        (
            z as f32 +
            0.5
        )
        /
        grid as f32;

    Vec3::new(
        fx * WORLD_SIZE -
            WORLD_HALF,

        0.5,

        fz * WORLD_SIZE -
            WORLD_HALF,
    )
}


// ============================================================================
// NEURAL NETWORK
// ============================================================================
//
// 37 -> 64 -> 64 -> 5
//
// Output:
//
//   Q(FORWARD)
//   Q(LEFT)
//   Q(RIGHT)
//   Q(BRAKE)
//   Q(REVERSE)
//
// ============================================================================

#[derive(Clone)]
struct Network {
    w1: Vec<f32>,
    b1: Vec<f32>,

    w2: Vec<f32>,
    b2: Vec<f32>,

    w3: Vec<f32>,
    b3: Vec<f32>,
}

impl Network {
    fn new(
        rng: &mut Lcg,
    ) -> Self {
        let mut network =
            Self {
                w1: vec![
                    0.0;
                    OBS_SIZE *
                    HIDDEN_1
                ],

                b1: vec![
                    0.0;
                    HIDDEN_1
                ],

                w2: vec![
                    0.0;
                    HIDDEN_1 *
                    HIDDEN_2
                ],

                b2: vec![
                    0.0;
                    HIDDEN_2
                ],

                w3: vec![
                    0.0;
                    HIDDEN_2 *
                    ACTION_COUNT
                ],

                b3: vec![
                    0.0;
                    ACTION_COUNT
                ],
            };

        let limit_1 =
            (
                6.0 /
                (
                    OBS_SIZE +
                    HIDDEN_1
                ) as f32
            )
            .sqrt();

        let limit_2 =
            (
                6.0 /
                (
                    HIDDEN_1 +
                    HIDDEN_2
                ) as f32
            )
            .sqrt();

        let limit_3 =
            (
                6.0 /
                (
                    HIDDEN_2 +
                    ACTION_COUNT
                ) as f32
            )
            .sqrt();

        for weight in
            &mut network.w1
        {
            *weight =
                rng.range_f32(
                    -limit_1,
                    limit_1,
                );
        }

        for weight in
            &mut network.w2
        {
            *weight =
                rng.range_f32(
                    -limit_2,
                    limit_2,
                );
        }

        for weight in
            &mut network.w3
        {
            *weight =
                rng.range_f32(
                    -limit_3,
                    limit_3,
                );
        }

        network
    }

    fn parameter_count()
        -> usize
    {
        OBS_SIZE * HIDDEN_1
            + HIDDEN_1
            + HIDDEN_1 * HIDDEN_2
            + HIDDEN_2
            + HIDDEN_2 * ACTION_COUNT
            + ACTION_COUNT
    }


    // =========================================================================
    // FORWARD
    // =========================================================================

    fn forward(
        &self,
        input: &[f32; OBS_SIZE],
    )
        -> (
            [f32; HIDDEN_1],
            [f32; HIDDEN_2],
            [f32; ACTION_COUNT],
        )
    {
        let mut hidden_1 =
            [0.0; HIDDEN_1];

        let mut hidden_2 =
            [0.0; HIDDEN_2];

        let mut q =
            [0.0; ACTION_COUNT];


        // ---------------------------------------------------------------------
        // LAYER 1
        // ---------------------------------------------------------------------

        for output in
            0..HIDDEN_1
        {
            let mut value =
                self.b1[output];

            for input_index in
                0..OBS_SIZE
            {
                value +=
                    input[input_index] *
                    self.w1[
                        input_index *
                        HIDDEN_1 +
                        output
                    ];
            }

            hidden_1[output] =
                value.max(0.0);
        }


        // ---------------------------------------------------------------------
        // LAYER 2
        // ---------------------------------------------------------------------

        for output in
            0..HIDDEN_2
        {
            let mut value =
                self.b2[output];

            for input_index in
                0..HIDDEN_1
            {
                value +=
                    hidden_1[input_index] *
                    self.w2[
                        input_index *
                        HIDDEN_2 +
                        output
                    ];
            }

            hidden_2[output] =
                value.max(0.0);
        }


        // ---------------------------------------------------------------------
        // OUTPUT
        // ---------------------------------------------------------------------

        for output in
            0..ACTION_COUNT
        {
            let mut value =
                self.b3[output];

            for input_index in
                0..HIDDEN_2
            {
                value +=
                    hidden_2[input_index] *
                    self.w3[
                        input_index *
                        ACTION_COUNT +
                        output
                    ];
            }

            q[output] =
                value;
        }

        (
            hidden_1,
            hidden_2,
            q,
        )
    }


    fn q_values(
        &self,
        input: &[f32; OBS_SIZE],
    ) -> [f32; ACTION_COUNT] {
        self.forward(
            input
        ).2
    }


    // =========================================================================
    // TRAIN ONE SAMPLE
    // =========================================================================

    fn train_sample(
        &mut self,
        target_network: &Network,
        transition: &Transition,
        learning_rate: f32,
        gamma: f32,
    ) -> f32 {
        let (
            hidden_1,
            hidden_2,
            q_values,
        ) =
            self.forward(
                &transition.state
            );

        let next_q =
            target_network.q_values(
                &transition.next_state
            );

        let mut max_next =
            next_q[0];

        for value in
            &next_q[1..]
        {
            max_next =
                max_next.max(
                    *value
                );
        }

        let target =
            transition.reward +
            if transition.done {
                0.0
            } else {
                gamma * max_next
            };

        let action =
            (
                transition.action as usize
            )
            .min(
                ACTION_COUNT - 1
            );

        let error =
            q_values[action] -
            target;

        let mut delta_3 =
            [0.0; ACTION_COUNT];

        delta_3[action] =
            error.clamp(
                -5.0,
                5.0,
            );


        // ---------------------------------------------------------------------
        // DELTA LAYER 2
        // ---------------------------------------------------------------------

        let mut delta_2 =
            [0.0; HIDDEN_2];

        for hidden in
            0..HIDDEN_2
        {
            let mut value =
                0.0;

            for output in
                0..ACTION_COUNT
            {
                value +=
                    self.w3[
                        hidden *
                        ACTION_COUNT +
                        output
                    ]
                    *
                    delta_3[
                        output
                    ];
            }

            delta_2[hidden] =
                if hidden_2[hidden] > 0.0 {
                    value
                } else {
                    0.0
                };
        }


        // ---------------------------------------------------------------------
        // DELTA LAYER 1
        // ---------------------------------------------------------------------

        let mut delta_1 =
            [0.0; HIDDEN_1];

        for hidden in
            0..HIDDEN_1
        {
            let mut value =
                0.0;

            for next_hidden in
                0..HIDDEN_2
            {
                value +=
                    self.w2[
                        hidden *
                        HIDDEN_2 +
                        next_hidden
                    ]
                    *
                    delta_2[
                        next_hidden
                    ];
            }

            delta_1[hidden] =
                if hidden_1[hidden] > 0.0 {
                    value
                } else {
                    0.0
                };
        }

        let step =
            learning_rate.clamp(
                1.0e-6,
                1.0,
            );


        // ---------------------------------------------------------------------
        // UPDATE LAYER 3
        // ---------------------------------------------------------------------

        for hidden in
            0..HIDDEN_2
        {
            for output in
                0..ACTION_COUNT
            {
                let gradient =
                    (
                        hidden_2[hidden] *
                        delta_3[output]
                    )
                    .clamp(
                        -5.0,
                        5.0,
                    );

                self.w3[
                    hidden *
                    ACTION_COUNT +
                    output
                ] -=
                    step * gradient;
            }
        }

        for output in
            0..ACTION_COUNT
        {
            self.b3[output] -=
                step *
                delta_3[output];
        }


        // ---------------------------------------------------------------------
        // UPDATE LAYER 2
        // ---------------------------------------------------------------------

        for hidden in
            0..HIDDEN_1
        {
            for next_hidden in
                0..HIDDEN_2
            {
                let gradient =
                    (
                        hidden_1[hidden] *
                        delta_2[next_hidden]
                    )
                    .clamp(
                        -5.0,
                        5.0,
                    );

                self.w2[
                    hidden *
                    HIDDEN_2 +
                    next_hidden
                ] -=
                    step * gradient;
            }
        }

        for hidden in
            0..HIDDEN_2
        {
            self.b2[hidden] -=
                step *
                delta_2[hidden];
        }


        // ---------------------------------------------------------------------
        // UPDATE LAYER 1
        // ---------------------------------------------------------------------

        for input_index in
            0..OBS_SIZE
        {
            for hidden in
                0..HIDDEN_1
            {
                let gradient =
                    (
                        transition.state[
                            input_index
                        ] *
                        delta_1[hidden]
                    )
                    .clamp(
                        -5.0,
                        5.0,
                    );

                self.w1[
                    input_index *
                    HIDDEN_1 +
                    hidden
                ] -=
                    step * gradient;
            }
        }

        for hidden in
            0..HIDDEN_1
        {
            self.b1[hidden] -=
                step *
                delta_1[hidden];
        }


        // ---------------------------------------------------------------------
        // MSE
        // ---------------------------------------------------------------------

        0.5 *
        error *
        error
    }


    // =========================================================================
    // FLAT PARAMETER STREAM
    // =========================================================================

    fn parameters(
        &self,
    ) -> impl Iterator<Item = f32> + '_ {
        self.w1
            .iter()
            .chain(
                self.b1.iter()
            )
            .chain(
                self.w2.iter()
            )
            .chain(
                self.b2.iter()
            )
            .chain(
                self.w3.iter()
            )
            .chain(
                self.b3.iter()
            )
            .copied()
    }

    fn load_parameters(
        &mut self,
        data: &[f32],
    ) -> bool {
        if data.len() !=
           Self::parameter_count()
        {
            return false;
        }

        let mut offset =
            0usize;

        let copy_block =
            |target: &mut [f32],
             offset: &mut usize,
             data: &[f32]| {
                let end =
                    *offset +
                    target.len();

                target.copy_from_slice(
                    &data[
                        *offset..end
                    ],
                );

                *offset = end;
            };

        copy_block(
            &mut self.w1,
            &mut offset,
            data,
        );

        copy_block(
            &mut self.b1,
            &mut offset,
            data,
        );

        copy_block(
            &mut self.w2,
            &mut offset,
            data,
        );

        copy_block(
            &mut self.b2,
            &mut offset,
            data,
        );

        copy_block(
            &mut self.w3,
            &mut offset,
            data,
        );

        copy_block(
            &mut self.b3,
            &mut offset,
            data,
        );

        true
    }
}


// ============================================================================
// ACTION ARGMAX
// ============================================================================

fn argmax(
    values: &[f32],
) -> usize {
    if values.is_empty() {
        return 0;
    }

    let mut best =
        0usize;

    for index in
        1..values.len()
    {
        if values[index] >
           values[best]
        {
            best = index;
        }
    }

    best
}


// ============================================================================
// WASM ENGINE
// ============================================================================

#[wasm_bindgen]
pub struct RayTracerEngine {
    environment: Environment,

    network: Network,
    target_network: Network,

    replay: ReplayBuffer,

    rng: Lcg,


    // -------------------------------------------------------------------------
    // TRAINING PARAMETERS
    // -------------------------------------------------------------------------

    learning_rate: f32,
    gamma: f32,

    epsilon: f32,
    epsilon_min: f32,
    epsilon_decay: f32,

    batch_size: usize,
    target_update: u64,


    // -------------------------------------------------------------------------
    // TRAINING METRICS
    // -------------------------------------------------------------------------

    training_steps: u64,
    completed_episodes: u64,

    success_episodes: u64,
    collision_episodes: u64,

    last_episode_reward: f32,
    last_episode_steps: u32,

    last_loss: f32,
    last_reward: f32,

    last_action: u32,

    // Terminal event from the most recently completed episode.
    // Kept separately from Environment so training can reset immediately
    // without losing the outcome that just happened.
    last_terminal_done: bool,
    last_terminal_success: bool,
    last_terminal_collision: bool,
    last_terminal_distance: f32,
}


// ============================================================================
// WASM API
// ============================================================================

#[wasm_bindgen]
impl RayTracerEngine {


    // =========================================================================
    // CREATE
    // =========================================================================

    #[wasm_bindgen(constructor)]
    pub fn new() -> RayTracerEngine {
        let mut rng =
            Lcg::new(
                0x5241595452435F41
            );

        let network =
            Network::new(
                &mut rng
            );

        let target_network =
            network.clone();

        Self {
            environment:
                Environment::new(
                    1,
                    1,
                    4,
                    DEFAULT_MAX_STEPS,
                ),

            network,

            target_network,

            replay:
                ReplayBuffer::new(
                    DEFAULT_REPLAY_CAPACITY,
                ),

            rng,

            learning_rate:
                0.001,

            gamma:
                0.99,

            epsilon:
                1.0,

            epsilon_min:
                0.05,

            epsilon_decay:
                0.995,

            batch_size:
                DEFAULT_BATCH_SIZE,

            target_update:
                DEFAULT_TARGET_UPDATE,

            training_steps:
                0,

            completed_episodes:
                0,

            success_episodes:
                0,

            collision_episodes:
                0,

            last_episode_reward:
                0.0,

            last_episode_steps:
                0,

            last_loss:
                0.0,

            last_reward:
                0.0,

            last_action:
                0,

            last_terminal_done:
                false,

            last_terminal_success:
                false,

            last_terminal_collision:
                false,

            last_terminal_distance:
                0.0,
        }
    }


    // =========================================================================
    // ENGINE INFO
    // =========================================================================

    pub fn version(
        &self,
    ) -> String {
        "RAYTRC.AI.NAV/1.1.0"
            .to_string()
    }

    pub fn backend(
        &self,
    ) -> String {
        "RUST + WASM + DQN + 32-RAY SENSORS"
            .to_string()
    }

    pub fn sensor_count(
        &self,
    ) -> u32 {
        SENSOR_COUNT as u32
    }

    pub fn observation_size(
        &self,
    ) -> u32 {
        OBS_SIZE as u32
    }

    pub fn action_count(
        &self,
    ) -> u32 {
        ACTION_COUNT as u32
    }

    pub fn model_parameter_count(
        &self,
    ) -> u32 {
        Network::parameter_count()
            as u32
    }


    // =========================================================================
    // ACTION NAMES
    // =========================================================================

    pub fn action_name(
        &self,
        action: u32,
    ) -> String {
        match action {
            0 => "FORWARD",
            1 => "LEFT",
            2 => "RIGHT",
            3 => "BRAKE",
            4 => "REVERSE",
            _ => "UNKNOWN",
        }
        .to_string()
    }


    // =========================================================================
    // RESET AI
    // =========================================================================

    pub fn reset_ai(
        &mut self,
    ) {
        self.network =
            Network::new(
                &mut self.rng
            );

        self.target_network =
            self.network.clone();

        self.replay.clear();

        self.epsilon =
            1.0;

        self.training_steps =
            0;

        self.completed_episodes =
            0;

        self.success_episodes =
            0;

        self.collision_episodes =
            0;

        self.last_episode_reward =
            0.0;

        self.last_episode_steps =
            0;

        self.last_loss =
            0.0;

        self.last_reward =
            0.0;

        self.last_action =
            0;

        self.last_terminal_done =
            false;

        self.last_terminal_success =
            false;

        self.last_terminal_collision =
            false;

        self.last_terminal_distance =
            0.0;
    }


    // =========================================================================
    // RESET ENVIRONMENT
    // =========================================================================

    pub fn reset_environment(
        &mut self,
        seed: u32,
    ) {
        self.environment.reset(
            seed
        );

        self.last_action =
            0;

        self.last_reward =
            0.0;

        self.last_terminal_done =
            false;

        self.last_terminal_success =
            false;

        self.last_terminal_collision =
            false;

        self.last_terminal_distance =
            self.environment.last_distance;
    }


    // =========================================================================
    // COLLISION RECOVERY / SAME MAP
    // =========================================================================
    //
    // Used by the foreground autonomous preview. A collision is a terminal
    // training event, but the visual RUN mode should not freeze or generate a
    // different map. Restart the agent on the SAME map instead.
    //
    // This deliberately preserves:
    //   - seed
    //   - obstacles
    //   - target
    //   - difficulty
    //   - configured obstacle count
    //
    // and only resets the agent/episode-local state.
    // =========================================================================

    pub fn recover_from_collision(
        &mut self,
    ) {
        if !self.environment.done ||
           !self.environment.collision
        {
            return;
        }

        let dx =
            self.environment.target.x -
            self.environment.start.x;

        let dz =
            self.environment.target.z -
            self.environment.start.z;

        let target_heading =
            dz.atan2(dx);

        self.environment.step =
            0;

        self.environment.episode_reward =
            0.0;

        self.environment.done =
            false;

        self.environment.success =
            false;

        self.environment.collision =
            false;

        self.environment.agent =
            Agent {
                position: Vec3::new(
                    self.environment.start.x,
                    0.5,
                    self.environment.start.z,
                ),

                heading:
                    target_heading -
                    PI * 0.5,

                speed: 0.0,
            };

        self.environment.path.clear();

        self.environment.path.push(
            self.environment.agent.position,
        );

        self.environment.last_distance =
            self.environment.agent.position
                .distance_xz(
                    self.environment.target,
                );

        self.environment.update_sensors();

        self.last_action =
            0;

        self.last_reward =
            0.0;

        self.last_terminal_done =
            false;

        self.last_terminal_success =
            false;

        self.last_terminal_collision =
            false;

        self.last_terminal_distance =
            self.environment.last_distance;
    }


    // =========================================================================
    // HARD RESET
    // =========================================================================

    pub fn hard_reset(
        &mut self,
    ) {
        self.reset_ai();

        self.environment.difficulty =
            1;

        self.environment.obstacle_count =
            4;

        self.environment.max_steps =
            DEFAULT_MAX_STEPS;

        let seed =
            self.rng
                .next_u32()
                .max(1);

        self.environment.reset(
            seed
        );
    }


    // =========================================================================
    // ENVIRONMENT CONFIG
    // =========================================================================

    pub fn set_difficulty(
        &mut self,
        difficulty: u32,
    ) {
        self.environment.difficulty =
            difficulty.clamp(
                1,
                10,
            );

        let seed =
            self.environment.seed;

        self.environment.reset(
            seed
        );
    }

    pub fn set_obstacle_count(
        &mut self,
        count: u32,
    ) {
        self.environment.obstacle_count =
            count.clamp(
                1,
                64,
            );

        let seed =
            self.environment.seed;

        self.environment.reset(
            seed
        );
    }

    pub fn set_episode_steps(
        &mut self,
        steps: u32,
    ) {
        self.environment.max_steps =
            steps.clamp(
                50,
                5000,
            );
    }

    pub fn difficulty(
        &self,
    ) -> u32 {
        self.environment.difficulty
    }

    pub fn configured_obstacle_count(
        &self,
    ) -> u32 {
        self.environment.obstacle_count
    }

    pub fn episode_limit(
        &self,
    ) -> u32 {
        self.environment.max_steps
    }


    // =========================================================================
    // AI CONFIG
    // =========================================================================

    pub fn set_learning_rate(
        &mut self,
        value: f32,
    ) {
        self.learning_rate =
            value.clamp(
                0.00001,
                0.1,
            );
    }

    pub fn set_gamma(
        &mut self,
        value: f32,
    ) {
        self.gamma =
            value.clamp(
                0.80,
                0.9999,
            );
    }

    pub fn set_batch_size(
        &mut self,
        value: u32,
    ) {
        self.batch_size =
            (value as usize)
                .clamp(
                    1,
                    256,
                );
    }

    pub fn set_epsilon(
        &mut self,
        value: f32,
    ) {
        self.epsilon =
            value.clamp(
                0.0,
                1.0,
            );
    }

    pub fn set_epsilon_min(
        &mut self,
        value: f32,
    ) {
        self.epsilon_min =
            value.clamp(
                0.0,
                1.0,
            );

        if self.epsilon <
           self.epsilon_min
        {
            self.epsilon =
                self.epsilon_min;
        }
    }

    pub fn set_epsilon_decay(
        &mut self,
        value: f32,
    ) {
        self.epsilon_decay =
            value.clamp(
                0.9,
                0.999999,
            );
    }

    pub fn set_target_update(
        &mut self,
        value: u32,
    ) {
        self.target_update =
            (value as u64)
                .clamp(
                    1,
                    100_000,
                );
    }


    // =========================================================================
    // CURRENT STATE
    // =========================================================================

    pub fn current_seed(
        &self,
    ) -> u32 {
        self.environment.seed
    }

    pub fn current_step(
        &self,
    ) -> u32 {
        self.environment.step
    }

    pub fn current_episode(
        &self,
    ) -> u64 {
        self.completed_episodes
    }

    pub fn training_step_count(
        &self,
    ) -> u64 {
        self.training_steps
    }

    pub fn is_done(
        &self,
    ) -> bool {
        self.environment.done
    }

    pub fn is_success(
        &self,
    ) -> bool {
        self.environment.success
    }

    pub fn is_collision(
        &self,
    ) -> bool {
        self.environment.collision
    }

    pub fn last_terminal_done(
        &self,
    ) -> bool {
        self.last_terminal_done
    }

    pub fn last_terminal_success(
        &self,
    ) -> bool {
        self.last_terminal_success
    }

    pub fn last_terminal_collision(
        &self,
    ) -> bool {
        self.last_terminal_collision
    }

    pub fn last_terminal_distance(
        &self,
    ) -> f32 {
        self.last_terminal_distance
    }

    pub fn epsilon(
        &self,
    ) -> f32 {
        self.epsilon
    }

    pub fn learning_rate(
        &self,
    ) -> f32 {
        self.learning_rate
    }

    pub fn gamma(
        &self,
    ) -> f32 {
        self.gamma
    }

    pub fn last_loss(
        &self,
    ) -> f32 {
        self.last_loss
    }

    pub fn last_reward(
        &self,
    ) -> f32 {
        self.last_reward
    }

    pub fn last_episode_reward(
        &self,
    ) -> f32 {
        self.last_episode_reward
    }

    pub fn last_episode_steps(
        &self,
    ) -> u32 {
        self.last_episode_steps
    }

    pub fn replay_size(
        &self,
    ) -> u32 {
        self.replay.len() as u32
    }


    // =========================================================================
    // RAW ENVIRONMENT BUFFERS
    // =========================================================================

    pub fn get_observation(
        &self,
    ) -> Vec<f32> {
        self.environment
            .observation()
            .to_vec()
    }

    pub fn get_sensor_buffer(
        &self,
    ) -> Vec<f32> {
        self.environment
            .sensors
            .to_vec()
    }

    pub fn get_agent_buffer(
        &self,
    ) -> Vec<f32> {
        self.environment
            .agent_buffer()
    }

    pub fn get_target_buffer(
        &self,
    ) -> Vec<f32> {
        self.environment
            .target_buffer()
    }

    pub fn get_obstacle_buffer(
        &self,
    ) -> Vec<f32> {
        self.environment
            .obstacle_buffer()
    }

    pub fn get_path_buffer(
        &self,
    ) -> Vec<f32> {
        self.environment
            .path_buffer()
    }

    pub fn obstacle_count(
        &self,
    ) -> u32 {
        self.environment
            .obstacles
            .len() as u32
    }


    // =========================================================================
    // AI PREDICTION
    // =========================================================================

    pub fn predict_action(
        &self,
    ) -> u32 {
        let observation =
            self.environment
                .observation();

        argmax(
            &self.network.q_values(
                &observation
            )
        ) as u32
    }

    pub fn q_values(
        &self,
    ) -> Vec<f32> {
        self.network
            .q_values(
                &self.environment
                    .observation()
            )
            .to_vec()
    }


    // =========================================================================
    // MANUAL STEP
    // =========================================================================
    //
    // Return:
    //
    //   [reward,
    //    done,
    //    success,
    //    collision,
    //    distance,
    //    action]
    //
    // =========================================================================

    pub fn step(
        &mut self,
        action: u32,
    ) -> Vec<f32> {
        let action =
            (action as usize)
                .min(
                    ACTION_COUNT - 1
                );

        /*
         * TERMINAL STATES ARE STICKY.
         *
         * A manual/preview step after SUCCESS or COLLISION must never:
         *   - move the agent,
         *   - generate a second episode,
         *   - increment completed_episodes again,
         *   - silently create a new map.
         *
         * The UI explicitly calls reset_environment() / RUN to begin
         * the next episode.
         */
        if self.environment.done {
            self.last_action =
                action as u32;

            self.last_reward =
                0.0;

            return vec![
                0.0,

                1.0,

                if self.environment.success {
                    1.0
                } else {
                    0.0
                },

                if self.environment.collision {
                    1.0
                } else {
                    0.0
                },

                self.environment.last_distance,

                action as f32,
            ];
        }

        let result =
            self.environment.step(
                action
            );

        self.last_action =
            action as u32;

        self.last_reward =
            result.reward;

        if result.done {
            self.record_terminal_result(
                result
            );

            self.finish_episode(
                result
            );
        }

        vec![
            result.reward,

            if result.done {
                1.0
            } else {
                0.0
            },

            if result.success {
                1.0
            } else {
                0.0
            },

            if result.collision {
                1.0
            } else {
                0.0
            },

            result.distance_to_target,

            action as f32,
        ]
    }


    // =========================================================================
    // TRAIN ONE STEP
    // =========================================================================

    pub fn train_step(
        &mut self,
    ) -> Vec<f32> {
        if self.environment.done {
            self.environment.reset(0);
        }

        let state =
            self.environment
                .observation();

        let action =
            self.select_action(
                &state
            );

        let result =
            self.environment.step(
                action
            );

        let next_state =
            self.environment
                .observation();

        self.replay.push(
            Transition {
                state,
                action: action as u8,
                reward: result.reward,
                next_state,
                done: result.done,
            }
        );

        self.last_reward =
            result.reward;

        self.last_action =
            action as u32;

        self.training_steps +=
            1;


        // ---------------------------------------------------------------------
        // LEARN
        // ---------------------------------------------------------------------

        if self.replay.len() >=
           self.batch_size
        {
            self.last_loss =
                self.train_batch();
        }


        // ---------------------------------------------------------------------
        // EXPLORATION DECAY
        // ---------------------------------------------------------------------

        self.epsilon =
            (
                self.epsilon *
                self.epsilon_decay
            )
            .max(
                self.epsilon_min
            );


        // ---------------------------------------------------------------------
        // TARGET NETWORK
        // ---------------------------------------------------------------------

        if self.training_steps %
            self.target_update
            ==
            0
        {
            self.target_network =
                self.network.clone();
        }


        // ---------------------------------------------------------------------
        // EPISODE END
        // ---------------------------------------------------------------------

        if result.done {
            self.record_terminal_result(
                result
            );

            self.finish_episode(
                result
            );

            /*
             * TRAINING episodes automatically continue on the next
             * environment. This is intentionally different from the
             * manual/preview step() API, which holds terminal maps.
             */
            self.environment.reset(0);
        }

        vec![
            self.training_steps as f32,
            self.completed_episodes as f32,
            result.reward,
            self.last_loss,
            self.epsilon,
            action as f32,

            if result.done {
                1.0
            } else {
                0.0
            },

            if result.success {
                1.0
            } else {
                0.0
            },

            if result.collision {
                1.0
            } else {
                0.0
            },
        ]
    }


    // =========================================================================
    // TRAIN N STEPS
    // =========================================================================

    pub fn train_steps(
        &mut self,
        count: u32,
    ) -> Vec<f32> {
        let count =
            count.min(
                1_000_000
            );

        for _ in 0..count {
            self.train_step();
        }

        self.metrics()
    }


    // =========================================================================
    // TRAIN ONE EPISODE
    // =========================================================================

    pub fn train_episode(
        &mut self,
    ) -> Vec<f32> {
        let starting_episode =
            self.completed_episodes;

        let mut guard =
            0usize;

        while self.completed_episodes ==
              starting_episode
              &&
              guard <
              self.environment.max_steps
                  as usize
                  + 2
        {
            self.train_step();

            guard += 1;
        }

        self.metrics()
    }


    // =========================================================================
    // TRAIN MULTIPLE EPISODES
    // =========================================================================

    pub fn train_episodes(
        &mut self,
        count: u32,
    ) -> Vec<f32> {
        let count =
            count.min(
                10_000
            );

        let target =
            self.completed_episodes +
            count as u64;

        let max_work =
            count as usize *
            (
                self.environment.max_steps
                    as usize
                +
                2
            );

        let mut guard =
            0usize;

        while self.completed_episodes <
              target
              &&
              guard <
              max_work
        {
            self.train_step();

            guard += 1;
        }

        self.metrics()
    }


    // =========================================================================
    // EVALUATION
    // =========================================================================

    pub fn evaluate_episodes(
        &self,
        count: u32,
        base_seed: u32,
    ) -> Vec<f32> {
        let total =
            count
                .max(1)
                .min(10_000);

        let mut evaluation =
            self.environment.clone();

        let mut successes =
            0u32;

        let mut collisions =
            0u32;

        let mut total_steps =
            0u64;

        let mut total_reward =
            0.0f64;

        for episode in
            0..total
        {
            let seed =
                base_seed
                    .wrapping_add(
                        episode
                            .wrapping_mul(
                                7919
                            )
                    )
                    .max(1);

            evaluation.reset(
                seed
            );

            let mut steps =
                0u32;

            loop {
                let observation =
                    evaluation
                        .observation();

                let action =
                    argmax(
                        &self.network
                            .q_values(
                                &observation
                            )
                    );

                let result =
                    evaluation.step(
                        action
                    );

                steps += 1;

                total_reward +=
                    result.reward as f64;

                if result.done {
                    if result.success {
                        successes += 1;
                    }

                    if result.collision {
                        collisions += 1;
                    }

                    break;
                }

                if steps >=
                   evaluation.max_steps
                {
                    break;
                }
            }

            total_steps +=
                steps as u64;
        }

        vec![
            successes as f32 /
                total as f32,

            collisions as f32 /
                total as f32,

            total_steps as f32 /
                total as f32,

            (
                total_reward /
                total as f64
            ) as f32,
        ]
    }


    // =========================================================================
    // METRICS
    // =========================================================================

    pub fn metrics(
        &self,
    ) -> Vec<f32> {
        let episode_divisor =
            self.completed_episodes
                .max(1)
                as f32;

        vec![
            self.completed_episodes as f32,
            self.training_steps as f32,
            self.last_reward,
            self.last_loss,
            self.epsilon,

            self.success_episodes as f32 /
                episode_divisor,

            self.collision_episodes as f32 /
                episode_divisor,

            self.environment.episode_reward,
            self.environment.step as f32,
            self.environment.difficulty as f32,
            self.environment.obstacles.len() as f32,
            self.environment.seed as f32,
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


    // =========================================================================
    // STATUS JSON
    // =========================================================================
    //
    // IMPORTANT:
    //
    // Do NOT use:
    //
    //   format!(concat!("{", ... "}"), ...)
    //
    // because `{` and `}` are interpreted by Rust's format parser.
    //
    // This implementation builds valid JSON directly.
    //
    // =========================================================================

    pub fn status_json(
        &self,
    ) -> String {
        let completed =
            self.completed_episodes.max(
                1
            ) as f32;

        let status =
            if self.environment.done {
                if self.environment.success {
                    "SUCCESS"
                } else if self.environment.collision {
                    "COLLISION"
                } else {
                    "DONE"
                }
            } else {
                "RUNNING"
            };

        let action =
            self.action_name(
                self.last_action
            );

        let mut output =
            String::with_capacity(
                2048
            );

        output.push('{');

        output.push_str(
            "\"version\":\"",
        );

        output.push_str(
            &json_escape(
                &self.version()
            ),
        );

        output.push_str(
            "\",",
        );

        output.push_str(
            "\"backend\":\"",
        );

        output.push_str(
            &json_escape(
                &self.backend()
            ),
        );

        output.push_str(
            "\",",
        );

        output.push_str(
            "\"status\":\"",
        );

        output.push_str(status);

        output.push_str(
            "\",",
        );

        output.push_str(
            "\"seed\":",
        );

        output.push_str(
            &self.environment
                .seed
                .to_string(),
        );

        output.push(',');

        output.push_str(
            "\"difficulty\":",
        );

        output.push_str(
            &self.environment
                .difficulty
                .to_string(),
        );

        output.push(',');

        output.push_str(
            "\"obstacles\":",
        );

        output.push_str(
            &self.environment
                .obstacles
                .len()
                .to_string(),
        );

        output.push(',');

        output.push_str(
            "\"configuredObstacles\":",
        );

        output.push_str(
            &self.environment
                .obstacle_count
                .to_string(),
        );

        output.push(',');

        output.push_str(
            "\"episode\":",
        );

        output.push_str(
            &self.completed_episodes
                .to_string(),
        );

        output.push(',');

        output.push_str(
            "\"step\":",
        );

        output.push_str(
            &self.environment
                .step
                .to_string(),
        );

        output.push(',');

        output.push_str(
            "\"trainingSteps\":",
        );

        output.push_str(
            &self.training_steps
                .to_string(),
        );

        output.push(',');

        output.push_str(
            "\"reward\":",
        );

        push_json_f32(
            &mut output,
            self.last_reward,
        );

        output.push(',');

        output.push_str(
            "\"episodeReward\":",
        );

        push_json_f32(
            &mut output,
            self.last_episode_reward,
        );

        output.push(',');

        output.push_str(
            "\"loss\":",
        );

        push_json_f32(
            &mut output,
            self.last_loss,
        );

        output.push(',');

        output.push_str(
            "\"epsilon\":",
        );

        push_json_f32(
            &mut output,
            self.epsilon,
        );

        output.push(',');

        output.push_str(
            "\"successRate\":",
        );

        push_json_f32(
            &mut output,
            self.success_episodes as f32 /
                completed,
        );

        output.push(',');

        output.push_str(
            "\"collisionRate\":",
        );

        push_json_f32(
            &mut output,
            self.collision_episodes as f32 /
                completed,
        );

        output.push(',');

        output.push_str(
            "\"avgEpisodeSteps\":",
        );

        let average_steps =
            if self.completed_episodes == 0 {
                0.0
            } else {
                self.last_episode_steps as f32
            };

        push_json_f32(
            &mut output,
            average_steps,
        );

        output.push(',');

        output.push_str(
            "\"replay\":",
        );

        output.push_str(
            &self.replay.len()
                .to_string(),
        );

        output.push(',');

        output.push_str(
            "\"modelParameters\":",
        );

        output.push_str(
            &Network::parameter_count()
                .to_string(),
        );

        output.push(',');

        output.push_str(
            "\"sensorCount\":",
        );

        output.push_str(
            &SENSOR_COUNT.to_string(),
        );

        output.push(',');

        output.push_str(
            "\"observationSize\":",
        );

        output.push_str(
            &OBS_SIZE.to_string(),
        );

        output.push(',');

        output.push_str(
            "\"actionCount\":",
        );

        output.push_str(
            &ACTION_COUNT.to_string(),
        );

        output.push(',');

        output.push_str(
            "\"done\":",
        );

        output.push_str(
            if self.environment.done {
                "true"
            } else {
                "false"
            },
        );

        output.push(',');

        output.push_str(
            "\"success\":",
        );

        output.push_str(
            if self.environment.success {
                "true"
            } else {
                "false"
            },
        );

        output.push(',');

        output.push_str(
            "\"collision\":",
        );

        output.push_str(
            if self.environment.collision {
                "true"
            } else {
                "false"
            },
        );

        output.push(',');

        output.push_str(
            "\"action\":\"",
        );

        output.push_str(
            &json_escape(
                &action
            ),
        );

        output.push_str(
            "\",",
        );

        output.push_str(
            "\"learningRate\":",
        );

        push_json_f32(
            &mut output,
            self.learning_rate,
        );

        output.push(',');

        output.push_str(
            "\"gamma\":",
        );

        push_json_f32(
            &mut output,
            self.gamma,
        );

        output.push(',');

        output.push_str(
            "\"batchSize\":",
        );

        output.push_str(
            &self.batch_size
                .to_string(),
        );

        output.push(',');

        output.push_str(
            "\"targetUpdate\":",
        );

        output.push_str(
            &self.target_update
                .to_string(),
        );

        output.push('}');

        output
    }


    // =========================================================================
    // EXPORT MODEL
    // =========================================================================
    //
    // Custom text format.
    //
    // No serde.
    // No extra dependency.
    //
    // Network weights are represented as hexadecimal f32 bit patterns.
    //
    // =========================================================================

    pub fn export_model(
        &self,
    ) -> String {
        let mut output =
            String::new();

        let _ =
            write!(
                output,
                "RAYTRC_NAV_MODEL_V1|{}|{}|{}|{}|{}|{}|{}|",
                self.epsilon,
                self.learning_rate,
                self.gamma,
                self.training_steps,
                self.completed_episodes,
                self.environment.difficulty,
                self.environment.obstacle_count,
            );

        for parameter in
            self.network.parameters()
        {
            let _ =
                write!(
                    output,
                    "{:08X}",
                    parameter.to_bits()
                );
        }

        output
    }


    // =========================================================================
    // IMPORT MODEL
    // =========================================================================

    pub fn import_model(
        &mut self,
        data: String,
    ) -> bool {
        let parts:
            Vec<&str> =
            data.splitn(
                9,
                '|',
            )
            .collect();

        if parts.len() != 9 {
            return false;
        }

        if parts[0] !=
           "RAYTRC_NAV_MODEL_V1"
        {
            return false;
        }

        let epsilon =
            match parts[1]
                .parse::<f32>()
            {
                Ok(value) =>
                    value,

                Err(_) =>
                    return false,
            };

        let learning_rate =
            match parts[2]
                .parse::<f32>()
            {
                Ok(value) =>
                    value,

                Err(_) =>
                    return false,
            };

        let gamma =
            match parts[3]
                .parse::<f32>()
            {
                Ok(value) =>
                    value,

                Err(_) =>
                    return false,
            };

        let training_steps =
            match parts[4]
                .parse::<u64>()
            {
                Ok(value) =>
                    value,

                Err(_) =>
                    return false,
            };

        let completed_episodes =
            match parts[5]
                .parse::<u64>()
            {
                Ok(value) =>
                    value,

                Err(_) =>
                    return false,
            };

        let difficulty =
            match parts[6]
                .parse::<u32>()
            {
                Ok(value) =>
                    value.clamp(
                        1,
                        10,
                    ),

                Err(_) =>
                    return false,
            };

        let obstacle_count =
            match parts[7]
                .parse::<u32>()
            {
                Ok(value) =>
                    value.clamp(
                        1,
                        64,
                    ),

                Err(_) =>
                    return false,
            };

        let encoded =
            parts[8].trim();

        let expected_length =
            Network::parameter_count() *
            8;

        if encoded.len() !=
           expected_length
        {
            return false;
        }

        let mut parameters =
            Vec::<f32>::with_capacity(
                Network::parameter_count()
            );

        for chunk in
            encoded.as_bytes()
                .chunks_exact(8)
        {
            let text =
                match std::str::from_utf8(
                    chunk
                )
                {
                    Ok(value) =>
                        value,

                    Err(_) =>
                        return false,
                };

            let bits =
                match u32::from_str_radix(
                    text,
                    16,
                )
                {
                    Ok(value) =>
                        value,

                    Err(_) =>
                        return false,
                };

            let value =
                f32::from_bits(
                    bits
                );

            if !value.is_finite() {
                return false;
            }

            parameters.push(
                value
            );
        }

        if !self.network
            .load_parameters(
                &parameters
            )
        {
            return false;
        }

        self.target_network =
            self.network.clone();

        self.replay.clear();

        self.epsilon =
            epsilon.clamp(
                0.0,
                1.0,
            );

        self.learning_rate =
            learning_rate.clamp(
                0.00001,
                0.1,
            );

        self.gamma =
            gamma.clamp(
                0.80,
                0.9999,
            );

        self.training_steps =
            training_steps;

        self.completed_episodes =
            completed_episodes;

        self.environment.difficulty =
            difficulty;

        self.environment.obstacle_count =
            obstacle_count;

        true
    }


    // =========================================================================
    // SAVE / LOAD ALIASES
    // =========================================================================

    pub fn save_model(
        &self,
    ) -> String {
        self.export_model()
    }

    pub fn load_model(
        &mut self,
        data: String,
    ) -> bool {
        self.import_model(
            data
        )
    }


    // =========================================================================
    // INTERNAL: ACTION SELECTION
    // =========================================================================

    fn select_action(
        &mut self,
        state: &[f32; OBS_SIZE],
    ) -> usize {
        if self.rng.next_f32() <
           self.epsilon
        {
            self.rng.range_usize(
                ACTION_COUNT
            )
        } else {
            argmax(
                &self.network.q_values(
                    state
                )
            )
        }
    }


    // =========================================================================
    // INTERNAL: BATCH TRAINING
    // =========================================================================

    fn train_batch(
        &mut self,
    ) -> f32 {
        if self.replay.len() <
           self.batch_size
        {
            return self.last_loss;
        }

        let mut total_loss =
            0.0f32;

        for _ in
            0..self.batch_size
        {
            if let Some(
                transition
            ) =
            self.replay.sample(
                &mut self.rng
            )
            {
                total_loss +=
                    self.network
                        .train_sample(
                            &self.target_network,
                            &transition,
                            self.learning_rate,
                            self.gamma,
                        );
            }
        }

        total_loss /
        self.batch_size as f32
    }


    // =========================================================================
    // INTERNAL: RECORD TERMINAL RESULT
    // =========================================================================

    fn record_terminal_result(
        &mut self,
        result: StepResult,
    ) {
        self.last_terminal_done =
            result.done;

        self.last_terminal_success =
            result.success;

        self.last_terminal_collision =
            result.collision;

        self.last_terminal_distance =
            result.distance_to_target;
    }


    // =========================================================================
    // INTERNAL: EPISODE FINISH
    // =========================================================================

    fn finish_episode(
        &mut self,
        result: StepResult,
    ) {
        self.completed_episodes +=
            1;

        self.last_episode_reward =
            self.environment
                .episode_reward;

        self.last_episode_steps =
            self.environment.step;

        if result.success {
            self.success_episodes +=
                1;
        }

        if result.collision {
            self.collision_episodes +=
                1;
        }
    }
}


// ============================================================================
// JSON HELPERS
// ============================================================================

fn json_escape(
    input: &str,
) -> String {
    let mut output =
        String::with_capacity(
            input.len()
        );

    for character in
        input.chars()
    {
        match character {
            '"' => {
                output.push_str(
                    "\\\""
                );
            }

            '\\' => {
                output.push_str(
                    "\\\\"
                );
            }

            '\n' => {
                output.push_str(
                    "\\n"
                );
            }

            '\r' => {
                output.push_str(
                    "\\r"
                );
            }

            '\t' => {
                output.push_str(
                    "\\t"
                );
            }

            character
                if character.is_control() =>
            {
                let _ =
                    write!(
                        output,
                        "\\u{:04X}",
                        character as u32
                    );
            }

            character => {
                output.push(
                    character
                );
            }
        }
    }

    output
}

fn push_json_f32(
    output: &mut String,
    value: f32,
) {
    if value.is_finite() {
        let _ =
            write!(
                output,
                "{:.8}",
                value
            );
    } else {
        output.push_str(
            "0.0"
        );
    }
}


// ============================================================================
// WASM START
// ============================================================================

#[wasm_bindgen(start)]
pub fn wasm_start() {
    // Intentionally empty.
    //
    // The engine is created explicitly from JavaScript:
    //
    //   const engine = new RayTracerEngine();
    //
}


// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn network_parameter_count_is_correct() {
        assert_eq!(
            Network::parameter_count(),
            6917
        );
    }

    #[test]
    fn observation_size_is_correct() {
        let environment =
            Environment::new(
                123,
                3,
                8,
                500,
            );

        let observation =
            environment
                .observation();

        assert_eq!(
            observation.len(),
            OBS_SIZE
        );

        assert!(
            observation.iter().all(
                |value| value.is_finite()
            )
        );
    }

    #[test]
    fn sensors_are_normalized() {
        let environment =
            Environment::new(
                9876,
                4,
                10,
                500,
            );

        assert!(
            environment
                .sensors
                .iter()
                .all(
                    |value| {
                        *value >= 0.0 &&
                        *value <= 1.0
                    }
                )
        );
    }

    #[test]
    fn generated_map_has_valid_agent_position() {
        let environment =
            Environment::new(
                4567,
                4,
                10,
                500,
            );

        assert!(
            environment
                .agent
                .position
                .x
                .abs()
                <=
                WORLD_HALF
        );

        assert!(
            environment
                .agent
                .position
                .z
                .abs()
                <=
                WORLD_HALF
        );
    }

    #[test]
    fn network_forward_has_five_actions() {
        let mut rng =
            Lcg::new(
                12345
            );

        let network =
            Network::new(
                &mut rng
            );

        let input =
            [0.0f32; OBS_SIZE];

        let q =
            network.q_values(
                &input
            );

        assert_eq!(
            q.len(),
            ACTION_COUNT
        );
    }

    #[test]
    fn status_json_is_valid_shape() {
        let engine =
            RayTracerEngine::new();

        let json =
            engine.status_json();

        assert!(
            json.starts_with('{')
        );

        assert!(
            json.ends_with('}')
        );

        assert!(
            json.contains(
                "\"version\""
            )
        );

        assert!(
            json.contains(
                "\"backend\""
            )
        );

        assert!(
            json.contains(
                "\"status\""
            )
        );

        assert!(
            json.contains(
                "\"sensorCount\""
            )
        );

        assert!(
            json.contains(
                "\"observationSize\""
            )
        );

        assert!(
            json.contains(
                "\"actionCount\""
            )
        );
    }

    #[test]
    fn terminal_result_is_recorded_without_double_counting() {
        let mut engine =
            RayTracerEngine::new();

        engine.environment.done =
            true;

        engine.environment.success =
            true;

        engine.environment.collision =
            false;

        engine.environment.last_distance =
            0.25;

        let before =
            engine.current_episode();

        let result =
            engine.step(0);

        assert_eq!(
            result[0],
            0.0
        );

        assert_eq!(
            result[1],
            1.0
        );

        assert_eq!(
            result[2],
            1.0
        );

        assert_eq!(
            engine.current_episode(),
            before
        );
    }


    #[test]
    fn model_export_import_roundtrip() {
        let mut first =
            RayTracerEngine::new();

        first.set_epsilon(0.42);
        first.set_learning_rate(
            0.002,
        );
        first.set_gamma(
            0.97,
        );

        let exported =
            first.export_model();

        assert!(
            exported.starts_with(
                "RAYTRC_NAV_MODEL_V1|"
            )
        );

        let mut second =
            RayTracerEngine::new();

        assert!(
            second.import_model(
                exported
            )
        );

        assert!(
            (
                second.epsilon() -
                0.42
            )
            .abs()
            < 0.0001
        );

        assert!(
            (
                second.learning_rate() -
                0.002
            )
            .abs()
            < 0.0001
        );

        assert!(
            (
                second.gamma() -
                0.97
            )
            .abs()
            < 0.0001
        );
    }
}
