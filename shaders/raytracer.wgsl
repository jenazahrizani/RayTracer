// ============================================================================
// RAYTRC.GPU / shaders/raytracer.wgsl
// ============================================================================
// Compute path tracer used by the RAYTRC renderer.
//
// IMPORTANT HOST CONTRACT
// -----------------------
// Params is exactly 10 x vec4<f32> = 160 bytes. Keep the order unchanged.
//
//   resolutionTime : x width, y height, z frame/sample serial, w seed/time
//   quality        : x samples-per-dispatch, y bounce limit,
//                    z accumulation enabled, w exposure (display shader)
//   cameraPosition : xyz position, w reserved
//   cameraRotation : x pitch degrees, y yaw degrees
//   cameraProjection: x FOV degrees, y aspect ratio
//   lightPosition  : xyz point-light position, w power
//   lightParams    : x light intensity multiplier, y/z/w reserved
//   material       : x roughness, y metallic, z IOR, w material mode
//   sceneParams    : x object count, y reserved, z scene id, w revision/reset
//   padding        : reserved
//
// OUTPUT BUFFER
// ------------
// accumulation[index].rgb = running HDR SUM
// accumulation[index].a   = accumulated sample count
//
// The display shader divides rgb by a and then applies exposure/tonemapping.
// ============================================================================

const PI: f32 = 3.14159265358979323846;
const INV_PI: f32 = 0.31830988618379067154;
const TWO_PI: f32 = 6.28318530717958647692;

const EPSILON: f32 = 0.0005;
const FAR_DISTANCE: f32 = 250.0;
const MAX_BOUNCES: u32 = 16u;
const MAX_SAMPLES: u32 = 2048u;

struct Params {
    resolutionTime: vec4f,
    quality: vec4f,
    cameraPosition: vec4f,
    cameraRotation: vec4f,
    cameraProjection: vec4f,
    lightPosition: vec4f,
    lightParams: vec4f,
    material: vec4f,
    sceneParams: vec4f,
    padding: vec4f,
};

struct Ray {
    origin: vec3f,
    direction: vec3f,
};

struct Hit {
    hit: f32,
    distance: f32,
    position: vec3f,
    normal: vec3f,
    albedo: vec3f,
    roughness: f32,
    metallic: f32,
    ior: f32,
    transmission: f32,
};

@group(0) @binding(0)
var<uniform> params: Params;

@group(0) @binding(1)
var<storage, read_write> accumulation: array<vec4f>;

fn saturate(value: f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn saturate3(value: vec3f) -> vec3f {
    return clamp(value, vec3f(0.0), vec3f(1.0));
}

fn safeNormalize(value: vec3f) -> vec3f {
    let lengthSquared = dot(value, value);
    if (lengthSquared <= 1.0e-12) {
        return vec3f(0.0, 1.0, 0.0);
    }
    return value * inverseSqrt(lengthSquared);
}

fn maxComponent(value: vec3f) -> f32 {
    return max(value.x, max(value.y, value.z));
}

fn makeMiss() -> Hit {
    return Hit(
        0.0,
        FAR_DISTANCE,
        vec3f(0.0),
        vec3f(0.0, 1.0, 0.0),
        vec3f(0.0),
        1.0,
        0.0,
        1.5,
        0.0
    );
}

fn makeHit(
    distance: f32,
    position: vec3f,
    normal: vec3f,
    albedo: vec3f,
    roughness: f32,
    metallic: f32,
    ior: f32,
    transmission: f32
) -> Hit {
    return Hit(
        1.0,
        distance,
        position,
        safeNormalize(normal),
        max(albedo, vec3f(0.0)),
        clamp(roughness, 0.045, 1.0),
        saturate(metallic),
        max(ior, 1.001),
        saturate(transmission)
    );
}

// ---------------------------------------------------------------------------
// Deterministic hash RNG. No mutable pointer parameters are used so this
// remains friendly to current WGSL implementations in Chromium/Firefox/Safari.
// ---------------------------------------------------------------------------

fn hash32(value: u32) -> u32 {
    var x = value;
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

fn hashToUnit(value: u32) -> f32 {
    return f32(hash32(value) & 0x00ffffffu) / 16777216.0;
}

fn random01(pixel: vec2u, frame: u32, sampleIndex: u32, dimension: u32) -> f32 {
    var seed = 0xA341316Cu;
    seed ^= hash32(pixel.x * 1973u + 0x9E3779B9u);
    seed ^= hash32(pixel.y * 9277u + 0x85EBCA6Bu);
    seed ^= hash32(frame * 26699u + 0xC2B2AE35u);
    seed ^= hash32(sampleIndex * 2246822519u + dimension * 3266489917u);
    return hashToUnit(seed + dimension * 0x632BE59Bu);
}

fn random2(pixel: vec2u, frame: u32, sampleIndex: u32, dimension: u32) -> vec2f {
    return vec2f(
        random01(pixel, frame, sampleIndex, dimension),
        random01(pixel, frame, sampleIndex, dimension + 1u)
    );
}

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

fn intersectSphere(ray: Ray, center: vec3f, radius: f32) -> f32 {
    let offset = ray.origin - center;
    let halfB = dot(offset, ray.direction);
    let c = dot(offset, offset) - radius * radius;
    let discriminant = halfB * halfB - c;

    if (discriminant < 0.0) {
        return FAR_DISTANCE;
    }

    let root = sqrt(discriminant);
    var t = -halfB - root;

    if (t <= EPSILON) {
        t = -halfB + root;
    }

    if (t <= EPSILON || t >= FAR_DISTANCE) {
        return FAR_DISTANCE;
    }

    return t;
}

fn intersectPlane(ray: Ray, y: f32) -> f32 {
    if (abs(ray.direction.y) <= 1.0e-6) {
        return FAR_DISTANCE;
    }

    let distance = (y - ray.origin.y) / ray.direction.y;

    if (distance <= EPSILON || distance >= FAR_DISTANCE) {
        return FAR_DISTANCE;
    }

    return distance;
}

fn checkerColor(position: vec3f) -> vec3f {
    let cell = i32(floor(position.x)) + i32(floor(position.z));
    if ((cell & 1) == 0) {
        return vec3f(0.72, 0.74, 0.76);
    }
    return vec3f(0.10, 0.11, 0.13);
}

fn intersectWorld(ray: Ray) -> Hit {
    var closest = FAR_DISTANCE;
    var result = makeMiss();

    let objectCount = u32(clamp(params.sceneParams.x, 0.0, 3.0));
    let sceneId = u32(max(params.sceneParams.z, 0.0));

    // Ground.
    let floorDistance = intersectPlane(ray, 0.0);
    if (floorDistance < closest) {
        let position = ray.origin + ray.direction * floorDistance;
        closest = floorDistance;
        result = makeHit(
            floorDistance,
            position,
            vec3f(0.0, 1.0, 0.0),
            checkerColor(position),
            max(params.material.x, 0.72),
            0.0,
            1.50,
            0.0
        );
    }

    // Primary sphere. Its material follows the UI controls.
    if (objectCount >= 1u) {
        let center = vec3f(0.0, 1.0, 0.0);
        let distance = intersectSphere(ray, center, 1.0);

        if (distance < closest) {
            let position = ray.origin + ray.direction * distance;
            let materialMode = u32(max(params.material.w, 0.0));
            let glassMode = select(0.0, 1.0, materialMode >= 2u);

            closest = distance;
            result = makeHit(
                distance,
                position,
                position - center,
                vec3f(0.78, 0.20, 0.055),
                params.material.x,
                params.material.y,
                params.material.z,
                glassMode
            );
        }
    }

    // Blue sphere.
    if (objectCount >= 2u) {
        let center = vec3f(-2.0, 0.75, -1.4);
        let distance = intersectSphere(ray, center, 0.75);

        if (distance < closest) {
            let position = ray.origin + ray.direction * distance;
            closest = distance;
            result = makeHit(
                distance,
                position,
                position - center,
                vec3f(0.055, 0.22, 0.78),
                0.32,
                0.15,
                1.50,
                0.0
            );
        }
    }

    // Green metallic sphere. Scene id creates a tiny deterministic variation
    // so map/scene changes are visible without changing host-side geometry.
    if (objectCount >= 3u) {
        let center = vec3f(1.85, 0.65, -1.15);
        let distance = intersectSphere(ray, center, 0.65);

        if (distance < closest) {
            let position = ray.origin + ray.direction * distance;
            let roughness = mix(0.30, 0.18, saturate(f32(sceneId & 1u)));
            closest = distance;
            result = makeHit(
                distance,
                position,
                position - center,
                vec3f(0.08, 0.68, 0.22),
                roughness,
                0.72,
                1.50,
                0.0
            );
        }
    }

    return result;
}

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

fn cameraRay(pixel: vec2u, sampleIndex: u32, frame: u32) -> Ray {
    let resolution = max(params.resolutionTime.xy, vec2f(1.0));
    let jitter = random2(pixel, frame, sampleIndex, 11u) - vec2f(0.5);

    let uv = (
        vec2f(f32(pixel.x), f32(pixel.y)) +
        vec2f(0.5) +
        jitter
    ) / resolution;

    let ndc = uv * 2.0 - vec2f(1.0);
    let aspect = max(params.cameraProjection.y, 0.001);
    let fovRadians = radians(clamp(params.cameraProjection.x, 1.0, 170.0));
    let tanHalfFov = tan(0.5 * fovRadians);

    // Camera convention matches the current controls: yaw 0 / pitch 0 faces -Z.
    let pitch = radians(clamp(params.cameraRotation.x, -89.0, 89.0));
    let yaw = radians(params.cameraRotation.y);

    let cp = cos(pitch);
    let sp = sin(pitch);
    let cy = cos(yaw);
    let sy = sin(yaw);

    let forward = safeNormalize(vec3f(
        -sy * cp,
        sp,
        -cy * cp
    ));

    let right = safeNormalize(cross(forward, vec3f(0.0, 1.0, 0.0)));
    let up = safeNormalize(cross(right, forward));

    let direction = safeNormalize(
        forward +
        right * (ndc.x * aspect * tanHalfFov) +
        up * (-ndc.y * tanHalfFov)
    );

    return Ray(
        params.cameraPosition.xyz,
        direction
    );
}

// ---------------------------------------------------------------------------
// BRDF / lighting
// ---------------------------------------------------------------------------

fn fresnelSchlick(cosTheta: f32, f0: vec3f) -> vec3f {
    let factor = pow(1.0 - saturate(cosTheta), 5.0);
    return f0 + (vec3f(1.0) - f0) * factor;
}

fn distributionGGX(normal: vec3f, halfVector: vec3f, roughness: f32) -> f32 {
    let a = max(roughness * roughness, 0.002025);
    let a2 = a * a;
    let nDotH = max(dot(normal, halfVector), 0.0);
    let nDotH2 = nDotH * nDotH;
    let denominator = nDotH2 * (a2 - 1.0) + 1.0;

    return a2 / max(PI * denominator * denominator, 1.0e-6);
}

fn geometrySchlickGGX(nDotDirection: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return nDotDirection / max(nDotDirection * (1.0 - k) + k, 1.0e-6);
}

fn geometrySmith(
    normal: vec3f,
    viewDirection: vec3f,
    lightDirection: vec3f,
    roughness: f32
) -> f32 {
    let nDotV = max(dot(normal, viewDirection), 0.0);
    let nDotL = max(dot(normal, lightDirection), 0.0);
    return geometrySchlickGGX(nDotV, roughness) * geometrySchlickGGX(nDotL, roughness);
}

fn tangentFor(normal: vec3f) -> vec3f {
    if (abs(normal.y) < 0.999) {
        return safeNormalize(cross(vec3f(0.0, 1.0, 0.0), normal));
    }
    return safeNormalize(cross(vec3f(1.0, 0.0, 0.0), normal));
}

fn cosineHemisphere(normal: vec3f, randomValue: vec2f) -> vec3f {
    let phi = TWO_PI * randomValue.x;
    let radial = sqrt(max(randomValue.y, 0.0));
    let x = radial * cos(phi);
    let y = sqrt(max(0.0, 1.0 - randomValue.y));
    let z = radial * sin(phi);

    let tangent = tangentFor(normal);
    let bitangent = cross(normal, tangent);

    return safeNormalize(
        tangent * x +
        normal * y +
        bitangent * z
    );
}

fn sampleGGX(normal: vec3f, viewDirection: vec3f, roughness: f32, randomValue: vec2f) -> vec3f {
    let alpha = max(roughness * roughness, 0.002025);
    let alpha2 = alpha * alpha;
    let phi = TWO_PI * randomValue.x;

    let cosTheta = sqrt(
        (1.0 - randomValue.y) /
        max(1.0 + (alpha2 - 1.0) * randomValue.y, 1.0e-6)
    );

    let sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    let tangent = tangentFor(normal);
    let bitangent = cross(normal, tangent);

    let halfVector = safeNormalize(
        tangent * (sinTheta * cos(phi)) +
        bitangent * (sinTheta * sin(phi)) +
        normal * cosTheta
    );

    return safeNormalize(reflect(-viewDirection, halfVector));
}

fn skyColor(direction: vec3f) -> vec3f {
    let t = saturate(0.5 * (direction.y + 1.0));

    let horizon = vec3f(0.012, 0.018, 0.028);
    let zenith = vec3f(0.060, 0.085, 0.130);
    var result = mix(horizon, zenith, t);

    let lightDirection = safeNormalize(params.lightPosition.xyz - params.cameraPosition.xyz);
    let sun = pow(max(dot(direction, lightDirection), 0.0), 700.0);
    result += vec3f(1.0, 0.92, 0.78) * sun * 2.0;

    let horizonBand = pow(1.0 - abs(direction.y), 5.0);
    result += vec3f(0.02, 0.028, 0.04) * horizonBand;

    return max(result, vec3f(0.0));
}

fn shadowVisible(origin: vec3f, direction: vec3f, maxDistance: f32) -> bool {
    let ray = Ray(origin, direction);
    let blocker = intersectWorld(ray);
    return blocker.hit < 0.5 || blocker.distance >= maxDistance;
}

fn directLight(hit: Hit, viewDirection: vec3f) -> vec3f {
    let lightVector = params.lightPosition.xyz - hit.position;
    let lightDistance = length(lightVector);

    if (lightDistance <= 1.0e-4) {
        return vec3f(0.0);
    }

    let lightDirection = lightVector / lightDistance;
    let nDotL = max(dot(hit.normal, lightDirection), 0.0);

    if (nDotL <= 0.0) {
        return vec3f(0.0);
    }

    let shadowOrigin = hit.position + hit.normal * (EPSILON * 8.0);
    if (!shadowVisible(shadowOrigin, lightDirection, lightDistance - EPSILON * 10.0)) {
        return vec3f(0.0);
    }

    let power = max(params.lightPosition.w, 0.0) * max(params.lightParams.x, 0.0);
    let attenuation = power / max(lightDistance * lightDistance, 1.0);

    let halfVector = safeNormalize(viewDirection + lightDirection);
    let nDotV = max(dot(hit.normal, viewDirection), 0.0);
    let nDotH = max(dot(hit.normal, halfVector), 0.0);
    let hDotV = max(dot(halfVector, viewDirection), 0.0);

    let roughness = clamp(hit.roughness, 0.045, 1.0);
    let f0 = mix(vec3f(0.04), hit.albedo, vec3f(hit.metallic));
    let fresnel = fresnelSchlick(hDotV, f0);

    let distribution = distributionGGX(hit.normal, halfVector, roughness);
    let geometry = geometrySmith(hit.normal, viewDirection, lightDirection, roughness);

    let specular = (
        distribution * geometry * fresnel
    ) / max(4.0 * nDotV * nDotL, 1.0e-5);

    let diffuseWeight = (vec3f(1.0) - fresnel) * (1.0 - hit.metallic);
    let diffuse = diffuseWeight * hit.albedo * INV_PI;

    return (diffuse + specular) * attenuation * nDotL;
}

// ---------------------------------------------------------------------------
// Path tracing
// ---------------------------------------------------------------------------

fn tracePath(pixel: vec2u, frame: u32, sampleIndex: u32) -> vec3f {
    var ray = cameraRay(pixel, sampleIndex, frame);
    var throughput = vec3f(1.0);
    var radiance = vec3f(0.0);

    let bounceLimit = min(
        u32(floor(max(params.quality.y, 1.0))),
        MAX_BOUNCES
    );

    for (var bounce = 0u; bounce < MAX_BOUNCES; bounce += 1u) {
        if (bounce >= bounceLimit) {
            break;
        }

        let hit = intersectWorld(ray);

        if (hit.hit < 0.5) {
            radiance += throughput * skyColor(ray.direction);
            break;
        }

        let viewDirection = safeNormalize(-ray.direction);
        let localRoughness = clamp(hit.roughness, 0.045, 1.0);

        radiance += throughput * directLight(hit, viewDirection);

        // Very small ambient contribution so fully shadowed faces remain readable.
        let ambient = max(params.lightParams.w, 0.0);
        radiance += throughput * hit.albedo * ambient * 0.12;

        let isGlass = hit.transmission > 0.5;

        if (isGlass) {
            let entering = dot(ray.direction, hit.normal) < 0.0;
            let normal = select(-hit.normal, hit.normal, entering);
            let eta = select(hit.ior, 1.0 / hit.ior, entering);
            let cosTheta = min(dot(-ray.direction, normal), 1.0);
            let sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
            let cannotRefract = eta * sinTheta > 1.0;

            let f0 = vec3f(
                pow((1.0 - hit.ior) / (1.0 + hit.ior), 2.0)
            );
            let reflectProbability = fresnelSchlick(cosTheta, f0).x;
            let randomValue = random01(pixel, frame, sampleIndex, bounce * 17u + 71u);

            var nextDirection: vec3f;
            if (cannotRefract || randomValue < reflectProbability) {
                nextDirection = safeNormalize(reflect(ray.direction, normal));
            } else {
                nextDirection = safeNormalize(refract(ray.direction, normal, eta));
            }

            throughput *= mix(hit.albedo, vec3f(1.0), 0.75);
            let offsetNormal = select(-normal, normal, dot(nextDirection, normal) >= 0.0);
            ray = Ray(
                hit.position + offsetNormal * (EPSILON * 8.0),
                nextDirection
            );
            continue;
        }

        let f0 = mix(vec3f(0.04), hit.albedo, vec3f(hit.metallic));
        let fresnel = fresnelSchlick(max(dot(hit.normal, viewDirection), 0.0), f0);
        let specularWeight = clamp(maxComponent(fresnel), 0.04, 0.96);
        let diffuseWeight = max((1.0 - hit.metallic) * (1.0 - specularWeight), 0.02);
        let probabilitySum = specularWeight + diffuseWeight;
        let specularProbability = specularWeight / probabilitySum;

        let chooseSpecular = random01(
            pixel,
            frame,
            sampleIndex,
            bounce * 23u + 5u
        ) < specularProbability;

        var nextDirection = vec3f(0.0);

        if (chooseSpecular) {
            nextDirection = sampleGGX(
                hit.normal,
                viewDirection,
                localRoughness,
                random2(pixel, frame, sampleIndex, bounce * 29u + 11u)
            );

            if (dot(nextDirection, hit.normal) <= 0.0) {
                nextDirection = safeNormalize(reflect(ray.direction, hit.normal));
            }

            throughput *= fresnel / max(specularProbability, 0.05);
        } else {
            nextDirection = cosineHemisphere(
                hit.normal,
                random2(pixel, frame, sampleIndex, bounce * 31u + 17u)
            );

            throughput *= hit.albedo * (1.0 - hit.metallic) / max(diffuseWeight, 0.02);
        }

        // Russian roulette after the first two bounces.
        if (bounce >= 2u) {
            let survival = clamp(maxComponent(throughput), 0.05, 0.95);
            let roulette = random01(pixel, frame, sampleIndex, bounce * 37u + 29u);

            if (roulette > survival) {
                break;
            }

            throughput /= survival;
        }

        // NaN/Inf guard. Keep one broken sample from poisoning the accumulation.
        if (any(throughput != throughput)) {
            break;
        }

        let offsetNormal = select(-hit.normal, hit.normal, dot(nextDirection, hit.normal) >= 0.0);
        ray = Ray(
            hit.position + offsetNormal * (EPSILON * 8.0),
            safeNormalize(nextDirection)
        );
    }

    return max(radiance, vec3f(0.0));
}

// ---------------------------------------------------------------------------
// Compute entry
// ---------------------------------------------------------------------------

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) globalId: vec3u) {
    let width = max(u32(max(params.resolutionTime.x, 1.0)), 1u);
    let height = max(u32(max(params.resolutionTime.y, 1.0)), 1u);

    if (globalId.x >= width || globalId.y >= height) {
        return;
    }

    let pixelIndex = globalId.y * width + globalId.x;
    let frame = u32(max(params.resolutionTime.z, 0.0));

    var previous = accumulation[pixelIndex];
    var previousColor = previous.rgb;
    var previousCount = max(previous.a, 0.0);

    let accumulationEnabled = params.quality.z > 0.5;
    if (!accumulationEnabled) {
        previousColor = vec3f(0.0);
        previousCount = 0.0;
    }

    let sampleCount = min(
        u32(floor(max(params.quality.x, 1.0))),
        MAX_SAMPLES
    );

    var sampleSum = vec3f(0.0);

    for (var sampleIndex = 0u; sampleIndex < MAX_SAMPLES; sampleIndex += 1u) {
        if (sampleIndex >= sampleCount) {
            break;
        }

        sampleSum += tracePath(
            globalId.xy,
            frame,
            sampleIndex
        );
    }

    let sampleWeight = f32(sampleCount);
    let newCount = previousCount + sampleWeight;

    // Store a true running HDR SUM. display.wgsl divides rgb by alpha, so
    // the render remains correctly normalized after every additional batch.
    var result = previousColor + sampleSum;

    // Keep the accumulation finite and comfortably inside the HDR range.
    result = clamp(result, vec3f(0.0), vec3f(65504.0));

    accumulation[pixelIndex] = vec4f(result, newCount);
}
