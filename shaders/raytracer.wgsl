struct Params {
    resolutionTime: vec4f,     // width, height, frame, time
    quality: vec4f,            // samples, bounces, accumulation, exposure
    cameraPosition: vec4f,    // xyz + unused
    cameraRotation: vec4f,    // pitch, yaw, unused, unused
    cameraProjection: vec4f,   // fovDeg, aspect, unused, unused
    lightPosition: vec4f,     // xyz + unused
    lightParams: vec4f,        // power, unused, unused, unused
    material: vec4f,           // roughness, metallic, ior, materialId
    sceneParams: vec4f,        // objectCount, unused, unused, unused
    padding: vec4f
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
};

struct Ray {
    origin: vec3f,
    direction: vec3f,
};

struct Material {
    albedo: vec3f,
    roughness: f32,
    metallic: f32,
    ior: f32,
};

struct Sphere {
    center: vec3f,
    radius: f32,
    material: Material,
};

@group(0) @binding(0)
var<uniform> params: Params;

@group(0) @binding(1)
var<storage, read_write> accumulation: array<vec4f>;

const PI: f32 = 3.141592653589793;
const INV_PI: f32 = 0.3183098861837907;
const EPSILON: f32 = 0.0005;
const MAX_BOUNCES: u32 = 32u;

const SPHERE_0: Sphere = Sphere(
    vec3f(-1.45, 0.85, 0.0),
    0.85,
    Material(
        vec3f(0.12, 0.72, 0.92),
        0.22,
        0.05,
        1.5
    )
);

const SPHERE_1: Sphere = Sphere(
    vec3f(1.25, 1.05, -0.55),
    1.05,
    Material(
        vec3f(0.92, 0.18, 0.08),
        0.16,
        0.0,
        1.5
    )
);

const SPHERE_2: Sphere = Sphere(
    vec3f(0.15, 0.52, 1.45),
    0.52,
    Material(
        vec3f(0.96, 0.82, 0.16),
        0.32,
        0.78,
        1.5
    )
);

fn saturate3(value: vec3f) -> vec3f {
    return clamp(value, vec3f(0.0), vec3f(1.0));
}

fn saturate1(value: f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn max3(value: vec3f) -> f32 {
    return max(value.x, max(value.y, value.z));
}

fn hash11(value: f32) -> f32 {
    var x = fract(value * 0.1031);
    x *= x + 33.33;
    x *= x + x;
    return fract(x);
}

fn hash21(value: vec2f) -> f32 {
    var p = fract(value * vec2f(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

fn hash31(value: vec3f) -> f32 {
    var p = fract(value * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}

fn random01(pixel: vec2u, frame: u32, dimension: u32) -> f32 {
    let pixelSeed =
        f32(pixel.x) * 0.06711056 +
        f32(pixel.y) * 0.00583715 +
        f32(frame) * 0.0007134 +
        f32(dimension) * 17.123;

    return hash11(pixelSeed + 0.1234567);
}

fn random2(
    pixel: vec2u,
    frame: u32,
    dimension: u32
) -> vec2f {
    return vec2f(
        random01(pixel, frame, dimension),
        random01(pixel, frame, dimension + 1u)
    );
}

fn safeNormalize(value: vec3f) -> vec3f {
    let lengthSquared = dot(value, value);

    if (lengthSquared <= 0.0000001) {
        return vec3f(0.0, 1.0, 0.0);
    }

    return value * inverseSqrt(lengthSquared);
}

fn reflectDirection(
    incident: vec3f,
    normal: vec3f
) -> vec3f {
    return safeNormalize(
        incident - 2.0 * dot(incident, normal) * normal
    );
}

fn refractDirection(
    incident: vec3f,
    normal: vec3f,
    eta: f32
) -> vec3f {
    let cosTheta = min(
        dot(-incident, normal),
        1.0
    );

    let perpendicular =
        eta *
        (incident + cosTheta * normal);

    let parallelMagnitude =
        -sqrt(
            max(
                0.0,
                1.0 -
                dot(perpendicular, perpendicular)
            )
        );

    return safeNormalize(
        perpendicular +
        parallelMagnitude * normal
    );
}

fn fresnelSchlick(
    cosTheta: f32,
    f0: vec3f
) -> vec3f {
    let factor =
        pow(
            1.0 - saturate1(cosTheta),
            5.0
        );

    return f0 +
        (vec3f(1.0) - f0) * factor;
}

fn distributionGGX(
    normal: vec3f,
    halfVector: vec3f,
    roughness: f32
) -> f32 {
    let a = max(
        0.045,
        roughness
    );

    let a2 =
        a * a;

    let normalDotHalf =
        max(
            dot(normal, halfVector),
            0.0
        );

    let normalDotHalf2 =
        normalDotHalf *
        normalDotHalf;

    let denominator =
        normalDotHalf2 *
        (a2 - 1.0) +
        1.0;

    return a2 /
        max(
            PI *
            denominator *
            denominator,
            0.000001
        );
}

fn geometrySchlickGGX(
    normalDotDirection: f32,
    roughness: f32
) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;

    return normalDotDirection /
        max(
            normalDotDirection *
            (1.0 - k) +
            k,
            0.000001
        );
}

fn geometrySmith(
    normal: vec3f,
    viewDirection: vec3f,
    lightDirection: vec3f,
    roughness: f32
) -> f32 {
    let normalDotView =
        max(
            dot(normal, viewDirection),
            0.0
        );

    let normalDotLight =
        max(
            dot(normal, lightDirection),
            0.0
        );

    return
        geometrySchlickGGX(
            normalDotView,
            roughness
        ) *
        geometrySchlickGGX(
            normalDotLight,
            roughness
        );
}

fn randomCosineHemisphere(
    normal: vec3f,
    randomValue: vec2f
) -> vec3f {
    let phi =
        2.0 *
        PI *
        randomValue.x;

    let radial =
        sqrt(
            max(
                randomValue.y,
                0.0
            )
        );

    let localX =
        radial *
        cos(phi);

    let localY =
        radial *
        sin(phi);

    let localZ =
        sqrt(
            max(
                0.0,
                1.0 -
                randomValue.y
            )
        );

    let tangentReference =
        select(
            vec3f(0.0, 1.0, 0.0),
            vec3f(1.0, 0.0, 0.0),
            abs(normal.y) > 0.95
        );

    let tangent =
        safeNormalize(
            cross(
                tangentReference,
                normal
            )
        );

    let bitangent =
        cross(
            normal,
            tangent
        );

    return safeNormalize(
        tangent * localX +
        bitangent * localY +
        normal * localZ
    );
}

fn sampleGGX(
    normal: vec3f,
    viewDirection: vec3f,
    roughness: f32,
    randomValue: vec2f
) -> vec3f {
    let a =
        max(
            roughness,
            0.045
        );

    let a2 =
        a * a;

    let phi =
        2.0 *
        PI *
        randomValue.x;

    let cosTheta =
        sqrt(
            (1.0 - randomValue.y) /
            (
                1.0 +
                (a2 - 1.0) *
                randomValue.y
            )
        );

    let sinTheta =
        sqrt(
            max(
                0.0,
                1.0 -
                cosTheta *
                cosTheta
            )
        );

    let tangentReference =
        select(
            vec3f(0.0, 1.0, 0.0),
            vec3f(1.0, 0.0, 0.0),
            abs(normal.y) > 0.95
        );

    let tangent =
        safeNormalize(
            cross(
                tangentReference,
                normal
            )
        );

    let bitangent =
        cross(
            normal,
            tangent
        );

    let halfVector =
        safeNormalize(
            tangent *
            (sinTheta * cos(phi)) +
            bitangent *
            (sinTheta * sin(phi)) +
            normal *
            cosTheta
        );

    return reflectDirection(
        -viewDirection,
        halfVector
    );
}

fn intersectSphere(
    ray: Ray,
    sphere: Sphere,
    currentClosest: f32
) -> Hit {
    let offset =
        ray.origin -
        sphere.center;

    let a =
        dot(
            ray.direction,
            ray.direction
        );

    let halfB =
        dot(
            offset,
            ray.direction
        );

    let c =
        dot(
            offset,
            offset
        ) -
        sphere.radius *
        sphere.radius;

    let discriminant =
        halfB *
        halfB -
        a *
        c;

    if (
        discriminant <
        0.0
    ) {
        return Hit(
            0.0,
            currentClosest,
            vec3f(0.0),
            vec3f(0.0, 1.0, 0.0),
            vec3f(0.0),
            sphere.material.roughness,
            sphere.material.metallic,
            sphere.material.ior
        );
    }

    let sqrtDiscriminant =
        sqrt(
            discriminant
        );

    var root =
        (-halfB -
            sqrtDiscriminant) /
        a;

    if (
        root <= EPSILON ||
        root >= currentClosest
    ) {
        root =
            (-halfB +
                sqrtDiscriminant) /
            a;
    }

    if (
        root <= EPSILON ||
        root >= currentClosest
    ) {
        return Hit(
            0.0,
            currentClosest,
            vec3f(0.0),
            vec3f(0.0, 1.0, 0.0),
            vec3f(0.0),
            sphere.material.roughness,
            sphere.material.metallic,
            sphere.material.ior
        );
    }

    let position =
        ray.origin +
        ray.direction * root;

    let normal =
        safeNormalize(
            position -
            sphere.center
        );

    return Hit(
        1.0,
        root,
        position,
        normal,
        sphere.material.albedo,
        sphere.material.roughness,
        sphere.material.metallic,
        sphere.material.ior
    );
}

fn intersectPlane(
    ray: Ray,
    currentClosest: f32
) -> Hit {
    let denominator =
        ray.direction.y;

    if (
        abs(denominator) <
        0.000001
    ) {
        return Hit(
            0.0,
            currentClosest,
            vec3f(0.0),
            vec3f(0.0, 1.0, 0.0),
            vec3f(0.0),
            params.material.x,
            params.material.y,
            params.material.z
        );
    }

    let distance =
        -ray.origin.y /
        denominator;

    if (
        distance <= EPSILON ||
        distance >= currentClosest
    ) {
        return Hit(
            0.0,
            currentClosest,
            vec3f(0.0),
            vec3f(0.0, 1.0, 0.0),
            vec3f(0.0),
            params.material.x,
            params.material.y,
            params.material.z
        );
    }

    let position =
        ray.origin +
        ray.direction *
        distance;

    let checker =
        (
            i32(floor(position.x)) +
            i32(floor(position.z))
        ) & 1;

    let baseColor =
        select(
            vec3f(0.08, 0.08, 0.08),
            vec3f(0.32, 0.32, 0.32),
            checker == 1
        );

    return Hit(
        1.0,
        distance,
        position,
        vec3f(0.0, 1.0, 0.0),
        baseColor,
        params.material.x,
        params.material.y,
        params.material.z
    );
}

fn intersectWorld(
    ray: Ray
) -> Hit {
    var closest =
        1.0e30;

    var result =
        Hit(
            0.0,
            closest,
            vec3f(0.0),
            vec3f(0.0, 1.0, 0.0),
            vec3f(0.0),
            params.material.x,
            params.material.y,
            params.material.z
        );

    let floorHit =
        intersectPlane(
            ray,
            closest
        );

    if (
        floorHit.hit > 0.5
    ) {
        closest =
            floorHit.distance;

        result =
            floorHit;
    }

    let sphere0Hit =
        intersectSphere(
            ray,
            SPHERE_0,
            closest
        );

    if (
        sphere0Hit.hit > 0.5
    ) {
        closest =
            sphere0Hit.distance;

        result =
            sphere0Hit;
    }

    let sphere1Hit =
        intersectSphere(
            ray,
            SPHERE_1,
            closest
        );

    if (
        sphere1Hit.hit > 0.5
    ) {
        closest =
            sphere1Hit.distance;

        result =
            sphere1Hit;
    }

    let sphere2Hit =
        intersectSphere(
            ray,
            SPHERE_2,
            closest
        );

    if (
        sphere2Hit.hit > 0.5
    ) {
        result =
            sphere2Hit;
    }

    return result;
}

fn skyColor(
    direction: vec3f
) -> vec3f {
    let t =
        0.5 *
        (
            direction.y +
            1.0
        );

    let horizon =
        vec3f(
            0.72,
            0.80,
            0.94
        );

    let zenith =
        vec3f(
            0.05,
            0.10,
            0.20
        );

    return mix(
        horizon,
        zenith,
        saturate1(t)
    );
}

fn shadowVisible(
    origin: vec3f,
    direction: vec3f,
    maxDistance: f32
) -> bool {
    let ray =
        Ray(
            origin,
            direction
        );

    let blocker =
        intersectWorld(
            ray
        );

    return
        blocker.hit < 0.5 ||
        blocker.distance >=
            maxDistance - EPSILON;
}

fn directLight(
    hit: Hit,
    viewDirection: vec3f
) -> vec3f {
    let lightPosition =
        params.lightPosition.xyz;

    let lightVector =
        lightPosition -
        hit.position;

    let lightDistance =
        length(
            lightVector
        );

    if (
        lightDistance <= 0.0001
    ) {
        return vec3f(0.0);
    }

    let lightDirection =
        lightVector /
        lightDistance;

    let normalDotLight =
        max(
            dot(
                hit.normal,
                lightDirection
            ),
            0.0
        );

    if (
        normalDotLight <= 0.0
    ) {
        return vec3f(0.0);
    }

    let shadowOrigin =
        hit.position +
        hit.normal *
        EPSILON *
        4.0;

    if (
        !shadowVisible(
            shadowOrigin,
            lightDirection,
            lightDistance
        )
    ) {
        return vec3f(0.0);
    }

    let attenuation =
        params.lightParams.x /
        max(
            lightDistance *
            lightDistance,
            1.0
        );

    let radiance =
        vec3f(
            attenuation
        );

    let halfVector =
        safeNormalize(
            viewDirection +
            lightDirection
        );

    let f0 =
        mix(
            vec3f(0.04),
            hit.albedo,
            vec3f(
                hit.metallic
            )
        );

    let fresnel =
        fresnelSchlick(
            max(
                dot(
                    halfVector,
                    viewDirection
                ),
                0.0
            ),
            f0
        );

    let distribution =
        distributionGGX(
            hit.normal,
            halfVector,
            hit.roughness
        );

    let geometry =
        geometrySmith(
            hit.normal,
            viewDirection,
            lightDirection,
            hit.roughness
        );

    let denominator =
        max(
            4.0 *
            max(
                dot(
                    hit.normal,
                    viewDirection
                ),
                0.0
            ) *
            normalDotLight,
            0.0001
        );

    let specular =
        (
            distribution *
            geometry *
            fresnel
        ) /
        denominator;

    let diffuseStrength =
        1.0 -
        hit.metallic;

    let diffuse =
        (
            vec3f(1.0) -
            fresnel
        ) *
        hit.albedo *
        diffuseStrength *
        INV_PI;

    return
        (
            diffuse +
            specular
        ) *
        radiance *
        normalDotLight;
}

fn tracePath(
    ray: Ray,
    pixel: vec2u,
    frame: u32
) -> vec3f {
    var currentRay =
        ray;

    var throughput =
        vec3f(1.0);

    var radiance =
        vec3f(0.0);

    let configuredBounces =
        max(
            1.0,
            params.quality.y
        );

    let bounceLimit =
        min(
            u32(
                floor(
                    configuredBounces
                )
            ),
            MAX_BOUNCES
        );

    for (
        var bounce: u32 = 0u;
        bounce < bounceLimit;
        bounce += 1u
    ) {
        let hit =
            intersectWorld(
                currentRay
            );

        if (
            hit.hit < 0.5
        ) {
            radiance +=
                throughput *
                skyColor(
                    currentRay.direction
                );

            break;
        }

        let viewDirection =
            safeNormalize(
                -currentRay.direction
            );

        let emitted =
            hit.albedo *
            0.0;

        radiance +=
            throughput *
            emitted;

        let light =
            directLight(
                hit,
                viewDirection
            );

        radiance +=
            throughput *
            light;

        let materialId =
            params.material.w;

        let isGlass =
            materialId >= 2.0;

        let localRoughness =
            clamp(
                hit.roughness,
                0.045,
                1.0
            );

        if (
            isGlass
        ) {
            let hitNormal =
                hit.normal;

            let entering =
                dot(
                    currentRay.direction,
                    hitNormal
                ) < 0.0;

            let orientedNormal =
                select(
                    -hitNormal,
                    hitNormal,
                    entering
                );

            let eta =
                select(
                    hit.ior,
                    1.0 / hit.ior,
                    entering
                );

            let cosTheta =
                min(
                    dot(
                        -currentRay.direction,
                        orientedNormal
                    ),
                    1.0
                );

            let sinTheta =
                sqrt(
                    max(
                        0.0,
                        1.0 -
                        cosTheta *
                        cosTheta
                    )
                );

            let cannotRefract =
                eta *
                sinTheta >
                1.0;

            let reflectProbability =
                fresnelSchlick(
                    cosTheta,
                    vec3f(
                        0.04
                    )
                ).x;

            let randomValue =
                random01(
                    pixel,
                    frame,
                    bounce * 11u +
                    7u
                );

            var nextDirection =
                vec3f(0.0);

            if (
                cannotRefract ||
                randomValue <
                    reflectProbability
            ) {
                nextDirection =
                    reflectDirection(
                        currentRay.direction,
                        orientedNormal
                    );
            } else {
                nextDirection =
                    refractDirection(
                        currentRay.direction,
                        orientedNormal,
                        eta
                    );
            }

            throughput *=
                mix(
                    hit.albedo,
                    vec3f(1.0),
                    0.65
                );

            currentRay =
                Ray(
                    hit.position +
                        orientedNormal *
                        EPSILON *
                        6.0,
                    nextDirection
                );

            continue;
        }

        let metallic =
            saturate1(
                hit.metallic
            );

        let specularProbability =
            clamp(
                mix(
                    0.04,
                    1.0,
                    metallic
                ),
                0.05,
                0.95
            );

        let randomSelect =
            random01(
                pixel,
                frame,
                bounce * 7u +
                3u
            );

        var nextDirection =
            vec3f(0.0);

        if (
            randomSelect <
            specularProbability
        ) {
            nextDirection =
                sampleGGX(
                    hit.normal,
                    viewDirection,
                    localRoughness,
                    random2(
                        pixel,
                        frame,
                        bounce * 13u +
                        19u
                    )
                );

            if (
                dot(
                    nextDirection,
                    hit.normal
                ) <= 0.0
            ) {
                nextDirection =
                    randomCosineHemisphere(
                        hit.normal,
                        random2(
                            pixel,
                            frame,
                            bounce * 17u +
                            29u
                        )
                    );
            }

            let fresnel =
                fresnelSchlick(
                    max(
                        dot(
                            nextDirection,
                            hit.normal
                        ),
                        0.0
                    ),
                    mix(
                        vec3f(0.04),
                        hit.albedo,
                        vec3f(
                            metallic
                        )
                    )
                );

            throughput *=
                mix(
                    hit.albedo,
                    vec3f(1.0),
                    fresnel
                );
        } else {
            nextDirection =
                randomCosineHemisphere(
                    hit.normal,
                    random2(
                        pixel,
                        frame,
                        bounce * 23u +
                        37u
                    )
                );

            throughput *=
                hit.albedo;
        }

        currentRay =
            Ray(
                hit.position +
                    hit.normal *
                    EPSILON *
                    6.0,
                safeNormalize(
                    nextDirection
                )
            );

        if (
            bounce >= 2u
        ) {
            let survive =
                min(
                    max3(
                        throughput
                    ),
                    0.95
                );

            let roulette =
                random01(
                    pixel,
                    frame,
                    bounce * 31u +
                    53u
                );

            if (
                roulette >
                survive
            ) {
                break;
            }

            throughput /=
                max(
                    survive,
                    0.05
                );
        }
    }

    return max(
        radiance,
        vec3f(0.0)
    );
}

fn makeCameraRay(
    pixel: vec2u,
    sampleIndex: u32
) -> Ray {
    let width =
        max(
            params.resolutionTime.x,
            1.0
        );

    let height =
        max(
            params.resolutionTime.y,
            1.0
        );

    let frame =
        u32(
            max(
                params.resolutionTime.z,
                0.0
            )
        );

    let randomJitter =
        random2(
            pixel,
            frame + sampleIndex * 17u,
            101u
        ) -
        vec2f(0.5);

    let uv =
        (
            (
                vec2f(
                    f32(pixel.x),
                    f32(pixel.y)
                ) +
                vec2f(0.5) +
                randomJitter
            ) /
            vec2f(
                width,
                height
            )
        ) *
        2.0 -
        vec2f(1.0);

    let aspect =
        params.cameraProjection.y;

    let fovRadians =
        radians(
            params.cameraProjection.x
        );

    let tanHalfFov =
        tan(
            fovRadians *
            0.5
        );

    var direction =
        safeNormalize(
            vec3f(
                uv.x *
                    aspect *
                    tanHalfFov,
                -uv.y *
                    tanHalfFov,
                -1.0
            )
        );

    let pitch =
        radians(
            params.cameraRotation.x
        );

    let yaw =
        radians(
            params.cameraRotation.y
        );

    let cosPitch =
        cos(pitch);

    let sinPitch =
        sin(pitch);

    let cosYaw =
        cos(yaw);

    let sinYaw =
        sin(yaw);

    direction =
        vec3f(
            direction.x * cosYaw -
                direction.z * sinYaw,
            direction.y,
            direction.x * sinYaw +
                direction.z * cosYaw
        );

    direction =
        vec3f(
            direction.x,
            direction.y * cosPitch -
                direction.z * sinPitch,
            direction.y * sinPitch +
                direction.z * cosPitch
        );

    direction =
        safeNormalize(
            direction
        );

    return Ray(
        params.cameraPosition.xyz,
        direction
    );
}

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(global_invocation_id)
    globalId: vec3u
) {
    let width =
        u32(
            params.resolutionTime.x
        );

    let height =
        u32(
            params.resolutionTime.y
        );

    if (
        globalId.x >= width ||
        globalId.y >= height
    ) {
        return;
    }

    let pixelIndex =
        globalId.y *
        width +
        globalId.x;

    let configuredSamples =
        max(
            1.0,
            params.quality.x
        );

    let sampleIndex =
        u32(
            max(
                params.resolutionTime.z,
                0.0
            )
        ) %
        max(
            u32(
                floor(
                    configuredSamples
                )
            ),
            1u
        );

    let frame =
        u32(
            max(
                params.resolutionTime.z,
                0.0
            )
        );

    let ray =
        makeCameraRay(
            globalId.xy,
            sampleIndex
        );

    let sample =
        tracePath(
            ray,
            globalId.xy,
            frame
        );

    let accumulationEnabled =
        params.quality.z >
        0.5;

    let firstFrame =
        frame < 0.5;

    if (
        !accumulationEnabled ||
        firstFrame
    ) {
        accumulation[pixelIndex] =
            vec4f(
                sample,
                1.0
            );
    } else {
        let previous =
            accumulation[
                pixelIndex
            ];

        accumulation[
            pixelIndex
        ] =
            vec4f(
                previous.rgb +
                    sample,
                previous.a +
                    1.0
            );
    }
}