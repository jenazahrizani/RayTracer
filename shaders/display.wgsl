struct Params {
    resolutionTime: vec4f,     // width, height, frame, time
    quality: vec4f,            // samples, bounces, accumulation, exposure
    cameraPosition: vec4f,
    cameraRotation: vec4f,
    cameraProjection: vec4f,
    lightPosition: vec4f,
    lightParams: vec4f,
    material: vec4f,
    sceneParams: vec4f,
    padding: vec4f
};

@group(0) @binding(0)
var<uniform> params: Params;

@group(0) @binding(1)
var<storage, read> accumulation: array<vec4f>;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f
};

const FULLSCREEN_POSITIONS: array<vec2f, 3> = array<vec2f, 3>(
    vec2f(-1.0, -3.0),
    vec2f(-1.0,  1.0),
    vec2f( 3.0,  1.0)
);

const FULLSCREEN_UVS: array<vec2f, 3> = array<vec2f, 3>(
    vec2f(0.0, 2.0),
    vec2f(0.0, 0.0),
    vec2f(2.0, 0.0)
);

fn saturate(value: f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn saturate3(value: vec3f) -> vec3f {
    return clamp(
        value,
        vec3f(0.0),
        vec3f(1.0)
    );
}

fn acesTonemap(
    color: vec3f
) -> vec3f {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;

    let numerator =
        color *
        (
            a * color +
            vec3f(b)
        );

    let denominator =
        color *
        (
            c * color +
            vec3f(d)
        ) +
        vec3f(e);

    return saturate3(
        numerator /
        max(
            denominator,
            vec3f(0.000001)
        )
    );
}

fn gammaEncode(
    color: vec3f
) -> vec3f {
    return pow(
        saturate3(color),
        vec3f(1.0 / 2.2)
    );
}

fn safeAverage(
    value: vec4f
) -> vec3f {
    let sampleCount =
        max(
            value.a,
            1.0
        );

    return value.rgb /
        sampleCount;
}

@vertex
fn vertexMain(
    @builtin(vertex_index)
    vertexIndex: u32
) -> VertexOutput {
    var output: VertexOutput;

    output.position =
        vec4f(
            FULLSCREEN_POSITIONS[
                vertexIndex
            ],
            0.0,
            1.0
        );

    output.uv =
        FULLSCREEN_UVS[
            vertexIndex
        ];

    return output;
}

@fragment
fn fragmentMain(
    @builtin(position)
    pixelPosition: vec4f,
    @location(0)
    uv: vec2f
) -> @location(0) vec4f {
    let renderWidth =
        max(
            u32(
                params.resolutionTime.x
            ),
            1u
        );

    let renderHeight =
        max(
            u32(
                params.resolutionTime.y
            ),
            1u
        );

    let pixelX =
        min(
            u32(
                max(
                    pixelPosition.x,
                    0.0
                )
            ),
            renderWidth - 1u
        );

    let pixelY =
        min(
            u32(
                max(
                    pixelPosition.y,
                    0.0
                )
            ),
            renderHeight - 1u
        );

    let pixelIndex =
        pixelY *
        renderWidth +
        pixelX;

    let accumulated =
        accumulation[
            pixelIndex
        ];

    var color =
        safeAverage(
            accumulated
        );

    let exposure =
        params.quality.w;

    color *=
        pow(
            2.0,
            exposure
        );

    color =
        acesTonemap(
            max(
                color,
                vec3f(0.0)
            )
        );

    color =
        gammaEncode(
            color
        );

    let vignetteUv =
        uv * 0.5;

    let centered =
        vignetteUv -
        vec2f(0.25);

    let vignetteDistance =
        dot(
            centered,
            centered
        );

    let vignette =
        1.0 -
        smoothstep(
            0.02,
            0.18,
            vignetteDistance
        ) *
        0.12;

    color *=
        vignette;

    return vec4f(
        saturate3(color),
        1.0
    );
}