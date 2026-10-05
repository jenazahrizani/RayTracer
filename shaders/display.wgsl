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

// Single fullscreen triangle. UV range across the visible triangle is 0..1.
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

const EPSILON: f32 = 0.000001;
const MAX_DISPLAY_HDR: f32 = 65504.0;

fn saturate3(value: vec3f) -> vec3f {
    return clamp(value, vec3f(0.0), vec3f(1.0));
}

fn finiteSafe(value: vec3f) -> vec3f {
    let finiteX = select(0.0, value.x, value.x == value.x);
    let finiteY = select(0.0, value.y, value.y == value.y);
    let finiteZ = select(0.0, value.z, value.z == value.z);

    return vec3f(
        finiteX,
        finiteY,
        finiteZ
    );
}

fn acesTonemap(color: vec3f) -> vec3f {
    // Narkowicz-style ACES approximation.
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;

    let numerator =
        color *
        (a * color + vec3f(b));

    let denominator =
        color *
        (c * color + vec3f(d)) +
        vec3f(e);

    return saturate3(
        numerator /
        max(denominator, vec3f(EPSILON))
    );
}

fn gammaEncode(color: vec3f) -> vec3f {
    // Rec.709/sRGB-like display transfer approximation.
    return pow(
        saturate3(color),
        vec3f(1.0 / 2.2)
    );
}

fn safeAverage(value: vec4f) -> vec3f {
    let finiteCount = select(1.0, value.a, value.a == value.a);
    let safeCount = max(finiteCount, 1.0);
    return value.rgb / safeCount;
}

fn applyExposure(color: vec3f, exposure: f32) -> vec3f {
    // Keep exposure bounded so a corrupt uniform cannot overflow the display path.
    let safeExposure = clamp(exposure, -16.0, 16.0);
    let scale = exp2(safeExposure);
    return color * scale;
}

fn applyVignette(
    color: vec3f,
    uv: vec2f,
    resolution: vec2f
) -> vec3f {
    let safeResolution = max(resolution, vec2f(1.0));
    let aspect = safeResolution.x / safeResolution.y;

    // Correct centered coordinates; the old implementation used a quarter-offset
    // center, which made the vignette visibly asymmetric.
    let centered = uv - vec2f(0.5);
    let corrected = vec2f(centered.x * aspect, centered.y);
    let radius = length(corrected);

    let vignette = 1.0 -
        0.13 *
        smoothstep(0.28, 0.82, radius);

    return color * vignette;
}

fn resolvePixelIndex(
    pixelPosition: vec4f,
    width: u32,
    height: u32
) -> u32 {
    let x = min(
        u32(max(pixelPosition.x, 0.0)),
        width - 1u
    );

    let y = min(
        u32(max(pixelPosition.y, 0.0)),
        height - 1u
    );

    return y * width + x;
}

@vertex
fn vertexMain(
    @builtin(vertex_index) vertexIndex: u32
) -> VertexOutput {
    var output: VertexOutput;

    output.position = vec4f(
        FULLSCREEN_POSITIONS[vertexIndex],
        0.0,
        1.0
    );

    output.uv = FULLSCREEN_UVS[vertexIndex];

    return output;
}

@fragment
fn fragmentMain(
    @builtin(position) pixelPosition: vec4f,
    @location(0) uv: vec2f
) -> @location(0) vec4f {
    let renderWidth = max(
        u32(max(params.resolutionTime.x, 1.0)),
        1u
    );

    let renderHeight = max(
        u32(max(params.resolutionTime.y, 1.0)),
        1u
    );

    let pixelIndex = resolvePixelIndex(
        pixelPosition,
        renderWidth,
        renderHeight
    );

    // accumulation.rgb is the summed HDR radiance and accumulation.a is
    // the number of samples contributing to that pixel.
    let accumulated = accumulation[pixelIndex];

    var color = safeAverage(accumulated);
    color = finiteSafe(color);
    color = clamp(
        color,
        vec3f(0.0),
        vec3f(MAX_DISPLAY_HDR)
    );

    color = applyExposure(
        color,
        params.quality.w
    );

    // HDR -> display-referred -> gamma encoded.
    color = acesTonemap(color);
    color = gammaEncode(color);

    color = applyVignette(
        color,
        uv,
        vec2f(
            f32(renderWidth),
            f32(renderHeight)
        )
    );

    return vec4f(
        saturate3(color),
        1.0
    );
}
