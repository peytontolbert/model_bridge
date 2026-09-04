import { g as glMatrix } from './shader_program-sEzzVVPq.js';

const PIT = [-43750.859375, -19906.48046875, 586.9697875976562];
const TOPOLOGY_URL = 'assets/matrix-universe/world-packages/nohesi-415/topology/traffic-network.json';
const GAMEPLAY_ZONES_URL = 'assets/matrix-universe/world-packages/nohesi-415/runtime/gameplay-zones.json';
const RESOLVED_TEXTURE_MANIFEST_URL = 'assets/matrix-universe/world-packages/nohesi-415/recovered-csp-textures-512/manifest.json';
const RECOVERED_ROAD_URL = 'assets/matrix-universe/world-packages/nohesi-415/recovered-road-material-web/manifest.json';
const RECOVERED_ENVIRONMENT_URL = 'assets/matrix-universe/world-packages/nohesi-415/recovered-environment-resolved-web/manifest.json';
const RECOVERED_PIT_FOCUS = [-0.16015625, 30.990234375, -0.09442138671875];
const canvas = document.querySelector('#trackCanvas');
const status = document.querySelector('#status');
const overview = document.querySelector('#mapOverview');
const topologyCanvas = document.querySelector('#topologyCanvas');
const toggleOverview = document.querySelector('#toggleOverview');
const moreButton = document.querySelector('#more');
const toggleZonesButton = document.querySelector('#toggleZones');
const teleportSelect = document.querySelector('#teleportSelect');
const teleportGoButton = document.querySelector('#teleportGo');
const gl = canvas.getContext('webgl2', { antialias: true, alpha: false, depth: true });
if (!gl) throw new Error('WebGL 2 is required for the 415 highway renderer.');

const vertexShader = `#version 300 es
precision highp float;
layout(location=0) in vec3 position;
layout(location=1) in vec3 color;
uniform mat4 viewProjection;
out vec3 vColor;
out vec3 vPosition;
void main() { vColor = color; vPosition = position; gl_Position = viewProjection * vec4(position, 1.0); }
`;
const fragmentShader = `#version 300 es
precision highp float;
in vec3 vColor;
in vec3 vPosition;
uniform sampler2D roadTexture;
uniform float nightMix;
out vec4 outColor;
void main() {
  vec3 shaded = vColor;
  if (vColor.r < 0.16 && vColor.g < 0.17) {
    vec3 asphalt = texture(roadTexture, vPosition.xy * 0.035).rgb;
    shaded = mix(vColor, asphalt * 0.72, 0.78);
  }
  float heightShade = clamp(0.88 + vPosition.z * 0.00035, 0.72, 1.08);
  float localLights = exp(-length(vPosition.xy - vec2(-38.516, 8.549)) / 65.93)
    + exp(-length(vPosition.xy - vec2(-39.211, 9.065)) / 65.93)
    + exp(-length(vPosition.xy - vec2(-152.821, 15.467)) / 65.93)
    + exp(-length(vPosition.xy - vec2(-153.567, 14.391)) / 65.93);
  shaded *= mix(1.0, 0.22, nightMix);
  shaded += vec3(0.30, 0.29, 0.25) * min(localLights, 1.5) * nightMix;
  outColor = vec4(shaded * heightShade, 1.0);
}
`;

function compile(type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
  return shader;
}
const program = gl.createProgram();
gl.attachShader(program, compile(gl.VERTEX_SHADER, vertexShader));
gl.attachShader(program, compile(gl.FRAGMENT_SHADER, fragmentShader));
gl.linkProgram(program);
if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
const matrixLocation = gl.getUniformLocation(program, 'viewProjection');
const textureLocation = gl.getUniformLocation(program, 'roadTexture');
const nightLocation = gl.getUniformLocation(program, 'nightMix');
function loadTexture(texture, url, useMipmaps, attempt = 1) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.decoding = 'async';
    image.onload = () => {
      try {
        gl.bindTexture(gl.TEXTURE_2D, texture);
        // KN5 UVs retain DirectX's top-left texture convention, and DDS-to-WebP
        // recovery preserves row order. Flipping here turns trees and signs
        // upside down.
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
        // Source images are already display-encoded. Keep them in RGBA8 because this
        // canvas does not use an sRGB framebuffer; SRGB8_ALPHA8 made asphalt decode
        // to linear and then display almost black.
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, image);
        if (useMipmaps) gl.generateMipmap(gl.TEXTURE_2D);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, useMipmaps ? gl.LINEAR_MIPMAP_LINEAR : gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        const error = gl.getError();
        if (error !== gl.NO_ERROR) throw new Error(`WebGL texture upload failed (${error}): ${url}`);
        resolve(texture);
      } catch (error) {
        gl.deleteTexture(texture);
        reject(error);
      } finally {
        image.onload = null;
        image.onerror = null;
      }
    };
    image.onerror = () => {
      image.onload = null;
      image.onerror = null;
      if (attempt < 3) {
        window.setTimeout(() => {
          loadTexture(texture, url, useMipmaps, attempt + 1).then(resolve, reject);
        }, attempt * 250);
        return;
      }
      gl.deleteTexture(texture);
      reject(new Error(`Recovered texture failed after 3 attempts: ${url}`));
    };
    const separator = url.includes('?') ? '&' : '?';
    image.src = attempt === 1 ? url : `${url}${separator}decodeRetry=${attempt}`;
  });
}

function materialKey(material) {
  return String(material || '').trim().toLowerCase();
}

async function loadResolvedMaterialTextures(requiredMaterials, mipmappedMaterials) {
  const response = await fetch(RESOLVED_TEXTURE_MANIFEST_URL, { cache: 'no-store' });
  if (!response.ok) throw new Error(`Resolved texture manifest failed (${response.status})`);
  const manifest = await response.json();
  if (manifest.format !== 'CSP_RESOLVED_TEXTURE_WEB_1' || manifest.strictNoFallback !== true) {
    throw new Error('Resolved texture manifest is not a strict CSP recovery.');
  }
  const recordByMaterial = new Map();
  for (const record of manifest.textures) {
    for (const binding of record.bindings || []) {
      if (String(binding.slot).toLowerCase() === 'txdiffuse') {
        recordByMaterial.set(materialKey(binding.material), record);
      }
    }
  }
  const missing = [...requiredMaterials].filter(material => !recordByMaterial.has(materialKey(material)));
  if (missing.length) throw new Error(`CSP texture bindings missing for: ${missing.join(', ')}`);

  const records = [...new Set([...requiredMaterials].map(
    material => recordByMaterial.get(materialKey(material))
  ))];
  const mipmappedRecordIDs = new Set([...mipmappedMaterials].map(
    material => recordByMaterial.get(materialKey(material)).id
  ));
  const texturesByID = new Map();
  let cursor = 0;
  const baseURL = RESOLVED_TEXTURE_MANIFEST_URL.replace(/manifest\.json$/, '');
  async function worker() {
    while (cursor < records.length) {
      const record = records[cursor++];
      const texture = gl.createTexture();
      await loadTexture(texture, `${baseURL}${record.webPath}`, mipmappedRecordIDs.has(record.id));
      texturesByID.set(record.id, texture);
    }
  }
  // Decode and upload one recovered image at a time. Parallel image decodes caused
  // Chromium/SwiftShader to exhaust transient resources on this large scene.
  await worker();

  const diffuseByMaterial = new Map();
  for (const material of requiredMaterials) {
    const record = recordByMaterial.get(materialKey(material));
    diffuseByMaterial.set(materialKey(material), texturesByID.get(record.id));
  }
  return { manifest, diffuseByMaterial, loadedCount: records.length, mipmappedCount: mipmappedRecordIDs.size };
}

const nativeVertexShader = `#version 300 es
precision highp float;
layout(location=0) in vec3 position;
layout(location=1) in vec3 normal;
layout(location=2) in vec2 uv;
uniform mat4 viewProjection;
out vec3 vNormal;
out vec2 vUv;
out vec3 vPosition;
void main() {
  // Use a proper rotation from Assetto Y-up to viewer Z-up. The former
  // [X,Z,Y] swap had determinant -1 and reflected the highway.
  vPosition = vec3(position.x - ${PIT[0]}, -(position.z - ${PIT[1]}), position.y - ${PIT[2]});
  vNormal = normalize(vec3(normal.x, -normal.z, normal.y));
  vUv = uv;
  gl_Position = viewProjection * vec4(vPosition, 1.0);
}
`;
const nativeFragmentShader = `#version 300 es
precision highp float;
in vec3 vNormal;
in vec2 vUv;
in vec3 vPosition;
uniform sampler2D roadTexture;
uniform vec3 baseColor;
uniform float textured;
uniform float nightMix;
out vec4 outColor;
void main() {
  vec3 sun = normalize(vec3(-0.35, 0.24, 0.91));
  float diffuse = 0.82 + 0.18 * abs(dot(normalize(vNormal), sun));
  vec3 shaded = baseColor;
  if (textured > 0.5) {
    vec4 texel = texture(roadTexture, vUv);
    if (texel.a < 0.2) discard;
    shaded = texel.rgb * 1.18;
  }
  shaded *= diffuse;
  float localLights = exp(-length(vPosition.xy - vec2(-38.516, -8.549)) / 65.93)
    + exp(-length(vPosition.xy - vec2(-39.211, -9.065)) / 65.93)
    + exp(-length(vPosition.xy - vec2(-152.821, -15.467)) / 65.93)
    + exp(-length(vPosition.xy - vec2(-153.567, -14.391)) / 65.93);
  shaded *= mix(1.0, 0.56, nightMix);
  shaded += vec3(0.12, 0.115, 0.10) * min(localLights, 1.25) * nightMix;
  outColor = vec4(clamp(shaded, 0.0, 1.0), 1.0);
}
`;

function createProgram(vertexSource, fragmentSource) {
  const linked = gl.createProgram();
  gl.attachShader(linked, compile(gl.VERTEX_SHADER, vertexSource));
  gl.attachShader(linked, compile(gl.FRAGMENT_SHADER, fragmentSource));
  gl.linkProgram(linked);
  if (!gl.getProgramParameter(linked, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(linked));
  return linked;
}

const nativeProgram = createProgram(nativeVertexShader, nativeFragmentShader);
const nativeUniforms = {
  matrix: gl.getUniformLocation(nativeProgram, 'viewProjection'),
  texture: gl.getUniformLocation(nativeProgram, 'roadTexture'),
  color: gl.getUniformLocation(nativeProgram, 'baseColor'),
  textured: gl.getUniformLocation(nativeProgram, 'textured'),
  night: gl.getUniformLocation(nativeProgram, 'nightMix')
};

function resolvedMaterialStyle(material, diffuseByMaterial) {
  const texture = diffuseByMaterial.get(materialKey(material));
  if (!texture) throw new Error(`No resolved CSP diffuse texture for material: ${material}`);
  return { color: [1, 1, 1], textured: 1, texture };
}

async function uploadRecoveredRoad() {
  const response = await fetch(RECOVERED_ROAD_URL, { cache: 'no-store' });
  if (!response.ok) throw new Error(`Recovered road manifest failed (${response.status})`);
  const manifest = await response.json();
  if (manifest.format !== 'CSP_WEB_ROAD_MATERIAL_1' || manifest.vertexStride !== 32) {
    throw new Error('Recovered road format is not supported.');
  }
  const geometryUrl = `${RECOVERED_ROAD_URL.replace(/manifest\.json$/, '')}${manifest.geometry}`;
  const geometryResponse = await fetch(geometryUrl, { cache: 'force-cache' });
  if (!geometryResponse.ok) throw new Error(`Recovered road geometry failed (${geometryResponse.status})`);
  const buffer = gl.createBuffer();
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, await geometryResponse.arrayBuffer(), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, manifest.vertexStride, 0);
  gl.enableVertexAttribArray(1);
  gl.vertexAttribPointer(1, 3, gl.FLOAT, false, manifest.vertexStride, 12);
  gl.enableVertexAttribArray(2);
  gl.vertexAttribPointer(2, 2, gl.FLOAT, false, manifest.vertexStride, 24);
  gl.bindVertexArray(null);
  const meshes = manifest.meshes.map(mesh => ({ ...mesh }));
  return { manifest, meshes, vao, triangles: manifest.triangleCount };
}

async function uploadRecoveredEnvironment() {
  const response = await fetch(RECOVERED_ENVIRONMENT_URL, { cache: 'no-store' });
  if (!response.ok) throw new Error(`Recovered environment manifest failed (${response.status})`);
  const manifest = await response.json();
  if (manifest.format !== 'CSP_WEB_ENVIRONMENT_MATERIAL_1' || manifest.vertexStride !== 32) {
    throw new Error('Recovered environment format is not supported.');
  }
  const geometryUrl = `${RECOVERED_ENVIRONMENT_URL.replace(/manifest\.json$/, '')}${manifest.geometry}`;
  const geometryResponse = await fetch(geometryUrl, { cache: 'force-cache' });
  if (!geometryResponse.ok) throw new Error(`Recovered environment geometry failed (${geometryResponse.status})`);
  const buffer = gl.createBuffer();
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, await geometryResponse.arrayBuffer(), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, manifest.vertexStride, 0);
  gl.enableVertexAttribArray(1);
  gl.vertexAttribPointer(1, 3, gl.FLOAT, false, manifest.vertexStride, 12);
  gl.enableVertexAttribArray(2);
  gl.vertexAttribPointer(2, 2, gl.FLOAT, false, manifest.vertexStride, 24);
  gl.bindVertexArray(null);
  const segments = manifest.segments
    .filter(segment => segment.vertexCount > 0)
    .map(segment => ({ ...segment }));
  return { manifest, segments, vao, triangles: manifest.triangleCount };
}

let topology;
let gameplayZones;
let zoneMesh;
let recoveredRoad;
let recoveredEnvironment;
let yaw = -3.11;
let pitch = 0.62;
let distance = 600;
let dragging = false;
let lastPointer;
let overviewVisible = false;
let zonesVisible = true;
let focus = [0, 0, 10];
let nightMode = false;
const renderHealth = {
  frame: 0,
  glError: 0,
  contextLost: false,
  passes: {
    environment: { enabled: true, draws: 0, triangles: 0 },
    road: { enabled: true, draws: 0, triangles: 0 },
    zones: { enabled: true, draws: 0, triangles: 0 }
  },
  textures: { manifest: RESOLVED_TEXTURE_MANIFEST_URL, loaded: 0, required: 0 }
};

function pushVertex(target, point, offsetX, offsetY, offsetZ, color) {
  target.push(point[0] - PIT[0] + offsetX, -(point[1] - PIT[1] + offsetY), point[2] - PIT[2] + offsetZ, ...color);
}

function addRibbon(target, points, halfWidth, height, color, dashed = false) {
  for (let i = 0; i < points.length - 1; i++) {
    if (dashed && (i % 8) >= 5) continue;
    const a = points[i];
    const b = points[i + 1];
    const dx = b[0] - a[0];
    const dy = b[1] - a[1];
    const length = Math.hypot(dx, dy) || 1;
    const nx = -dy / length * halfWidth;
    const ny = dx / length * halfWidth;
    pushVertex(target, a, nx, ny, height, color);
    pushVertex(target, a, -nx, -ny, height, color);
    pushVertex(target, b, nx, ny, height, color);
    pushVertex(target, b, nx, ny, height, color);
    pushVertex(target, a, -nx, -ny, height, color);
    pushVertex(target, b, -nx, -ny, height, color);
  }
}

function addOffsetStripe(target, points, lateralOffset, halfWidth, height, color) {
  for (let i = 0; i < points.length - 1; i++) {
    const a = points[i];
    const b = points[i + 1];
    const dx = b[0] - a[0];
    const dy = b[1] - a[1];
    const length = Math.hypot(dx, dy) || 1;
    const nx = -dy / length;
    const ny = dx / length;
    const inner = lateralOffset - halfWidth;
    const outer = lateralOffset + halfWidth;
    pushVertex(target, a, nx * inner, ny * inner, height, color);
    pushVertex(target, a, nx * outer, ny * outer, height, color);
    pushVertex(target, b, nx * inner, ny * inner, height, color);
    pushVertex(target, b, nx * inner, ny * inner, height, color);
    pushVertex(target, a, nx * outer, ny * outer, height, color);
    pushVertex(target, b, nx * outer, ny * outer, height, color);
  }
}
function addBarrier(target, points, lateralOffset, color) {
  for (let i = 0; i < points.length - 1; i++) {
    const a = points[i]; const b = points[i + 1];
    const dx = b[0] - a[0]; const dy = b[1] - a[1];
    const length = Math.hypot(dx, dy) || 1;
    const nx = -dy / length * lateralOffset;
    const ny = dx / length * lateralOffset;
    pushVertex(target, a, nx, ny, 0.12, color);
    pushVertex(target, b, nx, ny, 0.12, color);
    pushVertex(target, a, nx, ny, 1.08, color);
    pushVertex(target, a, nx, ny, 1.08, color);
    pushVertex(target, b, nx, ny, 0.12, color);
    pushVertex(target, b, nx, ny, 1.08, color);
  }
  addOffsetStripe(target, points, lateralOffset, 0.16, 1.08, [0.52, 0.54, 0.52]);
}
function uploadTopology(data) {
  const vertices = [];
  for (const lane of data.lanes || []) {
    if (!Array.isArray(lane.points) || lane.points.length < 2) continue;
    const width = lane.role === 4 ? 15.5 : 13.5;
    addRibbon(vertices, lane.points, width, 0, [0.105, 0.12, 0.13]);
    addRibbon(vertices, lane.points, width + 0.85, -0.08, [0.34, 0.36, 0.35]);
    addRibbon(vertices, lane.points, 0.12, 0.07, [0.92, 0.77, 0.22], true);
    addOffsetStripe(vertices, lane.points, width - 0.55, 0.12, 0.055, [0.86, 0.88, 0.86]);
    addOffsetStripe(vertices, lane.points, -width + 0.55, 0.12, 0.055, [0.86, 0.88, 0.86]);
    addBarrier(vertices, lane.points, width + 0.72, [0.39, 0.41, 0.40]);
    addBarrier(vertices, lane.points, -width - 0.72, [0.39, 0.41, 0.40]);
  }
  const vao = gl.createVertexArray();
  const buffer = gl.createBuffer();
  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 24, 0);
  gl.enableVertexAttribArray(1);
  gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 24, 12);
  gl.bindVertexArray(null);
  return { vao, count: vertices.length / 6, triangles: vertices.length / 18 };
}

function addDisc(target, point, radius, height, color, steps = 12) {
  for (let i = 0; i < steps; i++) {
    const a = i / steps * Math.PI * 2;
    const b = (i + 1) / steps * Math.PI * 2;
    pushVertex(target, point, 0, 0, height, color);
    pushVertex(target, point, Math.cos(a) * radius, Math.sin(a) * radius, height, color);
    pushVertex(target, point, Math.cos(b) * radius, Math.sin(b) * radius, height, color);
  }
}

function uploadGameplayZones(data) {
  const vertices = [];
  for (const zone of data.trafficZones || []) {
    addRibbon(vertices, zone.centerline, 0.5, 0.85, [0.21, 0.89, 1.0]);
    for (const spawn of zone.spawnCandidates || []) {
      addDisc(vertices, spawn.position, 1.25, 1.0, [0.4, 1.0, 0.51], 8);
    }
  }
  for (const spawn of data.pitSpawnZones || []) {
    addDisc(vertices, spawn.position, 1.7, 1.15, [1.0, 0.68, 0.22], 10);
  }
  for (const teleport of data.teleportZones || []) {
    addDisc(vertices, teleport.position, 4.25, 1.35, [1.0, 0.31, 0.85], 16);
  }
  const vao = gl.createVertexArray();
  const buffer = gl.createBuffer();
  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 24, 0);
  gl.enableVertexAttribArray(1);
  gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 24, 12);
  gl.bindVertexArray(null);
  return { vao, count: vertices.length / 6, triangles: vertices.length / 18 };
}

function mapPoint(point) {
  return [
    ((point[0] + 45715.05078125) / 4.58755683898926) * 0.25,
    ((point[1] + 19981.3125) / 4.58755683898926) * 0.25
  ];
}

function teleportToZone(id) {
  const destination = gameplayZones?.teleportZones?.find(zone => zone.id === id);
  if (!destination) return null;
  focus = [destination.position[0] - PIT[0], -(destination.position[1] - PIT[1]), destination.position[2] - PIT[2] + 1.5];
  yaw = -destination.headingRadians + Math.PI;
  pitch = 0.5;
  distance = destination.kind === 'pit' ? 180 : 260;
  setOverview(false);
  teleportSelect.value = destination.id;
  window.dispatchEvent(new CustomEvent('nohesi415:teleport', { detail: structuredClone(destination) }));
  return destination;
}

function populateTeleportControls() {
  teleportSelect.replaceChildren();
  for (const zone of gameplayZones.teleportZones || []) {
    const option = document.createElement('option');
    option.value = zone.id;
    option.textContent = zone.label;
    teleportSelect.append(option);
  }
}

function resize() {
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.round(canvas.clientWidth * ratio));
  const height = Math.max(1, Math.round(canvas.clientHeight * ratio));
  if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
  gl.viewport(0, 0, width, height);
  return width / height;
}

function render() {
  const aspect = resize();
  const projection = glMatrix.mat4.create();
  const view = glMatrix.mat4.create();
  const viewProjection = glMatrix.mat4.create();
  glMatrix.mat4.perspective(projection, Math.PI / 3, aspect, 0.5, 80000);
  const cp = Math.cos(pitch);
  const eye = [focus[0] + distance * Math.cos(yaw) * cp, focus[1] + distance * Math.sin(yaw) * cp, focus[2] + distance * Math.sin(pitch) + 24];
  glMatrix.mat4.lookAt(view, eye, focus, [0, 0, 1]);
  glMatrix.mat4.multiply(viewProjection, projection, view);
  if (nightMode) gl.clearColor(0.024, 0.038, 0.068, 1); else gl.clearColor(0.38, 0.58, 0.76, 1);
  gl.clearDepth(1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.disable(gl.BLEND);
  gl.disable(gl.POLYGON_OFFSET_FILL);
  gl.enable(gl.DEPTH_TEST);
  gl.depthMask(true);
  gl.depthFunc(gl.LESS);
  gl.disable(gl.CULL_FACE);
  renderHealth.passes.environment.draws = 0;
  renderHealth.passes.road.draws = 0;
  renderHealth.passes.zones.draws = 0;
  if (!overviewVisible && recoveredEnvironment && renderHealth.passes.environment.enabled) {
    gl.useProgram(nativeProgram);
    gl.uniformMatrix4fv(nativeUniforms.matrix, false, viewProjection);
    gl.activeTexture(gl.TEXTURE0);
    gl.uniform1i(nativeUniforms.texture, 0);
    gl.uniform1f(nativeUniforms.night, nightMode ? 1 : 0);
    gl.bindVertexArray(recoveredEnvironment.vao);
    for (const segment of recoveredEnvironment.segments) {
      gl.bindTexture(gl.TEXTURE_2D, segment.style.texture);
      gl.uniform3fv(nativeUniforms.color, segment.style.color);
      gl.uniform1f(nativeUniforms.textured, segment.style.textured);
      gl.drawArrays(gl.TRIANGLES, segment.firstVertex, segment.vertexCount);
      renderHealth.passes.environment.draws++;
    }
  }
  if (!overviewVisible && recoveredRoad && renderHealth.passes.road.enabled) {
    // Environment recovery contains coplanar roadway fragments. The authored
    // road owns those pixels, so draw it last with LEQUAL and a small bias.
    gl.depthFunc(gl.LEQUAL);
    gl.enable(gl.POLYGON_OFFSET_FILL);
    gl.polygonOffset(-1.0, -1.0);
    gl.useProgram(nativeProgram);
    gl.uniformMatrix4fv(nativeUniforms.matrix, false, viewProjection);
    gl.activeTexture(gl.TEXTURE0);
    gl.uniform1i(nativeUniforms.texture, 0);
    gl.uniform1f(nativeUniforms.night, nightMode ? 1 : 0);
    gl.bindVertexArray(recoveredRoad.vao);
    for (const mesh of recoveredRoad.meshes) {
      gl.bindTexture(gl.TEXTURE_2D, mesh.style.texture);
      gl.uniform3fv(nativeUniforms.color, mesh.style.color);
      gl.uniform1f(nativeUniforms.textured, mesh.style.textured);
      gl.drawArrays(gl.TRIANGLES, mesh.firstVertex, mesh.vertexCount);
      renderHealth.passes.road.draws++;
    }
    gl.disable(gl.POLYGON_OFFSET_FILL);
    gl.depthFunc(gl.LESS);
  }
  if (!overviewVisible && zonesVisible && zoneMesh && renderHealth.passes.zones.enabled) {
    gl.depthFunc(gl.LEQUAL);
    gl.enable(gl.POLYGON_OFFSET_FILL);
    gl.polygonOffset(-2.0, -2.0);
    gl.useProgram(program);
    gl.uniformMatrix4fv(matrixLocation, false, viewProjection);
    gl.uniform1f(nightLocation, nightMode ? 1 : 0);
    gl.bindVertexArray(zoneMesh.vao);
    gl.drawArrays(gl.TRIANGLES, 0, zoneMesh.count);
    gl.disable(gl.POLYGON_OFFSET_FILL);
    gl.depthFunc(gl.LESS);
    renderHealth.passes.zones.draws = 1;
  }
  renderHealth.frame++;
  renderHealth.glError = gl.getError();
  requestAnimationFrame(render);
}

function drawOverview() {
  if (!topology || !topologyCanvas) return;
  const context = topologyCanvas.getContext('2d');
  context.clearRect(0, 0, topologyCanvas.width, topologyCanvas.height);
  context.lineJoin = 'round'; context.lineCap = 'round';
  for (const lane of topology.lanes || []) {
    context.beginPath();
    context.strokeStyle = lane.role === 4 ? '#ffd04d' : '#32e6ff';
    context.lineWidth = lane.role === 4 ? 2.5 : 2;
    lane.points.forEach((point, index) => {
      const [x, y] = mapPoint(point);
      if (index) context.lineTo(x, y); else context.moveTo(x, y);
    });
    context.stroke();
  }
  if (!zonesVisible || !gameplayZones) return;
  const drawPoints = (items, radius, color) => {
    context.fillStyle = color;
    for (const item of items) {
      const [x, y] = mapPoint(item.position);
      context.beginPath();
      context.arc(x, y, radius, 0, Math.PI * 2);
      context.fill();
    }
  };
  drawPoints(gameplayZones.trafficZones.flatMap(zone => zone.spawnCandidates || []), 1.5, '#65ff82');
  drawPoints(gameplayZones.pitSpawnZones, 2, '#ffae38');
  drawPoints(gameplayZones.teleportZones, 4, '#ff4fd8');
}

function setOverview(show) {
  overviewVisible = Boolean(show);
  overview.hidden = !overviewVisible;
  canvas.hidden = overviewVisible;
  toggleOverview.textContent = overviewVisible ? 'Show reconstructed 3D highway' : 'Show authored overview';
}

canvas.addEventListener('pointerdown', event => { dragging = true; lastPointer = [event.clientX, event.clientY]; canvas.setPointerCapture(event.pointerId); });
canvas.addEventListener('pointermove', event => {
  if (!dragging || !lastPointer) return;
  yaw -= (event.clientX - lastPointer[0]) * 0.006;
  pitch = Math.max(-0.1, Math.min(1.35, pitch - (event.clientY - lastPointer[1]) * 0.006));
  lastPointer = [event.clientX, event.clientY];
});
canvas.addEventListener('pointerup', () => { dragging = false; lastPointer = null; });
canvas.addEventListener('wheel', event => { event.preventDefault(); distance = Math.max(35, Math.min(12000, distance * Math.exp(event.deltaY * 0.001))); }, { passive: false });
canvas.addEventListener('webglcontextlost', event => {
  event.preventDefault();
  renderHealth.contextLost = true;
  status.textContent = 'Render failed: WebGL context lost. Reload required; no fallback renderer was started.';
});
toggleOverview.addEventListener('click', () => setOverview(!overviewVisible));
document.querySelector('#reset').addEventListener('click', () => { yaw = -3.11; pitch = 0.62; distance = 600; focus = [...RECOVERED_PIT_FOCUS]; });
document.querySelector('#reload').addEventListener('click', () => location.reload());
toggleZonesButton.addEventListener('click', () => {
  zonesVisible = !zonesVisible;
  renderHealth.passes.zones.enabled = zonesVisible;
  toggleZonesButton.textContent = zonesVisible ? 'Hide gameplay zones' : 'Show gameplay zones';
  drawOverview();
});
teleportGoButton.addEventListener('click', () => teleportToZone(teleportSelect.value));
topologyCanvas.addEventListener('click', event => {
  if (!zonesVisible || !gameplayZones) return;
  const bounds = topologyCanvas.getBoundingClientRect();
  const pointer = [
    (event.clientX - bounds.left) * topologyCanvas.width / bounds.width,
    (event.clientY - bounds.top) * topologyCanvas.height / bounds.height
  ];
  let nearest = null;
  let nearestDistance = 18;
  for (const zone of gameplayZones.teleportZones || []) {
    const point = mapPoint(zone.position);
    const distanceToPointer = Math.hypot(point[0] - pointer[0], point[1] - pointer[1]);
    if (distanceToPointer < nearestDistance) { nearestDistance = distanceToPointer; nearest = zone; }
  }
  if (nearest) teleportToZone(nearest.id);
});
moreButton.disabled = false;
moreButton.textContent = 'Switch to CSP night';
moreButton.addEventListener('click', () => {
  nightMode = !nightMode;
  moreButton.textContent = nightMode ? 'Switch to daylight' : 'Switch to CSP night';
});

async function bootstrap() {
  status.textContent = 'Loading authored 415 traffic splines…';
  const response = await fetch(TOPOLOGY_URL, { cache: 'no-store' });
  if (!response.ok) throw new Error(`Topology request failed (${response.status})`);
  topology = await response.json();
  const zonesResponse = await fetch(GAMEPLAY_ZONES_URL, { cache: 'no-store' });
  if (!zonesResponse.ok) throw new Error(`Gameplay zones request failed (${zonesResponse.status})`);
  gameplayZones = await zonesResponse.json();
  if (gameplayZones.schema !== 'nohesi-415-gameplay-zones/v1') throw new Error('Unsupported 415 gameplay-zone catalogue.');
  zoneMesh = uploadGameplayZones(gameplayZones);
  populateTeleportControls();
  let nearest = null;
  let nearestDistanceSq = Infinity;
  for (const lane of topology.lanes || []) {
    for (const point of lane.points || []) {
      const dx = point[0] - PIT[0];
      const dy = point[1] - PIT[1];
      const dz = point[2] - PIT[2];
      const distanceSq = dx * dx + dy * dy + dz * dz;
      if (distanceSq < nearestDistanceSq) { nearestDistanceSq = distanceSq; nearest = point; }
    }
  }
  if (nearest) focus = [nearest[0] - PIT[0], -(nearest[1] - PIT[1]), nearest[2] - PIT[2] + 1.5];
  recoveredRoad = await uploadRecoveredRoad();
  recoveredEnvironment = await uploadRecoveredEnvironment();
  const roadMaterials = new Set(recoveredRoad.meshes.map(mesh => mesh.material));
  const requiredMaterials = new Set([
    ...roadMaterials,
    ...recoveredEnvironment.segments.map(segment => segment.material)
  ]);
  renderHealth.textures.required = requiredMaterials.size;
  status.textContent = `Loading ${requiredMaterials.size} CSP-resolved material textures...`;
  const resolvedMaterials = await loadResolvedMaterialTextures(requiredMaterials, roadMaterials);
  for (const mesh of recoveredRoad.meshes) {
    mesh.style = resolvedMaterialStyle(mesh.material, resolvedMaterials.diffuseByMaterial);
  }
  for (const segment of recoveredEnvironment.segments) {
    segment.style = resolvedMaterialStyle(segment.material, resolvedMaterials.diffuseByMaterial);
  }
  renderHealth.textures.loaded = resolvedMaterials.loadedCount;
  focus = [...RECOVERED_PIT_FOCUS];
  drawOverview();
  status.textContent = `${gameplayZones.stats.trafficZones} traffic zones · ${gameplayZones.stats.trafficTrainingSpawns} training spawns · ${gameplayZones.stats.pitSpawns} pit spawns · ${gameplayZones.stats.teleports} teleports`;
  renderHealth.passes.road.triangles = recoveredRoad.triangles;
  renderHealth.passes.environment.triangles = recoveredEnvironment.triangles;
  window.__noHesi415Inspector = {
    topology: () => topology,
    gameplayZones: () => gameplayZones,
    recoveredRoad,
    recoveredEnvironment,
    pit: PIT,
    teleport: teleportToZone,
    renderHealth,
    camera: () => ({ yaw, pitch, distance, focus: [...focus], nightMode }),
    setCamera: next => {
      if (Number.isFinite(next?.yaw)) yaw = next.yaw;
      if (Number.isFinite(next?.pitch)) pitch = Math.max(-0.1, Math.min(1.35, next.pitch));
      if (Number.isFinite(next?.distance)) distance = Math.max(35, Math.min(12000, next.distance));
      if (Array.isArray(next?.focus) && next.focus.length === 3 && next.focus.every(Number.isFinite)) focus = [...next.focus];
      if (typeof next?.nightMode === 'boolean') nightMode = next.nightMode;
      return { yaw, pitch, distance, focus: [...focus], nightMode };
    },
    setPassEnabled: (name, enabled) => {
      if (renderHealth.passes[name]) renderHealth.passes[name].enabled = Boolean(enabled);
      return renderHealth.passes;
    }
  };
  render();
}
bootstrap().catch(error => {
  status.textContent = `Render failed: ${error.message}. No fallback renderer was started.`;
  console.error(error);
});
