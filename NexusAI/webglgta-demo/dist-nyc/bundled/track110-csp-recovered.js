import { T as TrackSceneRenderer } from './track_scene_renderer-110-fixed.js';
import { g as glMatrix } from './shader_program-sEzzVVPq.js';

const { mat4, vec3 } = glMatrix;
const canvas = document.querySelector('#trackCanvas');
const status = document.querySelector('#status');
const gl = canvas.getContext('webgl2', { antialias: true, alpha: false, depth: true });
if (!gl) throw new Error('WebGL 2 is required for the 110 reconstruction.');

const DETAIL_SCENE_URL = 'assets/matrix-universe/world-packages/nohesi-110/scene/scene.json';
const OVERVIEW_SCENE_URL = 'assets/matrix-universe/world-packages/nohesi-110/overview/scene.json';
const START_FOCUS = [3432.5544433594, -3389.9748535156, -24.5];
const MAP_CENTER = [-963.05859375, -2941.4168701172, 50];
const MAP_BOUNDS = { minX: -7983.9931640625, maxX: 6057.8759765625, minY: -9832.884765625, maxY: 3950.0510253906 };
const OVERVIEW_ENTER_DISTANCE = 2600;
const OVERVIEW_EXIT_DISTANCE = 2200;
const WHOLE_MAP_AUTO_DISTANCE = 9000;
const INITIAL_WHOLE_MAP = new URLSearchParams(location.search).get('view') === 'whole';
const focusData = START_FOCUS.slice();
const dataToView = mat4.create();
mat4.rotateX(dataToView, dataToView, -Math.PI / 2);
const focusView = vec3.transformMat4(vec3.create(), focusData, dataToView);
const detailRenderer = new TrackSceneRenderer(gl);
const overviewRenderer = new TrackSceneRenderer(gl);
globalThis.__track110Ready = false;
globalThis.__track110OverviewReady = false;
let yaw = 0.72, pitch = 0.34, distance = 260;
let dragging = false, px = 0, py = 0;
let lastFrame = performance.now(), lastStreamUpdate = 0;
let overviewLoading = null, overviewVisible = false;
let wholeMapFramed = INITIAL_WHOLE_MAP;
const pressed = new Set();
const tiers = [
  { label: 'near', radiusM: 1050, retainRadiusM: 1450, maxLoaded: 72 },
  { label: 'wide', radiusM: 1900, retainRadiusM: 2450, maxLoaded: 192 },
  { label: 'regional', radiusM: 3600, retainRadiusM: 4300, maxLoaded: 512 },
];
let tierIndex = 0;
const projection = mat4.create(), view = mat4.create(), viewProjection = mat4.create();

function resize() {
  const scale = Math.min(devicePixelRatio || 1, overviewVisible ? 1.15 : 1.5);
  const width = Math.max(1, Math.floor(innerWidth * scale));
  const height = Math.max(1, Math.floor(innerHeight * scale));
  if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
  gl.viewport(0, 0, width, height);
}

function updateFocusView() {
  vec3.transformMat4(focusView, focusData, dataToView);
}

async function applyTier() {
  if (!detailRenderer.streaming) return;
  Object.assign(detailRenderer.streaming, tiers[tierIndex]);
  detailRenderer._lastAppliedStreamFocus = null;
  await detailRenderer.updateStreamingFocus(focusData);
  document.querySelector('#loadMore').textContent = `Detail: ${tiers[tierIndex].label}`;
}

async function ensureOverview() {
  if (globalThis.__track110OverviewReady) return true;
  if (overviewLoading) return overviewLoading;
  overviewLoading = (async () => {
    status.textContent = 'Loading compiled whole-map overview...';
    await overviewRenderer.init(dataToView);
    if (!await overviewRenderer.load(OVERVIEW_SCENE_URL)) {
      throw new Error(overviewRenderer.error || '110 overview scene did not load');
    }
    if (!overviewRenderer.models.length) throw new Error('110 overview contains no geometry');
    globalThis.__track110OverviewReady = {
      scene: 'nohesi-110-csp-overview-v1',
      residentCells: overviewRenderer.stats.sectors,
      totalCells: overviewRenderer.stats.totalSectors,
      compiledTriangles: overviewRenderer.stats.triangles,
      textureCount: overviewRenderer.textureCache.size,
      readyAt: new Date().toISOString(),
    };
    return true;
  })().catch(error => {
    overviewLoading = null;
    console.error(error);
    status.textContent = `Overview failed: ${error.message || error}`;
    status.style.color = '#ff8585';
    return false;
  });
  return overviewLoading;
}

function resetCamera() {
  yaw = 0.72; pitch = 0.34; distance = 260;
  overviewVisible = false;
  wholeMapFramed = false;
  focusData.splice(0, 3, ...START_FOCUS);
  updateFocusView();
  detailRenderer._lastAppliedStreamFocus = null;
  detailRenderer.updateStreamingFocus(focusData);
}

function fittedWholeMapDistance() {
  const mapWidth = MAP_BOUNDS.maxX - MAP_BOUNDS.minX;
  const mapHeight = MAP_BOUNDS.maxY - MAP_BOUNDS.minY;
  const aspect = Math.max(0.25, canvas.clientWidth / Math.max(1, canvas.clientHeight));
  const verticalTangent = Math.tan(Math.PI / 6);
  const fitHeight = mapHeight * 0.5 / verticalTangent;
  const fitWidth = mapWidth * 0.5 / (verticalTangent * aspect);
  // Perspective at the intentionally slight oblique angle expands the near
  // edge. The additional margin keeps every authored bound inside the canvas.
  return Math.min(58000, Math.max(fitHeight, fitWidth) * 1.7);
}

function showWholeMap() {
  // Align the nearly top-down view to the authored bounds. Rotating this
  // almost-square map by 41 degrees made its diagonal the limiting dimension
  // and left the complete map needlessly tiny on screen.
  yaw = 0;
  pitch = 1.45;
  distance = fittedWholeMapDistance();
  focusData.splice(0, 3, ...MAP_CENTER);
  updateFocusView();
  wholeMapFramed = true;
  ensureOverview();
}

document.querySelector('#reset').addEventListener('click', resetCamera);
document.querySelector('#wholeMap').addEventListener('click', showWholeMap);
document.querySelector('#loadMore').addEventListener('click', () => {
  tierIndex = (tierIndex + 1) % tiers.length;
  void applyTier();
});
addEventListener('keydown', event => pressed.add(event.key.toLowerCase()));
addEventListener('keyup', event => pressed.delete(event.key.toLowerCase()));
canvas.addEventListener('pointerdown', event => {
  dragging = true; px = event.clientX; py = event.clientY;
  canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener('pointerup', () => { dragging = false; });
canvas.addEventListener('pointermove', event => {
  if (!dragging) return;
  yaw -= (event.clientX - px) * 0.005;
  pitch = Math.max(-0.15, Math.min(1.48, pitch + (event.clientY - py) * 0.004));
  px = event.clientX; py = event.clientY;
});
canvas.addEventListener('wheel', event => {
  event.preventDefault();
  const previousDistance = distance;
  distance = Math.max(35, Math.min(58000, distance * Math.exp(event.deltaY * 0.001)));
  if (!wholeMapFramed && previousDistance < WHOLE_MAP_AUTO_DISTANCE && distance >= WHOLE_MAP_AUTO_DISTANCE) {
    focusData.splice(0, 3, ...MAP_CENTER);
    updateFocusView();
    pitch = Math.max(pitch, 1.05);
    wholeMapFramed = true;
  } else if (distance < WHOLE_MAP_AUTO_DISTANCE) {
    wholeMapFramed = false;
  }
  if (distance >= OVERVIEW_ENTER_DISTANCE) ensureOverview();
}, { passive: false });

function frame() {
  const overviewReady = Boolean(globalThis.__track110OverviewReady);
  if (!overviewVisible && overviewReady && distance >= OVERVIEW_ENTER_DISTANCE) overviewVisible = true;
  if (overviewVisible && distance <= OVERVIEW_EXIT_DISTANCE) overviewVisible = false;
  if (!overviewReady && distance >= OVERVIEW_ENTER_DISTANCE) ensureOverview();
  resize();
  const now = performance.now();
  const dt = Math.min(0.05, (now - lastFrame) / 1000);
  lastFrame = now;
  const baseSpeed = overviewVisible ? Math.max(900, distance * 0.16) : (pressed.has('shift') ? 900 : 280);
  const speed = baseSpeed * dt;
  const forward = (pressed.has('w') || pressed.has('arrowup') ? 1 : 0) - (pressed.has('s') || pressed.has('arrowdown') ? 1 : 0);
  const right = (pressed.has('d') || pressed.has('arrowright') ? 1 : 0) - (pressed.has('a') || pressed.has('arrowleft') ? 1 : 0);
  if (forward || right) {
    focusData[0] += (Math.cos(yaw) * forward - Math.sin(yaw) * right) * speed;
    focusData[1] += (Math.sin(yaw) * forward + Math.cos(yaw) * right) * speed;
    focusData[0] = Math.max(MAP_BOUNDS.minX, Math.min(MAP_BOUNDS.maxX, focusData[0]));
    focusData[1] = Math.max(MAP_BOUNDS.minY, Math.min(MAP_BOUNDS.maxY, focusData[1]));
    updateFocusView();
    if (!overviewVisible && now - lastStreamUpdate > 250) {
      lastStreamUpdate = now;
      detailRenderer.updateStreamingFocus(focusData);
    }
  }
  const cp = Math.cos(pitch);
  const eye = vec3.fromValues(
    focusView[0] + Math.cos(yaw) * cp * distance,
    focusView[1] + Math.sin(pitch) * distance,
    focusView[2] + Math.sin(yaw) * cp * distance
  );
  mat4.perspective(projection, Math.PI / 3, canvas.width / canvas.height, 0.5, 100000);
  mat4.lookAt(view, eye, focusView, [0, 1, 0]);
  mat4.multiply(viewProjection, projection, view);
  const activeRenderer = overviewVisible ? overviewRenderer : detailRenderer;
  // Always clear and draw in the same animation frame. The prior overview
  // throttle cleared/presented blank buffers between its 30 FPS draw frames.
  gl.clearColor(0.035, 0.06, 0.085, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  activeRenderer.render(viewProjection);
  const sceneStats = activeRenderer.stats;
  const renderStats = activeRenderer.getRenderStats();
  if (overviewVisible) {
    status.style.color = '#9fe8b2';
    status.textContent = `WHOLE MAP READY - ${sceneStats.sectors.toLocaleString()}/${sceneStats.totalSectors.toLocaleString()} compiled cells - ${Math.round(renderStats.triangles || 0).toLocaleString()} visible triangles - zero overview textures`;
  } else {
    const records = [...detailRenderer.textureCache.values()];
    const resident = records.filter(record => record.ready).length;
    const failed = records.filter(record => record.failed).length;
    const pending = Math.max(0, records.length - resident - failed);
    if (!globalThis.__track110Ready && sceneStats.sectors > 0 && (renderStats.triangles || 0) > 0 && records.length > 0 && pending === 0 && failed === 0) {
      globalThis.__track110Ready = {
        scene: 'nohesi-110-csp-full-r4',
        residentTiles: sceneStats.sectors,
        totalTiles: sceneStats.totalSectors,
        visibleTriangles: Math.round(renderStats.triangles || 0),
        residentTextures: resident,
        failedTextures: failed,
        readyAt: new Date().toISOString(),
      };
    }
    status.style.color = failed ? '#ff8585' : '#ffe08a';
    status.textContent = sceneStats.sectors
      ? `${sceneStats.sectors.toLocaleString()} resident / ${sceneStats.totalSectors.toLocaleString()} map tiles - ${Math.round(renderStats.triangles || 0).toLocaleString()} visible triangles - textures ${resident}/${records.length}${failed ? ` - ${failed} failed` : ''}`
      : 'Streaming nearby geometry...';
    if (globalThis.__track110Ready) status.textContent = `READY - ${status.textContent}`;
    if (overviewLoading && !overviewReady && distance >= OVERVIEW_ENTER_DISTANCE) {
      status.textContent = `Loading compiled whole-map overview - ${status.textContent}`;
    }
  }
  requestAnimationFrame(frame);
}

async function boot() {
  await detailRenderer.init(dataToView);
  detailRenderer.onProgress = () => {};
  globalThis.__track110Audit = {
    detailRenderer,
    overviewRenderer,
    detailSceneUrl: DETAIL_SCENE_URL,
    overviewSceneUrl: OVERVIEW_SCENE_URL,
    focus: focusData,
    tiers,
    mapBounds: MAP_BOUNDS,
    exportRevision: 'csp-full-r4-overview-v1',
  };
  if (INITIAL_WHOLE_MAP) {
    showWholeMap();
    requestAnimationFrame(frame);
    return;
  }
  if (!await detailRenderer.load(DETAIL_SCENE_URL)) throw new Error(detailRenderer.error || '110 scene metadata did not load');
  await applyTier();
  if (!detailRenderer.models.length) throw new Error(detailRenderer.error || 'No nearby 110 tiles loaded');
  requestAnimationFrame(frame);
}
boot().catch(error => {
  console.error(error);
  status.textContent = `Failed: ${error.message || error}`;
  status.style.color = '#ff8585';
});
