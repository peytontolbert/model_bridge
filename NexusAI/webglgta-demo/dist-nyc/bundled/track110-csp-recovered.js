import { T as TrackSceneRenderer } from './track_scene_renderer-110-fixed.js';
import { g as glMatrix } from './shader_program-sEzzVVPq.js';

const { mat4, vec3 } = glMatrix;
const canvas = document.querySelector('#trackCanvas');
const guideCanvas = document.querySelector('#guideCanvas');
const guideContext = guideCanvas.getContext('2d');
const status = document.querySelector('#status');
const gl = canvas.getContext('webgl2', { antialias: true, alpha: false, depth: true });
if (!gl) throw new Error('WebGL 2 is required for the 110 reconstruction.');

const DETAIL_SCENE_URL = 'assets/matrix-universe/world-packages/nohesi-110/scene/scene.json';
const REGIONAL_SCENE_URL = 'assets/matrix-universe/world-packages/nohesi-110/regional/scene.json';
const OVERVIEW_SCENE_URL = 'assets/matrix-universe/world-packages/nohesi-110/overview/scene.json';
const GUIDE_URL = 'assets/matrix-universe/world-packages/nohesi-110/guide/gameplay-overlay.json';
// AC_PIT_0 recovered directly from PITS.kn5. The WebGL scene stores the
// Assetto coordinates as [x, z, y]; lift the focus 1.68 m to eye height.
const PIT_SPAWN = [3432.5544433594, -3389.9748535156, -24.5];
// AC_PIT_0's authored forward vector resolves to yaw 0 in the viewer basis.
const PIT_YAW = 0;
// Present the authored aerial map a literal 90 degrees clockwise from the
// X-right/Z-down calibration. The projection-X reflection below removes the
// right-handed camera mirror; yaw 0 then gives screen X = -Z, screen Y = X.
// Detail/pit view remains a normal, non-reflected 3D camera.
const ASSETTO_MAP_YAW = 0;
const START_FOCUS = PIT_SPAWN;
const MAP_CENTER = [-963.05859375, -2941.4168701172, 50];
const MAP_BOUNDS = { minX: -7983.9931640625, maxX: 6057.8759765625, minY: -9832.884765625, maxY: 3950.0510253906 };
const OVERVIEW_ENTER_DISTANCE = 2600;
const OVERVIEW_EXIT_DISTANCE = 2200;
const COARSE_ENTER_DISTANCE = 10500;
const COARSE_EXIT_DISTANCE = 9000;
const WHOLE_MAP_AUTO_DISTANCE = 9000;
const SOURCE_TILE_COUNT = 6326;
const SOURCE_TRIANGLE_COUNT = 30634683;
const INITIAL_VIEW = new URLSearchParams(location.search).get('view') || 'detail';
const INITIAL_WHOLE_MAP = INITIAL_VIEW === 'whole';
const INITIAL_REGIONAL_MAP = INITIAL_VIEW === 'regional';
const focusData = START_FOCUS.slice();
const dataToView = mat4.create();
mat4.rotateX(dataToView, dataToView, -Math.PI / 2);
const focusView = vec3.transformMat4(vec3.create(), focusData, dataToView);
const detailRenderer = new TrackSceneRenderer(gl);
const regionalRenderer = new TrackSceneRenderer(gl);
const overviewRenderer = new TrackSceneRenderer(gl);
globalThis.__track110Ready = false;
globalThis.__track110RegionalReady = false;
globalThis.__track110OverviewReady = false;
globalThis.__track110GuideReady = false;
let yaw = PIT_YAW, pitch = 0.24, distance = 45;
let dragging = false, px = 0, py = 0;
let lastFrame = performance.now(), lastStreamUpdate = 0;
let regionalLoading = null, overviewLoading = null, detailLoading = null;
let guideLoading = null, guideData = null;
let guideEnabled = new URLSearchParams(location.search).get('guides') !== '0';
let activeMode = 'detail';
let wholeMapFramed = INITIAL_WHOLE_MAP;
let assettoMapPresentation = INITIAL_WHOLE_MAP || INITIAL_REGIONAL_MAP;
const pressed = new Set();
const tiers = [
  { label: 'near', radiusM: 1050, retainRadiusM: 1450, maxLoaded: 72 },
  { label: 'wide', radiusM: 1900, retainRadiusM: 2450, maxLoaded: 192 },
  { label: 'regional', radiusM: 3600, retainRadiusM: 4300, maxLoaded: 512 },
];
let tierIndex = 0;
const projection = mat4.create(), view = mat4.create(), viewProjection = mat4.create();

function resize() {
  const scale = Math.min(devicePixelRatio || 1, activeMode === 'detail' ? 1.5 : 1.15);
  const width = Math.max(1, Math.floor(innerWidth * scale));
  const height = Math.max(1, Math.floor(innerHeight * scale));
  if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
  if (guideCanvas.width !== width || guideCanvas.height !== height) { guideCanvas.width = width; guideCanvas.height = height; }
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

async function ensureGuide() {
  if (guideData) return true;
  if (guideLoading) return guideLoading;
  guideLoading = fetch(GUIDE_URL, { cache: 'no-store' })
    .then(response => {
      if (!response.ok) throw new Error(`Guide data request failed (${response.status})`);
      return response.json();
    })
    .then(data => {
      if (data?.schema !== 'nohesi-110-viewer-guide/v1' || !Array.isArray(data.trafficZones)) {
        throw new Error('Guide data is invalid');
      }
      guideData = data;
      globalThis.__track110GuideReady = {
        trafficZones: data.stats?.trafficZones || data.trafficZones.length,
        centerlinePoints: data.stats?.trafficCenterlinePoints || 0,
        blockedZones: data.stats?.blockedZones || 0,
        blockerMeshes: data.stats?.blockerMeshes || 0,
        readyAt: new Date().toISOString(),
      };
      document.querySelector('#guideToggle').textContent = `Guides: ${guideEnabled ? 'on' : 'off'}`;
      return true;
    })
    .catch(error => {
      console.error(error);
      document.querySelector('#guideToggle').textContent = 'Guides: unavailable';
      return false;
    });
  return guideLoading;
}

async function ensureDetail() {
  if (detailRenderer.models.length) return true;
  if (detailLoading) return detailLoading;
  detailLoading = (async () => {
    status.textContent = 'Loading full-resolution CSP geometry near the pits...';
    if (!await detailRenderer.load(DETAIL_SCENE_URL)) throw new Error(detailRenderer.error || '110 detail scene did not load');
    await applyTier();
    return detailRenderer.models.length > 0;
  })().catch(error => {
    detailLoading = null;
    console.error(error);
    status.textContent = `Detail failed: ${error.message || error}`;
    status.style.color = '#ff8585';
    return false;
  });
  return detailLoading;
}

async function ensureRegional() {
  if (globalThis.__track110RegionalReady) return true;
  if (regionalLoading) return regionalLoading;
  regionalLoading = (async () => {
    status.textContent = 'Loading road-preserving regional geometry...';
    await regionalRenderer.init(dataToView);
    if (!await regionalRenderer.load(REGIONAL_SCENE_URL)) throw new Error(regionalRenderer.error || '110 regional scene did not load');
    if (!regionalRenderer.models.length) throw new Error('110 regional scene contains no geometry');
    globalThis.__track110RegionalReady = {
      scene: 'nohesi-110-csp-regional-v1',
      residentCells: regionalRenderer.stats.sectors,
      totalCells: regionalRenderer.stats.totalSectors,
      compiledTriangles: regionalRenderer.stats.triangles,
      sourceTilesRepresented: SOURCE_TILE_COUNT,
      sourceTriangles: SOURCE_TRIANGLE_COUNT,
      preservedCellBorders: true,
      readyAt: new Date().toISOString(),
    };
    return true;
  })().catch(error => {
    regionalLoading = null;
    console.error(error);
    status.textContent = `Regional scene failed: ${error.message || error}`;
    status.style.color = '#ff8585';
    return false;
  });
  return regionalLoading;
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
      sourceTilesRepresented: SOURCE_TILE_COUNT,
      sourceTriangles: SOURCE_TRIANGLE_COUNT,
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

async function resetCamera() {
  yaw = PIT_YAW; pitch = 0.24; distance = 45;
  activeMode = 'detail';
  wholeMapFramed = false;
  assettoMapPresentation = false;
  focusData.splice(0, 3, ...START_FOCUS);
  updateFocusView();
  await ensureDetail();
  detailRenderer._lastAppliedStreamFocus = null;
  void detailRenderer.updateStreamingFocus(focusData);
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
  yaw = ASSETTO_MAP_YAW;
  pitch = 1.45;
  distance = fittedWholeMapDistance();
  focusData.splice(0, 3, ...MAP_CENTER);
  updateFocusView();
  wholeMapFramed = true;
  assettoMapPresentation = true;
  void ensureOverview();
}

function showRegionalMap() {
  yaw = ASSETTO_MAP_YAW;
  pitch = 1.15;
  distance = 6000;
  focusData.splice(0, 3, ...MAP_CENTER);
  updateFocusView();
  wholeMapFramed = false;
  assettoMapPresentation = true;
  void ensureRegional();
}

document.querySelector('#reset').addEventListener('click', resetCamera);
document.querySelector('#wholeMap').addEventListener('click', showWholeMap);
document.querySelector('#loadMore').addEventListener('click', () => {
  tierIndex = (tierIndex + 1) % tiers.length;
  void applyTier();
});
document.querySelector('#guideToggle').addEventListener('click', () => {
  guideEnabled = !guideEnabled;
  document.querySelector('#guideToggle').textContent = `Guides: ${guideEnabled ? 'on' : 'off'}`;
  document.querySelector('#guideLegend').style.opacity = guideEnabled ? '1' : '0.45';
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
  if (distance >= COARSE_ENTER_DISTANCE) void ensureOverview();
  else if (distance >= OVERVIEW_ENTER_DISTANCE) void ensureRegional();
}, { passive: false });

function projectGuidePoint(point, matrix, lift = 2) {
  const x = Number(point?.[0]), y = Number(point?.[2]) + lift, z = -Number(point?.[1]);
  if (![x, y, z].every(Number.isFinite)) return null;
  const w = matrix[3] * x + matrix[7] * y + matrix[11] * z + matrix[15];
  if (w <= 0.01) return null;
  const clipX = (matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12]) / w;
  const clipY = (matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13]) / w;
  return [(clipX * 0.5 + 0.5) * guideCanvas.width, (0.5 - clipY * 0.5) * guideCanvas.height];
}

function drawDirectionArrow(points, matrix, color) {
  if (points.length < 3) return;
  const index = Math.max(1, Math.floor(points.length * 0.55));
  const from = projectGuidePoint(points[index - 1], matrix);
  const to = projectGuidePoint(points[index], matrix);
  if (!from || !to) return;
  const dx = to[0] - from[0], dy = to[1] - from[1];
  const length = Math.hypot(dx, dy);
  if (length < 2) return;
  const ux = dx / length, uy = dy / length;
  const size = Math.max(5, Math.min(10, 14000 / Math.max(1400, distance)));
  guideContext.beginPath();
  guideContext.moveTo(to[0] + ux * size, to[1] + uy * size);
  guideContext.lineTo(to[0] - ux * size - uy * size * 0.7, to[1] - uy * size + ux * size * 0.7);
  guideContext.lineTo(to[0] - ux * size + uy * size * 0.7, to[1] - uy * size - ux * size * 0.7);
  guideContext.closePath();
  guideContext.fillStyle = color;
  guideContext.fill();
}

function drawGuide(matrix) {
  guideContext.clearRect(0, 0, guideCanvas.width, guideCanvas.height);
  if (!guideEnabled || !guideData) return;
  const roleColors = { 4: '#48e8ff', 2: '#66ff9a', 0: '#ffd35c' };
  guideContext.lineJoin = 'round';
  guideContext.lineCap = 'round';
  guideContext.globalAlpha = activeMode === 'detail' ? 0.78 : 0.92;
  for (const zone of guideData.trafficZones) {
    const points = zone.centerline || [];
    const color = roleColors[zone.role] || '#77e6ff';
    guideContext.beginPath();
    let drawing = false;
    for (const point of points) {
      const screen = projectGuidePoint(point, matrix);
      if (!screen) { drawing = false; continue; }
      if (drawing) guideContext.lineTo(screen[0], screen[1]);
      else { guideContext.moveTo(screen[0], screen[1]); drawing = true; }
    }
    guideContext.strokeStyle = '#041017';
    guideContext.lineWidth = activeMode === 'detail' ? 6 : 4;
    guideContext.stroke();
    guideContext.strokeStyle = color;
    guideContext.lineWidth = activeMode === 'detail' ? 2.5 : 1.7;
    guideContext.stroke();
    drawDirectionArrow(points, matrix, color);
  }
  guideContext.globalAlpha = 0.92;
  guideContext.font = `${Math.max(11, Math.round(11 * Math.min(devicePixelRatio || 1, 1.5)))}px system-ui,sans-serif`;
  for (const zone of guideData.blockedZones || []) {
    const min = zone.bounds?.min, max = zone.bounds?.max;
    if (!min || !max) continue;
    const altitude = Math.max(min[2], max[2]) + 4;
    const corners = [
      [min[0], min[1], altitude], [max[0], min[1], altitude],
      [max[0], max[1], altitude], [min[0], max[1], altitude],
    ].map(point => projectGuidePoint(point, matrix, 0));
    if (corners.some(point => !point)) continue;
    guideContext.beginPath();
    guideContext.moveTo(corners[0][0], corners[0][1]);
    for (let index = 1; index < corners.length; index++) guideContext.lineTo(corners[index][0], corners[index][1]);
    guideContext.closePath();
    guideContext.fillStyle = '#ff263f55';
    guideContext.strokeStyle = '#ff4055';
    guideContext.lineWidth = activeMode === 'detail' ? 4 : 2.5;
    guideContext.fill();
    guideContext.stroke();
    const center = projectGuidePoint(zone.center, matrix, 5);
    if (center) {
      const size = activeMode === 'detail' ? 9 : 6;
      guideContext.beginPath();
      guideContext.moveTo(center[0] - size, center[1] - size);
      guideContext.lineTo(center[0] + size, center[1] + size);
      guideContext.moveTo(center[0] + size, center[1] - size);
      guideContext.lineTo(center[0] - size, center[1] + size);
      guideContext.strokeStyle = '#fff0f2';
      guideContext.lineWidth = 2;
      guideContext.stroke();
      if (distance < 5000) {
        guideContext.fillStyle = '#fff0f2';
        guideContext.fillText(zone.label, center[0] + size + 4, center[1] - 4);
      }
    }
  }
  guideContext.globalAlpha = 1;
}

function frame() {
  const regionalReady = Boolean(globalThis.__track110RegionalReady);
  const overviewReady = Boolean(globalThis.__track110OverviewReady);
  if (distance >= COARSE_ENTER_DISTANCE) {
    if (overviewReady) activeMode = 'coarse';
    else void ensureOverview();
  } else if (activeMode === 'coarse' && distance > COARSE_EXIT_DISTANCE) {
    // Hysteresis prevents rapid LOD swapping near the threshold.
  } else if (distance >= OVERVIEW_ENTER_DISTANCE) {
    if (regionalReady) activeMode = 'regional';
    else void ensureRegional();
  } else if (activeMode === 'regional' && distance > OVERVIEW_EXIT_DISTANCE) {
    // Keep regional active until the full-detail exit threshold is crossed.
  } else {
    activeMode = 'detail';
    if (!detailRenderer.models.length) void ensureDetail();
  }
  resize();
  const now = performance.now();
  const dt = Math.min(0.05, (now - lastFrame) / 1000);
  lastFrame = now;
  const baseSpeed = activeMode !== 'detail' ? Math.max(900, distance * 0.16) : (pressed.has('shift') ? 900 : 280);
  const speed = baseSpeed * dt;
  const forward = (pressed.has('w') || pressed.has('arrowup') ? 1 : 0) - (pressed.has('s') || pressed.has('arrowdown') ? 1 : 0);
  const right = (pressed.has('d') || pressed.has('arrowright') ? 1 : 0) - (pressed.has('a') || pressed.has('arrowleft') ? 1 : 0);
  if (forward || right) {
    // Convert viewer camera-relative motion back into Assetto X/Z. The old X
    // term had the opposite sign, making forward navigation feel mirrored.
    focusData[0] += (-Math.cos(yaw) * forward + Math.sin(yaw) * right) * speed;
    focusData[1] += (Math.sin(yaw) * forward + Math.cos(yaw) * right) * speed;
    focusData[0] = Math.max(MAP_BOUNDS.minX, Math.min(MAP_BOUNDS.maxX, focusData[0]));
    focusData[1] = Math.max(MAP_BOUNDS.minY, Math.min(MAP_BOUNDS.maxY, focusData[1]));
    updateFocusView();
    if (activeMode === 'detail' && now - lastStreamUpdate > 250) {
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
  // Match Assetto's authored map.png axes exactly. A yaw alone cannot remove
  // the reflection introduced when a right-handed 3D camera is projected into
  // Assetto's 2D X-right/Z-down map convention.
  // Never reflect full-detail geometry. The map-only reflection previously
  // leaked through after zooming in from Whole map, swapping real junction
  // handedness (for example, moving a right-side entrance ramp to the left).
  if (assettoMapPresentation && activeMode !== 'detail') projection[0] *= -1;
  mat4.lookAt(view, eye, focusView, [0, 1, 0]);
  mat4.multiply(viewProjection, projection, view);
  const activeRenderer = activeMode === 'coarse' ? overviewRenderer : activeMode === 'regional' ? regionalRenderer : detailRenderer;
  // Always clear and draw in the same animation frame. The prior overview
  // throttle cleared/presented blank buffers between its 30 FPS draw frames.
  gl.clearColor(0.035, 0.06, 0.085, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  activeRenderer.render(viewProjection);
  drawGuide(viewProjection);
  const sceneStats = activeRenderer.stats;
  const renderStats = activeRenderer.getRenderStats();
  if (activeMode === 'coarse') {
    status.style.color = '#9fe8b2';
    status.textContent = `EXTREME WHOLE-MAP LOD - ${sceneStats.sectors.toLocaleString()}/${sceneStats.totalSectors.toLocaleString()} cells compiled from the ${SOURCE_TILE_COUNT.toLocaleString()}-tile CSP scene - ${Math.round(renderStats.triangles || 0).toLocaleString()} triangles`;
  } else if (activeMode === 'regional') {
    status.style.color = '#9fe8b2';
    status.textContent = `ROAD-PRESERVING REGIONAL LOD - ${sceneStats.sectors.toLocaleString()}/${sceneStats.totalSectors.toLocaleString()} cells - preserved boundaries - ${Math.round(renderStats.triangles || 0).toLocaleString()} triangles`;
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
    if (regionalLoading && !regionalReady && distance >= OVERVIEW_ENTER_DISTANCE) {
      status.textContent = `Loading road-preserving regional geometry - ${status.textContent}`;
    } else if (overviewLoading && !overviewReady && distance >= COARSE_ENTER_DISTANCE) {
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
    regionalRenderer,
    overviewRenderer,
    detailSceneUrl: DETAIL_SCENE_URL,
    regionalSceneUrl: REGIONAL_SCENE_URL,
    overviewSceneUrl: OVERVIEW_SCENE_URL,
    guideUrl: GUIDE_URL,
    get guideData() { return guideData; },
    get guideEnabled() { return guideEnabled; },
    focus: focusData,
    tiers,
    mapBounds: MAP_BOUNDS,
    pitSpawn: PIT_SPAWN,
    get activeMode() { return activeMode; },
    get assettoMapPresentation() { return assettoMapPresentation; },
    get assettoMapProjectionActive() { return assettoMapPresentation && activeMode !== 'detail'; },
    pitYaw: PIT_YAW,
    assettoMapYaw: ASSETTO_MAP_YAW,
    coordinateParity: 'assetto-map-clockwise-90',
    exportRevision: 'csp-full-r4-regional-v1-overview-v1-guides-r20',
  };
  void ensureGuide();
  if (INITIAL_WHOLE_MAP) {
    showWholeMap();
    requestAnimationFrame(frame);
    return;
  }
  if (INITIAL_REGIONAL_MAP) {
    showRegionalMap();
    requestAnimationFrame(frame);
    return;
  }
  if (!await ensureDetail()) throw new Error(detailRenderer.error || 'No nearby 110 tiles loaded');
  requestAnimationFrame(frame);
}
boot().catch(error => {
  console.error(error);
  status.textContent = `Failed: ${error.message || error}`;
  status.style.color = '#ff8585';
});
