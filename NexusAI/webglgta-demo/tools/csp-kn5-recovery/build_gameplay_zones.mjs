import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';

const root = resolve(process.argv[2] || 'recovered/nohesi_415');
const trackId = root.replace(/\\/g, '/').split('/').filter(Boolean).at(-1) || 'nohesi_track';
const topology = JSON.parse(await readFile(resolve(root, 'runtime/traffic-network.json'), 'utf8'));
const pitPath = process.argv[3]
  ? resolve(process.argv[3])
  : resolve(root, trackId === 'nohesi_415' ? 'pits/manifest.json' : 'runtime/pit-instances.json');
const recoveredPits = JSON.parse(await readFile(pitPath, 'utf8'));
const outputPath = resolve(root, 'runtime/gameplay-zones.json');

const parsePoint = value => Array.isArray(value)
  ? value.map(Number)
  : String(value).trim().split(/\s+/).map(Number);
const round = value => Number(value.toFixed(6));
const distance2d = (a, b) => Math.hypot(b[0] - a[0], b[1] - a[1]);
const normalize2d = (a, b) => {
  const length = distance2d(a, b) || 1;
  return [round((b[0] - a[0]) / length), round((b[1] - a[1]) / length)];
};
const boundsFor = points => ({
  min: [0, 1, 2].map(axis => Math.min(...points.map(point => point[axis]))),
  max: [0, 1, 2].map(axis => Math.max(...points.map(point => point[axis])))
});

function sampleLane(lane, spacing = 350, endpointMargin = 120) {
  const points = lane.points.map(parsePoint);
  const candidates = [];
  let total = 0;
  const cumulative = [0];
  for (let i = 1; i < points.length; i++) {
    total += distance2d(points[i - 1], points[i]);
    cumulative.push(total);
  }
  for (let target = endpointMargin; target <= total - endpointMargin; target += spacing) {
    let index = 1;
    while (index < cumulative.length - 1 && cumulative[index] < target) index++;
    const before = points[index - 1];
    const after = points[index];
    const segmentLength = Math.max(0.001, cumulative[index] - cumulative[index - 1]);
    const amount = Math.max(0, Math.min(1, (target - cumulative[index - 1]) / segmentLength));
    const position = [0, 1, 2].map(axis => round(before[axis] + (after[axis] - before[axis]) * amount));
    const forward = normalize2d(before, after);
    candidates.push({
      id: `traffic-${lane.id}-${String(candidates.length + 1).padStart(3, '0')}`,
      position,
      forward,
      headingRadians: round(Math.atan2(forward[1], forward[0])),
      clearanceMeters: 18
    });
  }
  return { points, totalLengthMeters: round(total), candidates };
}

const trafficZones = topology.lanes.map(lane => {
  const sampled = sampleLane(lane);
  return {
    id: `traffic-lane-${lane.id}`,
    label: `${lane.name} training corridor`,
    laneId: lane.id,
    role: lane.role,
    enabledForTraffic: true,
    enabledForTrainingSpawn: true,
    halfWidthMeters: lane.role === 4 ? 15.5 : 13.5,
    totalLengthMeters: sampled.totalLengthMeters,
    bounds: boundsFor(sampled.points),
    centerline: sampled.points,
    spawnCandidates: sampled.candidates
  };
});

const recoveredPitNodes = recoveredPits.meshes || recoveredPits.instances || [];
const uniquePitNodes = [...recoveredPitNodes
  .filter(mesh => /^AC_PIT_\d+$/i.test(mesh.name) && (mesh.worldTransform?.position || mesh.position))
  .reduce((byName, mesh) => {
    const previous = byName.get(mesh.name.toUpperCase());
    if (!previous || (mesh.nodeClass === 1 && previous.nodeClass !== 1)) byName.set(mesh.name.toUpperCase(), mesh);
    return byName;
  }, new Map()).values()];
const pitSpawnZones = uniquePitNodes
  .map(mesh => {
    const sourcePosition = (mesh.worldTransform?.position || mesh.position).map(Number);
    const sourceLook = (mesh.worldTransform?.look || mesh.forward || [0, 0, 1]).map(Number);
    const position = [sourcePosition[0], sourcePosition[2], sourcePosition[1]].map(round);
    const rawForward = [sourceLook[0], sourceLook[2]];
    const length = Math.hypot(...rawForward) || 1;
    const forward = rawForward.map(value => round(value / length));
    return {
      id: mesh.name.toLowerCase(),
      name: mesh.name,
      position,
      forward,
      headingRadians: round(Math.atan2(forward[1], forward[0])),
      radiusMeters: 2.25,
      source: `${pitPath.split(/[\\/]/).at(-1)} KN5 world transform`
    };
  })
  .sort((a, b) => Number(a.name.slice(7)) - Number(b.name.slice(7)));

const laneTeleports = [];
for (const zone of trafficZones) {
  const candidates = zone.spawnCandidates;
  if (!candidates.length) continue;
  const selections = [
    ['entry', 0],
    ['quarter', Math.floor((candidates.length - 1) * 0.25)],
    ['midpoint', Math.floor((candidates.length - 1) * 0.5)],
    ['three-quarter', Math.floor((candidates.length - 1) * 0.75)],
    ['exit', candidates.length - 1]
  ];
  const seen = new Set();
  for (const [section, index] of selections) {
    if (seen.has(index)) continue;
    seen.add(index);
    const candidate = candidates[index];
    laneTeleports.push({
      id: `teleport-lane-${zone.laneId}-${section}`,
      label: `${zone.label.replace(' training corridor', '')} ${section.replace('-', ' ')}`,
      kind: 'training',
      laneId: zone.laneId,
      position: candidate.position,
      forward: candidate.forward,
      headingRadians: candidate.headingRadians,
      triggerRadiusMeters: 7.5,
      safeSpawnCandidateId: candidate.id
    });
  }
}

const primaryPit = pitSpawnZones.find(zone => zone.name === 'AC_PIT_0') || pitSpawnZones[0];
const teleportZones = [
  {
    id: 'teleport-pit',
    label: 'Pit / staging area',
    kind: 'pit',
    position: primaryPit.position,
    forward: primaryPit.forward,
    headingRadians: primaryPit.headingRadians,
    triggerRadiusMeters: 10,
    pitSpawnId: primaryPit.id
  },
  ...laneTeleports
];

const result = {
  schema: `${trackId.replace(/_/g, '-')}-gameplay-zones/v1`,
  coordinateSystem: topology.coordinateSystem,
  generatedFrom: {
    traffic: 'topology/traffic-network.json',
    pits: pitPath
  },
  policies: {
    trafficSpawn: 'lane-aligned candidates spaced 350 m apart with 120 m endpoint exclusion',
    trainingSpawn: 'same safe candidates; consumers must run collision/occupancy checks before committing',
    teleport: 'named destinations resolve to an existing safe traffic or pit spawn'
  },
  trafficZones,
  pitSpawnZones,
  teleportZones,
  stats: {
    trafficZones: trafficZones.length,
    trafficCenterlinePoints: trafficZones.reduce((sum, zone) => sum + zone.centerline.length, 0),
    trafficTrainingSpawns: trafficZones.reduce((sum, zone) => sum + zone.spawnCandidates.length, 0),
    pitSpawns: pitSpawnZones.length,
    teleports: teleportZones.length
  }
};

await mkdir(dirname(outputPath), { recursive: true });
await writeFile(outputPath, `${JSON.stringify(result, null, 2)}\n`);
console.log(JSON.stringify({ outputPath, ...result.stats }));
