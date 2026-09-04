import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';

const [inputArg, outputArg, idArg = 'nohesi-110'] = process.argv.slice(2);
if (!inputArg || !outputArg) {
  throw new Error('usage: node repair_scene_manifest.mjs INPUT OUTPUT [TRACK_ID]');
}

const inputPath = resolve(inputArg);
const outputPath = resolve(outputArg);
const scene = JSON.parse(await readFile(inputPath, 'utf8'));
if (!Array.isArray(scene.models) || scene.models.length === 0) throw new Error('scene has no models');

const min = [Infinity, Infinity, Infinity];
const max = [-Infinity, -Infinity, -Infinity];
let groupCount = 0;
let boundGroupCount = 0;
let totalTriangles = 0;
let unboundTriangles = 0;
const missingMaterials = new Map();

for (const model of scene.models) {
  const bounds = model.bounds;
  const modelMin = Array.isArray(bounds?.min)
    ? bounds.min
    : [bounds?.minX, bounds?.minY, bounds?.minZ];
  const modelMax = Array.isArray(bounds?.max)
    ? bounds.max
    : [bounds?.maxX, bounds?.maxY, bounds?.maxZ];
  if (![...modelMin, ...modelMax].every(Number.isFinite)) {
    throw new Error(`model ${model.file} has invalid bounds`);
  }
  for (let axis = 0; axis < 3; axis++) {
    min[axis] = Math.min(min[axis], modelMin[axis]);
    max[axis] = Math.max(max[axis], modelMax[axis]);
  }
  totalTriangles += Number(model.trianglesOutput || 0);
  for (const group of model.groups || []) {
    groupCount++;
    const triangles = Number(group.count || 0) / 3;
    if (group.materialBinding && scene.materialBindings?.[group.materialBinding]) {
      boundGroupCount++;
    } else {
      unboundTriangles += triangles;
      const key = `${group.material || '(unnamed)'}\u0000${group.shader || '(unknown)'}`;
      const item = missingMaterials.get(key) || {
        material: group.material || '(unnamed)',
        shader: group.shader || '(unknown)',
        groups: 0,
        triangles: 0,
      };
      item.groups++;
      item.triangles += triangles;
      missingMaterials.set(key, item);
    }
  }
}

scene.id = idArg;
scene.bounds = {
  minX: min[0], minY: min[1], minZ: min[2],
  maxX: max[0], maxY: max[1], maxZ: max[2],
};
scene.recovery = {
  track: idArg,
  policy: 'strict-authoritative-assets-only',
  placeholdersAllowed: false,
  geometrySource: 'preliminary installed KN5 scene conversion',
  geometryComplete: false,
  geometryIssue: 'legacy conversion contains coordinates clamped to its declared bounds; replace affected meshes from CSP runtime capture',
  boundsStatus: 'preliminary-until-runtime-geometry-finalization',
  textureSource: 'installed KN5 plus CSP runtime-resolved capture',
};

const report = {
  schema: 'track-scene-recovery-report/v1',
  track: idArg,
  inputPath,
  outputPath,
  bounds: scene.bounds,
  modelCount: scene.models.length,
  groupCount,
  boundGroupCount,
  unboundGroupCount: groupCount - boundGroupCount,
  totalTriangles,
  unboundTriangles,
  missingMaterials: [...missingMaterials.values()]
    .sort((a, b) => b.triangles - a.triangles),
};

await mkdir(dirname(outputPath), { recursive: true });
await writeFile(outputPath, JSON.stringify(scene));
await writeFile(`${outputPath}.report.json`, `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify({
  outputPath,
  bounds: report.bounds,
  modelCount: report.modelCount,
  groupCount: report.groupCount,
  unboundGroupCount: report.unboundGroupCount,
  unboundTriangles: report.unboundTriangles,
}, null, 2));
