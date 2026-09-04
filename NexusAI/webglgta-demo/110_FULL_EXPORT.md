# No Hesi 110 full CSP export

## Authoritative inputs

- Geometry: `recovered/nohesi_110/component_runs/recovery_20260903_120328_022767`
- Textures: `recovered/nohesi_110/resolved_texture_runs/recovery_20260903_complete_110`
- Loader: Assetto Corsa CSP `SceneReference:loadKN5` via `tools/csp-kn5-recovery/kn5_recovery.lua`
- No RenderDoc or external GPU-capture software is part of this pipeline.

The geometry run contains all 24 requested KN5 models, 16,961 meshes,
42,164,397 vertices, 92,570,217 indices, and zero skipped meshes.  The texture
run contains 307/307 captured loader-resolved textures with strict fallback
rejection enabled.

CSP's runtime size report is off for 51 tall atlas records (for example,
`2048x8492` is reported for a valid `2048x8192` DDS). The browser finalizer
therefore records both reported dimensions and decoded DDS-header dimensions,
and uses the decoded DDS header as the authoritative size. DDS signature, byte
length, decoded image, output dimensions, and WebP decode are still validated.

## Browser compilation

`build_web_scene_from_csp.py` reads each CSP batch slice, validates every index,
applies the emitted `worldTransform`, converts AC `(X, up, Z)` to demo
`(X, Z, up)`, and writes gzip-compressed TNM v4 tiles.  Tiles are split per KN5
component and 384 m spatial cell, matching the successful 415 streaming shape.

Pit locators, tunnel audio volumes, collider-only KN5 geometry, and embedded
`M_Collider` meshes remain accounted for in `EXPORT_AUDIT.json` but are not
drawn as scenery.

```powershell
python K:\LANParty\tools\csp-kn5-recovery\finalize_resolved_textures_corrected.py `
  --run K:\LANParty\recovered\nohesi_110\resolved_texture_runs\recovery_20260903_complete_110 `
  --output K:\LANParty\recovered\nohesi_110\resolved_texture_runs\recovery_20260903_complete_110\web-1024-complete `
  --cache K:\LANParty\recovered\nohesi_110\resolved_texture_runs\recovery_20260903_complete_110\web-1024-complete.resume-cache `
  --max-web-dimension 1024 --workers 6

python K:\LANParty\tools\csp-kn5-recovery\build_web_scene_from_csp.py `
  --recovery K:\LANParty\recovered\nohesi_110\component_runs\recovery_20260903_120328_022767 `
  --web-textures K:\LANParty\recovered\nohesi_110\resolved_texture_runs\recovery_20260903_complete_110\web-1024-complete\manifest.json `
  --material-scene K:\LANParty\recovered\nohesi_110\runtime\scene.json `
  --output K:\LANParty\recovered\nohesi_110\web_csp_full_r4 `
  --tile-size 384

python K:\LANParty\tools\csp-kn5-recovery\validate_web_scene.py `
  K:\LANParty\recovered\nohesi_110\web_csp_full_r4\scene.json `
  --require-textures
```

## Required audit result

- 24/24 CSP component models complete
- 6,326 streamed visual tiles
- 12,932 visual material groups
- 42,161,717 packed visual vertices
- 30,634,683 visual triangles
- 100 explicitly separated nonvisual meshes (127,560 vertices, 666,168 indices)
- 303 distinct referenced browser textures backed by the 307-record recovery
- zero invalid indices, missing payloads, missing referenced textures, or fallbacks

The previously hosted 5,188-tile scene is not an authoritative export.  It was
compiled by the generic KN5 pipeline before the CSP component run completed and
must not be used as input for later revisions.

## Whole-map display LOD

`build_web_scene_overview.py` compiles a second TNM scene from the authoritative
full CSP scene. It never reads protected KN5 payloads. The pass merges 384 m
tiles into 1,536 m cells, welds matching source-tile seams, simplifies each
original material independently, and only then merges finished geometry into a
small semantic road/marking/ground/structure palette.

`Mountain_2D`, `Terrain_Horizon`, background cards, skyboxes, transparent glass,
and sub-pixel alpha foliage are excluded from the aerial display LOD. They are
not removed from the full-detail scene. Excluding `Terrain_Horizon` is required
to frame the playable map rather than its 20 km distant backdrop shell.

```powershell
python K:\LANParty\tools\csp-kn5-recovery\build_web_scene_overview.py `
  --scene K:\LANParty\recovered\nohesi_110\web_csp_full_r4\scene.json `
  --output K:\LANParty\recovered\nohesi_110\web_csp_overview_v6 `
  --cell-size 1536 `
  --final-reduction 0.82

python K:\LANParty\tools\csp-kn5-recovery\validate_web_scene.py `
  K:\LANParty\recovered\nohesi_110\web_csp_overview_v6\scene.json
```

Final overview audit:

- 46/46 compiled cells
- 1,415,412 vertices and 1,187,133 triangles
- 282 material groups (281 visible draw calls in the audited camera)
- 17.86 MiB compressed geometry, 51,048,332 decoded bytes
- zero overview texture allocations
- zero invalid indices, missing payloads, console errors, or failed requests

The viewer now uses this coarse package only past 10,500 m (with a 9,000 m
return threshold). Between 2,600 m and 10,500 m it uses the road-preserving
regional package described below, then returns to exact CSP detail inside
2,200 m. It exposes a `Whole map` button and supports a direct aerial URL:

`http://192.168.0.85:5173/demo2/110.html?v=20260903-csp-parity-r17&view=whole`

### Road-preserving regional LOD

The regional profile retains substantially more road and marking geometry,
keeps important transparent road/structure groups as stable opaque geometry,
and preserves merged-cell boundaries. It is compiled from the same complete
6,326-tile CSP scene:

```powershell
python K:\LANParty\tools\csp-kn5-recovery\build_web_scene_overview.py `
  --scene K:\LANParty\recovered\nohesi_110\web_csp_full_r4\scene.json `
  --output K:\LANParty\recovered\nohesi_110\web_csp_regional_v1 `
  --profile regional --cell-size 768 --final-reduction 0.35
```

- 144/144 compiled cells with preserved boundaries
- 3,657,006 vertices and 3,091,922 triangles
- 638 material groups
- 49,466,835 compressed package bytes
- zero invalid indices or missing payloads

The Assetto traffic coordinates were also checked directly against 11,018,999
recovered road vertices. The authored transform is the clear winner (1.74 m
median horizontal distance, 4.16 m p95); every tested flip or axis swap is
hundreds to thousands of metres worse. The map and lanes must not be flipped.

Viewer parity is handled separately from geometry alignment. The exporter packs
Assetto X/Y/Z as data X/Z/Y. The original viewer used a -90 degree X rotation,
which converted that data to WebGL X/Y/-Z and therefore mirrored the authored
road handedness. The corrected viewer uses the parity-preserving X/Y/Z mapping:
data X remains viewer X, data Z-up becomes viewer Y, and data Y remains viewer
Z without a sign inversion. Entrance and exit ramps consequently remain on
their authored sides. With this basis, aerial yaw 0 naturally presents the
requested 90-degree-clockwise map orientation and no projection reflection is
used in any LOD. Camera-relative movement and guide projection use the same
preserved Assetto X/Z basis.

## Traffic and closure guide overlay

`build_gameplay_overlay.py` reads the recovered gameplay-zone export and CSP
scene rather than inferring lanes from the display LOD. Those traffic
centerlines were exported from the installed track's `data/traffic.json` and
`data/traffic_connections.json`. Physical closures come from recovered
`Blockade_Prop` mesh batches with their emitted world transforms. Decorative
materials and `Light_Blocker_OG` are deliberately excluded.

```powershell
python K:\LANParty\tools\csp-kn5-recovery\build_gameplay_overlay.py `
  --gameplay-zones K:\LANParty\recovered\nohesi_110\runtime\gameplay-zones.json `
  --components K:\LANParty\recovered\nohesi_110\component_runs\recovery_20260903_120328_022767 `
  --output K:\LANParty\recovered\nohesi_110\viewer-guide-v1\gameplay-overlay.json
```

Generated guide audit:

- 101 authored traffic zones and 6,203 centerline points
- 74 physical blocker meshes grouped into 8 closure zones
- 628 training spawns, 101 pit spawns, and 219 teleports retained as metadata
- zero speculative lane-end cutoffs (none satisfied the blocker proximity test)

The viewer projects the guide with the same Assetto-to-WebGL transform as the
geometry. Cyan marks highway traffic, green marks main traffic, amber marks
local/parking traffic, and red X/bounds mark physical blocked zones. Direction
arrows follow the authored traffic point order. `Guides: on/off` toggles the
overlay, and `?guides=0` disables it on first load.

Live audited URL:

`http://192.168.0.85:5173/demo2/110.html?v=20260903-csp-basis-r21`

Both whole-map and close-detail audits complete with guide readiness
`101 / 6203 / 8 / 74`, zero console failures, and zero failed requests.

The source-preservation branch is `codex/110-csp-whole-map`. Deployed recovered
binary packages remain outside Git; the authoritative recovery runs are on
`K:\LANParty\recovered\nohesi_110`, and server-side pre-activation copies are
kept under `/data/NexusAI/webglgta-demo/backups/` and beside the live overview.
