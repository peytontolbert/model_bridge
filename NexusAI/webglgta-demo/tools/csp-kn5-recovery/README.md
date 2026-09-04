# 110 CSP resolved texture recovery

This CSP Lua app performs an authorized, one-time recovery of textures from the
currently loaded `nohesi_110` track. It reads the live CSP `trackRoot`, after CSP
has applied the track's runtime material and texture resolution, and refuses to
run on any other track.

## Strict recovery rules

- Only meshes below the live `trackRoot` are scanned. CSP does not retain a
  reliable `$KN5` attribute on these runtime meshes, so blank source attributes
  are explicitly recorded as `runtime-track-root` instead of being skipped.
- Each live material slot and its CSP runtime texture reference is recorded.
- Named embedded textures are resolved through CSP's documented
  `track::track::<name>` image-source namespace; CSP temporary IDs are used
  directly while the session is alive.
- A texture is exported only if CSP reports it ready and larger than 1x1.
- Missing, invalid, timed-out, and 1x1 images are reported as unresolved.
- The app never creates placeholder or fallback textures.
- The original Assetto Corsa track files are read-only and are never modified.

## Output

Each run is written to:

`K:/LANParty/recovered/nohesi_110/resolved_texture_runs/recovery_<timestamp>`

The run contains:

- `resolved_####_<name>.dds`: GPU-resolved texture payloads captured from CSP.
- `manifest.json`: dimensions, byte lengths, status, runtime references, and
  material/mesh/slot provenance.
- `DONE.txt`: present only when every discovered runtime texture was captured.
- `INCOMPLETE.txt`: present if any discovered texture was rejected or unresolved.

The DDS files are the authoritative recovery output. Browser conversion and
material wiring must consume only entries marked `captured` in `manifest.json`.

After a run reaches `DONE.txt`, validate and convert it with:

```powershell
python K:\LANParty\tools\csp-kn5-recovery\finalize_resolved_textures.py
```

The finalizer refuses incomplete runs, validates every DDS signature, byte
length, decoded dimensions, and placeholder threshold, then creates
browser-optimized WebP files plus a material/slot binding manifest.
The authoritative DDS files retain their complete runtime resolution. Web
copies are compiled to a maximum 1024-pixel dimension at WebP quality 92 to
bound browser GPU and transfer cost; source dimensions, web dimensions, and
encoding settings are recorded in the manifest.
