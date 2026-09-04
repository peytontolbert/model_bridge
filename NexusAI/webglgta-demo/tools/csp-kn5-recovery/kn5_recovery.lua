local TARGET_TRACK = 'nohesi_110'
local RECOVERY_REVISION = 7
local OUTPUT_ROOT = 'K:/LANParty/recovered/nohesi_110/resolved_texture_runs'
local PROTECTED_ONLY = true
local OUTPUT_RESOLUTION = 4096
local CAPTURE_SETTLE_SECONDS = 2
local MIN_SCENE_MESHES = 16000
local EXPECTED_PROTECTED_TEXTURES = 17
local SCENE_STABLE_SECONDS = 10
local READINESS_POLL_SECONDS = 1

io.createDir(OUTPUT_ROOT)
io.save(OUTPUT_ROOT..'/APP_LOADED.txt', '110 texture recovery script loaded at '..os.date('%Y-%m-%d %H:%M:%S')..'\n', true)

local state = {
  running = false,
  finished = false,
  incomplete = false,
  failed = nil,
  message = 'Load nohesi_110, then start resolved texture recovery.',
  phase = 'idle',
  outputDir = nil,
  root = nil,
  meshes = nil,
  meshIndex = 0,
  meshCount = 0,
  textureIndex = 0,
  textures = {},
  textureByRef = {},
  textureOwnerByRef = {},
  bindingByKey = {},
  bindingCount = 0,
  capturedCount = 0,
  unresolvedCount = 0,
  skippedUnattributedMeshes = 0,
  currentTexture = nil,
  currentTextureStartedAt = 0,
  captureParent = nil,
  captureShot = nil,
  readinessNextPollAt = 0,
  readinessMeshCount = 0,
  readinessStableSince = nil,
  readinessProtectedCount = 0
}

local function cleanString(value)
  if value == nil then return '' end
  return tostring(value):gsub('[%z\1-\31]', '')
end

local function sourceBasename(value)
  local s = cleanString(value):gsub('\\', '/')
  return (s:match('([^/]+)$') or s):lower()
end

local function safeStem(value)
  local s = cleanString(value):gsub('\\', '/')
  s = s:match('([^/]+)$') or s
  s = s:gsub('%.[^%.]+$', '')
  s = s:gsub('[^%w%._%-]+', '_'):gsub('_+', '_')
  s = s:gsub('^_+', ''):gsub('_+$', '')
  if #s == 0 then s = 'texture' end
  if #s > 72 then s = s:sub(1, 72) end
  return s
end

local function recordUnresolved(texture, reason, width, height)
  texture.status = 'unresolved'
  texture.reason = reason
  texture.width = width or 0
  texture.height = height or 0
  state.unresolvedCount = state.unresolvedCount + 1
  state.currentTexture = nil
end

local function imageSourcesFor(textureRef, resolvedRuntimeSource)
  local sources = {}
  if resolvedRuntimeSource ~= nil and #resolvedRuntimeSource > 0 then
    sources[#sources + 1] = resolvedRuntimeSource
  end
  if textureRef:find('::', 1, true) or textureRef:match('^%a:[/\\]') or
      textureRef:sub(1, 1) == '/' or textureRef:sub(1, 1) == '$' then
    sources[#sources + 1] = textureRef
    return sources
  end
  sources[#sources + 1] = textureRef
  sources[#sources + 1] = 'track::track::'..textureRef
  sources[#sources + 1] = 'track::'..textureRef
  return sources
end

local function addBinding(mesh, sourceFile, material, shader, slotName, textureRef, resolvedRuntimeSource)
  if resolvedRuntimeSource ~= nil and #resolvedRuntimeSource > 0 then
    local owner = state.textureOwnerByRef[textureRef]
    if owner == nil or (not owner.mesh:isActive() and mesh:isActive()) then
      state.textureOwnerByRef[textureRef] = {
        mesh = mesh,
        sourceFile = sourceFile,
        material = material,
        sourceSlot = slotName,
        dumpedHandle = resolvedRuntimeSource
      }
    end
  end
  local bindingKey = table.concat({sourceFile, material, shader, slotName, textureRef}, '\0')
  local existing = state.bindingByKey[bindingKey]
  if existing ~= nil then
    existing.meshUseCount = existing.meshUseCount + 1
    return
  end

  local texture = state.textureByRef[textureRef]
  if texture == nil then
    texture = {
      id = #state.textures + 1,
      runtimeReference = textureRef,
      status = 'queued',
      width = 0,
      height = 0,
      outputFile = nil,
      imageSources = imageSourcesFor(textureRef, resolvedRuntimeSource),
      sourceAttempts = {},
      candidateIndex = 1,
      bindings = {}
    }
    state.textures[#state.textures + 1] = texture
    state.textureByRef[textureRef] = texture
  end

  local binding = {
    sourceFile = sourceFile,
    material = material,
    shader = shader,
    slot = slotName,
    firstMesh = cleanString(mesh:name()),
    meshUseCount = 1
  }
  texture.bindings[#texture.bindings + 1] = binding
  state.bindingByKey[bindingKey] = binding
  state.bindingCount = state.bindingCount + 1
end

local function dumpedSourcesFor(mesh)
  local ok, dump = pcall(function() return mesh:dumpShaderReplacements() end)
  if not ok or dump == nil then return {} end
  local result = {}
  local dumpString = tostring(dump)
  for slotName, runtimeSource in dumpString:gmatch('RESOURCE_%d+=([^,]+),([^\r\n]+)') do
    slotName = cleanString(slotName):gsub('^%s+', ''):gsub('%s+$', '')
    runtimeSource = runtimeSource:gsub('^%s+', ''):gsub('%s+$', '')
    -- CSP 0.2.11 places a private control byte between "dumped" and "::".
    -- It is part of the live image handle and must not be sanitized away.
    local dumped = runtimeSource:match('^(dumped.::%d+)') or runtimeSource:match('^(dumped::%d+)')
    if dumped ~= nil then result[slotName] = dumped end
  end
  if state.dumpDiagnostic == nil then
    local parsed = {}
    for key, value in pairs(result) do parsed[#parsed + 1] = key..'='..cleanString(value) end
    state.dumpDiagnostic = dumpString..'\nPARSED: '..table.concat(parsed, ', ')
  end
  return result
end

local function scanOneMesh()
  state.meshIndex = state.meshIndex + 1
  if state.meshIndex > state.meshCount then
    state.phase = 'export'
    state.textureIndex = 0
    state.currentTexture = nil
    state.message = string.format('Live scan complete: %d unique runtime textures in %d bindings.',
      #state.textures, state.bindingCount)
    io.save(state.outputDir..'/PROGRESS.txt', state.message..'\n', true)
    ac.log('[110 texture recovery] '..state.message)
    return
  end

  local mesh = state.meshes:at(state.meshIndex)
  local sourceFile = sourceBasename(mesh:getAttribute('$KN5'))
  if #sourceFile == 0 then
    state.skippedUnattributedMeshes = state.skippedUnattributedMeshes + 1
    sourceFile = 'runtime-track-root'
  end

  local material = cleanString(mesh:materialName())
  local shader = cleanString(mesh:shaderName())
  local slotCount = mesh:getTextureSlotsCount()
  local dumpedSources = nil
  for slotIndex = 1, slotCount do
    local slotName = cleanString(mesh:getTextureSlotName(nil, slotIndex))
    local textureRef = cleanString(mesh:getTextureSlotFilename(nil, slotIndex))
    local protected = textureRef:lower():find('encrypted_textures', 1, true) ~= nil
    if #slotName > 0 and #textureRef > 0 and (not PROTECTED_ONLY or protected) then
      if protected and dumpedSources == nil then dumpedSources = dumpedSourcesFor(mesh) end
      addBinding(mesh, sourceFile, material, shader, slotName, textureRef,
        dumpedSources ~= nil and dumpedSources[slotName] or nil)
    end
  end
  state.message = string.format('Scanning mesh %d/%d: %s', state.meshIndex,
    state.meshCount, cleanString(mesh:name()))
end

local function disposeCapture()
  if state.captureShot ~= nil then pcall(function() state.captureShot:dispose() end) end
  if state.captureParent ~= nil then pcall(function() state.captureParent:setParent(nil) end) end
  state.captureShot = nil
  state.captureParent = nil
end

local function hasNonClearPixels(data)
  if data == nil or #data <= 128 then return false end
  local pixelBytes = #data - 128
  local step = math.max(4, math.floor(pixelBytes / 65536 / 4) * 4)
  for i = 129, #data - 3, step do
    local r, g, b, a = data:byte(i, i + 3)
    if r ~= 255 or g ~= 0 or b ~= 255 or a ~= 255 then return true end
  end
  return false
end

local function prepareMaterialCapture(texture, owner)
  local dynamicRoot = ac.findNodes('dynamicRoot:yes')
  if dynamicRoot == nil or #dynamicRoot == 0 then error('CSP dynamic root is unavailable') end
  state.captureParent = dynamicRoot:createNode('__KN5_RECOVERY_110_TEXTURE_'..texture.id, false)
  state.captureParent:setPosition(vec3(0, 0, 0))
  local vertices = ac.VertexBuffer({
    ac.MeshVertex(vec3(-1, -1, 0), vec3(0, 0, 1), vec2(0, 1)),
    ac.MeshVertex(vec3( 1, -1, 0), vec3(0, 0, 1), vec2(1, 1)),
    ac.MeshVertex(vec3( 1,  1, 0), vec3(0, 0, 1), vec2(1, 0)),
    ac.MeshVertex(vec3(-1,  1, 0), vec3(0, 0, 1), vec2(0, 0))
  })
  local indices = ac.IndicesBuffer({0, 1, 2, 0, 2, 3})
  local quad = state.captureParent:createMesh('__KN5_RECOVERY_110_QUAD_'..texture.id,
    owner.material, vertices, indices, false, true)
  quad:setMaterialsFrom(owner.mesh)
  quad:ensureUniqueMaterials()
  quad:setCullMode(render.CullMode.None)
  quad:setVisible(true)
  quad:applyShaderReplacements('RESOURCE_0=txDiffuse,'..owner.dumpedHandle)
  state.captureShot = ac.GeometryShot(quad, vec2(OUTPUT_RESOLUTION, OUTPUT_RESOLUTION), 1, false)
  state.captureShot:setName('110 recovery texture '..texture.id)
  state.captureShot:setShadersType(render.ShadersType.SampleColor)
  state.captureShot:setClearColor(rgbm(1, 0, 1, 1))
  state.captureShot:setOrthogonalParams(vec2(2, 2), 10)
  state.captureShot:update(vec3(0, 0, 2), vec3(0, 0, -1), vec3(0, 1, 0), 0)
  state.currentTextureStartedAt = os.preciseClock()
end

local function exportCurrentTexture()
  if state.currentTexture == nil then
    state.textureIndex = state.textureIndex + 1
    if state.textureIndex > #state.textures then return false end
    state.currentTexture = state.textures[state.textureIndex]
    state.currentTexture.status = 'rendering'
    local owner = state.textureOwnerByRef[state.currentTexture.runtimeReference]
    if owner == nil then
      recordUnresolved(state.currentTexture, 'No live owner with a dumped session resource')
      return true
    end
    local ok, err = pcall(function() prepareMaterialCapture(state.currentTexture, owner) end)
    if not ok then
      disposeCapture()
      recordUnresolved(state.currentTexture, 'CSP material capture setup failed: '..cleanString(err))
      return true
    end
  end

  local texture = state.currentTexture
  state.message = string.format('Rendering protected texture %d/%d through CSP material API: %s',
    state.textureIndex, #state.textures, safeStem(texture.runtimeReference))
  if os.preciseClock() - state.currentTextureStartedAt < CAPTURE_SETTLE_SECONDS then return true end

  local ok, result = pcall(function() return state.captureShot:encode() end)
  if not ok or result == nil or #result < 128 or result:sub(1, 4) ~= 'DDS ' then
    disposeCapture()
    recordUnresolved(texture, ok and 'CSP GeometryShot returned invalid DDS data' or
      ('CSP GeometryShot encode failed: '..cleanString(result)))
    return true
  end
  if not hasNonClearPixels(result) then
    disposeCapture()
    recordUnresolved(texture, 'CSP GeometryShot contained only the magenta clear color',
      OUTPUT_RESOLUTION, OUTPUT_RESOLUTION)
    return true
  end

  local filename = string.format('resolved_%04d_%s.dds', texture.id, safeStem(texture.runtimeReference))
  if not io.save(state.outputDir..'/'..filename, result, true) then
    disposeCapture()
    recordUnresolved(texture, 'Failed to write captured DDS', OUTPUT_RESOLUTION, OUTPUT_RESOLUTION)
    return true
  end
  local owner = state.textureOwnerByRef[texture.runtimeReference]
  texture.status = 'captured'
  texture.width = OUTPUT_RESOLUTION
  texture.height = OUTPUT_RESOLUTION
  texture.byteLength = #result
  texture.outputFile = filename
  texture.resolvedImageSource = cleanString(owner.dumpedHandle)
  texture.captureMethod = 'CSP applyShaderReplacements + GeometryShot SampleColor'
  state.capturedCount = state.capturedCount + 1
  disposeCapture()
  state.currentTexture = nil
  return true
end

local function finishRecovery()
  local manifest = {
    format = 'CSP_RESOLVED_TEXTURE_RECOVERY_1',
    revision = RECOVERY_REVISION,
    track = ac.getTrackID(),
    trackFullID = ac.getTrackFullID(),
    cspVersion = ac.getPatchVersion(),
    source = 'scene-readiness-gated live CSP shader replacements rendered with GeometryShot',
    readiness = {
      minimumMeshCount = MIN_SCENE_MESHES,
      stableSeconds = SCENE_STABLE_SECONDS,
      expectedProtectedTextures = EXPECTED_PROTECTED_TEXTURES,
      observedMeshCount = state.readinessMeshCount,
      observedProtectedTextures = state.readinessProtectedCount
    },
    strictNoFallback = true,
    protectedOnly = PROTECTED_ONLY,
    meshCount = state.meshCount,
    bindingCount = state.bindingCount,
    textureCount = #state.textures,
    capturedCount = state.capturedCount,
    unresolvedCount = state.unresolvedCount,
    skippedUnattributedMeshes = state.skippedUnattributedMeshes,
    textures = state.textures
    ,dumpDiagnostic = state.dumpDiagnostic
  }
  local manifestPath = state.outputDir..'/manifest.json'
  if not io.save(manifestPath, JSON.stringify(manifest), true) then
    state.failed = 'Failed to write manifest: '..manifestPath
    state.running = false
    return
  end

  local emptyScan = #state.textures == 0 or state.bindingCount == 0
  state.incomplete = #state.textures ~= EXPECTED_PROTECTED_TEXTURES or
    state.bindingCount == 0 or state.unresolvedCount > 0
  local marker = emptyScan and 'INVALID_EMPTY_SCAN.txt' or
    (state.incomplete and 'INCOMPLETE.txt' or 'DONE.txt')
  io.save(state.outputDir..'/'..marker, string.format(
    '110 CSP resolved texture recovery\nCaptured: %d\nUnresolved: %d\nBindings: %d\n',
    state.capturedCount, state.unresolvedCount, state.bindingCount), true)
  state.running = false
  state.finished = true
  state.phase = 'done'
  state.root = nil
  state.meshes = nil
  state.textureOwnerByRef = {}
  state.message = string.format('Captured %d/%d resolved textures; %d unresolved.',
    state.capturedCount, #state.textures, state.unresolvedCount)
  ac.log('[110 texture recovery] '..state.message..' Output: '..state.outputDir)
  if state.incomplete then
    ui.toast(ui.Icons.Warning, '110 texture recovery incomplete: '..state.unresolvedCount..' unresolved')
  else
    ui.toast(ui.Icons.Confirm, '110 texture recovery complete: '..state.capturedCount..' textures')
  end
end

local function startRecovery(root, meshes)
  if ac.getTrackID() ~= TARGET_TRACK then
    state.failed = 'Refusing to run: current track is '..tostring(ac.getTrackID())..', expected '..TARGET_TRACK
    return
  end
  if root == nil or #root == 0 or meshes == nil or #meshes < MIN_SCENE_MESHES then
    state.failed = 'Live CSP track scene was not ready at capture start.'
    return
  end

  local runID = os.date('%Y%m%d_%H%M%S')..string.format('_%06d',
    math.floor(os.preciseClock() * 1000) % 1000000)
  state.outputDir = OUTPUT_ROOT..'/recovery_'..runID
  if not io.createDir(state.outputDir) and not io.dirExists(state.outputDir) then
    state.failed = 'Could not create output directory: '..state.outputDir
    return
  end

  ui.setAsynchronousImagesLoading(false)
  state.running = true
  state.finished = false
  state.incomplete = false
  state.failed = nil
  state.phase = 'scan'
  state.root = root
  state.meshes = meshes
  state.meshIndex = 0
  state.meshCount = #meshes
  state.textureIndex = 0
  state.textures = {}
  state.textureByRef = {}
  state.textureOwnerByRef = {}
  state.bindingByKey = {}
  state.bindingCount = 0
  state.capturedCount = 0
  state.unresolvedCount = 0
  state.skippedUnattributedMeshes = 0
  state.currentTexture = nil
  state.dumpDiagnostic = nil
  state.message = string.format('Scanning %d stable live-scene meshes...', state.meshCount)
  ac.log('[110 texture recovery] Starting. Output: '..state.outputDir)
end

local function countProtectedTextures(meshes)
  local unique = {}
  for i = 1, #meshes do
    local mesh = meshes:at(i)
    for j = 1, mesh:getTextureSlotsCount() do
      local ref = cleanString(mesh:getTextureSlotFilename(nil, j))
      if ref:lower():find('encrypted_textures', 1, true) ~= nil then unique[ref] = true end
    end
  end
  local count = 0
  for _ in pairs(unique) do count = count + 1 end
  return count
end

local function pollSceneReadiness()
  local now = os.preciseClock()
  if now < state.readinessNextPollAt then return end
  state.readinessNextPollAt = now + READINESS_POLL_SECONDS
  local root = ac.findNodes('trackRoot:yes')
  local meshes = root ~= nil and #root > 0 and root:findMeshes('?') or nil
  local meshCount = meshes ~= nil and #meshes or 0
  if meshCount ~= state.readinessMeshCount then
    state.readinessMeshCount = meshCount
    state.readinessStableSince = now
    state.readinessProtectedCount = 0
  elseif state.readinessStableSince == nil then
    state.readinessStableSince = now
  end
  local stableFor = now - state.readinessStableSince
  if meshCount < MIN_SCENE_MESHES or stableFor < SCENE_STABLE_SECONDS then
    state.message = string.format('Waiting for stable live scene: %d/%d meshes, %.1f/%.1f s stable.',
      meshCount, MIN_SCENE_MESHES, stableFor, SCENE_STABLE_SECONDS)
    return
  end
  state.readinessProtectedCount = countProtectedTextures(meshes)
  if state.readinessProtectedCount ~= EXPECTED_PROTECTED_TEXTURES then
    state.message = string.format('Scene stable, waiting for protected materials: %d/%d runtime textures.',
      state.readinessProtectedCount, EXPECTED_PROTECTED_TEXTURES)
    return
  end
  startRecovery(root, meshes)
end

function script.update(dt)
  if not state.running and not state.finished and not state.failed and ac.getTrackID() == TARGET_TRACK then
    pollSceneReadiness()
  end
  if not state.running then return end

  if state.phase == 'scan' then
    local deadline = os.preciseClock() + 0.006
    repeat
      scanOneMesh()
    until state.phase ~= 'scan' or os.preciseClock() >= deadline
  elseif state.phase == 'export' then
    if not exportCurrentTexture() then finishRecovery() end
  end
end

function windowMain(dt)
  ui.text('Authorized recovery target: '..TARGET_TRACK)
  ui.text('Current track: '..tostring(ac.getTrackID()))
  ui.separator()

  if state.running then
    local current, total
    if state.phase == 'scan' then
      current, total = state.meshIndex, state.meshCount
    else
      current, total = state.textureIndex, #state.textures
    end
    local progress = total > 0 and current / total or 0
    ui.progressBar(progress, vec2(-0.1, 20), string.format('%d / %d', current, total))
    ui.textWrapped(state.message)
    ui.text(string.format('Captured: %d   Unresolved: %d', state.capturedCount, state.unresolvedCount))
  elseif state.finished then
    local color = state.incomplete and rgbm(1, 0.72, 0.2, 1) or rgbm(0.3, 1, 0.4, 1)
    ui.textColored(state.message, color)
    ui.textWrapped('Output: '..state.outputDir)
    if state.incomplete then
      ui.textWrapped('No fallback files were created. See manifest.json and INCOMPLETE.txt for exact unresolved slots.')
    end
  elseif state.failed then
    ui.textColored(state.failed, rgbm(1, 0.3, 0.3, 1))
    if ui.button('Retry readiness gate') then
      state.failed = nil
      state.readinessStableSince = nil
      state.readinessNextPollAt = 0
    end
  else
    ui.textWrapped(state.message)
    ui.textWrapped(string.format('Gate: at least %d meshes, stable for %.0f seconds, with %d protected textures.',
      MIN_SCENE_MESHES, SCENE_STABLE_SECONDS, EXPECTED_PROTECTED_TEXTURES))
  end

  ui.separator()
  ui.textWrapped('Waits for the fully loaded live scene, then captures its CSP-resolved dumped texture handles. Images at 1x1 or unavailable are rejected and reported; no placeholders or fallback assets are emitted.')
end
