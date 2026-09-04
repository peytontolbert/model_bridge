import childProcess from 'node:child_process';
import fs from 'node:fs';
import WebSocket from '/data/NexusAI/webglgta-demo/node_modules/ws/wrapper.mjs';

const port = 9238;
const mode = process.env.TRACK110_AUDIT_MODE === 'detail' ? 'detail' : 'overview';
const url = `http://127.0.0.1:5173/demo2/110.html?v=20260903-csp-overview-r9${mode === 'overview' ? '&view=whole' : ''}`;
const chrome = childProcess.spawn('/opt/google/chrome/chrome', [
  '--headless=new', '--no-sandbox', '--disable-dev-shm-usage',
  '--enable-unsafe-swiftshader', '--use-angle=swiftshader',
  '--window-size=1280,800', `--remote-debugging-port=${port}`,
  '--remote-allow-origins=*', `--user-data-dir=/tmp/map110-audit-${process.pid}`,
  'about:blank'
], { stdio: ['ignore', 'ignore', 'ignore'] });
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
let socket;
try {
  let page;
  for (let attempt = 0; attempt < 100; attempt++) {
    try {
      const targets = await fetch(`http://127.0.0.1:${port}/json/list`).then(r => r.json());
      page = targets.find(target => target.type === 'page');
      if (page) break;
    } catch {}
    await delay(100);
  }
  if (!page) throw new Error('Chrome page target unavailable');
  socket = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => { socket.once('open', resolve); socket.once('error', reject); });
  let id = 0;
  const pending = new Map(), failures = [], failedRequests = [], requestUrls = new Map();
  socket.on('message', raw => {
    const message = JSON.parse(String(raw));
    if (message.id && pending.has(message.id)) {
      const waiter = pending.get(message.id); pending.delete(message.id);
      message.error ? waiter.reject(new Error(message.error.message)) : waiter.resolve(message.result || {});
      return;
    }
    if (message.method === 'Runtime.exceptionThrown') failures.push(message.params?.exceptionDetails?.exception?.description || message.params?.exceptionDetails?.text);
    if (message.method === 'Log.entryAdded' && message.params?.entry?.level === 'error') failures.push(message.params.entry.text);
    if (message.method === 'Network.requestWillBeSent') {
      requestUrls.set(message.params?.requestId, message.params?.request?.url || '');
    }
    if (message.method === 'Network.loadingFailed') failedRequests.push({
      url: requestUrls.get(message.params?.requestId) || '',
      error: message.params?.errorText || 'request failed',
      canceled: Boolean(message.params?.canceled),
      type: message.params?.type || '',
    });
  });
  const send = (method, params = {}) => new Promise((resolve, reject) => {
    const requestId = ++id; pending.set(requestId, { resolve, reject });
    socket.send(JSON.stringify({ id: requestId, method, params }));
  });
  const evaluate = async expression => {
    const result = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
    if (result.exceptionDetails) throw new Error(result.exceptionDetails.text || 'evaluation failed');
    return result.result?.value;
  };
  await Promise.all([send('Page.enable'), send('Runtime.enable'), send('Log.enable'), send('Network.enable')]);
  await send('Page.navigate', { url });
  for (let attempt = 0; attempt < 180; attempt++) {
    const ready = mode === 'overview'
      ? await evaluate('Boolean(window.__track110OverviewReady)')
      : await evaluate('Boolean(window.__track110Ready)');
    if (ready) break;
    await delay(1000);
  }
  await delay(2000);
  const detailReport = await evaluate(`({
    status: document.querySelector('#status')?.textContent || '',
    models: window.__track110Audit?.detailRenderer?.models?.length || 0,
    focus: window.__track110Audit?.focus?.slice() || [],
    maxLoaded: window.__track110Audit?.detailRenderer?.streaming?.maxLoaded || 0,
    readiness: window.__track110Ready || false
  })`);
  if (mode === 'overview') await delay(5000);
  const report = await evaluate(`(() => {
    const api = window.__track110Audit;
    const detail = api?.detailRenderer;
    const overview = api?.overviewRenderer;
    const resources = performance.getEntriesByType('resource');
    return {
      status: document.querySelector('#status')?.textContent || '',
      readiness: window.__track110Ready || false,
      overviewReadiness: window.__track110OverviewReady || false,
      hasAuditApi: !!api,
      detail: {
        reportBeforeOverview: ${JSON.stringify(detailReport)},
        rendererError: detail?.error || '',
        models: detail?.models?.length ?? 0,
        stats: detail?.stats || null,
        renderStats: detail?.getRenderStats?.() || null,
        sceneUrl: detail?.sceneUrl || ''
      },
      overview: {
        rendererError: overview?.error || '',
        models: overview?.models?.length ?? 0,
        stats: overview?.stats || null,
        renderStats: overview?.getRenderStats?.() || null,
        sceneUrl: overview?.sceneUrl || '',
        textureRecords: overview?.textureCache?.size ?? -1
      },
      focus: api?.focus?.slice() || [],
      canvas: { width: document.querySelector('#trackCanvas')?.width, height: document.querySelector('#trackCanvas')?.height },
      resources: { count: resources.length, transfer: resources.reduce((n, e) => n + (e.transferSize || 0), 0) }
    };
  })()`);
  const capture = await send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: false });
  fs.writeFileSync(`/data/NexusAI/webglgta-demo/.codex-map110-${mode}-audit.png`, Buffer.from(capture.data, 'base64'));
  console.log(JSON.stringify({ mode, report, failures, failedRequests }, null, 2));
} finally {
  socket?.close();
  chrome.kill('SIGTERM');
}
