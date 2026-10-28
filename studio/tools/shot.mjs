// Headless screenshot at an exact viewport (phone sizes included) through the Chrome DevTools
// protocol. Usage: node tools/shot.mjs <url> <out.png> [width] [height] [waitMs] [scrollY]
// Needs Chrome; no npm dependencies (Node ≥ 22 has WebSocket).
import { spawn } from 'node:child_process';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const [url, out, w = '390', h = '844', waitMs = '4000', scrollY = '0'] = process.argv.slice(2);
if (!url || !out) {
  console.error('usage: node tools/shot.mjs <url> <out.png> [width] [height] [waitMs] [scrollY]');
  process.exit(2);
}
const chrome = process.env['CHROME'] ?? 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const port = 9300 + Math.floor(Math.random() * 500);
const profile = mkdtempSync(join(tmpdir(), 'shot-'));
const proc = spawn(chrome, [`--remote-debugging-port=${port}`, `--user-data-dir=${profile}`, '--headless=new', '--disable-gpu', '--hide-scrollbars', '--no-first-run', `--window-size=${Math.max(Number(w), 800)},${Math.max(Number(h), 600)}`, 'about:blank'], { stdio: 'ignore' });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function targets() {
  for (let i = 0; i < 50; i++) {
    try {
      const res = await fetch(`http://127.0.0.1:${port}/json`);
      const list = await res.json();
      const page = list.find((t) => t.type === 'page');
      if (page) {
        return page;
      }
    } catch {
      // not up yet
    }
    await sleep(200);
  }
  throw new Error('Chrome did not expose a page target');
}

async function main() {
  const page = await targets();
  const ws = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((r) => (ws.onopen = r));
  let id = 0;
  const pending = new Map();
  ws.onmessage = (m) => {
    const msg = JSON.parse(m.data);
    if (msg.id && pending.has(msg.id)) {
      pending.get(msg.id)(msg);
      pending.delete(msg.id);
    }
  };
  const send = (method, params = {}) =>
    new Promise((resolve) => {
      const n = ++id;
      pending.set(n, resolve);
      ws.send(JSON.stringify({ id: n, method, params }));
    });
  await send('Emulation.setDeviceMetricsOverride', { width: Number(w), height: Number(h), deviceScaleFactor: 2, mobile: Number(w) < 720 });
  await send('Emulation.setTouchEmulationEnabled', { enabled: Number(w) < 720 });
  await send('Page.enable');
  await send('Page.navigate', { url });
  await sleep(Number(waitMs));
  if (Number(scrollY) > 0) {
    await send('Runtime.evaluate', { expression: `window.scrollTo(0, ${Number(scrollY)}); document.querySelector('app-list-page, .content, main')?.scrollTo?.(0, ${Number(scrollY)});` });
    await sleep(400);
  }
  const shot = await send('Page.captureScreenshot', { format: 'png' });
  writeFileSync(out, Buffer.from(shot.result.data, 'base64'));
  console.log(`${out}: ${Number(w)}x${Number(h)}`);
  ws.close();
}

main()
  .catch((e) => {
    console.error(e);
    process.exitCode = 1;
  })
  .finally(() => proc.kill());
