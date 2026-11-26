import assert from 'node:assert/strict';
import { resolveObjectURL } from 'node:buffer';
import { after, afterEach, before, beforeEach, test } from 'node:test';
import { Window } from 'happy-dom';
import React, { act } from 'react';
import { createServer } from 'vite';

let browser;
let server;
let App;
let createRoot;
let root;
let container;
let requests;
let cameraRequests;

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

before(async () => {
  browser = new Window({ url: 'http://127.0.0.1:5173', settings: {
    disableJavaScriptEvaluation: true, disableJavaScriptFileLoading: true, disableCSSFileLoading: true,
  } });
  for (const name of ['window', 'document', 'navigator', 'HTMLElement', 'Element', 'SVGElement', 'HTMLVideoElement']) {
    Object.defineProperty(globalThis, name, { configurable: true, value: name === 'window' ? browser : browser[name] });
  }
  globalThis.requestAnimationFrame = browser.requestAnimationFrame.bind(browser);
  globalThis.cancelAnimationFrame = browser.cancelAnimationFrame.bind(browser);
  globalThis.getComputedStyle = browser.getComputedStyle.bind(browser);
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  // Use the JavaScript animation fallback instead of the emulated WAAPI.
  delete browser.Element.prototype.animate;
  ({ createRoot } = await import('react-dom/client'));
  server = await createServer({
    server: { middlewareMode: true, hmr: false, ws: false },
    appType: 'custom',
    plugins: [{
      name: 'test-surface-viewer',
      enforce: 'pre',
      resolveId(source) {
        if (source === './components/ThreeDViewer') return '\0test-surface-viewer';
      },
      load(id) {
        if (id === '\0test-surface-viewer') return `import React from 'react';
          export default function Surface({ imageUrl }) {
            return React.createElement('div', { 'data-image-url': imageUrl || '' });
          }`;
      },
    }],
  });
  ({ default: App } = await server.ssrLoadModule('/src/App.jsx'));
});

after(async () => {
  await server?.close();
  await browser?.close();
});

beforeEach(async () => {
  requests = { analyze: [], augment: [] };
  cameraRequests = [];
  globalThis.fetch = (url, options) => {
    if (url === '/api/health') return Promise.resolve(Response.json({ status: 'ok' }));
    const pending = deferred();
    requests[url.split('/').at(-1)].push({ ...pending, file: options.body.get('file') });
    return pending.promise;
  };
  Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: {
    getUserMedia: (constraints) => {
      const pending = deferred();
      cameraRequests.push({ ...pending, constraints });
      return pending.promise;
    },
  } });
  container = document.createElement('div');
  document.body.appendChild(container);
  root = createRoot(container);
  await act(async () => root.render(React.createElement(App)));
});

afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
});

function button(label) {
  return [...container.querySelectorAll('button')].find(node => node.textContent.trim() === label
    || node.getAttribute('aria-label') === label || node.title === label);
}

async function click(label) {
  const node = button(label);
  assert.ok(node, `Missing button: ${label}`);
  await act(async () => node.click());
}

async function until(predicate) {
  for (let tries = 0; tries < 100; tries++) {
    if (predicate()) return;
    await act(async () => new Promise(resolve => setTimeout(resolve, 20)));
  }
  assert.ok(predicate(), 'The expected workflow did not appear');
}

async function tab(label, ready) {
  await click(label);
  await until(ready);
}

async function upload(name) {
  const input = container.querySelector('input[type=file]');
  assert.ok(input);
  Object.defineProperty(input, 'files', { configurable: true, value: [new File([name], name, { type: 'image/png' })] });
  await act(async () => input.dispatchEvent(new browser.Event('change', { bubbles: true })));
}

async function respond(pending, data, status = 200) {
  assert.ok(pending);
  await act(async () => pending.resolve(Response.json(data, { status })));
}

function classification(label) {
  return { model_trained: true, prediction: label, top_probability: 1,
    inference_time_ms: 1, all_predictions: [{ class: label, probability: 1 }] };
}

function augmentation(label) {
  return { steps: { original: 'data:image/png;base64,AA==' }, step_order: ['original'], step_labels: { original: label } };
}

function stream() {
  const tracks = [0, 0];
  const media = new browser.MediaStream();
  media.tracks = tracks;
  media.getTracks = () => tracks.map((_, index) => ({ stop() { tracks[index]++; } }));
  return media;
}

test('new image invalidates pending analysis, including its loading state', async () => {
  await upload('first.png');
  const firstPreview = container.querySelector('img[alt=Captured]').src;
  await tab('Augmentation', () => button('Run 3D Augmentation'));
  await tab('Scan', () => button('Clear image'));
  assert.equal(container.querySelector('img[alt=Captured]').src, firstPreview);
  assert.ok(resolveObjectURL(firstPreview));
  await click('Clear image');
  assert.equal(resolveObjectURL(firstPreview), undefined);
  await upload('second.png');
  assert.equal(requests.analyze[1].file.name, 'second.png');
  await respond(requests.analyze[0], classification('Old classification'));
  assert.match(container.textContent, /Analyzing/);
  assert.doesNotMatch(container.textContent, /Old classification/);
  await respond(requests.analyze[1], classification('New classification'));
  assert.match(container.textContent, /New classification/);
  assert.doesNotMatch(container.textContent, /Analyzing|Old classification/);
});

test('a cleared image cannot be restored by a late analysis failure', async () => {
  await upload('first.png');
  await click('Clear image');
  await upload('second.png');
  await respond(requests.analyze[1], classification('New classification'));
  await respond(requests.analyze[0], { detail: 'Old analysis error' }, 500);
  assert.match(container.textContent, /New classification/);
  assert.doesNotMatch(container.textContent, /Old analysis error/);
  await click('Clear image');
  assert.equal(container.querySelector('img[alt=Captured]'), null);
  assert.doesNotMatch(container.textContent, /New classification/);
  await tab('3D View', () => container.querySelector('[data-image-url]'));
  assert.equal(container.querySelector('[data-image-url]').dataset.imageUrl, '');
  await tab('Augmentation', () => container.textContent.includes('Capture or upload an image first'));
  assert.equal(button('Run 3D Augmentation'), undefined);
});

test('late classification cannot overwrite a newer result or restore a cleared image', async () => {
  await upload('first.png');
  await click('Clear image');
  await upload('second.png');
  await respond(requests.analyze[1], classification('Current classification'));
  await respond(requests.analyze[0], classification('Obsolete classification'));
  assert.match(container.textContent, /Current classification/);
  assert.doesNotMatch(container.textContent, /Obsolete classification/);
  await click('Clear image');
  await upload('third.png');
  await click('Clear image');
  await respond(requests.analyze[2], classification('Cleared classification'));
  assert.doesNotMatch(container.textContent, /Cleared classification/);
  assert.ok(button('Open Camera'));
});

test('only the latest augmentation for the current image can be displayed', async () => {
  await upload('first.png');
  await respond(requests.analyze[0], classification('First image'));
  await click('View 3D Augmentation');
  await until(() => button('Run 3D Augmentation'));
  await tab('Scan', () => button('Clear image'));
  await click('Clear image');
  await upload('second.png');
  await respond(requests.analyze[1], classification('Second image'));
  await tab('Augmentation', () => button('Run 3D Augmentation'));
  await click('Run 3D Augmentation');
  await click('Run 3D Augmentation');
  assert.deepEqual(requests.augment.map(request => request.file.name), ['first.png', 'second.png', 'second.png']);
  await respond(requests.augment[2], augmentation('Latest augmentation'));
  await respond(requests.augment[1], augmentation('Superseded augmentation'));
  await respond(requests.augment[0], augmentation('Old image augmentation'));
  assert.match(container.textContent, /Latest augmentation/);
  assert.doesNotMatch(container.textContent, /Superseded augmentation|Old image augmentation/);
  await click('Re-augment');
  await tab('Scan', () => button('Clear image'));
  await click('Clear image');
  await respond(requests.augment[3], augmentation('Cleared augmentation'));
  assert.ok(button('Open Camera'));
  await tab('Augmentation', () => container.textContent.includes('Capture or upload an image first'));
  assert.doesNotMatch(container.textContent, /Latest augmentation|Cleared augmentation/);
  assert.equal(button('Run 3D Augmentation'), undefined);
});

test('augmentation errors are visible on the augmentation tab and stale errors are ignored', async () => {
  await upload('first.png');
  await respond(requests.analyze[0], classification('First image'));
  await tab('Augmentation', () => button('Run 3D Augmentation'));
  await click('Run 3D Augmentation');
  await respond(requests.augment[0], { detail: 'Augmentation unavailable' }, 500);
  assert.equal(container.querySelector('[role=alert]').textContent, 'Augmentation unavailable');
  await click('Run 3D Augmentation');
  await click('Run 3D Augmentation');
  assert.equal(container.querySelector('[role=alert]'), null);
  await respond(requests.augment[2], augmentation('Successful retry'));
  await respond(requests.augment[1], { detail: 'Obsolete error' }, 500);
  assert.match(container.textContent, /Successful retry/);
  assert.equal(container.querySelector('[role=alert]'), null);
});

test('navigation while camera permission is pending stops the late stream', async () => {
  await click('Open Camera');
  assert.equal(cameraRequests.length, 1);
  await tab('About', () => container.querySelector('h2')?.textContent.includes('About DermScreen'));
  assert.equal(button('Open Camera'), undefined);
  const late = stream();
  await act(async () => cameraRequests[0].resolve(late));
  assert.deepEqual(late.tracks, [1, 1]);
  await tab('Scan', () => button('Open Camera'));
  assert.equal(container.querySelector('video'), null);
});

test('superseded camera requests and active streams are stopped', async () => {
  await click('Open Camera');
  await click('Open Camera');
  const old = stream();
  const current = stream();
  await act(async () => cameraRequests[1].resolve(current));
  await act(async () => cameraRequests[0].resolve(old));
  assert.equal(container.querySelector('video').srcObject, current);
  assert.deepEqual(old.tracks, [1, 1]);
  assert.deepEqual(current.tracks, [0, 0]);
  await click('Switch camera');
  assert.deepEqual(current.tracks, [1, 1]);
  assert.equal(cameraRequests[2].constraints.video.facingMode, 'user');
  const switched = stream();
  await act(async () => cameraRequests[2].resolve(switched));
  assert.equal(container.querySelector('video').srcObject, switched);
  await click('Stop camera');
  assert.deepEqual(switched.tracks, [1, 1]);
});

test('uploading while camera permission is pending stops the late stream', async () => {
  await click('Open Camera');
  await upload('selected.png');
  const late = stream();
  await act(async () => cameraRequests[0].resolve(late));
  assert.deepEqual(late.tracks, [1, 1]);
  assert.ok(container.querySelector('img[alt=Captured]'));
  assert.equal(container.querySelector('video'), null);
});

test('a captured frame selects an image and releases the camera', async () => {
  await click('Open Camera');
  const current = stream();
  await act(async () => cameraRequests[0].resolve(current));
  const canvas = container.querySelector('canvas');
  canvas.getContext = () => ({ drawImage() {} });
  canvas.toBlob = callback => callback(new Blob(['frame'], { type: 'image/jpeg' }));
  await click('Capture');
  assert.deepEqual(current.tracks, [1, 1]);
  assert.equal(requests.analyze.length, 1);
  assert.equal(await requests.analyze[0].file.text(), 'frame');
  assert.ok(container.querySelector('img[alt=Captured]'));
  assert.equal(container.querySelector('video'), null);
});

test('navigation releases the active camera and discards an unfinished frame', async () => {
  await click('Open Camera');
  const current = stream();
  await act(async () => cameraRequests[0].resolve(current));
  const canvas = container.querySelector('canvas');
  let finishFrame;
  canvas.getContext = () => ({ drawImage() {} });
  canvas.toBlob = callback => { finishFrame = callback; };
  await click('Capture');
  await tab('About', () => container.querySelector('h2')?.textContent.includes('About DermScreen'));
  assert.deepEqual(current.tracks, [1, 1]);
  await act(async () => finishFrame(new Blob(['obsolete frame'], { type: 'image/jpeg' })));
  assert.equal(requests.analyze.length, 0);
  await tab('Scan', () => button('Open Camera'));
  assert.equal(container.querySelector('img[alt=Captured]'), null);
});
