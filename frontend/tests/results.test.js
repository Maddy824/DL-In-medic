import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { createServer } from 'vite';

let server;
let Results;

before(async () => {
  server = await createServer({ server: { middlewareMode: true, hmr: false, ws: false }, appType: 'custom' });
  ({ default: Results } = await server.ssrLoadModule('/src/components/Results.jsx'));
});

after(async () => {
  await server?.close();
});

function render(result) {
  return renderToStaticMarkup(React.createElement(Results, { result }));
}

test('untrained result shows a notice without class scores', () => {
  const html = render({ model_trained: false });
  assert.match(html, /No trained model is loaded/);
  assert.match(html, /still explore the 3D surface and image augmentation/);
  assert.doesNotMatch(html, /Top Softmax Score|Class Scores|HIGH RISK|LOW RISK/);
});

test('legacy response without a trained flag cannot render predictions', () => {
  const html = render({ prediction: 'Melanoma', risk_level: 'critical', top_probability: 0.9 });
  assert.match(html, /No trained model is loaded/);
  assert.doesNotMatch(html, /Melanoma|90\.0%|CRITICAL/);
});

test('trained result shows experimental class scores, not risk', () => {
  const html = render({
    model_trained: true,
    prediction: 'Melanoma',
    top_probability: 0.6,
    inference_time_ms: 42,
    all_predictions: [{ class: 'Melanoma', probability: 0.6 }],
    disclaimer: 'Educational model output only',
  });
  assert.match(html, /Experimental top class/);
  assert.match(html, /60\.0%/);
  assert.match(html, /uncalibrated/);
  assert.doesNotMatch(html, /CRITICAL|HIGH RISK|LOW RISK/);
});
