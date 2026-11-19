import { test } from '@e2e-dev/web';
import { expect } from 'e2e';
import { readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { CareResponse, ErrorResponse, PredictResponse } from '../src/api/client';

interface CnnGruReleaseManifest {
  model_version: string;
  threshold: number;
}

const fixtures = fileURLToPath(new URL('../../desktop/tests/fixtures/', import.meta.url));
const catalog = JSON.parse(readFileSync(new URL('../../infra/knowledge/care-v1.json', import.meta.url), 'utf8'));

function loadCnnGruManifest(): CnnGruReleaseManifest {
  const modelDir = process.env.TOMATO_E2E_MODEL_DIR;
  if (!modelDir) {
    throw new Error('Set TOMATO_E2E_MODEL_DIR to the exported CNN-GRU bundle directory.');
  }
  return JSON.parse(
    readFileSync(join(resolve(modelDir), 'cnn-gru-release.json'), 'utf8'),
  ) as CnnGruReleaseManifest;
}

test('CNN-GRU bundle serves calibrated predictions and dynamic threshold in app', async ({ app, browser, screen }) => {
  const manifest = loadCnnGruManifest();
  expect(typeof manifest.model_version).toBe('string');
  expect(typeof manifest.threshold).toBe('number');
  expect(Number.isFinite(manifest.threshold)).toBe(true);
  const leaf = process.env.TOMATO_E2E_LEAF_IMAGE;
  const nonLeaf = process.env.TOMATO_E2E_NON_LEAF_IMAGE;
  const classId = process.env.TOMATO_E2E_LEAF_CLASS_ID;
  if (!leaf || !nonLeaf || !classId) {
    throw new Error('Set TOMATO_E2E_LEAF_IMAGE, TOMATO_E2E_NON_LEAF_IMAGE and TOMATO_E2E_LEAF_CLASS_ID.');
  }

  const health = await fetch(new URL('/api/v1/health', app.baseUrl), {
    headers: { 'X-Session-Token': process.env.TOMATO_E2E_SESSION_TOKEN! },
    signal: AbortSignal.timeout(3000),
  });
  expect(health.status).toBe(200);
  expect(await health.json()).toEqual({
    schema_version: '1',
    ready: true,
    model_version: manifest.model_version,
    knowledge_version: catalog.knowledge_version,
    reason: null,
  });

  await app.open('/');
  await expect(screen.getByText(`Mô hình: ${manifest.model_version}`, { exact: true })).toBeVisible();

  const upload = screen.getByLabel('Chọn ảnh lá', { exact: true });
  const recognize = screen.getByRole('button', 'Nhận diện', { exact: true });
  const result = screen.getByRole('region', 'Kết quả nhận diện', { exact: true });
  const care = screen.getByRole('region', 'Thông tin chăm sóc', { exact: true });

  await upload.setInputFiles(leaf);
  const accepted = browser.waitForResponse('**/api/v1/predict');
  const careResponse = browser.waitForResponse(`**/api/v1/care/${classId}`);
  const careGuard = careResponse.catch(() => undefined);
  await recognize.tap();
  const acceptedResponse = await accepted;
  expect(acceptedResponse.status).toBe(200);
  const acceptedPrediction = (await acceptedResponse.json()) as PredictResponse;
  expect(acceptedPrediction).toMatchObject({
    status: 'ok',
    model_version: manifest.model_version,
    class_id: classId,
    care_card_id: classId,
  });
  expect(acceptedPrediction.top_k).toHaveLength(3);
  expect(acceptedPrediction.top_k[0].class_id).toBe(classId);
  expect(acceptedPrediction.top_k[0].score).toBeGreaterThanOrEqual(manifest.threshold);
  const cardRes = await careGuard;
  expect(cardRes).toBeDefined();
  expect(cardRes!.status).toBe(200);
  const card = (await cardRes!.json()) as CareResponse;
  expect(card.class_id).toBe(classId);
  await expect(result.getByRole('heading', card.name_vi, { exact: true })).toBeVisible();
  await expect(care).toBeVisible();
  await app.screenshot('cnn-gru-accepted-leaf');

  await upload.setInputFiles(nonLeaf);
  const rejected = browser.waitForResponse('**/api/v1/predict');
  await recognize.tap();
  const rejectedResponse = await rejected;
  expect(rejectedResponse.status).toBe(200);
  const rejectedPrediction = (await rejectedResponse.json()) as PredictResponse;
  expect(rejectedPrediction).toMatchObject({
    status: 'uncertain',
    model_version: manifest.model_version,
    class_id: null,
    care_card_id: null,
  });
  await expect(result.getByRole('heading', 'Chưa đủ cơ sở xác định', { exact: true })).toBeVisible();
  await expect(care).toHaveCount(0);
  await app.screenshot('cnn-gru-rejected-non-leaf');
});

test('CNN-GRU runtime preserves malformed-image rejection', async ({ app, screen, browser }) => {
  await app.open('/');
  await screen.getByLabel('Chọn ảnh lá').setInputFiles(`${fixtures}/fake.png`);
  const response = browser.waitForResponse('**/api/v1/predict');
  await screen.getByRole('button', 'Nhận diện', { exact: true }).tap();
  const res = await response;
  expect(res.status).toBe(422);
  const payload = (await res.json()) as ErrorResponse;
  expect(payload.error.code).toBe('invalid_input');
  await expect(screen.getByRole('alert')).toHaveText(
    'Không đọc được tệp ảnh. Hãy chọn một ảnh JPEG hoặc PNG khác.',
  );
  await expect(screen.getByRole('region', 'Thông tin chăm sóc')).toHaveCount(0);
});
