import { test } from '@e2e-dev/web';
import { expect } from 'e2e';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import type { CareResponse, ErrorResponse, PredictResponse } from '../src/api/client';

const repoRoot = fileURLToPath(new URL('../../', import.meta.url));
const fixtures = fileURLToPath(new URL('../../desktop/tests/fixtures/', import.meta.url));
const catalog = JSON.parse(readFileSync(new URL('../../infra/knowledge/care-v1.json', import.meta.url), 'utf8'));
const healthyCard = catalog.cards.find((item: { class_id: string }) => item.class_id === 'healthy');

test('trained yolo26s-cls-v1 bundle serves accepted and uncertain PlantDoc predictions in app', async ({ app, screen, browser }) => {
  const health = await fetch(new URL('/api/v1/health', app.baseUrl), {
    headers: { 'X-Session-Token': process.env.TOMATO_E2E_SESSION_TOKEN! },
    signal: AbortSignal.timeout(3000),
  });
  expect(health.status).toBe(200);
  expect(await health.json()).toEqual({
    schema_version: '1',
    ready: true,
    model_version: 'yolo26s-cls-v1',
    knowledge_version: catalog.knowledge_version,
    reason: null,
  });

  await app.open('/');

  // 1. Accepted PlantDoc validation sample (plantdoc:PD0160 -> healthy)
  await screen.getByLabel('Chọn ảnh lá', { exact: true }).setInputFiles(
    `${repoRoot}infra/plant_doc/train/Tomato leaf/2015-09-17-01.jpg`,
  );
  const okPredict = browser.waitForResponse('**/api/v1/predict');
  const okCare = browser.waitForResponse('**/api/v1/care/healthy');
  await screen.getByRole('button', 'Nhận diện', { exact: true }).tap();
  const okRes = await okPredict;
  expect(okRes.status).toBe(200);
  const okPayload = (await okRes.json()) as PredictResponse;
  expect(okPayload.model_version).toBe('yolo26s-cls-v1');
  expect(okPayload.status).toBe('ok');
  expect(okPayload.class_id).toBe('healthy');
  expect(okPayload.care_card_id).toBe('healthy');
  expect(okPayload.top_k.map((row: { class_id: string }) => row.class_id)).toEqual([
    'healthy',
    'bacterial_spot',
    'leaf_mold',
  ]);
  expect(Math.abs(okPayload.top_k[0].score - 0.8202740550041199)).toBeLessThan(0.00001);
  expect(((await (await okCare).json()) as CareResponse).class_id).toBe('healthy');
  await expect(
    screen.getByRole('region', 'Kết quả nhận diện').getByRole('heading', healthyCard.name_vi, { exact: true }),
  ).toBeVisible();
  await expect(screen.getByRole('region', 'Thông tin chăm sóc')).toBeVisible();

  // 2. Uncertain PlantDoc validation sample (plantdoc:PD0072 -> score 0.4936 < 0.5 threshold)
  await screen.getByLabel('Chọn ảnh lá', { exact: true }).setInputFiles(
    `${repoRoot}infra/plant_doc/train/Tomato Early blight leaf/126689-004-D622CD2F.jpg`,
  );
  const uncPredict = browser.waitForResponse('**/api/v1/predict');
  await screen.getByRole('button', 'Nhận diện', { exact: true }).tap();
  const uncRes = await uncPredict;
  expect(uncRes.status).toBe(200);
  const uncPayload = (await uncRes.json()) as PredictResponse;
  expect(uncPayload.model_version).toBe('yolo26s-cls-v1');
  expect(uncPayload.status).toBe('uncertain');
  expect(uncPayload.class_id).toBeNull();
  expect(uncPayload.care_card_id).toBeNull();
  expect(uncPayload.top_k.map((row: { class_id: string }) => row.class_id)).toEqual([
    'bacterial_spot',
    'early_blight',
    'late_blight',
  ]);
  expect(Math.abs(uncPayload.top_k[0].score - 0.49364784359931946)).toBeLessThan(0.00001);
  await expect(screen.getByRole('region', 'Thông tin chăm sóc')).toHaveCount(0);
  await app.screenshot('real-yolo26s-cls-v1');
});

test('trained yolo26s-cls-v1 runtime keeps malformed-image rejection', async ({ app, screen, browser }) => {
  await app.open('/');
  await screen.getByLabel('Chọn ảnh lá').setInputFiles(`${fixtures}/fake.png`);
  const response = browser.waitForResponse('**/api/v1/predict');
  await screen.getByRole('button', 'Nhận diện', { exact: true }).tap();
  expect((await response).status).toBe(422);
  await expect(screen.getByRole('alert')).toHaveText(
    'Không đọc được tệp ảnh. Hãy chọn một ảnh JPEG hoặc PNG khác.',
  );
  await expect(screen.getByRole('region', 'Thông tin chăm sóc')).toHaveCount(0);
});

test('animated PNG is rejected before diagnosis', async ({ app, screen, browser }) => {
  await app.open('/');
  await screen.getByLabel('Chọn ảnh lá').setInputFiles(`${fixtures}/animated.png`);
  const response = browser.waitForResponse('**/api/v1/predict');
  await screen.getByRole('button', 'Nhận diện', { exact: true }).tap();
  const result = await response;
  expect(result.status).toBe(415);
  const payload = (await result.json()) as ErrorResponse;
  expect(payload.error.code).toBe('unsupported_image');
  await expect(screen.getByRole('alert')).toHaveText(
    'Tệp không phải ảnh JPEG hoặc PNG hợp lệ. Hãy chọn ảnh khác.',
  );
  await expect(screen.getByRole('region', 'Thông tin chăm sóc')).toHaveCount(0);
});
