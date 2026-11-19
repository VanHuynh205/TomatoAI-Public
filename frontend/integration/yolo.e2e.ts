import { test } from '@e2e-dev/web';
import { expect } from 'e2e';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import type { CareResponse, PredictResponse } from '../src/api/client';

const fixtures = fileURLToPath(new URL('../../desktop/tests/fixtures/', import.meta.url));
const catalog = JSON.parse(readFileSync(new URL('../../infra/knowledge/care-v1.json', import.meta.url), 'utf8'));
const card = catalog.cards.find((item: { class_id: string }) => item.class_id === 'bacterial_spot');

test('synthetic YOLO bundle renders logits-derived result and care', async ({ app, screen, browser }) => {
  const health = await fetch(new URL('/api/v1/health', app.baseUrl), {
    headers: { 'X-Session-Token': process.env.TOMATO_E2E_SESSION_TOKEN! },
    signal: AbortSignal.timeout(3000),
  });
  expect(health.status).toBe(200);
  expect(await health.json()).toEqual({ schema_version: '1', ready: true, model_version: 'yolo-fixture-v1', knowledge_version: catalog.knowledge_version, reason: null });
  await app.open('/');
  const denominator = [3, 2, 1, 0, -1, -2, -3, -4].reduce((sum, bias) => sum + Math.exp(bias), 0);
  for (const name of ['healthy.png', 'healthy.jpg']) {
    await screen.getByLabel('Chọn ảnh lá', { exact: true }).setInputFiles(`${fixtures}/${name}`);
    const prediction = browser.waitForResponse('**/api/v1/predict');
    const care = browser.waitForResponse('**/api/v1/care/bacterial_spot');
    await screen.getByRole('button', 'Nhận diện', { exact: true }).tap();
    const response = await prediction;
    expect(response.status).toBe(200);
    const payload = await response.json() as PredictResponse;
    expect(payload.model_version).toBe('yolo-fixture-v1');
    expect(payload.status).toBe('ok');
    expect(payload.class_id).toBe('bacterial_spot');
    expect(payload.care_card_id).toBe('bacterial_spot');
    expect(payload.top_k.map((row: { class_id: string }) => row.class_id)).toEqual(['bacterial_spot', 'early_blight', 'late_blight']);
    payload.top_k.forEach((row: { score: number }, index: number) => {
      expect(Math.abs(row.score - Math.exp(3 - index) / denominator)).toBeLessThan(0.000001);
    });
    expect((await (await care).json() as CareResponse).class_id).toBe('bacterial_spot');
    await expect(screen.getByRole('region', 'Kết quả nhận diện').getByRole('heading', card.name_vi, { exact: true })).toBeVisible();
    await expect(screen.getByRole('region', 'Thông tin chăm sóc')).toBeVisible();
  }
  await app.screenshot('synthetic-yolo-tracer');
});

test('YOLO runtime keeps malformed-image rejection', async ({ app, screen, browser }) => {
  await app.open('/');
  await screen.getByLabel('Chọn ảnh lá').setInputFiles(`${fixtures}/fake.png`);
  const response = browser.waitForResponse('**/api/v1/predict');
  await screen.getByRole('button', 'Nhận diện', { exact: true }).tap();
  expect((await response).status).toBe(422);
  await expect(screen.getByRole('alert')).toHaveText('Không đọc được tệp ảnh. Hãy chọn một ảnh JPEG hoặc PNG khác.');
  await expect(screen.getByRole('region', 'Thông tin chăm sóc')).toHaveCount(0);
});
