import { test } from '@e2e-dev/web';
import { expect } from 'e2e';
import type { CareResponse, PredictResponse } from '../src/api/client';

test('Desktop displays care for an accepted leaf and clears it after rejecting a non-leaf', async ({ app, browser, screen }) => {
  const leaf = process.env.TOMATO_E2E_LEAF_IMAGE;
  const nonLeaf = process.env.TOMATO_E2E_NON_LEAF_IMAGE;
  const classId = process.env.TOMATO_E2E_LEAF_CLASS_ID;
  const modelVersion = process.env.TOMATO_E2E_MODEL_VERSION
    ?? (process.env.TOMATO_E2E_MODEL_DIR ? 'dinov3-field-v1' : 'cnn-bigru-20261118-g2-convnext_tiny_axial_bigru');
  if (!leaf || !nonLeaf || !classId) {
    throw new Error('Set TOMATO_E2E_LEAF_IMAGE, TOMATO_E2E_NON_LEAF_IMAGE and TOMATO_E2E_LEAF_CLASS_ID from the verified benchmark.');
  }

  const origin = new URL('/', app.baseUrl).origin;
  const externalOrigins: string[] = [];
  await browser.route(/^https?:\/\//, async route => {
    const destination = new URL(route.request.url).origin;
    if (destination === origin) {
      await route.continue();
      return;
    }
    externalOrigins.push(destination);
    await route.abort();
  });
  await app.open('/');
  await expect(screen.getByText(`Mô hình: ${modelVersion}`, { exact: true })).toBeVisible();

  const upload = screen.getByLabel('Chọn ảnh lá', { exact: true });
  const recognize = screen.getByRole('button', 'Nhận diện', { exact: true });
  const result = screen.getByRole('region', 'Kết quả nhận diện', { exact: true });
  const care = screen.getByRole('region', 'Thông tin chăm sóc', { exact: true });

  await upload.setInputFiles(leaf);
  const accepted = browser.waitForResponse('**/api/v1/predict');
  const careResponse = browser.waitForResponse(`**/api/v1/care/${classId}`);
  await recognize.tap();
  const acceptedResponse = await accepted;
  expect(acceptedResponse.status).toBe(200);
  const acceptedPrediction = (await acceptedResponse.json()) as PredictResponse;
  expect(acceptedPrediction).toMatchObject({
    status: 'ok', model_version: modelVersion, class_id: classId, care_card_id: classId,
  });
  expect(acceptedPrediction.top_k[0].class_id).toBe(classId);
  expect(acceptedPrediction.top_k[0].score).toBeGreaterThanOrEqual(0.5);
  const cardResponse = await careResponse;
  expect(cardResponse.status).toBe(200);
  const card = (await cardResponse.json()) as CareResponse;
  expect(card.class_id).toBe(classId);
  await expect(result.getByRole('heading', card.name_vi, { exact: true })).toBeVisible();
  await expect(care).toBeVisible();
  await app.screenshot('accepted-leaf');

  await upload.setInputFiles(nonLeaf);
  const rejected = browser.waitForResponse('**/api/v1/predict');
  await recognize.tap();
  const rejectedResponse = await rejected;
  expect(rejectedResponse.status).toBe(200);
  const rejectedPrediction = (await rejectedResponse.json()) as PredictResponse;
  expect(rejectedPrediction).toMatchObject({
    status: 'uncertain', model_version: modelVersion, class_id: null, care_card_id: null,
  });
  await expect(result.getByRole('heading', 'Chưa đủ cơ sở xác định', { exact: true })).toBeVisible();
  await expect(care).toHaveCount(0);
  expect(externalOrigins).toEqual([]);
  await app.screenshot('rejected-non-leaf');
});
