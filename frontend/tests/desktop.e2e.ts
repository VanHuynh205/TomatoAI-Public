import { beforeEach, test } from '@e2e-dev/web';
import { expect } from 'e2e';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const fixtures = fileURLToPath(new URL('../../desktop/tests/fixtures/', import.meta.url));
const expected = JSON.parse(readFileSync(new URL('./fixtures/expected.json', import.meta.url), 'utf8'));

beforeEach(async ({ app }) => {
  // The runner accepts HTTP 401 as reachable; only authenticated health proves readiness.
  const response = await fetch(new URL('/api/v1/health', app.baseUrl), {
    headers: { 'X-Session-Token': process.env.TOMATO_E2E_SESSION_TOKEN! },
    signal: AbortSignal.timeout(3000),
  });
  expect(response.status).toBe(200);
  expect(await response.json()).toEqual({ schema_version: '1', ready: true, model_version: 'fixture-model-v1', knowledge_version: 'fixture-knowledge-v1', reason: null });
});

test('PNG and JPEG produce the exact fixture result and care', async ({ app, screen, browser }) => {
  await app.open('/');
  for (const name of ['healthy.png', 'healthy.jpg']) {
    await screen.getByLabel('Chọn ảnh lá', { exact: true }).setInputFiles(`${fixtures}/${name}`);
    const prediction = browser.waitForResponse('**/api/v1/predict');
    const care = browser.waitForResponse('**/api/v1/care/healthy');
    await screen.getByRole('button', 'Nhận diện', { exact: true }).tap();
    expect(await (await prediction).json()).toEqual(expected.prediction);
    expect(await (await care).json()).toEqual(expected.care);
    await expect(screen.getByRole('region', 'Kết quả nhận diện').getByRole('heading', 'Lá khỏe theo nhãn dữ liệu', { exact: true })).toBeVisible();
    await expect(screen.getByRole('region', 'Thông tin chăm sóc')).toContainText('Thông tin tham khảo đã đối chiếu tài liệu');
  }
  await app.screenshot('desktop-healthy');
});

test('uncertain response has no selected care', async ({ app, screen, browser }) => {
  await app.open('/');
  await screen.getByLabel('Chọn ảnh lá').setInputFiles(`${fixtures}/uncertain.png`);
  const response = browser.waitForResponse('**/api/v1/predict');
  await screen.getByRole('button', 'Nhận diện', { exact: true }).tap();
  expect(await (await response).json()).toEqual(expected.uncertain);
  await expect(screen.getByText('Chưa đủ cơ sở xác định', { exact: true })).toBeVisible();
  await expect(screen.getByRole('region', 'Thông tin chăm sóc')).toHaveCount(0);
});

test('fake PNG is rejected by the real API', async ({ app, screen, browser }) => {
  await app.open('/');
  await screen.getByLabel('Chọn ảnh lá').setInputFiles(`${fixtures}/fake.png`);
  const response = browser.waitForResponse('**/api/v1/predict');
  await screen.getByRole('button', 'Nhận diện', { exact: true }).tap();
  expect((await response).status).toBe(422);
  await expect(screen.getByRole('alert')).toHaveText('Không đọc được tệp ảnh. Hãy chọn một ảnh JPEG hoặc PNG khác.');
  await expect(screen.getByRole('region', 'Thông tin chăm sóc')).toHaveCount(0);
});

test('API failure is actionable and leaves retry available', async ({ app, screen, browser }) => {
  await browser.route('**/api/v1/predict', route => route.fulfill({ status: 503, json: { error: { code: 'model_unavailable', message: 'Fixture model failure' } } }));
  await app.open('/');
  await screen.getByLabel('Chọn ảnh lá').setInputFiles(`${fixtures}/healthy.png`);
  await screen.getByRole('button', 'Nhận diện', { exact: true }).tap();
  await expect(screen.getByRole('alert')).toHaveText('Mô hình chưa sẵn sàng. Hãy khởi động lại ứng dụng.');
  await expect(screen.getByRole('button', 'Nhận diện', { exact: true })).toBeEnabled();
  await app.screenshot('api-failure');
});

test('a stalled prediction times out and lets the user retry', { timeout: 45_000 }, async ({ app, screen, browser }) => {
  let release!: () => void;
  const pending = new Promise<void>(resolve => { release = resolve; });
  let first = true;
  await browser.route('**/api/v1/predict', async route => {
    if (!first) { await route.continue(); return; }
    first = false;
    await pending;
    await route.fulfill({ status: 503, json: { error: { code: 'model_unavailable', message: 'Delayed fixture' } } });
  });
  try {
    await app.open('/');
    await screen.getByLabel('Chọn ảnh lá').setInputFiles(`${fixtures}/healthy.png`);
    await screen.getByRole('button', 'Nhận diện', { exact: true }).tap();
    await expect(screen.getByRole('status')).toHaveText('Đang nhận diện ảnh…');
    await expect(screen.getByRole('alert')).toHaveText('Không thể kết nối dịch vụ nhận diện trên máy. Hãy khởi động lại ứng dụng.', { timeout: 35_000 });
    await expect(screen.getByRole('button', 'Nhận diện', { exact: true })).toBeEnabled();
    await expect(screen.getByRole('region', 'Thông tin chăm sóc')).toHaveCount(0);
    await app.screenshot('request-timeout');
    release();
    await screen.getByRole('button', 'Nhận diện', { exact: true }).tap();
    await expect(screen.getByRole('region', 'Thông tin chăm sóc')).toBeVisible();
    await expect(screen.getByRole('heading', 'Lá khỏe theo nhãn dữ liệu', { exact: true })).toBeVisible();
  } finally { release(); }
});

test('selecting a new image discards the pending old response', async ({ app, screen, browser }) => {
  let release!: () => void;
  let received!: () => void;
  const pending = new Promise<void>(resolve => { release = resolve; });
  const intercepted = new Promise<void>(resolve => { received = resolve; });
  let first = true;
  await browser.route('**/api/v1/predict', async route => {
    if (!first) { await route.continue(); return; }
    first = false;
    const response = await route.fetch();
    const body = await response.text();
    received();
    await pending;
    await route.fulfill({ status: response.status, contentType: 'application/json', body });
  });
  await app.open('/');
  await screen.getByLabel('Chọn ảnh lá').setInputFiles(`${fixtures}/healthy.png`);
  await screen.getByRole('button', 'Nhận diện', { exact: true }).tap();
  await intercepted;
  await expect(screen.getByRole('status')).toHaveText('Đang nhận diện ảnh…');
  await app.screenshot('pending-prediction');
  await screen.getByLabel('Chọn ảnh lá').setInputFiles(`${fixtures}/uncertain.png`);
  release();
  await screen.getByRole('button', 'Nhận diện', { exact: true }).tap();
  await expect(screen.getByText('Chưa đủ cơ sở xác định', { exact: true })).toBeVisible();
  await expect(screen.getByRole('region', 'Thông tin chăm sóc')).toHaveCount(0);
  await expect(screen.getByRole('heading', 'Lá khỏe theo nhãn dữ liệu', { exact: true })).toHaveCount(0);
});

test('source link uses the bridge and keeps the privileged page', async ({ app, screen, browser }) => {
  await app.open('/');
  await screen.getByLabel('Chọn ảnh lá').setInputFiles(`${fixtures}/healthy.png`);
  await screen.getByRole('button', 'Nhận diện', { exact: true }).tap();
  await screen.getByRole('link', 'example.com — Fixture section 1', { exact: true }).tap();
  expect(await browser.evaluate('window.__sourceUrls')).toEqual(['https://example.com/fixture']);
  await expect(browser).toHaveURL('/');
  expect(await browser.evaluate('localStorage.length + sessionStorage.length')).toBe(0);
});

test('small viewport supports keyboard recognition without horizontal overflow', async ({ app, screen, browser }) => {
  await browser.setViewport({ width: 390, height: 844 });
  await app.open('/');
  await screen.getByLabel('Chọn ảnh lá').setInputFiles(`${fixtures}/healthy.png`);
  await screen.getByLabel('Chọn ảnh lá').focus();
  await screen.getByLabel('Chọn ảnh lá').press('Tab');
  await expect(screen.getByRole('button', 'Nhận diện', { exact: true })).toBeFocused();
  await screen.getByRole('button', 'Nhận diện', { exact: true }).press('Enter');
  await expect(screen.getByRole('region', 'Thông tin chăm sóc')).toBeVisible();
  expect(await browser.evaluate('document.documentElement.scrollWidth <= innerWidth')).toBe(true);
  await app.screenshot('mobile-healthy');
  await screen.getByRole('region', 'Kết quả nhận diện').scrollIntoView();
  await app.screenshot('mobile-result');
});

test('640px layout, CSS 200 percent zoom and reduced motion keep controls usable', async ({ app, screen, browser }) => {
  await browser.setViewport({ width: 640, height: 900 });
  await browser.emulateMedia({ reducedMotion: 'reduce' });
  await app.open('/');
  await expect(screen.getByLabel('Chọn ảnh lá')).toBeVisible();
  expect(await browser.evaluate("getComputedStyle(document.querySelector('.diagnosis-layout')).gridTemplateColumns.split(' ').length")).toBe(1);
  expect(await browser.evaluate('document.documentElement.scrollWidth <= innerWidth')).toBe(true);
  await app.screenshot('viewport-640');
  await browser.setViewport({ width: 1280, height: 1800 });
  // CSS zoom is a browser layout check, not evidence of native WebView2 zoom/offline.
  await browser.evaluate("document.documentElement.style.zoom = '2'");
  await screen.getByLabel('Chọn ảnh lá').setInputFiles(`${fixtures}/healthy.png`);
  await screen.getByRole('button', 'Nhận diện', { exact: true }).tap();
  await expect(screen.getByRole('heading', 'Lá khỏe theo nhãn dữ liệu', { exact: true })).toBeVisible();
  expect(await browser.evaluate('document.documentElement.scrollWidth <= document.documentElement.clientWidth')).toBe(true);
  expect(await browser.evaluate("getComputedStyle(document.querySelector('.primary')).transitionDuration")).toBe('0s');
  await app.screenshot('css-zoom-200');
});
