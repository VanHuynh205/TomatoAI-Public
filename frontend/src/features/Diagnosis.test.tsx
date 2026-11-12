import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { ApiError, type CareResponse, type PredictResponse } from '../api/client';
import fixtures from '../../tests/fixtures/expected.json';
import { Diagnosis } from './Diagnosis';

const prediction = fixtures.prediction as PredictResponse;
const card = fixtures.care as CareResponse;
const uncertain = fixtures.uncertain as PredictResponse;
const newFile = (name = 'leaf.png') => new File(['synthetic'], name, { type: 'image/png' });
function api() {
  return {
    health: vi.fn(),
    predict: vi.fn<(file: File, signal?: AbortSignal) => Promise<PredictResponse>>().mockResolvedValue(prediction),
    care: vi.fn<(id: string, signal?: AbortSignal) => Promise<CareResponse>>().mockResolvedValue(card),
  };
}
function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => { resolve = done; });
  return { promise, resolve };
}
let revoke: ReturnType<typeof vi.fn>;
beforeEach(() => {
  let sequence = 0;
  revoke = vi.fn();
  Object.defineProperty(URL, 'createObjectURL', { configurable: true, value: vi.fn(() => `blob:test-${++sequence}`) });
  Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, value: revoke });
  window.pywebview = { api: { get_session_token: async () => 'test-session', open_external_url: vi.fn(async () => undefined) } };
});
afterEach(() => { delete window.pywebview; });

it('shows the exact nonzero file size and layout-independent instructions', async () => {
  render(<Diagnosis api={api()} />);
  expect(screen.getByText('Chọn ảnh rồi bấm Nhận diện. Kết quả và thông tin chăm sóc có nguồn sẽ xuất hiện tại đây.')).toBeVisible();
  await userEvent.upload(screen.getByLabelText('Chọn ảnh lá'), newFile());
  expect(screen.getByText('9 byte')).toBeVisible();
});

it('requires an image, announces loading, and maps the returned card name', async () => {
  const client = api();
  const pending = deferred<PredictResponse>();
  client.predict.mockReturnValue(pending.promise);
  render(<Diagnosis api={client} />);
  const button = screen.getByRole('button', { name: 'Nhận diện' });
  expect(button).toBeDisabled();
  await userEvent.upload(screen.getByLabelText('Chọn ảnh lá'), newFile());
  expect(screen.getByAltText('Ảnh lá đã chọn')).toHaveAttribute('src', 'blob:test-1');
  await userEvent.click(button);
  expect(button).toBeDisabled();
  expect(screen.getByRole('status')).toHaveTextContent('Đang nhận diện');
  await act(async () => { pending.resolve(prediction); });
  expect(await screen.findByRole('heading', { name: 'Lá khỏe theo nhãn dữ liệu' })).toBeVisible();
  expect(screen.getByRole('region', { name: 'Thông tin chăm sóc' })).toBeVisible();
  expect(screen.getByText('Điểm mô hình')).toBeVisible();
  expect(screen.getByText('Thông tin tham khảo đã đối chiếu tài liệu; chưa thẩm định chuyên môn tại địa phương')).toBeVisible();
  expect(client.care).toHaveBeenCalledWith('healthy', expect.any(AbortSignal));
});

it('shows uncertain status without requesting or rendering care', async () => {
  const client = api();
  client.predict.mockResolvedValue(uncertain);
  render(<Diagnosis api={client} />);
  await userEvent.upload(screen.getByLabelText('Chọn ảnh lá'), newFile());
  await userEvent.click(screen.getByRole('button', { name: 'Nhận diện' }));
  expect(await screen.findByRole('heading', { name: 'Chưa đủ cơ sở xác định' })).toBeVisible();
  expect(client.care).not.toHaveBeenCalled();
  expect(screen.queryByRole('region', { name: 'Thông tin chăm sóc' })).not.toBeInTheDocument();
});

it.each(['prediction', 'care'] as const)('ignores an old %s after selecting another image', async (stage) => {
  const client = api();
  const oldPrediction = deferred<PredictResponse>();
  const oldCare = deferred<CareResponse>();
  if (stage === 'prediction') client.predict.mockReturnValueOnce(oldPrediction.promise);
  else client.care.mockReturnValueOnce(oldCare.promise);
  render(<Diagnosis api={client} />);
  await userEvent.upload(screen.getByLabelText('Chọn ảnh lá'), newFile('old.png'));
  await userEvent.click(screen.getByRole('button', { name: 'Nhận diện' }));
  if (stage === 'care') await waitFor(() => expect(client.care).toHaveBeenCalledTimes(1));
  const signal = client.predict.mock.calls[0][1];
  await userEvent.upload(screen.getByLabelText('Chọn ảnh lá'), newFile('new.png'));
  expect(signal?.aborted).toBe(true);
  await act(async () => {
    oldPrediction.resolve({ ...prediction, class_id: 'early_blight', care_card_id: 'early_blight' });
    oldCare.resolve({ ...card, class_id: 'early_blight', name_vi: 'Đốm vòng' });
  });
  expect(screen.queryByText('Đốm vòng')).not.toBeInTheDocument();
  expect(screen.queryByRole('region', { name: 'Thông tin chăm sóc' })).not.toBeInTheDocument();
  expect(revoke).toHaveBeenCalledWith('blob:test-1');
  await userEvent.click(screen.getByRole('button', { name: 'Nhận diện' }));
  expect(await screen.findByRole('heading', { name: 'Lá khỏe theo nhãn dữ liệu' })).toBeVisible();
});

it('aborts pending work and releases the preview on unmount', async () => {
  const client = api();
  client.predict.mockReturnValue(new Promise(() => {}));
  const { unmount } = render(<Diagnosis api={client} />);
  await userEvent.upload(screen.getByLabelText('Chọn ảnh lá'), newFile());
  await userEvent.click(screen.getByRole('button', { name: 'Nhận diện' }));
  const signal = client.predict.mock.calls[0][1];
  unmount();
  expect(signal?.aborted).toBe(true);
  expect(revoke).toHaveBeenCalledWith('blob:test-1');
});

it('shows busy without automatic retries and allows an explicit retry', async () => {
  const client = api();
  client.predict.mockRejectedValueOnce(new ApiError(429, 'busy', 'busy'));
  render(<Diagnosis api={client} />);
  await userEvent.upload(screen.getByLabelText('Chọn ảnh lá'), newFile());
  await userEvent.click(screen.getByRole('button', { name: 'Nhận diện' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('đang nhận diện');
  expect(client.predict).toHaveBeenCalledTimes(1);
  await userEvent.click(screen.getByRole('button', { name: 'Nhận diện' }));
  expect(await screen.findByRole('heading', { name: 'Lá khỏe theo nhãn dữ liệu' })).toBeVisible();
});

it.each([[413, 'payload_too_large'], [503, 'model_unavailable'], [500, 'internal_error']] as const)('shows API %i as an error without a card', async (status, code) => {
  const client = api();
  client.predict.mockRejectedValue(new ApiError(status, code, 'server failure'));
  render(<Diagnosis api={client} />);
  await userEvent.upload(screen.getByLabelText('Chọn ảnh lá'), newFile());
  await userEvent.click(screen.getByRole('button', { name: 'Nhận diện' }));
  expect(await screen.findByRole('alert')).not.toBeEmptyDOMElement();
  expect(screen.queryByRole('region', { name: 'Thông tin chăm sóc' })).not.toBeInTheDocument();
});

it('rejects unsupported or oversized input before calling the API', async () => {
  const client = api();
  render(<Diagnosis api={client} />);
  const input = screen.getByLabelText('Chọn ảnh lá');
  fireEvent.change(input, { target: { files: [new File(['x'], 'file.txt', { type: 'text/plain' })] } });
  expect(screen.getByRole('alert')).toHaveTextContent('JPEG hoặc PNG');
  const large = new File([new Uint8Array(10_000_001)], 'large.png', { type: 'image/png' });
  fireEvent.change(input, { target: { files: [large] } });
  expect(screen.getByRole('alert')).toHaveTextContent('10 MB');
  expect(client.predict).not.toHaveBeenCalled();
});

it('shows source, geographic limits, review date, and opens the external bridge', async () => {
  render(<Diagnosis api={api()} />);
  await userEvent.upload(screen.getByLabelText('Chọn ảnh lá'), newFile());
  await userEvent.click(screen.getByRole('button', { name: 'Nhận diện' }));
  const source = await screen.findByRole('link', { name: /example.com/ });
  await userEvent.click(source);
  expect(window.pywebview?.api.open_external_url).toHaveBeenCalledWith('https://example.com/fixture');
  expect(screen.getAllByText(/Synthetic fixture only/).length).toBeGreaterThan(0);
  expect(screen.getByText(/02\/11\/2026/)).toBeVisible();
  expect(screen.getByText('Synthetic fixture; no clinical or agronomic validity')).toBeVisible();
});

it('reports care lookup failure without stale care content', async () => {
  const client = api();
  client.care.mockRejectedValue(new ApiError(404, 'care_not_found', 'missing'));
  render(<Diagnosis api={client} />);
  await userEvent.upload(screen.getByLabelText('Chọn ảnh lá'), newFile());
  await userEvent.click(screen.getByRole('button', { name: 'Nhận diện' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('thẻ chăm sóc');
  expect(screen.queryByRole('region', { name: 'Thông tin chăm sóc' })).not.toBeInTheDocument();
});
