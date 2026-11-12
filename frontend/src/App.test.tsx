import { act, render, screen } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import { App } from './App';

afterEach(() => { delete window.pywebview; vi.unstubAllGlobals(); });
it('waits for the desktop bridge and uses its RAM token for readiness', async () => {
  const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(new Response(JSON.stringify({ schema_version: '1', ready: true, model_version: 'fixture-model', knowledge_version: 'fixture-care', reason: null })));
  vi.stubGlobal('fetch', fetchMock);
  render(<App />);
  expect(screen.getByRole('status')).toHaveTextContent('Đang kết nối');
  expect(fetchMock).not.toHaveBeenCalled();
  window.pywebview = { api: { get_session_token: vi.fn(async () => 'bridge-session'), open_external_url: async () => undefined } };
  await act(async () => { window.dispatchEvent(new Event('pywebviewready')); });
  expect(await screen.findByLabelText('Chọn ảnh lá')).toBeVisible();
  expect(fetchMock.mock.calls[0][0]).toBe('/api/v1/health');
  expect(new Headers(fetchMock.mock.calls[0][1]?.headers).get('X-Session-Token')).toBe('bridge-session');
  expect(document.body).not.toHaveTextContent('bridge-session');
});

it('keeps inference unavailable when the model fails readiness', async () => {
  window.pywebview = { api: { get_session_token: async () => 'bridge-session', open_external_url: async () => undefined } };
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('{}', { status: 503 })));
  render(<App />);
  expect(await screen.findByRole('alert')).toHaveTextContent('khởi động lại');
  expect(screen.queryByLabelText('Chọn ảnh lá')).not.toBeInTheDocument();
});
