import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError, createApiClient } from './client';

const fetchMock = vi.fn<typeof fetch>();
const healthy = { schema_version: '1', ready: true, model_version: 'fixture-v1', knowledge_version: 'fixture-care-v1', reason: null };
beforeEach(() => {
  fetchMock.mockReset();
  vi.stubGlobal('fetch', fetchMock);
});
afterEach(() => { vi.unstubAllGlobals(); vi.useRealTimers(); });

describe('local API client', () => {
  it('uses a relative URL and session header for health', async () => {
    fetchMock.mockResolvedValue(new Response(JSON.stringify(healthy)));
    expect(await createApiClient('test-session').health()).toEqual(healthy);
    expect(fetchMock.mock.calls[0][0]).toBe('/api/v1/health');
    expect(new Headers(fetchMock.mock.calls[0][1]?.headers).get('X-Session-Token')).toBe('test-session');
  });

  it('sends a single image FormData and signal without overriding its boundary', async () => {
    const prediction = { schema_version: '1', status: 'uncertain', class_id: null, top_k: [], model_version: 'fixture-v1', knowledge_version: 'fixture-care-v1', care_card_id: null, warnings: [] };
    fetchMock.mockResolvedValue(new Response(JSON.stringify(prediction)));
    const file = new File(['synthetic'], 'leaf.png', { type: 'image/png' });
    const controller = new AbortController();
    expect(await createApiClient('test-session').predict(file, controller.signal)).toEqual(prediction);
    const [url, options] = fetchMock.mock.calls[0];
    expect(url).toBe('/api/v1/predict');
    expect(options?.method).toBe('POST');
    expect(options?.signal?.aborted).toBe(false);
    expect(new Headers(options?.headers).has('Content-Type')).toBe(false);
    expect(options?.body).toBeInstanceOf(FormData);
    expect([...((options?.body as FormData).keys())]).toEqual(['image']);
    expect((options?.body as FormData).get('image')).toBe(file);
  });

  it('encodes a class ID and propagates care cancellation', async () => {
    fetchMock.mockResolvedValue(new Response('{}'));
    const controller = new AbortController();
    await createApiClient('test-session').care('bad/id?x=#', controller.signal);
    const [url, options] = fetchMock.mock.calls[0];
    expect(url).toBe('/api/v1/care/bad%2Fid%3Fx%3D%23');
    expect(options?.signal?.aborted).toBe(false);
  });

  it.each([[401, 'unauthorized'], [404, 'care_not_found'], [413, 'payload_too_large'], [429, 'busy'], [503, 'model_unavailable']] as const)('preserves HTTP %i and error code %s', async (status, code) => {
    fetchMock.mockResolvedValue(new Response(JSON.stringify({ schema_version: '1', error: { code, message: 'Yêu cầu không thể xử lý.' } }), { status }));
    const request = createApiClient('test-session').care('healthy');
    await expect(request).rejects.toBeInstanceOf(ApiError);
    await expect(request).rejects.toMatchObject({ status, code });
  });

  it('keeps network errors as failures', async () => {
    const failure = new TypeError('network unavailable');
    fetchMock.mockRejectedValue(failure);
    await expect(createApiClient('test-session').health()).rejects.toBe(failure);
  });

  it('keeps aborts distinct from a prediction result', async () => {
    const failure = new DOMException('Aborted', 'AbortError');
    fetchMock.mockRejectedValue(failure);
    await expect(createApiClient('test-session').predict(new File([], 'leaf.png'))).rejects.toBe(failure);
  });

  it('reports a non-JSON server error without exposing its response body', async () => {
    fetchMock.mockResolvedValue(new Response('private stack trace', { status: 500 }));
    await expect(createApiClient('test-session').health()).rejects.toMatchObject({ status: 500, code: 'internal_error' });
  });

  it('requires a nonempty token and keeps sessions separate', async () => {
    expect(() => createApiClient('')).toThrow();
    fetchMock.mockImplementation(async () => new Response(JSON.stringify(healthy)));
    await createApiClient('session-one').health();
    await createApiClient('session-two').health();
    expect(fetchMock.mock.calls.map(([, options]) => new Headers(options?.headers).get('X-Session-Token'))).toEqual(['session-one', 'session-two']);
  });

  it.each(['health', 'predict', 'care'] as const)('bounds a stalled %s request at 30 seconds', async (method) => {
    vi.useFakeTimers();
    fetchMock.mockImplementation((_url, options) => new Promise((_resolve, reject) => {
      options?.signal?.addEventListener('abort', () => reject(options.signal?.reason), { once: true });
    }));
    const api = createApiClient('test-session');
    const outcomes: unknown[] = [];
    const request = method === 'health' ? api.health() : method === 'care' ? api.care('healthy') : api.predict(new File(['x'], 'leaf.png'));
    void request.then((value) => outcomes.push(value), (error) => outcomes.push(error));
    await vi.advanceTimersByTimeAsync(29_999);
    expect(outcomes).toEqual([]);
    await vi.advanceTimersByTimeAsync(1);
    expect(outcomes).toHaveLength(1);
    expect(outcomes[0]).toMatchObject({ name: 'TimeoutError' });
  });

  it.each(['predict', 'care'] as const)('propagates caller cancellation during %s without waiting for the deadline', async (method) => {
    fetchMock.mockImplementation((_url, options) => new Promise((_resolve, reject) => {
      const signal = options?.signal;
      if (signal?.aborted) reject(signal.reason);
      else signal?.addEventListener('abort', () => reject(signal.reason), { once: true });
    }));
    const controller = new AbortController();
    const api = createApiClient('test-session');
    const request = method === 'care' ? api.care('healthy', controller.signal) : api.predict(new File(['x'], 'leaf.png'), controller.signal);
    const assertion = expect(request).rejects.toMatchObject({ name: 'AbortError' });
    controller.abort();
    await assertion;
  });

  it('keeps the deadline active until the JSON body finishes and clears it after success', async () => {
    vi.useFakeTimers();
    fetchMock.mockImplementation(async (_url, options) => new Response(new ReadableStream({
      start(controller) {
        options?.signal?.addEventListener('abort', () => controller.error(options.signal?.reason), { once: true });
      },
    })));
    const outcomes: unknown[] = [];
    void createApiClient('test-session').health().catch((error) => outcomes.push(error));
    await vi.advanceTimersByTimeAsync(30_000);
    expect(outcomes).toHaveLength(1);
    expect(outcomes[0]).toMatchObject({ name: 'TimeoutError' });
    expect(vi.getTimerCount()).toBe(0);
    fetchMock.mockResolvedValue(new Response(JSON.stringify(healthy)));
    expect(await createApiClient('test-session').health()).toEqual(healthy);
    expect(vi.getTimerCount()).toBe(0);
  });
});
