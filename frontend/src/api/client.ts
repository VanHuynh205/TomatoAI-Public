import type { components } from './schema';

export type HealthResponse = components['schemas']['HealthResponse'];
export type PredictResponse = components['schemas']['PredictResponse'];
export type CareResponse = components['schemas']['CareResponse'];
export type ErrorResponse = components['schemas']['ErrorResponse'];
export type ErrorCode = ErrorResponse['error']['code'];

export class ApiError extends Error {
  constructor(public readonly status: number, public readonly code: ErrorCode, message: string) {
    super(message);
    this.name = 'ApiError';
  }
}

export function createApiClient(sessionToken: string) {
  if (!sessionToken.trim()) throw new Error('Thiếu token phiên làm việc.');

  async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
    const controller = new AbortController();
    const cancel = () => controller.abort(options.signal?.reason);
    if (options.signal?.aborted) cancel();
    else options.signal?.addEventListener('abort', cancel, { once: true });
    const timeout = setTimeout(() => controller.abort(new DOMException('Yêu cầu quá thời gian chờ.', 'TimeoutError')), 30_000);
    try {
      const response = await fetch(path, {
        ...options,
        signal: controller.signal,
        headers: { 'X-Session-Token': sessionToken },
        cache: 'no-store',
        credentials: 'omit',
      });
      if (!response.ok) {
        const body: ErrorResponse | null = await response.json().catch(() => null);
        throw new ApiError(response.status, body?.error?.code ?? 'internal_error', body?.error?.message ?? 'Không thể xử lý yêu cầu.');
      }
      return await response.json() as T;
    } finally {
      clearTimeout(timeout);
      options.signal?.removeEventListener('abort', cancel);
    }
  }

  return {
    health: () => request<HealthResponse>('/api/v1/health'),
    predict(image: File, signal?: AbortSignal) {
      const body = new FormData();
      body.append('image', image);
      return request<PredictResponse>('/api/v1/predict', { method: 'POST', body, signal });
    },
    care: (classId: string, signal?: AbortSignal) => request<CareResponse>(`/api/v1/care/${encodeURIComponent(classId)}`, { signal }),
  };
}
