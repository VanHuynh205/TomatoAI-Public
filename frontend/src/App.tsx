import { useEffect, useState } from 'react';
import { createApiClient, type HealthResponse } from './api/client';
import { Diagnosis } from './features/Diagnosis';

export function App() {
  const [session, setSession] = useState<{ api: ReturnType<typeof createApiClient>; health: HealthResponse } | null>(null);
  const [error, setError] = useState('');
  useEffect(() => {
    let active = true;
    let started = false;
    const timeout = window.setTimeout(() => {
      if (active) setError('Chưa kết nối được ứng dụng desktop. Hãy đóng cửa sổ và khởi động lại ứng dụng.');
    }, 30_000);
    async function connect() {
      if (started || !window.pywebview?.api) return;
      started = true;
      try {
        const token = await window.pywebview.api.get_session_token();
        const api = createApiClient(token);
        const health = await api.health();
        if (!health.ready) throw new Error('Not ready');
        if (active) { setSession({ api, health }); setError(''); }
      } catch {
        if (active) setError('Dịch vụ nhận diện chưa sẵn sàng. Hãy khởi động lại ứng dụng và kiểm tra gói mô hình, tri thức.');
      } finally { window.clearTimeout(timeout); }
    }
    const ready = () => { void connect(); };
    window.addEventListener('pywebviewready', ready);
    ready();
    return () => { active = false; window.clearTimeout(timeout); window.removeEventListener('pywebviewready', ready); };
  }, []);
  return (
    <div className="app-shell">
      <a className="skip-link" href="#main">Đến nội dung nhận diện</a>
      <header className="app-header"><div><h1>Nhận diện lá cà chua</h1><p>Đối chiếu ảnh lá và đọc hướng dẫn có nguồn.</p></div><span className="local-label">Xử lý trên máy</span></header>
      <main id="main">
        {error ? <p role="alert" className="error">{error}</p> : session ? <Diagnosis api={session.api} /> : <p role="status">Đang kết nối ứng dụng desktop…</p>}
      </main>
      <footer>{session ? <><span>Mô hình: {session.health.model_version}</span><span>Tri thức: {session.health.knowledge_version}</span></> : <span>Ứng dụng desktop · Tiếng Việt</span>}</footer>
    </div>
  );
}
