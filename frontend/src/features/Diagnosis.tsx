import { useEffect, useLayoutEffect, useRef, useState, type MouseEvent } from 'react';
import { gsap } from 'gsap';
import Lenis from 'lenis';
import { ApiError, type CareResponse, type PredictResponse, type createApiClient } from '../api/client';
import { CareCard } from './CareCard';

function errorMessage(error: unknown): string {
  if (!(error instanceof ApiError)) return 'Không thể kết nối dịch vụ nhận diện trên máy. Hãy khởi động lại ứng dụng.';
  switch (error.code) {
    case 'busy': return 'Ứng dụng đang nhận diện ảnh trước. Chờ một chút rồi bấm Nhận diện để thử lại.';
    case 'payload_too_large': return 'Ảnh vượt giới hạn 10 MB. Hãy chọn ảnh nhỏ hơn.';
    case 'image_too_large': return 'Ảnh vượt 20 triệu điểm ảnh. Hãy giảm kích thước ảnh rồi thử lại.';
    case 'unsupported_image': return 'Tệp không phải ảnh JPEG hoặc PNG hợp lệ. Hãy chọn ảnh khác.';
    case 'invalid_input': return 'Không đọc được tệp ảnh. Hãy chọn một ảnh JPEG hoặc PNG khác.';
    case 'care_not_found': return 'Không tìm thấy thẻ chăm sóc phù hợp. Hãy kiểm tra gói dữ liệu của ứng dụng.';
    case 'model_unavailable': return 'Mô hình chưa sẵn sàng. Hãy khởi động lại ứng dụng.';
    case 'unauthorized':
    case 'forbidden_host':
    case 'forbidden_origin': return 'Phiên làm việc không hợp lệ. Hãy mở lại ứng dụng desktop.';
    default: return 'Chưa thể hoàn tất nhận diện. Hãy thử lại hoặc khởi động lại ứng dụng.';
  }
}

export function Diagnosis({ api }: { api: ReturnType<typeof createApiClient> }) {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState('');
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [card, setCard] = useState<CareResponse | null>(null);
  const [phase, setPhase] = useState<'idle' | 'predict' | 'care'>('idle');
  const [error, setError] = useState('');
  const sequence = useRef(0);
  const request = useRef<AbortController | null>(null);
  const previewImage = useRef<HTMLImageElement>(null);
  const resultRegion = useRef<HTMLElement>(null);
  const cancelScroll = useRef<(() => void) | null>(null);
  const busy = phase !== 'idle';

  useEffect(() => {
    if (!file) { setPreview(''); return; }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);
  useEffect(() => () => { sequence.current++; request.current?.abort(); }, []);
  useEffect(() => () => cancelScroll.current?.(), [card]);

  useLayoutEffect(() => {
    const image = previewImage.current;
    if (!preview || !image || !window.matchMedia) return;
    const motion = gsap.matchMedia();
    motion.add('(prefers-reduced-motion: no-preference)', () => {
      gsap.from(image, { opacity: 0.55, scale: 0.98, duration: 0.24, ease: 'power3.out', clearProps: 'opacity,transform' });
    });
    return () => motion.revert();
  }, [preview]);

  useLayoutEffect(() => {
    const region = resultRegion.current;
    if (!result || !region || !window.matchMedia) return;
    const motion = gsap.matchMedia(region);
    motion.add('(prefers-reduced-motion: no-preference)', () => {
      gsap.from('[data-result-reveal]', {
        opacity: 0.6, y: 6, duration: 0.28, stagger: 0.04, ease: 'power3.out', clearProps: 'opacity,transform',
      });
    });
    return () => motion.revert();
  }, [result]);

  function followReference(event: MouseEvent<HTMLDivElement>) {
    if (event.defaultPrevented || event.button !== 0 || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
    const anchor = event.target instanceof Element ? event.target.closest<HTMLAnchorElement>('a[href^="#"]') : null;
    const target = anchor && document.getElementById(anchor.hash.slice(1));
    if (!anchor || !target || !event.currentTarget.contains(target)) return;

    event.preventDefault();
    cancelScroll.current?.();
    if (window.location.hash !== anchor.hash) window.history.pushState(null, '', anchor.hash);
    target.focus({ preventScroll: true });
    if (!window.matchMedia?.('(prefers-reduced-motion: no-preference)').matches) {
      target.scrollIntoView({ behavior: 'instant', block: 'start' });
      return;
    }

    const scroll = new Lenis({ smoothWheel: false, autoResize: false });
    const frame = (seconds: number) => scroll.raf(seconds * 1000);
    const finish = () => {
      gsap.ticker.remove(frame);
      scroll.destroy();
      cancelScroll.current = null;
    };
    cancelScroll.current = finish;
    scroll.on('virtual-scroll', finish);
    gsap.ticker.add(frame);
    scroll.scrollTo(target, { duration: 0.32, easing: t => 1 - (1 - t) ** 3, onComplete: finish });
  }

  function select(next: File | undefined) {
    sequence.current++;
    request.current?.abort();
    setResult(null); setCard(null); setError(''); setPhase('idle'); setFile(null);
    if (!next) return;
    if (!['image/jpeg', 'image/png'].includes(next.type)) {
      setError('Chỉ hỗ trợ ảnh JPEG hoặc PNG. Hãy chọn ảnh khác.'); return;
    }
    if (next.size > 10_000_000) { setError('Ảnh vượt giới hạn 10 MB. Hãy chọn ảnh nhỏ hơn.'); return; }
    if (next.size === 0) { setError('Tệp ảnh rỗng. Hãy chọn ảnh khác.'); return; }
    setFile(next);
  }

  async function recognize() {
    if (!file || busy) return;
    const current = ++sequence.current;
    const controller = new AbortController();
    request.current?.abort(); request.current = controller;
    setResult(null); setCard(null); setError(''); setPhase('predict');
    try {
      const prediction = await api.predict(file, controller.signal);
      if (current !== sequence.current) return;
      setResult(prediction);
      if (prediction.status === 'ok' && prediction.care_card_id) {
        setPhase('care');
        const care = await api.care(prediction.care_card_id, controller.signal);
        if (current === sequence.current) setCard(care);
      }
    } catch (failure) {
      if (current === sequence.current && !controller.signal.aborted) setError(errorMessage(failure));
    } finally {
      if (current === sequence.current) setPhase('idle');
    }
  }

  const score = result?.top_k.find((item) => item.class_id === result.class_id)?.score;
  const status = phase === 'predict' ? 'Đang nhận diện ảnh…' : phase === 'care' ? 'Đang tải thông tin chăm sóc…' : result ? 'Đã có kết quả nhận diện.' : file ? 'Ảnh đã sẵn sàng để nhận diện.' : 'Chưa chọn ảnh.';
  return (
    <div className="diagnosis-layout">
      <section className="image-panel" aria-labelledby="image-heading">
        <h2 id="image-heading">Ảnh cần nhận diện</h2>
        <p className="secondary">Chọn ảnh rõ nét của một lá, thấy đủ phiến lá.</p>
        {preview && <div className="preview"><img ref={previewImage} src={preview} alt="Ảnh lá đã chọn" /></div>}
        <label className={`file-label${preview ? ' has-preview' : ''}`} htmlFor="leaf-image">
          <span className="upload-icon" aria-hidden="true" />
          <span>Chọn ảnh lá</span>
          <input id="leaf-image" type="file" accept="image/jpeg,image/png" aria-describedby="image-limits" onChange={(event) => select(event.target.files?.[0])} />
        </label>
        <p id="image-limits" className="secondary">JPEG hoặc PNG · Tối đa 10 MB, 20 triệu điểm ảnh</p>
        {file && <p className="filename"><strong>{file.name}</strong><span>{new Intl.NumberFormat('vi-VN').format(file.size)} byte</span></p>}
        <button className="primary" disabled={!file || busy} onClick={() => void recognize()}>Nhận diện<span className="action-arrow" aria-hidden="true" /></button>
        <p role="status" aria-live="polite" className="status" data-phase={phase}>
          <span className="status-mark" data-ready={Boolean(file)} aria-hidden="true" />{status}
        </p>
        <p className="privacy">Ảnh được xử lý trên máy, không lưu sau phiên.</p>
      </section>

      <div className="reading-panel" onClick={followReference}>
        {error && <p role="alert" className="error">{error}</p>}
        <section ref={resultRegion} className={`result ${result?.status === 'uncertain' ? 'uncertain' : ''}`} aria-labelledby="result-heading" aria-busy={busy}>
          <h2 id="result-heading">Kết quả nhận diện</h2>
          {!result && !error && <div className="result-placeholder">
            <div className="result-page" aria-hidden="true"><span /><span /><span /></div>
            <h3>{busy ? 'Đang xem ảnh của bạn' : 'Nhận diện để bắt đầu đối chiếu'}</h3>
            {!busy && <p>{file ? 'Bấm Nhận diện để đối chiếu ảnh đã chọn.' : 'Chọn ảnh rồi bấm Nhận diện. Kết quả và thông tin chăm sóc có nguồn sẽ xuất hiện tại đây.'}</p>}
          </div>}
          {result?.status === 'uncertain' && <>
            <h3 className="result-title" data-result-reveal>Chưa đủ cơ sở xác định</h3>
            <p data-result-reveal>Hãy thử ảnh rõ hơn, đủ ánh sáng và chỉ có một lá. Kết quả này chưa chọn được thẻ chăm sóc.</p>
          </>}
          {result?.status === 'ok' && <>
            <h3 className="result-title" data-result-reveal>{card?.name_vi ?? (phase === 'care' ? 'Đang tải tên lớp nhận diện…' : 'Đã nhận diện ảnh')}</h3>
            {score !== undefined && <p className="score" data-result-reveal><span>Điểm mô hình</span><strong>{new Intl.NumberFormat('vi-VN', { style: 'percent', maximumFractionDigits: 1 }).format(score)}</strong></p>}
            <p className="secondary">Điểm dùng để xếp hạng nhãn, không phải xác nhận bệnh học.</p>
          </>}
          {result && <>
            {result.warnings.map((warning, index) => <p key={index} className="warning">{warning}</p>)}
            <details className="prediction-details"><summary>Xem điểm các nhãn</summary><dl>{result.top_k.map((item) => <div key={item.class_id}><dt>{item.class_id}</dt><dd>{new Intl.NumberFormat('vi-VN', { maximumFractionDigits: 4 }).format(item.score)}</dd></div>)}</dl><p className="secondary">Mô hình: {result.model_version}<br />Tri thức: {result.knowledge_version}</p></details>
          </>}
        </section>
        {card && result?.status === 'ok' && <CareCard card={card} />}
      </div>
    </div>
  );
}
