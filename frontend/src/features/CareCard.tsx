import { useState } from 'react';
import type { CareResponse } from '../api/client';

const date = (iso: string) => new Intl.DateTimeFormat('vi-VN', { day: '2-digit', month: '2-digit', year: 'numeric' }).format(new Date(`${iso}T00:00:00`));

export function CareCard({ card }: { card: CareResponse }) {
  const [linkError, setLinkError] = useState('');
  async function openSource(url: string) {
    try {
      if (!window.pywebview?.api) throw new Error('Desktop bridge unavailable');
      await window.pywebview.api.open_external_url(url);
      setLinkError('');
    } catch {
      setLinkError('Không thể mở nguồn. Hãy mở ứng dụng desktop và kiểm tra trình duyệt mặc định.');
    }
  }
  return (
    <section className="care" aria-labelledby="care-heading">
      <h2 id="care-heading">Thông tin chăm sóc</h2>
      <p className="reference-notice">Thông tin tham khảo đã đối chiếu tài liệu; chưa thẩm định chuyên môn tại địa phương</p>
      <p className="secondary">Đối chiếu: <time dateTime={card.checked_at}>{date(card.checked_at)}</time> · {card.reviewer}</p>
      <h3>Dấu hiệu cần đối chiếu</h3>
      <ul>{card.symptoms.map((text, index) => <li key={index}>{text}</li>)}</ul>
      <h3>Dễ nhầm với</h3>
      <ul>{card.confusable_conditions.map((text, index) => <li key={index}>{text}</li>)}</ul>
      <h3>Quan sát thêm</h3>
      <ul>{card.observe_next.map((text, index) => <li key={index}>{text}</li>)}</ul>
      <h3>Việc có thể làm</h3>
      {card.actions.length ? <ul className="actions">{card.actions.map((action, index) => (
        <li key={index}><p>{action.text}</p><p className="secondary">Điều kiện áp dụng: {action.applicability}</p><p className="secondary">Căn cứ: {action.source_ids.map((id, sourceIndex) => <span key={id}>{sourceIndex > 0 && ', '}<a href={`#source-${id}`}>nguồn {card.sources.findIndex((source) => source.source_id === id) + 1}</a></span>)}</p></li>
      ))}</ul> : <p>Chưa có hướng dẫn hành động đủ căn cứ cho trường hợp này.</p>}
      <h3>Giới hạn thông tin</h3>
      <ul>{card.limits.map((text, index) => <li key={index}>{text}</li>)}</ul>
      <h3>Nguồn tham khảo</h3>
      <p className="secondary">Liên kết mở trong trình duyệt ngoài và cần kết nối mạng.</p>
      <ol className="sources">{card.sources.map((source) => (
        <li key={source.source_id} id={`source-${source.source_id}`}>
          <a href={source.url} onClick={(event) => { event.preventDefault(); void openSource(source.url); }} onAuxClick={(event) => event.preventDefault()}>{new URL(source.url).hostname} — {source.locator}</a>
          <p>Phạm vi: {source.geographic_scope}</p>
          <details><summary>Thông tin đối chiếu nguồn</summary>
            <p>Mã nguồn: {source.source_id}. Ngày kiểm tra: {source.checked_at}.</p>
            <p>{source.rights_basis}</p><p>{source.rights_evidence}</p>
          </details>
        </li>
      ))}</ol>
      {linkError && <p role="alert" className="error">{linkError}</p>}
    </section>
  );
}
