/* hanoiFlow — sidebar navigation (one page per hash), mobile menu. */
'use strict';

const PAGES = { intro: 'Giới thiệu', video: 'Phân tích video', overview: 'Bản đồ và thời tiết', live: 'Camera trực tiếp', tech: 'Thông tin kỹ thuật' };

function pageFromHash(hash) {
  const key = String(hash || '').replace(/^#/, '');
  if (key === '') return 'intro';
  if (key === 'how') return 'tech';
  return Object.prototype.hasOwnProperty.call(PAGES, key) ? key : 'video';
}

if (typeof module !== 'undefined') module.exports = { pageFromHash, PAGES };

if (typeof document !== 'undefined') {
  const listeners = [];
  window.hanoiFlow = { onPage: fn => listeners.push(fn), page: null };
  const $ = id => document.getElementById(id);
  const escapeHtml = text => String(text ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  function show() {
    const page = pageFromHash(location.hash);
    document.querySelectorAll('.page').forEach(el => el.classList.toggle('active', el.id === `page-${page}`));
    document.querySelectorAll('.nav-item').forEach(el => el.classList.toggle('active', el.dataset.page === page));
    $('pageCrumb').textContent = PAGES[page];
    $('sidebar').classList.remove('open');
    if (location.hash === '#how') { $('how').open = true; setTimeout(() => $('how').scrollIntoView({ behavior: 'smooth' }), 50); }
    window.hanoiFlow.page = page;
    listeners.forEach(fn => { try { fn(page); } catch (e) { console.error(e); } });
  }

  $('mobileMenu').addEventListener('click', () => $('sidebar').classList.toggle('open'));
  $('reloadBtn').addEventListener('click', () => location.reload());
  window.addEventListener('hashchange', show);
  show();
}
