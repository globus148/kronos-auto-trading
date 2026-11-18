/* Kronos — общие функции интерфейса. */
(function (w, d) {
    'use strict';

    // ─── Иконки ────────────────────────────────────────────
    function icon(name, cls) {
        return '<svg class="icon ' + (cls || '') + '" aria-hidden="true"><use href="#i-' + name + '"></use></svg>';
    }

    // ─── Форматирование ────────────────────────────────────
    // Адаптивные знаки для дешёвых монет (как фильтр pricefmt на сервере).
    function pricefmt(v) {
        if (v === null || v === undefined || isNaN(v)) return '—';
        return pricefmtRaw(v);
    }
    // Для полей ввода — без разделителей тысяч.
    function pricefmtRaw(v) {
        if (v === null || v === undefined || isNaN(v)) return '';
        var a = Math.abs(v);
        if (a >= 1000) return v.toFixed(2);
        if (a >= 1) return v.toFixed(4);
        if (a >= 0.1) return v.toFixed(5);
        if (a >= 0.001) return v.toFixed(6);
        return v.toFixed(8);
    }
    function escapeHtml(s) {
        return String(s).replace(/[&<>"']/g, function (c) {
            return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
        });
    }

    // ─── Тосты ─────────────────────────────────────────────
    function toast(msg, kind) {
        kind = kind || 'ok';
        var box = d.getElementById('toasts');
        if (!box) return;
        var el = d.createElement('div');
        el.className = 'toast ' + kind;
        var ic = kind === 'err' ? 'alert' : (kind === 'warn' ? 'info' : 'check');
        el.innerHTML = icon(ic) + '<span></span>';
        el.querySelector('span').textContent = msg;
        box.appendChild(el);
        setTimeout(function () { el.classList.add('out'); }, 3200);
        setTimeout(function () { el.remove(); }, 3500);
    }

    // ─── Модальные окна ────────────────────────────────────
    function openModal(id) {
        var m = d.getElementById(id);
        if (m) m.classList.add('open');
        return m;
    }
    function closeModal(id) {
        var m = typeof id === 'string' ? d.getElementById(id) : id;
        if (m) m.classList.remove('open');
    }
    d.addEventListener('click', function (e) {
        if (e.target.classList && e.target.classList.contains('modal-overlay')) closeModal(e.target);
        var closer = e.target.closest && e.target.closest('[data-close-modal]');
        if (closer) closeModal(closer.closest('.modal-overlay'));
    });
    d.addEventListener('keydown', function (e) {
        if (e.key !== 'Escape') return;
        d.querySelectorAll('.modal-overlay.open').forEach(function (m) { closeModal(m); });
    });

    function confirmDialog(opts) {
        return new Promise(function (resolve) {
            var m = d.getElementById('confirm-modal');
            if (!m) { resolve(w.confirm(opts.text || '')); return; }
            d.getElementById('confirm-title').textContent = opts.title || 'Подтверждение';
            d.getElementById('confirm-text').innerHTML = opts.html || escapeHtml(opts.text || '');
            var ok = d.getElementById('confirm-ok');
            ok.textContent = opts.ok || 'Подтвердить';
            ok.className = 'btn ' + (opts.danger ? 'btn-danger' : 'btn-primary');
            function done(v) {
                m.removeEventListener('click', onClick);
                closeModal(m);
                resolve(v);
            }
            function onClick(e) {
                var b = e.target.closest('[data-confirm]');
                if (b) done(b.getAttribute('data-confirm') === 'ok');
                else if (e.target === m) done(false);
            }
            m.addEventListener('click', onClick);
            openModal('confirm-modal');
            ok.focus();
        });
    }

    // ─── Шкала S_entry ─────────────────────────────────────
    // value: 0..1 или null, thr: порог, direction: 'up' | 'down'.
    function meterState(value, thr, direction) {
        var side = direction === 'down' ? 'is-short' : 'is-long';
        if (value === null || value === undefined || isNaN(value)) return 'is-idle ' + side;
        return (value >= thr ? 'is-pass ' : 'is-below ') + side;
    }
    function setState(el, base, state) {
        if (!el) return;
        el.className = base + ' ' + state;
    }
    function updateMeter(meterEl, valEl, value, thr, direction) {
        var st = meterState(value, thr, direction);
        if (meterEl) {
            var size = meterEl.classList.contains('meter-lg') ? ' meter-lg' : '';
            setState(meterEl, 'meter' + size, st);
            var fill = meterEl.querySelector('.meter-fill');
            var mark = meterEl.querySelector('.meter-mark');
            if (fill) fill.style.width = (value == null ? 0 : Math.max(0, Math.min(1, value)) * 100) + '%';
            if (mark && thr != null) mark.style.left = (thr * 100) + '%';
            meterEl.title = 'S_entry ' + (value == null ? '—' : value.toFixed(3)) + ' · порог ' + thr.toFixed(2);
        }
        if (valEl) {
            var base = valEl.classList.contains('big') ? 'big' : 'sval';
            setState(valEl, base, st);
            valEl.textContent = value == null ? '—' : value.toFixed(base === 'big' ? 3 : 2);
        }
    }
    function dirHtml(direction, prob) {
        if (prob === null || prob === undefined) return '';
        return icon(direction === 'down' ? 'arrow-down' : 'arrow-up') + Math.round(prob * 100) + '%';
    }
    function updateDir(el, direction, prob) {
        if (!el) return;
        el.className = 'dir' + (prob == null ? '' : (direction === 'down' ? ' down' : ' up'));
        el.innerHTML = dirHtml(direction, prob);
    }

    // ─── Plotly ────────────────────────────────────────────
    function cssVar(name) {
        return getComputedStyle(d.documentElement).getPropertyValue(name).trim();
    }
    function theme() {
        return {
            bg: cssVar('--surface'), grid: cssVar('--border'), text: cssVar('--text-2'),
            textDim: cssVar('--text-3'), accent: cssVar('--accent'),
            green: cssVar('--green'), red: cssVar('--red'), amber: cssVar('--amber'),
        };
    }
    function plotLayout(extra) {
        var t = theme();
        var base = {
            paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
            font: { family: 'Inter, Segoe UI, sans-serif', color: t.text, size: 11 },
            margin: { t: 8, r: 12, b: 40, l: 56 },
            xaxis: { showgrid: false, linecolor: t.grid, tickcolor: t.grid, zeroline: false },
            yaxis: { gridcolor: t.grid, zeroline: false, linecolor: t.grid },
            showlegend: true,
            legend: { orientation: 'h', x: 0, y: 1.08, bgcolor: 'rgba(0,0,0,0)', font: { color: t.text } },
            hoverlabel: { bgcolor: cssVar('--surface-3'), bordercolor: t.grid, font: { color: cssVar('--text') } },
            hovermode: 'x unified',
        };
        extra = extra || {};
        Object.keys(extra).forEach(function (k) {
            if (base[k] && typeof base[k] === 'object' && typeof extra[k] === 'object' && !Array.isArray(extra[k])) {
                base[k] = Object.assign({}, base[k], extra[k]);
            } else {
                base[k] = extra[k];
            }
        });
        return base;
    }
    var plotConfig = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'] };

    // ─── Переключатель режима Paper / Live ─────────────────
    function initModeSwitch() {
        var box = d.getElementById('mode-switch');
        if (!box) return;
        box.addEventListener('click', async function (e) {
            var btn = e.target.closest('button[data-mode]');
            if (!btn || btn.classList.contains('active')) return;
            var mode = btn.getAttribute('data-mode');
            if (mode === 'live') {
                var ok = await confirmDialog({
                    title: 'Перейти в Live-режим?',
                    html: 'В Live-режиме сделки идут на бирже <strong>с настоящими деньгами</strong>. ' +
                          'Авто-торговля в Live работает только если она отдельно разрешена в настройках.',
                    ok: 'Перейти в Live', danger: true,
                });
                if (!ok) return;
            }
            box.querySelectorAll('button').forEach(function (b) { b.disabled = true; });
            var fd = new FormData();
            fd.append('mode', mode);
            try {
                var r = await fetch('/api/mode', { method: 'POST', body: fd });
                var data = await r.json();
                if (data.mode) {
                    // Режим — это другой счёт (баланс, позиции), поэтому данные страницы перечитываем.
                    location.reload();
                    return;
                }
            } catch (err) { /* ниже */ }
            box.querySelectorAll('button').forEach(function (b) { b.disabled = false; });
            toast('Не удалось переключить режим', 'err');
        });
    }

    // ─── HTMX: единая обработка ошибок ─────────────────────
    d.addEventListener('htmx:responseError', function (e) {
        var s = e.detail.xhr ? e.detail.xhr.status : 0;
        if (s === 401) { location.href = '/login'; return; }
        toast('Ошибка запроса (' + s + ')', 'err');
    });
    d.addEventListener('htmx:sendError', function () { toast('Нет связи с сервером', 'err'); });

    // hx-confirm → своё окно подтверждения вместо системного confirm().
    d.addEventListener('htmx:confirm', function (e) {
        if (!e.detail.question) return;
        e.preventDefault();
        confirmDialog({ title: 'Подтверждение', text: e.detail.question, ok: 'Удалить', danger: true })
            .then(function (ok) { if (ok) e.detail.issueRequest(true); });
    });

    d.addEventListener('DOMContentLoaded', initModeSwitch);

    w.K = {
        icon: icon, pricefmt: pricefmt, pricefmtRaw: pricefmtRaw, esc: escapeHtml,
        toast: toast, openModal: openModal, closeModal: closeModal, confirm: confirmDialog,
        meterState: meterState, updateMeter: updateMeter, updateDir: updateDir,
        theme: theme, plotLayout: plotLayout, plotConfig: plotConfig,
    };
    // Совместимость со старым кодом.
    w.pricefmtJS = pricefmtRaw;
})(window, document);
