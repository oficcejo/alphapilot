/* ═══════════════════════════════════════════════════════════════
 * OKX AlphaPilot — 前端 SPA 应用
 * ═══════════════════════════════════════════════════════════════ */

const API = '/api';

// ── Utility ────────────────────────────────────────────────────────
async function fetchJSON(url, opts = {}) {
  const resp = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  });
  if (!resp.ok) {
    let msg = `HTTP ${resp.status}`;
    try { const e = await resp.json(); msg = e.detail || e.message || msg; } catch {}
    throw new Error(msg);
  }
  return resp.json();
}

function toast(msg, type = 'info') {
  const c = document.getElementById('toast-container');
  const t = document.createElement('div');
  t.className = `toast toast-${type}`;
  t.textContent = msg;
  c.appendChild(t);
  setTimeout(() => { t.style.opacity = '0'; setTimeout(() => t.remove(), 300); }, 4000);
}

function fmtPct(v) { return (v >= 0 ? '+' : '') + v.toFixed(2) + '%'; }
function fmtNum(v, d = 2) { return Number(v).toFixed(d); }
function fmtTime(ts) {
  if (!ts) return '-';
  const n = Number(ts);
  if (n > 1e12) return new Date(n).toLocaleString('zh-CN');
  if (n > 1e9) return new Date(n * 1000).toLocaleString('zh-CN');
  return String(ts);
}

function destroyChart(id) {
  if (window._charts && window._charts[id]) { window._charts[id].destroy(); delete window._charts[id]; }
}
function setChart(id, chart) {
  if (!window._charts) window._charts = {};
  destroyChart(id);
  window._charts[id] = chart;
}

// ── SPA Router ─────────────────────────────────────────────────────
const App = {
  currentPage: 'dashboard',
  pollTimer: null,

  navigate(page) {
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    document.getElementById(`page-${page}`).classList.add('active');
    document.querySelector(`.nav-item[data-page="${page}"]`)?.classList.add('active');
    this.currentPage = page;
    if (this.pollTimer) { clearInterval(this.pollTimer); this.pollTimer = null; }
    this.render(page);
  },

  async render(page) {
    const fn = this[`render_${page}`];
    if (fn) { try { await fn.call(this); } catch (e) { toast(`页面加载失败: ${e.message}`, 'error'); } }
  },

  // ════════ Dashboard ════════
  async render_dashboard() {
    const el = document.getElementById('dashboard-content');
    el.innerHTML = '<div class="loading-overlay"><span class="spinner"></span> 加载系统信息...</div>';
    try {
      const [sys, strategies, parquets] = await Promise.all([
        fetchJSON(`${API}/system`),
        fetchJSON(`${API}/training/strategies`),
        fetchJSON(`${API}/data/parquets`),
      ]);
      const stratCount = strategies.strategies?.length || 0;
      const dataCount = parquets.files?.length || 0;
      const mode = sys.is_live ? 'LIVE' : 'PAPER';
      const modeClass = sys.is_live ? 'live' : 'paper';

      el.innerHTML = `
        <div class="grid-4 mb-6">
          <div class="stat-card">
            <div class="stat-label">交易模式</div>
            <div class="stat-value ${sys.is_live ? 'negative' : 'warning'}">${mode}</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">策略数量</div>
            <div class="stat-value neutral">${stratCount}</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">数据文件</div>
            <div class="stat-value neutral">${dataCount}</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">词表大小</div>
            <div class="stat-value neutral">${sys.vocab_size}</div>
          </div>
        </div>

        <div class="grid-2">
          <div class="card">
            <div class="card-title">最近策略</div>
            ${stratCount > 0 ? `
              <div class="table-wrapper">
                <table>
                  <thead><tr><th>品种</th><th>分数</th><th>公式</th></tr></thead>
                  <tbody>
                    ${strategies.strategies.slice(0, 5).map(s => `
                      <tr>
                        <td>${s.symbol || '-'}</td>
                        <td class="text-mono ${s.best_score > 0 ? 'text-success' : 'text-muted'}">${fmtNum(s.best_score, 3)}</td>
                        <td class="text-mono text-sm text-secondary" style="max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${s.formula_decoded || '-'}</td>
                      </tr>`).join('')}
                  </tbody>
                </table>
              </div>` : '<div class="text-muted text-center" style="padding:20px">暂无策略，请先训练模型</div>'}
          </div>
          <div class="card">
            <div class="card-title">数据文件</div>
            ${dataCount > 0 ? `
              <div class="table-wrapper">
                <table>
                  <thead><tr><th>品种</th><th>周期</th><th>K线数</th><th>大小</th></tr></thead>
                  <tbody>
                    ${parquets.files.slice(0, 5).map(f => `
                      <tr>
                        <td>${f.symbol || '-'}</td>
                        <td>${f.timeframe || '-'}</td>
                        <td class="text-mono">${f.n_bars || 0}</td>
                        <td class="text-mono text-muted">${f.file_size_mb || 0} MB</td>
                      </tr>`).join('')}
                  </tbody>
                </table>
              </div>` : '<div class="text-muted text-center" style="padding:20px">暂无数据，请先下载 K 线</div>'}
          </div>
        </div>

        <div class="card mt-6">
          <div class="card-title">合规声明</div>
          <div class="alert alert-warning">
            <strong>重要边界：</strong>
            <ul style="margin-left:16px;margin-top:4px">
              <li>默认模式是 <code>paper</code>（模拟盘），不会发送真实订单。</li>
              <li>训练和回测只使用本地 Parquet，不调用 OKX 私有接口。</li>
              <li>真实交易必须显式开启 live 闸门，并配置完整 API 凭证。</li>
              <li>项目不保证收益，也不构成投资建议。</li>
            </ul>
          </div>
          <div class="disclosure">
            本项目与 OKX 官方无隶属或背书关系；名称中的 OKX 仅表示主要适配的交易所接口。
          </div>
        </div>
      `;
    } catch (e) {
      el.innerHTML = `<div class="alert alert-danger">加载失败: ${e.message}</div>`;
    }
  },

  // ════════ Training ════════
  async render_training() {
    const el = document.getElementById('training-content');
    try {
      const [parquets, strategies] = await Promise.all([
        fetchJSON(`${API}/data/parquets`),
        fetchJSON(`${API}/training/strategies`),
      ]);

      // 读取待训练数据队列（从数据页"一键入训练"传入）
      const pending = JSON.parse(localStorage.getItem('pending_train_data') || '[]');
      let pendingHtml = '';
      if (pending.length > 0) {
        pendingHtml = `
          <div class="card" style="border-color:var(--success);background:rgba(16,185,129,0.05)">
            <div class="card-title" style="color:var(--success)">待训练数据队列（${pending.length} 项）</div>
            <div class="table-wrapper">
              <table>
                <thead><tr><th>数据文件</th><th>品种</th><th>操作</th></tr></thead>
                <tbody>
                  ${pending.map((p, idx) => `
                    <tr>
                      <td class="text-mono text-sm">${(p.file_path || '').split(/[\\\\/]/).pop() || '-'}</td>
                      <td>${p.symbol || '-'}</td>
                      <td>
                        <button class="btn btn-success btn-sm" onclick="App.usePendingData(${idx})">选用</button>
                        <button class="btn btn-ghost btn-sm" onclick="App.removePendingData(${idx})">移除</button>
                      </td>
                    </tr>`).join('')}
                </tbody>
              </table>
            </div>
            <button class="btn btn-ghost btn-sm mt-2" onclick="App.clearPendingData()">清空队列</button>
          </div>`;
      }

      const parquetOptions = (parquets.files || []).map(f =>
        `<option value="${f.file_path}" data-symbol="${f.symbol || ''}" data-timeframe="${f.timeframe || ''}">${f.file_name} (${f.symbol} ${f.timeframe}, ${f.n_bars} bars)</option>`
      ).join('') || '<option value="">无可用数据</option>';

      el.innerHTML = `
        ${pendingHtml}
        <div class="grid-2">
          <div>
            <div class="card">
              <div class="card-title">训练配置</div>
              <div class="form-group">
                <label class="form-label">数据文件 (Parquet)</label>
                <select class="form-select" id="train-data-file" onchange="App.onTrainDataChange()">${parquetOptions}</select>
              </div>
              <div class="form-row">
                <div class="form-group">
                  <label class="form-label">品种标识（自动填充）</label>
                  <input class="form-input" id="train-symbol" placeholder="如 BTC-USDT-SWAP" onchange="App.onTrainDataChange()">
                </div>
                <div class="form-group">
                  <label class="form-label">训练步数（0=默认9000）</label>
                  <input class="form-input" id="train-steps" type="number" value="0" min="0">
                </div>
              </div>
              <div class="form-row">
                <div class="form-group">
                  <label class="form-label">奖励模式</label>
                  <select class="form-select" id="train-reward-mode">
                    <option value="ftmo" selected>FTMO（年化收益优先）</option>
                    <option value="standard">Standard（收益+风险平衡）</option>
                    <option value="forex">Forex（均值回归）</option>
                  </select>
                </div>
              <div class="flex gap-2 mt-4">
                <button class="btn btn-primary flex-1" onclick="App.startTraining()">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"/><circle cx="12" cy="12" r="9"/></svg>
                  启动训练
                </button>
                <button class="btn btn-danger" id="btn-stop-training" onclick="App.stopTraining()" style="display:none">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="6" y="6" width="12" height="12" rx="1"/></svg>
                  停止训练
                </button>
                <button class="btn btn-ghost" id="btn-reset-training" onclick="App.resetTraining()" title="清除检查点和历史，下次从头训练">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/></svg>
                  重置训练
                </button>
              </div>
              <div id="train-stop-msg" class="mt-2"></div>
              <div id="train-reset-msg" class="mt-2"></div>
            </div>

            <div class="card mt-6">
              <div class="card-title">训练状态</div>
              <div id="train-status"><div class="text-muted text-center" style="padding:20px">无训练任务</div></div>
            </div>
          </div>

          <div>
            <div class="card">
              <div class="card-title">训练曲线</div>
              <div id="train-chart-area">
                <div class="text-muted text-center" style="padding:40px">启动训练后显示曲线</div>
              </div>
            </div>
          </div>
        </div>

        <div class="card mt-6">
          <div class="card-title">已保存策略</div>
          ${(strategies.strategies || []).length > 0 ? `
            <div class="table-wrapper">
              <table>
                <thead><tr><th>文件</th><th>品种</th><th>周期</th><th>分数</th><th>公式</th><th>操作</th></tr></thead>
                <tbody>
              ${strategies.strategies.map(s => `
                <tr>
                  <td class="text-sm">${s.file_name || '-'}</td>
                  <td>${s.symbol || '-'}</td>
                  <td>${s.timeframe ? `<span class="badge badge-info">${s.timeframe}</span>` : '-'}</td>
                  <td class="text-mono ${s.best_score > 0 ? 'text-success' : 'text-muted'}">${fmtNum(s.best_score, 3)}</td>
                  <td class="text-mono text-sm text-secondary" style="max-width:400px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${s.formula_decoded || '-'}</td>
                  <td><button class="btn btn-ghost btn-sm" onclick="App.exportStrategy('${s.file_path}')">导出</button> <button class="btn btn-danger btn-sm" onclick="App.deleteStrategy('${s.file_path}', '${s.file_name || ''}', '${s.symbol || ''}', '${s.timeframe || ''}')">删除</button></td>
                </tr>`).join('')}
              </tbody></table>
            </div>` : '<div class="text-muted text-center" style="padding:20px">暂无策略</div>'}
        </div>
      `;

      // 页面加载时检查是否有训练在运行，自动启动轮询
      this.refreshTrainingStatus();
      fetchJSON(`${API}/training/status`).then(s => {
        if (s.running && !this.pollTimer) {
          this.pollTimer = setInterval(() => this.refreshTrainingStatus(), 3000);
        }
      }).catch(() => {});
    } catch (e) {
      el.innerHTML = `<div class="alert alert-danger">加载失败: ${e.message}</div>`;
    }
  },

  async startTraining() {
    const dataFile = document.getElementById('train-data-file').value;
    if (!dataFile) { toast('请选择数据文件', 'error'); return; }
    const symbol = document.getElementById('train-symbol').value || null;
const steps = parseInt(document.getElementById('train-steps').value) || 0;
const rewardMode = document.getElementById('train-reward-mode').value;
try {
const r = await fetchJSON(`${API}/training/start`, {
method: 'POST',
body: JSON.stringify({ data_file: dataFile, symbol, train_steps: steps, reward_mode: rewardMode }),
});
      toast(`训练已启动: ${r.symbol || '未知品种'} ${r.timeframe || ''}, ${r.steps} 步`, 'success');
      this.pollTimer = setInterval(() => this.refreshTrainingStatus(), 3000);
      this.refreshTrainingStatus();
    } catch (e) {
      toast(`启动失败: ${e.message}`, 'error');
    }
  },

  async stopTraining() {
    try {
      const r = await fetchJSON(`${API}/training/stop`, { method: 'POST' });
      toast(r.message || '已请求停止', 'warning');
      const msgEl = document.getElementById('train-stop-msg');
      if (msgEl) msgEl.innerHTML = '<div class="alert alert-warning">已请求停止训练，等待当前步完成...</div>';
    } catch (e) {
      toast(`停止失败: ${e.message}`, 'error');
    }
  },

  async resetTraining() {
    // 获取当前选择的品种和周期
    const symbol = document.getElementById('train-symbol').value || null;
    const dataSel = document.getElementById('train-data-file');
    const selOpt = dataSel ? dataSel.options[dataSel.selectedIndex] : null;
    const timeframe = selOpt ? (selOpt.getAttribute('data-timeframe') || '') : '';

    // 确认对话框
    const target = [symbol, timeframe].filter(Boolean).join(' / ') || '所有品种和周期';
    if (!confirm(`确定要重置训练吗？\n\n目标：${target}\n\n将删除：\n- 训练历史（.json）\n\n注意：策略 JSON 默认保留，如需删除请勾选。\n\n重置后下次训练将从头开始。`)) {
      return;
    }
    const deleteStrategy = confirm('是否同时删除已保存的策略 JSON 文件？\n（取消 = 保留策略，确定 = 同时删除策略）');

    try {
      const r = await fetchJSON(`${API}/training/reset`, {
        method: 'POST',
body: JSON.stringify({
symbol: symbol || null,
timeframe: timeframe || null,
delete_strategy: deleteStrategy,
delete_history: true,
}),
      });
      if (r.status === 'error') {
        toast(r.message, 'error');
        return;
      }
      toast(r.message, 'success');
      const msgEl = document.getElementById('train-reset-msg');
      if (msgEl) {
        const d = r.deleted || {};
        msgEl.innerHTML = `<div class="alert alert-success">
<strong>训练已重置</strong><br>
删除策略: ${d.strategies?.length || 0} 个<br>
删除历史: ${d.histories?.length || 0} 个<br>
          <span class="text-muted text-sm">下次训练将从头开始</span>
        </div>`;
      }
      // 刷新页面以更新检查点下拉列表和策略表
      setTimeout(() => this.render_training(), 1500);
    } catch (e) {
      toast(`重置失败: ${e.message}`, 'error');
    }
  },

  onTrainDataChange() {
    // 数据文件变化时，自动填充品种标识，并清空检查点选择（默认从头训练）
    const dataSel = document.getElementById('train-data-file');
    if (!dataSel) return;
    const opt = dataSel.options[dataSel.selectedIndex];
    if (!opt) return;
    const symbol = opt.getAttribute('data-symbol') || '';
    const timeframe = opt.getAttribute('data-timeframe') || '';

    // 自动填充品种标识（仅当用户未手动修改时）
    const symInput = document.getElementById('train-symbol');
    if (symInput && !symInput.dataset.userModified) {
      symInput.value = symbol;
    }

    // 标记品种输入框为用户可修改
    if (symInput) {
      symInput.addEventListener('input', () => { symInput.dataset.userModified = '1'; }, { once: true });
    }
  },

  exportStrategy(strategyPath) {
    window.open(`${API}/training/export?strategy_path=${encodeURIComponent(strategyPath)}`, '_blank');
    toast('策略下载已开始', 'success');
  },

  async deleteStrategy(strategyPath, fileName, symbol, timeframe) {
    const label = [symbol, timeframe].filter(Boolean).join(' / ') || fileName || '该策略';
    if (!confirm(`确定要删除策略吗？\n\n策略：${label}\n文件：${fileName || '-'}\n\n此操作不可恢复，删除后需要重新训练才能恢复该策略。`)) {
      return;
    }
    try {
      const r = await fetchJSON(`${API}/training/strategies/delete`, {
        method: 'POST',
        body: JSON.stringify({ strategy_path: strategyPath }),
      });
      if (r.ok) {
        toast(`策略已删除: ${r.deleted || fileName}`, 'success');
        this.render_training();
      } else {
        toast(r.msg || '删除失败', 'error');
      }
    } catch (e) {
      toast(`删除失败: ${e.message}`, 'error');
    }
  },

  usePendingData(idx) {
    const pending = JSON.parse(localStorage.getItem('pending_train_data') || '[]');
    const item = pending[idx];
    if (!item) return;
    const sel = document.getElementById('train-data-file');
    if (sel) sel.value = item.file_path;
    const sym = document.getElementById('train-symbol');
    if (sym) sym.value = item.symbol || '';
    toast(`已选用: ${item.symbol || item.file_path}`, 'success');
  },

  removePendingData(idx) {
    const pending = JSON.parse(localStorage.getItem('pending_train_data') || '[]');
    pending.splice(idx, 1);
    localStorage.setItem('pending_train_data', JSON.stringify(pending));
    this.render_training();
  },

  clearPendingData() {
    localStorage.removeItem('pending_train_data');
    this.render_training();
    toast('已清空训练队列', 'info');
  },

  async refreshTrainingStatus() {
    try {
      const s = await fetchJSON(`${API}/training/status`);
      const el = document.getElementById('train-status');
      if (!el) return;
      if (!s.running && !s.error && s.best_score < -900) {
        el.innerHTML = '<div class="text-muted text-center" style="padding:20px">无训练任务</div>';
        const stopBtn = document.getElementById('btn-stop-training');
        if (stopBtn) stopBtn.style.display = 'none';
        return;
      }
      // 显示/隐藏停止按钮
      const stopBtn = document.getElementById('btn-stop-training');
      if (stopBtn) stopBtn.style.display = s.running ? '' : 'none';
      const progress = s.end_step > 0 ? Math.min(100, (s.current_step / s.end_step * 100)) : 0;
      const scoreClass = s.best_score > 0 ? 'positive' : 'negative';
      el.innerHTML = `
        <div class="grid-2 mb-4">
          <div>
            <div class="stat-label">品种 / 周期</div>
            <div class="text-mono" style="font-size:16px;font-weight:600">${s.symbol || '-'} ${s.timeframe ? '<span class="badge badge-info">' + s.timeframe + '</span>' : ''}</div>
          </div>
          <div>
            <div class="stat-label">状态</div>
            <div>${s.running ? '<span class="badge badge-info">运行中</span>' : '<span class="badge badge-muted">已停止</span>'}</div>
          </div>
        </div>
        <div class="form-group">
          <label class="form-label">进度: ${s.current_step} / ${s.end_step} (${progress.toFixed(1)}%)</label>
          <div class="progress-bar"><div class="progress-fill ${s.running ? '' : 'success'}" style="width:${progress}%"></div></div>
        </div>
        <div class="grid-3 mt-4">
          <div><div class="stat-label">最优分数</div><div class="stat-value ${scoreClass}" style="font-size:18px">${fmtNum(s.best_score, 3)}</div></div>
          <div><div class="stat-label">重启次数</div><div class="text-mono" style="font-size:18px;font-weight:600">${s.restart_count || 0}</div></div>
          <div><div class="stat-label">策略路径</div><div class="text-mono text-sm" style="word-break:break-all">${(s.strategy_path || '-').split(/[\\/]/).pop()}</div></div>
        </div>
        ${s.best_formula_decoded ? `<div class="mt-4"><div class="stat-label">最优公式</div><div class="text-mono text-sm" style="background:var(--bg-primary);padding:10px;border-radius:6px;border:1px solid var(--border)">${s.best_formula_decoded}</div></div>` : ''}
        ${s.error ? `<div class="alert alert-danger mt-4">${s.error}</div>` : ''}
      `;
      if (s.running) {
        this.loadTrainingChart(s.symbol, s.timeframe);
      } else {
        // 训练已停止/完成：清除轮询，加载最终曲线
        if (this.pollTimer) { clearInterval(this.pollTimer); this.pollTimer = null; }
        const msgEl = document.getElementById('train-stop-msg');
        if (msgEl) msgEl.innerHTML = '';
        this.loadTrainingChart(s.symbol, s.timeframe);
      }
    } catch (e) { /* silent */ }
  },

  async loadTrainingChart(symbol, timeframe) {
    try {
      const params = new URLSearchParams();
      if (symbol) params.set('symbol', symbol);
      if (timeframe) params.set('timeframe', timeframe);
      const qs = params.toString();
      const hist = await fetchJSON(`${API}/training/history${qs ? '?' + qs : ''}`);
      if (hist.error) return;
      const el = document.getElementById('train-chart-area');
      if (!el) return;
      const steps = hist.step || [];
      const scores = hist.best_score || [];
      const rewards = hist.avg_reward || [];
      const entropy = hist.entropy || [];
      if (steps.length === 0) { el.innerHTML = '<div class="text-muted text-center" style="padding:40px">暂无训练数据</div>'; return; }
      el.innerHTML = '<div class="chart-container lg"><canvas id="train-chart"></canvas></div>';
      const ctx = document.getElementById('train-chart').getContext('2d');
      setChart('train', new Chart(ctx, {
        type: 'line',
        data: {
          labels: steps.map(s => s + 1),
          datasets: [
            { label: '最优分数', data: scores, borderColor: '#10b981', backgroundColor: 'rgba(16,185,129,0.1)', yAxisID: 'y', tension: 0.3, pointRadius: 0 },
            { label: '平均奖励', data: rewards, borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.1)', yAxisID: 'y', tension: 0.3, pointRadius: 0 },
            { label: '策略熵', data: entropy, borderColor: '#f59e0b', backgroundColor: 'rgba(245,158,11,0.1)', yAxisID: 'y1', tension: 0.3, pointRadius: 0, borderDash: [4, 4] },
          ],
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          interaction: { mode: 'index', intersect: false },
          scales: {
            y: { position: 'left', grid: { color: '#2a3447' }, ticks: { color: '#94a3b8' } },
            y1: { position: 'right', grid: { drawOnChartArea: false }, ticks: { color: '#f59e0b' } },
            x: { grid: { color: '#2a3447' }, ticks: { color: '#94a3b8', maxTicksLimit: 10 } },
          },
          plugins: { legend: { labels: { color: '#e2e8f0', font: { size: 11 } } } },
        },
      }));
    } catch (e) { /* silent */ }
  },

  // ════════ Backtest ════════
  async render_backtest() {
    const el = document.getElementById('backtest-content');
    try {
      const [strategies, parquets] = await Promise.all([
        fetchJSON(`${API}/training/strategies`),
        fetchJSON(`${API}/data/parquets`),
      ]);
      const stratOpts = (strategies.strategies || []).map(s =>
        `<option value="${s.file_path}">${s.file_name} (${s.symbol || '-'}, score=${fmtNum(s.best_score, 2)})</option>`
      ).join('') || '<option value="">无策略</option>';
      const dataOpts = (parquets.files || []).map(f =>
        `<option value="${f.file_path}">${f.file_name} (${f.symbol} ${f.timeframe}, ${f.n_bars} bars)</option>`
      ).join('') || '<option value="">无数据</option>';

      el.innerHTML = `
        <div class="grid-2">
          <div class="card">
            <div class="card-title flex items-center justify-between">
              <span>回测配置</span>
              <div class="flex gap-1">
                <input type="file" id="bt-import-file" accept=".json" style="display:none" onchange="App.importStrategy(this, 'bt-strategy')">
                <button class="btn btn-sm btn-ghost" onclick="document.getElementById('bt-import-file').click()" style="padding:2px 8px;font-size:12px" title="导入外部或导出的策略 JSON 文件">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="vertical-align:-2px;margin-right:2px"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
                  导入策略
                </button>
                <button class="btn btn-sm btn-ghost" onclick="App.openPortfolioModal('bt-strategy')" style="padding:2px 8px;font-size:12px" title="将多个单因子策略融合成多因子组合策略">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="vertical-align:-2px;margin-right:2px"><path d="M12 20v-6M6 20V10M18 20V4"/></svg>
                  构建组合
                </button>
              </div>
            </div>
            <div class="form-group">
              <label class="form-label">策略文件</label>
              <select class="form-select" id="bt-strategy">${stratOpts}</select>
            </div>
            <div class="form-group">
              <label class="form-label">数据文件</label>
              <select class="form-select" id="bt-data">${dataOpts}</select>
            </div>
            <div class="form-row">
              <div class="form-group"><label class="form-label">手续费率</label><input class="form-input" id="bt-cost" type="number" value="0.0005" step="0.0001"></div>
              <div class="form-group"><label class="form-label">滑点</label><input class="form-input" id="bt-slippage" type="number" value="0.0003" step="0.0001"></div>
            </div>
            <div class="form-row">
              <div class="form-group"><label class="form-label">初始资金 (USDT)</label><input class="form-input" id="bt-capital" type="number" value="10000" step="100"></div>
              <div class="form-group"><label class="form-label">杠杆</label><input class="form-input" id="bt-leverage" type="number" value="5" min="1" max="100"></div>
            </div>
            <button class="btn btn-success w-full" onclick="App.runBacktest()">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"/><circle cx="12" cy="12" r="9"/></svg>
              运行回测
            </button>
          </div>
          <div id="bt-result-area">
            <div class="card"><div class="card-title">回测结果</div><div class="text-muted text-center" style="padding:40px">配置参数后点击「运行回测」</div></div>
          </div>
        </div>
      `;
    } catch (e) {
      el.innerHTML = `<div class="alert alert-danger">加载失败: ${e.message}</div>`;
    }
  },

  async runBacktest() {
    const strategy = document.getElementById('bt-strategy').value;
    const dataFile = document.getElementById('bt-data').value;
    if (!strategy) { toast('请选择策略', 'error'); return; }
    if (!dataFile) { toast('请选择数据文件', 'error'); return; }
    const area = document.getElementById('bt-result-area');
    area.innerHTML = '<div class="card"><div class="card-title">回测结果</div><div class="loading-overlay"><span class="spinner"></span> 回测计算中...</div></div>';
    try {
      const r = await fetchJSON(`${API}/backtest/run`, {
        method: 'POST',
        body: JSON.stringify({
          strategy_path: strategy, data_file: dataFile,
          cost_rate: parseFloat(document.getElementById('bt-cost').value),
          slippage: parseFloat(document.getElementById('bt-slippage').value),
          initial_capital: parseFloat(document.getElementById('bt-capital').value),
          leverage: parseInt(document.getElementById('bt-leverage').value),
        }),
      });
      this.renderBacktestResult(r);
      toast('回测完成', 'success');
    } catch (e) {
      area.innerHTML = `<div class="card"><div class="card-title">回测结果</div><div class="alert alert-danger">回测失败: ${e.message}</div></div>`;
      toast(`回测失败: ${e.message}`, 'error');
    }
  },

  renderBacktestResult(r) {
    const p = r.performance;
    const dq = r.data_quality || {};
    const area = document.getElementById('bt-result-area');
    const retClass = p.total_return >= 0 ? 'positive' : 'negative';
    const ddClass = p.max_drawdown > 10 ? 'negative' : 'neutral';

    // 警告横幅
    let warningBanner = '';
    if (dq.warnings && dq.warnings.length > 0) {
      const items = dq.warnings.map(w => `<li>${w}</li>`).join('');
      warningBanner = `
        <div class="card" style="border-color:var(--warning);background:rgba(245,158,11,0.08);margin-bottom:16px">
          <div class="card-title" style="color:var(--warning)">⚠ 数据质量警告</div>
          <ul style="margin:8px 0 0 20px;color:var(--text-secondary);font-size:13px;line-height:1.8">${items}</ul>
        </div>`;
    }

    // 数据时长信息
    const durInfo = dq.data_duration_days != null
      ? `<div class="text-muted text-sm" style="margin-bottom:12px">
           数据时长: <span class="text-mono">${dq.data_duration_days.toFixed(1)} 天</span>
           (${dq.data_duration_years.toFixed(4)} 年) ·
           每年周期数: <span class="text-mono">${dq.periods_per_year || '-'}</span>
           ${dq.is_in_sample ? ' · <span style="color:var(--danger)">样本内回测</span>' : ''}
         </div>` : '';

    // 截断标记
    const clipMark = (clipped) => clipped
      ? ' <span style="color:var(--warning);font-size:10px" title="原始值超出截断上限，不可信">⚠截断</span>' : '';

    // 年化收益：短样本时标注
    const annRetLabel = dq.is_short_sample ? '年化收益(参考)' : '年化收益';

    area.innerHTML = `
      ${warningBanner}
      <div class="card">
        <div class="card-title">绩效指标</div>
        ${durInfo}
        <div class="grid-4 mb-4">
          <div class="stat-card"><div class="stat-label">总收益</div><div class="stat-value ${retClass}">${fmtPct(p.total_return)}</div></div>
          <div class="stat-card"><div class="stat-label">${annRetLabel}</div><div class="stat-value ${retClass}">${fmtPct(p.annualized_return)}</div></div>
          <div class="stat-card"><div class="stat-label">最大回撤</div><div class="stat-value ${ddClass}">${p.max_drawdown.toFixed(2)}%</div></div>
          <div class="stat-card"><div class="stat-label">夏普比率${clipMark(p.sharpe_clipped)}</div><div class="stat-value neutral">${fmtNum(p.sharpe)}</div></div>
          <div class="stat-card"><div class="stat-label">Sortino${clipMark(p.sortino_clipped)}</div><div class="stat-value neutral">${fmtNum(p.sortino)}</div></div>
          <div class="stat-card"><div class="stat-label">Calmar${clipMark(p.calmar_clipped)}</div><div class="stat-value neutral">${fmtNum(p.calmar)}</div></div>
          <div class="stat-card"><div class="stat-label">胜率</div><div class="stat-value neutral">${p.win_rate.toFixed(1)}%</div></div>
          <div class="stat-card"><div class="stat-label">综合评分</div><div class="stat-value neutral">${fmtNum(p.score)}</div></div>
        </div>
        <div class="grid-3">
          <div><span class="text-muted text-sm">在场时间</span> <span class="text-mono">${p.exposure.toFixed(1)}%</span></div>
          <div><span class="text-muted text-sm">多头比例</span> <span class="text-mono text-success">${p.long_ratio.toFixed(1)}%</span></div>
          <div><span class="text-muted text-sm">空头比例</span> <span class="text-mono text-danger">${p.short_ratio.toFixed(1)}%</span></div>
        </div>
        ${(p.annualized_return_linear != null && Math.abs(p.annualized_return_linear - p.annualized_return) > 1) ? `
        <div class="text-muted text-sm" style="margin-top:12px;padding-top:8px;border-top:1px solid var(--border)">
          年化收益对比 — CAGR(几何): <span class="text-mono">${fmtPct(p.annualized_return_cagr || p.annualized_return)}</span>
          · 线性外推: <span class="text-mono" style="color:var(--warning)">${fmtPct(p.annualized_return_linear)}</span>
        </div>` : (dq.is_short_sample ? `
        <div class="text-muted text-sm" style="margin-top:12px;padding-top:8px;border-top:1px solid var(--border)">
          年化收益(线性外推): <span class="text-mono" style="color:var(--warning)">${fmtPct(p.annualized_return_linear)}</span>
          <span style="color:var(--text-muted)">· CAGR 因样本过短无意义，已隐藏</span>
        </div>` : '')}
      </div>
      <div class="card mt-4">
        <div class="card-title">资金曲线</div>
        <div class="chart-container lg"><canvas id="bt-equity-chart"></canvas></div>
      </div>
      <div class="card mt-4">
        <div class="card-title">策略公式</div>
        <div class="text-mono text-sm" style="background:var(--bg-primary);padding:12px;border-radius:6px;border:1px solid var(--border);word-break:break-all">${r.strategy.formula_decoded}</div>
      </div>
    `;
    // 资金曲线
    const ctx = document.getElementById('bt-equity-chart').getContext('2d');
    setChart('bt-equity', new Chart(ctx, {
      type: 'line',
      data: {
        labels: r.equity_curve.time.map(t => fmtTime(t)),
        datasets: [{
          label: '资金曲线', data: r.equity_curve.equity,
          borderColor: '#10b981', backgroundColor: 'rgba(16,185,129,0.1)',
          fill: true, tension: 0.3, pointRadius: 0,
        }],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        scales: {
          y: { grid: { color: '#2a3447' }, ticks: { color: '#94a3b8' } },
          x: { grid: { color: '#2a3447' }, ticks: { color: '#94a3b8', maxTicksLimit: 8 } },
        },
        plugins: { legend: { labels: { color: '#e2e8f0' } } },
      },
    }));
  },

  // ════════ Analysis ════════
  async render_analysis() {
    const el = document.getElementById('analysis-content');
    try {
      const [strategies, parquets] = await Promise.all([
        fetchJSON(`${API}/training/strategies`),
        fetchJSON(`${API}/data/parquets`),
      ]);
      const stratOpts = (strategies.strategies || []).map(s =>
        `<option value="${s.file_path}">${s.file_name} (${s.symbol || '-'})</option>`
      ).join('') || '<option value="">无策略</option>';
      const dataOpts = (parquets.files || []).map(f =>
        `<option value="${f.file_path}">${f.file_name}</option>`
      ).join('') || '<option value="">无数据</option>';

      el.innerHTML = `
        <div class="card">
          <div class="card-title flex items-center justify-between">
            <span>分析配置</span>
            <div class="flex gap-1">
              <input type="file" id="an-import-file" accept=".json" style="display:none" onchange="App.importStrategy(this, 'an-strategy')">
              <button class="btn btn-sm btn-ghost" onclick="document.getElementById('an-import-file').click()" style="padding:2px 8px;font-size:12px" title="导入外部或导出的策略 JSON 文件">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="vertical-align:-2px;margin-right:2px"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
                导入策略
              </button>
              <button class="btn btn-sm btn-ghost" onclick="App.openPortfolioModal('an-strategy')" style="padding:2px 8px;font-size:12px" title="将多个单因子策略融合成多因子组合策略">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="vertical-align:-2px;margin-right:2px"><path d="M12 20v-6M6 20V10M18 20V4"/></svg>
                构建组合
              </button>
            </div>
          </div>
          <div class="form-row">
            <div class="form-group"><label class="form-label">策略文件</label><select class="form-select" id="an-strategy">${stratOpts}</select></div>
            <div class="form-group"><label class="form-label">数据源</label><select class="form-select" id="an-source" onchange="App.toggleAnalysisSource()">
              <option value="okx">OKX 实时</option>
              <option value="parquet">本地 Parquet (MT5/TradingView)</option>
            </select></div>
          </div>
          <div id="an-okx-params" class="form-row">
            <div class="form-group"><label class="form-label">合约 ID</label><input class="form-input" id="an-inst-id" value="BTC-USDT-SWAP" placeholder="如 BTC-USDT-SWAP"></div>
<div class="form-group"><label class="form-label">K线周期</label><select class="form-select" id="an-bar">
<option value="1m">1分钟</option>
<option value="3m">3分钟</option>
<option value="5m">5分钟</option>
<option value="15m">15分钟</option>
<option value="30m">30分钟</option>
<option value="1H" selected>1小时</option>
<option value="2H">2小时</option>
<option value="4H">4小时</option>
<option value="6H">6小时</option>
<option value="12H">12小时</option>
<option value="1D">日线</option>
<option value="1W">周线</option>
</select></div>
            <div class="form-group"><label class="form-label">K线数量</label><input class="form-input" id="an-limit" type="number" value="300" min="50" max="1000"></div>
          </div>
          <div id="an-parquet-params" style="display:none">
            <div class="form-group"><label class="form-label">Parquet 文件</label><select class="form-select" id="an-parquet">${dataOpts}</select></div>
          </div>
          <button class="btn btn-primary" onclick="App.runAnalysis()">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/></svg>
            分析信号
          </button>
        </div>
        <div id="an-result-area"></div>
      `;
    } catch (e) {
      el.innerHTML = `<div class="alert alert-danger">加载失败: ${e.message}</div>`;
    }
  },

  toggleAnalysisSource() {
    const src = document.getElementById('an-source').value;
    document.getElementById('an-okx-params').style.display = src === 'okx' ? '' : 'none';
    document.getElementById('an-parquet-params').style.display = src === 'parquet' ? '' : 'none';
  },

  async runAnalysis() {
    const strategy = document.getElementById('an-strategy').value;
    if (!strategy) { toast('请选择策略', 'error'); return; }
    const source = document.getElementById('an-source').value;
    const area = document.getElementById('an-result-area');
    area.innerHTML = '<div class="card"><div class="loading-overlay"><span class="spinner"></span> 计算信号中...</div></div>';
    try {
      let r;
      if (source === 'okx') {
        r = await fetchJSON(`${API}/analysis/okx`, {
          method: 'POST',
          body: JSON.stringify({
            strategy_path: strategy,
            inst_id: document.getElementById('an-inst-id').value,
            bar: document.getElementById('an-bar').value,
            limit: parseInt(document.getElementById('an-limit').value),
          }),
        });
      } else {
        r = await fetchJSON(`${API}/analysis/parquet`, {
          method: 'POST',
          body: JSON.stringify({
            strategy_path: strategy,
            data_file: document.getElementById('an-parquet').value,
          }),
        });
      }
      this.renderAnalysisResult(r);
      toast('分析完成', 'success');
    } catch (e) {
      area.innerHTML = `<div class="card"><div class="alert alert-danger">分析失败: ${e.message}</div></div>`;
      toast(`分析失败: ${e.message}`, 'error');
    }
  },

  renderAnalysisResult(r) {
    const l = r.latest;
    const sigClass = l.position > 0.05 ? 'signal-long' : l.position < -0.05 ? 'signal-short' : 'signal-flat';
    const area = document.getElementById('an-result-area');
    area.innerHTML = `
      <div class="grid-2">
        <div class="card">
          <div class="card-title">最新信号</div>
          <div style="text-align:center;padding:12px">
            <div class="signal-indicator ${sigClass}" style="font-size:18px;padding:8px 24px">${l.action}</div>
            <div class="grid-3 mt-6">
              <div><div class="stat-label">最新价格</div><div class="text-mono" style="font-size:16px;font-weight:600">${l.price}</div></div>
              <div><div class="stat-label">因子值</div><div class="text-mono" style="font-size:16px;font-weight:600;color:var(--cyan)">${l.factor}</div></div>
              <div><div class="stat-label">仓位信号</div><div class="text-mono" style="font-size:16px;font-weight:600">${l.position}</div></div>
            </div>
          </div>
        </div>
        <div class="card">
          <div class="card-title">信号统计</div>
          <div class="grid-3">
            <div><div class="stat-label">做多K线</div><div class="text-mono text-success" style="font-size:18px;font-weight:600">${r.signal_stats.long_bars} (${r.signal_stats.long_pct}%)</div></div>
            <div><div class="stat-label">做空K线</div><div class="text-mono text-danger" style="font-size:18px;font-weight:600">${r.signal_stats.short_bars} (${r.signal_stats.short_pct}%)</div></div>
            <div><div class="stat-label">空仓K线</div><div class="text-mono text-muted" style="font-size:18px;font-weight:600">${r.signal_stats.flat_bars}</div></div>
          </div>
          <div class="mt-4"><div class="stat-label">估算收益（无成本）</div><div class="text-mono ${r.estimated_return >= 0 ? 'text-success' : 'text-danger'}" style="font-size:20px;font-weight:700">${fmtPct(r.estimated_return)}</div></div>
        </div>
      </div>
      <div class="card mt-4">
        <div class="card-title">价格与信号</div>
        <div class="chart-container lg"><canvas id="an-chart"></canvas></div>
      </div>
      <div class="card mt-4">
        <div class="card-title">仓位信号</div>
        <div class="chart-container"><canvas id="an-position-chart"></canvas></div>
      </div>
    `;
    // 价格 + 因子图
    const s = r.series;
    const ctx1 = document.getElementById('an-chart').getContext('2d');
    setChart('an', new Chart(ctx1, {
      type: 'line',
      data: {
        labels: s.time.map(t => fmtTime(t)),
        datasets: [
          { label: '价格', data: s.close, borderColor: '#3b82f6', yAxisID: 'y', tension: 0.3, pointRadius: 0, borderWidth: 2 },
          { label: '因子值', data: s.factor, borderColor: '#06b6d4', yAxisID: 'y1', tension: 0.3, pointRadius: 0, borderWidth: 1.5, borderDash: [4,4] },
        ],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        scales: {
          y: { position: 'left', grid: { color: '#2a3447' }, ticks: { color: '#94a3b8' } },
          y1: { position: 'right', grid: { drawOnChartArea: false }, ticks: { color: '#06b6d4' } },
          x: { grid: { color: '#2a3447' }, ticks: { color: '#94a3b8', maxTicksLimit: 8 } },
        },
        plugins: { legend: { labels: { color: '#e2e8f0' } } },
      },
    }));
    // 仓位图
    const ctx2 = document.getElementById('an-position-chart').getContext('2d');
    setChart('an-pos', new Chart(ctx2, {
      type: 'bar',
      data: {
        labels: s.time.map(t => fmtTime(t)),
        datasets: [{
          label: '仓位', data: s.position,
          backgroundColor: s.position.map(p => p > 0 ? 'rgba(16,185,129,0.4)' : p < 0 ? 'rgba(239,68,68,0.4)' : 'rgba(100,116,139,0.2)'),
          borderColor: s.position.map(p => p > 0 ? '#10b981' : p < 0 ? '#ef4444' : '#64748b'),
          borderWidth: 1,
        }],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        scales: {
          y: { min: -1.1, max: 1.1, grid: { color: '#2a3447' }, ticks: { color: '#94a3b8' } },
          x: { grid: { color: '#2a3447' }, ticks: { color: '#94a3b8', maxTicksLimit: 8 } },
        },
        plugins: { legend: { display: false } },
      },
    }));
  },

  // ════════ Trading ════════
  async render_trading() {
    const el = document.getElementById('trading-content');
    try {
      const [config, strategies] = await Promise.all([
        fetchJSON(`${API}/trading/config`),
        fetchJSON(`${API}/training/strategies`),
      ]);
      const stratOpts = (strategies.strategies || []).map(s =>
        `<option value="${s.file_path}">${s.file_name} (${s.symbol || '-'})</option>`
      ).join('') || '<option value="">无策略</option>';
      const modeClass = config.is_live ? 'live' : 'paper';

      el.innerHTML = `
        <div class="alert alert-${config.is_live ? 'danger' : 'warning'}">
          <strong>当前模式: ${config.mode.toUpperCase()}</strong>
          ${config.is_live ? '— 实盘模式已启用，将发送真实订单！' : '— 模拟模式，不会发送真实订单。'}
          <span class="text-muted text-sm" style="margin-left:12px">风控: 杠杆≤${config.max_leverage} | 单日亏损≤${(config.max_daily_loss_pct * 100).toFixed(0)}% | 单品种仓位≤${(config.max_position_pct * 100).toFixed(0)}%</span>
        </div>

        <div class="grid-2">
          <div class="card">
            <div class="card-title flex items-center justify-between">
              <span>交易配置</span>
              <div class="flex gap-1">
                <input type="file" id="tr-import-file" accept=".json" style="display:none" onchange="App.importStrategy(this, 'tr-strategy')">
                <button class="btn btn-sm btn-ghost" onclick="document.getElementById('tr-import-file').click()" style="padding:2px 8px;font-size:12px" title="导入外部或导出的策略 JSON 文件">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="vertical-align:-2px;margin-right:2px"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
                  导入策略
                </button>
                <button class="btn btn-sm btn-ghost" onclick="App.openPortfolioModal('tr-strategy')" style="padding:2px 8px;font-size:12px" title="将多个单因子策略融合成多因子组合策略">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="vertical-align:-2px;margin-right:2px"><path d="M12 20v-6M6 20V10M18 20V4"/></svg>
                  构建组合
                </button>
              </div>
            </div>
            <div class="form-group"><label class="form-label">策略文件</label><select class="form-select" id="tr-strategy">${stratOpts}</select></div>
            <div class="form-group"><label class="form-label">合约 ID</label><input class="form-input" id="tr-inst-id" value="BTC-USDT-SWAP"></div>
            <div class="form-row">
              <div class="form-group"><label class="form-label">本金 (USDT) <button class="btn btn-sm" onclick="App.useAccountBalance()" style="margin-left:8px;padding:2px 8px">使用余额</button></label><input class="form-input" id="tr-capital" type="number" value="${config.default_capital}" step="10"></div>
              <div class="form-group"><label class="form-label">杠杆 (max ${config.max_leverage})</label><input class="form-input" id="tr-leverage" type="number" value="${config.default_leverage}" min="1" max="${config.max_leverage}"></div>
            </div>
            <div class="form-row">
              <div class="form-group"><label class="form-label">K线周期</label><select class="form-select" id="tr-bar">
                <option value="1m">1分钟</option>
                <option value="3m">3分钟</option>
                <option value="5m">5分钟</option>
                <option value="15m">15分钟</option>
                <option value="30m">30分钟</option>
                <option value="1H" selected>1小时</option>
                <option value="2H">2小时</option>
                <option value="4H">4小时</option>
                <option value="6H">6小时</option>
                <option value="12H">12小时</option>
                <option value="1D">日线</option>
                <option value="1W">周线</option>
              </select></div>
              <div class="form-group"><label class="form-label">最大仓位占比</label><input class="form-input" id="tr-max-pos" type="number" value="0.30" step="0.05" min="0.05" max="1.0"></div>
            </div>
            <div class="flex gap-2 mt-4">
              <button class="btn btn-success flex-1" onclick="App.executeTrade()">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 10l7-7m0 0l7 7m-7-7v18"/></svg>
                执行信号
              </button>
              <button class="btn btn-danger" onclick="App.closePosition()">平仓</button>
            </div>
          </div>

          <div class="card">
            <div class="card-title flex items-center justify-between">
              <span>自动执行</span>
              <span class="badge badge-muted" id="at-status-badge">未运行</span>
            </div>
            <div class="form-group"><label class="form-label">执行间隔（秒）</label><input class="form-input" id="at-interval" type="number" value="3600" min="60" step="60" placeholder="3600=1小时, 900=15分钟, 60=1分钟"></div>
            <div class="text-xs text-muted mb-3">按固定间隔自动执行信号（使用上方配置的策略/品种/本金/杠杆/周期）。建议与 K 线周期一致：1H=3600秒, 15m=900秒, 5m=300秒, 1m=60秒</div>
            <div id="at-controls">
              <button class="btn btn-success w-full" onclick="App.startAutoTrade()" id="at-start-btn">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 3l14 9-14 9V3z"/></svg>
                启动自动执行
              </button>
            </div>
            <div id="at-detail" style="display:none">
              <!-- 当前信号状态 -->
              <div id="at-signal-box" style="text-align:center;padding:12px;border-radius:8px;background:var(--bg-primary);margin-top:8px;border:1px solid var(--border)">
                <div class="stat-label">最新信号</div>
                <div id="at-signal-indicator" class="signal-indicator signal-flat" style="font-size:20px;padding:8px 24px;margin-top:6px;display:inline-block">等待执行</div>
                <div class="grid-3 mt-3">
                  <div><div class="stat-label">信号值</div><div class="text-mono" style="font-weight:600;color:var(--cyan)" id="at-signal-val">-</div></div>
                  <div><div class="stat-label">最新价</div><div class="text-mono" style="font-weight:600" id="at-signal-price">-</div></div>
                  <div><div class="stat-label">目标张数</div><div class="text-mono" style="font-weight:600" id="at-signal-sz">-</div></div>
                </div>
              </div>

              <!-- 统计计数 -->
              <div class="grid-3 mt-3">
                <div class="text-center" style="background:var(--bg-primary);padding:10px;border-radius:6px">
                  <div class="stat-label">已执行</div>
                  <div class="text-mono" style="font-size:18px;font-weight:600;color:var(--cyan)" id="at-total">0</div>
                </div>
                <div class="text-center" style="background:var(--bg-primary);padding:10px;border-radius:6px">
                  <div class="stat-label">已下单</div>
                  <div class="text-mono" style="font-size:18px;font-weight:600;color:var(--green)" id="at-orders">0</div>
                </div>
                <div class="text-center" style="background:var(--bg-primary);padding:10px;border-radius:6px">
                  <div class="stat-label">已跳过</div>
                  <div class="text-mono" style="font-size:18px;font-weight:600;color:var(--orange)" id="at-skips">0</div>
                </div>
              </div>

              <!-- 信号方向分布 -->
              <div style="margin-top:10px">
                <div class="stat-label mb-2">信号方向分布</div>
                <div id="at-stats-bar" style="display:flex;gap:2px;height:22px;border-radius:4px;overflow:hidden;background:var(--bg-primary)">
                  <div id="at-bar-long" style="background:var(--green);width:0%;transition:width .3s" title="做多"></div>
                  <div id="at-bar-short" style="background:var(--danger);width:0%;transition:width .3s" title="做空"></div>
                  <div id="at-bar-flat" style="background:var(--text-muted);width:0%;transition:width .3s" title="空仓"></div>
                  <div id="at-bar-skip" style="background:var(--orange);width:0%;transition:width .3s" title="跳过"></div>
                  <div id="at-bar-error" style="background:#666;width:0%;transition:width .3s" title="错误"></div>
                </div>
                <div class="flex gap-3 mt-2 text-xs text-muted" id="at-stats-legend">
                  <span><span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:var(--green);margin-right:3px"></span>做多 <b id="at-cnt-long">0</b></span>
                  <span><span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:var(--danger);margin-right:3px"></span>做空 <b id="at-cnt-short">0</b></span>
                  <span><span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:var(--text-muted);margin-right:3px"></span>空仓 <b id="at-cnt-flat">0</b></span>
                  <span><span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:var(--orange);margin-right:3px"></span>跳过 <b id="at-cnt-skip">0</b></span>
                  <span><span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:#666;margin-right:3px"></span>错误 <b id="at-cnt-error">0</b></span>
                </div>
              </div>

              <!-- 最近信号历史 -->
              <div style="margin-top:10px">
                <div class="stat-label mb-2">最近信号历史</div>
                <div id="at-history" style="max-height:200px;overflow-y:auto;font-size:12px"><div class="text-muted text-center" style="padding:10px">暂无记录</div></div>
              </div>

              <!-- 上次执行信息 -->
              <div class="text-xs text-muted mt-2" id="at-last-info">-</div>
              <button class="btn btn-danger w-full mt-3" onclick="App.stopAutoTrade()">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="6" y="6" width="12" height="12"/></svg>
                停止自动执行
              </button>
            </div>
          </div>

          <div class="card">
            <div class="card-title">交易状态</div>
            <div id="tr-status-area"><div class="text-muted text-center" style="padding:20px">点击「执行信号」查看结果</div></div>
          </div>
        </div>

        <div class="card mt-6">
          <div class="card-title flex items-center justify-between">
            <span>运行状态</span>
            <div class="flex gap-2 items-center">
              <span class="text-xs text-muted" id="rt-update-time">-</span>
              <button class="btn btn-sm" onclick="App.refreshRuntimeStatus()">刷新</button>
              <label class="text-xs flex items-center gap-1" style="cursor:pointer">
                <input type="checkbox" id="rt-auto-refresh" onchange="App.toggleRuntimeAutoRefresh()" checked> 自动刷新
              </label>
            </div>
          </div>
          <div id="rt-content"><div class="text-muted text-center" style="padding:20px">加载中...</div></div>
        </div>

        <div class="card mt-6">
          <div class="card-title">审计日志</div>
          <div id="tr-audit-area"><div class="text-muted text-center" style="padding:20px">暂无审计记录</div></div>
        </div>
      `;
      this.refreshAuditLog();
      this.refreshRuntimeStatus();
      this.refreshAutoTradeStatus();
      // 启动运行状态自动刷新（复用 pollTimer，离开页面时 navigate 会清除）
      if (this.pollTimer) clearInterval(this.pollTimer);
      this.pollTimer = setInterval(() => {
        this.refreshRuntimeStatus();
        this.refreshAutoTradeStatus();
      }, 5000);
    } catch (e) {
      el.innerHTML = `<div class="alert alert-danger">加载失败: ${e.message}</div>`;
    }
  },

  async importStrategy(inputEl, targetSelectId = null) {
    const file = inputEl.files && inputEl.files[0];
    if (!file) return;
    try {
      const formData = new FormData();
      formData.append('file', file);

      const resp = await fetch(`${API}/training/strategies/import`, {
        method: 'POST',
        body: formData,
      });

      if (!resp.ok) {
        let msg = `HTTP ${resp.status}`;
        try { const e = await resp.json(); msg = e.detail || e.message || msg; } catch {}
        throw new Error(msg);
      }

      const res = await resp.json();
      toast(res.msg || '策略导入成功', 'success');

      // 重新拉取策略列表
      const data = await fetchJSON(`${API}/training/strategies`);
      const newStrats = data.strategies || [];

      // 刷新所有包含策略选择的下拉框
      const targetIds = [targetSelectId, 'tr-strategy', 'bt-strategy', 'an-strategy'].filter(Boolean);
      const uniqueIds = [...new Set(targetIds)];

      uniqueIds.forEach(id => {
        const select = document.getElementById(id);
        if (select) {
          const currentVal = select.value;
          select.innerHTML = newStrats.map(s =>
            `<option value="${s.file_path}">${s.file_name} (${s.symbol || '-'})</option>`
          ).join('') || '<option value="">无策略</option>';

          if (res.strategy && res.strategy.file_path) {
            select.value = res.strategy.file_path;
          } else if (currentVal) {
            select.value = currentVal;
          }
        }
      });
    } catch (e) {
      toast(`导入策略失败: ${e.message}`, 'error');
    } finally {
      inputEl.value = '';
    }
  },

  async openPortfolioModal(targetSelectId = null) {
    this._targetPortfolioSelectId = targetSelectId;
    const modal = document.getElementById('portfolio-modal');
    const listEl = document.getElementById('pf-strategy-list');
    modal.style.display = 'flex';
    listEl.innerHTML = '<div class="loading-overlay"><span class="spinner"></span> 加载可用策略中...</div>';
    try {
      const res = await fetchJSON(`${API}/training/strategies`);
      const strats = (res.strategies || []).filter(s => !s.is_portfolio);
      if (strats.length < 2) {
        listEl.innerHTML = '<div class="alert alert-warning" style="margin:0">需至少有 2 个单因子策略才能构建多因子组合。<br>请先训练模型或导入更多策略文件。</div>';
        return;
      }
      listEl.innerHTML = strats.map(s => `
        <label class="flex items-center gap-2" style="padding:6px 0;border-bottom:1px solid var(--border);cursor:pointer">
          <input type="checkbox" name="pf-sub-strat" value="${s.file_path}">
          <div style="flex:1">
            <div class="text-sm font-mono">${s.file_name} (${s.symbol || '-'})</div>
            <div class="text-xs text-muted">评分: ${fmtNum(s.best_score, 3)} | ${s.formula_decoded || '-'}</div>
          </div>
        </label>
      `).join('');
    } catch (e) {
      listEl.innerHTML = `<div class="text-danger">加载策略列表失败: ${e.message}</div>`;
    }
  },

  closePortfolioModal() {
    document.getElementById('portfolio-modal').style.display = 'none';
  },

  async submitPortfolioStrategy() {
    const name = document.getElementById('pf-name').value.trim();
    if (!name) { toast('请输入组合策略名称', 'error'); return; }
    const checked = Array.from(document.querySelectorAll('input[name="pf-sub-strat"]:checked')).map(el => el.value);
    if (checked.length < 2) { toast('请至少勾选 2 个子策略进行融合', 'error'); return; }
    const weightMethod = document.getElementById('pf-weight-method').value;

    try {
      const res = await fetchJSON(`${API}/portfolio/create`, {
        method: 'POST',
        body: JSON.stringify({
          portfolio_name: name,
          strategy_paths: checked,
          weight_method: weightMethod,
        }),
      });

      toast(res.msg || '组合策略创建成功', 'success');
      this.closePortfolioModal();

      // 刷新界面上的策略选择框并自动选中新构建的组合策略
      const data = await fetchJSON(`${API}/training/strategies`);
      const newStrats = data.strategies || [];

      ['tr-strategy', 'bt-strategy', 'an-strategy'].forEach(id => {
        const select = document.getElementById(id);
        if (select) {
          select.innerHTML = newStrats.map(s =>
            `<option value="${s.file_path}">${s.is_portfolio ? '📈 [组合] ' : ''}${s.file_name} (${s.symbol || '-'})</option>`
          ).join('') || '<option value="">无策略</option>';

          if (res.portfolio && res.portfolio.file_path) {
            select.value = res.portfolio.file_path;
          }
        }
      });
    } catch (e) {
      toast(`创建组合策略失败: ${e.message}`, 'error');
    }
  },

  async executeTrade() {
    const strategy = document.getElementById('tr-strategy').value;
    if (!strategy) { toast('请选择策略', 'error'); return; }
    const instId = document.getElementById('tr-inst-id').value;
    if (!instId) { toast('请输入合约 ID', 'error'); return; }
    const area = document.getElementById('tr-status-area');
    area.innerHTML = '<div class="loading-overlay"><span class="spinner"></span> 执行交易信号中...</div>';
    try {
      const r = await fetchJSON(`${API}/trading/execute`, {
        method: 'POST',
        body: JSON.stringify({
          strategy_path: strategy,
          inst_id: instId,
          capital: parseFloat(document.getElementById('tr-capital').value),
          leverage: parseInt(document.getElementById('tr-leverage').value),
          bar: document.getElementById('tr-bar').value,
          max_position_pct: parseFloat(document.getElementById('tr-max-pos').value),
        }),
      });
      this.renderTradeResult(r);
      this.refreshAuditLog();
      this.refreshRuntimeStatus();
      toast(r.risk_passed ? '交易信号已执行' : '信号未通过风控', r.risk_passed ? 'success' : 'warning');
    } catch (e) {
      area.innerHTML = `<div class="alert alert-danger">执行失败: ${e.message}</div>`;
      toast(`执行失败: ${e.message}`, 'error');
    }
  },

  renderTradeResult(r) {
    const sigClass = r.signal > 0.05 ? 'signal-long' : r.signal < -0.05 ? 'signal-short' : 'signal-flat';
    const riskHtml = (r.risk_checks || []).map(rc =>
      `<div class="flex items-center gap-2"><span class="badge ${rc.passed ? 'badge-success' : 'badge-danger'}">${rc.passed ? '✓' : '✗'}</span><span class="text-sm">${rc.msg}</span></div>`
    ).join('');
    const order = r.order || {};
    const sd = r.size_detail || {};
    const hedgeHtml = (r.hedge_actions || []).length > 0
      ? `<div class="mt-4"><div class="stat-label mb-2">对冲平仓</div>${r.hedge_actions.map(h => `<div class="text-sm flex items-center gap-2"><span class="badge badge-warning">${h.action}</span><span>${h.pos_side} ${h.pos_sz} 张</span>${h.error ? `<span class="text-danger">${h.error}</span>` : ''}</div>`).join('')}</div>`
      : '';
    const levHtml = r.leverage_result
      ? `<div class="text-xs text-muted mt-1">杠杆设置: ${r.leverage_result.simulated ? 'PAPER 模拟' : (r.leverage_result.error ? '失败: ' + r.leverage_result.error : '已设置')}</div>`
      : '';
    const area = document.getElementById('tr-status-area');
    area.innerHTML = `
      <div style="text-align:center;padding:8px">
        <div class="signal-indicator ${sigClass}" style="font-size:16px;padding:6px 20px">${r.action}</div>
      </div>
      <div class="grid-3 mt-4">
        <div><div class="stat-label">最新价格</div><div class="text-mono" style="font-weight:600">${r.last_price}</div></div>
        <div><div class="stat-label">信号值</div><div class="text-mono" style="font-weight:600;color:var(--cyan)">${r.signal}</div></div>
        <div><div class="stat-label">目标张数</div><div class="text-mono" style="font-weight:600">${r.target_sz}</div></div>
      </div>
${sd.ct_val ? `<div class="text-xs text-muted mt-2" style="background:var(--bg-primary);padding:8px;border-radius:6px">${sd.note} | 目标价值: ${sd.target_value} USDT | 原始张数: ${sd.raw_sz}${sd.below_min ? ' | <span class="text-danger">低于最小下单量</span>' : ''}</div>` : ''}
${r.signal_diag ? `<div class="mt-3" style="background:var(--bg-primary);padding:10px;border-radius:6px;border:1px solid var(--border)">
<div class="stat-label mb-2">因子诊断 ${r.signal_diag.in_neutral_band ? '<span class="badge badge-warning">中性区间内（信号被过滤）</span>' : '<span class="badge badge-success">有效信号</span>'}</div>
<div class="grid-4 text-sm">
<div><span class="text-muted">因子值</span><br><span class="text-mono" style="font-weight:600">${r.signal_diag.factor}</span></div>
<div><span class="text-muted">tanh(因子)</span><br><span class="text-mono" style="font-weight:600;color:var(--cyan)">${r.signal_diag.tanh}</span></div>
<div><span class="text-muted">|tanh|</span><br><span class="text-mono" style="font-weight:600;color:${r.signal_diag.in_neutral_band ? 'var(--orange)' : 'var(--green)'}">${r.signal_diag.abs_tanh}</span></div>
<div><span class="text-muted">最终信号</span><br><span class="text-mono" style="font-weight:600;color:var(--cyan)">${r.signal_diag.signal_after_band}</span></div>
</div>
<div class="text-xs text-muted mt-2">中性区间: [${r.signal_diag.lower_band}, ${r.signal_diag.upper_band}] | K线数: ${r.signal_diag.bars_used} | 特征维度: ${r.signal_diag.feat_shape ? r.signal_diag.feat_shape.join('×') : '-'}</div>
${r.signal_diag.in_neutral_band ? `<div class="text-xs text-warning mt-1">⚠ 因子 |tanh|=${r.signal_diag.abs_tanh} 低于下限 ${r.signal_diag.lower_band}，被视为噪声，信号归零。可能原因：策略对该品种/周期不敏感，或市场处于震荡期。</div>` : ''}
</div>` : ''}
<div class="mt-4">
        <div class="stat-label mb-2">风控检查</div>
        ${riskHtml}
      </div>
      ${hedgeHtml}
      ${levHtml}
      <div class="mt-4">
        <div class="stat-label mb-2">订单结果</div>
        <div class="text-mono text-sm" style="background:var(--bg-primary);padding:10px;border-radius:6px;border:1px solid var(--border)">
          ${order.simulated ? '<span class="badge badge-warning">PAPER</span> ' : (order.live ? '<span class="badge badge-danger">LIVE</span> ' : '')}
          ${order.skipped ? '信号跳过: ' + (order.reason || '') : JSON.stringify(order, null, 2)}
        </div>
      </div>
      <div class="mt-4 flex gap-2">
        <span class="badge badge-info">模式: ${r.mode.toUpperCase()}</span>
        <span class="badge ${r.is_live ? 'badge-danger' : 'badge-muted'}">${r.is_live ? '实盘' : '模拟'}</span>
      </div>
    `;
  },

  async refreshRuntimeStatus() {
    const area = document.getElementById('rt-content');
    if (!area) return;
    try {
      const r = await fetchJSON(`${API}/trading/runtime`);
      if (r.ws_status) {
        const badge = document.getElementById('ws-badge');
        if (badge) {
          if (r.ws_status.connected) {
            badge.style.background = '#10b981';
            badge.style.color = '#fff';
            badge.textContent = 'WS 实时推送';
          } else {
            badge.style.background = '#f59e0b';
            badge.style.color = '#fff';
            badge.textContent = 'WS 备用(REST)';
          }
        }
      }
      const acct = r.account;
      const acctHtml = acct ? `
        <div class="grid-4">
          <div><div class="stat-label">总权益</div><div class="text-mono" style="font-weight:600;font-size:16px;color:var(--cyan)">${acct.total_eq.toFixed(2)} <span class="text-xs text-muted">${acct.currency}</span></div></div>
          <div><div class="stat-label">可用余额</div><div class="text-mono" style="font-weight:600">${acct.avail_bal.toFixed(2)}</div></div>
          <div><div class="stat-label">未实现盈亏</div><div class="text-mono" style="font-weight:600;color:${acct.upl >= 0 ? 'var(--green)' : 'var(--red)'}">${acct.upl >= 0 ? '+' : ''}${acct.upl.toFixed(2)} (${(acct.upl_ratio * 100).toFixed(2)}%)</div></div>
          <div><div class="stat-label">保证金占用</div><div class="text-mono" style="font-weight:600">${acct.margin.toFixed(2)} <span class="text-xs text-muted">(${(acct.margin_ratio * 100).toFixed(2)}%)</span></div></div>
        </div>
      ` : `<div class="alert alert-warning text-sm">${r.account_error ? '获取账户失败: ' + r.account_error : 'paper 模式无凭证，无法获取账户信息'}</div>`;

      const posList = r.positions || [];
      const posHtml = posList.length > 0 ? `
        <div class="table-wrapper mt-4">
          <table>
            <thead><tr><th>合约</th><th>方向</th><th>数量</th><th>均价</th><th>最新价</th><th>未实现盈亏</th><th>盈亏率</th><th>杠杆</th><th>强平价</th></tr></thead>
            <tbody>
              ${posList.map(p => `
                <tr>
                  <td class="text-mono text-sm">${p.inst_id}</td>
                  <td><span class="badge ${p.pos_side === 'long' ? 'badge-success' : p.pos_side === 'short' ? 'badge-danger' : 'badge-muted'}">${p.pos_side}</span></td>
                  <td class="text-mono text-sm">${p.pos}</td>
                  <td class="text-mono text-sm">${p.avg_px}</td>
                  <td class="text-mono text-sm">${p.last}</td>
                  <td class="text-mono text-sm" style="color:${p.upl >= 0 ? 'var(--green)' : 'var(--red)'}">${p.upl >= 0 ? '+' : ''}${p.upl.toFixed(2)}</td>
                  <td class="text-mono text-sm" style="color:${p.upl_ratio >= 0 ? 'var(--green)' : 'var(--red)'}">${(p.upl_ratio * 100).toFixed(2)}%</td>
                  <td class="text-mono text-sm">${p.lever}x</td>
                  <td class="text-mono text-sm text-muted">${p.liq_px || '-'}</td>
                </tr>`).join('')}
            </tbody>
          </table>
        </div>
      ` : `<div class="text-muted text-center" style="padding:12px">${r.positions_error ? '获取持仓失败: ' + r.positions_error : '当前无持仓'}</div>`;

      const dr = r.daily_risk || {};
      const drInfo = (dr.info || {});
      const drHtml = `
        <div class="flex items-center gap-4 mt-4" style="background:var(--bg-primary);padding:10px;border-radius:6px">
          <span class="badge ${dr.passed ? 'badge-success' : 'badge-danger'}">${dr.passed ? '风控正常' : '触发风控'}</span>
          <span class="text-sm">${dr.msg || '-'}</span>
          ${drInfo.initial_eq ? `<span class="text-xs text-muted">| 初始权益 ${drInfo.initial_eq.toFixed(2)} → 当前 ${drInfo.current_eq ? drInfo.current_eq.toFixed(2) : '-'} | 上限 ${(drInfo.max_daily_loss_pct * 100).toFixed(0)}%</span>` : ''}
        </div>
      `;

      const stats = r.audit_stats || {};
      const statsHtml = stats ? `
        <div class="grid-3 mt-4">
          <div class="text-center" style="background:var(--bg-primary);padding:10px;border-radius:6px">
            <div class="stat-label">今日执行</div>
            <div class="text-mono" style="font-size:20px;font-weight:600;color:var(--green)">${stats.today_executions || 0}</div>
          </div>
          <div class="text-center" style="background:var(--bg-primary);padding:10px;border-radius:6px">
            <div class="stat-label">今日跳过</div>
            <div class="text-mono" style="font-size:20px;font-weight:600;color:var(--orange)">${stats.today_skips || 0}</div>
          </div>
          <div class="text-center" style="background:var(--bg-primary);padding:10px;border-radius:6px">
            <div class="stat-label">今日平仓</div>
            <div class="text-mono" style="font-size:20px;font-weight:600;color:var(--cyan)">${stats.today_closes || 0}</div>
          </div>
        </div>
      ` : '';

      area.innerHTML = acctHtml + drHtml + posHtml + statsHtml;
      const timeEl = document.getElementById('rt-update-time');
      if (timeEl) timeEl.textContent = '更新于 ' + new Date().toLocaleTimeString('zh-CN');
    } catch (e) {
      area.innerHTML = `<div class="alert alert-danger text-sm">刷新失败: ${e.message}</div>`;
    }
  },

  toggleRuntimeAutoRefresh() {
    const cb = document.getElementById('rt-auto-refresh');
    if (!cb) return;
    if (cb.checked) {
      // 启动自动刷新（复用 pollTimer，离开页面时 navigate 会清除）
      if (this.pollTimer) clearInterval(this.pollTimer);
      this.pollTimer = setInterval(() => this.refreshRuntimeStatus(), 5000);
      toast('运行状态自动刷新已开启（5秒）', 'info');
    } else {
      if (this.pollTimer) { clearInterval(this.pollTimer); this.pollTimer = null; }
      toast('运行状态自动刷新已关闭', 'info');
    }
  },

  async useAccountBalance() {
    try {
      const r = await fetchJSON(`${API}/trading/runtime`);
      if (r.account && r.account.avail_bal > 0) {
        const inp = document.getElementById('tr-capital');
        inp.value = r.account.avail_bal.toFixed(2);
        toast(`已填充可用余额: ${r.account.avail_bal.toFixed(2)} USDT`, 'success');
      } else {
        toast(r.account_error ? '获取余额失败: ' + r.account_error : '可用余额为 0', 'error');
      }
    } catch (e) {
      toast(`获取余额失败: ${e.message}`, 'error');
    }
  },

  async startAutoTrade() {
    const strategy = document.getElementById('tr-strategy').value;
    if (!strategy) { toast('请选择策略', 'error'); return; }
    const instId = document.getElementById('tr-inst-id').value;
    if (!instId) { toast('请输入合约 ID', 'error'); return; }
    const interval = parseInt(document.getElementById('at-interval').value) || 3600;
    if (interval < 60) { toast('执行间隔不能小于 60 秒', 'error'); return; }
    try {
      const r = await fetchJSON(`${API}/trading/auto/start`, {
        method: 'POST',
        body: JSON.stringify({
          strategy_path: strategy,
          inst_id: instId,
          capital: parseFloat(document.getElementById('tr-capital').value),
          leverage: parseInt(document.getElementById('tr-leverage').value),
          bar: document.getElementById('tr-bar').value,
          max_position_pct: parseFloat(document.getElementById('tr-max-pos').value),
          interval_seconds: interval,
        }),
      });
      if (r.ok) {
        toast(r.msg, 'success');
        this.refreshAutoTradeStatus();
      } else {
        toast(r.msg, 'warning');
      }
    } catch (e) {
      toast(`启动失败: ${e.message}`, 'error');
    }
  },

  async stopAutoTrade() {
    if (!confirm('确定要停止自动执行吗？\n\n停止后将不再自动下单，已有持仓不会自动平仓。')) return;
    try {
      const r = await fetchJSON(`${API}/trading/auto/stop`, { method: 'POST' });
      toast(r.msg, r.ok ? 'success' : 'warning');
      this.refreshAutoTradeStatus();
    } catch (e) {
      toast(`停止失败: ${e.message}`, 'error');
    }
  },

  async refreshAutoTradeStatus() {
    try {
      const s = await fetchJSON(`${API}/trading/auto/status`);
      const badge = document.getElementById('at-status-badge');
      const controls = document.getElementById('at-controls');
      const detail = document.getElementById('at-detail');
      if (!badge) return;  // 不在交易页面

      if (s.running) {
        badge.textContent = '运行中';
        badge.className = 'badge badge-success';
        if (controls) controls.style.display = 'none';
        if (detail) detail.style.display = 'block';

        // 统计计数
        const elTotal = document.getElementById('at-total');
        const elOrders = document.getElementById('at-orders');
        const elSkips = document.getElementById('at-skips');
        if (elTotal) elTotal.textContent = s.total_executions || 0;
        if (elOrders) elOrders.textContent = s.total_orders || 0;
        if (elSkips) elSkips.textContent = s.total_skips || 0;

        // 当前信号状态
        const lr = s.last_result || {};
        const sigVal = lr.signal !== undefined ? lr.signal : null;
        const sigInd = document.getElementById('at-signal-indicator');
        const sigValEl = document.getElementById('at-signal-val');
        const sigPriceEl = document.getElementById('at-signal-price');
        const sigSzEl = document.getElementById('at-signal-sz');
        if (sigInd) {
          let cls = 'signal-flat', txt = '空仓';
          if (lr.skipped) { cls = 'signal-flat'; txt = '跳过'; }
          else if (sigVal > 0.05) { cls = 'signal-long'; txt = '做多'; }
          else if (sigVal < -0.05) { cls = 'signal-short'; txt = '做空'; }
          sigInd.className = `signal-indicator ${cls}`;
          sigInd.textContent = txt;
        }
        if (sigValEl) sigValEl.textContent = sigVal !== null ? sigVal : '-';
        if (sigPriceEl) sigPriceEl.textContent = lr.price || '-';
        if (sigSzEl) sigSzEl.textContent = lr.target_sz !== undefined && lr.target_sz !== null ? lr.target_sz : '-';

        // 信号方向分布
        const stats = s.signal_stats || {long:0, short:0, flat:0, skip:0, error:0};
        const total = (stats.long||0) + (stats.short||0) + (stats.flat||0) + (stats.skip||0) + (stats.error||0) || 1;
        const setBar = (id, cnt) => { const el = document.getElementById(id); if (el) el.style.width = `${(cnt/total*100).toFixed(1)}%`; };
        setBar('at-bar-long', stats.long||0);
        setBar('at-bar-short', stats.short||0);
        setBar('at-bar-flat', stats.flat||0);
        setBar('at-bar-skip', stats.skip||0);
        setBar('at-bar-error', stats.error||0);
        const setCnt = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
        setCnt('at-cnt-long', stats.long||0);
        setCnt('at-cnt-short', stats.short||0);
        setCnt('at-cnt-flat', stats.flat||0);
        setCnt('at-cnt-skip', stats.skip||0);
        setCnt('at-cnt-error', stats.error||0);

        // 最近信号历史
        const hist = s.signal_history || [];
        const histEl = document.getElementById('at-history');
        if (histEl) {
          if (hist.length === 0) {
            histEl.innerHTML = '<div class="text-muted text-center" style="padding:10px">暂无记录</div>';
          } else {
            histEl.innerHTML = hist.map(h => {
              const t = new Date(h.time * 1000).toLocaleTimeString('zh-CN');
              let badge = '';
              if (h.skipped) badge = '<span class="badge badge-warning" style="font-size:10px">跳过</span>';
              else if (h.signal > 0.05) badge = '<span class="badge badge-success" style="font-size:10px">多</span>';
              else if (h.signal < -0.05) badge = '<span class="badge badge-danger" style="font-size:10px">空</span>';
              else badge = '<span class="badge badge-muted" style="font-size:10px">平</span>';
              const ord = h.ordered ? '<span class="text-success">✓下单</span>' : '';
              return `<div style="display:flex;align-items:center;gap:6px;padding:4px 6px;border-bottom:1px solid var(--border)">
                <span class="text-muted" style="min-width:60px">${t}</span>
                ${badge}
                <span class="text-mono" style="min-width:50px">${h.signal}</span>
                <span class="text-muted" style="min-width:70px">@${h.price || '-'}</span>
                ${ord}
              </div>`;
            }).join('');
          }
        }

        // 上次执行信息
        const elLast = document.getElementById('at-last-info');
        const nextIn = s.next_execute_in ? `${s.next_execute_in}秒后执行` : '-';
        const lastTime = s.last_execute_time ? new Date(s.last_execute_time * 1000).toLocaleTimeString('zh-CN') : '-';
        let lastInfo = `上次: ${lastTime} | 下次: ${nextIn}`;
        if (s.last_error) lastInfo += ` | <span class="text-danger">错误: ${s.last_error}</span>`;
        if (elLast) elLast.innerHTML = lastInfo;
      } else {
        badge.textContent = '未运行';
        badge.className = 'badge badge-muted';
        if (controls) controls.style.display = 'block';
        if (detail) detail.style.display = 'none';
      }
    } catch (e) { /* silent */ }
  },

  async closePosition() {
    const instId = document.getElementById('tr-inst-id').value;
    if (!instId) { toast('请输入合约 ID', 'error'); return; }
    try {
      const r = await fetchJSON(`${API}/trading/close/${instId}`, { method: 'POST' });
      toast(r.result?.simulated ? '模拟平仓完成' : '平仓指令已发送', 'info');
      this.refreshAuditLog();
      this.refreshRuntimeStatus();
    } catch (e) {
      toast(`平仓失败: ${e.message}`, 'error');
    }
  },

  async refreshAuditLog() {
    try {
      const r = await fetchJSON(`${API}/trading/audit?n=20`);
      const area = document.getElementById('tr-audit-area');
      if (!area) return;
      const logs = r.logs || [];
      if (logs.length === 0) { area.innerHTML = '<div class="text-muted text-center" style="padding:20px">暂无审计记录</div>'; return; }
      area.innerHTML = `
        <div class="table-wrapper">
          <table>
            <thead><tr><th>时间</th><th>事件</th><th>品种</th><th>信号</th><th>模式</th></tr></thead>
            <tbody>
              ${logs.reverse().map(l => `
                <tr>
                  <td class="text-mono text-sm text-muted">${l.timestamp ? new Date(l.timestamp).toLocaleTimeString('zh-CN') : '-'}</td>
                  <td class="text-sm">${l.event || '-'}</td>
                  <td class="text-mono text-sm">${l.inst_id || '-'}</td>
                  <td class="text-mono text-sm">${l.signal !== undefined ? l.signal : '-'}</td>
                  <td><span class="badge ${l.is_live ? 'badge-danger' : 'badge-muted'}">${l.mode || '-'}</span></td>
                </tr>`).join('')}
            </tbody>
          </table>
        </div>`;
    } catch (e) { /* silent */ }
  },

  // ════════ Data ════════
  async render_data() {
    const el = document.getElementById('data-content');
    try {
      const [parquets] = await Promise.all([
        fetchJSON(`${API}/data/parquets`),
      ]);

      el.innerHTML = `
        <div class="grid-2">
          <div class="card">
            <div class="card-title">下载 K 线数据</div>
            <div class="form-group"><label class="form-label">品种类型</label>
              <select class="form-select" id="dl-inst-type" onchange="App.discoverInstruments()">
                <option value="SWAP" selected>永续合约 (SWAP)</option>
                <option value="SPOT">现货 (SPOT)</option>
                <option value="FUTURES">交割合约 (FUTURES)</option>
                <option value="OPTION">期权 (OPTION)</option>
              </select>
            </div>
            <div class="form-group"><label class="form-label">合约 ID</label><input class="form-input" id="dl-symbol" value="BTC-USDT-SWAP" placeholder="如 BTC-USDT-SWAP / XAU-USDT-SWAP / AAPL-USDT-SWAP"></div>
            <div class="form-row">
              <div class="form-group"><label class="form-label">K线周期</label><select class="form-select" id="dl-bar">
                <option value="1m">1分钟</option>
                <option value="3m">3分钟</option>
                <option value="5m">5分钟</option>
                <option value="15m">15分钟</option>
                <option value="30m">30分钟</option>
                <option value="1H" selected>1小时</option>
                <option value="2H">2小时</option>
                <option value="4H">4小时</option>
                <option value="6H">6小时</option>
                <option value="12H">12小时</option>
                <option value="1D">日线</option>
                <option value="1W">周线</option>
              </select></div>
              <div class="form-group"><label class="form-label">K线数量</label><input class="form-input" id="dl-total" type="number" value="2000" min="100" max="10000"></div>
            </div>
            <button class="btn btn-primary w-full" onclick="App.downloadData()">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"/></svg>
              下载并加入训练列表
            </button>
            <div class="disclosure">支持加密资产(BTC/ETH)、贵金属(XAU)、指数(SPX)、股票(AAPL)等 TradFi 品种。可用性取决于 OKX 区域和账户权限。</div>
          </div>

          <div class="card">
            <div class="card-title">OKX 品种发现</div>
            <div id="dl-instruments-area"><div class="text-muted text-center" style="padding:20px">点击下方按钮发现可用品种</div></div>
            <div class="mt-4">
              <button class="btn btn-ghost btn-sm" onclick="App.discoverInstruments()">发现当前类型品种</button>
            </div>
          </div>
        </div>

        <div class="card mt-6">
          <div class="card-title">本地数据文件（可一键入训练）</div>
          ${(parquets.files || []).length > 0 ? `
            <div class="table-wrapper">
              <table>
                <thead><tr><th>文件</th><th>品种</th><th>周期</th><th>K线数</th><th>大小</th><th>操作</th></tr></thead>
                <tbody>
                    ${parquets.files.map(f => `
                      <tr>
                        <td class="text-sm">${f.file_name}</td>
                        <td>${f.symbol || '-'}</td>
                        <td>${f.timeframe || '-'}</td>
                        <td class="text-mono">${f.n_bars || 0}</td>
                        <td class="text-mono text-muted">${f.file_size_mb || 0} MB</td>
                        <td>
                          <button class="btn btn-success btn-sm" onclick="App.sendToTraining('${f.file_path}','${f.symbol || ''}')">入训练</button>
                          <button class="btn btn-ghost btn-sm" onclick="App.deleteData('${f.file_name}')">删除</button>
                        </td>
                      </tr>`).join('')}
                </tbody>
              </table>
            </div>` : '<div class="text-muted text-center" style="padding:20px">暂无数据文件</div>'}
        </div>
      `;
      // 自动发现当前类型的品种
      this.discoverInstruments();
    } catch (e) {
      el.innerHTML = `<div class="alert alert-danger">加载失败: ${e.message}</div>`;
    }
  },

  async downloadData() {
    const symbol = document.getElementById('dl-symbol').value.trim();
    const bar = document.getElementById('dl-bar').value;
    const total = parseInt(document.getElementById('dl-total').value);
    if (!symbol) { toast('请输入合约 ID', 'error'); return; }
    toast(`开始下载 ${symbol} ${bar}...`, 'info');
    try {
      const r = await fetchJSON(`${API}/data/download`, {
        method: 'POST',
        body: JSON.stringify({ symbol, bar, total_bars: total }),
      });
      toast(`下载完成: ${r.n_bars} 根 K线`, 'success');
      // 下载后一键入训练列表
      this.sendToTraining(r.file_path, r.symbol, true);
      this.render_data();
    } catch (e) {
      toast(`下载失败: ${e.message}`, 'error');
    }
  },

  sendToTraining(filePath, symbol, silent = false) {
    // 保存到 localStorage，训练页面加载时读取
    const pending = JSON.parse(localStorage.getItem('pending_train_data') || '[]');
    pending.push({ file_path: filePath, symbol: symbol || '', time: Date.now() });
    localStorage.setItem('pending_train_data', JSON.stringify(pending));
    if (!silent) {
      toast(`已加入训练列表: ${symbol || filePath}`, 'success');
      this.navigate('training');
    } else {
      toast(`已下载并加入训练列表: ${symbol}`, 'success');
    }
  },

  async discoverInstruments() {
    const area = document.getElementById('dl-instruments-area');
    if (!area) return;
    const instType = document.getElementById('dl-inst-type')?.value || 'SWAP';
    area.innerHTML = '<div class="loading-overlay"><span class="spinner"></span> 获取品种列表...</div>';
    try {
      const r = await fetchJSON(`${API}/data/instruments?inst_type=${instType}`);
      const insts = (r.instruments || []).filter(i => i.state === 'live');
      area.innerHTML = `
        <div class="stat-label">${instType} 类型发现 ${r.count} 个品种（显示前 ${Math.min(insts.length, 80)} 个活跃品种）</div>
        <div class="table-wrapper mt-2" style="max-height:400px;overflow-y:auto">
          <table>
            <thead><tr><th>合约 ID</th><th>结算/计价</th><th>杠杆</th><th>操作</th></tr></thead>
            <tbody>
              ${insts.slice(0, 80).map(i => `
                <tr>
                  <td class="text-mono text-sm">${i.inst_id}</td>
                  <td class="text-sm text-muted">${i.settle_ccy || i.quote_ccy || '-'}</td>
                  <td class="text-mono text-sm">${i.lever || '-'}</td>
                  <td>
                    <button class="btn btn-ghost btn-sm" onclick="App.selectInstrument('${i.inst_id}')">选用</button>
                    <button class="btn btn-success btn-sm" onclick="App.quickDownload('${i.inst_id}')">快速下载</button>
                  </td>
                </tr>`).join('')}
            </tbody>
          </table>
        </div>`;
    } catch (e) {
      area.innerHTML = `<div class="alert alert-danger">获取失败: ${e.message}</div>`;
    }
  },

  selectInstrument(instId) {
    const inp = document.getElementById('dl-symbol');
    if (inp) inp.value = instId;
    toast(`已选用: ${instId}`, 'info');
  },

  async quickDownload(instId) {
    const bar = document.getElementById('dl-bar')?.value || '1H';
    const total = parseInt(document.getElementById('dl-total')?.value || '2000');
    toast(`快速下载 ${instId} ${bar}...`, 'info');
    try {
      const r = await fetchJSON(`${API}/data/download`, {
        method: 'POST',
        body: JSON.stringify({ symbol: instId, bar, total_bars: total }),
      });
      toast(`下载完成: ${r.n_bars} 根 K线`, 'success');
      this.sendToTraining(r.file_path, r.symbol, true);
      this.render_data();
    } catch (e) {
      toast(`下载失败: ${e.message}`, 'error');
    }
  },

  async deleteData(fileName) {
    if (!confirm(`确认删除 ${fileName}？`)) return;
    try {
      await fetchJSON(`${API}/data/parquet/${fileName}`, { method: 'DELETE' });
      toast('已删除', 'success');
      this.render_data();
    } catch (e) {
      toast(`删除失败: ${e.message}`, 'error');
    }
  },
};

// ── Init ───────────────────────────────────────────────────────────
(async function init() {
  try {
    const sys = await fetchJSON(`${API}/system`);
    const badge = document.getElementById('mode-badge');
    badge.textContent = sys.is_live ? 'LIVE' : 'PAPER';
    badge.className = `mode-badge ${sys.is_live ? 'live' : 'paper'}`;
  } catch (e) {
    console.warn('系统信息获取失败:', e);
  }
  App.navigate('dashboard');
})();
