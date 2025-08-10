const baselineForm = document.getElementById('baselineForm');
const detectForm   = document.getElementById('detectForm');
const resetBtn     = document.getElementById('resetBtn');
const toggle       = document.getElementById('monitorToggle');
const wifi         = document.getElementById('wifi');

let chart;
const ctx = document.getElementById('baselineChart').getContext('2d');

function ensureChart() {
  if (chart) return chart;
  chart = new Chart(ctx, {
    type: 'line',
    data: { labels: [], datasets: [{ label: 'Baseline Signal', data: [], borderWidth: 2, tension: .25 }] },
    options: { responsive:true, maintainAspectRatio:false, scales:{ y:{ beginAtZero:false } } }
  });
  return chart;
}

toggle.addEventListener('change', async () => {
  wifi.classList.toggle('on', toggle.checked);
  wifi.classList.toggle('off', !toggle.checked);
  await fetch('/toggle', { method:'POST' });
});

baselineForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const fd = new FormData(baselineForm);
  const res = await fetch('/baseline', { method:'POST', body:fd });
  const out = await res.json();
  const msg = document.getElementById('baselineMsg');
  if (!out.ok) { msg.textContent = out.msg || 'Error'; return; }
  msg.textContent = `Baseline saved • rows used: ${out.count}`;
  const ch = ensureChart();
  ch.data.labels = out.signal.map((_, i) => i+1);
  ch.data.datasets[0].data = out.signal;
  ch.update();
});

detectForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const fd = new FormData(detectForm);
  const res = await fetch('/detect', { method:'POST', body:fd });
  const out = await res.json();
  const div = document.getElementById('detectOut');
  if (!out.ok) { div.textContent = out.msg || 'Error'; return; }

  const driftTxt = (out.drift_score === null || out.drift_score === undefined)
    ? 'No baseline yet'
    : out.drift_score.toFixed(3);

  const rep = out.report && out.report['1'] ? out.report['1'] : null;
  const prec = rep ? (rep.precision || rep['precision']) : null;
  const rec  = rep ? (rep.recall || rep['recall']) : null;

  div.innerHTML = `
    <p><b>Rows used:</b> ${out.n_rows}</p>
    <p><b>Predicted malware rate:</b> ${(out.pred_malware_rate*100).toFixed(2)}%</p>
    <p><b>Drift score (vs baseline):</b> ${driftTxt}</p>
    <p><b>Precision/Recall (label=1):</b> ${prec?prec.toFixed(3):'n/a'} / ${rec?rec.toFixed(3):'n/a'}</p>
  `;
});

resetBtn.addEventListener('click', async () => {
  await fetch('/reset', { method:'POST' });
  if (chart) { chart.data.labels = []; chart.data.datasets[0].data = []; chart.update(); }
  document.getElementById('baselineMsg').textContent = 'State reset. No baseline yet';
  document.getElementById('detectOut').textContent = '';
});
