import { score } from './model.js';
import { buildFeatures } from './features.js';

const $ = (id) => document.getElementById(id);
const money = (v) => '$' + Math.round(v / 1000).toLocaleString('en-AU') + 'k';
const pct = (v) => (v * 100).toFixed(1) + '%';

const [trees, meta] = await Promise.all([
  fetch('trees.json').then((r) => r.json()),
  fetch('spec.json').then((r) => r.json()),
]);

$('suburbs').innerHTML = meta.suburbs.map((s) => `<option value="${s}"></option>`).join('');
$('test-period').textContent = `${meta.test_period[0]} to ${meta.test_period[1]}`;
$('b-med').textContent = pct(meta.metrics.baseline.median_ape);
$('b-10').textContent = pct(meta.metrics.baseline.within_10pct);
$('b-mae').textContent = money(meta.metrics.baseline.mae);
$('m-med').textContent = pct(meta.metrics.model.median_ape);
$('m-10').textContent = pct(meta.metrics.model.within_10pct);
$('m-mae').textContent = money(meta.metrics.model.mae);
$('cov').textContent = pct(meta.metrics.coverage);

const num = (id) => {
  const v = $(id).value.trim();
  return v === '' ? null : Number(v);
};

$('form').addEventListener('submit', (e) => {
  e.preventDefault();
  const suburb = meta.suburbs.find((s) => s.toLowerCase() === $('suburb').value.trim().toLowerCase());
  $('suburb-err').hidden = !!suburb;
  if (!suburb) return $('suburb').focus();
  $('suburb').value = suburb;

  const clamp = (v, lo, hi) => (v == null ? null : Math.min(hi, Math.max(lo, v)));
  const input = {
    suburb,
    type: document.querySelector('input[name="type"]:checked').value,
    rooms: clamp(num('rooms') ?? 3, 1, 10),
    bath: clamp(num('bath'), 1, 8),
    car: clamp(num('car'), 0, 10),
    land: clamp(num('land'), 1, 20000),
    building: clamp(num('building'), 20, 1500),
    year: clamp(num('year'), 1840, 2018),
  };

  const { vector, baseline } = buildFeatures(meta, input);
  const logPrice = score(trees, vector);
  const price = Math.exp(logPrice);
  const lo = price * Math.exp(meta.interval_log[0]);
  const hi = price * Math.exp(meta.interval_log[1]);

  $('empty').hidden = true;
  $('out').hidden = false;
  $('price').textContent = money(price);
  $('range').textContent = `${money(lo)} to ${money(hi)}`;

  // bar: scale from 0.6x the low end to 1.2x the high end
  const min = Math.min(lo, baseline) * 0.85;
  const max = Math.max(hi, baseline) * 1.1;
  const at = (v) => ((v - min) / (max - min)) * 100 + '%';
  $('bar-range').style.left = at(lo);
  $('bar-range').style.width = `calc(${at(hi)} - ${at(lo)})`;
  $('bar-point').style.left = at(price);
  $('bar-base').style.left = at(baseline);

  const diff = price / baseline - 1;
  const typeName = { h: 'houses', t: 'townhouses', u: 'units' }[input.type];
  $('vs').innerHTML =
    Math.abs(diff) < 0.03
      ? `Right on the typical price for ${typeName} in ${suburb} (${money(baseline)}).`
      : `<strong>${pct(Math.abs(diff))} ${diff > 0 ? 'above' : 'below'}</strong> the typical price for ${typeName} in ${suburb} (${money(baseline)}), because of the details you entered.`;
});
