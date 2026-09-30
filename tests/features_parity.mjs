// Builds features in "browser mode" for the cases passed on argv and prints them as JSON.
import { readFileSync } from 'node:fs';
import { buildFeatures } from '../web/features.js';

const meta = JSON.parse(readFileSync(new URL('../web/spec.json', import.meta.url)));
const cases = JSON.parse(process.argv[2]);
console.log(JSON.stringify(cases.map((c) => buildFeatures(meta, c.input, c.months).vector)));
