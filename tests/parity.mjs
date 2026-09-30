// Scores the parity fixture with the browser model and prints the predictions as JSON.
import { readFileSync } from 'node:fs';
import { score } from '../web/model.js';

const root = new URL('..', import.meta.url);
const model = JSON.parse(readFileSync(new URL('web/trees.json', root)));
const { X } = JSON.parse(readFileSync(new URL('tests/parity_fixture.json', root)));
console.log(JSON.stringify(X.map((x) => score(model, x))));
