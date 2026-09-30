// Evaluates the exported XGBoost trees. Each tree is [value] for a leaf, or
// [featureIndex, threshold, yes, no, missing] for a split. XGBoost compares in float32,
// so we do too - that keeps browser predictions identical to Python's.
const f32 = Math.fround;

function walk(node, x) {
  while (node.length > 1) {
    const v = x[node[0]];
    if (v === null || v === undefined || Number.isNaN(v)) node = node[4] === 'y' ? node[2] : node[3];
    else node = f32(v) < f32(node[1]) ? node[2] : node[3];
  }
  return node[0];
}

/** Returns the predicted log price for one feature vector. */
export function score(model, x) {
  let sum = model.base;
  for (const tree of model.trees) sum += walk(tree, x);
  return sum;
}
