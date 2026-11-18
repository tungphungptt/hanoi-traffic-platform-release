const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');

// Classic <script> files share one global scope: a top-level const/let/function declared twice is a SyntaxError that
// silently stops the second file. Check the files the page loads together.
test('page scripts declare no top-level name twice', () => {
  const html = fs.readFileSync(path.join(__dirname, '../static/index.html'), 'utf8');
  const files = [...html.matchAll(/<script src="\/static\/([\w.]+\.js)/g)].map(m => m[1]);
  assert.ok(files.length >= 4);
  const seen = new Map();
  for (const file of files) {
    const text = fs.readFileSync(path.join(__dirname, '../static', file), 'utf8');
    for (const m of text.matchAll(/^(?:async\s+)?(?:const|let|var|function|class)\s+([A-Za-z_$][\w$]*)/gm)) {
      assert.ok(!seen.has(m[1]), `${m[1]} is declared in both ${seen.get(m[1])} and ${file}`);
      seen.set(m[1], file);
    }
  }
});
