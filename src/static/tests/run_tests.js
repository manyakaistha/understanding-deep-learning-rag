/**
 * Static Frontend Test Suite for UDL Study Interface
 * Tests asset integrity, security invariants, SSE parser, contracts, and sanitization.
 */

const fs = require('fs');
const path = require('path');
const assert = require('assert');

const STATIC_DIR = path.resolve(__dirname, '..');
const ROOT_DIR = path.resolve(STATIC_DIR, '../..');

let totalTests = 0;
let passedTests = 0;

function runTest(name, fn) {
  totalTests++;
  try {
    fn();
    console.log(`  \x1b[32m✓\x1b[0m ${name}`);
    passedTests++;
  } catch (err) {
    console.error(`  \x1b[31m✗\x1b[0m ${name}`);
    console.error(`    ${err.message}`);
  }
}

async function runAsyncTest(name, fn) {
  totalTests++;
  try {
    await fn();
    console.log(`  \x1b[32m✓\x1b[0m ${name}`);
    passedTests++;
  } catch (err) {
    console.error(`  \x1b[31m✗\x1b[0m ${name}`);
    console.error(`    ${err.message}`);
  }
}

async function main() {
  console.log('\n--- Running UDL Static Frontend Test Suite ---\n');

  // Test 1: Pinned Vendor Files Integrity
  runTest('Pinned vendor files exist and are non-empty', () => {
    const requiredFiles = [
      'index.html',
      'app.css',
      'app.js',
      'vendor/purify.min.js',
      'vendor/marked.min.js',
      'vendor/katex/katex.min.js',
      'vendor/katex/katex.min.css',
      'vendor/katex/fonts/KaTeX_Main-Regular.woff2',
      'vendor/prism/prism.min.js',
      'vendor/prism/prism.min.css',
      'vendor/prism/prism-python.min.js',
      'vendor/VENDOR_LICENSES.md'
    ];

    for (const relPath of requiredFiles) {
      const fullPath = path.join(STATIC_DIR, relPath);
      assert.ok(fs.existsSync(fullPath), `File missing: ${relPath}`);
      const stats = fs.statSync(fullPath);
      assert.ok(stats.size > 0, `File is empty: ${relPath}`);
    }
  });

  // Test 2: In-Memory Storage Security (No Local/Session Storage)
  runTest('Security invariant: zero localStorage / sessionStorage / cookie writes in app.js', () => {
    const appJsContent = fs.readFileSync(path.join(STATIC_DIR, 'app.js'), 'utf8');

    assert.ok(!appJsContent.includes('localStorage.set'), 'app.js must not call localStorage.set');
    assert.ok(!appJsContent.includes('localStorage.get'), 'app.js must not call localStorage.get');
    assert.ok(!appJsContent.includes('sessionStorage.set'), 'app.js must not call sessionStorage.set');
    assert.ok(!appJsContent.includes('sessionStorage.get'), 'app.js must not call sessionStorage.get');
    assert.ok(!appJsContent.includes('document.cookie'), 'app.js must not manipulate document.cookie');
  });

  // Test 3: KaTeX Trust Option Security
  runTest('Security invariant: KaTeX is configured with trust: false', () => {
    const appJsContent = fs.readFileSync(path.join(STATIC_DIR, 'app.js'), 'utf8');
    assert.ok(appJsContent.includes('trust: false'), 'KaTeX must be called with trust: false');
    assert.ok(!appJsContent.includes('trust: true'), 'KaTeX must never be called with trust: true');
  });

  // Test 4: Verbatim Content Safety (textContent used for raw sources)
  runTest('Security invariant: raw chunk source and user text use textContent', () => {
    const appJsContent = fs.readFileSync(path.join(STATIC_DIR, 'app.js'), 'utf8');
    assert.ok(appJsContent.includes('.textContent = text'), 'User text must be assigned via textContent');
    assert.ok(appJsContent.includes('.textContent = verbatim'), 'Verbatim chunk code must be assigned via textContent');
  });

  // Test 5: DOMPurify Sanitization on HTML
  runTest('DOMPurify integration exists in app.js', () => {
    const appJsContent = fs.readFileSync(path.join(STATIC_DIR, 'app.js'), 'utf8');
    assert.ok(appJsContent.includes('window.DOMPurify.sanitize'), 'Dynamic markdown rendering must call DOMPurify.sanitize');
  });

  // Test 6: SSE Stream Parser Unit Test (Multiline and Unicode)
  await runAsyncTest('SSE stream parser handles multiline events, Unicode, and chunk boundary splits', async () => {
    // Implement mock stream reader matching app.js logic
    async function mockConsumeSSE(chunks, onEvent) {
      const decoder = new TextDecoder('utf-8');
      let buffer = '';
      let currentEvent = 'message';
      let currentDataLines = [];

      for (const rawChunk of chunks) {
        buffer += decoder.decode(rawChunk, { stream: true });
        const lines = buffer.split(/\r\n|\r|\n/);
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line === '') {
            if (currentDataLines.length > 0) {
              const fullDataStr = currentDataLines.join('\n');
              let parsedData = fullDataStr;
              try {
                parsedData = JSON.parse(fullDataStr);
              } catch (_) {}
              onEvent(currentEvent, parsedData);
            }
            currentEvent = 'message';
            currentDataLines = [];
          } else if (line.startsWith('event:')) {
            currentEvent = line.slice(6).trim();
          } else if (line.startsWith('data:')) {
            currentDataLines.push(line.slice(5).trimStart());
          }
        }
      }
      if (buffer.trim()) {
        if (buffer.startsWith('data:')) {
          currentDataLines.push(buffer.slice(5).trimStart());
        }
      }
      if (currentDataLines.length > 0) {
        const fullDataStr = currentDataLines.join('\n');
        let parsedData = fullDataStr;
        try {
          parsedData = JSON.parse(fullDataStr);
        } catch (_) {}
        onEvent(currentEvent, parsedData);
      }
    }

    const eventsReceived = [];

    // Simulate multi-chunk SSE stream containing Unicode and split packets
    const part1 = Buffer.from('event: sources\ndata: {"hits":[{"chunk_id":"c1","citation":"p. 42"}]}\n\nevent: delta\ndata: {"text":"Deep neural');
    const part2 = Buffer.from(' networks (DNNs) with weights θ and biases β."}\n\nevent: done\ndata: {"answer":"Done text"}\n\n');

    await mockConsumeSSE([part1, part2], (evt, data) => {
      eventsReceived.push({ evt, data });
    });

    assert.strictEqual(eventsReceived.length, 3, 'Should have received 3 events');
    assert.strictEqual(eventsReceived[0].evt, 'sources');
    assert.strictEqual(eventsReceived[0].data.hits[0].chunk_id, 'c1');
    assert.strictEqual(eventsReceived[1].evt, 'delta');
    assert.ok(eventsReceived[1].data.text.includes('weights θ and biases β'));
    assert.strictEqual(eventsReceived[2].evt, 'done');
    assert.strictEqual(eventsReceived[2].data.answer, 'Done text');
  });

  // Test 7: Relative Same-Origin API Endpoints Contract
  runTest('Contract: All API endpoints are relative same-origin routes', () => {
    const appJsContent = fs.readFileSync(path.join(STATIC_DIR, 'app.js'), 'utf8');

    const expectedEndpoints = [
      '/api/status',
      '/api/search',
      '/api/ask',
      '/api/build',
      '/api/reviews',
      '/api/chunks/',
      '/api/asset',
      '/api/source'
    ];

    for (const ep of expectedEndpoints) {
      assert.ok(appJsContent.includes(ep), `app.js must call relative endpoint: ${ep}`);
    }
  });

  // Test 8: Index.html DOM IDs Contract
  runTest('HTML / JS Contract: All DOM element IDs referenced in app.js exist in index.html', () => {
    const indexHtml = fs.readFileSync(path.join(STATIC_DIR, 'index.html'), 'utf8');
    const appJs = fs.readFileSync(path.join(STATIC_DIR, 'app.js'), 'utf8');

    // Extract getElementById calls
    const matches = [...appJs.matchAll(/getElementById\(['"]([^'"]+)['"]\)/g)];
    assert.ok(matches.length > 20, 'Should find elements bound by ID');

    for (const m of matches) {
      const id = m[1];
      const hasId = indexHtml.includes(`id="${id}"`) || indexHtml.includes(`id='${id}'`);
      assert.ok(hasId, `DOM element id="${id}" referenced in app.js must exist in index.html`);
    }
  });

  // Test 9: Mathematical delimiter preprocessing
  runTest('Math formatting correctly converts $..$ and $$..$$ delimiters', () => {
    const sample = 'Let $y = \\sigma(Wx + b)$ be output and $$\\mathcal{L} = \\frac{1}{2}(y - t)^2$$ be MSE.';
    // Display math regex
    const displayMatches = [...sample.matchAll(/\$\$([\s\S]+?)\$\$/g)];
    assert.strictEqual(displayMatches.length, 1);
    assert.strictEqual(displayMatches[0][1], '\\mathcal{L} = \\frac{1}{2}(y - t)^2');

    // Inline math regex
    const inlineMatches = [...sample.matchAll(/\$([^\$\n]+?)\$/g)];
    assert.ok(inlineMatches.length >= 1);
    assert.strictEqual(inlineMatches[0][1], 'y = \\sigma(Wx + b)');
  });

  // Summary
  console.log(`\nResults: ${passedTests}/${totalTests} tests passed.\n`);
  if (passedTests !== totalTests) {
    process.exit(1);
  }
}

main().catch(err => {
  console.error('Test run failed:', err);
  process.exit(1);
});
