import { web } from '@e2e-dev/web';
import type { E2EConfig } from 'e2e';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = fileURLToPath(new URL('../', import.meta.url));
const token = process.env.TOMATO_E2E_SESSION_TOKEN;
const modelDir = process.env.TOMATO_E2E_MODEL_DIR;
if (!token) throw new Error('Set TOMATO_E2E_SESSION_TOKEN before starting the run.');
if (!modelDir) throw new Error('Set TOMATO_E2E_MODEL_DIR to the exported CNN-GRU bundle directory.');
const port = process.env.TOMATO_E2E_PORT ?? '8769';
const origin = `http://127.0.0.1:${port}`;

export default {
  tests: 'integration/cnn-gru.e2e.ts',
  workers: 1,
  retries: 0,
  output: '.e2e/cnn-gru',
  secrets: { sessionToken: token },
  targets: [{
    name: 'chromium',
    engine: web({
      viewport: { width: 1280, height: 900 },
      locale: 'vi-VN',
      initScripts: [`
        if (location.origin === ${JSON.stringify(origin)}) {
          window.__sourceUrls = [];
          window.pywebview = { api: {
            get_session_token: async () => ${JSON.stringify(token)},
            open_external_url: async (url) => { window.__sourceUrls.push(url); },
          }};
        }
      `],
    }),
    app: {
      url: origin,
      command: {
        executable: process.env.TOMATO_E2E_PYTHON ?? join(repoRoot, '.venv-dev', 'Scripts', 'python.exe'),
        args: ['-m', 'desktop.tests.browser_server', '--port', port, '--mode', 'cnn-gru'],
        cwd: repoRoot,
        env: {
          TOMATO_E2E_SESSION_TOKEN: token,
          TOMATO_E2E_MODEL_DIR: resolve(modelDir),
          TOMATO_E2E_CARE_PATH: join(repoRoot, 'infra', 'knowledge', 'care-v1.json'),
          TOMATO_E2E_REQUIRE_TRAINED: process.env.TOMATO_E2E_REQUIRE_TRAINED ?? '1',
        },
        reuseExisting: false,
        log: '.e2e/logs/cnn-gru.log',
      },
    },
  }],
} satisfies E2EConfig;
