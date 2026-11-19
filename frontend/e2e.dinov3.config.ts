import { web } from '@e2e-dev/web';
import type { E2EConfig } from 'e2e';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = fileURLToPath(new URL('../', import.meta.url));
const token = process.env.TOMATO_E2E_SESSION_TOKEN;
const modelDir = process.env.TOMATO_E2E_MODEL_DIR;
if (!token) throw new Error('Set TOMATO_E2E_SESSION_TOKEN before starting the run.');
const port = '8768';
const origin = `http://127.0.0.1:${port}`;

export default {
  tests: 'integration/dinov3.e2e.ts',
  workers: 1,
  retries: 0,
  output: '.e2e/dinov3',
  secrets: { sessionToken: token },
  targets: [{
    name: 'chromium',
    engine: web({
      viewport: { width: 1280, height: 900 },
      locale: 'vi-VN',
      initScripts: [`
        if (location.origin === ${JSON.stringify(origin)}) {
          window.pywebview = { api: { get_session_token: async () => ${JSON.stringify(token)} } };
        }
      `],
    }),
    app: {
      url: origin,
      command: {
        executable: process.env.TOMATO_E2E_PYTHON ?? join(repoRoot, '.venv-dev', 'Scripts', 'python.exe'),
        args: ['-m', 'desktop.tests.browser_server', '--port', port, '--mode', modelDir ? 'dinov3' : 'real'],
        cwd: repoRoot,
        env: {
          HF_HUB_OFFLINE: '1',
          TOMATO_E2E_SESSION_TOKEN: token,
          TOMATO_E2E_MODEL_DIR: modelDir ? resolve(modelDir) : '',
          TOMATO_E2E_CARE_PATH: join(repoRoot, 'infra', 'knowledge', 'care-v1.json'),
        },
        reuseExisting: false,
        log: '.e2e/logs/dinov3.log',
      },
    },
  }],
} satisfies E2EConfig;
