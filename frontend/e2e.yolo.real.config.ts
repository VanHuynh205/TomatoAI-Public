import type { E2EConfig } from 'e2e';
import { web } from '@e2e-dev/web';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = fileURLToPath(new URL('../', import.meta.url));
const token = process.env.TOMATO_E2E_SESSION_TOKEN;
if (!token) throw new Error('Set TOMATO_E2E_SESSION_TOKEN once before starting the run.');
const port = process.env.TOMATO_E2E_PORT ?? '8767';
const origin = `http://127.0.0.1:${port}`;

export default {
  tests: 'integration/yolo.real.e2e.ts',
  workers: 1,
  retries: 0,
  output: '.e2e-yolo-real',
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
            open_external_url: async (url) => { window.__sourceUrls.push(url); }
          }};
        }
      `],
    }),
    app: {
      url: origin,
      command: {
        executable: process.env.TOMATO_E2E_PYTHON ?? join(repoRoot, '.venv-dev', 'Scripts', 'python.exe'),
        args: ['-m', 'desktop.tests.browser_server', '--port', port, '--mode', 'real'],
        cwd: repoRoot,
        env: {
          TOMATO_E2E_SESSION_TOKEN: token,
          YOLO_OFFLINE: 'true',
          YOLO_AUTOINSTALL: 'false',
          TOMATO_E2E_MODEL_DIR: join(repoRoot, 'ai/artifacts/yolo26s-cls-v1'),
          TOMATO_E2E_CARE_PATH: fileURLToPath(new URL('../infra/knowledge/care-v1.json', import.meta.url)),
        },
        reuseExisting: false,
        log: '.e2e/logs/app.log',
      },
    },
  }],
} satisfies E2EConfig;
