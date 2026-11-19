import { fileURLToPath } from 'node:url';
import base from './e2e.config';

const original = base.targets[0];

export default {
  ...base,
  tests: 'integration/yolo.e2e.ts',
  output: '.e2e-yolo-tracer',
  targets: [{
    ...original,
    app: {
      ...original.app,
      command: {
        ...original.app.command,
        args: ['-m', 'desktop.tests.browser_server', '--port', '8765', '--mode', 'fixture-bundle'],
        env: {
          ...original.app.command.env,
          YOLO_OFFLINE: 'true',
          YOLO_AUTOINSTALL: 'false',
          TOMATO_E2E_MODEL_DIR: process.env.TOMATO_E2E_MODEL_DIR ?? '',
          TOMATO_E2E_CARE_PATH: fileURLToPath(new URL('../infra/knowledge/care-v1.json', import.meta.url)),
        },
      },
    },
  }],
};
