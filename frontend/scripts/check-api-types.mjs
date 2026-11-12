import { mkdtempSync, readFileSync, rmSync, rmdirSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../', import.meta.url));
const directory = mkdtempSync(join(root, '.api-types-'));
const temporary = join(directory, 'schema.d.ts');
try {
  const generated = spawnSync(process.execPath, [
    join(root, 'node_modules/openapi-typescript/bin/cli.js'),
    fileURLToPath(new URL('../../docs/contracts/openapi-v1.json', import.meta.url)),
    '-o', temporary,
  ], { cwd: root, encoding: 'utf8' });
  if (generated.error || generated.status !== 0) {
    throw new Error(generated.error?.message ?? generated.stderr);
  }
  if (!readFileSync(temporary).equals(readFileSync(join(root, 'src/api/schema.d.ts')))) {
    throw new Error('API types differ from OpenAPI. Run npm run api:types and review the change.');
  }
  console.log('API types match OpenAPI.');
} catch (error) {
  console.error(error.message);
  process.exitCode = 1;
} finally {
  rmSync(temporary, { force: true });
  rmdirSync(directory);
}
