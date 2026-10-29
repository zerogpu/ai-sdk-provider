import { defineConfig } from 'tsdown';

export default defineConfig({
  target: 'es2022',
  tsconfig: 'tsconfig.build.json',
  dts: true,
  sourcemap: true,
  fixedExtension: false,
  define: {
    __PACKAGE_VERSION__: JSON.stringify(
      (await import('./package.json', { with: { type: 'json' } })).default
        .version,
    ),
  },
});
