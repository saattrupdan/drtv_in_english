// Build the extension into dist/chrome and dist/firefox.
//
// Each target gets: bundled JS, the right manifest, options/index.html,
// and icons. Two outputs because the manifests differ
// (background.service_worker on Chrome, background.scripts on Firefox MV3,
// plus browser_specific_settings on Firefox).

import { build, context } from "esbuild";
import { cp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { spawn } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import sharp from "sharp";

const here = dirname(fileURLToPath(import.meta.url));
const watch = process.argv.includes("--watch");
const pkg = process.argv.includes("--package");

// package.json is the single source of truth for the version; it gets
// injected into each manifest at build time (see writeManifest).
const { version } = JSON.parse(
  await readFile(resolve(here, "package.json"), "utf8"),
);

const entryPoints = {
  "background/index": resolve(here, "src/background/index.ts"),
  "content/index": resolve(here, "src/content/index.ts"),
  "content/early": resolve(here, "src/content/early.ts"),
  "content/inject": resolve(here, "src/content/inject.ts"),
  "content/subtitle-fetcher": resolve(here, "src/content/subtitle-fetcher.ts"),
  "options/options": resolve(here, "src/options/options.ts"),
};

const targets = [
  { name: "chrome", manifest: "manifest.chrome.json" },
  { name: "firefox", manifest: "manifest.firefox.json" },
];

const ICON_SIZES = [16, 32, 48, 128];

// Raw source only — no dist/ output, node_modules, or store artifacts — so
// Mozilla's "no transpiled/concatenated/minified/machine-generated source"
// rule holds. SOURCE_SUBMISSION.md carries the build instructions.
const SOURCE_ENTRIES = [
  "src",
  "build.mjs",
  "package.json",
  "package-lock.json",
  "tsconfig.json",
  "manifest.chrome.json",
  "manifest.firefox.json",
  "icons",
  "README.md",
  "AGENTS.md",
  "CHANGELOG.md",
  "package/submission/SOURCE_SUBMISSION.md",
];

async function copyStatic(outDir) {
  await mkdir(resolve(outDir, "options"), { recursive: true });
  await cp(
    resolve(here, "src/options/index.html"),
    resolve(outDir, "options/index.html"),
  );
  await renderIcons(resolve(outDir, "icons"));
}

async function renderIcons(iconsOutDir) {
  const svgPath = resolve(here, "icons/icon.svg");
  if (!existsSync(svgPath)) return;
  await mkdir(iconsOutDir, { recursive: true });
  const svg = await readFile(svgPath);
  await Promise.all(
    ICON_SIZES.map((size) =>
      sharp(svg)
        .resize(size, size)
        .png()
        .toFile(resolve(iconsOutDir, `icon-${size}.png`)),
    ),
  );
}

async function writeManifest(outDir, manifestName) {
  const manifest = JSON.parse(await readFile(resolve(here, manifestName), "utf8"));
  // package.json wins: a bump there propagates to both manifests, so the
  // manifests' own version field is just a placeholder.
  manifest.version = version;
  await writeFile(
    resolve(outDir, "manifest.json"),
    JSON.stringify(manifest, null, 2) + "\n",
  );
}

async function buildTarget({ name, manifest }) {
  const outDir = resolve(here, "dist", name);
  await rm(outDir, { recursive: true, force: true });
  await mkdir(outDir, { recursive: true });

  // IIFE for background + content so neither manifest needs ESM
  // module wiring; ESM for the options page (loaded from index.html as
  // a <script type="module">).
  const common = {
    bundle: true,
    target: "es2022",
    sourcemap: true,
    logLevel: "info",
  };
  const jobs = [
    {
      ...common,
      entryPoints: {
        "background/index": entryPoints["background/index"],
        "content/index": entryPoints["content/index"],
        "content/early": entryPoints["content/early"],
        "content/inject": entryPoints["content/inject"],
        "content/subtitle-fetcher": entryPoints["content/subtitle-fetcher"],
      },
      outdir: outDir,
      format: "iife",
    },
    {
      ...common,
      entryPoints: { "options/options": entryPoints["options/options"] },
      outdir: outDir,
      format: "esm",
    },
  ];

  if (watch) {
    for (const j of jobs) {
      const ctx = await context(j);
      await ctx.watch();
    }
  } else {
    for (const j of jobs) await build(j);
  }

  await copyStatic(outDir);
  await writeManifest(outDir, manifest);
  console.log(`[build] ${name} -> ${outDir}`);
}

for (const t of targets) {
  await buildTarget(t);
}

if (watch) {
  console.log("[build] watching for changes…");
}

if (pkg) {
  for (const t of targets) {
    await packageTarget(t.name, version);
  }
  // Firefox AMO requires the source for bundled add-ons. Always produced
  // alongside the store zips so it can never drift to a stale version.
  await packageSource(version);
}

async function packageTarget(name, version) {
  const outDir = resolve(here, "dist", name);
  const zipName = `drtv-in-english-${name}-${version}.zip`;
  const zipPath = resolve(here, "dist", zipName);
  await rm(zipPath, { force: true });
  // `zip -r ... .` from inside outDir so paths inside the archive are
  // relative to the extension root (manifest.json at top level — what
  // both stores expect).
  await runZip(outDir, zipPath, ["."], ["*.map"]);
  console.log(`[build] packaged ${name} -> dist/${zipName}`);
}

async function packageSource(version) {
  const zipName = `drtv-in-english-source-${version}.zip`;
  const zipPath = resolve(here, "dist", zipName);
  await mkdir(resolve(here, "dist"), { recursive: true });
  await rm(zipPath, { force: true });
  // Zipped from the repo root so archive paths mirror the project layout.
  await runZip(here, zipPath, SOURCE_ENTRIES, ["*.DS_Store"]);
  console.log(`[build] packaged source -> dist/${zipName}`);
}

function runZip(cwd, zipPath, entries, excludes) {
  return new Promise((resolveP, reject) => {
    const args = ["-r", "-q", "-X", zipPath, ...entries, "-x", ...excludes];
    const child = spawn("zip", args, { cwd, stdio: "inherit" });
    child.on("error", reject);
    child.on("exit", (code) =>
      code === 0 ? resolveP() : reject(new Error(`zip exited ${code}`)),
    );
  });
}
