import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

export default /** @type {import('eslint').Linter.Config[]} */ [
  {
    files: ["**/*.{ts,tsx}"],
    extends: ["eslint:recommended", "plugin:@typescript-eslint/recommended"],
    languageOptions: {
      ecmaVersion: "latest",
      sourceType: "module",
      parser: await import("@typescript-eslint/parser").then((m) => m.default),
      parserOptions: {
        project: `${__dirname}/tsconfig.json`,
        ecmaFeatures: { jsx: true },
      },
      globals: {
        browser: true,
        es2021: true,
        node: true,
      },
    },
    plugins: {
      "@typescript-eslint": await import("@typescript-eslint/eslint-plugin").then(
        (m) => m.default
      ),
    },
    rules: {
      "no-unused-vars": "off",
      "@typescript-eslint/no-unused-vars": ["warn", { argsIgnorePattern: "^_" }],
      "no-console": "warn",
    },
  },
];
