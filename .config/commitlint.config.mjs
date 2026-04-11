export default {
  extends: ["@commitlint/config-conventional"],
  plugins: [
    {
      rules: {
        "no-co-authored-by": (parsed, when) => {
          const hasCoAuthorToken = /\bco-authored-by\b/i.test(parsed.raw ?? "");
          const valid = when === "never" ? hasCoAuthorToken : !hasCoAuthorToken;
          return [valid, "Co-authored-by is not allowed in commit messages"];
        },
      },
    },
  ],
  rules: {
    "type-enum": [
      2,
      "always",
      [
        "feat",
        "fix",
        "docs",
        "style",
        "refactor",
        "perf",
        "test",
        "build",
        "ci",
        "chore",
        "revert",
      ],
    ],
    "type-case": [2, "always", "lower-case"],
    "type-empty": [2, "never"],
    "scope-case": [2, "always", "lower-case"],
    "subject-empty": [2, "never"],
    "subject-full-stop": [2, "never", "."],
    "header-max-length": [2, "always", 100],
    "body-leading-blank": [1, "always"],
    "footer-leading-blank": [1, "always"],
    "no-co-authored-by": [2, "always"],
  },
};
