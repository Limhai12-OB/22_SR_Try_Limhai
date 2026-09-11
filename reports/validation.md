# Validation notes

- `poetry install`: succeeded using the existing project `.venv`.
- Standalone model check: nomic-embed-text returned a 768-dimensional embedding;
  llama3.2:3b answered `Hello!`.
- Offline ingestion: five text files, 20 chunks persisted in Chroma.
- Standalone vector check for “How do I reset a jammed printer?” returned printer
  chunks 0, 3, and 2, with distances approximately 0.2487, 0.4066, and 0.4905.
- Regression tests: 18 passed. Three deprecation warnings originate in dependencies.
- Actual five-question evidence: see `test_results.md` and `test_results.json`.

## Manual review of recorded answers

| Question | Observation |
| --- | --- |
| First action for a printer jam | Correctly says to turn off the printer; cites the file |
| Configure VPN | Main setup steps are supported; citation is `[3]` rather than a filename |
| Back up important files | Supported but incomplete: omits backup verification and scheduling |
| Secure wireless network | Supported but incomplete: omits later steps such as firmware updates |
| Vacation policy | Correctly refuses; retrieved chunks contain no vacation policy |

All four in-corpus questions retrieved all four chunks of their corresponding
source. The omissions are therefore generation limitations in these runs, not
missing source chunks. The fifth question still retrieved unrelated material,
illustrating why nearest-neighbor search alone does not establish answerability.
