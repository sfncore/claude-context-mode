---
name: context-mode
description: Route large-output operations through context-mode MCP tools instead of raw shell/file tools
globs: ["*"]
alwaysApply: true
---

# Context-Mode Tool Routing

You have access to context-mode MCP tools that keep large outputs in a sandbox
instead of flooding your context window. **Use these tools by default for any
operation where output size is uncertain.**

## Decision Tree

```
About to run a command or read a file?
│
├── Output will definitely be small (<20 lines)?
│   └── Use shell tool directly
│
├── Output MIGHT be large or you're UNSURE?
│   └── Use context-mode execute(language, code, intent?)
│
├── Reading a large file for analysis?
│   └── Use context-mode execute_file(path, language, code)
│
├── Fetching web documentation?
│   └── Use context-mode fetch_and_index(url) → search(queries)
│
├── Running multiple commands + searching results?
│   └── Use context-mode batch_execute(commands, queries)
│
└── Need to index content for later search?
    └── Use context-mode index(content/path, source) → search(queries)
```

## Tool Reference

| Tool | Use For |
|------|---------|
| `execute` | Run code/commands, only stdout enters context |
| `execute_file` | Process a file in sandbox without loading it into context |
| `batch_execute` | Multiple commands + search queries in one call |
| `index` | Store content in FTS5 knowledge base |
| `search` | Query indexed content with BM25 ranking |
| `fetch_and_index` | Fetch URL, convert to markdown, index for search |
| `stats` | Show context savings for this session |

## When to Use context-mode

- `git log`, `git diff`, `git show` — history and diffs (large output)
- `npm test`, `pytest`, `cargo test` — test output
- `curl`/API calls — response bodies
- Reading log files, JSON data, CSV
- `ls -laR` on large directory trees
- Any command piped through `head`/`tail`/`grep` — use execute instead

## When NOT to Use context-mode

- File mutations: `git add`, `git commit`, `git push`, `mkdir`, `cp`, `mv`
- Small known output: `echo`, `pwd`, `which`, `date`
- Interactive commands that need stdin

**Note:** Monitoring commands (`gt`, `bd`, `tmux`, `git status`) work fine through
context-mode — output is always fresh. But do NOT use `intent` on monitoring commands
(it filters output), and always re-run commands instead of searching stale indexes.

## Key Pattern: intent Parameter

When using `execute`, pass the `intent` parameter to describe what you're
looking for. If output exceeds 5KB, context-mode auto-indexes it and returns
only matching sections instead of truncating.

```
execute(language: "shell", code: "npm test", intent: "failing tests")
```

## Anti-Patterns

- Using shell for `cat large-file.json` → floods context. Use `execute_file`.
- Piping through `| head -20` → loses data. Use `execute` to analyze ALL data.
- Running `gh pr list` via shell → raw output. Use `execute` with `--jq` filter.
- Calling `index(content: huge_string)` → data enters context as parameter. Use `index(path: file)`.
