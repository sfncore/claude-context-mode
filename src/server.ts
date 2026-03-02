#!/usr/bin/env node
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { PolyglotExecutor } from "./executor.js";
import { ContentStore, cleanupStaleDBs, loadDatabase, type SearchResult } from "./store.js";
import { readdirSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  detectRuntimes,
  getRuntimeSummary,
  getAvailableLanguages,
  hasBunRuntime,
} from "./runtime.js";

const VERSION = "0.8.1";
const runtimes = detectRuntimes();
const available = getAvailableLanguages(runtimes);
const server = new McpServer({
  name: "context-mode",
  version: VERSION,
});

const executor = new PolyglotExecutor({
  runtimes,
  projectRoot: process.env.CLAUDE_PROJECT_DIR,
});

// Lazy singleton — no DB overhead unless index/search is used
let _store: ContentStore | null = null;
function getStore(): ContentStore {
  if (!_store) _store = new ContentStore();
  return _store;
}

// Named persistent store cache
const _namedStores = new Map<string, ContentStore>();
function getNamedStore(name: string): ContentStore {
  let store = _namedStores.get(name);
  if (!store) {
    store = ContentStore.openNamed(name);
    _namedStores.set(name, store);
  }
  return store;
}

/** Get the ephemeral store or a named persistent store. */
function resolveStore(database?: string): ContentStore {
  return database ? getNamedStore(database) : getStore();
}

// ─────────────────────────────────────────────────────────
// Session stats — track context consumption per tool
// ─────────────────────────────────────────────────────────

const sessionStats = {
  calls: {} as Record<string, number>,
  bytesReturned: {} as Record<string, number>,
  bytesIndexed: 0,
  bytesSandboxed: 0, // network I/O consumed inside sandbox (never enters context)
  sessionStart: Date.now(),
};

type ToolResult = {
  content: Array<{ type: "text"; text: string }>;
  isError?: boolean;
};

function trackResponse(toolName: string, response: ToolResult): ToolResult {
  const bytes = response.content.reduce(
    (sum, c) => sum + Buffer.byteLength(c.text),
    0,
  );
  sessionStats.calls[toolName] = (sessionStats.calls[toolName] || 0) + 1;
  sessionStats.bytesReturned[toolName] =
    (sessionStats.bytesReturned[toolName] || 0) + bytes;
  return response;
}

function trackIndexed(bytes: number): void {
  sessionStats.bytesIndexed += bytes;
}

// ─────────────────────────────────────────────────────────
// Safety: Block dangerous Gas Town commands
// ─────────────────────────────────────────────────────────

/** Commands that mutate rig lifecycle state — must never run inside context-mode. */
const BLOCKED_GT_PATTERNS: RegExp[] = [
  /\bgt\s+rig\s+unpark\b/,
  /\bgt\s+rig\s+undock\b/,
  /\bgt\s+rig\s+start\b/,
  /\bgt\s+rig\s+restart\b/,
  /\bgt\s+rig\s+reboot\b/,
  /\bgt\s+rig\s+remove\b/,
  /\bgt\s+rig\s+add\b/,
  /\bgt\s+shutdown\b/,
  /\bgt\s+dolt\s+stop\b/,
  /\bgt\s+dolt\s+start\b/,
  /\bgt\s+polecat\s+nuke\b/,
  /\bgt\s+deacon\s+stop\b/,
  /\bgt\s+deacon\s+start\b/,
];

/**
 * Check if code/command contains blocked Gas Town operations.
 * Returns the matched command string if blocked, or null if safe.
 */
function checkBlockedCommand(code: string): string | null {
  for (const pattern of BLOCKED_GT_PATTERNS) {
    const match = code.match(pattern);
    if (match) return match[0];
  }
  return null;
}

/** Return a JSON-structured response for better UI rendering (OMP JSON tree view). */
function jsonResponse(toolName: string, data: Record<string, unknown>, isError = false): ToolResult {
  return trackResponse(toolName, {
    content: [{ type: "text" as const, text: JSON.stringify(data) }],
    ...(isError && { isError: true }),
  });
}

// Build description dynamically based on detected runtimes
const langList = available.join(", ");
const bunNote = hasBunRuntime()
  ? " (Bun detected — JS/TS runs 3-5x faster)"
  : "";

// ─────────────────────────────────────────────────────────
// Helper: smart snippet extraction — returns windows around
// matching query terms instead of dumb truncation
//
// When `highlighted` is provided (from FTS5 `highlight()` with
// STX/ETX markers), match positions are derived from the markers.
// This is the authoritative source — FTS5 uses the exact same
// tokenizer that produced the BM25 match, so stemmed variants
// like "configuration" matching query "configure" are found
// correctly. Falls back to indexOf on raw terms when highlighted
// is absent (non-FTS codepath).
// ─────────────────────────────────────────────────────────

const STX = "\x02";
const ETX = "\x03";

/**
 * Parse FTS5 highlight markers to find match positions in the
 * original (marker-free) text. Returns character offsets into the
 * stripped content where each matched token begins.
 */
export function positionsFromHighlight(highlighted: string): number[] {
  const positions: number[] = [];
  let cleanOffset = 0;

  let i = 0;
  while (i < highlighted.length) {
    if (highlighted[i] === STX) {
      // Record position of this match in the clean text
      positions.push(cleanOffset);
      i++; // skip STX
      // Advance through matched text until ETX
      while (i < highlighted.length && highlighted[i] !== ETX) {
        cleanOffset++;
        i++;
      }
      if (i < highlighted.length) i++; // skip ETX
    } else {
      cleanOffset++;
      i++;
    }
  }

  return positions;
}

/** Strip STX/ETX markers to recover original content. */
function stripMarkers(highlighted: string): string {
  return highlighted.replaceAll(STX, "").replaceAll(ETX, "");
}

export function extractSnippet(
  content: string,
  query: string,
  maxLen = 1500,
  highlighted?: string,
): string {
  if (content.length <= maxLen) return content;

  // Derive match positions from FTS5 highlight markers when available
  const positions: number[] = [];

  if (highlighted) {
    for (const pos of positionsFromHighlight(highlighted)) {
      positions.push(pos);
    }
  }

  // Fallback: indexOf on raw query terms (non-FTS codepath)
  if (positions.length === 0) {
    const terms = query
      .toLowerCase()
      .split(/\s+/)
      .filter((t) => t.length > 2);
    const lower = content.toLowerCase();

    for (const term of terms) {
      let idx = lower.indexOf(term);
      while (idx !== -1) {
        positions.push(idx);
        idx = lower.indexOf(term, idx + 1);
      }
    }
  }

  // No matches at all — return prefix
  if (positions.length === 0) {
    return content.slice(0, maxLen) + "\n…";
  }

  // Sort positions, merge overlapping windows
  positions.sort((a, b) => a - b);
  const WINDOW = 300;
  const windows: Array<[number, number]> = [];

  for (const pos of positions) {
    const start = Math.max(0, pos - WINDOW);
    const end = Math.min(content.length, pos + WINDOW);
    if (windows.length > 0 && start <= windows[windows.length - 1][1]) {
      windows[windows.length - 1][1] = end;
    } else {
      windows.push([start, end]);
    }
  }

  // Collect windows until maxLen
  const parts: string[] = [];
  let total = 0;
  for (const [start, end] of windows) {
    if (total >= maxLen) break;
    const part = content.slice(start, Math.min(end, start + (maxLen - total)));
    parts.push(
      (start > 0 ? "…" : "") + part + (end < content.length ? "…" : ""),
    );
    total += part.length;
  }

  return parts.join("\n\n");
}

// ─────────────────────────────────────────────────────────
// Tool: execute
// ─────────────────────────────────────────────────────────

server.registerTool(
  "execute",
  {
    title: "Run Code",
    description: `Execute code in a sandboxed subprocess. Only stdout enters context.${bunNote} Available: ${langList}. Prefer over Bash for commands with large output.`,
    annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: false, openWorldHint: true },
    inputSchema: z.object({
      language: z
        .enum(["javascript", "typescript", "python", "shell", "ruby", "go", "rust", "php", "perl", "r", "elixir"])
        .describe("Runtime language"),
      code: z.string().describe("Source code to execute. Print output to stdout."),
      timeout: z.number().optional().default(30000).describe("Max execution time in ms"),
      intent: z.string().optional().describe("What you're looking for. Large output (>5KB) gets indexed; use search() to retrieve sections."),
    }),
  },
  async ({ language, code, timeout, intent }) => {
    try {
      // Safety: block dangerous gt commands in shell code
      if (language === "shell") {
        const blocked = checkBlockedCommand(code);
        if (blocked) {
          return jsonResponse("execute", {
            error: `Blocked: "${blocked}" is a protected Gas Town operation and cannot be run through context-mode.`,
          }, true);
        }
      }

      // For JS/TS: wrap in async IIFE with fetch interceptor to track network bytes
      let instrumentedCode = code;
      if (language === "javascript" || language === "typescript") {
        instrumentedCode = `
let __cm_net=0;const __cm_f=globalThis.fetch;
globalThis.fetch=async(...a)=>{const r=await __cm_f(...a);
try{const cl=r.clone();const b=await cl.arrayBuffer();__cm_net+=b.byteLength}catch{}
return r};
async function __cm_main(){
${code}
}
__cm_main().catch(e=>{console.error(e);process.exitCode=1}).finally(()=>{
if(__cm_net>0)process.stderr.write('__CM_NET__:'+__cm_net+'\\n');
});`;
      }
      const result = await executor.execute({ language, code: instrumentedCode, timeout });

      // Parse sandbox network metrics from stderr
      const netMatch = result.stderr?.match(/__CM_NET__:(\d+)/);
      if (netMatch) {
        sessionStats.bytesSandboxed += parseInt(netMatch[1]);
        // Clean the metric line from stderr
        result.stderr = result.stderr.replace(/\n?__CM_NET__:\d+\n?/g, "");
      }

      if (result.timedOut) {
        return jsonResponse("execute", {
          error: `Execution timed out after ${timeout}ms`,
          language, output: result.stdout, stderr: result.stderr,
        }, true);
      }

      if (result.exitCode !== 0) {
        const output = `Exit code: ${result.exitCode}\n\nstdout:\n${result.stdout}\n\nstderr:\n${result.stderr}`;
        if (intent && intent.trim().length > 0 && Buffer.byteLength(output) > INTENT_SEARCH_THRESHOLD) {
          trackIndexed(Buffer.byteLength(output));
          return jsonResponse("execute", {
            language, exit_code: result.exitCode, indexed: true,
            ...parseIntentResult(intentSearch(output, intent, `execute:${language}:error`)),
          }, true);
        }
        return jsonResponse("execute", {
          language, exit_code: result.exitCode,
          output: result.stdout, stderr: result.stderr,
        }, true);
      }

      const stdout = result.stdout || "(no output)";

      // Intent-driven search: if intent provided and output is large enough
      if (intent && intent.trim().length > 0 && Buffer.byteLength(stdout) > INTENT_SEARCH_THRESHOLD) {
        trackIndexed(Buffer.byteLength(stdout));
        return jsonResponse("execute", {
          language, exit_code: 0, indexed: true,
          ...parseIntentResult(intentSearch(stdout, intent, `execute:${language}`)),
        });
      }

      return jsonResponse("execute", {
        language, exit_code: 0, output: stdout,
      });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      return jsonResponse("execute", { error: message }, true);
    }
  },
);

// ─────────────────────────────────────────────────────────
// Helper: index stdout into FTS5 knowledge base
// ─────────────────────────────────────────────────────────

function indexStdout(
  stdout: string,
  source: string,
): { content: Array<{ type: "text"; text: string }> } {
  const store = getStore();
  trackIndexed(Buffer.byteLength(stdout));
  const indexed = store.index({ content: stdout, source });
  return {
    content: [
      {
        type: "text" as const,
        text: `Indexed ${indexed.totalChunks} sections (${indexed.codeChunks} with code) from: ${indexed.label}\nUse search(queries: ["..."]) to query this content. Use source: "${indexed.label}" to scope results.`,
      },
    ],
  };
}

// ─────────────────────────────────────────────────────────
// Helper: intent-driven search on execution output
// ─────────────────────────────────────────────────────────

const INTENT_SEARCH_THRESHOLD = 5_000; // bytes — ~80-100 lines

function intentSearch(
  stdout: string,
  intent: string,
  source: string,
  maxResults: number = 5,
): string {
  const totalLines = stdout.split("\n").length;
  const totalBytes = Buffer.byteLength(stdout);

  // Index into the PERSISTENT store so user can search() later
  const persistent = getStore();
  const indexed = persistent.indexPlainText(stdout, source);

  // Search the persistent store directly (porter → trigram → fuzzy)
  let results = persistent.searchWithFallback(intent, maxResults, source);

  // Extract distinctive terms as vocabulary hints for the LLM
  const distinctiveTerms = persistent.getDistinctiveTerms(indexed.sourceId);

  if (results.length === 0) {
    const lines = [
      `Indexed ${indexed.totalChunks} sections from "${source}" into knowledge base.`,
      `No sections matched intent "${intent}" in ${totalLines}-line output (${(totalBytes / 1024).toFixed(1)}KB).`,
    ];
    if (distinctiveTerms.length > 0) {
      lines.push("");
      lines.push(`Searchable terms: ${distinctiveTerms.join(", ")}`);
    }
    lines.push("");
    lines.push("Use search() to explore the indexed content.");
    return lines.join("\n");
  }

  // Return ONLY titles + first-line previews — not full content
  const lines = [
    `Indexed ${indexed.totalChunks} sections from "${source}" into knowledge base.`,
    `${results.length} sections matched "${intent}" (${totalLines} lines, ${(totalBytes / 1024).toFixed(1)}KB):`,
    "",
  ];

  for (const r of results) {
    const preview = r.content.split("\n")[0].slice(0, 120);
    lines.push(`  - ${r.title}: ${preview}`);
  }

  if (distinctiveTerms.length > 0) {
    lines.push("");
    lines.push(`Searchable terms: ${distinctiveTerms.join(", ")}`);
  }

  lines.push("");
  lines.push("Use search(queries: [...]) to retrieve full content of any section.");

  return lines.join("\n");
}

// ─────────────────────────────────────────────────────────
// Tool: execute_file
// ─────────────────────────────────────────────────────────

/** Parse intentSearch text output into structured fields for JSON response. */
function parseIntentResult(text: string): Record<string, unknown> {
  const lines = text.split("\n");
  const sections: string[] = [];
  let searchable_terms: string[] = [];
  let summary = "";

  for (const line of lines) {
    if (line.startsWith("Indexed ")) summary = line;
    else if (line.startsWith("  - ")) sections.push(line.slice(4).trim());
    else if (line.startsWith("Searchable terms: ")) {
      searchable_terms = line.slice(18).split(", ").filter(Boolean);
    }
  }

  return {
    ...(summary && { summary }),
    ...(sections.length > 0 && { sections }),
    ...(searchable_terms.length > 0 && { searchable_terms }),
  };
}

server.registerTool(
  "execute_file",
  {
    title: "Process File",
    description: "Read a file into FILE_CONTENT variable and process it in a sandbox. Only printed output enters context. Prefer over Read/cat for large files.",
    annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
    inputSchema: z.object({
      path: z.string().describe("File path (absolute or relative)"),
      language: z
        .enum(["javascript", "typescript", "python", "shell", "ruby", "go", "rust", "php", "perl", "r", "elixir"])
        .describe("Runtime language"),
      code: z.string().describe("Code to process FILE_CONTENT. Print summary to stdout."),
      timeout: z.number().optional().default(30000).describe("Max execution time in ms"),
      intent: z.string().optional().describe("What you're looking for. Large output gets indexed; use search() to retrieve."),
    }),
  },
  async ({ path, language, code, timeout, intent }) => {
    try {
      // Safety: block dangerous gt commands in shell code
      if (language === "shell") {
        const blocked = checkBlockedCommand(code);
        if (blocked) {
          return jsonResponse("execute_file", {
            error: `Blocked: "${blocked}" is a protected Gas Town operation and cannot be run through context-mode.`,
          }, true);
        }
      }

      const result = await executor.executeFile({ path, language, code, timeout });

      if (result.timedOut) {
        return jsonResponse("execute_file", { error: `Timed out after ${timeout}ms`, path, language }, true);
      }

      if (result.exitCode !== 0) {
        const output = `Error processing ${path} (exit ${result.exitCode}):\n${result.stderr || result.stdout}`;
        if (intent && intent.trim().length > 0 && Buffer.byteLength(output) > INTENT_SEARCH_THRESHOLD) {
          trackIndexed(Buffer.byteLength(output));
          return jsonResponse("execute_file", {
            path, language, exit_code: result.exitCode, indexed: true,
            ...parseIntentResult(intentSearch(output, intent, `file:${path}:error`)),
          }, true);
        }
        return jsonResponse("execute_file", {
          path, language, exit_code: result.exitCode,
          output: result.stdout, stderr: result.stderr,
        }, true);
      }

      const stdout = result.stdout || "(no output)";

      if (intent && intent.trim().length > 0 && Buffer.byteLength(stdout) > INTENT_SEARCH_THRESHOLD) {
        trackIndexed(Buffer.byteLength(stdout));
        return jsonResponse("execute_file", {
          path, language, exit_code: 0, indexed: true,
          ...parseIntentResult(intentSearch(stdout, intent, `file:${path}`)),
        });
      }

      return jsonResponse("execute_file", { path, language, exit_code: 0, output: stdout });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      return jsonResponse("execute_file", { error: message, path }, true);
    }
  },
);

// ─────────────────────────────────────────────────────────
// Tool: index
// ─────────────────────────────────────────────────────────

server.registerTool(
  "index",
  {
    title: "Index",
    description: "Index content into a searchable BM25 knowledge base. Chunks by headings, stores in FTS5. Use search() to query.",
    annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: true, openWorldHint: false },
    inputSchema: z.object({
      content: z.string().optional().describe("Raw text/markdown to index. Provide this OR path."),
      path: z.string().optional().describe("File path to index server-side. Provide this OR content."),
      source: z.string().optional().describe("Label for the content (e.g., 'React docs')"),
      database: z.string().optional().describe("Named persistent DB. Omit for ephemeral storage."),
    }),
  },
  async ({ content, path, source, database }) => {
    if (!content && !path) {
      return jsonResponse("index", { error: "Either content or path must be provided" }, true);
    }

    try {
      if (content) trackIndexed(Buffer.byteLength(content));
      else if (path) {
        try {
          const fs = await import("fs");
          trackIndexed(fs.readFileSync(path).byteLength);
        } catch { /* ignore — file read errors handled by store */ }
      }
      const store = resolveStore(database);
      const result = store.index({ content, path, source });

      return jsonResponse("index", {
        source: result.label, chunks: result.totalChunks, code_chunks: result.codeChunks,
      });
    } catch (err: unknown) {
      return jsonResponse("index", { error: err instanceof Error ? err.message : String(err) }, true);
    }
  },
);

// ─────────────────────────────────────────────────────────
// Tool: search — progressive throttling
// ─────────────────────────────────────────────────────────

// Track search calls per 60-second window for progressive throttling
let searchCallCount = 0;
let searchWindowStart = Date.now();
const SEARCH_WINDOW_MS = 60_000;
const SEARCH_MAX_RESULTS_AFTER = 3; // after 3 calls: 1 result per query
const SEARCH_BLOCK_AFTER = 8; // after 8 calls: refuse, demand batching

server.registerTool(
  "search",
  {
    title: "Search",
    description: "Search indexed content. Pass ALL queries as array in ONE call. 2-4 terms per query.",
    annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
    inputSchema: z.object({
      queries: z.array(z.string()).optional().describe("Search queries. Batch all in one call."),
      limit: z.number().optional().default(3).describe("Results per query (default: 3)"),
      source: z.string().optional().describe("Filter to source (partial match)."),
      database: z.string().optional().describe("Named persistent DB. Omit for ephemeral."),
    }),
  },
  async (params) => {
    try {
      const raw = params as Record<string, unknown>;
      const store = resolveStore(raw.database as string | undefined);

      // Normalize: accept both query (string) and queries (array)
      const queryList: string[] = [];
      if (Array.isArray(raw.queries) && raw.queries.length > 0) {
        queryList.push(...(raw.queries as string[]));
      } else if (typeof raw.query === "string" && raw.query.length > 0) {
        queryList.push(raw.query as string);
      }

      if (queryList.length === 0) {
        return jsonResponse("search", { error: "Provide query or queries." }, true);
      }

      const { limit = 3, source } = params as { limit?: number; source?: string };

      // Progressive throttling: track calls in time window
      const now = Date.now();
      if (now - searchWindowStart > SEARCH_WINDOW_MS) {
        searchCallCount = 0;
        searchWindowStart = now;
      }
      searchCallCount++;

      // After SEARCH_BLOCK_AFTER calls: refuse
      if (searchCallCount > SEARCH_BLOCK_AFTER) {
        return jsonResponse("search", {
          error: "Rate limited",
          calls: searchCallCount,
          window_seconds: Math.round((now - searchWindowStart) / 1000),
          hint: "Use batch_execute(commands, queries) instead.",
        }, true);
      }

      // Determine per-query result limit based on throttle level
      const effectiveLimit = searchCallCount > SEARCH_MAX_RESULTS_AFTER
        ? 1 // after 3 calls: only 1 result per query
        : Math.min(limit, 2); // normal: max 2

      const MAX_TOTAL = 40 * 1024; // 40KB total cap
      let totalSize = 0;
      const sections: string[] = [];

      for (const q of queryList) {
        if (totalSize > MAX_TOTAL) {
          sections.push(`## ${q}\n(output cap reached)\n`);
          continue;
        }

        const results = store.searchWithFallback(q, effectiveLimit, source);

        if (results.length === 0) {
          sections.push(`## ${q}\nNo results found.`);
          continue;
        }

        const formatted = results
          .map((r, i) => {
            const header = `--- [${r.source}] ---`;
            const heading = `### ${r.title}`;
            const snippet = extractSnippet(r.content, q, 1500, r.highlighted);
            return `${header}\n${heading}\n\n${snippet}`;
          })
          .join("\n\n");

        sections.push(`## ${q}\n\n${formatted}`);
        totalSize += formatted.length;
      }

      let output = sections.join("\n\n---\n\n");

      // Add throttle warning after threshold
      if (searchCallCount >= SEARCH_MAX_RESULTS_AFTER) {
        output += `\n\n⚠ search call #${searchCallCount}/${SEARCH_BLOCK_AFTER} in this window. ` +
          `Results limited to ${effectiveLimit}/query. ` +
          `Batch queries: search(queries: ["q1","q2","q3"]) or use batch_execute.`;
      }

      if (output.trim().length === 0) {
        const sources = store.listSources();
        return jsonResponse("search", {
          results: 0, queries: queryList,
          ...(sources.length > 0 && { indexed_sources: sources.map(s => ({ label: s.label, chunks: s.chunkCount })) }),
        });
      }

      return trackResponse("search", {
        content: [{ type: "text" as const, text: output }],
      });
    } catch (err: unknown) {
      return jsonResponse("search", { error: err instanceof Error ? err.message : String(err) }, true);
    }
  },
);

// ─────────────────────────────────────────────────────────
// Tool: fetch_and_index
// ─────────────────────────────────────────────────────────

const HTML_TO_MARKDOWN_CODE = `
const url = process.argv[1];
if (!url) { console.error("No URL provided"); process.exit(1); }

async function main() {
  const resp = await fetch(url);
  if (!resp.ok) { console.error("HTTP " + resp.status); process.exit(1); }

  let html = await resp.text();

  // Strip script, style, nav, header, footer tags with content
  html = html.replace(/<script[^>]*>[\\s\\S]*?<\\/script>/gi, "");
  html = html.replace(/<style[^>]*>[\\s\\S]*?<\\/style>/gi, "");
  html = html.replace(/<nav[^>]*>[\\s\\S]*?<\\/nav>/gi, "");
  html = html.replace(/<header[^>]*>[\\s\\S]*?<\\/header>/gi, "");
  html = html.replace(/<footer[^>]*>[\\s\\S]*?<\\/footer>/gi, "");

  // Convert headings to markdown
  html = html.replace(/<h1[^>]*>(.*?)<\\/h1>/gi, "\\n# $1\\n");
  html = html.replace(/<h2[^>]*>(.*?)<\\/h2>/gi, "\\n## $1\\n");
  html = html.replace(/<h3[^>]*>(.*?)<\\/h3>/gi, "\\n### $1\\n");
  html = html.replace(/<h4[^>]*>(.*?)<\\/h4>/gi, "\\n#### $1\\n");

  // Convert code blocks
  html = html.replace(/<pre[^>]*><code[^>]*class="[^"]*language-(\\w+)"[^>]*>([\\s\\S]*?)<\\/code><\\/pre>/gi,
    (_, lang, code) => "\\n\\\`\\\`\\\`" + lang + "\\n" + decodeEntities(code) + "\\n\\\`\\\`\\\`\\n");
  html = html.replace(/<pre[^>]*><code[^>]*>([\\s\\S]*?)<\\/code><\\/pre>/gi,
    (_, code) => "\\n\\\`\\\`\\\`\\n" + decodeEntities(code) + "\\n\\\`\\\`\\\`\\n");
  html = html.replace(/<code[^>]*>([^<]*)<\\/code>/gi, "\\\`$1\\\`");

  // Convert links
  html = html.replace(/<a[^>]*href="([^"]*)"[^>]*>(.*?)<\\/a>/gi, "[$2]($1)");

  // Convert lists
  html = html.replace(/<li[^>]*>(.*?)<\\/li>/gi, "- $1\\n");

  // Convert paragraphs and line breaks
  html = html.replace(/<p[^>]*>(.*?)<\\/p>/gi, "\\n$1\\n");
  html = html.replace(/<br\\s*\\/?>/gi, "\\n");
  html = html.replace(/<hr\\s*\\/?>/gi, "\\n---\\n");

  // Strip remaining HTML tags
  html = html.replace(/<[^>]+>/g, "");

  // Decode HTML entities
  html = decodeEntities(html);

  // Clean up whitespace
  html = html.replace(/\\n{3,}/g, "\\n\\n").trim();

  console.log(html);
}

function decodeEntities(s) {
  return s
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&#x27;/g, "'")
    .replace(/&#x2F;/g, "/")
    .replace(/&nbsp;/g, " ");
}

main();
`;

server.registerTool(
  "fetch_and_index",
  {
    title: "Fetch URL",
    description: "Fetch URL, convert HTML to markdown, index into knowledge base. Returns ~3KB preview. Use search() for full content.",
    annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: true, openWorldHint: true },
    inputSchema: z.object({
      url: z.string().describe("URL to fetch"),
      source: z.string().optional().describe("Label for the content (e.g., 'React docs')"),
    }),
  },
  async ({ url, source }) => {
    try {
      // Execute fetch inside subprocess — raw HTML never enters context
      const fetchCode = `process.argv[1] = ${JSON.stringify(url)};\n${HTML_TO_MARKDOWN_CODE}`;
      const result = await executor.execute({
        language: "javascript",
        code: fetchCode,
        timeout: 30_000,
      });

      if (result.exitCode !== 0) {
        return jsonResponse("fetch_and_index", { error: `Failed to fetch ${url}`, detail: result.stderr || result.stdout }, true);
      }

      if (!result.stdout || result.stdout.trim().length === 0) {
        return jsonResponse("fetch_and_index", { error: "Empty content after HTML conversion", url }, true);
      }

      // Index the markdown into FTS5
      const store = getStore();
      const markdown = result.stdout.trim();
      trackIndexed(Buffer.byteLength(markdown));
      const indexed = store.index({ content: markdown, source: source ?? url });

      // Build preview — first ~3KB of markdown for immediate use
      const PREVIEW_LIMIT = 3072;
      const preview = markdown.length > PREVIEW_LIMIT
        ? markdown.slice(0, PREVIEW_LIMIT) + "\n\n…[truncated — use search() for full content]"
        : markdown;

      return jsonResponse("fetch_and_index", {
        url, source: indexed.label, chunks: indexed.totalChunks,
        size_kb: +(Buffer.byteLength(markdown) / 1024).toFixed(1),
        preview,
      });
    } catch (err: unknown) {
      return jsonResponse("fetch_and_index", { error: err instanceof Error ? err.message : String(err) }, true);
    }
  },
);

// ─────────────────────────────────────────────────────────
// Tool: batch_execute
// ─────────────────────────────────────────────────────────

server.registerTool(
  "batch_execute",
  {
    title: "Batch Run",
    description: "Execute multiple commands in ONE call, auto-index output, search with queries. Returns results directly — no follow-up needed.",
    annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: false, openWorldHint: true },
    inputSchema: z.object({
      commands: z.array(z.object({
        label: z.string().describe("Section header for output"),
        command: z.string().describe("Shell command to execute"),
      })).min(1).describe("Commands to run sequentially."),
      queries: z.array(z.string()).min(1).describe("Search queries for indexed output. Put ALL questions here."),
      timeout: z.number().optional().default(60000).describe("Max execution time in ms (default: 60s)"),
      database: z.string().optional().describe("Named persistent DB. Omit for ephemeral."),
    }),
  },
  async ({ commands, queries, timeout, database }) => {
    try {
      // Safety: block dangerous gt commands in any batch command
      for (const c of commands) {
        const blocked = checkBlockedCommand(c.command);
        if (blocked) {
          return jsonResponse("batch_execute", {
            error: `Blocked: "${blocked}" in command "${c.label}" is a protected Gas Town operation and cannot be run through context-mode.`,
          }, true);
        }
      }

      // Build batch script with markdown section headers for proper chunking
      const script = commands
        .map((c) => {
          const safeLabel = c.label.replace(/'/g, "'\\''");
          return `echo '# ${safeLabel}'\necho ''\n${c.command} 2>&1\necho ''`;
        })
        .join("\n");

      const result = await executor.execute({
        language: "shell",
        code: script,
        timeout,
      });

      if (result.timedOut) {
        return jsonResponse("batch_execute", {
          error: `Timed out after ${timeout}ms`,
          partial_output: result.stdout?.slice(0, 2000) || "(none)",
        }, true);
      }

      const stdout = result.stdout || "(no output)";
      const totalBytes = Buffer.byteLength(stdout);
      const totalLines = stdout.split("\n").length;

      // Track indexed bytes (raw data that stays in sandbox)
      trackIndexed(totalBytes);

      // Index into knowledge base — markdown heading chunking splits by # labels
      const store = resolveStore(database);
      const source = `batch:${commands
        .map((c) => c.label)
        .join(",")
        .slice(0, 80)}`;
      const indexed = store.index({ content: stdout, source });

      // Build section inventory — direct query by source_id (no FTS5 MATCH needed)
      const allSections = store.getChunksBySource(indexed.sourceId);
      const inventory: string[] = ["## Indexed Sections", ""];
      const sectionTitles: string[] = [];
      for (const s of allSections) {
        const bytes = Buffer.byteLength(s.content);
        inventory.push(`- ${s.title} (${(bytes / 1024).toFixed(1)}KB)`);
        sectionTitles.push(s.title);
      }

      // Run all search queries — 3 results each, smart snippets
      // Three-tier fallback: scoped → boosted → global
      const MAX_OUTPUT = 80 * 1024; // 80KB total output cap
      const queryResults: string[] = [];
      let outputSize = 0;

      for (const query of queries) {
        if (outputSize > MAX_OUTPUT) {
          queryResults.push(`## ${query}\n(output cap reached — use search(queries: ["${query}"]) for details)\n`);
          continue;
        }

        // Tier 1: scoped search with fallback (porter → trigram → fuzzy)
        let results = store.searchWithFallback(query, 3, source);

        // Tier 2: global fallback (no source filter)
        if (results.length === 0) {
          results = store.searchWithFallback(query, 3);
        }

        queryResults.push(`## ${query}`);
        queryResults.push("");
        if (results.length > 0) {
          for (const r of results) {
            const snippet = extractSnippet(r.content, query, 1500, r.highlighted);
            queryResults.push(`### ${r.title}`);
            queryResults.push(snippet);
            queryResults.push("");
            outputSize += snippet.length + r.title.length;
          }
        } else {
          queryResults.push("No matching sections found.");
          queryResults.push("");
        }
      }

      // Get searchable terms for edge cases where follow-up is needed
      const distinctiveTerms = store.getDistinctiveTerms
        ? store.getDistinctiveTerms(indexed.sourceId)
        : [];

      const output = [
        `Executed ${commands.length} commands (${totalLines} lines, ${(totalBytes / 1024).toFixed(1)}KB). ` +
          `Indexed ${indexed.totalChunks} sections. Searched ${queries.length} queries.`,
        "",
        ...inventory,
        "",
        ...queryResults,
        distinctiveTerms.length > 0
          ? `\nSearchable terms for follow-up: ${distinctiveTerms.join(", ")}`
          : "",
      ].join("\n");

      return trackResponse("batch_execute", {
        content: [{ type: "text" as const, text: output }],
      });
    } catch (err: unknown) {
      return jsonResponse("batch_execute", { error: err instanceof Error ? err.message : String(err) }, true);
    }
  },
);

// ─────────────────────────────────────────────────────────
// Tool: list_databases
// ─────────────────────────────────────────────────────────

server.registerTool(
  "list_databases",
  {
    title: "List KBs",
    description: "List persistent knowledge bases with names and sizes.",
    annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
    inputSchema: z.object({}),
  },
  async () => {
    try {
      const dbs = ContentStore.listPersistent();
      const kb = (b: number) => b < 1024 ? `${b}B` : b < 1048576 ? `${(b / 1024).toFixed(1)}KB` : `${(b / 1048576).toFixed(1)}MB`;
      return jsonResponse("list_databases", {
        databases: dbs.map(db => ({ name: db.name, size: kb(db.sizeBytes) })),
        path: ContentStore.persistentDir,
      });
    } catch (err: unknown) {
      return jsonResponse("list_databases", { error: err instanceof Error ? err.message : String(err) }, true);
    }
  },
);

// ─────────────────────────────────────────────────────────
// Tool: delete_database
// ─────────────────────────────────────────────────────────

server.registerTool(
  "delete_database",
  {
    title: "Delete KB",
    description: "Delete a named persistent knowledge base.",
    annotations: { readOnlyHint: false, destructiveHint: true, idempotentHint: false, openWorldHint: false },
    inputSchema: z.object({
      name: z.string().describe("Database name to delete"),
    }),
  },
  async ({ name }) => {
    try {
      const cached = _namedStores.get(name);
      if (cached) {
        cached.cleanup();
        _namedStores.delete(name);
      }
      const deleted = ContentStore.deletePersistent(name);
      return jsonResponse("delete_database", { name, deleted });
    } catch (err: unknown) {
      return jsonResponse("delete_database", { error: err instanceof Error ? err.message : String(err) }, true);
    }
  },
);

// ─────────────────────────────────────────────────────────
// Tool: list_peers
// ─────────────────────────────────────────────────────────

/** Discover peer context-mode DBs in /tmp/. Returns paths excluding own PID. */
function discoverPeerDbs(): string[] {
  const ownFile = `context-mode-${process.pid}.db`;
  try {
    return readdirSync(tmpdir())
      .filter(f => f.startsWith("context-mode-") && f.endsWith(".db")
        && !f.endsWith("-wal") && !f.endsWith("-shm") && f !== ownFile)
      .map(f => join(tmpdir(), f));
  } catch { return []; }
}

/** Extract PID from a context-mode DB filename. */
function pidFromDbPath(dbPath: string): number {
  const m = dbPath.match(/context-mode-(\d+)\.db$/);
  return m ? Number(m[1]) : 0;
}

/** Resolve PID to a human-readable role label via /proc/{pid}/environ (Linux only). */
function roleForPid(pid: number): string {
  try {
    const raw = readFileSync(`/proc/${pid}/environ`, "utf-8");
    const env = Object.fromEntries(
      raw.split("\0").filter(Boolean).map(e => {
        const i = e.indexOf("=");
        return i > 0 ? [e.slice(0, i), e.slice(i + 1)] : [e, ""];
      }),
    );
    // GT_ROLE gives the full path like "sfgastown/polecats/jade" or "mayor"
    const role = env.GT_ROLE;
    const agent = env.GT_AGENT;
    if (role) {
      // Extract the meaningful part: "mayor", "polecat/jade", "refinery", "witness", "deacon", "crew/forge"
      const parts = role.split("/");
      // Patterns: "mayor", "deacon", "rig/refinery", "rig/witness", "rig/polecats/name", "rig/crew/name"
      let label: string;
      if (parts.includes("polecats")) {
        const name = parts[parts.indexOf("polecats") + 1];
        label = name ? `polecat/${name}` : "polecat";
      } else if (parts.includes("crew")) {
        const name = parts[parts.indexOf("crew") + 1];
        label = name ? `crew/${name}` : "crew";
      } else {
        // Last meaningful segment: "mayor", "deacon", "refinery", "witness"
        label = parts[parts.length - 1];
      }
      return agent ? `${label} (${agent})` : label;
    }
    return agent ? `unknown (${agent})` : `PID ${pid}`;
  } catch {
    return `PID ${pid}`;
  }
}

server.registerTool(
  "list_peers",
  {
    title: "List Peers",
    description: "Discover other running context-mode agents and their indexed knowledge. Use before search_peers.",
    annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
    inputSchema: z.object({}),
  },
  async () => {
    try {
      const peers = discoverPeerDbs();
      if (peers.length === 0) {
        return jsonResponse("list_peers", { peers: [] });
      }

      const Database = loadDatabase();
      const peerList: Array<Record<string, unknown>> = [];

      for (const dbPath of peers) {
        const pid = pidFromDbPath(dbPath);
        try {
          const db = new Database(dbPath, { readonly: true, timeout: 1000 });
          db.pragma("journal_mode = WAL");
          const sources = db.prepare(
            "SELECT label, chunk_count FROM sources ORDER BY id DESC"
          ).all() as Array<{ label: string; chunk_count: number }>;
          db.close();

          const totalChunks = sources.reduce((s, r) => s + r.chunk_count, 0);
          peerList.push({
            role: roleForPid(pid), pid, chunks: totalChunks,
            sources: sources.slice(0, 10).map(s => `${s.label} (${s.chunk_count})`),
          });
        } catch { /* DB cleaned up */ }
      }

      return jsonResponse("list_peers", { peers: peerList });
    } catch (err: unknown) {
      return jsonResponse("list_peers", { error: err instanceof Error ? err.message : String(err) }, true);
    }
  },
);

// ─────────────────────────────────────────────────────────
// Tool: search_peers
// ─────────────────────────────────────────────────────────

server.registerTool(
  "search_peers",
  {
    title: "Search Peers",
    description: "Search other agents' indexed knowledge. Read-only. Use list_peers first.",
    annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
    inputSchema: z.object({
      queries: z.array(z.string()).min(1).describe("Search queries (2-4 terms each)."),
      source: z.string().optional().describe("Filter by source substring."),
      limit: z.number().optional().default(5).describe("Max results per query (default 5)"),
    }),
  },
  async ({ queries, source, limit }) => {
    try {
      const peers = discoverPeerDbs();
      if (peers.length === 0) {
        return jsonResponse("search_peers", { results: [], peers: 0 });
      }

      const Database = loadDatabase();
      const allResults: Array<{ title: string; content: string; source: string; rank: number; peer_pid: number; peer_role: string }> = [];

      for (const dbPath of peers) {
        const pid = pidFromDbPath(dbPath);
        const role = roleForPid(pid);
        let db;
        try {
          db = new Database(dbPath, { readonly: true, timeout: 1000 });
          db.pragma("journal_mode = WAL");

          const sourceFilter = source ? "AND sources.label LIKE ?" : "";
          const stmt = db.prepare(`
            SELECT
              chunks.title,
              chunks.content,
              sources.label,
              bm25(chunks, 2.0, 1.0) AS rank
            FROM chunks
            JOIN sources ON sources.id = chunks.source_id
            WHERE chunks MATCH ? ${sourceFilter}
            ORDER BY rank
            LIMIT ?
          `);

          for (const query of queries) {
            // Sanitize: remove FTS5 special chars
            const sanitized = query.replace(/[*"():^~{}<>]/g, " ").trim();
            if (!sanitized) continue;
            try {
              const params = source
                ? [sanitized, `%${source}%`, limit]
                : [sanitized, limit];
              const rows = stmt.all(...params) as Array<{
                title: string; content: string; label: string; rank: number;
              }>;
              for (const r of rows) {
                allResults.push({
                  title: r.title,
                  content: r.content,
                  source: r.label,
                  rank: r.rank,
                  peer_pid: pid,
                  peer_role: role,
                });
              }
            } catch { /* query may fail on empty/corrupt DB — skip */ }
          }

          db.close();
        } catch {
          // DB unavailable — skip
          try { db?.close(); } catch { /* ignore */ }
        }
      }

      if (allResults.length === 0) {
        return jsonResponse("search_peers", { results: [], queries });
      }

      // Sort by BM25 rank (lower = better), deduplicate by content, limit total
      const seen = new Set<string>();
      const unique = allResults
        .sort((a, b) => a.rank - b.rank)
        .filter(r => {
          const key = r.content.slice(0, 200);
          if (seen.has(key)) return false;
          seen.add(key);
          return true;
        })
        .slice(0, limit * queries.length);

      return jsonResponse("search_peers", {
        results: unique.map(r => ({
          title: r.title, source: r.source,
          from: r.peer_role, pid: r.peer_pid,
          preview: r.content.slice(0, 1500),
        })),
        queries,
      });
    } catch (err: unknown) {
      return jsonResponse("search_peers", { error: err instanceof Error ? err.message : String(err) }, true);
    }
  },
);

// ─────────────────────────────────────────────────────────
// Tool: stats
// ─────────────────────────────────────────────────────────

server.registerTool(
  "stats",
  {
    title: "Stats",
    description: "Context consumption statistics for the current session.",
    annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
    inputSchema: z.object({}),
  },
  async () => {
    const totalBytesReturned = Object.values(sessionStats.bytesReturned).reduce((sum, b) => sum + b, 0);
    const totalCalls = Object.values(sessionStats.calls).reduce((sum, c) => sum + c, 0);
    const uptimeMs = Date.now() - sessionStats.sessionStart;
    const keptOut = sessionStats.bytesIndexed + sessionStats.bytesSandboxed;
    const totalProcessed = keptOut + totalBytesReturned;
    const reductionPct = totalProcessed > 0
      ? +((1 - totalBytesReturned / totalProcessed) * 100).toFixed(0)
      : 0;

    const fmtKb = (b: number) => b >= 1024 * 1024 ? `${(b / 1024 / 1024).toFixed(1)}MB` : `${(b / 1024).toFixed(1)}KB`;

    const toolBreakdown: Record<string, { calls: number; context: string; tokens: number }> = {};
    const toolNames = new Set([...Object.keys(sessionStats.calls), ...Object.keys(sessionStats.bytesReturned)]);
    for (const tool of Array.from(toolNames).sort()) {
      const calls = sessionStats.calls[tool] || 0;
      const bytes = sessionStats.bytesReturned[tool] || 0;
      toolBreakdown[tool] = { calls, context: fmtKb(bytes), tokens: Math.round(bytes / 4) };
    }

    return jsonResponse("stats", {
      session_minutes: +(uptimeMs / 60_000).toFixed(1),
      total_calls: totalCalls,
      data_processed: fmtKb(totalProcessed),
      kept_in_sandbox: fmtKb(keptOut),
      entered_context: fmtKb(totalBytesReturned),
      tokens_consumed: Math.round(totalBytesReturned / 4),
      tokens_saved: Math.round(keptOut / 4),
      context_savings: `${(totalProcessed / Math.max(totalBytesReturned, 1)).toFixed(1)}x (${reductionPct}% reduction)`,
      tools: toolBreakdown,
    });
  },
);

// ─────────────────────────────────────────────────────────
// Server startup
// ─────────────────────────────────────────────────────────

async function main() {
  // Clean up stale DB files from previous sessions
  const cleaned = cleanupStaleDBs();
  if (cleaned > 0) {
    console.error(`Cleaned up ${cleaned} stale DB file(s) from previous sessions`);
  }

  // Clean up own DB on shutdown (persistent stores are only closed, not deleted)
  const shutdown = () => {
    if (_store) _store.cleanup();
    for (const store of _namedStores.values()) store.cleanup();
    _namedStores.clear();
  };
  process.on("exit", shutdown);
  process.on("SIGINT", () => { shutdown(); process.exit(0); });
  process.on("SIGTERM", () => { shutdown(); process.exit(0); });

  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error(`Context Mode MCP server v${VERSION} running on stdio`);
  console.error(`Detected runtimes:\n${getRuntimeSummary(runtimes)}`);
  if (!hasBunRuntime()) {
    console.error(
      "\nPerformance tip: Install Bun for 3-5x faster JS/TS execution",
    );
    console.error("  curl -fsSL https://bun.sh/install | bash");
  }
}

main().catch((err) => {
  console.error("Fatal:", err);
  process.exit(1);
});
