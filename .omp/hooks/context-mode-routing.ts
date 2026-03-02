/**
 * Context-Mode Tool Routing Hook
 *
 * Fires on tool_call events and sends a steering message when the agent
 * is about to use shell/file tools for operations that would benefit from
 * context-mode's sandboxed execution. Does NOT block — just reminds.
 */
import type { HookAPI } from "@oh-my-pi/pi-coding-agent";

// Commands that are safe to run directly (small output, mutations, navigation)
const BASH_ALLOWLIST = /^(echo|printf|pwd|cd|mkdir|cp|mv|rm|touch|chmod|chown|git\s+(add|commit|push|pull|checkout|branch|merge|rebase|stash|tag|remote|init|clone)|npm\s+(install|ci|publish)|bun\s+(install|add)|pip\s+install|cargo\s+build|make\b|which|type|date|whoami|hostname|export|source|kill|pkill)\b/;


// Commands likely to produce large output
const LARGE_OUTPUT_PATTERNS = [
	/\b(git\s+(log|diff|show|shortlog|blame))\b/,
	/\b(npm\s+test|npx\s+jest|pytest|cargo\s+test|bun\s+test)\b/,
	/\bcat\s+/,
	/\bhead\s+/,
	/\btail\s+/,
	/\bcurl\s+/,
	/\bwget\s+/,
	/\bfind\s+/,
	/\bls\s+(-la?|--all|-R)/,
	/\bgrep\s+/,
	/\brg\s+/,
	/\bgh\s+(pr|issue|api|run)\b/,
	/\bdocker\s+(ps|logs|inspect)\b/,
	/\bkubectl\s+(get|describe|logs)\b/,
	/\bbd\s+(search)\b/,
];

export default function (pi: HookAPI) {
	pi.on("tool_call", async (event) => {
		if (event.toolName !== "bash") return undefined;

		const command = String(event.input.command ?? "").trim();
		if (!command) return undefined;

		// Skip if it's on the allowlist (small/mutation commands)
		if (BASH_ALLOWLIST.test(command)) return undefined;

		// Check if it matches large-output patterns
		const isLikelyLarge = LARGE_OUTPUT_PATTERNS.some((p) => p.test(command));

		if (isLikelyLarge) {
			pi.sendMessage(
				{
					customType: "context-mode-tip",
					content: `CONTEXT TIP: This command may produce large output. Consider using context-mode execute(language: "shell", code: "${command.replace(/"/g, '\\"').slice(0, 100)}") instead — output stays in sandbox, only your summary enters context.`,
					display: false,
				},
				{ deliverAs: "steer" },
			);
		}

		return undefined;
	});

	// Also intercept file reads
	pi.on("tool_call", async (event) => {
		if (event.toolName !== "read" && event.toolName !== "Read") return undefined;

		pi.sendMessage(
			{
				customType: "context-mode-tip",
				content:
					"CONTEXT TIP: If this file is large (>50 lines), prefer context-mode execute_file(path, language, code) — processes in sandbox, only stdout enters context.",
				display: false,
			},
			{ deliverAs: "steer" },
		);

		return undefined;
	});
}
