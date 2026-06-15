/**
 * Nebius Token Factory — pi extension
 *
 * Fetches the current model catalog from the Token Factory API on startup
 * and registers all tool-capable text-generation models as a "nebius" provider.
 *
 * Environment:
 *   NEBIUS_API_KEY — required, Token Factory API key
 *
 * Usage:
 *   pi -e /path/to/tokenfactory-pi
 *   pi -e /path/to/tokenfactory-pi --provider nebius
 *   pi -e /path/to/tokenfactory-pi --provider nebius --model Qwen/Qwen3-32B
 */

import { gunzipSync } from "node:zlib";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const PROVIDER_NAME = "nebius";
const BASE_URL = "https://api.tokenfactory.nebius.com/v1";
const ENV_VAR = "NEBIUS_API_KEY";
const API_KEY_CONFIG = `$${ENV_VAR}`;

// ============================================================================
// Token Factory API types
// ============================================================================

interface TokenFactoryModel {
	id: string;
	name?: string;
	context_length?: number;
	supported_features?: string[];
	architecture?: { modality?: string };
	pricing?: { prompt?: string; completion?: string };
}

interface TokenFactoryResponse {
	data: TokenFactoryModel[];
}

// ============================================================================
// Helpers
// ============================================================================

function isGzip(bytes: Buffer): boolean {
	return bytes.length >= 2 && bytes[0] === 0x1f && bytes[1] === 0x8b;
}

async function readTokenFactoryResponse(res: Response): Promise<TokenFactoryResponse> {
	const bytes = Buffer.from(await res.arrayBuffer());
	const body = (isGzip(bytes) ? gunzipSync(bytes) : bytes).toString("utf8");
	try {
		return JSON.parse(body) as TokenFactoryResponse;
	} catch {
		const preview = body.slice(0, 200).replace(/\s+/g, " ");
		throw new Error(
			`Invalid Token Factory JSON response (${res.status} ${res.statusText}, ` +
				`content-type=${res.headers.get("content-type") || "unknown"}, ` +
				`content-encoding=${res.headers.get("content-encoding") || "none"}): ${preview}`,
		);
	}
}

function isToolCapableTextModel(m: TokenFactoryModel): boolean {
	const features = m.supported_features || [];
	const modality = m.architecture?.modality || "";
	return features.includes("tools") && modality.includes("->text");
}

function parseInputModalities(modality: string): ("text" | "image")[] {
	const input: ("text" | "image")[] = ["text"];
	if (modality.includes("image")) input.push("image");
	return input;
}

function parseCostPerMillion(raw: string | undefined): number {
	const parsed = parseFloat(raw || "0");
	return Number.isNaN(parsed) ? 0 : parsed * 1_000_000;
}

function isReasoningModel(m: TokenFactoryModel): boolean {
	const features = m.supported_features || [];
	return features.includes("reasoning") || /(-R1|-Thinking|QwQ)/.test(m.id);
}

// ============================================================================
// Extension entry point
// ============================================================================

export default async function (pi: ExtensionAPI) {
	const apiKey = process.env[ENV_VAR];
	if (!apiKey) {
		return;
	}

	let response: TokenFactoryResponse;
	try {
		const res = await fetch(`${BASE_URL}/models?verbose=true`, {
			headers: {
				Authorization: `Bearer ${apiKey}`,
				"Accept-Encoding": "identity",
			},
		});
		if (!res.ok) {
			console.warn(`[${PROVIDER_NAME}] API returned ${res.status}: ${res.statusText}`);
			return;
		}
		response = await readTokenFactoryResponse(res);
	} catch (error) {
		console.warn(`[${PROVIDER_NAME}] Failed to fetch models:`, error);
		return;
	}

	if (!Array.isArray(response.data)) {
		console.warn(`[${PROVIDER_NAME}] Unexpected API response shape`);
		return;
	}

	const models: Array<{
		id: string;
		name: string;
		reasoning: boolean;
		input: ("text" | "image")[];
		cost: { input: number; output: number; cacheRead: number; cacheWrite: number };
		contextWindow: number;
		maxTokens: number;
		compat: { supportsDeveloperRole: boolean; maxTokensField: "max_tokens" };
	}> = [];
	for (const m of response.data) {
		if (!m.id || m.id.trim() === "") continue;
		if (!isToolCapableTextModel(m)) continue;

		const modality = m.architecture?.modality || "";
		const contextLength = m.context_length && m.context_length > 0 ? m.context_length : 131072;

		models.push({
			id: m.id,
			name: m.name || m.id,
			reasoning: isReasoningModel(m),
			input: parseInputModalities(modality),
			cost: {
				input: parseCostPerMillion(m.pricing?.prompt),
				output: parseCostPerMillion(m.pricing?.completion),
				cacheRead: 0,
				cacheWrite: 0,
			},
			contextWindow: contextLength,
			maxTokens: Math.min(contextLength, 32768),
			compat: {
				supportsDeveloperRole: false,
				maxTokensField: "max_tokens" as const,
			},
		});
	}

	pi.registerProvider(PROVIDER_NAME, {
		baseUrl: BASE_URL,
		apiKey: API_KEY_CONFIG,
		api: "openai-completions",
		headers: {
			"Accept-Encoding": "identity",
		},
		models,
	});

	// /nebius-models command to list available models
	pi.registerCommand("nebius-models", {
		description: "List available Nebius Token Factory models",
		handler: async (_args, ctx) => {
			if (models.length === 0) {
				ctx.ui.notify("No Nebius models available", "warning");
				return;
			}
			const items = models
				.sort((a, b) => a.id.localeCompare(b.id))
				.map((m) => {
					const tags = [];
					if (m.reasoning) tags.push("reasoning");
					if (m.input.includes("image")) tags.push("vision");
					const suffix = tags.length > 0 ? ` (${tags.join(", ")})` : "";
					return `${m.id}${suffix}`;
				});
			await ctx.ui.select(`Nebius Token Factory — ${models.length} models`, items);
		},
	});
}
