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

import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";

const PROVIDER_NAME = "nebius";
const BASE_URL = "https://api.tokenfactory.nebius.com/v1";
const ENV_VAR = "NEBIUS_API_KEY";

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
	return parseFloat(raw || "0") * 1_000_000;
}

function isReasoningModel(id: string): boolean {
	return /(-R1|-Thinking|QwQ)/.test(id);
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
			headers: { Authorization: `Bearer ${apiKey}` },
		});
		if (!res.ok) {
			console.warn(`[${PROVIDER_NAME}] API returned ${res.status}: ${res.statusText}`);
			return;
		}
		response = (await res.json()) as TokenFactoryResponse;
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
		if (!isToolCapableTextModel(m)) continue;

		const modality = m.architecture?.modality || "";

		models.push({
			id: m.id,
			name: m.name || m.id,
			reasoning: isReasoningModel(m.id),
			input: parseInputModalities(modality),
			cost: {
				input: parseCostPerMillion(m.pricing?.prompt),
				output: parseCostPerMillion(m.pricing?.completion),
				cacheRead: 0,
				cacheWrite: 0,
			},
			contextWindow: m.context_length || 131072,
			maxTokens: Math.min(m.context_length || 32768, 32768),
			compat: {
				supportsDeveloperRole: false,
				maxTokensField: "max_tokens" as const,
			},
		});
	}

	pi.registerProvider(PROVIDER_NAME, {
		baseUrl: BASE_URL,
		apiKey: ENV_VAR,
		api: "openai-completions",
		models,
	});

	// /nebius-models command to list and select a model
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
