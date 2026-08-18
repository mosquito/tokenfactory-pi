/**
 * Nebius Token Factory — pi extension
 *
 * Fetches the current model catalog from the Token Factory API on startup
 * and registers all tool-capable text-generation models as a "nebius" provider.
 *
 * Environment:
 *   NEBIUS_API_KEY — required, Token Factory API key
 *   NEBIUS_MODEL_CONTEXT — optional, context-window override in tokens
 *
 * Usage:
 *   pi -e /path/to/tokenfactory-pi
 *   pi -e /path/to/tokenfactory-pi --provider nebius
 *   pi -e /path/to/tokenfactory-pi --provider nebius --model Qwen/Qwen3-32B
 */

import { gunzipSync } from "node:zlib";
import {
	type ExtensionAPI,
	type ExtensionCommandContext,
	type ProviderModelConfig,
} from "@earendil-works/pi-coding-agent";
import {
	getDefaultModelOverridesPath,
	isPositiveInteger,
	isTemperature,
	loadModelOverrides,
	REASONING_EFFORTS,
	type ModelOverrides,
	type ModelParameterOverride,
	type ReasoningEffort,
	type SupportedFeatureOverrides,
	UnsafeModelOverridesFileError,
	updateModelOverrides,
} from "./model-overrides.js";

const PROVIDER_NAME = "nebius";
const BASE_URL = "https://api.tokenfactory.nebius.com/v1";
const ENV_VAR = "NEBIUS_API_KEY";
const API_KEY_CONFIG = `$${ENV_VAR}`;
const MODEL_CONTEXT_ENV_VAR = "NEBIUS_MODEL_CONTEXT";
const DEFAULT_CONTEXT_WINDOW = 131_072;
const DEFAULT_MAX_TOKENS = 32_768;

const EDIT_CONTEXT_ACTION = "Edit context window";
const EDIT_MAX_TOKENS_ACTION = "Edit max output tokens";
const EDIT_FEATURES_ACTION = "Edit supported features";
const EDIT_REASONING_ACTION = "Edit reasoning effort";
const EDIT_TEMPERATURE_ACTION = "Edit temperature";
const RESET_ACTION = "Reset saved overrides";
const BACK_ACTION = "Back to model list";
const BACK = "Back";
const INHERIT = "Inherit catalog/default";
const ENABLED = "Enabled";
const DISABLED = "Disabled";

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

export interface TokenFactoryExtensionOptions {
	configPath?: string;
	environment?: NodeJS.ProcessEnv;
	fetchImpl?: typeof fetch;
}

type SupportedFeature = keyof SupportedFeatureOverrides;

// ============================================================================
// Helpers
// ============================================================================

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

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

function isTextOutputModel(m: TokenFactoryModel): boolean {
	return (m.architecture?.modality || "").includes("->text");
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

function catalogFeatureValue(m: TokenFactoryModel, feature: SupportedFeature): boolean {
	const features = m.supported_features || [];
	if (feature === "reasoning") {
		return features.includes("reasoning") || /(-R1|-Thinking|QwQ)/.test(m.id);
	}
	return features.includes(feature);
}

function resolveFeatureValue(
	m: TokenFactoryModel,
	modelOverride: ModelParameterOverride | undefined,
	feature: SupportedFeature,
): boolean {
	return modelOverride?.supportedFeatures?.[feature] ?? catalogFeatureValue(m, feature);
}

function readModelContextOverride(environment: NodeJS.ProcessEnv): number | undefined {
	const raw = environment[MODEL_CONTEXT_ENV_VAR];
	if (raw === undefined || raw.trim() === "") return undefined;

	const parsed = Number(raw);
	if (!isPositiveInteger(parsed)) {
		console.warn(
			`[${PROVIDER_NAME}] Ignoring invalid ${MODEL_CONTEXT_ENV_VAR}; expected a positive integer token count`,
		);
		return undefined;
	}

	return parsed;
}

function resolveModelLimits(
	m: TokenFactoryModel,
	contextOverride: number | undefined,
	modelOverride: ModelParameterOverride | undefined,
): { contextWindow: number; maxTokens: number } {
	const reportedContextWindow = isPositiveInteger(m.context_length)
		? m.context_length
		: DEFAULT_CONTEXT_WINDOW;
	const contextWindow = modelOverride?.contextWindow ?? contextOverride ?? reportedContextWindow;
	const requestedMaxTokens = modelOverride?.maxTokens ?? DEFAULT_MAX_TOKENS;

	return {
		contextWindow,
		maxTokens: Math.min(contextWindow, requestedMaxTokens),
	};
}

function buildProviderModel(
	m: TokenFactoryModel,
	contextOverride: number | undefined,
	modelOverride: ModelParameterOverride | undefined,
): ProviderModelConfig {
	const modality = m.architecture?.modality || "";
	const { contextWindow, maxTokens } = resolveModelLimits(m, contextOverride, modelOverride);

	return {
		id: m.id,
		name: m.name || m.id,
		reasoning: resolveFeatureValue(m, modelOverride, "reasoning"),
		input: parseInputModalities(modality),
		cost: {
			input: parseCostPerMillion(m.pricing?.prompt),
			output: parseCostPerMillion(m.pricing?.completion),
			cacheRead: 0,
			cacheWrite: 0,
		},
		contextWindow,
		maxTokens,
		compat: {
			supportsDeveloperRole: false,
			maxTokensField: "max_tokens" as const,
		},
	};
}

function buildProviderModels(
	catalog: TokenFactoryModel[],
	contextOverride: number | undefined,
	modelOverrides: ModelOverrides,
): ProviderModelConfig[] {
	const models: ProviderModelConfig[] = [];
	for (const m of catalog) {
		if (!m.id || m.id.trim() === "" || !isTextOutputModel(m)) continue;
		const modelOverride = modelOverrides.get(m.id);
		if (!resolveFeatureValue(m, modelOverride, "tools")) continue;
		models.push(buildProviderModel(m, contextOverride, modelOverride));
	}
	return models;
}

function registerTokenFactoryProvider(pi: ExtensionAPI, models: ProviderModelConfig[]): void {
	if (models.length === 0) {
		pi.unregisterProvider(PROVIDER_NAME);
		return;
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
}

function formatTokens(value: number): string {
	return value.toLocaleString("en-US");
}

function errorMessage(error: unknown): string {
	return error instanceof Error ? error.message : String(error);
}

function hasOverrideValues(value: ModelParameterOverride): boolean {
	return (
		value.contextWindow !== undefined ||
		value.maxTokens !== undefined ||
		value.supportedFeatures !== undefined ||
		value.reasoningEffort !== undefined ||
		value.temperature !== undefined
	);
}

function mutateModelOverride(
	overrides: ModelOverrides,
	modelId: string,
	mutate: (modelOverride: ModelParameterOverride) => void,
): void {
	const current = overrides.get(modelId);
	const next: ModelParameterOverride = {
		...current,
		supportedFeatures: current?.supportedFeatures
			? { ...current.supportedFeatures }
			: undefined,
	};
	mutate(next);
	if (next.supportedFeatures && Object.keys(next.supportedFeatures).length === 0) {
		delete next.supportedFeatures;
	}
	if (hasOverrideValues(next)) overrides.set(modelId, next);
	else overrides.delete(modelId);
}

async function promptPositiveInteger(
	ctx: ExtensionCommandContext,
	title: string,
	maximum?: number,
): Promise<number | undefined> {
	while (true) {
		const raw = await ctx.ui.input(title);
		if (raw === undefined) return undefined;
		const value = Number(raw.trim());
		if (!isPositiveInteger(value)) {
			ctx.ui.notify("Enter a positive integer token count", "warning");
			continue;
		}
		if (maximum !== undefined && value > maximum) {
			ctx.ui.notify(`Value cannot exceed the ${formatTokens(maximum)}-token context window`, "warning");
			continue;
		}
		return value;
	}
}

function featureStateLabel(
	m: TokenFactoryModel,
	modelOverride: ModelParameterOverride | undefined,
	feature: SupportedFeature,
): string {
	const forced = modelOverride?.supportedFeatures?.[feature];
	const effective = resolveFeatureValue(m, modelOverride, feature) ? "enabled" : "disabled";
	return forced === undefined ? `inherit (${effective})` : `forced ${effective}`;
}

// ============================================================================
// Extension entry point
// ============================================================================

export function createTokenFactoryExtension(options: TokenFactoryExtensionOptions = {}) {
	return async function tokenFactoryExtension(pi: ExtensionAPI): Promise<void> {
		const environment = options.environment ?? process.env;
		const apiKey = environment[ENV_VAR];
		if (!apiKey) return;

		let response: TokenFactoryResponse;
		try {
			const fetchImpl = options.fetchImpl ?? globalThis.fetch;
			const res = await fetchImpl(`${BASE_URL}/models?verbose=true`, {
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

		const catalog = response.data.filter(
			(model) => model.id && model.id.trim() !== "" && isTextOutputModel(model),
		);
		const configPath = options.configPath ?? getDefaultModelOverridesPath();
		const contextOverride = readModelContextOverride(environment);
		const loadedOverrides = await loadModelOverrides(configPath);
		let modelOverrides = loadedOverrides.overrides;
		let overrideFileWritable = loadedOverrides.writable;
		let models = buildProviderModels(catalog, contextOverride, modelOverrides);

		const refreshProvider = () => {
			models = buildProviderModels(catalog, contextOverride, modelOverrides);
			registerTokenFactoryProvider(pi, models);
		};

		const saveMutation = async (
			mutate: (overrides: ModelOverrides) => void,
			ctx: ExtensionCommandContext,
			refresh: boolean,
		): Promise<boolean> => {
			let overwriteInvalid: { revision?: string } | undefined;
			while (true) {
				try {
					modelOverrides = await updateModelOverrides(configPath, mutate, { overwriteInvalid });
					overrideFileWritable = true;
					break;
				} catch (error) {
					if (error instanceof UnsafeModelOverridesFileError) {
						const confirmed = await ctx.ui.confirm(
							"Repair model override file?",
							`${error.problem}. Replace invalid content in ${configPath} with validated settings?`,
						);
						if (!confirmed) return false;
						overwriteInvalid = { revision: error.revision };
						continue;
					}
					ctx.ui.notify(`Failed to save ${configPath}: ${errorMessage(error)}`, "error");
					return false;
				}
			}

			if (refresh) {
				try {
					refreshProvider();
				} catch (error) {
					ctx.ui.notify(
						`Overrides were saved, but the provider refresh failed: ${errorMessage(error)}`,
						"warning",
					);
				}
			}
			return true;
		};

		refreshProvider();

		pi.on("before_provider_request", (event, ctx) => {
			if (ctx.model?.provider !== PROVIDER_NAME || !isRecord(event.payload)) return;
			const requestModel = event.payload.model;
			if (typeof requestModel !== "string" || requestModel !== ctx.model.id) return;

			const modelOverride = modelOverrides.get(requestModel);
			if (
				modelOverride?.reasoningEffort === undefined &&
				modelOverride?.temperature === undefined
			) {
				return;
			}

			return {
				...event.payload,
				...(modelOverride.reasoningEffort === undefined
					? {}
					: { reasoning_effort: modelOverride.reasoningEffort }),
				...(modelOverride.temperature === undefined
					? {}
					: { temperature: modelOverride.temperature }),
			};
		});

		pi.registerCommand("nebius-model", {
			description: "Edit and persist Nebius model parameters",
			getArgumentCompletions: (argumentPrefix) => {
				const prefix = argumentPrefix.trim().toLowerCase();
				return catalog
					.filter((model) => model.id.toLowerCase().includes(prefix))
					.map((model) => {
						const modelOverride = modelOverrides.get(model.id);
						const preview = buildProviderModel(model, contextOverride, modelOverride);
						const tools = resolveFeatureValue(model, modelOverride, "tools");
						return {
							value: model.id,
							label: model.id,
							description: `${tools ? "available" : "hidden"}; context ${formatTokens(preview.contextWindow)}, max ${formatTokens(preview.maxTokens)}`,
						};
					});
			},
			handler: async (args, ctx) => {
				if (!ctx.hasUI) {
					ctx.ui.notify("/nebius-model requires interactive or RPC UI mode", "error");
					return;
				}
				await ctx.waitForIdle();

				let selectedModelId = args.trim() || undefined;
				if (selectedModelId && !catalog.some((model) => model.id === selectedModelId)) {
					ctx.ui.notify(`Nebius model not found: ${selectedModelId}`, "error");
					return;
				}

				while (true) {
					if (!selectedModelId) {
						const menuItems = [...catalog]
							.sort((left, right) => left.id.localeCompare(right.id))
							.map((model) => {
								const modelOverride = modelOverrides.get(model.id);
								const preview = buildProviderModel(model, contextOverride, modelOverride);
								const tools = resolveFeatureValue(model, modelOverride, "tools");
								return {
									id: model.id,
									label: `${model.id}  [${tools ? "available" : "hidden"}; context ${formatTokens(preview.contextWindow)}, max ${formatTokens(preview.maxTokens)}]`,
								};
							});
						const selectedLabel = await ctx.ui.select(
							"Select a Nebius model",
							menuItems.map((item) => item.label),
						);
						if (selectedLabel === undefined) return;
						selectedModelId = menuItems.find((item) => item.label === selectedLabel)?.id;
						if (!selectedModelId) return;
					}

					const catalogModel = catalog.find((model) => model.id === selectedModelId);
					if (!catalogModel) {
						ctx.ui.notify(`Nebius model is no longer available: ${selectedModelId}`, "error");
						return;
					}
					const savedOverride = modelOverrides.get(selectedModelId);
					const currentModel = buildProviderModel(catalogModel, contextOverride, savedOverride);
					const toolsEnabled = resolveFeatureValue(catalogModel, savedOverride, "tools");
					const reasoningEnabled = resolveFeatureValue(catalogModel, savedOverride, "reasoning");
					const action = await ctx.ui.select(
						`${selectedModelId} — ${toolsEnabled ? "available" : "hidden"}, context ${formatTokens(currentModel.contextWindow)}, max ${formatTokens(currentModel.maxTokens)}, reasoning ${reasoningEnabled ? "on" : "off"}`,
						[
							EDIT_CONTEXT_ACTION,
							EDIT_MAX_TOKENS_ACTION,
							EDIT_FEATURES_ACTION,
							EDIT_REASONING_ACTION,
							EDIT_TEMPERATURE_ACTION,
							RESET_ACTION,
							BACK_ACTION,
						],
					);

					if (action === undefined) return;
					if (action === BACK_ACTION) {
						selectedModelId = undefined;
						continue;
					}

					switch (action) {
						case EDIT_CONTEXT_ACTION: {
							const value = await promptPositiveInteger(
								ctx,
								`Context window for ${selectedModelId} (current: ${formatTokens(currentModel.contextWindow)})`,
							);
							if (value === undefined) continue;

							let adjustedMaxTokens = false;
							const saved = await saveMutation(
								(overrides) =>
									mutateModelOverride(overrides, selectedModelId!, (modelOverride) => {
										modelOverride.contextWindow = value;
										if (
											modelOverride.maxTokens !== undefined &&
											modelOverride.maxTokens > value
										) {
											modelOverride.maxTokens = value;
											adjustedMaxTokens = true;
										}
									}),
								ctx,
								true,
							);
							if (saved) {
								ctx.ui.notify(
									`Saved ${selectedModelId} context window: ${formatTokens(value)}${adjustedMaxTokens ? " (max output reduced to match)" : ""}`,
									"info",
								);
							}
							continue;
						}

						case EDIT_MAX_TOKENS_ACTION: {
							const value = await promptPositiveInteger(
								ctx,
								`Max output tokens for ${selectedModelId} (current: ${formatTokens(currentModel.maxTokens)})`,
								currentModel.contextWindow,
							);
							if (value === undefined) continue;

							if (
								await saveMutation(
									(overrides) =>
										mutateModelOverride(overrides, selectedModelId!, (modelOverride) => {
											modelOverride.maxTokens = value;
										}),
									ctx,
									true,
								)
							) {
								ctx.ui.notify(
									`Saved ${selectedModelId} max output tokens: ${formatTokens(value)}`,
									"info",
								);
							}
							continue;
						}

						case EDIT_FEATURES_ACTION: {
							while (true) {
								const currentOverride = modelOverrides.get(selectedModelId);
								const toolsItem = `Tools: ${featureStateLabel(catalogModel, currentOverride, "tools")}`;
								const reasoningItem = `Reasoning: ${featureStateLabel(catalogModel, currentOverride, "reasoning")}`;
								const featureItem = await ctx.ui.select("Supported features", [
									toolsItem,
									reasoningItem,
									BACK,
								]);
								if (featureItem === undefined || featureItem === BACK) break;

								const feature: SupportedFeature = featureItem === toolsItem ? "tools" : "reasoning";
								const catalogEnabled = catalogFeatureValue(catalogModel, feature);
								const mode = await ctx.ui.select(
									`${feature === "tools" ? "Tools" : "Reasoning"} support`,
									[`${INHERIT} (${catalogEnabled ? "enabled" : "disabled"})`, ENABLED, DISABLED, BACK],
								);
								if (mode === undefined || mode === BACK) continue;
								const forcedValue = mode === ENABLED ? true : mode === DISABLED ? false : undefined;
								const nextFeatureValue = forcedValue ?? catalogEnabled;

								if (
									feature === "tools" &&
									!nextFeatureValue &&
									ctx.model?.provider === PROVIDER_NAME &&
									ctx.model.id === selectedModelId
								) {
									ctx.ui.notify("Switch to another model before disabling tools for the active model", "warning");
									continue;
								}

								if (
									await saveMutation(
										(overrides) =>
											mutateModelOverride(overrides, selectedModelId!, (modelOverride) => {
												const supportedFeatures = {
													...(modelOverride.supportedFeatures ?? {}),
												};
												if (forcedValue === undefined) delete supportedFeatures[feature];
												else supportedFeatures[feature] = forcedValue;
												modelOverride.supportedFeatures = supportedFeatures;
											}),
										ctx,
										true,
									)
								) {
									ctx.ui.notify(
										`${feature === "tools" ? "Tools" : "Reasoning"} support now ${forcedValue === undefined ? "inherits the catalog" : forcedValue ? "enabled" : "disabled"}`,
										"info",
									);
								}
							}
							continue;
						}

						case EDIT_REASONING_ACTION: {
							const currentEffort = savedOverride?.reasoningEffort;
							const inheritReasoning = "Inherit Pi /thinking (reset)";
							const effortOptions = REASONING_EFFORTS.map((effort) => `Force ${effort}`);
							const choice = await ctx.ui.select(
								`Reasoning effort (current: ${currentEffort ? `force ${currentEffort}` : "inherit Pi /thinking"})`,
								[inheritReasoning, ...effortOptions, BACK],
							);
							if (choice === undefined || choice === BACK) continue;
							const reasoningEffort: ReasoningEffort | undefined =
								choice === inheritReasoning
									? undefined
									: REASONING_EFFORTS[effortOptions.indexOf(choice)];
							if (
								await saveMutation(
									(overrides) =>
										mutateModelOverride(overrides, selectedModelId!, (modelOverride) => {
											if (reasoningEffort === undefined) delete modelOverride.reasoningEffort;
											else modelOverride.reasoningEffort = reasoningEffort;
										}),
									ctx,
									false,
								)
							) {
								ctx.ui.notify(
									reasoningEffort === undefined
										? `${selectedModelId} reasoning effort now inherits Pi /thinking`
										: `Forcing reasoning_effort=${reasoningEffort} for ${selectedModelId}`,
									"info",
								);
							}
							continue;
						}

						case EDIT_TEMPERATURE_ACTION: {
							const currentTemperature = savedOverride?.temperature;
							const raw = await ctx.ui.input(
								`Temperature for ${selectedModelId} (current: ${currentTemperature ?? "inherit"}; 0-2, blank = inherit)`,
							);
							if (raw === undefined) continue;
							const trimmed = raw.trim();
							const temperature = trimmed === "" ? undefined : Number(trimmed);
							if (temperature !== undefined && !isTemperature(temperature)) {
								ctx.ui.notify("Enter a number from 0 through 2, or leave blank to inherit", "warning");
								continue;
							}

							if (
								await saveMutation(
									(overrides) =>
										mutateModelOverride(overrides, selectedModelId!, (modelOverride) => {
											if (temperature === undefined) delete modelOverride.temperature;
											else modelOverride.temperature = temperature;
										}),
									ctx,
									false,
								)
							) {
								ctx.ui.notify(
									temperature === undefined
										? `${selectedModelId} temperature now inherits the request default`
										: `Forcing temperature=${temperature} for ${selectedModelId}`,
									"info",
								);
							}
							continue;
						}

						case RESET_ACTION: {
							if (!modelOverrides.has(selectedModelId) && overrideFileWritable) {
								ctx.ui.notify(`No saved overrides for ${selectedModelId}`, "info");
								continue;
							}
							if (
								!catalogFeatureValue(catalogModel, "tools") &&
								ctx.model?.provider === PROVIDER_NAME &&
								ctx.model.id === selectedModelId
							) {
								ctx.ui.notify(
									"Switch to another model before resetting the tools override for the active model",
									"warning",
								);
								continue;
							}
							const confirmed = await ctx.ui.confirm(
								`Reset ${selectedModelId}?`,
								"Remove all saved parameter overrides for this model?",
							);
							if (!confirmed) continue;

							if (
								await saveMutation(
									(overrides) => overrides.delete(selectedModelId!),
									ctx,
									true,
								)
							) {
								ctx.ui.notify(`Reset saved overrides for ${selectedModelId}`, "info");
							}
							continue;
						}
						}
				}
			},
		});
	};
}

export default createTokenFactoryExtension();
