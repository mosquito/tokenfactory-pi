import { createHash, randomUUID } from "node:crypto";
import { mkdir, open, readFile, rename, unlink } from "node:fs/promises";
import { basename, dirname, join } from "node:path";
import { getAgentDir, withFileMutationQueue } from "@earendil-works/pi-coding-agent";

const CONFIG_VERSION = 1;
const CONFIG_FILE_NAME = "tokenfactory-pi.json";
const MAX_CONFIG_BYTES = 1024 * 1024;

export const REASONING_EFFORTS = [
	"none",
	"minimal",
	"low",
	"medium",
	"high",
	"xhigh",
	"max",
] as const;

export type ReasoningEffort = (typeof REASONING_EFFORTS)[number];

export interface SupportedFeatureOverrides {
	tools?: boolean;
	reasoning?: boolean;
}

export interface ModelParameterOverride {
	contextWindow?: number;
	maxTokens?: number;
	supportedFeatures?: SupportedFeatureOverrides;
	reasoningEffort?: ReasoningEffort;
	temperature?: number;
}

export type ModelOverrides = Map<string, ModelParameterOverride>;

export interface ModelOverridesLoadResult {
	overrides: ModelOverrides;
	writable: boolean;
	problem?: string;
	revision?: string;
}

interface PersistedModelOverrides {
	version: typeof CONFIG_VERSION;
	models: Record<string, ModelParameterOverride>;
}

export class UnsafeModelOverridesFileError extends Error {
	constructor(
		readonly configPath: string,
		readonly problem: string,
		readonly revision?: string,
	) {
		super(`Refusing to overwrite ${configPath}: ${problem}`);
		this.name = "UnsafeModelOverridesFileError";
	}
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function isPositiveInteger(value: unknown): value is number {
	return typeof value === "number" && Number.isSafeInteger(value) && value > 0;
}

export function isReasoningEffort(value: unknown): value is ReasoningEffort {
	return typeof value === "string" && (REASONING_EFFORTS as readonly string[]).includes(value);
}

export function isTemperature(value: unknown): value is number {
	return typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 2;
}

export function getDefaultModelOverridesPath(): string {
	return join(getAgentDir(), "extensions", CONFIG_FILE_NAME);
}

function warnInvalidFile(
	configPath: string,
	problem: string,
	revision?: string,
): ModelOverridesLoadResult {
	console.warn(`[nebius] Model overrides in ${configPath} are not writable: ${problem}`);
	return { overrides: new Map(), writable: false, problem, revision };
}

export async function loadModelOverrides(configPath: string): Promise<ModelOverridesLoadResult> {
	let raw: string;
	try {
		raw = await readFile(configPath, "utf8");
	} catch (error) {
		if ((error as NodeJS.ErrnoException).code === "ENOENT") {
			return { overrides: new Map(), writable: true };
		}
		const problem = `failed to read the file: ${error instanceof Error ? error.message : String(error)}`;
		return warnInvalidFile(configPath, problem);
	}

	const revision = createHash("sha256").update(raw).digest("hex");
	if (Buffer.byteLength(raw, "utf8") > MAX_CONFIG_BYTES) {
		return warnInvalidFile(configPath, `file exceeds ${MAX_CONFIG_BYTES} bytes`, revision);
	}

	let parsed: unknown;
	try {
		parsed = JSON.parse(raw);
	} catch (error) {
		return warnInvalidFile(
			configPath,
			`invalid JSON: ${error instanceof Error ? error.message : String(error)}`,
			revision,
		);
	}

	if (!isRecord(parsed)) return warnInvalidFile(configPath, "expected a JSON object", revision);
	if (parsed.version !== CONFIG_VERSION) {
		return warnInvalidFile(configPath, `expected version ${CONFIG_VERSION}`, revision);
	}
	if (!isRecord(parsed.models)) {
		return warnInvalidFile(configPath, "expected a models object", revision);
	}

	let invalid = Object.keys(parsed).some((key) => key !== "version" && key !== "models");
	const overrides: ModelOverrides = new Map();
	for (const [modelId, value] of Object.entries(parsed.models)) {
		if (modelId.trim() === "" || !isRecord(value)) {
			invalid = true;
			continue;
		}

		const modelOverride: ModelParameterOverride = {};
		const knownKeys = new Set([
			"contextWindow",
			"maxTokens",
			"supportedFeatures",
			"reasoningEffort",
			"temperature",
		]);
		if (Object.keys(value).some((key) => !knownKeys.has(key))) invalid = true;

		if (value.contextWindow !== undefined) {
			if (isPositiveInteger(value.contextWindow)) modelOverride.contextWindow = value.contextWindow;
			else invalid = true;
		}
		if (value.maxTokens !== undefined) {
			if (isPositiveInteger(value.maxTokens)) modelOverride.maxTokens = value.maxTokens;
			else invalid = true;
		}
		if (value.supportedFeatures !== undefined) {
			if (isRecord(value.supportedFeatures)) {
				const supportedFeatures: SupportedFeatureOverrides = {};
				if (
					Object.keys(value.supportedFeatures).some(
						(key) => key !== "tools" && key !== "reasoning",
					)
				) {
					invalid = true;
				}
				for (const feature of ["tools", "reasoning"] as const) {
					const featureValue = value.supportedFeatures[feature];
					if (featureValue === undefined) continue;
					if (typeof featureValue === "boolean") supportedFeatures[feature] = featureValue;
					else invalid = true;
				}
				if (Object.keys(supportedFeatures).length > 0) {
					modelOverride.supportedFeatures = supportedFeatures;
				}
			} else {
				invalid = true;
			}
		}
		if (value.reasoningEffort !== undefined) {
			if (isReasoningEffort(value.reasoningEffort)) {
				modelOverride.reasoningEffort = value.reasoningEffort;
			} else {
				invalid = true;
			}
		}
		if (value.temperature !== undefined) {
			if (isTemperature(value.temperature)) modelOverride.temperature = value.temperature;
			else invalid = true;
		}

		if (
			modelOverride.contextWindow !== undefined &&
			modelOverride.maxTokens !== undefined &&
			modelOverride.maxTokens > modelOverride.contextWindow
		) {
			delete modelOverride.maxTokens;
			invalid = true;
		}

		if (hasOverrideValues(modelOverride)) overrides.set(modelId, modelOverride);
	}

	if (invalid) {
		const problem = "contains invalid or unknown values";
		console.warn(`[nebius] Model overrides in ${configPath} are not writable: ${problem}`);
		return { overrides, writable: false, problem, revision };
	}
	return { overrides, writable: true, revision };
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

function cloneModelOverride(value: ModelParameterOverride): ModelParameterOverride {
	return {
		...value,
		supportedFeatures: value.supportedFeatures ? { ...value.supportedFeatures } : undefined,
	};
}

function cloneModelOverrides(overrides: ModelOverrides): ModelOverrides {
	return new Map(
		[...overrides.entries()].map(([modelId, value]) => [modelId, cloneModelOverride(value)]),
	);
}

function serializeModelOverrides(overrides: ModelOverrides): string {
	const models = Object.fromEntries(
		[...overrides.entries()]
			.filter(([, value]) => hasOverrideValues(value))
			.sort(([left], [right]) => left.localeCompare(right))
			.map(([modelId, value]) => [modelId, cloneModelOverride(value)]),
	);
	const persisted: PersistedModelOverrides = { version: CONFIG_VERSION, models };
	return `${JSON.stringify(persisted, null, 2)}\n`;
}

async function writeAtomically(configPath: string, contents: string): Promise<void> {
	const parentDirectory = dirname(configPath);
	await mkdir(parentDirectory, { recursive: true, mode: 0o700 });
	const temporaryPath = join(
		parentDirectory,
		`.${basename(configPath)}.${process.pid}.${randomUUID()}.tmp`,
	);
	let handle: Awaited<ReturnType<typeof open>> | undefined;

	try {
		handle = await open(temporaryPath, "wx", 0o600);
		await handle.writeFile(contents, "utf8");
		await handle.sync();
		await handle.close();
		handle = undefined;
		await rename(temporaryPath, configPath);
	} finally {
		if (handle) await handle.close().catch(() => undefined);
		await unlink(temporaryPath).catch((error: NodeJS.ErrnoException) => {
			if (error.code !== "ENOENT") throw error;
		});
	}
}

export interface UpdateModelOverridesOptions {
	overwriteInvalid?: { revision?: string };
}

function validateMergedOverrides(overrides: ModelOverrides): void {
	for (const [modelId, modelOverride] of overrides) {
		if (
			modelOverride.contextWindow !== undefined &&
			modelOverride.maxTokens !== undefined &&
			modelOverride.maxTokens > modelOverride.contextWindow
		) {
			throw new Error(
				`Max output tokens for ${modelId} cannot exceed its saved context window`,
			);
		}
	}
}

export async function updateModelOverrides(
	configPath: string,
	mutate: (overrides: ModelOverrides) => void,
	options: UpdateModelOverridesOptions = {},
): Promise<ModelOverrides> {
	return withFileMutationQueue(configPath, async () => {
		const loaded = await loadModelOverrides(configPath);
		if (
			!loaded.writable &&
			(!options.overwriteInvalid || options.overwriteInvalid.revision !== loaded.revision)
		) {
			throw new UnsafeModelOverridesFileError(
				configPath,
				loaded.problem ?? "the file is invalid",
				loaded.revision,
			);
		}

		const nextOverrides = cloneModelOverrides(loaded.overrides);
		mutate(nextOverrides);
		validateMergedOverrides(nextOverrides);
		await writeAtomically(configPath, serializeModelOverrides(nextOverrides));
		return cloneModelOverrides(nextOverrides);
	});
}
