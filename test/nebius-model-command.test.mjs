import assert from "node:assert/strict";
import { access, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { createTokenFactoryExtension } from "../dist/index.js";

const KIMI_K3_ID = "moonshotai/Kimi-K3";
const OTHER_MODEL_ID = "example/Other-Model";
const HIDDEN_MODEL_ID = "example/Catalog-Missed-Tools";

const catalog = [
	{
		id: KIMI_K3_ID,
		name: "Kimi K3",
		context_length: 8_000,
		supported_features: ["tools", "reasoning"],
		architecture: { modality: "text+image->text" },
	},
	{
		id: OTHER_MODEL_ID,
		name: "Other Model",
		context_length: 262_144,
		supported_features: ["tools"],
		architecture: { modality: "text->text" },
	},
	{
		id: HIDDEN_MODEL_ID,
		name: "Catalog Missed Tools",
		context_length: 65_536,
		supported_features: [],
		architecture: { modality: "text->text" },
	},
];

async function loadExtension(configPath, modelContext) {
	const registrations = [];
	const unregistrations = [];
	const hooks = new Map();
	let command;
	const extension = createTokenFactoryExtension({
		configPath,
		environment: {
			NEBIUS_API_KEY: "test-api-key",
			NEBIUS_MODEL_CONTEXT: modelContext,
		},
		fetchImpl: async () =>
			new Response(JSON.stringify({ data: catalog }), {
				status: 200,
				headers: { "content-type": "application/json" },
			}),
	});

	await extension({
		registerProvider(name, config) {
			assert.equal(name, "nebius");
			registrations.push(config);
		},
		unregisterProvider(name) {
			assert.equal(name, "nebius");
			unregistrations.push(name);
		},
		on(name, handler) {
			hooks.set(name, handler);
		},
		registerCommand(name, spec) {
			assert.equal(name, "nebius-model");
			command = spec;
		},
	});

	assert.ok(command, "nebius-model command should be registered");
	assert.ok(hooks.has("before_provider_request"), "request override hook should be registered");
	return { command, hooks, registrations, unregistrations };
}

function scriptedContext({ select = [], input = [], confirm = [], model } = {}) {
	const selectResponses = [...select];
	const inputResponses = [...input];
	const confirmResponses = [...confirm];
	const notifications = [];
	let waitedForIdle = 0;

	return {
		ctx: {
			hasUI: true,
			model,
			async waitForIdle() {
				waitedForIdle += 1;
			},
			ui: {
				async select(title, options) {
					assert.ok(selectResponses.length > 0, `unexpected select dialog: ${title}`);
					const response = selectResponses.shift();
					const selected = typeof response === "function" ? response(options, title) : response;
					if (selected !== undefined) {
						assert.ok(options.includes(selected), `selection ${selected} was not offered by ${title}`);
					}
					return selected;
				},
				async input(title) {
					assert.ok(inputResponses.length > 0, `unexpected input dialog: ${title}`);
					return inputResponses.shift();
				},
				async confirm(title) {
					assert.ok(confirmResponses.length > 0, `unexpected confirm dialog: ${title}`);
					return confirmResponses.shift();
				},
				notify(message, type) {
					notifications.push({ message, type });
				},
			},
		},
		notifications,
		get waitedForIdle() {
			return waitedForIdle;
		},
	};
}

function findModel(registration, modelId) {
	const model = registration.models.find((candidate) => candidate.id === modelId);
	assert.ok(model, `missing registered model ${modelId}`);
	return model;
}

test("/nebius-model edits, persists, reloads, and resets per-model limits", async (t) => {
	const testDirectory = await mkdtemp(join(tmpdir(), "tokenfactory-pi-command-"));
	const configPath = join(testDirectory, "nested", "extensions", "tokenfactory-pi.json");
	t.after(() => rm(testDirectory, { recursive: true, force: true }));

	const { command, registrations } = await loadExtension(configPath);
	assert.equal(registrations.length, 1);
	assert.equal(findModel(registrations[0], KIMI_K3_ID).contextWindow, 8_000);

	const menu = scriptedContext({
		select: [
			(options) => options.find((option) => option.startsWith(KIMI_K3_ID)),
			"Edit context window",
			"Edit max output tokens",
			undefined,
		],
		input: ["1048576", "131072"],
	});
	await command.handler("", menu.ctx);

	assert.equal(menu.waitedForIdle, 1);
	assert.equal(registrations.length, 3);
	const editedKimi = findModel(registrations.at(-1), KIMI_K3_ID);
	assert.equal(editedKimi.contextWindow, 1_048_576);
	assert.equal(editedKimi.maxTokens, 131_072);
	const untouchedModel = findModel(registrations.at(-1), OTHER_MODEL_ID);
	assert.equal(untouchedModel.contextWindow, 262_144);
	assert.equal(untouchedModel.maxTokens, 32_768);

	assert.deepEqual(JSON.parse(await readFile(configPath, "utf8")), {
		version: 1,
		models: {
			[KIMI_K3_ID]: {
				contextWindow: 1_048_576,
				maxTokens: 131_072,
			},
		},
	});

	const reloaded = await loadExtension(configPath, "524288");
	const reloadedKimi = findModel(reloaded.registrations[0], KIMI_K3_ID);
	assert.equal(reloadedKimi.contextWindow, 1_048_576, "saved per-model context should win over the env override");
	assert.equal(reloadedKimi.maxTokens, 131_072);
	assert.equal(findModel(reloaded.registrations[0], OTHER_MODEL_ID).contextWindow, 524_288);

	const resetMenu = scriptedContext({
		select: ["Reset saved overrides", undefined],
		confirm: [true],
	});
	await command.handler(KIMI_K3_ID, resetMenu.ctx);
	assert.deepEqual(JSON.parse(await readFile(configPath, "utf8")), { version: 1, models: {} });
	assert.equal(findModel(registrations.at(-1), KIMI_K3_ID).contextWindow, 8_000);
});

test("supported feature overrides can expose and hide text models", async (t) => {
	const testDirectory = await mkdtemp(join(tmpdir(), "tokenfactory-pi-features-"));
	const configPath = join(testDirectory, "tokenfactory-pi.json");
	t.after(() => rm(testDirectory, { recursive: true, force: true }));

	const { command, registrations } = await loadExtension(configPath);
	assert.equal(registrations[0].models.some((model) => model.id === HIDDEN_MODEL_ID), false);

	const enableMenu = scriptedContext({
		select: [
			"Edit supported features",
			(options) => options.find((option) => option.startsWith("Tools:")),
			"Enabled",
			(options) => options.find((option) => option.startsWith("Reasoning:")),
			"Enabled",
			"Back",
			undefined,
		],
	});
	await command.handler(HIDDEN_MODEL_ID, enableMenu.ctx);

	const enabledModel = findModel(registrations.at(-1), HIDDEN_MODEL_ID);
	assert.equal(enabledModel.reasoning, true);
	assert.deepEqual(JSON.parse(await readFile(configPath, "utf8")).models[HIDDEN_MODEL_ID], {
		supportedFeatures: { tools: true, reasoning: true },
	});

	const resetRegistrationCount = registrations.length;
	const activeResetMenu = scriptedContext({
		model: { provider: "nebius", id: HIDDEN_MODEL_ID },
		select: ["Reset saved overrides", undefined],
	});
	await command.handler(HIDDEN_MODEL_ID, activeResetMenu.ctx);
	assert.equal(registrations.length, resetRegistrationCount);
	assert.ok(activeResetMenu.notifications.some(({ type }) => type === "warning"));
	assert.deepEqual(JSON.parse(await readFile(configPath, "utf8")).models[HIDDEN_MODEL_ID], {
		supportedFeatures: { tools: true, reasoning: true },
	});
	const activeInheritMenu = scriptedContext({
		model: { provider: "nebius", id: HIDDEN_MODEL_ID },
		select: [
			"Edit supported features",
			(options) => options.find((option) => option.startsWith("Tools:")),
			"Inherit catalog/default (disabled)",
			"Back",
			undefined,
		],
	});
	await command.handler(HIDDEN_MODEL_ID, activeInheritMenu.ctx);
	assert.equal(registrations.length, resetRegistrationCount);
	assert.ok(activeInheritMenu.notifications.some(({ type }) => type === "warning"));
	assert.deepEqual(JSON.parse(await readFile(configPath, "utf8")).models[HIDDEN_MODEL_ID], {
		supportedFeatures: { tools: true, reasoning: true },
	});

	const disableMenu = scriptedContext({
		select: [
			"Edit supported features",
			(options) => options.find((option) => option.startsWith("Tools:")),
			"Disabled",
			"Back",
			undefined,
		],
	});
	await command.handler(HIDDEN_MODEL_ID, disableMenu.ctx);
	assert.equal(registrations.at(-1).models.some((model) => model.id === HIDDEN_MODEL_ID), false);
	assert.deepEqual(JSON.parse(await readFile(configPath, "utf8")).models[HIDDEN_MODEL_ID], {
		supportedFeatures: { tools: false, reasoning: true },
	});

	const registrationCount = registrations.length;
	const activeModelMenu = scriptedContext({
		model: { provider: "nebius", id: KIMI_K3_ID },
		select: [
			"Edit supported features",
			(options) => options.find((option) => option.startsWith("Tools:")),
			"Disabled",
			"Back",
			undefined,
		],
	});
	await command.handler(KIMI_K3_ID, activeModelMenu.ctx);
	assert.equal(registrations.length, registrationCount);
	assert.ok(activeModelMenu.notifications.some(({ type }) => type === "warning"));
});

test("reasoning effort and temperature override matching Nebius requests", async (t) => {
	const testDirectory = await mkdtemp(join(tmpdir(), "tokenfactory-pi-request-"));
	const configPath = join(testDirectory, "tokenfactory-pi.json");
	t.after(() => rm(testDirectory, { recursive: true, force: true }));

	const { command, hooks, registrations } = await loadExtension(configPath);
	const menu = scriptedContext({
		select: ["Edit reasoning effort", "Force max", "Edit temperature", undefined],
		input: ["0.2"],
	});
	await command.handler(KIMI_K3_ID, menu.ctx);

	assert.equal(registrations.length, 1, "request-only settings should not rebuild the provider");
	assert.deepEqual(JSON.parse(await readFile(configPath, "utf8")).models[KIMI_K3_ID], {
		reasoningEffort: "max",
		temperature: 0.2,
	});

	const hook = hooks.get("before_provider_request");
	const originalPayload = {
		model: KIMI_K3_ID,
		messages: [{ role: "user", content: "hello" }],
		reasoning_effort: "low",
		temperature: 1,
	};
	const overridden = await hook(
		{ type: "before_provider_request", payload: originalPayload },
		{ model: { provider: "nebius", id: KIMI_K3_ID } },
	);
	assert.notEqual(overridden, originalPayload);
	assert.equal(overridden.reasoning_effort, "max");
	assert.equal(overridden.temperature, 0.2);
	assert.equal(originalPayload.reasoning_effort, "low", "the hook must not mutate the original payload");
	assert.equal(originalPayload.temperature, 1);

	assert.equal(
		await hook(
			{ type: "before_provider_request", payload: originalPayload },
			{ model: { provider: "other", id: KIMI_K3_ID } },
		),
		undefined,
	);
	assert.equal(
		await hook(
			{ type: "before_provider_request", payload: { ...originalPayload, model: OTHER_MODEL_ID } },
			{ model: { provider: "nebius", id: KIMI_K3_ID } },
		),
		undefined,
	);
	assert.equal(
		await hook(
			{ type: "before_provider_request", payload: null },
			{ model: { provider: "nebius", id: KIMI_K3_ID } },
		),
		undefined,
	);

	const resetRequestMenu = scriptedContext({
		select: [
			"Edit reasoning effort",
			"Inherit Pi /thinking (reset)",
			"Edit temperature",
			undefined,
		],
		input: [""],
	});
	await command.handler(KIMI_K3_ID, resetRequestMenu.ctx);
	assert.deepEqual(JSON.parse(await readFile(configPath, "utf8")), { version: 1, models: {} });
	assert.equal(
		await hook(
			{ type: "before_provider_request", payload: originalPayload },
			{ model: { provider: "nebius", id: KIMI_K3_ID } },
		),
		undefined,
	);
});

test("invalid and cancelled edits do not persist or re-register", async (t) => {
	const testDirectory = await mkdtemp(join(tmpdir(), "tokenfactory-pi-command-invalid-"));
	const configPath = join(testDirectory, "tokenfactory-pi.json");
	t.after(() => rm(testDirectory, { recursive: true, force: true }));

	const { command, registrations } = await loadExtension(configPath);
	const menu = scriptedContext({
		select: ["Edit context window", "Edit temperature", undefined],
		input: ["not-a-number", undefined, "2.1"],
	});
	await command.handler(KIMI_K3_ID, menu.ctx);

	assert.equal(registrations.length, 1);
	assert.ok(menu.notifications.some(({ type }) => type === "warning"));
	await assert.rejects(access(configPath), (error) => error.code === "ENOENT");
});

test("malformed persisted JSON is preserved until explicit repair", async (t) => {
	const testDirectory = await mkdtemp(join(tmpdir(), "tokenfactory-pi-command-malformed-"));
	const configPath = join(testDirectory, "tokenfactory-pi.json");
	t.after(() => rm(testDirectory, { recursive: true, force: true }));
	await writeFile(configPath, "{not-json", "utf8");

	const originalWarn = console.warn;
	const warnings = [];
	console.warn = (...args) => warnings.push(args);
	t.after(() => {
		console.warn = originalWarn;
	});

	const { command, registrations } = await loadExtension(configPath);
	assert.equal(findModel(registrations[0], KIMI_K3_ID).contextWindow, 8_000);
	assert.ok(warnings.length >= 1);

	const refusedRepair = scriptedContext({
		select: ["Edit context window", undefined],
		input: ["1048576"],
		confirm: [false],
	});
	await command.handler(KIMI_K3_ID, refusedRepair.ctx);
	assert.equal(await readFile(configPath, "utf8"), "{not-json");

	const explicitRepair = scriptedContext({
		select: ["Reset saved overrides", undefined],
		confirm: [true, true],
	});
	await command.handler(KIMI_K3_ID, explicitRepair.ctx);
	assert.deepEqual(JSON.parse(await readFile(configPath, "utf8")), { version: 1, models: {} });
});

test("stale extension instances merge edits made by another session", async (t) => {
	const testDirectory = await mkdtemp(join(tmpdir(), "tokenfactory-pi-command-merge-"));
	const configPath = join(testDirectory, "tokenfactory-pi.json");
	t.after(() => rm(testDirectory, { recursive: true, force: true }));

	const first = await loadExtension(configPath);
	const second = await loadExtension(configPath);
	await first.command.handler(
		KIMI_K3_ID,
		scriptedContext({ select: ["Edit context window", undefined], input: ["1048576"] }).ctx,
	);
	await second.command.handler(
		OTHER_MODEL_ID,
		scriptedContext({ select: ["Edit max output tokens", undefined], input: ["65536"] }).ctx,
	);

	assert.deepEqual(JSON.parse(await readFile(configPath, "utf8")), {
		version: 1,
		models: {
			[OTHER_MODEL_ID]: { maxTokens: 65_536 },
			[KIMI_K3_ID]: { contextWindow: 1_048_576 },
		},
	});
});

test("a stale max-token edit cannot violate a concurrently lowered context", async (t) => {
	const testDirectory = await mkdtemp(join(tmpdir(), "tokenfactory-pi-command-invariant-"));
	const configPath = join(testDirectory, "tokenfactory-pi.json");
	t.after(() => rm(testDirectory, { recursive: true, force: true }));

	const contextEditor = await loadExtension(configPath);
	const staleMaxEditor = await loadExtension(configPath);
	await contextEditor.command.handler(
		KIMI_K3_ID,
		scriptedContext({ select: ["Edit context window", undefined], input: ["100"] }).ctx,
	);
	const staleEdit = scriptedContext({
		select: ["Edit max output tokens", undefined],
		input: ["1000"],
	});
	await staleMaxEditor.command.handler(KIMI_K3_ID, staleEdit.ctx);

	assert.ok(staleEdit.notifications.some(({ type }) => type === "error"));
	assert.deepEqual(JSON.parse(await readFile(configPath, "utf8")), {
		version: 1,
		models: { [KIMI_K3_ID]: { contextWindow: 100 } },
	});
});
