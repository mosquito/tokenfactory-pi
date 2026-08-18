import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { createTokenFactoryExtension } from "../dist/index.js";

const KIMI_K3_ID = "moonshotai/Kimi-K3";
const MODEL_CATALOG_URL = "https://api.tokenfactory.nebius.com/v1/models?verbose=true";

function catalogModel(overrides) {
	return {
		id: "test/model",
		name: "Test model",
		context_length: 8_000,
		supported_features: ["tools"],
		architecture: { modality: "text->text" },
		...overrides,
	};
}

test("registers environment-overridden and upstream model context limits", async (t) => {
	const testDirectory = await mkdtemp(join(tmpdir(), "tokenfactory-pi-catalog-"));
	let configNumber = 0;
	t.after(() => rm(testDirectory, { recursive: true, force: true }));

	async function registeredModel(model, contextOverride) {
		const fetchImpl = async (url, init) => {
			assert.equal(url, MODEL_CATALOG_URL);
			assert.deepEqual(init?.headers, {
				Authorization: "Bearer test-api-key",
				"Accept-Encoding": "identity",
			});

			return new Response(JSON.stringify({ data: [model] }), {
				status: 200,
				headers: { "content-type": "application/json" },
			});
		};

		let registration;
		const registerTokenFactory = createTokenFactoryExtension({
			configPath: join(testDirectory, `overrides-${configNumber++}.json`),
			environment: {
				NEBIUS_API_KEY: "test-api-key",
				NEBIUS_MODEL_CONTEXT: contextOverride,
			},
			fetchImpl,
		});
		await registerTokenFactory({
			registerProvider(name, config) {
				assert.equal(name, "nebius");
				assert.equal(registration, undefined, "provider should register once");
				registration = config;
			},
			unregisterProvider() {
				assert.fail("provider should have at least one model");
			},
			on() {},
			registerCommand() {},
		});

		assert.ok(registration, "provider should be registered");
		assert.equal(registration.models.length, 1);
		return registration.models[0];
	}

	await t.test("overrides the live Kimi K3 8000-token catalog value", async () => {
		const model = await registeredModel(
			catalogModel({
				id: KIMI_K3_ID,
				name: "Kimi K3",
				context_length: 8_000,
			}),
			"1048576",
		);

		assert.equal(model.contextWindow, 1_048_576);
		assert.equal(model.maxTokens, 32_768);
	});

	await t.test("trusts a future corrected catalog value when no override is set", async () => {
		const model = await registeredModel(
			catalogModel({
				id: KIMI_K3_ID,
				name: "Kimi K3",
				context_length: 524_288,
			}),
		);

		assert.equal(model.contextWindow, 524_288);
		assert.equal(model.maxTokens, 32_768);
	});

	await t.test("uses an 8000-token catalog value when no override is set", async () => {
		const model = await registeredModel(
			catalogModel({
				id: "example/Unrelated-8K",
				name: "Unrelated 8K model",
				description: "A model advertised as supporting a 1M-token context window",
				context_length: 8_000,
			}),
		);

		assert.equal(model.contextWindow, 8_000);
		assert.equal(model.maxTokens, 8_000);
	});
});
