import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import {
	getDefaultModelOverridesPath,
	loadModelOverrides,
	REASONING_EFFORTS,
	UnsafeModelOverridesFileError,
	updateModelOverrides,
} from "../dist/model-overrides.js";

const EXPECTED_REASONING_EFFORTS = [
	"none",
	"minimal",
	"low",
	"medium",
	"high",
	"xhigh",
	"max",
];

test("loads every supported reasoning effort and temperature boundary", async (t) => {
	const testDirectory = await mkdtemp(join(tmpdir(), "tokenfactory-pi-validation-"));
	const configPath = join(testDirectory, "tokenfactory-pi.json");
	t.after(() => rm(testDirectory, { recursive: true, force: true }));

	assert.deepEqual(REASONING_EFFORTS, EXPECTED_REASONING_EFFORTS);
	const models = Object.fromEntries(
		EXPECTED_REASONING_EFFORTS.map((reasoningEffort, index) => [
			`example/model-${index}`,
			{ reasoningEffort, temperature: index % 2 === 0 ? 0 : 2 },
		]),
	);
	await writeFile(configPath, `${JSON.stringify({ version: 1, models })}\n`, "utf8");

	const loaded = await loadModelOverrides(configPath);
	assert.equal(loaded.writable, true);
	assert.equal(loaded.overrides.size, EXPECTED_REASONING_EFFORTS.length);
	for (const [index, reasoningEffort] of EXPECTED_REASONING_EFFORTS.entries()) {
		assert.deepEqual(loaded.overrides.get(`example/model-${index}`), {
			reasoningEffort,
			temperature: index % 2 === 0 ? 0 : 2,
		});
	}
});

test("default override path follows PI_CODING_AGENT_DIR", () => {
	const originalAgentDirectory = process.env.PI_CODING_AGENT_DIR;
	const agentDirectory = join(tmpdir(), "tokenfactory-pi-agent-dir-test");
	process.env.PI_CODING_AGENT_DIR = agentDirectory;
	try {
		assert.equal(
			getDefaultModelOverridesPath(),
			join(agentDirectory, "extensions", "tokenfactory-pi.json"),
		);
	} finally {
		if (originalAgentDirectory === undefined) delete process.env.PI_CODING_AGENT_DIR;
		else process.env.PI_CODING_AGENT_DIR = originalAgentDirectory;
	}
});

test("repair approval is tied to the invalid file revision", async (t) => {
	const testDirectory = await mkdtemp(join(tmpdir(), "tokenfactory-pi-repair-revision-"));
	const configPath = join(testDirectory, "tokenfactory-pi.json");
	t.after(() => rm(testDirectory, { recursive: true, force: true }));
	const originalWarn = console.warn;
	console.warn = () => {};
	t.after(() => {
		console.warn = originalWarn;
	});
	await writeFile(configPath, "{first-invalid", "utf8");

	let firstError;
	try {
		await updateModelOverrides(configPath, () => {});
	} catch (error) {
		firstError = error;
	}
	assert.ok(firstError instanceof UnsafeModelOverridesFileError);

	await writeFile(configPath, "{newer-invalid", "utf8");
	await assert.rejects(
		updateModelOverrides(configPath, () => {}, {
			overwriteInvalid: { revision: firstError.revision },
		}),
		UnsafeModelOverridesFileError,
	);
	assert.equal(await readFile(configPath, "utf8"), "{newer-invalid");
});
