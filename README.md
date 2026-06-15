# tokenfactory-pi

[Nebius Token Factory](https://tokenfactory.nebius.com/) provider extension for [pi coding agent](https://pi.dev).

Fetches the current model catalog from the Token Factory API on startup and registers all tool-capable text models. No pi core changes required.

## Prerequisites

```bash
# Install pi coding agent globally (required)
npm install -g --ignore-scripts @earendil-works/pi-coding-agent
```

## Installation

```bash
# Install the extension using pi's built-in package manager
pi install npm:tokenfactory-pi
```

## Setup

```bash
# Get an API key from https://tokenfactory.nebius.com/
export NEBIUS_API_KEY=your-key-here
```

## Usage

```bash
# List available models to verify installation
pi --list-models | grep nebius
```

Once running in interactive mode, use `/nebius-models` to list all available models.

## Development

For local development:
```bash
# Build the TypeScript
npm run build

# Test locally from the project directory
cd path/to/tokenfactory-pi
pi -e . --provider nebius
```

## How it works

On startup the extension:

1. Reads `NEBIUS_API_KEY` from environment (no-op if missing)
2. Fetches `GET /v1/models?verbose=true` from the Token Factory API
3. Filters for models with `tools` support and `->text` output modality
4. Registers them as the `nebius` provider via `pi.registerProvider()`

All models use the `openai-completions` API with
`compat: { supportsDeveloperRole: false, maxTokensField: "max_tokens" }`.
The registered provider resolves auth from `$NEBIUS_API_KEY`, matching pi's current custom-provider config syntax.
This is important: a literal `NEBIUS_API_KEY` value would be sent as the bearer token and Token Factory returns `401`.
