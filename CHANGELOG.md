# Changelog

All notable changes to `@zerogpu/ai-sdk-provider` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-10-30

### Changed

- Docs: use the AI SDK 7 names `isStepCount` (was `stepCountIs`) and `instructions` (was `system`) in the examples.
- First release published from CI with npm provenance.

## [0.1.0] - 2026-10-29

### Added

- Initial release of the ZeroGPU provider for AI SDK 7 (Language Model Specification V4).
- Language models: `generateText` and `streamText`, tool calling, structured output, image input and reasoning effort.
- ZeroGPU classification and extraction models (IAB, signal extraction, GLiNER, DeBERTa) via `providerOptions.zerogpu`.
- Embedding models (`embed`, `embedMany`).
- Speech models (`generateSpeech`).
- Transcription models (`transcribe`).

[0.1.1]: https://github.com/zerogpu/ai-sdk-provider/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/zerogpu/ai-sdk-provider/releases/tag/v0.1.0
