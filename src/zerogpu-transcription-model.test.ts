import { createTestServer } from '@ai-sdk/test-server/with-vitest';
import fs from 'node:fs';
import { describe, expect, it, vi } from 'vitest';
import { createZeroGPU } from './zerogpu-provider';

vi.mock('./version', () => ({
  VERSION: '0.0.0-test',
}));

const TRANSCRIPTIONS_URL = 'https://api.zerogpu.ai/v1/audio/transcriptions';
const audioData = new Uint8Array([0, 1, 2, 3, 4]);
const transcriptionResponse = JSON.parse(
  fs.readFileSync('src/__fixtures__/zerogpu-transcription.json', 'utf8'),
);

const provider = createZeroGPU({ apiKey: 'test-api-key' });
const model = provider.transcription('whisper-tiny');

const server = createTestServer({
  [TRANSCRIPTIONS_URL]: {},
});

function prepareJsonResponse({
  body = transcriptionResponse,
  headers,
}: {
  body?: Record<string, unknown>;
  headers?: Record<string, string>;
} = {}) {
  server.urls[TRANSCRIPTIONS_URL].response = {
    type: 'json-value',
    headers,
    body,
  };
}

describe('doGenerate', () => {
  it('should send audio as a multipart file with verbose_json', async () => {
    prepareJsonResponse();

    await model.doGenerate({ audio: audioData, mediaType: 'audio/mpeg' });

    const formData = await server.calls[0].requestBodyMultipart;

    expect(formData).toMatchObject({
      model: 'whisper-tiny',
      response_format: 'verbose_json',
    });
    expect(formData?.file).toBeInstanceOf(File);
    expect(formData?.file.name).toBe('audio.mp3');
    expect(new Uint8Array(await formData?.file.arrayBuffer())).toStrictEqual(
      audioData,
    );
  });

  it('should send base64 audio', async () => {
    prepareJsonResponse();

    await model.doGenerate({ audio: 'AAECAwQ=', mediaType: 'audio/wav' });

    const formData = await server.calls[0].requestBodyMultipart;

    expect(formData?.file.name).toBe('audio.wav');
    expect(new Uint8Array(await formData?.file.arrayBuffer())).toStrictEqual(
      audioData,
    );
  });

  it('should pass provider options', async () => {
    prepareJsonResponse();

    await model.doGenerate({
      audio: audioData,
      mediaType: 'audio/mpeg',
      providerOptions: { zerogpu: { language: 'en', temperature: 0 } },
    });

    expect(await server.calls[0].requestBodyMultipart).toMatchObject({
      language: 'en',
      temperature: '0',
    });
  });

  it('should map the verbose_json response', async () => {
    prepareJsonResponse();

    const result = await model.doGenerate({
      audio: audioData,
      mediaType: 'audio/mpeg',
    });

    expect(result.text).toBe('Hello from zero GPU in the AISDK.');
    expect(result.language).toBe('en');
    expect(result.durationInSeconds).toBe(3.12);
    expect(result.segments).toStrictEqual([
      {
        text: ' Hello from zero GPU in the AISDK.',
        startSecond: 0,
        endSecond: 3,
      },
    ]);
    expect(result.warnings).toStrictEqual([]);
    expect(result.response.modelId).toBe('whisper-tiny');
  });

  it('should keep ISO-639-1 language codes and drop unknown languages', async () => {
    prepareJsonResponse({ body: { text: 'Bonjour', language: 'fr' } });
    const french = await model.doGenerate({
      audio: audioData,
      mediaType: 'audio/mpeg',
    });

    prepareJsonResponse({ body: { text: 'Hello', language: 'klingon' } });
    const unknown = await model.doGenerate({
      audio: audioData,
      mediaType: 'audio/mpeg',
    });

    expect(french.language).toBe('fr');
    expect(unknown.language).toBeUndefined();
    expect(unknown.segments).toStrictEqual([]);
    expect(unknown.durationInSeconds).toBeUndefined();
  });

  it('should pass headers', async () => {
    prepareJsonResponse();

    await createZeroGPU({
      apiKey: 'test-api-key',
      headers: { 'Custom-Provider-Header': 'provider-header-value' },
    })
      .transcription('whisper-tiny')
      .doGenerate({
        audio: audioData,
        mediaType: 'audio/mpeg',
        headers: { 'Custom-Request-Header': 'request-header-value' },
      });

    expect(server.calls[0].requestHeaders).toMatchObject({
      authorization: 'Bearer test-api-key',
      'custom-provider-header': 'provider-header-value',
      'custom-request-header': 'request-header-value',
    });
  });
});
