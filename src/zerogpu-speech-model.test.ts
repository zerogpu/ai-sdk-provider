import { createTestServer } from '@ai-sdk/test-server/with-vitest';
import { describe, expect, it, vi } from 'vitest';
import { createZeroGPU } from './zerogpu-provider';

vi.mock('./version', () => ({
  VERSION: '0.0.0-test',
}));

const SPEECH_URL = 'https://api.zerogpu.ai/v1/audio/speech';
const audio = new Uint8Array([73, 68, 51, 4, 0]);

const provider = createZeroGPU({ apiKey: 'test-api-key' });
const model = provider.speech('chatterbox-nano');

const server = createTestServer({
  [SPEECH_URL]: {},
});

function prepareAudioResponse({
  headers = { 'content-type': 'audio/mpeg' },
}: { headers?: Record<string, string> } = {}) {
  server.urls[SPEECH_URL].response = {
    type: 'binary',
    headers,
    body: Buffer.from(audio),
  };
}

describe('doGenerate', () => {
  it('should send the default voice and mp3 output', async () => {
    prepareAudioResponse();

    await model.doGenerate({ text: 'Hello from ZeroGPU.' });

    expect(await server.calls[0].requestBodyJson).toStrictEqual({
      model: 'chatterbox-nano',
      input: 'Hello from ZeroGPU.',
      voice: 'alloy',
      response_format: 'mp3',
    });
  });

  it('should pass voice, output format and provider options', async () => {
    prepareAudioResponse({ headers: { 'content-type': 'audio/wav' } });

    await model.doGenerate({
      text: 'Hello [laugh]',
      voice: 'nova',
      outputFormat: 'wav',
      providerOptions: {
        zerogpu: {
          seed: 7,
          temperature: 0.8,
          topP: 0.9,
          topK: 50,
          repetitionPenalty: 1.2,
        },
      },
    });

    expect(await server.calls[0].requestBodyJson).toStrictEqual({
      model: 'chatterbox-nano',
      input: 'Hello [laugh]',
      voice: 'nova',
      response_format: 'wav',
      seed: 7,
      temperature: 0.8,
      top_p: 0.9,
      top_k: 50,
      repetition_penalty: 1.2,
    });
  });

  it('should return binary audio', async () => {
    prepareAudioResponse();

    const result = await model.doGenerate({ text: 'Hello' });

    expect(result.audio).toStrictEqual(audio);
    expect(result.warnings).toStrictEqual([]);
  });

  it('should warn and use mp3 for unsupported output formats', async () => {
    prepareAudioResponse();

    const result = await model.doGenerate({
      text: 'Hello',
      outputFormat: 'webm',
    });

    expect(
      ((await server.calls[0].requestBodyJson) as { response_format: string })
        .response_format,
    ).toBe('mp3');
    expect(result.warnings).toStrictEqual([
      {
        type: 'unsupported',
        feature: 'outputFormat',
        details: 'Unsupported output format: webm. Using mp3 instead.',
      },
    ]);
  });

  it('should warn about unsupported standard options and not send them', async () => {
    prepareAudioResponse();

    const result = await model.doGenerate({
      text: 'Hello',
      instructions: 'Speak happily',
      speed: 1.5,
      language: 'en',
    });

    expect(await server.calls[0].requestBodyJson).toStrictEqual({
      model: 'chatterbox-nano',
      input: 'Hello',
      voice: 'alloy',
      response_format: 'mp3',
    });
    expect(result.warnings).toMatchObject([
      { type: 'unsupported', feature: 'instructions' },
      { type: 'unsupported', feature: 'speed' },
      { type: 'unsupported', feature: 'language' },
    ]);
  });

  it('should include response metadata', async () => {
    prepareAudioResponse({
      headers: {
        'content-type': 'audio/mpeg',
        'x-audio-duration-seconds': '1.76',
      },
    });
    const result = await model.doGenerate({ text: 'Hello' });

    expect(result.response).toMatchObject({
      modelId: 'chatterbox-nano',
      headers: {
        'content-type': 'audio/mpeg',
        'x-audio-duration-seconds': '1.76',
      },
    });
    expect(result.response.timestamp).toBeInstanceOf(Date);
  });

  it('should pass headers', async () => {
    prepareAudioResponse();

    await createZeroGPU({
      apiKey: 'test-api-key',
      headers: { 'Custom-Provider-Header': 'provider-header-value' },
    })
      .speech('chatterbox-nano')
      .doGenerate({
        text: 'Hello',
        headers: { 'Custom-Request-Header': 'request-header-value' },
      });

    expect(server.calls[0].requestHeaders).toStrictEqual({
      authorization: 'Bearer test-api-key',
      'content-type': 'application/json',
      'custom-provider-header': 'provider-header-value',
      'custom-request-header': 'request-header-value',
    });
  });

  it('should throw an APICallError for error responses', async () => {
    server.urls[SPEECH_URL].response = {
      type: 'error',
      status: 400,
      body: JSON.stringify({
        error: {
          message: "'instructions' is not supported by this endpoint.",
          type: 'invalid_request_error',
          param: 'instructions',
          code: 'instructions_not_supported',
        },
      }),
    };

    await expect(model.doGenerate({ text: 'Hello' })).rejects.toThrow(
      "'instructions' is not supported by this endpoint.",
    );
  });
});
