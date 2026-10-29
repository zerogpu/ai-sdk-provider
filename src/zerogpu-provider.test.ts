import { NoSuchModelError } from '@ai-sdk/provider';
import { createTestServer } from '@ai-sdk/test-server/with-vitest';
import { describe, expect, it, vi } from 'vitest';
import { ZeroGPUChatLanguageModel } from './zerogpu-chat-language-model';
import { ZeroGPUEmbeddingModel } from './zerogpu-embedding-model';
import { createZeroGPU } from './zerogpu-provider';
import { ZeroGPUSpeechModel } from './zerogpu-speech-model';
import { ZeroGPUTranscriptionModel } from './zerogpu-transcription-model';

vi.mock('./version', () => ({
  VERSION: '0.0.0-test',
}));

const server = createTestServer({
  'https://api.zerogpu.ai/v1/chat/completions': {
    response: {
      type: 'json-value',
      body: {
        choices: [
          {
            message: { role: 'assistant', content: 'Hello' },
            finish_reason: 'stop',
          },
        ],
      },
    },
  },
  'https://custom.example.com/v1/chat/completions': {
    response: {
      type: 'json-value',
      body: {
        choices: [
          {
            message: { role: 'assistant', content: 'Hello' },
            finish_reason: 'stop',
          },
        ],
      },
    },
  },
});

const prompt = [
  { role: 'user' as const, content: [{ type: 'text' as const, text: 'Hi' }] },
];

describe('createZeroGPU', () => {
  const provider = createZeroGPU({ apiKey: 'test-api-key' });

  it('should create chat models', () => {
    const model = provider('gpt-4.1-mini');

    expect(model).toBeInstanceOf(ZeroGPUChatLanguageModel);
    expect(model.provider).toBe('zerogpu.chat');
    expect(model.modelId).toBe('gpt-4.1-mini');
    expect(provider.languageModel('gpt-4.1-mini')).toBeInstanceOf(
      ZeroGPUChatLanguageModel,
    );
    expect(provider.chat('gpt-4.1-mini')).toBeInstanceOf(
      ZeroGPUChatLanguageModel,
    );
  });

  it('should create embedding models', () => {
    const model = provider.embeddingModel('bge-small-en-v1.5');

    expect(model).toBeInstanceOf(ZeroGPUEmbeddingModel);
    expect(model.provider).toBe('zerogpu.embedding');
    expect(provider.embedding('bge-small-en-v1.5')).toBeInstanceOf(
      ZeroGPUEmbeddingModel,
    );
  });

  it('should create speech models', () => {
    const model = provider.speechModel('chatterbox-nano');

    expect(model).toBeInstanceOf(ZeroGPUSpeechModel);
    expect(model.provider).toBe('zerogpu.speech');
    expect(provider.speech('chatterbox-nano')).toBeInstanceOf(
      ZeroGPUSpeechModel,
    );
  });

  it('should create transcription models', () => {
    const model = provider.transcriptionModel('whisper-tiny');

    expect(model).toBeInstanceOf(ZeroGPUTranscriptionModel);
    expect(model.provider).toBe('zerogpu.transcription');
    expect(provider.transcription('whisper-tiny')).toBeInstanceOf(
      ZeroGPUTranscriptionModel,
    );
  });

  it('should throw NoSuchModelError for image models', () => {
    expect(() => provider.imageModel('any')).toThrow(NoSuchModelError);
  });

  it('should not be callable with the new keyword', () => {
    expect(
      () => new (provider as unknown as new (modelId: string) => unknown)('x'),
    ).toThrow(
      'The ZeroGPU model function cannot be called with the new keyword.',
    );
  });

  it('should have specification version v4', () => {
    expect(provider.specificationVersion).toBe('v4');
  });
});

describe('request configuration', () => {
  it('should load the API key from the environment', async () => {
    vi.stubEnv('ZEROGPU_API_KEY', 'env-api-key');

    try {
      await createZeroGPU().chat('gpt-4.1-mini').doGenerate({ prompt });
    } finally {
      vi.unstubAllEnvs();
    }

    expect(server.calls[0].requestHeaders.authorization).toBe(
      'Bearer env-api-key',
    );
  });

  it('should only send the project id header when configured', async () => {
    await createZeroGPU({ apiKey: 'test-api-key' })
      .chat('gpt-4.1-mini')
      .doGenerate({ prompt });
    await createZeroGPU({ apiKey: 'test-api-key', projectId: 'project-1' })
      .chat('gpt-4.1-mini')
      .doGenerate({ prompt });

    expect(server.calls[0].requestHeaders).not.toHaveProperty('x-project-id');
    expect(server.calls[1].requestHeaders['x-project-id']).toBe('project-1');
  });

  it('should use a custom base URL', async () => {
    await createZeroGPU({
      apiKey: 'test-api-key',
      baseURL: 'https://custom.example.com/v1/',
    })
      .chat('gpt-4.1-mini')
      .doGenerate({ prompt });

    expect(server.calls[0].requestUrl).toBe(
      'https://custom.example.com/v1/chat/completions',
    );
  });

  it('should use a custom fetch implementation', async () => {
    const customFetch = vi.fn(fetch);

    await createZeroGPU({ apiKey: 'test-api-key', fetch: customFetch })
      .chat('gpt-4.1-mini')
      .doGenerate({ prompt });

    expect(customFetch).toHaveBeenCalledOnce();
  });
});
