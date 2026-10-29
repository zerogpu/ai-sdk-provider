import {
  NoSuchModelError,
  type EmbeddingModelV4,
  type LanguageModelV4,
  type ProviderV4,
  type SpeechModelV4,
  type TranscriptionModelV4,
} from '@ai-sdk/provider';
import {
  loadApiKey,
  withoutTrailingSlash,
  withUserAgentSuffix,
  type FetchFunction,
} from '@ai-sdk/provider-utils';
import { ZeroGPUChatLanguageModel } from './zerogpu-chat-language-model';
import type { ZeroGPUChatModelId } from './zerogpu-chat-language-model-options';
import { ZeroGPUEmbeddingModel } from './zerogpu-embedding-model';
import type { ZeroGPUEmbeddingModelId } from './zerogpu-embedding-model-options';
import { ZeroGPUSpeechModel } from './zerogpu-speech-model';
import type { ZeroGPUSpeechModelId } from './zerogpu-speech-model-options';
import { ZeroGPUTranscriptionModel } from './zerogpu-transcription-model';
import type { ZeroGPUTranscriptionModelId } from './zerogpu-transcription-model-options';
import { VERSION } from './version';

export interface ZeroGPUProvider extends ProviderV4 {
  (modelId: ZeroGPUChatModelId): LanguageModelV4;

  /**
   * Creates a model for text generation.
   */
  languageModel(modelId: ZeroGPUChatModelId): LanguageModelV4;

  /**
   * Creates a model for text generation.
   */
  chat(modelId: ZeroGPUChatModelId): LanguageModelV4;

  /**
   * Creates a model for text embeddings.
   */
  embedding(modelId: ZeroGPUEmbeddingModelId): EmbeddingModelV4;

  /**
   * Creates a model for text embeddings.
   */
  embeddingModel: (modelId: ZeroGPUEmbeddingModelId) => EmbeddingModelV4;

  /**
   * Creates a model for speech generation (text-to-speech).
   */
  speech(modelId: ZeroGPUSpeechModelId): SpeechModelV4;

  /**
   * Creates a model for speech generation (text-to-speech).
   */
  speechModel(modelId: ZeroGPUSpeechModelId): SpeechModelV4;

  /**
   * Creates a model for audio transcription.
   */
  transcription(modelId: ZeroGPUTranscriptionModelId): TranscriptionModelV4;

  /**
   * Creates a model for audio transcription.
   */
  transcriptionModel(
    modelId: ZeroGPUTranscriptionModelId,
  ): TranscriptionModelV4;
}

export interface ZeroGPUProviderSettings {
  /**
   * Use a different URL prefix for API calls, e.g. to use proxy servers.
   * The default prefix is `https://api.zerogpu.ai/v1`.
   */
  baseURL?: string;

  /**
   * API key that is being sent using the `Authorization` header.
   * It defaults to the `ZEROGPU_API_KEY` environment variable.
   */
  apiKey?: string;

  /**
   * ZeroGPU project ID, sent using the `x-project-id` header.
   */
  projectId?: string;

  /**
   * Custom headers to include in the requests.
   */
  headers?: Record<string, string>;

  /**
   * Custom fetch implementation. You can use it as a middleware to intercept requests,
   * or to provide a custom fetch implementation for e.g. testing.
   */
  fetch?: FetchFunction;

  generateId?: () => string;
}

/**
 * Create a ZeroGPU provider instance.
 */
export function createZeroGPU(
  options: ZeroGPUProviderSettings = {},
): ZeroGPUProvider {
  const baseURL =
    withoutTrailingSlash(options.baseURL) ?? 'https://api.zerogpu.ai/v1';

  const getHeaders = () =>
    withUserAgentSuffix(
      {
        Authorization: `Bearer ${loadApiKey({
          apiKey: options.apiKey,
          environmentVariableName: 'ZEROGPU_API_KEY',
          description: 'ZeroGPU',
        })}`,
        'x-project-id': options.projectId,
        ...options.headers,
      },
      `ai-sdk-zerogpu/${VERSION}`,
    );

  const createChatModel = (modelId: ZeroGPUChatModelId) =>
    new ZeroGPUChatLanguageModel(modelId, {
      provider: 'zerogpu.chat',
      baseURL,
      headers: getHeaders,
      fetch: options.fetch,
      generateId: options.generateId,
    });

  const createEmbeddingModel = (modelId: ZeroGPUEmbeddingModelId) =>
    new ZeroGPUEmbeddingModel(modelId, {
      provider: 'zerogpu.embedding',
      baseURL,
      headers: getHeaders,
      fetch: options.fetch,
    });

  const createSpeechModel = (modelId: ZeroGPUSpeechModelId) =>
    new ZeroGPUSpeechModel(modelId, {
      provider: 'zerogpu.speech',
      baseURL,
      headers: getHeaders,
      fetch: options.fetch,
    });

  const createTranscriptionModel = (modelId: ZeroGPUTranscriptionModelId) =>
    new ZeroGPUTranscriptionModel(modelId, {
      provider: 'zerogpu.transcription',
      baseURL,
      headers: getHeaders,
      fetch: options.fetch,
    });

  const provider = function (modelId: ZeroGPUChatModelId) {
    if (new.target) {
      throw new Error(
        'The ZeroGPU model function cannot be called with the new keyword.',
      );
    }

    return createChatModel(modelId);
  };

  provider.specificationVersion = 'v4' as const;
  provider.languageModel = createChatModel;
  provider.chat = createChatModel;
  provider.embedding = createEmbeddingModel;
  provider.embeddingModel = createEmbeddingModel;
  provider.speech = createSpeechModel;
  provider.speechModel = createSpeechModel;
  provider.transcription = createTranscriptionModel;
  provider.transcriptionModel = createTranscriptionModel;

  provider.imageModel = (modelId: string) => {
    throw new NoSuchModelError({ modelId, modelType: 'imageModel' });
  };

  return provider;
}

/**
 * Default ZeroGPU provider instance.
 */
export const zerogpu = createZeroGPU();
