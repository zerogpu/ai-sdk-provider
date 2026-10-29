import type { SharedV4Warning, SpeechModelV4 } from '@ai-sdk/provider';
import {
  combineHeaders,
  createBinaryResponseHandler,
  parseProviderOptions,
  postJsonToApi,
  serializeModelOptions,
  WORKFLOW_DESERIALIZE,
  WORKFLOW_SERIALIZE,
  type FetchFunction,
} from '@ai-sdk/provider-utils';
import { zerogpuFailedResponseHandler } from './zerogpu-error';
import {
  zerogpuSpeechModelOptions,
  type ZeroGPUSpeechModelId,
} from './zerogpu-speech-model-options';

interface ZeroGPUSpeechModelConfig {
  provider: string;
  baseURL: string;
  headers?: () => Record<string, string | undefined>;
  fetch?: FetchFunction;
  _internal?: {
    currentDate?: () => Date;
  };
}

const supportedOutputFormats = ['mp3', 'wav', 'opus', 'aac', 'flac', 'pcm'];

export class ZeroGPUSpeechModel implements SpeechModelV4 {
  readonly specificationVersion = 'v4';

  static [WORKFLOW_SERIALIZE](model: ZeroGPUSpeechModel) {
    return serializeModelOptions({
      modelId: model.modelId,
      config: model.config,
    });
  }

  static [WORKFLOW_DESERIALIZE](options: {
    modelId: ZeroGPUSpeechModelId;
    config: ZeroGPUSpeechModelConfig;
  }) {
    return new ZeroGPUSpeechModel(options.modelId, options.config);
  }

  get provider(): string {
    return this.config.provider;
  }

  constructor(
    readonly modelId: ZeroGPUSpeechModelId,
    private readonly config: ZeroGPUSpeechModelConfig,
  ) {}

  private async getArgs({
    text,
    voice = 'alloy',
    outputFormat = 'mp3',
    instructions,
    speed,
    language,
    providerOptions,
  }: Parameters<SpeechModelV4['doGenerate']>[0]) {
    const warnings: SharedV4Warning[] = [];
    const zerogpuOptions = await parseProviderOptions({
      provider: 'zerogpu',
      providerOptions,
      schema: zerogpuSpeechModelOptions,
    });

    let responseFormat = 'mp3';
    if (supportedOutputFormats.includes(outputFormat)) {
      responseFormat = outputFormat;
    } else {
      warnings.push({
        type: 'unsupported',
        feature: 'outputFormat',
        details: `Unsupported output format: ${outputFormat}. Using mp3 instead.`,
      });
    }

    if (instructions != null) {
      warnings.push({
        type: 'unsupported',
        feature: 'instructions',
        details:
          'ZeroGPU speech models do not support the `instructions` option. ' +
          'Use inline tags such as [laugh] in the text to guide delivery.',
      });
    }

    if (speed != null) {
      warnings.push({
        type: 'unsupported',
        feature: 'speed',
        details:
          'ZeroGPU speech models do not support the `speed` option. It was ignored.',
      });
    }

    if (language != null) {
      warnings.push({
        type: 'unsupported',
        feature: 'language',
        details:
          'ZeroGPU speech models do not support the `language` option. It was ignored.',
      });
    }

    const requestBody = {
      model: this.modelId,
      input: text,
      voice,
      response_format: responseFormat,
      seed: zerogpuOptions?.seed,
      temperature: zerogpuOptions?.temperature,
      top_p: zerogpuOptions?.topP,
      top_k: zerogpuOptions?.topK,
      repetition_penalty: zerogpuOptions?.repetitionPenalty,
    };

    return { requestBody, warnings };
  }

  async doGenerate(
    options: Parameters<SpeechModelV4['doGenerate']>[0],
  ): Promise<Awaited<ReturnType<SpeechModelV4['doGenerate']>>> {
    const currentDate = this.config._internal?.currentDate?.() ?? new Date();
    const { requestBody, warnings } = await this.getArgs(options);

    const {
      value: audio,
      responseHeaders,
      rawValue: rawResponse,
    } = await postJsonToApi({
      url: `${this.config.baseURL}/audio/speech`,
      headers: combineHeaders(this.config.headers?.(), options.headers),
      body: requestBody,
      failedResponseHandler: zerogpuFailedResponseHandler,
      successfulResponseHandler: createBinaryResponseHandler(),
      abortSignal: options.abortSignal,
      fetch: this.config.fetch,
    });

    return {
      audio,
      warnings,
      request: {
        body: JSON.stringify(requestBody),
      },
      response: {
        timestamp: currentDate,
        modelId: this.modelId,
        headers: responseHeaders,
        body: rawResponse,
      },
    };
  }
}
