import type { TranscriptionModelV4 } from '@ai-sdk/provider';
import {
  lazySchema,
  zodSchema,
  combineHeaders,
  convertBase64ToUint8Array,
  createJsonResponseHandler,
  mediaTypeToExtension,
  parseProviderOptions,
  postFormDataToApi,
  serializeModelOptions,
  WORKFLOW_DESERIALIZE,
  WORKFLOW_SERIALIZE,
  type FetchFunction,
} from '@ai-sdk/provider-utils';
import { z } from 'zod/v4';
import { zerogpuFailedResponseHandler } from './zerogpu-error';
import {
  zerogpuTranscriptionModelOptions,
  type ZeroGPUTranscriptionModelId,
} from './zerogpu-transcription-model-options';

type ZeroGPUTranscriptionModelConfig = {
  provider: string;
  baseURL: string;
  headers?: () => Record<string, string | undefined>;
  fetch?: FetchFunction;
  _internal?: {
    currentDate?: () => Date;
  };
};

// Whisper reports the detected language by name; map it to ISO-639-1.
// https://platform.openai.com/docs/guides/speech-to-text#supported-languages
const languageMap = {
  afrikaans: 'af',
  arabic: 'ar',
  armenian: 'hy',
  azerbaijani: 'az',
  belarusian: 'be',
  bosnian: 'bs',
  bulgarian: 'bg',
  catalan: 'ca',
  chinese: 'zh',
  croatian: 'hr',
  czech: 'cs',
  danish: 'da',
  dutch: 'nl',
  english: 'en',
  estonian: 'et',
  finnish: 'fi',
  french: 'fr',
  galician: 'gl',
  german: 'de',
  greek: 'el',
  hebrew: 'he',
  hindi: 'hi',
  hungarian: 'hu',
  icelandic: 'is',
  indonesian: 'id',
  italian: 'it',
  japanese: 'ja',
  kannada: 'kn',
  kazakh: 'kk',
  korean: 'ko',
  latvian: 'lv',
  lithuanian: 'lt',
  macedonian: 'mk',
  malay: 'ms',
  marathi: 'mr',
  maori: 'mi',
  nepali: 'ne',
  norwegian: 'no',
  persian: 'fa',
  polish: 'pl',
  portuguese: 'pt',
  romanian: 'ro',
  russian: 'ru',
  serbian: 'sr',
  slovak: 'sk',
  slovenian: 'sl',
  spanish: 'es',
  swahili: 'sw',
  swedish: 'sv',
  tagalog: 'tl',
  tamil: 'ta',
  thai: 'th',
  turkish: 'tr',
  ukrainian: 'uk',
  urdu: 'ur',
  vietnamese: 'vi',
  welsh: 'cy',
};

function isLanguageCode(
  language: string | null | undefined,
): language is string {
  return language != null && /^[a-z]{2}$/.test(language);
}

function toLanguageCode(
  language: string | null | undefined,
): string | undefined {
  if (isLanguageCode(language)) {
    return language;
  }

  return language != null &&
    Object.prototype.hasOwnProperty.call(languageMap, language)
    ? languageMap[language as keyof typeof languageMap]
    : undefined;
}

export class ZeroGPUTranscriptionModel implements TranscriptionModelV4 {
  readonly specificationVersion = 'v4';

  static [WORKFLOW_SERIALIZE](model: ZeroGPUTranscriptionModel) {
    return serializeModelOptions({
      modelId: model.modelId,
      config: model.config,
    });
  }

  static [WORKFLOW_DESERIALIZE](options: {
    modelId: ZeroGPUTranscriptionModelId;
    config: ZeroGPUTranscriptionModelConfig;
  }) {
    return new ZeroGPUTranscriptionModel(options.modelId, options.config);
  }

  get provider(): string {
    return this.config.provider;
  }

  constructor(
    readonly modelId: ZeroGPUTranscriptionModelId,
    private readonly config: ZeroGPUTranscriptionModelConfig,
  ) {}

  private async getArgs({
    audio,
    mediaType,
    providerOptions,
  }: Parameters<TranscriptionModelV4['doGenerate']>[0]) {
    const zerogpuOptions = await parseProviderOptions({
      provider: 'zerogpu',
      providerOptions,
      schema: zerogpuTranscriptionModelOptions,
    });

    const formData = new FormData();
    const blob =
      audio instanceof Uint8Array
        ? new Blob([audio as Uint8Array<ArrayBuffer>])
        : new Blob([convertBase64ToUint8Array(audio)]);

    formData.append('model', this.modelId);
    formData.append(
      'file',
      new File([blob], 'audio', { type: mediaType }),
      `audio.${mediaTypeToExtension(mediaType)}`,
    );
    formData.append('response_format', 'verbose_json');

    if (zerogpuOptions?.language != null) {
      formData.append('language', zerogpuOptions.language);
    }
    if (zerogpuOptions?.temperature != null) {
      formData.append('temperature', String(zerogpuOptions.temperature));
    }

    return { formData, warnings: [] };
  }

  async doGenerate(
    options: Parameters<TranscriptionModelV4['doGenerate']>[0],
  ): Promise<Awaited<ReturnType<TranscriptionModelV4['doGenerate']>>> {
    const currentDate = this.config._internal?.currentDate?.() ?? new Date();
    const { formData, warnings } = await this.getArgs(options);

    const {
      value: response,
      responseHeaders,
      rawValue: rawResponse,
    } = await postFormDataToApi({
      url: `${this.config.baseURL}/audio/transcriptions`,
      headers: combineHeaders(this.config.headers?.(), options.headers),
      formData,
      failedResponseHandler: zerogpuFailedResponseHandler,
      successfulResponseHandler: createJsonResponseHandler(
        zerogpuTranscriptionResponseSchema,
      ),
      abortSignal: options.abortSignal,
      fetch: this.config.fetch,
    });

    const segments =
      response.segments?.map(segment => ({
        text: segment.text,
        startSecond: segment.start,
        endSecond: segment.end,
      })) ?? [];

    return {
      text: response.text,
      segments,
      language: toLanguageCode(response.language),
      durationInSeconds:
        response.duration ?? segments.at(-1)?.endSecond ?? undefined,
      warnings,
      response: {
        timestamp: currentDate,
        modelId: this.modelId,
        headers: responseHeaders,
        body: rawResponse,
      },
    };
  }
}

// minimal version of the schema, focussed on what is needed for the implementation
// this approach limits breakages when the API changes and increases efficiency
const zerogpuTranscriptionResponseSchema = lazySchema(() =>
  zodSchema(
    z.object({
      text: z.string(),
      language: z.string().nullish(),
      duration: z.number().nullish(),
      segments: z
        .array(
          z.object({
            text: z.string(),
            start: z.number(),
            end: z.number(),
          }),
        )
        .nullish(),
    }),
  ),
);
