import type {
  LanguageModelV4,
  LanguageModelV4CallOptions,
  LanguageModelV4Content,
  LanguageModelV4FinishReason,
  LanguageModelV4GenerateResult,
  LanguageModelV4StreamPart,
  LanguageModelV4StreamResult,
  SharedV4Warning,
} from '@ai-sdk/provider';
import {
  lazySchema,
  zodSchema,
  type InferSchema,
  combineHeaders,
  createEventSourceResponseHandler,
  createJsonResponseHandler,
  generateId,
  isCustomReasoning,
  mapReasoningToProviderEffort,
  parseProviderOptions,
  postJsonToApi,
  serializeModelOptions,
  StreamingToolCallTracker,
  WORKFLOW_SERIALIZE,
  WORKFLOW_DESERIALIZE,
  type FetchFunction,
  type ParseResult,
  type StreamingToolCallDelta,
} from '@ai-sdk/provider-utils';
import { z } from 'zod/v4';
import {
  convertZeroGPUUsage,
  zerogpuUsageSchema,
  type ZeroGPUUsage,
} from './convert-zerogpu-usage';
import { convertToZeroGPUChatMessages } from './convert-to-zerogpu-chat-messages';
import { getResponseMetadata } from './get-response-metadata';
import { mapZeroGPUFinishReason } from './map-zerogpu-finish-reason';
import {
  zerogpuLanguageModelOptionsValidator,
  type ZeroGPUChatModelId,
} from './zerogpu-chat-language-model-options';
import {
  zerogpuErrorDataObject,
  zerogpuFailedResponseHandler,
} from './zerogpu-error';
import { prepareTools } from './zerogpu-prepare-tools';

type ZeroGPUChatConfig = {
  provider: string;
  baseURL: string;
  headers?: () => Record<string, string | undefined>;
  fetch?: FetchFunction;
  generateId?: () => string;
};

export class ZeroGPUChatLanguageModel implements LanguageModelV4 {
  readonly specificationVersion = 'v4';

  readonly modelId: ZeroGPUChatModelId;

  private readonly config: ZeroGPUChatConfig;
  private readonly generateId: () => string;

  static [WORKFLOW_SERIALIZE](model: ZeroGPUChatLanguageModel) {
    return serializeModelOptions({
      modelId: model.modelId,
      config: model.config,
    });
  }

  static [WORKFLOW_DESERIALIZE](options: {
    modelId: ZeroGPUChatModelId;
    config: ZeroGPUChatConfig;
  }) {
    return new ZeroGPUChatLanguageModel(options.modelId, options.config);
  }

  constructor(modelId: ZeroGPUChatModelId, config: ZeroGPUChatConfig) {
    this.modelId = modelId;
    this.config = config;
    this.generateId = config.generateId ?? generateId;
  }

  get provider(): string {
    return this.config.provider;
  }

  readonly supportedUrls: Record<string, RegExp[]> = {
    'image/*': [/^https?:\/\/.*$/],
  };

  private async getArgs({
    prompt,
    maxOutputTokens,
    temperature,
    topP,
    topK,
    frequencyPenalty,
    presencePenalty,
    reasoning,
    stopSequences,
    responseFormat,
    seed,
    providerOptions,
    tools,
    toolChoice,
  }: LanguageModelV4CallOptions) {
    const warnings: SharedV4Warning[] = [];

    const options =
      (await parseProviderOptions({
        provider: 'zerogpu',
        providerOptions,
        schema: zerogpuLanguageModelOptionsValidator,
      })) ?? {};

    const reasoningEffort =
      options.reasoningEffort ??
      (reasoning === 'none'
        ? 'none'
        : isCustomReasoning(reasoning)
          ? mapReasoningToProviderEffort({
              reasoning,
              effortMap: {
                minimal: 'minimal',
                low: 'low',
                medium: 'medium',
                high: 'high',
                xhigh: 'high',
                max: 'high',
              },
              warnings,
            })
          : undefined);

    const {
      tools: zerogpuTools,
      toolChoice: zerogpuToolChoice,
      toolWarnings,
    } = prepareTools({
      tools,
      toolChoice,
    });

    return {
      args: {
        // model id:
        model: this.modelId,

        // standardized settings:
        max_tokens: maxOutputTokens,
        temperature,
        top_p: topP,
        top_k: topK,
        frequency_penalty: frequencyPenalty,
        presence_penalty: presencePenalty,
        stop: stopSequences,
        seed,
        reasoning_effort: reasoningEffort,

        // response format:
        response_format:
          responseFormat?.type === 'json'
            ? responseFormat.schema != null
              ? {
                  type: 'json_schema',
                  json_schema: {
                    schema: responseFormat.schema,
                    name: responseFormat.name ?? 'response',
                    description: responseFormat.description,
                  },
                }
              : { type: 'json_object' }
            : undefined,

        // zerogpu-specific provider options:
        metadata: options.metadata,
        categories: options.categories,
        max_keywords: options.maxKeywords,

        // messages:
        messages: convertToZeroGPUChatMessages(prompt),

        // tools:
        tools: zerogpuTools,
        tool_choice: zerogpuToolChoice,
      },
      warnings: [...warnings, ...toolWarnings],
    };
  }

  async doGenerate(
    options: LanguageModelV4CallOptions,
  ): Promise<LanguageModelV4GenerateResult> {
    const { args: body, warnings } = await this.getArgs(options);

    const {
      responseHeaders,
      value: response,
      rawValue: rawResponse,
    } = await postJsonToApi({
      url: `${this.config.baseURL}/chat/completions`,
      headers: combineHeaders(this.config.headers?.(), options.headers),
      body,
      failedResponseHandler: zerogpuFailedResponseHandler,
      successfulResponseHandler: createJsonResponseHandler(
        zerogpuChatResponseSchema,
      ),
      abortSignal: options.abortSignal,
      fetch: this.config.fetch,
    });

    const choice = response.choices[0];
    const content: Array<LanguageModelV4Content> = [];

    const text = choice.message.content;
    if (text != null && text.length > 0) {
      content.push({ type: 'text', text });
    }

    if (choice.message.tool_calls != null) {
      for (const toolCall of choice.message.tool_calls) {
        content.push({
          type: 'tool-call',
          toolCallId: toolCall.id ?? this.generateId(),
          toolName: toolCall.function.name,
          input: toolCall.function.arguments,
        });
      }
    }

    return {
      content,
      finishReason: {
        unified: mapZeroGPUFinishReason(choice.finish_reason),
        raw: choice.finish_reason ?? undefined,
      },
      usage: convertZeroGPUUsage(response.usage),
      request: { body },
      response: {
        ...getResponseMetadata(response),
        headers: responseHeaders,
        body: rawResponse,
      },
      warnings,
    };
  }

  async doStream(
    options: LanguageModelV4CallOptions,
  ): Promise<LanguageModelV4StreamResult> {
    const { args, warnings } = await this.getArgs(options);
    const body = {
      ...args,
      stream: true,
      stream_options: { include_usage: true },
    };

    const { responseHeaders, value: response } = await postJsonToApi({
      url: `${this.config.baseURL}/chat/completions`,
      headers: combineHeaders(this.config.headers?.(), options.headers),
      body,
      failedResponseHandler: zerogpuFailedResponseHandler,
      successfulResponseHandler: createEventSourceResponseHandler(
        zerogpuChatChunkSchema,
      ),
      abortSignal: options.abortSignal,
      fetch: this.config.fetch,
    });

    // A clean end without the provider's completion signal is interrupted.
    let finishReason: LanguageModelV4FinishReason = {
      unified: 'interrupted',
      raw: undefined,
    };
    let usage: ZeroGPUUsage | undefined = undefined;

    let isFirstChunk = true;
    let activeText = false;
    let toolCallTracker: StreamingToolCallTracker<StreamingToolCallDelta>;

    const generateId = this.generateId;

    return {
      stream: response.pipeThrough(
        new TransformStream<
          ParseResult<InferSchema<typeof zerogpuChatChunkSchema>>,
          LanguageModelV4StreamPart
        >({
          start(controller) {
            toolCallTracker = new StreamingToolCallTracker(controller, {
              generateId,
            });
            controller.enqueue({ type: 'stream-start', warnings });
          },

          transform(chunk, controller) {
            // Emit raw chunk if requested (before anything else)
            if (options.includeRawChunks) {
              controller.enqueue({ type: 'raw', rawValue: chunk.rawValue });
            }

            if (!chunk.success) {
              finishReason = { unified: 'error', raw: undefined };
              controller.enqueue({ type: 'error', error: chunk.error });
              return;
            }

            const value = chunk.value;

            // handle error chunks:
            if ('error' in value) {
              finishReason = { unified: 'error', raw: undefined };
              controller.enqueue({ type: 'error', error: value.error });
              return;
            }

            if (isFirstChunk) {
              isFirstChunk = false;

              controller.enqueue({
                type: 'response-metadata',
                ...getResponseMetadata(value),
              });
            }

            if (value.usage != null) {
              usage = value.usage;
            }

            // the final usage chunk has no choices:
            const choice = value.choices[0];
            if (choice == null) {
              return;
            }

            const delta = choice.delta;

            if (delta?.content != null && delta.content.length > 0) {
              if (!activeText) {
                controller.enqueue({ type: 'text-start', id: '0' });
                activeText = true;
              }

              controller.enqueue({
                type: 'text-delta',
                id: '0',
                delta: delta.content,
              });
            }

            if (delta?.tool_calls != null) {
              for (const toolCall of delta.tool_calls) {
                toolCallTracker.processDelta(toolCall);
              }
            }

            if (choice.finish_reason != null) {
              finishReason = {
                unified: mapZeroGPUFinishReason(choice.finish_reason),
                raw: choice.finish_reason,
              };
            }
          },

          flush(controller) {
            if (activeText) {
              controller.enqueue({ type: 'text-end', id: '0' });
            }

            toolCallTracker.flush();

            controller.enqueue({
              type: 'finish',
              finishReason,
              usage: convertZeroGPUUsage(usage),
            });
          },
        }),
      ),
      request: { body },
      response: { headers: responseHeaders },
    };
  }
}

// limited version of the schema, focussed on what is needed for the implementation
// this approach limits breakages when the API changes and increases efficiency
const zerogpuChatResponseSchema = lazySchema(() =>
  zodSchema(
    z.object({
      id: z.string().nullish(),
      created: z.number().nullish(),
      model: z.string().nullish(),
      choices: z.array(
        z.object({
          message: z.object({
            role: z.literal('assistant').nullish(),
            content: z.string().nullish(),
            tool_calls: z
              .array(
                z.object({
                  id: z.string().nullish(),
                  function: z.object({
                    name: z.string(),
                    arguments: z.string(),
                  }),
                }),
              )
              .nullish(),
          }),
          finish_reason: z.string().nullish(),
        }),
      ),
      usage: zerogpuUsageSchema.nullish(),
    }),
  ),
);

// limited version of the schema, focussed on what is needed for the implementation
// this approach limits breakages when the API changes and increases efficiency
const zerogpuChatChunkSchema = lazySchema(() =>
  zodSchema(
    z.union([
      z.object({
        id: z.string().nullish(),
        created: z.number().nullish(),
        model: z.string().nullish(),
        choices: z.array(
          z.object({
            delta: z
              .object({
                role: z.enum(['assistant']).nullish(),
                content: z.string().nullish(),
                tool_calls: z
                  .array(
                    z.object({
                      index: z.number().nullish(),
                      id: z.string().nullish(),
                      function: z
                        .object({
                          name: z.string().nullish(),
                          arguments: z.string().nullish(),
                        })
                        .nullish(),
                    }),
                  )
                  .nullish(),
              })
              .nullish(),
            finish_reason: z.string().nullish(),
          }),
        ),
        usage: zerogpuUsageSchema.nullish(),
      }),
      zerogpuErrorDataObject(),
    ]),
  ),
);
