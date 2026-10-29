import { lazySchema, zodSchema } from '@ai-sdk/provider-utils';
import { z } from 'zod/v4';

// https://docs.zerogpu.ai/docs/model-catalog
export type ZeroGPUChatModelId =
  // text generation:
  | 'deepseek-v4.1-flash'
  | 'glm-5.3-flash'
  | 'gpt-4.1-mini'
  | 'gpt-5.4-nano'
  | 'gpt-5.6-luna'
  | 'gpt-6-luna'
  | 'gpt-oss-120b'
  | 'qwen3-30b-a3b-fp8'
  | 'qwen3.6-35b-a3b'
  | 'llama-3.1-8b-instruct-fast'
  | 'llama-3.2-3b-instruct-fast'
  | 'LFM2.5-1.2B-Instruct'
  | 'LFM2.5-1.2B-Thinking'
  | 'zlm-v1-tool-decisions'
  // classification and extraction (JSON output, no streaming):
  | 'zlm-v1-iab-classify-edge'
  | 'zlm-v1-iab-domain-classifier'
  | 'zlm-v2-iab-classify-edge-enriched'
  | 'zlm-v1-signal-extract'
  | 'gliner2.5-base-v1'
  | 'gliner2.5-multi-v1'
  | 'gliner2.5-decide-v1'
  | 'deberta-v3-small'
  | (string & {});

export const zerogpuLanguageModelOptions = () =>
  z.object({
    /**
     * Controls the reasoning effort for models that support adjustable
     * reasoning.
     */
    reasoningEffort: z
      .enum(['none', 'minimal', 'low', 'medium', 'high'])
      .optional(),

    /**
     * Task configuration for the GLiNER models (`gliner2.5-*`), e.g.
     * `{ usecase: 'ner', labels: ['person', 'company'], threshold: 0.3 }`,
     * `{ usecase: 'json', schema: { ... } }` or
     * `{ usecase: 'classification', schema: { ... } }`.
     */
    metadata: z.record(z.string(), z.unknown()).optional(),

    /**
     * Signal categories to extract with `zlm-v1-signal-extract`.
     */
    categories: z.array(z.string()).optional(),

    /**
     * Maximum number of keywords to return with `zlm-v1-signal-extract`.
     */
    maxKeywords: z.number().int().positive().optional(),
  });

export const zerogpuLanguageModelOptionsValidator = lazySchema(() =>
  zodSchema(zerogpuLanguageModelOptions()),
);

export type ZeroGPULanguageModelOptions = z.infer<
  ReturnType<typeof zerogpuLanguageModelOptions>
>;
