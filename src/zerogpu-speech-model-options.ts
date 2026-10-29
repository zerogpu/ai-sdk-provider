import {
  lazySchema,
  zodSchema,
  type InferSchema,
} from '@ai-sdk/provider-utils';
import { z } from 'zod/v4';

// https://docs.zerogpu.ai/api-reference/audio-speech
export type ZeroGPUSpeechModelId = 'chatterbox-nano' | (string & {});

export const zerogpuSpeechModelOptions = lazySchema(() =>
  zodSchema(
    z.object({
      /**
       * Seed for reproducible generation.
       */
      seed: z.number().int().optional(),

      /**
       * Sampling temperature.
       */
      temperature: z.number().optional(),

      /**
       * Nucleus sampling probability.
       */
      topP: z.number().optional(),

      /**
       * Top-k sampling.
       */
      topK: z.number().int().optional(),

      /**
       * Penalty applied to repeated tokens.
       */
      repetitionPenalty: z.number().optional(),
    }),
  ),
);

export type ZeroGPUSpeechModelOptions = InferSchema<
  typeof zerogpuSpeechModelOptions
>;
