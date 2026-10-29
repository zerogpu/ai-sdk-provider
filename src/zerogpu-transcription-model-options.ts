import {
  lazySchema,
  zodSchema,
  type InferSchema,
} from '@ai-sdk/provider-utils';
import { z } from 'zod/v4';

// https://docs.zerogpu.ai/api-reference/audio-transcriptions
export type ZeroGPUTranscriptionModelId = 'whisper-tiny' | (string & {});

export const zerogpuTranscriptionModelOptions = lazySchema(() =>
  zodSchema(
    z.object({
      /**
       * The language of the input audio in ISO-639-1 format, e.g. `en`.
       * Supplying the language improves accuracy and latency.
       */
      language: z.string().min(1).optional(),

      /**
       * The sampling temperature, between 0 and 1.
       */
      temperature: z.number().min(0).max(1).optional(),
    }),
  ),
);

export type ZeroGPUTranscriptionModelOptions = InferSchema<
  typeof zerogpuTranscriptionModelOptions
>;
