import {
  lazySchema,
  zodSchema,
  type InferSchema,
  createJsonErrorResponseHandler,
} from '@ai-sdk/provider-utils';
import { z } from 'zod/v4';

export const zerogpuErrorDataObject = () =>
  z.object({
    error: z.object({
      message: z.string(),
      type: z.string().nullish(),
      param: z.any().nullish(),
      code: z.union([z.string(), z.number()]).nullish(),
    }),
  });

const zerogpuErrorDataSchema = lazySchema(() =>
  zodSchema(zerogpuErrorDataObject()),
);

export type ZeroGPUErrorData = InferSchema<typeof zerogpuErrorDataSchema>;

export const zerogpuFailedResponseHandler = createJsonErrorResponseHandler({
  errorSchema: zerogpuErrorDataSchema,
  errorToMessage: data => data.error.message,
});
