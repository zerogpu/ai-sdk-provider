import { describe, expect, it } from 'vitest';
import { convertZeroGPUUsage } from './convert-zerogpu-usage';

describe('convertZeroGPUUsage', () => {
  it('should return undefined token counts when usage is missing', () => {
    expect(convertZeroGPUUsage(undefined)).toStrictEqual({
      inputTokens: {
        total: undefined,
        noCache: undefined,
        cacheRead: undefined,
        cacheWrite: undefined,
      },
      outputTokens: { total: undefined, text: undefined, reasoning: undefined },
      raw: undefined,
    });
  });

  it('should map cached and reasoning tokens', () => {
    const usage = {
      prompt_tokens: 100,
      completion_tokens: 50,
      total_tokens: 150,
      prompt_tokens_details: { cached_tokens: 40 },
      completion_tokens_details: { reasoning_tokens: 30 },
    };

    expect(convertZeroGPUUsage(usage)).toStrictEqual({
      inputTokens: {
        total: 100,
        noCache: 60,
        cacheRead: 40,
        cacheWrite: undefined,
      },
      outputTokens: { total: 50, text: 20, reasoning: 30 },
      raw: usage,
    });
  });

  it('should handle usage without details', () => {
    const usage = { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 };

    expect(convertZeroGPUUsage(usage)).toStrictEqual({
      inputTokens: {
        total: 10,
        noCache: 10,
        cacheRead: undefined,
        cacheWrite: undefined,
      },
      outputTokens: { total: 5, text: 5, reasoning: undefined },
      raw: usage,
    });
  });
});
