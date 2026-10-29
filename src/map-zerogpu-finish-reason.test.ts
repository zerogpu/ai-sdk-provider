import { describe, expect, it } from 'vitest';
import { mapZeroGPUFinishReason } from './map-zerogpu-finish-reason';

describe('mapZeroGPUFinishReason', () => {
  it.each([
    ['stop', 'stop'],
    ['length', 'length'],
    ['content_filter', 'content-filter'],
    ['tool_calls', 'tool-calls'],
    ['unknown', 'other'],
    [null, 'other'],
    [undefined, 'other'],
  ] as const)('should map %s to %s', (finishReason, expected) => {
    expect(mapZeroGPUFinishReason(finishReason)).toBe(expected);
  });
});
