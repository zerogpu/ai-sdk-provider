import { UnsupportedFunctionalityError } from '@ai-sdk/provider';
import { describe, expect, it } from 'vitest';
import { convertToZeroGPUChatMessages } from './convert-to-zerogpu-chat-messages';

describe('system messages', () => {
  it('should convert system messages', () => {
    expect(
      convertToZeroGPUChatMessages([
        { role: 'system', content: '[Sports,Technology,Finance]' },
      ]),
    ).toStrictEqual([
      { role: 'system', content: '[Sports,Technology,Finance]' },
    ]);
  });
});

describe('user messages', () => {
  it('should join text-only content into a string', () => {
    expect(
      convertToZeroGPUChatMessages([
        {
          role: 'user',
          content: [
            { type: 'text', text: 'Hello, ' },
            { type: 'text', text: 'world' },
          ],
        },
      ]),
    ).toStrictEqual([{ role: 'user', content: 'Hello, world' }]);
  });

  it('should convert image data to a base64 data url', () => {
    expect(
      convertToZeroGPUChatMessages([
        {
          role: 'user',
          content: [
            { type: 'text', text: 'What is this?' },
            {
              type: 'file',
              mediaType: 'image/png',
              data: { type: 'data', data: new Uint8Array([0, 1, 2, 3]) },
            },
          ],
        },
      ]),
    ).toStrictEqual([
      {
        role: 'user',
        content: [
          { type: 'text', text: 'What is this?' },
          {
            type: 'image_url',
            image_url: { url: 'data:image/png;base64,AAECAw==' },
          },
        ],
      },
    ]);
  });

  it('should pass image urls through', () => {
    expect(
      convertToZeroGPUChatMessages([
        {
          role: 'user',
          content: [
            {
              type: 'file',
              mediaType: 'image/jpeg',
              data: {
                type: 'url',
                url: new URL('https://example.com/image.jpg'),
              },
            },
          ],
        },
      ]),
    ).toStrictEqual([
      {
        role: 'user',
        content: [
          {
            type: 'image_url',
            image_url: { url: 'https://example.com/image.jpg' },
          },
        ],
      },
    ]);
  });

  it('should throw for non-image files', () => {
    expect(() =>
      convertToZeroGPUChatMessages([
        {
          role: 'user',
          content: [
            {
              type: 'file',
              mediaType: 'application/pdf',
              data: { type: 'data', data: 'AAECAw==' },
            },
          ],
        },
      ]),
    ).toThrow(UnsupportedFunctionalityError);
  });
});

describe('assistant messages', () => {
  it('should convert text and tool calls and drop reasoning', () => {
    expect(
      convertToZeroGPUChatMessages([
        {
          role: 'assistant',
          content: [
            { type: 'reasoning', text: 'I should check the weather.' },
            { type: 'text', text: 'Let me check.' },
            {
              type: 'tool-call',
              toolCallId: 'call-1',
              toolName: 'weather',
              input: { city: 'Paris' },
            },
          ],
        },
      ]),
    ).toStrictEqual([
      {
        role: 'assistant',
        content: 'Let me check.',
        tool_calls: [
          {
            id: 'call-1',
            type: 'function',
            function: { name: 'weather', arguments: '{"city":"Paris"}' },
          },
        ],
      },
    ]);
  });
});

describe('tool messages', () => {
  it('should convert every tool output type', () => {
    const result = convertToZeroGPUChatMessages([
      {
        role: 'tool',
        content: [
          {
            type: 'tool-result',
            toolCallId: 'call-1',
            toolName: 'weather',
            output: { type: 'json', value: { tempC: 21 } },
          },
          {
            type: 'tool-result',
            toolCallId: 'call-2',
            toolName: 'weather',
            output: { type: 'text', value: 'sunny' },
          },
          {
            type: 'tool-result',
            toolCallId: 'call-3',
            toolName: 'weather',
            output: { type: 'error-text', value: 'failed' },
          },
          {
            type: 'tool-result',
            toolCallId: 'call-4',
            toolName: 'weather',
            output: { type: 'error-json', value: { error: 'failed' } },
          },
          {
            type: 'tool-result',
            toolCallId: 'call-5',
            toolName: 'weather',
            output: { type: 'execution-denied' },
          },
          {
            type: 'tool-result',
            toolCallId: 'call-6',
            toolName: 'weather',
            output: {
              type: 'content',
              value: [{ type: 'text', text: 'sunny' }],
            },
          },
        ],
      },
    ]);

    expect(result).toStrictEqual([
      { role: 'tool', tool_call_id: 'call-1', content: '{"tempC":21}' },
      { role: 'tool', tool_call_id: 'call-2', content: 'sunny' },
      { role: 'tool', tool_call_id: 'call-3', content: 'failed' },
      { role: 'tool', tool_call_id: 'call-4', content: '{"error":"failed"}' },
      {
        role: 'tool',
        tool_call_id: 'call-5',
        content: 'Tool call execution denied.',
      },
      {
        role: 'tool',
        tool_call_id: 'call-6',
        content: '[{"type":"text","text":"sunny"}]',
      },
    ]);
  });
});
