import { APICallError, type LanguageModelV4Prompt } from '@ai-sdk/provider';
import {
  convertReadableStreamToArray,
  mockId,
} from '@ai-sdk/provider-utils/test';
import { createTestServer } from '@ai-sdk/test-server/with-vitest';
import fs from 'node:fs';
import { describe, expect, it, vi } from 'vitest';
import { createZeroGPU } from './zerogpu-provider';

vi.mock('./version', () => ({
  VERSION: '0.0.0-test',
}));

const TEST_PROMPT: LanguageModelV4Prompt = [
  { role: 'user', content: [{ type: 'text', text: 'Hello' }] },
];

const CHAT_COMPLETIONS_URL = 'https://api.zerogpu.ai/v1/chat/completions';

const provider = createZeroGPU({
  apiKey: 'test-api-key',
  generateId: mockId(),
});
const model = provider.chat('gpt-4.1-mini');

const server = createTestServer({
  [CHAT_COMPLETIONS_URL]: {},
});

const weatherTool = {
  type: 'function' as const,
  name: 'weather',
  description: 'Get the weather for a city',
  inputSchema: {
    type: 'object' as const,
    properties: { city: { type: 'string' as const } },
    required: ['city'],
    additionalProperties: false,
    $schema: 'http://json-schema.org/draft-07/schema#',
  },
};

function prepareJsonFixtureResponse(
  filename: string,
  { headers }: { headers?: Record<string, string> } = {},
) {
  server.urls[CHAT_COMPLETIONS_URL].response = {
    type: 'json-value',
    headers,
    body: JSON.parse(
      fs.readFileSync(`src/__fixtures__/${filename}.json`, 'utf8'),
    ),
  };
}

function prepareChunksFixtureResponse(filename: string) {
  const chunks = fs
    .readFileSync(`src/__fixtures__/${filename}.chunks.txt`, 'utf8')
    .split('\n')
    .filter(line => line.length > 0)
    .map(line => `data: ${line}\n\n`);
  chunks.push('data: [DONE]\n\n');

  server.urls[CHAT_COMPLETIONS_URL].response = {
    type: 'stream-chunks',
    chunks,
  };
}

describe('doGenerate', () => {
  it('should extract text content', async () => {
    prepareJsonFixtureResponse('zerogpu-text');

    const result = await model.doGenerate({ prompt: TEST_PROMPT });

    expect(result).toMatchSnapshot();
  });

  it('should extract tool call content', async () => {
    prepareJsonFixtureResponse('zerogpu-tool-call');

    const result = await model.doGenerate({
      prompt: TEST_PROMPT,
      tools: [weatherTool],
    });

    expect(result.content).toMatchInlineSnapshot(`
      [
        {
          "input": "{"city":"Paris"}",
          "toolCallId": "call_7Zlq8pVJAsMAispduiY6jHcQ",
          "toolName": "weather",
          "type": "tool-call",
        },
      ]
    `);
    expect(result.finishReason).toStrictEqual({
      unified: 'tool-calls',
      raw: 'tool_calls',
    });
  });

  it('should extract usage with reasoning tokens', async () => {
    prepareJsonFixtureResponse('zerogpu-reasoning-usage');

    const { usage } = await provider
      .chat('glm-5.3-flash')
      .doGenerate({ prompt: TEST_PROMPT });

    expect(usage).toMatchInlineSnapshot(`
      {
        "inputTokens": {
          "cacheRead": undefined,
          "cacheWrite": undefined,
          "noCache": 20,
          "total": 20,
        },
        "outputTokens": {
          "reasoning": 143,
          "text": 47,
          "total": 190,
        },
        "raw": {
          "completion_tokens": 190,
          "completion_tokens_details": {
            "reasoning_tokens": 143,
          },
          "prompt_tokens": 20,
          "prompt_tokens_details": {
            "cached_tokens": 0,
          },
          "total_tokens": 210,
        },
      }
    `);
  });

  it('should return classification output as text', async () => {
    prepareJsonFixtureResponse('zerogpu-classification');

    const result = await provider
      .chat('zlm-v1-iab-classify-edge')
      .doGenerate({ prompt: TEST_PROMPT });

    expect(result.content).toHaveLength(1);
    expect(result.content[0].type).toBe('text');
    expect(
      JSON.parse((result.content[0] as { text: string }).text),
    ).toHaveProperty('content.iab_2_2');
  });

  it('should expose the raw response headers', async () => {
    prepareJsonFixtureResponse('zerogpu-text', {
      headers: { 'test-header': 'test-value' },
    });

    const { response } = await model.doGenerate({ prompt: TEST_PROMPT });

    expect(response?.headers).toStrictEqual({
      'content-length': expect.any(String),
      'content-type': 'application/json',
      'test-header': 'test-value',
    });
  });

  it('should send the model id and messages', async () => {
    prepareJsonFixtureResponse('zerogpu-text');

    await model.doGenerate({ prompt: TEST_PROMPT });

    expect(await server.calls[0].requestBodyJson).toMatchInlineSnapshot(`
      {
        "messages": [
          {
            "content": "Hello",
            "role": "user",
          },
        ],
        "model": "gpt-4.1-mini",
      }
    `);
  });

  it('should pass standard settings', async () => {
    prepareJsonFixtureResponse('zerogpu-text');

    await model.doGenerate({
      prompt: TEST_PROMPT,
      maxOutputTokens: 100,
      temperature: 0.5,
      topP: 0.9,
      topK: 40,
      frequencyPenalty: 0.1,
      presencePenalty: 0.2,
      stopSequences: ['END'],
      seed: 42,
    });

    expect(await server.calls[0].requestBodyJson).toMatchInlineSnapshot(`
      {
        "frequency_penalty": 0.1,
        "max_tokens": 100,
        "messages": [
          {
            "content": "Hello",
            "role": "user",
          },
        ],
        "model": "gpt-4.1-mini",
        "presence_penalty": 0.2,
        "seed": 42,
        "stop": [
          "END",
        ],
        "temperature": 0.5,
        "top_k": 40,
        "top_p": 0.9,
      }
    `);
  });

  it('should pass tools and tool choice', async () => {
    prepareJsonFixtureResponse('zerogpu-tool-call');

    await model.doGenerate({
      prompt: TEST_PROMPT,
      tools: [weatherTool],
      toolChoice: { type: 'tool', toolName: 'weather' },
    });

    expect(await server.calls[0].requestBodyJson).toMatchInlineSnapshot(`
      {
        "messages": [
          {
            "content": "Hello",
            "role": "user",
          },
        ],
        "model": "gpt-4.1-mini",
        "tool_choice": {
          "function": {
            "name": "weather",
          },
          "type": "function",
        },
        "tools": [
          {
            "function": {
              "description": "Get the weather for a city",
              "name": "weather",
              "parameters": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "additionalProperties": false,
                "properties": {
                  "city": {
                    "type": "string",
                  },
                },
                "required": [
                  "city",
                ],
                "type": "object",
              },
            },
            "type": "function",
          },
        ],
      }
    `);
  });

  it.each(['auto', 'none', 'required'] as const)(
    'should pass tool choice %s',
    async type => {
      prepareJsonFixtureResponse('zerogpu-tool-call');

      await model.doGenerate({
        prompt: TEST_PROMPT,
        tools: [weatherTool],
        toolChoice: { type },
      });

      expect(
        ((await server.calls[0].requestBodyJson) as { tool_choice: unknown })
          .tool_choice,
      ).toBe(type);
    },
  );

  it('should warn about provider-defined tools', async () => {
    prepareJsonFixtureResponse('zerogpu-text');

    const { warnings } = await model.doGenerate({
      prompt: TEST_PROMPT,
      tools: [
        {
          type: 'provider',
          id: 'some.provider-tool',
          name: 'providerTool',
          args: {},
        },
      ],
    });

    expect(warnings).toStrictEqual([
      {
        type: 'unsupported',
        feature: 'provider-defined tool some.provider-tool',
      },
    ]);
  });

  it('should pass a json schema response format', async () => {
    prepareJsonFixtureResponse('zerogpu-text');

    await model.doGenerate({
      prompt: TEST_PROMPT,
      responseFormat: {
        type: 'json',
        name: 'person',
        description: 'A person',
        schema: {
          type: 'object',
          properties: { name: { type: 'string' } },
          required: ['name'],
          additionalProperties: false,
        },
      },
    });

    expect(
      ((await server.calls[0].requestBodyJson) as { response_format: unknown })
        .response_format,
    ).toMatchInlineSnapshot(`
      {
        "json_schema": {
          "description": "A person",
          "name": "person",
          "schema": {
            "additionalProperties": false,
            "properties": {
              "name": {
                "type": "string",
              },
            },
            "required": [
              "name",
            ],
            "type": "object",
          },
        },
        "type": "json_schema",
      }
    `);
  });

  it('should pass a json object response format without schema', async () => {
    prepareJsonFixtureResponse('zerogpu-text');

    await model.doGenerate({
      prompt: TEST_PROMPT,
      responseFormat: { type: 'json' },
    });

    expect(
      ((await server.calls[0].requestBodyJson) as { response_format: unknown })
        .response_format,
    ).toStrictEqual({ type: 'json_object' });
  });

  it.each([
    ['high', 'high', 0],
    ['none', 'none', 0],
    ['minimal', 'minimal', 0],
    ['xhigh', 'high', 1],
  ] as const)(
    'should map reasoning %s to reasoning_effort %s',
    async (reasoning, effort, warningCount) => {
      prepareJsonFixtureResponse('zerogpu-text');

      const { warnings } = await model.doGenerate({
        prompt: TEST_PROMPT,
        reasoning,
      });

      expect(
        (
          (await server.calls[0].requestBodyJson) as {
            reasoning_effort: unknown;
          }
        ).reasoning_effort,
      ).toBe(effort);
      expect(warnings).toHaveLength(warningCount);
    },
  );

  it('should not send reasoning_effort for the provider default', async () => {
    prepareJsonFixtureResponse('zerogpu-text');

    await model.doGenerate({
      prompt: TEST_PROMPT,
      reasoning: 'provider-default',
    });

    expect(await server.calls[0].requestBodyJson).not.toHaveProperty(
      'reasoning_effort',
    );
  });

  it('should prefer the reasoningEffort provider option', async () => {
    prepareJsonFixtureResponse('zerogpu-text');

    await model.doGenerate({
      prompt: TEST_PROMPT,
      reasoning: 'high',
      providerOptions: { zerogpu: { reasoningEffort: 'low' } },
    });

    expect(
      (
        (await server.calls[0].requestBodyJson) as {
          reasoning_effort: unknown;
        }
      ).reasoning_effort,
    ).toBe('low');
  });

  it('should pass classification provider options', async () => {
    prepareJsonFixtureResponse('zerogpu-classification');

    await provider.chat('gliner2.5-base-v1').doGenerate({
      prompt: TEST_PROMPT,
      providerOptions: {
        zerogpu: {
          metadata: { usecase: 'ner', labels: ['person'], threshold: 0.3 },
          categories: ['signals'],
          maxKeywords: 3,
        },
      },
    });

    expect(await server.calls[0].requestBodyJson).toMatchInlineSnapshot(`
      {
        "categories": [
          "signals",
        ],
        "max_keywords": 3,
        "messages": [
          {
            "content": "Hello",
            "role": "user",
          },
        ],
        "metadata": {
          "labels": [
            "person",
          ],
          "threshold": 0.3,
          "usecase": "ner",
        },
        "model": "gliner2.5-base-v1",
      }
    `);
  });

  it('should reject invalid provider options', async () => {
    prepareJsonFixtureResponse('zerogpu-text');

    await expect(
      model.doGenerate({
        prompt: TEST_PROMPT,
        providerOptions: { zerogpu: { maxKeywords: 'many' } },
      }),
    ).rejects.toThrow();
  });

  it('should pass headers', async () => {
    prepareJsonFixtureResponse('zerogpu-text');

    await createZeroGPU({
      apiKey: 'test-api-key',
      headers: { 'Custom-Provider-Header': 'provider-header-value' },
    })
      .chat('gpt-4.1-mini')
      .doGenerate({
        prompt: TEST_PROMPT,
        headers: { 'Custom-Request-Header': 'request-header-value' },
      });

    expect(server.calls[0].requestHeaders).toStrictEqual({
      authorization: 'Bearer test-api-key',
      'content-type': 'application/json',
      'custom-provider-header': 'provider-header-value',
      'custom-request-header': 'request-header-value',
    });
    expect(server.calls[0].requestUserAgent).toContain(
      'ai-sdk-zerogpu/0.0.0-test',
    );
  });

  it('should throw an APICallError for JSON error responses', async () => {
    server.urls[CHAT_COMPLETIONS_URL].response = {
      type: 'error',
      status: 400,
      body: JSON.stringify({
        error: {
          message: 'Tool calling is not supported for model: test-model',
          type: 'invalid_request_error',
          param: 'tools',
          code: 'tools_not_supported',
        },
      }),
    };

    const error = await Promise.resolve(
      model.doGenerate({ prompt: TEST_PROMPT }),
    ).catch(error => error);

    expect(APICallError.isInstance(error)).toBe(true);
    expect(error.message).toBe(
      'Tool calling is not supported for model: test-model',
    );
    expect(error.statusCode).toBe(400);
    expect(error.isRetryable).toBe(false);
  });

  it('should throw an APICallError for plain-text error responses', async () => {
    server.urls[CHAT_COMPLETIONS_URL].response = {
      type: 'error',
      status: 403,
      body: 'Invalid credentials',
    };

    const error = await Promise.resolve(
      model.doGenerate({ prompt: TEST_PROMPT }),
    ).catch(error => error);

    expect(APICallError.isInstance(error)).toBe(true);
    expect(error.statusCode).toBe(403);
    expect(error.responseBody).toBe('Invalid credentials');
  });
});

describe('doStream', () => {
  it('should stream text', async () => {
    prepareChunksFixtureResponse('zerogpu-text');

    const { stream } = await provider
      .chat('gpt-5.4-nano')
      .doStream({ prompt: TEST_PROMPT });

    expect(await convertReadableStreamToArray(stream)).toMatchSnapshot();
  });

  it('should send stream options', async () => {
    prepareChunksFixtureResponse('zerogpu-text');

    await model.doStream({ prompt: TEST_PROMPT });

    expect(await server.calls[0].requestBodyJson).toMatchInlineSnapshot(`
      {
        "messages": [
          {
            "content": "Hello",
            "role": "user",
          },
        ],
        "model": "gpt-4.1-mini",
        "stream": true,
        "stream_options": {
          "include_usage": true,
        },
      }
    `);
  });

  it('should stream parallel tool call deltas', async () => {
    prepareChunksFixtureResponse('zerogpu-tool-call');

    const { stream } = await model.doStream({
      prompt: TEST_PROMPT,
      tools: [weatherTool],
    });

    expect(await convertReadableStreamToArray(stream)).toMatchSnapshot();
  });

  it('should stream tool calls that arrive in a single chunk', async () => {
    prepareChunksFixtureResponse('zerogpu-single-chunk-tool-call');

    const { stream } = await provider.chat('qwen3-30b-a3b-fp8').doStream({
      prompt: TEST_PROMPT,
      tools: [weatherTool],
    });

    const parts = await convertReadableStreamToArray(stream);

    expect(parts.filter(part => part.type === 'tool-call')).toStrictEqual([
      {
        type: 'tool-call',
        toolCallId: expect.any(String),
        toolName: 'weather',
        input: '{"city": "Paris"}',
      },
    ]);
    expect(parts.at(-1)).toMatchObject({
      type: 'finish',
      finishReason: { unified: 'tool-calls', raw: 'tool_calls' },
    });
  });

  it('should emit an error part for error chunks', async () => {
    server.urls[CHAT_COMPLETIONS_URL].response = {
      type: 'stream-chunks',
      chunks: [
        `data: ${JSON.stringify({ id: 'chatcmpl-1', created: 1, model: 'gpt-4.1-mini', choices: [{ index: 0, delta: { content: 'Partial' } }] })}\n\n`,
        `data: ${JSON.stringify({ error: { message: 'Upstream inference timed out', type: 'server_error' } })}\n\n`,
        'data: [DONE]\n\n',
      ],
    };

    const { stream } = await model.doStream({ prompt: TEST_PROMPT });
    const parts = await convertReadableStreamToArray(stream);

    expect(parts.filter(part => part.type === 'error')).toStrictEqual([
      {
        type: 'error',
        error: {
          message: 'Upstream inference timed out',
          type: 'server_error',
        },
      },
    ]);
    expect(parts.at(-1)).toMatchObject({
      type: 'finish',
      finishReason: { unified: 'error', raw: undefined },
    });
  });

  it.each([
    ['missing', 'interrupted'],
    ['unknown_reason', 'other'],
    ['invalid-json', 'error'],
  ] as const)(
    'handles a stream ending with %s as %s',
    async (ending, unified) => {
      server.urls[CHAT_COMPLETIONS_URL].response = {
        type: 'stream-chunks',
        chunks: [
          `data: ${JSON.stringify({ id: 'chatcmpl-1', created: 1, model: 'gpt-4.1-mini', choices: [{ index: 0, delta: { content: 'Partial answer' } }] })}\n\n`,
          ...(ending === 'missing'
            ? []
            : ending === 'invalid-json'
              ? ['data: {invalid-json}\n\n']
              : [
                  `data: ${JSON.stringify({ choices: [{ index: 0, delta: {}, finish_reason: ending }] })}\n\n`,
                ]),
        ],
      };

      const { stream } = await model.doStream({ prompt: TEST_PROMPT });
      const parts = await convertReadableStreamToArray(stream);

      expect(parts).toContainEqual({
        type: 'text-delta',
        id: '0',
        delta: 'Partial answer',
      });
      expect(parts.filter(part => part.type === 'error')).toHaveLength(
        ending === 'invalid-json' ? 1 : 0,
      );
      expect(parts.filter(part => part.type === 'finish')).toEqual([
        expect.objectContaining({
          finishReason: {
            unified,
            raw: ending === 'unknown_reason' ? ending : undefined,
          },
        }),
      ]);
    },
  );

  it('should include raw chunks when requested', async () => {
    prepareChunksFixtureResponse('zerogpu-text');

    const { stream } = await model.doStream({
      prompt: TEST_PROMPT,
      includeRawChunks: true,
    });
    const parts = await convertReadableStreamToArray(stream);

    expect(parts.filter(part => part.type === 'raw')).toHaveLength(12);
  });

  it('should throw an APICallError when the model does not support streaming', async () => {
    server.urls[CHAT_COMPLETIONS_URL].response = {
      type: 'error',
      status: 400,
      body: JSON.stringify({
        error: {
          message: 'stream is not supported for this model',
          type: 'invalid_request_error',
          param: 'stream',
          code: 'stream_not_supported',
        },
      }),
    };

    await expect(
      provider
        .chat('zlm-v1-iab-classify-edge')
        .doStream({ prompt: TEST_PROMPT }),
    ).rejects.toThrow('stream is not supported for this model');
  });

  it('should expose the request body and response headers', async () => {
    prepareChunksFixtureResponse('zerogpu-text');

    const { request, response } = await model.doStream({
      prompt: TEST_PROMPT,
    });

    expect(request?.body).toMatchObject({
      model: 'gpt-4.1-mini',
      stream: true,
    });
    expect(response?.headers).toMatchObject({
      'content-type': 'text/event-stream',
    });
  });
});
