import { TooManyEmbeddingValuesForCallError } from '@ai-sdk/provider';
import { createTestServer } from '@ai-sdk/test-server/with-vitest';
import fs from 'node:fs';
import { describe, expect, it, vi } from 'vitest';
import { createZeroGPU } from './zerogpu-provider';

vi.mock('./version', () => ({
  VERSION: '0.0.0-test',
}));

const EMBEDDINGS_URL = 'https://api.zerogpu.ai/v1/embeddings';
const values = ['sunny day at the beach', 'rainy day in the city'];

const provider = createZeroGPU({ apiKey: 'test-api-key' });
const model = provider.embeddingModel('bge-small-en-v1.5');

const server = createTestServer({
  [EMBEDDINGS_URL]: {},
});

function prepareJsonFixtureResponse({
  headers,
}: { headers?: Record<string, string> } = {}) {
  server.urls[EMBEDDINGS_URL].response = {
    type: 'json-value',
    headers,
    body: JSON.parse(
      fs.readFileSync('src/__fixtures__/zerogpu-embedding.json', 'utf8'),
    ),
  };
}

describe('doEmbed', () => {
  it('should extract embeddings', async () => {
    prepareJsonFixtureResponse();

    const { embeddings } = await model.doEmbed({ values });

    expect(embeddings).toMatchInlineSnapshot(`
      [
        [
          -0.02220393717288971,
          0.009346427395939827,
          -0.060692980885505676,
          -0.03030559979379177,
          0.06607820093631744,
        ],
        [
          0.021597152575850487,
          -0.017335740849375725,
          0.03263936936855316,
          0.00186991470400244,
          -0.003334558568894863,
        ],
      ]
    `);
  });

  it('should extract usage', async () => {
    prepareJsonFixtureResponse();

    const { usage } = await model.doEmbed({ values });

    expect(usage).toStrictEqual({ tokens: 14 });
  });

  it('should expose the raw response', async () => {
    prepareJsonFixtureResponse({ headers: { 'test-header': 'test-value' } });

    const { response } = await model.doEmbed({ values });

    expect(response?.headers).toMatchObject({ 'test-header': 'test-value' });
    expect(response?.body).toMatchObject({ model: 'bge-small-en-v1.5' });
  });

  it('should pass the model, values and dimensions', async () => {
    prepareJsonFixtureResponse();

    await model.doEmbed({ values, dimensions: 384 });

    expect(await server.calls[0].requestBodyJson).toStrictEqual({
      model: 'bge-small-en-v1.5',
      input: values,
      dimensions: 384,
      encoding_format: 'float',
    });
  });

  it('should pass headers', async () => {
    prepareJsonFixtureResponse();

    await createZeroGPU({
      apiKey: 'test-api-key',
      headers: { 'Custom-Provider-Header': 'provider-header-value' },
    })
      .embeddingModel('bge-small-en-v1.5')
      .doEmbed({
        values,
        headers: { 'Custom-Request-Header': 'request-header-value' },
      });

    expect(server.calls[0].requestHeaders).toStrictEqual({
      authorization: 'Bearer test-api-key',
      'content-type': 'application/json',
      'custom-provider-header': 'provider-header-value',
      'custom-request-header': 'request-header-value',
    });
  });

  it('should throw when there are too many values', async () => {
    expect(model.maxEmbeddingsPerCall).toBe(256);

    await expect(
      model.doEmbed({ values: Array.from({ length: 257 }, () => 'value') }),
    ).rejects.toThrow(TooManyEmbeddingValuesForCallError);
  });
});
