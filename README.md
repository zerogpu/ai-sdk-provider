# ZeroGPU Provider for the AI SDK

The **[ZeroGPU](https://zerogpu.ai) provider** for the [AI SDK](https://ai-sdk.dev/docs) gives you access to ZeroGPU's compute-efficient inference: purpose-built small and nano language models on an edge network, alongside open-weight and frontier LLMs, embeddings, text-to-speech and speech-to-text.

It implements the AI SDK [Language Model Specification V4](https://ai-sdk.dev/providers/community-providers/custom-providers) and supports:

- Text generation and streaming (`generateText`, `streamText`)
- Tool calling and structured output (`Output.object`)
- Image input on vision models
- ZeroGPU's classification and extraction models (IAB, signal extraction, GLiNER, DeBERTa)
- Embeddings (`embed`, `embedMany`)
- Speech generation (`generateSpeech`)
- Transcription (`transcribe`)

## Setup

The ZeroGPU provider is available in the `@zerogpu/ai-sdk-provider` module. You can install it with:

```bash
npm i @zerogpu/ai-sdk-provider ai
```

Get an API key from the [ZeroGPU dashboard](https://platform.zerogpu.ai/dashboard) and set it as an environment variable:

```bash
ZEROGPU_API_KEY=your-api-key
```

## Provider Instance

You can import the default provider instance `zerogpu` from `@zerogpu/ai-sdk-provider`:

```ts
import { zerogpu } from '@zerogpu/ai-sdk-provider';
```

If you need a customized setup, use `createZeroGPU` and create a provider instance with your settings:

```ts
import { createZeroGPU } from '@zerogpu/ai-sdk-provider';

const zerogpu = createZeroGPU({
  apiKey: process.env.ZEROGPU_API_KEY ?? '',
});
```

You can use the following optional settings to customize the ZeroGPU provider instance:

- **baseURL** _string_

  Use a different URL prefix for API calls, e.g. to use proxy servers.
  The default prefix is `https://api.zerogpu.ai/v1`.

- **apiKey** _string_

  API key that is being sent using the `Authorization` header.
  It defaults to the `ZEROGPU_API_KEY` environment variable.

- **headers** _Record&lt;string,string&gt;_

  Custom headers to include in the requests.

- **fetch** _(input: RequestInfo, init?: RequestInit) => Promise&lt;Response&gt;_

  Custom [fetch](https://developer.mozilla.org/en-US/docs/Web/API/fetch) implementation.

## Language Models

You can create language models using a provider instance.
The first argument is the model ID:

```ts
import { zerogpu } from '@zerogpu/ai-sdk-provider';
import { generateText } from 'ai';

const { text } = await generateText({
  model: zerogpu('gpt-oss-120b'),
  prompt: 'Write a vegetarian lasagna recipe for 4 people.',
});
```

ZeroGPU language models can also be used in the `streamText` function:

```ts
import { zerogpu } from '@zerogpu/ai-sdk-provider';
import { streamText } from 'ai';

const result = streamText({
  model: zerogpu('deepseek-v4.1-flash'),
  prompt: 'Explain edge inference in one paragraph.',
});

for await (const textPart of result.textStream) {
  process.stdout.write(textPart);
}
```

### Tool Calling

```ts
import { zerogpu } from '@zerogpu/ai-sdk-provider';
import { generateText, stepCountIs, tool } from 'ai';
import { z } from 'zod';

const { text } = await generateText({
  model: zerogpu('gpt-4.1-mini'),
  tools: {
    weather: tool({
      description: 'Get the weather for a city',
      inputSchema: z.object({ city: z.string() }),
      execute: async ({ city }) => ({ city, temperatureC: 21 }),
    }),
  },
  stopWhen: stepCountIs(3),
  prompt: 'What is the weather in Paris?',
});
```

### Structured Output

```ts
import { zerogpu } from '@zerogpu/ai-sdk-provider';
import { generateText, Output } from 'ai';
import { z } from 'zod';

const { output } = await generateText({
  model: zerogpu('gpt-5.4-nano'),
  output: Output.object({
    schema: z.object({
      name: z.string(),
      ingredients: z.array(z.string()),
    }),
  }),
  prompt: 'Generate a lasagna recipe.',
});
```

### Image Input

```ts
import { zerogpu } from '@zerogpu/ai-sdk-provider';
import { generateText } from 'ai';
import { readFile } from 'node:fs/promises';

const { text } = await generateText({
  model: zerogpu('qwen3.6-35b-a3b'),
  messages: [
    {
      role: 'user',
      content: [
        { type: 'text', text: 'Describe this image.' },
        {
          type: 'file',
          mediaType: 'image/png',
          data: await readFile('./image.png'),
        },
      ],
    },
  ],
});
```

Images can be passed as binary data or as `https` URLs.

### Reasoning

Use the AI SDK `reasoning` option, or `providerOptions.zerogpu.reasoningEffort`, to adjust the reasoning effort of models that support it:

```ts
const { text } = await generateText({
  model: zerogpu('gpt-5.6-luna'),
  reasoning: 'high',
  prompt: 'How many prime numbers are there below 100?',
});
```

ZeroGPU reports the reasoning token count in `usage`, but does not return the reasoning text.

### Classification and Extraction Models

ZeroGPU's classification and extraction models are called with `generateText` and return their result as a JSON string. They don't support streaming.

```ts
import { zerogpu } from '@zerogpu/ai-sdk-provider';
import { generateText } from 'ai';

// IAB Content Taxonomy classification
const { text } = await generateText({
  model: zerogpu('zlm-v1-iab-classify-edge'),
  prompt: 'Apple released a new MacBook laptop today.',
});

const classification = JSON.parse(text);
```

Some models take extra settings, passed as provider options:

```ts
// Named entity recognition with GLiNER
const { text } = await generateText({
  model: zerogpu('gliner2.5-base-v1'),
  prompt: 'The app is built with Python 3.11 and uses PostgreSQL 15.',
  providerOptions: {
    zerogpu: {
      metadata: {
        usecase: 'ner',
        labels: ['programming language', 'database'],
        threshold: 0.3,
      },
    },
  },
});

// Signal extraction
const { text: signals } = await generateText({
  model: zerogpu('zlm-v1-signal-extract'),
  prompt: 'How can I pay using my credit card?',
  providerOptions: {
    zerogpu: { categories: ['signals'], maxKeywords: 6 },
  },
});

// Zero-shot classification with DeBERTa: the labels are the system message
const { text: scores } = await generateText({
  model: zerogpu('deberta-v3-small'),
  system: '[Sports,Technology,Finance]',
  prompt: 'Apple unveiled its next-generation M5 chip.',
});
```

GLiNER models accept `usecase: 'ner'` with `labels`, `usecase: 'json'` with a `schema` of `field::type::description` entries, and `usecase: 'classification'` with a `schema` that maps each axis to its labels. See the [ZeroGPU model catalog](https://docs.zerogpu.ai/docs/model-catalog) for each model's input and output format.

### Provider Options

The following provider options are available for language models under `providerOptions.zerogpu`:

- **reasoningEffort** _'none' | 'minimal' | 'low' | 'medium' | 'high'_

  The reasoning effort for models that support adjustable reasoning. Takes precedence over the `reasoning` call option.

- **metadata** _Record&lt;string, unknown&gt;_

  Task configuration for the GLiNER models (`usecase`, `labels`, `schema`, `threshold`).

- **categories** _string[]_

  Signal categories to extract with `zlm-v1-signal-extract`.

- **maxKeywords** _number_

  The maximum number of keywords returned by `zlm-v1-signal-extract`.

### Model Capabilities

| Model                        | Image Input | Object Generation | Tool Usage | Tool Streaming |
| ---------------------------- | ----------- | ----------------- | ---------- | -------------- |
| `deepseek-v4.1-flash`        |             | ✅                | ✅         | ✅             |
| `glm-5.3-flash`              | ✅          | ✅                | ✅         | ✅             |
| `gpt-4.1-mini`               | ✅          | ✅                | ✅         | ✅             |
| `gpt-5.4-nano`               | ✅          | ✅                | ✅         | ✅             |
| `gpt-5.6-luna`               | ✅          | ✅                | ✅         | ✅             |
| `gpt-6-luna`                 | ✅          | ✅                | ✅         | ✅             |
| `gpt-oss-120b`               |             | ✅                | ✅         | ✅             |
| `qwen3-30b-a3b-fp8`          |             | ✅                | ✅         | ✅             |
| `qwen3.6-35b-a3b`            | ✅          | ✅                | ✅         | ✅             |
| `llama-3.1-8b-instruct-fast` |             | ✅                | ✅         | ✅             |
| `llama-3.2-3b-instruct-fast` |             |                   |            |                |
| `LFM2.5-1.2B-Instruct`       |             |                   |            |                |
| `LFM2.5-1.2B-Thinking`       |             |                   |            |                |
| `zlm-v1-tool-decisions`      |             | ✅                | ✅         | ✅             |

All of the models above support streaming. These classification and extraction models are also available, without streaming:

| Model                               | Task                                               |
| ----------------------------------- | -------------------------------------------------- |
| `zlm-v1-iab-classify-edge`          | IAB content and audience classification            |
| `zlm-v2-iab-classify-edge-enriched` | IAB classification with topics, keywords, intent   |
| `zlm-v1-iab-domain-classifier`      | IAB classification of a domain name                |
| `zlm-v1-signal-extract`             | Topic, keyword and intent extraction               |
| `gliner2.5-base-v1`                 | Entity, structured data and label extraction       |
| `gliner2.5-multi-v1`                | Multilingual entity and structured data extraction |
| `gliner2.5-decide-v1`               | Zero-shot decisions and classification             |
| `deberta-v3-small`                  | Zero-shot classification with your own labels      |

You can also pass any available ZeroGPU model ID as a string.

## Embedding Models

You can create models that call the ZeroGPU embeddings API using the `.embeddingModel()` factory method.

```ts
import { zerogpu } from '@zerogpu/ai-sdk-provider';
import { embed, embedMany } from 'ai';

const { embedding } = await embed({
  model: zerogpu.embeddingModel('bge-small-en-v1.5'),
  value: 'sunny day at the beach',
});

const { embeddings } = await embedMany({
  model: zerogpu.embeddingModel('all-minilm-l6-v2'),
  values: ['sunny day at the beach', 'rainy afternoon in the city'],
});
```

| Model               | Dimensions | Max Input Tokens |
| ------------------- | ---------- | ---------------- |
| `bge-small-en-v1.5` | 384        | 512              |
| `all-minilm-l6-v2`  | 384        | 256              |

## Speech Models

You can create models that call the ZeroGPU speech API using the `.speech()` factory method.

```ts
import { zerogpu } from '@zerogpu/ai-sdk-provider';
import { generateSpeech } from 'ai';

const { audio } = await generateSpeech({
  model: zerogpu.speech('chatterbox-nano'),
  text: 'Hello from ZeroGPU! [laugh]',
  voice: 'alloy',
  outputFormat: 'mp3',
});
```

- `voice` accepts OpenAI voice names such as `alloy`, `echo`, `fable`, `nova`, `onyx` and `shimmer`, and defaults to `alloy`.
- `outputFormat` can be `mp3` (default), `wav`, `opus`, `aac`, `flac` or `pcm`.
- Input is English text of up to 2,000 characters. Inline tags such as `[laugh]`, `[chuckle]` and `[cough]` add paralinguistic cues.

The following provider options are available under `providerOptions.zerogpu`: `seed`, `temperature`, `topP`, `topK` and `repetitionPenalty`.

## Transcription Models

You can create models that call the ZeroGPU transcription API using the `.transcription()` factory method.

```ts
import { zerogpu } from '@zerogpu/ai-sdk-provider';
import { transcribe } from 'ai';
import { readFile } from 'node:fs/promises';

const result = await transcribe({
  model: zerogpu.transcription('whisper-tiny'),
  audio: await readFile('./audio.mp3'),
  providerOptions: { zerogpu: { language: 'en' } },
});

console.log(result.text);
```

The result includes the transcript, timed `segments`, the detected `language` and the audio duration. Audio files can be up to 25 MB and 10 minutes long.

The following provider options are available under `providerOptions.zerogpu`:

- **language** _string_

  The language of the input audio in ISO-639-1 format (e.g. `en`). Supplying it improves accuracy and latency.

- **temperature** _number_

  The sampling temperature, between 0 and 1.

## Documentation

- [ZeroGPU documentation](https://docs.zerogpu.ai)
- [ZeroGPU model catalog](https://docs.zerogpu.ai/docs/model-catalog)
- [AI SDK documentation](https://ai-sdk.dev/docs)

## License

[MIT](./LICENSE)
