import {
  UnsupportedFunctionalityError,
  type LanguageModelV4FilePart,
  type LanguageModelV4Prompt,
} from '@ai-sdk/provider';
import {
  convertToBase64,
  getTopLevelMediaType,
  resolveFullMediaType,
} from '@ai-sdk/provider-utils';
import type {
  ZeroGPUAssistantMessage,
  ZeroGPUPrompt,
  ZeroGPUUserMessageContent,
} from './zerogpu-chat-prompt';

function formatImageUrl({ part }: { part: LanguageModelV4FilePart }): string {
  if (part.data.type === 'url') {
    return part.data.url.toString();
  }

  if (part.data.type === 'data') {
    return `data:${resolveFullMediaType({ part })};base64,${convertToBase64(part.data.data)}`;
  }

  throw new UnsupportedFunctionalityError({
    functionality: `file part data type ${part.data.type}`,
  });
}

export function convertToZeroGPUChatMessages(
  prompt: LanguageModelV4Prompt,
): ZeroGPUPrompt {
  const messages: ZeroGPUPrompt = [];

  for (const { role, content } of prompt) {
    switch (role) {
      case 'system': {
        messages.push({ role: 'system', content });
        break;
      }

      case 'user': {
        const parts: Array<ZeroGPUUserMessageContent> = content.map(part => {
          switch (part.type) {
            case 'text': {
              return { type: 'text', text: part.text };
            }

            case 'file': {
              if (getTopLevelMediaType(part.mediaType) !== 'image') {
                throw new UnsupportedFunctionalityError({
                  functionality: `file part media type ${part.mediaType}`,
                });
              }

              return {
                type: 'image_url',
                image_url: { url: formatImageUrl({ part }) },
              };
            }
          }
        });

        // text-only messages are sent as a plain string, which every
        // ZeroGPU model accepts (classification models require it):
        messages.push({
          role: 'user',
          content: parts.every(part => part.type === 'text')
            ? parts.map(part => part.text).join('')
            : parts,
        });
        break;
      }

      case 'assistant': {
        let text = '';
        const toolCalls: NonNullable<ZeroGPUAssistantMessage['tool_calls']> =
          [];

        for (const part of content) {
          switch (part.type) {
            case 'text': {
              text += part.text;
              break;
            }
            case 'tool-call': {
              toolCalls.push({
                id: part.toolCallId,
                type: 'function',
                function: {
                  name: part.toolName,
                  arguments: JSON.stringify(part.input),
                },
              });
              break;
            }
            case 'reasoning': {
              // reasoning is not sent back to the API
              break;
            }
            default: {
              throw new UnsupportedFunctionalityError({
                functionality: `assistant message part type ${part.type}`,
              });
            }
          }
        }

        messages.push({
          role: 'assistant',
          content: text,
          tool_calls: toolCalls.length > 0 ? toolCalls : undefined,
        });
        break;
      }

      case 'tool': {
        for (const toolResponse of content) {
          if (toolResponse.type === 'tool-approval-response') {
            continue;
          }

          const output = toolResponse.output;

          let contentValue: string;
          switch (output.type) {
            case 'text':
            case 'error-text':
              contentValue = output.value;
              break;
            case 'execution-denied':
              contentValue = output.reason ?? 'Tool call execution denied.';
              break;
            case 'content':
            case 'json':
            case 'error-json':
              contentValue = JSON.stringify(output.value);
              break;
          }

          messages.push({
            role: 'tool',
            tool_call_id: toolResponse.toolCallId,
            content: contentValue,
          });
        }
        break;
      }

      default: {
        const _exhaustiveCheck: never = role;
        throw new Error(`Unsupported role: ${_exhaustiveCheck}`);
      }
    }
  }

  return messages;
}
