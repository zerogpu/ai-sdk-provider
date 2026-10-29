export type ZeroGPUPrompt = Array<ZeroGPUMessage>;

export type ZeroGPUMessage =
  | ZeroGPUSystemMessage
  | ZeroGPUUserMessage
  | ZeroGPUAssistantMessage
  | ZeroGPUToolMessage;

export interface ZeroGPUSystemMessage {
  role: 'system';
  content: string;
}

export interface ZeroGPUUserMessage {
  role: 'user';
  content: string | Array<ZeroGPUUserMessageContent>;
}

export type ZeroGPUUserMessageContent =
  | { type: 'text'; text: string }
  | { type: 'image_url'; image_url: { url: string } };

export interface ZeroGPUAssistantMessage {
  role: 'assistant';
  content: string | null;
  tool_calls?: Array<{
    id: string;
    type: 'function';
    function: { name: string; arguments: string };
  }>;
}

export interface ZeroGPUToolMessage {
  role: 'tool';
  content: string;
  tool_call_id: string;
}

export type ZeroGPUToolChoice =
  | { type: 'function'; function: { name: string } }
  | 'auto'
  | 'none'
  | 'required';
