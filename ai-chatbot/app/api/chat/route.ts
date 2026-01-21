import { convertToModelMessages, streamText, UIMessage } from 'ai';
import { createOpenAICompatible } from '@ai-sdk/openai-compatible';

// 1. Define the SEA-LION provider
const seaLion = createOpenAICompatible({
  name: 'sea-lion',
  baseURL: 'https://api.sea-lion.ai/v1',
  apiKey: process.env.SEA_LION_API_KEY, 
});

export const maxDuration = 30;

export async function POST(req: Request) {
  const { messages }: { messages: UIMessage[] } = await req.json();

  // 1. Resolve the promise first
  const modelMessages = await convertToModelMessages(messages);

  const result = streamText({
    model: seaLion.chatModel('aisingapore/Qwen-SEA-LION-v4-32B-IT'),
    system: 'You are a helpful assistant.',
    // 2. Pass the resolved array here
    messages: modelMessages, 
  });

  return result.toUIMessageStreamResponse();
}