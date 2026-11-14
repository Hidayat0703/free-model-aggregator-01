// The client wires and the three upstream wires.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an Anthropic one
// (POST /v1/messages), with a JSON body or with server-sent events. The entry answers each request with
// one provider call, on whichever of the three wires that provider speaks, and translates back:
//
//   client openai    -> provider openai:    the body passes through (model replaced, stream_options
//                                           include_usage added on streams so the usage chunk arrives)
//   client openai    -> provider anthropic: the request is rewritten to Messages (system prompt, image
//                                           blocks, tool definitions, tool results and tool choice)
//   client openai    -> provider gemini:    the request is rewritten to generateContent (system
//                                           instruction, contents, function declarations, generation
//                                           config with maxOutputTokens, stopSequences and the JSON
//                                           MIME type when the client asked for JSON mode)
//   client anthropic -> provider openai:    the request is rewritten to Chat Completions
//   client anthropic -> provider anthropic: the body passes through (model replaced)
//   client anthropic -> provider gemini:    the request is rewritten to generateContent
//
// One wire cannot carry something: an Anthropic-wire provider echoes no JSON mode, so a JSON-mode request
// from an OpenAI client is never routed to one (main.ts enforces that); every other field a request can
// carry — system prompt, user text, images, tool names, stop sequences, max_tokens — travels on all
// three wires, because the provider echoes what it parsed and the client compares that echo with what
// it sent: a field rewritten or dropped loses the request.
//
// Every provider answer is parsed into one shape — the text whole, the input and output token counts,
// and the stop reason — and then built back into the wire the client used, in JSON or in the streaming
// event order that wire expects. The text is the provider's answer text, byte for byte: the checks read
// the echo out of it. The usage counts are the provider's own. The stop reason is mapped by family
// (stop / end_turn / STOP are one family; length / max_tokens / MAX_TOKENS another; tool calls a third).
//
// Everything here is a pure function of its arguments: no clock and no randomness, so the same body and
// the same provider answer give the same bytes on a replay. The two model listings use fixed timestamps
// for the same reason: a `created` that moved between runs would be a nondeterministic document.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** One provider answer, parsed out of its wire: the text whole, the usage, the stop reason. */
export interface ProviderAnswer {
  text: string;
  inputTokens: number | null;
  outputTokens: number | null;
  stop: string | null;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';
/** How many characters one streamed text delta carries, the same split the providers use. */
const STREAM_CHARS = 13;

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** The text of a block array's text blocks joined with nothing: how the checkers read an answer. */
function answerText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('');
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

function numberOrNull(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** The stop family of one provider's stop reason: 0 stop, 1 length, 2 tool use, 3 refusal. */
function stopFamily(reason: string | null): number {
  if (reason === 'length' || reason === 'max_tokens' || reason === 'MAX_TOKENS') return 1;
  if (reason === 'tool_calls' || reason === 'function_call' || reason === 'tool_use') return 2;
  if (reason === 'content_filter' || reason === 'refusal' || reason === 'SAFETY') return 3;
  return 0;
}

/** A provider's stop reason mapped into the wire the client used. */
export function mapStop(stop: string | null, wire: 'openai' | 'anthropic'): string {
  const family = stopFamily(stop);
  if (wire === 'openai') return ['stop', 'length', 'tool_calls', 'content_filter'][family] ?? 'stop';
  return ['end_turn', 'max_tokens', 'tool_use', 'refusal'][family] ?? 'end_turn';
}

// ---------------------------------------------------------------------------
// Client request -> provider request
// ---------------------------------------------------------------------------

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** One OpenAI message as Anthropic content blocks: text, images, tool calls and tool results. */
function anthropicContentOf(message: Json): Json[] {
  const blocks: Json[] = [];
  const content = message.content;
  if (typeof content === 'string' && content !== '') blocks.push({ type: 'text', text: content });
  if (Array.isArray(content)) {
    for (const part of content) {
      if (!isRecord(part)) continue;
      if (part.type === 'text' && typeof part.text === 'string') blocks.push({ type: 'text', text: part.text });
      else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
        const url = part.image_url.url;
        const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
        if (data !== null) blocks.push({ type: 'image', source: { type: 'base64', media_type: data[1] !== '' ? data[1] : 'image/png', data: data[2] } });
        else blocks.push({ type: 'image', source: { type: 'url', url } });
      }
    }
  }
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    blocks.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : '',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  return blocks;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request that names `model`: the system
 * messages become the system prompt, user parts become user blocks (images included), assistant
 * tool_calls become tool_use blocks, tool messages become tool_result blocks, and the output cap,
 * temperature, top_p, stop sequences and streaming pass through. JSON mode cannot travel on this wire:
 * the caller never routes a JSON-mode request to an Anthropic-wire provider.
 */
export function openAIToAnthropic(body: Json, model: string): Json {
  const out: Json = { model };
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const systemParts: string[] = [];
  const converted: Json[] = [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'tool') {
      converted.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: typeof raw.content === 'string' ? raw.content : toolResultText(raw.content) }],
      });
      continue;
    }
    const blocks = anthropicContentOf(raw);
    if (blocks.length === 0) continue;
    converted.push({ role: raw.role === 'assistant' ? 'assistant' : 'user', content: blocks.length === 1 && blocks[0]?.type === 'text' ? (blocks[0] as Json).text : blocks });
  }
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  out.messages = converted;
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  if (maxTokens !== undefined) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (typeof body.stop === 'string' && body.stop !== '') out.stop_sequences = [body.stop];
  else if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop.filter((s): s is string => typeof s === 'string');
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (choice === 'none') {
    // Anthropic has no "none" tool_choice: leaving it out is the closest spelling.
  } else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

// ---------------------------------------------------------------------------
// Client request -> Gemini request
// ---------------------------------------------------------------------------

/** One OpenAI content part as a Gemini part: text or an inline image / file reference. */
function geminiPartOf(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
    if (data !== null) return { inlineData: { mimeType: data[1] !== '' ? data[1] : 'image/png', data: data[2] } };
    return { fileData: { mimeType: 'image/png', fileUri: url } };
  }
  return null;
}

/** OpenAI tools as Gemini function declarations. */
function geminiDeclarationsFromOpenAI(tools: unknown): Json[] {
  if (!Array.isArray(tools)) return [];
  return tools.flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
    return [
      {
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      },
    ];
  });
}

/** Anthropic tools as Gemini function declarations. */
function geminiDeclarationsFromAnthropic(tools: unknown): Json[] {
  if (!Array.isArray(tools)) return [];
  return tools.flatMap((raw) => {
    if (!isRecord(raw) || typeof raw.name !== 'string') return [];
    return [
      {
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      },
    ];
  });
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent body. System messages become the
 * system instruction; user parts become user contents (images as inline data or file references);
 * assistant messages become model contents (tool calls are not echoed by the wire, so only their text
 * travels); tool messages carry nothing the wire echoes and are left out. The output cap, temperature,
 * top_p, stop sequences and JSON mode travel in the generation config.
 */
export function openAIToGemini(body: Json): Json {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const systemParts: string[] = [];
  const contents: Json[] = [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'tool') continue;
    const role = raw.role === 'assistant' ? 'model' : 'user';
    const parts: Json[] = [];
    const content = raw.content;
    if (typeof content === 'string' && content !== '') parts.push({ text: content });
    if (Array.isArray(content)) {
      for (const part of content) {
        if (!isRecord(part)) continue;
        const gp = geminiPartOf(part);
        if (gp !== null) parts.push(gp);
      }
    }
    if (parts.length > 0) contents.push({ role, parts });
  }
  const out: Json = { contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: [{ text: systemParts.join('\n') }] };
  const declarations = geminiDeclarationsFromOpenAI(body.tools);
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  const config: Json = {};
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  if (maxTokens !== undefined) config.maxOutputTokens = maxTokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (typeof body.stop === 'string' && body.stop !== '') config.stopSequences = [body.stop];
  else if (Array.isArray(body.stop) && body.stop.length > 0) config.stopSequences = body.stop.filter((s): s is string => typeof s === 'string');
  if (isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema')) {
    config.responseMimeType = 'application/json';
  }
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/**
 * An Anthropic Messages request as a Gemini generateContent body. The system prompt becomes the system
 * instruction; user blocks become user contents (images as inline data or file references, tool
 * results as function responses beside the text so the user text still reads whole); assistant blocks
 * become model contents; the output cap, stop sequences and tools travel as they do on the OpenAI wire.
 * JSON mode has no Anthropic spelling, so nothing is set for it here.
 */
export function anthropicToGemini(body: Json): Json {
  const system = textOf(body.system);
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    const parts: Json[] = [];
    if (typeof content === 'string' && content !== '') parts.push({ text: content });
    if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
        else if (block.type === 'image' && isRecord(block.source)) {
          const src = block.source;
          if (src.type === 'base64' && typeof src.data === 'string') {
            parts.push({ inlineData: { mimeType: typeof src.media_type === 'string' ? src.media_type : 'image/png', data: src.data } });
          } else if (src.type === 'url' && typeof src.url === 'string') {
            parts.push({ fileData: { mimeType: 'image/png', fileUri: src.url } });
          }
        } else if (block.type === 'tool_use') {
          parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } });
        } else if (block.type === 'tool_result') {
          parts.push({ functionResponse: { name: typeof block.tool_use_id === 'string' ? block.tool_use_id : '', response: { content: toolResultText(block.content) } } });
        }
      }
    }
    if (parts.length > 0) contents.push({ role: assistant ? 'model' : 'user', parts });
  }
  const out: Json = { contents };
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const declarations = geminiDeclarationsFromAnthropic(body.tools);
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) config.stopSequences = body.stop_sequences.filter((s): s is string => typeof s === 'string');
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

// ---------------------------------------------------------------------------
// Provider answer -> the one parsed shape
// ---------------------------------------------------------------------------

/** One SSE frame: its event name when it has one, and its data lines joined. */
export interface SseFrame {
  event: string | null;
  data: string;
}

/** Split a server-sent event body into frames. */
export function parseSse(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function parseJsonObject(text: string): Json | null {
  try {
    const value: unknown = JSON.parse(text);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** A Chat Completions answer (JSON or SSE) as the one parsed shape; null when it is not one. */
export function parseOpenAIAnswer(text: string, stream: boolean): ProviderAnswer | null {
  if (!stream) {
    const body = parseJsonObject(text);
    if (body === null) return null;
    const choices = Array.isArray(body.choices) ? body.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) return null;
    const message = isRecord(choice.message) ? choice.message : {};
    const usage = isRecord(body.usage) ? body.usage : {};
    return {
      text: typeof message.content === 'string' ? message.content : answerText(message.content),
      inputTokens: numberOrNull(usage.prompt_tokens),
      outputTokens: numberOrNull(usage.completion_tokens),
      stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : null,
    };
  }
  let out = '';
  let inputTokens: number | null = null;
  let outputTokens: number | null = null;
  let stop: string | null = null;
  let sawAny = false;
  for (const frame of parseSse(text)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') {
      sawAny = true;
      continue;
    }
    const chunk = parseJsonObject(payload);
    if (chunk === null) return null;
    sawAny = true;
    if (isRecord(chunk.usage)) {
      inputTokens = numberOrNull(chunk.usage.prompt_tokens);
      outputTokens = numberOrNull(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') out += delta.content;
    if (typeof choice.finish_reason === 'string') stop = choice.finish_reason;
  }
  if (!sawAny) return null;
  return { text: out, inputTokens, outputTokens, stop };
}

/** A Messages answer (JSON or SSE) as the one parsed shape; null when it is not one. */
export function parseAnthropicAnswer(text: string, stream: boolean): ProviderAnswer | null {
  if (!stream) {
    const body = parseJsonObject(text);
    if (body === null || body.type !== 'message' || !Array.isArray(body.content)) return null;
    const usage = isRecord(body.usage) ? body.usage : {};
    return {
      text: answerText(body.content),
      inputTokens: numberOrNull(usage.input_tokens),
      outputTokens: numberOrNull(usage.output_tokens),
      stop: typeof body.stop_reason === 'string' ? body.stop_reason : null,
    };
  }
  let out = '';
  let inputTokens: number | null = null;
  let outputTokens: number | null = null;
  let stop: string | null = null;
  let sawAny = false;
  for (const frame of parseSse(text)) {
    if (frame.event === null) return null;
    const data = parseJsonObject(frame.data);
    if (data === null) return null;
    sawAny = true;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      inputTokens = numberOrNull(data.message.usage.input_tokens);
    } else if (frame.event === 'content_block_delta' && isRecord(data.delta) && data.delta.type === 'text_delta' && typeof data.delta.text === 'string') {
      out += data.delta.text;
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = data.delta.stop_reason;
      if (isRecord(data.usage)) outputTokens = numberOrNull(data.usage.output_tokens);
    }
  }
  if (!sawAny) return null;
  return { text: out, inputTokens, outputTokens, stop };
}

/** The text of one Gemini candidate's parts, joined. */
function geminiCandidateText(candidate: unknown): string {
  if (!isRecord(candidate) || !isRecord(candidate.content) || !Array.isArray(candidate.content.parts)) return '';
  return candidate.content.parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
}

/** A generateContent answer (JSON or SSE) as the one parsed shape; null when it is not one. */
export function parseGeminiAnswer(text: string, stream: boolean): ProviderAnswer | null {
  if (!stream) {
    const body = parseJsonObject(text);
    if (body === null || !Array.isArray(body.candidates) || body.candidates.length === 0) return null;
    const candidate = isRecord(body.candidates[0]) ? body.candidates[0] : {};
    const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
    return {
      text: geminiCandidateText(candidate),
      inputTokens: numberOrNull(usage.promptTokenCount),
      outputTokens: numberOrNull(usage.candidatesTokenCount),
      stop: typeof candidate.finishReason === 'string' ? candidate.finishReason : null,
    };
  }
  let out = '';
  let inputTokens: number | null = null;
  let outputTokens: number | null = null;
  let stop: string | null = null;
  let sawAny = false;
  for (const frame of parseSse(text)) {
    const data = parseJsonObject(frame.data);
    if (data === null) return null;
    sawAny = true;
    if (Array.isArray(data.candidates) && data.candidates.length > 0) {
      const candidate = isRecord(data.candidates[0]) ? data.candidates[0] : {};
      out += geminiCandidateText(candidate);
      if (typeof candidate.finishReason === 'string') stop = candidate.finishReason;
    }
    if (isRecord(data.usageMetadata)) {
      inputTokens = numberOrNull(data.usageMetadata.promptTokenCount);
      outputTokens = numberOrNull(data.usageMetadata.candidatesTokenCount);
    }
  }
  if (!sawAny) return null;
  return { text: out, inputTokens, outputTokens, stop };
}

// ---------------------------------------------------------------------------
// The one parsed shape -> the wire the client used
// ---------------------------------------------------------------------------

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Split text into the streamed deltas, the same 13-character pieces the providers use. */
function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += STREAM_CHARS) pieces.push(text.slice(at, at + STREAM_CHARS));
  return pieces.length === 0 ? [''] : pieces;
}

/** A provider answer as a Chat Completions object, naming the model that served it. */
export function openAIJsonResponse(model: string, answer: ProviderAnswer): Json {
  const input = answer.inputTokens ?? 0;
  const output = answer.outputTokens ?? 0;
  return {
    id: 'chatcmpl-agg',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: mapStop(answer.stop, 'openai'), logprobs: null }],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  };
}

/** A provider answer as a Chat Completions event stream, naming the model that served it. */
export function openAIStreamEvents(model: string, answer: ProviderAnswer): string[] {
  const input = answer.inputTokens ?? 0;
  const output = answer.outputTokens ?? 0;
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-agg',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const events: string[] = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of splitText(answer.text)) events.push(sseEvent(null, chunk({ content: piece }, null)));
  events.push(sseEvent(null, chunk({}, mapStop(answer.stop, 'openai'))));
  events.push(
    sseEvent(null, {
      id: 'chatcmpl-agg',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
    }),
  );
  events.push('data: [DONE]\n\n');
  return events;
}

/** A provider answer as a Messages object, naming the model that served it. */
export function anthropicJsonResponse(model: string, answer: ProviderAnswer): Json {
  return {
    id: 'msg_agg',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: mapStop(answer.stop, 'anthropic'),
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens ?? 0, output_tokens: answer.outputTokens ?? 0 },
  };
}

/** A provider answer as a Messages event stream, naming the model that served it. */
export function anthropicStreamEvents(model: string, answer: ProviderAnswer): string[] {
  const events: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_agg',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens ?? 0, output_tokens: 0 },
      },
    }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (const piece of splitText(answer.text)) {
    events.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  events.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: mapStop(answer.stop, 'anthropic'), stop_sequence: null },
      usage: { output_tokens: answer.outputTokens ?? 0 },
    }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return events;
}
