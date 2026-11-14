# free-model-aggregator-01 — the best free AI model aggregator

One server (`main.ts`, Node 22, standard library only) that serves the OpenAI Chat
Completions API and the Anthropic Messages API — JSON and server-sent events — over ten
simulated free-tier model providers, on the OpenAI, Anthropic and Gemini wires.

## How it works

1. Read the brief at `$MAKEFUN_PROMPT_FILE` (provider directory, quality table, workload
   manifest) and probe every provider's catalog route in its own wire.
2. Build one catalog and print it with `node main.ts --list`, byte for byte as it is
   registered with the env.
3. Listen on a free port, register `{protocol_version: 1, port, catalog}` at
   `POST $MAKEFUN_ENV_URL/register`, serve the 20 requests, and exit 0 when
   `GET $MAKEFUN_ENV_URL/work` says done.

## Routing

Each request is served by the best free model that meets its requirements, ranked by the
quality table's `rating_milli`, with ties broken by the provider called least so far, then
provider id, then model id. A JSON-mode request from an OpenAI client never goes to an
Anthropic-wire provider (that wire cannot carry JSON mode). Every refusal (402/429/5xx) is
counted and the request is answered by the next candidate; a provider that fails three
calls in a row is treated as spent or out. When every candidate has refused once, the list
is tried a second time, since a scheduled refusal is a one-off.

Every request field the provider echoes — system prompt, user text, images, tool names,
stop sequences, `max_tokens`, JSON mode — is translated faithfully into the provider's
wire, and the provider's answer text, usage counts and stop reason are translated back
into the wire the client used. All choices are functions of the brief, the requests and
the providers' answers so far: no clock, no unseeded randomness, so a replay leaves the
same state export.

## Files

- `main.ts` — the server: brief, catalog, routing and failover, the two client routes.
- `lib/providers.ts` — the brief's documents, the catalog probe, the catalog builder.
- `lib/wire.ts` — the client wires and the three upstream wires, in both directions.

## Practice results (local, `program/env` + `program/scorer`)

| trial | score | quality | served | wire | failover | catalog | calls |
|------:|------:|--------:|-------:|-----:|---------:|--------:|------:|
| 0 | 99 | 99 | 100 | 100 | 100 | 100 | 27 |
| 1 | 99 | 99 | 100 | 100 | 100 | 100 | 32 |
| 2 | 99 | 99 | 100 | 100 | 100 | 100 | 30 |

Every trial was run twice; the state exports were byte-identical both times.
