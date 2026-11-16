# Chat history

`useChatHistory(session_id)` returns `{ status, messages, error, retry }`. It uses
the existing typed `getChatHistory` service and loads only on mount, session ID
changes, or explicit retry. Settings rerenders do not refetch. The Chat page
shows a localized greeting only after a successful response with no messages.
Loading and errors have separate accessible states; failures offer retry.

Each hook owns a small in-memory store consumed through `useSyncExternalStore`.
Unmount/session changes abort requests. A generation check also ignores late
successes/errors from transports that ignore cancellation. StrictMode may start
one aborted GET before its replacement; snapshots are replaced, never appended.
No messages are persisted in browser storage or fetched on informational pages.

History responses are checked for matching session ID and string role/content.
Server ordering is preserved. Local keys combine session and snapshot position,
since the API supplies no message IDs/timestamps. Repeated text remains distinct;
future submissions should keep pending messages separate from persisted snapshots
until reconciliation can be defined. Submission is not implemented here.

Content is React text with preserved newlines and wrapping; no raw HTML or
Markdown HTML renderer is used. Error details from the backend are not shown.
Set `VITE_API_BASE_URL` using the frontend environment example to enable live
history. The frontend still builds independently of the backend.

Run mocked transport/state tests and server-rendered UI safety/state checks:

```sh
node --test --test-isolation=none tests/history.test.ts
```

The test loader uses the existing TypeScript dependency to compile TSX for
Node's built-in test runner; no test dependencies are added. These checks do not
replace interactive browser or live backend/CORS verification.
