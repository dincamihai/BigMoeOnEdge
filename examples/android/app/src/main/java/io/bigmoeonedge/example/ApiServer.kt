package io.bigmoeonedge.example

import fi.iki.elonen.NanoHTTPD
import org.json.JSONArray
import org.json.JSONObject
import java.io.PipedInputStream
import java.io.PipedOutputStream
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Semaphore
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference

/**
 * Embedded HTTP API over the running session, so a machine elsewhere on the network (in practice:
 * another node on a Tailscale mesh) can prompt the model loaded on this phone. Hosted by
 * [RunService], so it exists exactly as long as a session process does — a request that arrives
 * with no model loaded is answered 503 rather than hanging.
 *
 *   POST /generate  {"prompt":"...", "n_predict":128, "think":false, "clear_kv":true}
 *                   → 200 {"text":..., "reasoning":..., "tokens":n, "tok_s":x, "cancelled":false}
 *   GET  /status    → 200 {"state":"READY", "session": true}
 *
 * `stream: true` on /v1/chat/completions is real: a chunk leaves as each token arrives.
 *
 * An OpenAI-shaped surface rides on the same path, so an off-the-shelf chat client — including one
 * on this very phone talking to 127.0.0.1, which works with no network at all — can hold the
 * conversation and its history while the app only owns the model:
 *
 *   GET  /v1/models            → the one loaded model
 *   POST /v1/chat/completions  {"messages":[...], "max_tokens":n, "stream":false}
 *
 * One model, one KV context, one engine queue: generations are strictly serialized here with a
 * single permit, mirroring the constraint the CLI's own session loop already has. On `/generate`
 * `clear_kv` defaults to true — a bare prompt is an independent completion. On the chat route it
 * defaults to FALSE, so a conversation is prefilled once and then extended; see [chat].
 *
 * No authentication: the bind is 0.0.0.0 so the server trusts the network it is on, which is the
 * mesh's job to keep private. Stated in the Settings toggle rather than hidden.
 */
class ApiServer(port: Int, private val service: RunService) : NanoHTTPD("0.0.0.0", port) {

    // One generation in flight at a time; fair, so queued callers run in arrival order.
    private val turn = Semaphore(1, true)

    /**
     * Nothing thrown by a handler may escape: NanoHTTPD runs each request on its own thread, and an
     * uncaught exception there reaches Android's default handler, which kills the whole process —
     * the app disappears, the model unloads, and the client sees a dropped connection with no clue
     * why. Answering 500 with the message turns a silent crash into something the caller can read.
     */
    override fun serve(session: IHTTPSession): Response =
        runCatching { route(session) }.getOrElse { e ->
            json(Response.Status.INTERNAL_ERROR, err("${e.javaClass.simpleName}: ${e.message}"))
        }

    private fun route(session: IHTTPSession): Response = when {
        session.method == Method.GET && session.uri == "/status" -> status()
        session.method == Method.POST && session.uri == "/generate" -> generate(session)
        session.method == Method.GET && session.uri == "/v1/models" -> models()
        session.method == Method.POST && session.uri == "/v1/chat/completions" -> chat(session)
        else -> json(Response.Status.NOT_FOUND,
            err("not found: use /generate, /status, /v1/models or /v1/chat/completions"))
    }

    private fun status(): Response {
        val ui = RunBus.state.value
        val o = JSONObject()
            .put("state", ui.state.name)
            .put("session", ui.sessionSig != null)
        return json(Response.Status.OK, o.toString())
    }

    private fun generate(session: IHTTPSession): Response {
        val req = readJson(session) ?: return json(Response.Status.BAD_REQUEST, err("body must be JSON"))
        val prompt = req.optString("prompt")
        if (prompt.isEmpty()) return json(Response.Status.BAD_REQUEST, err("missing \"prompt\""))

        return run(
            prompt = prompt,
            nPredict = req.optInt("n_predict", AppSettings.DEFAULT_N_PREDICT),
            think = req.optBoolean("think", false),
            clearKv = req.optBoolean("clear_kv", true),
        ) { r ->
            json(Response.Status.OK, JSONObject()
                .put("text", r.text)
                .put("reasoning", r.reasoning)
                .put("tokens", r.tokens)
                .put("tok_s", r.tokS)
                .put("cancelled", r.cancelled)
                .toString())
        }
    }

    /** The one model this phone has open. Clients call this to populate their model picker. */
    private fun models(): Response {
        val entry = JSONObject()
            .put("id", MODEL_ID)
            .put("object", "model")
            .put("created", now())
            .put("owned_by", "bigmoeonedge")
        return json(Response.Status.OK, JSONObject()
            .put("object", "list")
            .put("data", JSONArray().put(entry))
            .toString())
    }

    private fun chat(session: IHTTPSession): Response {
        val req = readJson(session) ?: return json(Response.Status.BAD_REQUEST, err("body must be JSON"))
        val messages = req.optJSONArray("messages")
        if (messages == null || messages.length() == 0) {
            return json(Response.Status.BAD_REQUEST, err("missing \"messages\""))
        }
        // The conversation crosses as it arrived. Flattening it into "User:/Assistant:" text cost
        // the model the turn structure its template was trained on, and no `tool` message could
        // have survived the trip.
        //
        // `clear_kv` defaults to FALSE here even though an OpenAI client re-sends its whole history
        // every turn: `messages` REPLACES the engine-held conversation rather than appending to it,
        // so a kept KV cannot duplicate anything. The engine diffs the re-rendered tokens against
        // the KV and prefills only the divergent tail — on a phone that is the difference between
        // paying for the new turn and paying for the entire conversation, on every question. A
        // second caller arriving with a different history is still correct; its common prefix is
        // just shorter, so more of the prompt gets prefilled. Send `"clear_kv": true` to force a
        // fresh conversation.
        val toolsJson = req.optJSONArray("tools")?.toString() ?: ""
        val conversation = withoutMedia(messages)
        val nPredict = req.optInt("max_tokens", AppSettings.DEFAULT_N_PREDICT)
        val clearKv = req.optBoolean("clear_kv", false)
        if (req.optBoolean("stream", false)) return stream(conversation, toolsJson, nPredict, clearKv)

        return run(
            messagesJson = conversation,
            toolsJson = toolsJson,
            nPredict = nPredict,
            think = false,
            clearKv = clearKv,
        ) { r -> completion(r) }
    }

    /**
     * One generation, answered as it is produced.
     *
     * This exists for the silence, not for the looks: the model writes about a token a second, so a
     * non-streamed answer means minutes with nothing on the socket, and every HTTP client gives up
     * long before that — the request is then lost with no error anywhere. A chunk per token keeps
     * bytes moving, so no idle timeout ever fires.
     *
     * The response returns immediately with a pipe the generation writes into; NanoHTTPD drains it
     * on its own thread. The semaphore is released by the completion callback, which the service
     * guarantees to run — on success, on engine error, and on session teardown — so no path leaks
     * the turn.
     *
     * ponytail: a client that stops reading eventually fills the pipe and blocks the engine's reader
     * thread. 256 KB is far more than any answer this model produces in the time a socket stays
     * open, so the ceiling is theoretical; a bounded queue with a drop policy is the upgrade.
     */
    private fun stream(messagesJson: String, toolsJson: String, nPredict: Int,
                       clearKv: Boolean): Response {
        if (!runCatching { turn.tryAcquire(GENERATE_TIMEOUT_MIN, TimeUnit.MINUTES) }.getOrDefault(false)) {
            return json(Response.Status.SERVICE_UNAVAILABLE, err("busy: another generation is in flight"))
        }
        val pipe = PipedInputStream(256 * 1024)
        val sink = PipedOutputStream(pipe).writer()
        // Writes race the client hanging up; a broken pipe is an ordinary end, not a failure.
        fun emit(s: String) = runCatching { sink.write(s); sink.flush() }

        val id = service.generateForApi("", nPredict, think = false, clearKv = clearKv,
            messagesJson = messagesJson, toolsJson = toolsJson,
            onDelta = { d -> emit("data: ${chunk(delta(d, null))}\n\n") },
            cb = { r ->
                r.fold(
                    onSuccess = { emit("data: ${chunk(delta(null, finish(it)))}\n\n") },
                    onFailure = { e -> emit("data: ${err(e.message ?: "engine error")}\n\n") },
                )
                emit("data: [DONE]\n\n")
                runCatching { sink.close() }
                turn.release()
            })
        if (id < 0) {
            runCatching { sink.close() }
            turn.release()
            return json(Response.Status.SERVICE_UNAVAILABLE,
                err("no model loaded: open a session in the app first"))
        }
        return newChunkedResponse(Response.Status.OK, "text/event-stream", pipe).apply {
            addHeader("Cache-Control", "no-cache")
        }
    }

    /** One choice of a streaming chunk: text arriving, or the terminal reason with an empty delta. */
    private fun delta(text: String?, finish: String?): JSONObject {
        val d = JSONObject()
        if (text != null) d.put("role", "assistant").put("content", text)
        return JSONObject().put("index", 0).put("delta", d)
            .put("finish_reason", finish ?: JSONObject.NULL)
    }

    /**
     * The conversation with non-text content parts removed.
     *
     * Several clients always send OpenAI's content-part array, and some put an image in it. The
     * engine hands messages to llama.cpp, which REJECTS a part type it cannot render — so passing
     * one through would turn a picture nobody could have described into a failed request. This
     * phone has no vision path; dropping the part and answering the text is what it did before
     * structured messages, and the tolerance is worth keeping.
     */
    private fun withoutMedia(messages: JSONArray): String {
        val out = JSONArray()
        for (i in 0 until messages.length()) {
            val m = messages.optJSONObject(i) ?: continue
            val parts = m.opt("content") as? JSONArray
            if (parts == null) {
                out.put(m)
                continue
            }
            val kept = JSONArray()
            for (j in 0 until parts.length()) {
                val part = parts.optJSONObject(j) ?: continue
                if (part.optString("type") == "text") kept.put(part)
            }
            // A message left with no renderable content is dropped whole: an empty content array
            // is not a message any template knows how to render.
            if (kept.length() > 0) out.put(JSONObject(m.toString()).put("content", kept))
        }
        return out.toString()
    }

    private fun completion(r: RunService.ApiResult): Response {
        val calls = JSONArray(r.toolCalls)
        // A turn that ends in tool calls carries no answer text, and an OpenAI client reads
        // finish_reason to know that before it reads content — reporting "stop" there would make a
        // waiting agent believe the model had simply said nothing.
        val message = JSONObject().put("role", "assistant")
            .put("content", if (r.text.isEmpty() && calls.length() > 0) JSONObject.NULL else r.text)
        if (calls.length() > 0) message.put("tool_calls", calls)
        val choice = JSONObject().put("index", 0).put("message", message)
            .put("finish_reason", if (calls.length() > 0) "tool_calls" else finish(r))
        val usage = JSONObject()
            .put("prompt_tokens", 0) // the engine reports generated tokens only
            .put("completion_tokens", r.tokens)
            .put("total_tokens", r.tokens)
        return json(Response.Status.OK, JSONObject()
            .put("id", "chatcmpl-${now()}")
            .put("object", "chat.completion")
            .put("created", now())
            .put("model", MODEL_ID)
            .put("choices", JSONArray().put(choice))
            .put("usage", usage)
            .toString())
    }

    private fun chunk(choice: JSONObject): String = JSONObject()
        .put("id", "chatcmpl-${now()}")
        .put("object", "chat.completion.chunk")
        .put("created", now())
        .put("model", MODEL_ID)
        .put("choices", JSONArray().put(choice))
        .toString()

    private fun finish(r: RunService.ApiResult): String = if (r.cancelled) "length" else "stop"

    private fun now(): Long = System.currentTimeMillis() / 1000

    private fun readJson(session: IHTTPSession): JSONObject? = runCatching {
        val files = HashMap<String, String>()
        session.parseBody(files) // NanoHTTPD: POST bodies only materialize through parseBody
        JSONObject(files["postData"] ?: "")
    }.getOrNull()

    /**
     * One generation, serialized against every other API caller, rendered by [onSuccess]. Every
     * failure path answers with a status a client can act on rather than leaving it hanging.
     */
    /**
     * One generation, whichever way the caller expressed it.
     *
     * `/generate` sends a bare prompt and lets the engine keep the conversation; `/v1/chat/...`
     * sends the whole conversation and owns it. Exactly one of [prompt] and [messagesJson] carries
     * the turn — the engine ignores `prompt` when messages are present.
     */
    private fun run(prompt: String = "", messagesJson: String = "", toolsJson: String = "",
                    nPredict: Int, think: Boolean, clearKv: Boolean,
                    onSuccess: (RunService.ApiResult) -> Response): Response {
        // Wait for the session to be free of other API callers. The wait is bounded by the same
        // ceiling as the generation itself, so a caller behind a long job times out rather than
        // queuing forever.
        if (!runCatching { turn.tryAcquire(GENERATE_TIMEOUT_MIN, TimeUnit.MINUTES) }.getOrDefault(false)) {
            return json(Response.Status.SERVICE_UNAVAILABLE, err("busy: another generation is in flight"))
        }
        try {
            val done = CountDownLatch(1)
            val result = AtomicReference<Result<RunService.ApiResult>>()
            val id = service.generateForApi(prompt, nPredict, think, clearKv, messagesJson, toolsJson) { r ->
                result.set(r); done.countDown()
            }
            if (id < 0) {
                return json(Response.Status.SERVICE_UNAVAILABLE,
                    err("no model loaded: open a session in the app first"))
            }
            if (!done.await(GENERATE_TIMEOUT_MIN, TimeUnit.MINUTES)) {
                service.abandonApi(id) // stop a late BMOE_DONE from resolving into nowhere
                // 408: NanoHTTPD's Status enum has no 504, and this is close enough to be honest.
                return json(Response.Status.REQUEST_TIMEOUT, err("generation timed out"))
            }
            return result.get().fold(
                onSuccess = onSuccess,
                onFailure = { e -> json(Response.Status.INTERNAL_ERROR, err(e.message ?: "engine error")) },
            )
        } finally {
            turn.release()
        }
    }

    private fun err(msg: String): String = JSONObject().put("error", msg).toString()

    private fun json(status: Response.Status, body: String): Response =
        newFixedLengthResponse(status, "application/json", body)

    companion object {
        // Ceiling for one generation, queue wait included. Generous because a >RAM model on a slow
        // flash can legitimately take minutes; a hung engine is caught by the session's own
        // teardown paths, which fail the pending callback well before this fires.
        private const val GENERATE_TIMEOUT_MIN = 15L

        // Clients need a model id to send back; the app has exactly one session open, so it is a
        // constant rather than anything discovered.
        private const val MODEL_ID = "bmoe-local"
    }
}
