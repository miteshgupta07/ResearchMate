"""ResearchMate Normal Chat voice worker. Run as python -m backend.voice.agent.

Credentials come from the repository root .env (shell values take precedence).
Optional shell settings: CARTESIA_VOICE_ID, CARTESIA_LANGUAGE (default: auto).
Diagnostics stay on the console; session recording is explicitly disabled.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
from collections.abc import AsyncIterable, AsyncIterator
from pathlib import Path
from urllib.parse import urlsplit
from uuid import UUID

from dotenv import load_dotenv
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, JobProcess, cli, llm
from livekit.plugins import cartesia, deepgram, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=ROOT / ".env", override=False)
# Keep model downloads with the worker environment unless a cache is configured.
os.environ.setdefault("HF_HOME", str(Path(sys.prefix) / "model-cache"))
os.environ.setdefault("HF_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub"))
REQUIRED = (
    "LIVEKIT_URL",
    "LIVEKIT_API_KEY",
    "LIVEKIT_API_SECRET",
    "DEEPGRAM_API_KEY",
    "CARTESIA_API_KEY",
    "GROQ_API_KEY",
)
DEFAULT_VOICE = "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"  # Jacqueline, multilingual with Sonic 3
INSTRUCTIONS = (
    "You are ResearchMate, an AI research assistant. Help users understand research, "
    "explain concepts, and discuss ideas using your general knowledge and this voice conversation. "
    "Be accurate, friendly, and concise: usually one or two short sentences. "
    "Acknowledge uncertainty and never invent evidence or citations. "
    "Respond in the user's language when supported. Use natural spoken language, "
    "without Markdown, code blocks, emojis, URLs, or formatting symbols. "
    "Never speak your internal reasoning. Ask a brief clarification when needed. "
    "You cannot access uploaded documents, search tools, or stored text chats. "
    "For document questions or Agent Mode tools, ask the user to use ResearchMate's text chat."
)
log = logging.getLogger("researchmate.voice")


def configure() -> None:
    missing = [key for key in REQUIRED if not os.environ.get(key, "").strip()]
    if missing:
        raise ValueError("Missing required environment variables: " + ", ".join(missing))
    url = urlsplit(os.environ["LIVEKIT_URL"])
    if url.scheme not in {"ws", "wss"} or not url.hostname or url.username or url.password:
        raise ValueError("LIVEKIT_URL must be a ws:// or wss:// server URL without credentials")
    try:
        UUID(os.environ.get("CARTESIA_VOICE_ID", DEFAULT_VOICE))
    except ValueError:
        raise ValueError("CARTESIA_VOICE_ID must be a voice UUID") from None


class SafeConsoleFormatter(logging.Formatter):
    """Redact credentials even from third-party exceptions and formatted tracebacks."""

    def format(self, record: logging.LogRecord) -> str:
        output = super().format(record)
        for key in REQUIRED:
            value = os.environ.get(key)
            if value:
                output = output.replace(value, "[redacted]")
        output = re.sub(r"(?i)(authorization[\"':\s]+)(?:bearer|token)\s+\S+", r"\1[redacted]", output)
        return output


def setup_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(SafeConsoleFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
    for name in ("httpx", "httpx2", "httpcore", "httpcore2", "openai"):
        logging.getLogger(name).setLevel(logging.WARNING)


async def remove_urls(text: AsyncIterable[str]) -> AsyncIterator[str]:
    """Hold only an incomplete word so URLs split across LLM chunks stay unspoken."""
    pending = ""
    async for chunk in text:
        pending += chunk
        parts = re.split(r"(\s+)", pending)
        pending = parts.pop()
        for index in range(0, len(parts), 2):
            word, space = parts[index : index + 2]
            if not re.search(r"(?i)(?:https?://|www\.)", word):
                yield word + space
    if pending and not re.search(r"(?i)(?:https?://|www\.)", pending):
        yield pending


def providers():
    language = os.environ.get("CARTESIA_LANGUAGE", "auto").strip()
    return (
        deepgram.STT(model="nova-3", language="multi", interim_results=True, mip_opt_out=True),
        # Official OpenAI-compatible plugin, pointed directly at Groq. Its parser
        # consumes delta.content only; Groq's separate delta.reasoning is discarded.
        # GPT-OSS does not support reasoning_format; do not pass that parameter.
        openai.LLM(
            model="openai/gpt-oss-20b",
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ["GROQ_API_KEY"],
            reasoning_effort="low",
            max_completion_tokens=1024,
            max_retries=0,  # AgentSession owns bounded provider retries.
        ),
        cartesia.TTS(
            model="sonic-3",
            voice=os.environ.get("CARTESIA_VOICE_ID", DEFAULT_VOICE),
            language=None if language == "auto" else language,
        ),
    )


def diagnostics(session: AgentSession, ctx: JobContext) -> None:
    @session.on("user_input_transcribed")
    def user_text(event):
        if event.is_final:
            log.info("Final user transcription (%s): %s", event.language, event.transcript)

    @session.on("conversation_item_added")
    def conversation_item(event):
        item = event.item
        if not isinstance(item, llm.ChatMessage):
            return
        if item.role == "assistant":
            log.info("Assistant%s: %s", " [interrupted]" if item.interrupted else "", item.text_content)
        # These are SDK per-turn measurements. In particular, streaming STT's
        # request duration (normally zero) is NOT a transcription latency.
        names = {
            "transcription_delay": "STT final transcript delay after speech end",
            "end_of_turn_delay": "End-of-turn decision delay",
            "llm_node_ttft": "LLM first content token",
            "tts_node_ttfb": "TTS first generated audio",
            "e2e_latency": "SDK speech-end to response (server/output boundary)",
            "playback_latency": "SDK output playback acknowledgement (not client audibility)",
        }
        for field, label in names.items():
            value = item.metrics.get(field)
            if value is not None and value >= 0:
                log.info("%s: %.1f ms", label, value * 1000)

    @session.on("user_state_changed")
    def user_state(event):
        log.info("User state: %s -> %s", event.old_state, event.new_state)

    @session.on("agent_state_changed")
    def agent_state(event):
        log.info("Agent state: %s -> %s", event.old_state, event.new_state)

    @session.on("agent_false_interruption")
    def false_interruption(event):
        log.info("False interruption; resumed=%s", event.resumed)

    @session.on("error")
    def provider_error(event):
        error = event.error
        log.error(
            "Provider %s: recoverable=%s error=%s",
            type(event.source).__name__,
            getattr(error, "recoverable", False),
            error,
        )

    @session.on("close")
    def closed(event):
        log.info("Session closed: %s", event.reason)

    @ctx.room.on("connection_state_changed")
    def connection_state(state):
        log.info("Room connection state: %s", state)

    @ctx.room.on("disconnected")
    def disconnected(reason):
        log.info("Room disconnected: %s", reason)


def prewarm(proc: JobProcess) -> None:
    setup_logging()
    configure()
    proc.userdata["vad"] = silero.VAD.load()


server = AgentServer(setup_fnc=prewarm, initialize_process_timeout=60, host="127.0.0.1")


@server.rtc_session(agent_name="researchmate-voice")
async def entrypoint(ctx: JobContext) -> None:
    setup_logging()
    configure()
    transcription, model, speech = providers()
    session = AgentSession(
        stt=transcription,
        llm=model,
        tts=speech,
        vad=ctx.proc.userdata["vad"],
        turn_handling={
            "turn_detection": MultilingualModel(),
            "endpointing": {"mode": "fixed", "min_delay": 0.5, "max_delay": 3.0},
            # Explicit VAD mode avoids adding a hosted adaptive interruption model.
            "interruption": {"enabled": True, "mode": "vad", "resume_false_interruption": True},
            "preemptive_generation": {"enabled": False},
        },
        tts_text_transforms=["filter_markdown", "filter_emoji", remove_urls],
    )
    diagnostics(session, ctx)

    async def shutdown() -> None:
        await session.aclose()
        await asyncio.gather(transcription.aclose(), model.aclose(), speech.aclose())
        log.info("Voice providers closed")

    ctx.add_shutdown_callback(shutdown)
    try:
        if not ctx.is_fake_job():  # The console supplies local audio IO without WebRTC.
            await ctx.connect()
        await session.start(room=ctx.room, agent=Agent(instructions=INSTRUCTIONS), record=False)
        log.info("Ready. No recordings or transcript files. Latencies exclude client/network audibility.")
        # Start in English; subsequent answers follow the user's language.
        await session.say("Hello! I am ResearchMate. What would you like to discuss?", allow_interruptions=True)
    except Exception:
        log.exception("Voice session failed; no provider or model substitution was attempted")
        ctx.shutdown(reason="voice pipeline startup failed")
        raise


def main() -> None:
    setup_logging()
    try:
        configure()
        if "--record" in sys.argv:
            raise ValueError("Audio recording is disabled for the voice worker; omit --record")
    except ValueError as error:
        log.error("%s", error)
        raise SystemExit(2) from None
    try:
        cli.run_app(server)
    except KeyboardInterrupt:
        log.info("Stopped")
    except Exception as error:
        log.error("Startup failed (%s): %s", type(error).__name__, error)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
