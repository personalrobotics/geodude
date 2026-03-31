"""Browser chat panel for LLM robot control via Viser."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import viser
from mj_viser import MujocoViewer, PanelBase

if TYPE_CHECKING:
    from geodude.chat import ChatSession


class ChatPanel(PanelBase):
    """LLM chat panel in the Viser browser GUI.

    Provides a text input, send button, and scrollable chat history.
    Tool calls are shown inline as they execute.
    """

    def __init__(self, chat_session: ChatSession) -> None:
        self._chat = chat_session
        self._history_md: viser.GuiMarkdownHandle | None = None
        self._input: viser.GuiTextHandle | None = None
        self._send_btn: viser.GuiButtonHandle | None = None
        self._stop_btn: viser.GuiButtonHandle | None = None
        self._messages: list[str] = []
        self._lock = threading.Lock()
        self._running = False
        self._stop_requested = False

    def name(self) -> str:
        return "Chat"

    def setup(self, gui: viser.GuiApi, viewer: MujocoViewer) -> None:
        with gui.add_folder("Chat", order=10):
            self._history_md = gui.add_markdown("*No messages yet.*")
            self._input = gui.add_text("Message", initial_value="", hint="e.g. 'pick up a can'")
            self._send_btn = gui.add_button(
                "Send", color="green", icon=viser.Icon.SEND,
            )
            self._stop_btn = gui.add_button(
                "Stop", color="red", icon=viser.Icon.PLAYER_STOP,
                visible=False,
            )

            @self._send_btn.on_click
            def _(_: viser.GuiEvent) -> None:
                msg = self._input.value.strip()
                if not msg or self._running:
                    return
                self._input.value = ""
                # Run chat in a thread so the UI stays responsive
                threading.Thread(
                    target=self._send_message, args=(msg, viewer), daemon=True,
                ).start()

            @self._stop_btn.on_click
            def _(_: viser.GuiEvent) -> None:
                self._stop_requested = True
                self._append("*(stopped by user)*")
                self._update_display()

    def on_sync(self, viewer: MujocoViewer) -> None:
        # Chat history is updated via _send_message, nothing needed per-frame
        pass

    def _send_message(self, message: str, viewer: MujocoViewer) -> None:
        """Send a message to the LLM and update the chat history."""
        self._running = True
        self._stop_requested = False
        self._send_btn.visible = False
        self._stop_btn.visible = True

        self._append(f"**You:** {message}")
        self._update_display()

        # Monkey-patch print to capture tool call output
        import builtins
        original_print = builtins.print

        def _capture_print(*args, **kwargs):
            text = " ".join(str(a) for a in args)
            # Capture tool call lines (→ and ✓/✗)
            stripped = text.strip()
            if stripped.startswith("\u2192") or stripped.startswith("\u2713") or stripped.startswith("\u2717"):
                self._append(f"`{stripped}`")
                self._update_display()
            original_print(*args, **kwargs)

        try:
            builtins.print = _capture_print
            response = self._chat.send(message)
        except Exception as e:
            response = f"Error: {e}"
        finally:
            builtins.print = original_print
            self._running = False
            self._send_btn.visible = True
            self._stop_btn.visible = False

        if self._stop_requested:
            return

        if response:
            self._append(f"**Geodude:** {response}")
        self._update_display()

    def _append(self, line: str) -> None:
        with self._lock:
            self._messages.append(line)
            # Keep last 50 messages
            if len(self._messages) > 50:
                self._messages = self._messages[-50:]

    def _update_display(self) -> None:
        if self._history_md is None:
            return
        with self._lock:
            md = "\n\n".join(self._messages) if self._messages else "*No messages yet.*"
        self._history_md.content = md
