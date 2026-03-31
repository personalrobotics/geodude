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
        self._messages: list[str] = []
        self._lock = threading.Lock()

    def name(self) -> str:
        return "Chat"

    def setup(self, gui: viser.GuiApi, viewer: MujocoViewer) -> None:
        with gui.add_folder("Chat", order=10):
            self._history_md = gui.add_markdown("*No messages yet.*")
            self._input = gui.add_text("Message", initial_value="", hint="e.g. 'pick up a can'")
            self._send_btn = gui.add_button(
                "Send", color="green", icon=viser.Icon.SEND,
            )

            @self._send_btn.on_click
            def _(_: viser.GuiEvent) -> None:
                msg = self._input.value.strip()
                if not msg:
                    return
                self._input.value = ""
                # Run chat in a thread so the UI stays responsive
                threading.Thread(
                    target=self._send_message, args=(msg, viewer), daemon=True,
                ).start()

    def on_sync(self, viewer: MujocoViewer) -> None:
        # Chat history is updated via _send_message, nothing needed per-frame
        pass

    def _send_message(self, message: str, viewer: MujocoViewer) -> None:
        """Send a message to the LLM and update the chat history."""
        self._append(f"**You:** {message}")
        self._update_display()

        # Monkey-patch print to capture tool call output
        original_print = __builtins__["print"] if isinstance(__builtins__, dict) else __builtins__.print
        tool_lines: list[str] = []

        def _capture_print(*args, **kwargs):
            text = " ".join(str(a) for a in args)
            # Capture tool call lines (→ and ✓/✗)
            if text.strip().startswith("\u2192") or text.strip().startswith("\u2713") or text.strip().startswith("\u2717"):
                tool_lines.append(text.strip())
                self._append(f"`{text.strip()}`")
                self._update_display()
            # Also call original print for terminal
            original_print(*args, **kwargs)

        try:
            if isinstance(__builtins__, dict):
                __builtins__["print"] = _capture_print
            else:
                import builtins
                builtins.print = _capture_print

            response = self._chat.send(message)
        finally:
            if isinstance(__builtins__, dict):
                __builtins__["print"] = original_print
            else:
                import builtins
                builtins.print = original_print

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
