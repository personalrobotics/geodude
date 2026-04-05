# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Browser chat panel for LLM robot control via Viser."""

from __future__ import annotations

import html
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import viser
from mj_viser import MujocoViewer, PanelBase

if TYPE_CHECKING:
    from geodude.chat import ChatSession


@dataclass
class _ChatEntry:
    """A single chat entry (user message, response, or tool group)."""

    kind: str  # "user", "bot", "tools", "status"
    text: str
    timestamp: float = field(default_factory=time.time)
    tool_lines: list[str] = field(default_factory=list)
    cost: float | None = None


class ChatPanel(PanelBase):
    """LLM chat panel in the Viser browser GUI.

    Note: Enter-to-send is not supported due to Viser's add_text
    firing on_update per keystroke (viser#681). Use Send button.
    """

    def __init__(self, chat_session: ChatSession) -> None:
        self._chat = chat_session
        self._history_html: viser.GuiHtmlHandle | None = None
        self._entries: list[_ChatEntry] = []
        self._lock = threading.Lock()
        self._running = False
        self._stop_requested = False
        self._current_tools: list[str] = []

    def name(self) -> str:
        return "Chat"

    def setup(self, gui: viser.GuiApi, viewer: MujocoViewer) -> None:
        with gui.add_folder("Chat", order=10):
            self._history_html = gui.add_html("")
            self._input = gui.add_text(
                "Message",
                initial_value="",
                hint="Type message, click Send",
            )
            self._send_btn = gui.add_button(
                "Send",
                color="green",
                icon=viser.Icon.SEND,
            )
            self._stop_btn = gui.add_button(
                "Stop",
                color="red",
                icon=viser.Icon.PLAYER_STOP,
                visible=False,
            )
            self._clear_btn = gui.add_button(
                "Clear",
                icon=viser.Icon.TRASH,
            )

            @self._send_btn.on_click
            def _(_: viser.GuiEvent) -> None:
                msg = self._input.value.strip()
                if not msg or self._running:
                    return
                self._input.value = ""
                threading.Thread(
                    target=self._send_message,
                    args=(msg, viewer),
                    daemon=True,
                ).start()

            @self._stop_btn.on_click
            def _(_: viser.GuiEvent) -> None:
                self._stop_requested = True
                self._add_entry(_ChatEntry(kind="status", text="Stopped by user"))
                self._render()

            @self._clear_btn.on_click
            def _(_: viser.GuiEvent) -> None:
                with self._lock:
                    self._entries.clear()
                self._render()

    def on_sync(self, viewer: MujocoViewer) -> None:
        pass

    def _send_message(self, message: str, viewer: MujocoViewer) -> None:
        self._running = True
        self._stop_requested = False
        self._current_tools = []
        self._send_btn.visible = False
        self._stop_btn.visible = True

        # User message
        self._add_entry(_ChatEntry(kind="user", text=message))

        # Thinking indicator
        thinking = _ChatEntry(kind="status", text="Thinking...")
        self._add_entry(thinking)
        self._render()

        cost_before = self._chat.estimated_cost()

        # Capture tool call prints
        import builtins

        original_print = builtins.print

        def _capture_print(*args, **kwargs):
            text = " ".join(str(a) for a in args)
            stripped = text.strip()
            if stripped.startswith("\u2192") or stripped.startswith("\u2713") or stripped.startswith("\u2717"):
                self._current_tools.append(stripped)
                # Update thinking to show progress
                thinking.text = (
                    f"Working... ({len(self._current_tools)} action{'s' if len(self._current_tools) != 1 else ''})"
                )
                self._render()
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

        # Remove thinking indicator
        with self._lock:
            if thinking in self._entries:
                self._entries.remove(thinking)

        if self._stop_requested:
            return

        # Add tool calls as a group (if any)
        if self._current_tools:
            self._add_entry(
                _ChatEntry(
                    kind="tools",
                    text="",
                    tool_lines=list(self._current_tools),
                )
            )

        turn_cost = self._chat.estimated_cost() - cost_before

        # Bot response
        if response:
            self._add_entry(
                _ChatEntry(
                    kind="bot",
                    text=response,
                    cost=turn_cost,
                )
            )

        self._render()

    def _add_entry(self, entry: _ChatEntry) -> None:
        with self._lock:
            self._entries.append(entry)
            if len(self._entries) > 30:
                self._entries = self._entries[-30:]

    def _render(self) -> None:
        if self._history_html is None:
            return
        with self._lock:
            entries = list(self._entries)

        if not entries:
            self._history_html.content = (
                '<div style="padding:12px;background:#f5f5f5;border-radius:6px;'
                'color:#999;font-size:13px;text-align:center;">'
                "No messages yet. Type a command and click Send.</div>"
            )
            return

        lines = []
        for entry in reversed(entries):
            ts = time.strftime("%H:%M", time.localtime(entry.timestamp))
            ts_html = f'<span style="color:#aaa;font-size:11px;margin-left:6px;">{ts}</span>'

            if entry.kind == "user":
                lines.append(
                    f'<div style="margin:6px 0;padding:6px 8px;'
                    f'background:#e8f0fe;border-radius:6px;">'
                    f'<b style="color:#1a6dd4;">You</b>{ts_html}<br>'
                    f"{html.escape(entry.text)}</div>"
                )
            elif entry.kind == "bot":
                cost_html = ""
                if entry.cost is not None:
                    cost_html = f'<span style="color:#aaa;font-size:11px;margin-left:6px;">(${entry.cost:.4f})</span>'
                lines.append(
                    f'<div style="margin:6px 0;padding:6px 8px;'
                    f'background:#e8f8e8;border-radius:6px;">'
                    f'<b style="color:#1a8a4a;">Geodude</b>{ts_html}{cost_html}<br>'
                    f"{_md_to_html(entry.text)}</div>"
                )
            elif entry.kind == "tools":
                n = len(entry.tool_lines)
                summary = f"{n} action{'s' if n != 1 else ''}"
                detail = "".join(f'<div style="margin:1px 0;">{html.escape(line)}</div>' for line in entry.tool_lines)
                lines.append(
                    f'<details style="margin:4px 0;font-size:12px;color:#666;">'
                    f'<summary style="cursor:pointer;font-family:monospace;">'
                    f"{summary}</summary>"
                    f'<div style="padding:4px 8px;font-family:monospace;'
                    f'background:#f0f0f0;border-radius:4px;margin-top:2px;">'
                    f"{detail}</div></details>"
                )
            elif entry.kind == "status":
                lines.append(
                    f'<div style="margin:4px 0;color:#888;font-style:italic;'
                    f'font-size:12px;">{html.escape(entry.text)}</div>'
                )

        inner = "\n".join(lines)
        self._history_html.content = (
            f'<div style="max-height:400px;overflow-y:auto;'
            f"padding:8px;background:#f5f5f5;border-radius:6px;"
            f'font-size:13px;color:#222;">{inner}</div>'
        )


def _md_to_html(text: str) -> str:
    import re

    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(
        r"`(.+?)`",
        r'<code style="background:#e0e0e0;padding:1px 4px;'
        r'border-radius:3px;">\1</code>',
        text,
    )
    text = re.sub(r"^(\d+)\.\s", r"<br>\1. ", text, flags=re.MULTILINE)
    text = text.replace("\n", "<br>")
    return text
