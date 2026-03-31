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
        self._history_html: viser.GuiHtmlHandle | None = None
        self._input: viser.GuiTextHandle | None = None
        self._send_btn: viser.GuiButtonHandle | None = None
        self._stop_btn: viser.GuiButtonHandle | None = None
        self._messages: list[str] = []
        self._lock = threading.Lock()
        self._running = False
        self._stop_requested = False
        self._update_counter = 0

    def name(self) -> str:
        return "Chat"

    def setup(self, gui: viser.GuiApi, viewer: MujocoViewer) -> None:
        with gui.add_folder("Chat", order=10):
            self._history_html = gui.add_html("")

            # Custom HTML input with Enter key support
            self._input_html = gui.add_html(
                '<div style="display:flex;gap:6px;margin:4px 0;">'
                '<input id="chat-input" type="text" placeholder="e.g. pick up a can" '
                'style="flex:1;padding:6px 10px;border:1px solid #ccc;border-radius:4px;'
                'font-size:13px;outline:none;" />'
                '<button id="chat-send" style="padding:6px 14px;background:#2ecc71;'
                'color:white;border:none;border-radius:4px;cursor:pointer;font-size:13px;">'
                'Send</button>'
                '<button id="chat-stop" style="padding:6px 14px;background:#e74c3c;'
                'color:white;border:none;border-radius:4px;cursor:pointer;font-size:13px;'
                'display:none;">Stop</button>'
                '</div>'
            )

            # Hidden Viser text field as JS→Python bridge
            self._bridge = gui.add_text("_bridge", initial_value="", visible=False)

            # JS: Enter key or Send button → write to bridge field
            self._js_html = gui.add_html(
                '<script>'
                'var inp=document.getElementById("chat-input");'
                'var btn=document.getElementById("chat-send");'
                'if(inp&&btn){'
                '  inp.addEventListener("keydown",function(e){'
                '    if(e.key==="Enter"&&inp.value.trim()){'
                '      e.preventDefault();'
                '      var el=document.querySelector(\'[data-label="_bridge"] input\');'
                '      if(el){var nv=new Event("input",{bubbles:true});'
                '        el.value=inp.value;el.dispatchEvent(nv);'
                '        el.dispatchEvent(new Event("change",{bubbles:true}));'
                '        inp.value="";}}'
                '  });'
                '  btn.addEventListener("click",function(){'
                '    if(inp.value.trim()){'
                '      var el=document.querySelector(\'[data-label="_bridge"] input\');'
                '      if(el){var nv=new Event("input",{bubbles:true});'
                '        el.value=inp.value;el.dispatchEvent(nv);'
                '        el.dispatchEvent(new Event("change",{bubbles:true}));'
                '        inp.value="";}}'
                '  });'
                '}'
                '</script>'
            )

            self._stop_btn = gui.add_button(
                "Stop", color="red", icon=viser.Icon.PLAYER_STOP,
                visible=False,
            )

            def _do_send() -> None:
                msg = self._bridge.value.strip()
                if not msg or self._running:
                    return
                self._bridge.value = ""
                threading.Thread(
                    target=self._send_message, args=(msg, viewer), daemon=True,
                ).start()

            @self._bridge.on_update
            def _(_: viser.GuiEvent) -> None:
                _do_send()

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
        if self._history_html is None:
            return
        with self._lock:
            if not self._messages:
                inner = '<em style="color:#999;">No messages yet.</em>'
            else:
                lines = []
                for msg in reversed(self._messages):
                    if msg.startswith("**You:**"):
                        text = msg[8:].strip()
                        lines.append(f'<div style="margin:4px 0;"><b style="color:#1a6dd4;">You:</b> {_esc(text)}</div>')
                    elif msg.startswith("**Geodude:**"):
                        text = msg[12:].strip()
                        lines.append(f'<div style="margin:4px 0;"><b style="color:#1a8a4a;">Geodude:</b> {_md_to_html(text)}</div>')
                    elif msg.startswith("`") and msg.endswith("`"):
                        text = msg[1:-1]
                        lines.append(f'<div style="margin:2px 0;font-family:monospace;font-size:12px;color:#666;">{_esc(text)}</div>')
                    else:
                        lines.append(f'<div style="margin:2px 0;color:#888;font-style:italic;">{_esc(msg)}</div>')
                inner = "\n".join(lines)

        html = (
            f'<div style="max-height:400px;overflow-y:auto;'
            f'padding:8px;background:#f5f5f5;border-radius:6px;font-size:13px;'
            f'color:#222;">'
            f'{inner}</div>'
        )
        self._history_html.content = html


def _esc(text: str) -> str:
    """Escape HTML special characters."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _md_to_html(text: str) -> str:
    """Minimal markdown → HTML for LLM responses."""
    import re
    text = _esc(text)
    # **bold**
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    # `code`
    text = re.sub(r'`(.+?)`', r'<code style="background:#e0e0e0;padding:1px 4px;border-radius:3px;">\1</code>', text)
    # Numbered lists: lines starting with "1. ", "2. " etc.
    text = re.sub(r'^(\d+)\.\s', r'<br>\1. ', text, flags=re.MULTILINE)
    # Line breaks
    text = text.replace('\n', '<br>')
    return text
