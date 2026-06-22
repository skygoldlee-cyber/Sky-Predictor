"""로그 뷰 포맷·플립/합의 라인 처리 (`gui_controller` 8단계 분리)."""

from __future__ import annotations

import html
import re
import threading
import time as _time_module
from typing import Any, Callable, Dict

__all__ = ["append_log_rich"]


def append_log_rich(
    log_view: Any,
    text: str,
    *,
    telegram_bridge_holder: Dict[str, Any],
    regime_led: Any,
    heur_led: Any,
    gpt_led: Any,
    gem_led: Any,
    set_cons_arrow: Callable[[str], None],
) -> None:
    """HEUR_FLIP / DIR_SUMMARY 특수 포맷, 그 외는 plain append."""
    try:
        text = str(text or "")
        try:
            from PySide6.QtGui import QColor
        except Exception:
            QColor = None  # type: ignore

        def _set_action_led(led: Any, action_s: str) -> None:
            try:
                a = str(action_s or "").strip().upper()
            except Exception:
                a = ""
            try:
                led.label.setText(a[:1] if a else "-")
            except Exception:
                pass
            if QColor is None:
                return
            try:
                if a == "BUY":
                    led.setColor(QColor("lawngreen"))
                elif a == "SELL":
                    led.setColor(QColor("red"))
                elif a == "HOLD":
                    led.setColor(QColor("yellow"))
                else:
                    led.setColor(QColor("gray"))
            except Exception:
                pass

        # HEURISTIC/GPT/GEMINI 블록의 action 라인을 직접 파싱해 LED를 즉시 갱신.
        # (DIR_SUMMARY 라인이 없거나 지연될 때도 LED가 BUY/SELL/HOLD를 반영)
        try:
            t = text.strip()
            if t in ("[HEURISTIC]", "[GPT]", "[GEMINI]"):
                telegram_bridge_holder["last_model_tag"] = t.strip("[]")
            ma = re.search(r'"action"\s*:\s*"([A-Za-z\-]+)"', text)
            if ma is not None:
                act = str(ma.group(1) or "").strip().upper()
                last_tag = str(telegram_bridge_holder.get("last_model_tag") or "").strip().upper()
                if last_tag == "HEURISTIC":
                    _set_action_led(heur_led, act)
                elif last_tag == "GPT":
                    _set_action_led(gpt_led, act)
                elif last_tag == "GEMINI":
                    _set_action_led(gem_led, act)
        except Exception:
            pass

        if "[HEUR_FLIP_TRIGGER]" in text:
            try:
                esc = html.escape(text)
                log_view.append(
                    f'<pre style="margin:0"><span style="color:#FFB74D; font-weight:700;">{esc}</span></pre>'
                )
            except Exception:
                try:
                    log_view.append(text)
                except Exception:
                    pass

            try:
                now_m = float(_time_module.monotonic())
                last_m = float(telegram_bridge_holder.get("last_flip_send_ts") or 0.0)
                if (now_m - last_m) >= 5.0:
                    telegram_bridge_holder["last_flip_send_ts"] = now_m

                    b = telegram_bridge_holder.get("bridge")
                    if b is not None and callable(getattr(b, "predict_now", None)):

                        def _send_now() -> None:
                            try:
                                b.predict_now(force=True, include_dir_summary=True)
                            except Exception:
                                pass

                        threading.Thread(target=_send_now, daemon=True).start()
            except Exception:
                pass
            return
        if "[DIR_SUMMARY]" in text and "CONS=" in text:
            try:
                m = re.search(r"(CONS=)([A-Z\-]+)", text)
                cons = (m.group(2) if m else "").strip().upper()

                mh = re.search(r"HEURISTIC=([A-Z\-]+)", text)
                mg = re.search(r"\bGPT=([A-Z\-]+)", text)
                mm = re.search(r"\bGEM(?:INI)?=([A-Z\-]+)", text)
                heur = (mh.group(1) if mh else "").strip().upper()
                gpt = (mg.group(1) if mg else "").strip().upper()
                gem = (mm.group(1) if mm else "").strip().upper()

                _set_action_led(heur_led, heur)
                _set_action_led(gpt_led, gpt)
                _set_action_led(gem_led, gem)

                mv = re.search(r"votes\s*=\s*(\d+)\s*/\s*3", text)
                try:
                    if mv is not None:
                        _ = str(int(mv.group(1)))
                except Exception:
                    pass

                color = None
                if cons == "BUY":
                    color = "lawngreen"
                elif cons == "SELL":
                    color = "red"
                elif cons == "HOLD":
                    color = "yellow"

                try:
                    set_cons_arrow(cons)
                except Exception:
                    pass

                esc = html.escape(text)
                if color and m is not None:
                    start, end = m.span(2)
                    esc2 = html.escape(text[:start])
                    esc_mid = html.escape(text[start:end])
                    esc3 = html.escape(text[end:])
                    esc = (
                        esc2
                        + f'<span style="color:{color}; font-weight:600;">{esc_mid}</span>'
                        + esc3
                    )

                log_view.append(f'<pre style="margin:0">{esc}</pre>')
                return
            except Exception:
                pass

        log_view.append(text)
    except Exception:
        pass
