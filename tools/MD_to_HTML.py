import markdown
from pathlib import Path
import unicodedata
import html
import re
import argparse
import sys
from dataclasses import dataclass



# UX toggles (can be overridden via CLI)
COLLAPSE_CODEBLOCK_MIN_LINES = 35
MERMAID_SANITIZE_MODE = "auto"  # auto|on|off

# Export toggles
EMBED_ASSETS = False
ASSETS_DIR = Path("assets")


@dataclass(frozen=True)
class RenderConfig:
    collapse_codeblock_min_lines: int = COLLAPSE_CODEBLOCK_MIN_LINES
    mermaid_sanitize_mode: str = MERMAID_SANITIZE_MODE
    embed_assets: bool = EMBED_ASSETS
    assets_dir: Path = ASSETS_DIR


def _read_text_if_exists(path: Path) -> str | None:
    try:
        if path and path.exists() and path.is_file():
            return path.read_text(encoding="utf-8")
    except Exception:
        return None
    return None


def embed_assets_into_html(doc_html: str, assets_dir: Path) -> str:
    """Inline local assets into the HTML so the output becomes a single-file document.

    This is intentionally opt-in and only inlines assets that exist in assets_dir.
    Missing files keep the original CDN tags.

    Expected filenames:
      - tailwind.css
      - highlight.min.js
      - highlight.css
      - mermaid.min.js
    """

    if not doc_html:
        return doc_html

    assets_dir = Path(assets_dir) if assets_dir else Path("assets")

    # Tailwind
    tw_css = _read_text_if_exists(assets_dir / "tailwind.css")
    if tw_css:
        # NOTE: This directly inlines local CSS as-is. If the CSS contains a literal
        # "</style>" sequence, it will break the surrounding HTML. This is acceptable
        # only if you trust the local asset contents.
        doc_html = doc_html.replace(
            '<script src="https://cdn.tailwindcss.com"></script>',
            '<style>\n' + tw_css + '\n</style>',
        )

    # highlight.js
    hl_css = _read_text_if_exists(assets_dir / "highlight.css")
    if hl_css:
        doc_html = doc_html.replace(
            '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">',
            '<style>\n' + hl_css + '\n</style>',
        )

    hl_js = _read_text_if_exists(assets_dir / "highlight.min.js")
    if hl_js:
        # NOTE: This directly inlines local JS as-is. If the JS contains a literal
        # "</script>" sequence, it will break the surrounding HTML. This is acceptable
        # only if you trust the local asset contents.
        doc_html = doc_html.replace(
            '<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>',
            '<script>\n' + hl_js + '\n</script>',
        )

    # Mermaid
    mm_js = _read_text_if_exists(assets_dir / "mermaid.min.js")
    if mm_js:
        doc_html = doc_html.replace(
            '<script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>',
            '<script>\n' + mm_js + '\n</script>',
        )

    return doc_html


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="ko" class="scroll-smooth">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>%%TITLE%%</title>

  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">

  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {
      darkMode: 'media',
      theme: {
        extend: {
          typography: {
            DEFAULT: {
              css: {
                maxWidth: '100%',
              }
            }
          }
        }
      }
    }
  </script>

  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
  <script>
    document.addEventListener('DOMContentLoaded', function () {
      try {
        // Only highlight code blocks NOT already processed by Pygments codehilite
        document.querySelectorAll('pre code').forEach(function (el) {
          if (!el.closest('.highlight')) {
            hljs.highlightElement(el);
          }
        });
      } catch (e) {}
    });
  </script>

  <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
  <script>
    document.addEventListener('DOMContentLoaded', function () {
      try {
        if (window.mermaid) {
          // Initial render happens after theme is applied (see theme script)
        }
      } catch (e) {}
    });
  </script>

  <style>
    /* --- Tailwind Fallback (CDN blocked) ---
       This project normally uses Tailwind CDN. If the CDN is blocked,
       these minimal utility class fallbacks keep the doc readable.
    */
    body {
      font-family: system-ui, -apple-system, "Segoe UI", Roboto, "Noto Sans KR", Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
    }

    .mx-auto { margin-left: auto; margin-right: auto; }
    .max-w-screen-2xl { max-width: 1536px; }
    .max-w-screen-xl { max-width: 1280px; }

    .px-4 { padding-left: 1rem; padding-right: 1rem; }
    .py-3 { padding-top: 0.75rem; padding-bottom: 0.75rem; }
    .py-8 { padding-top: 2rem; padding-bottom: 2rem; }

    @media (min-width: 640px) {
      .sm\:px-6 { padding-left: 1.5rem; padding-right: 1.5rem; }
    }
    @media (min-width: 1024px) {
      .lg\:px-8 { padding-left: 2rem; padding-right: 2rem; }
    }

    .min-h-screen { min-height: 100vh; }
    .flex { display: flex; }
    .grid { display: grid; }
    .items-center { align-items: center; }
    .justify-between { justify-content: space-between; }
    .gap-3 { gap: 0.75rem; }

    .grid-cols-1 { grid-template-columns: repeat(1, minmax(0, 1fr)); }
    @media (min-width: 1024px) {
      .lg\:grid-cols-12 { grid-template-columns: repeat(12, minmax(0, 1fr)); }
      .lg\:col-span-3 { grid-column: span 3 / span 3; }
      .lg\:col-span-9 { grid-column: span 9 / span 9; }
      .lg\:block { display: block; }
    }

    .hidden { display: none; }

    .sticky { position: sticky; }
    .top-0 { top: 0; }
    .top-24 { top: 6rem; }
    .z-40 { z-index: 40; }

    .rounded-xl { border-radius: 0.75rem; }
    .rounded-2xl { border-radius: 1rem; }

    .text-xs { font-size: 0.75rem; line-height: 1rem; }
    .text-sm { font-size: 0.875rem; line-height: 1.25rem; }
    .font-bold { font-weight: 700; }
    .font-semibold { font-weight: 600; }
    .font-medium { font-weight: 500; }

    .inline-flex { display: inline-flex; }
    .leading-tight { line-height: 1.25; }
    .overflow-auto { overflow: auto; }

    .fixed { position: fixed; }
    .inset-0 { top: 0; right: 0; bottom: 0; left: 0; }
    .right-0 { right: 0; }
    .w-80 { width: 20rem; }
    .h-screen { height: 100vh; }
    .p-4 { padding: 1rem; }
    .p-3 { padding: 0.75rem; }
    .pt-3 { padding-top: 0.75rem; }
    .mt-3 { margin-top: 0.75rem; }
    .w-full { width: 100%; }
    .border-b { border-bottom-width: 1px; }
    .border { border-width: 1px; }
    .shadow-xl { box-shadow: 0 20px 60px rgba(2,6,23,0.35); }
    .transition { transition: all 0.2s ease; }

    .sm\:hidden { display: block; }
    @media (min-width: 640px) { .sm\:hidden { display: none; } }

    /* Mobile TOC drawer */
    .drawer-backdrop {
      background: rgba(2, 6, 23, 0.55);
      backdrop-filter: blur(4px);
      -webkit-backdrop-filter: blur(4px);
    }
    html[data-theme="light"] .drawer-backdrop { background: rgba(15, 23, 42, 0.25); }

    .drawer-panel {
      height: 100vh;
      max-width: 22rem;
      width: 85vw;
      background: var(--panel);
      border-left: 1px solid var(--border);
      box-shadow: var(--shadow);
    }

    .drawer-hidden { display: none; }

    /* Toast */
    #toast {
      position: fixed;
      left: 50%;
      bottom: 20px;
      transform: translateX(-50%);
      padding: 0.6rem 0.9rem;
      border-radius: 9999px;
      border: 1px solid var(--border);
      background: rgba(2, 6, 23, 0.72);
      color: var(--fg);
      font-size: 0.85rem;
      box-shadow: var(--shadow);
      display: none;
      z-index: 60;
    }
    html[data-theme="light"] #toast {
      background: rgba(255, 255, 255, 0.92);
      color: rgba(15, 23, 42, 0.92);
    }

    /* Floating Theme button (failsafe when topbar buttons are hidden/clipped) */
    .theme-fab {
      position: fixed;
      right: 18px;
      bottom: 18px;
      z-index: 9999;
      border: 1px solid rgba(226, 232, 240, 0.18);
      background: rgba(2, 6, 23, 0.75);
      color: var(--fg);
      padding: 0.65rem 0.85rem;
      border-radius: 9999px;
      font-size: 0.8rem;
      font-weight: 700;
      box-shadow: var(--shadow);
      cursor: pointer;
      user-select: none;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      pointer-events: auto;
    }
    .theme-fab:hover { background: rgba(2, 6, 23, 0.88); }
    html[data-theme="light"] .theme-fab {
      background: rgba(255, 255, 255, 0.92);
      color: rgba(15, 23, 42, 0.92);
      border: 1px solid rgba(15, 23, 42, 0.14);
    }

    /* Collapsible code blocks */
    .codewrap.collapsed {
      max-height: 18rem;
      overflow: hidden;
      mask-image: linear-gradient(to bottom, rgba(0,0,0,1) 60%, rgba(0,0,0,0));
      -webkit-mask-image: linear-gradient(to bottom, rgba(0,0,0,1) 60%, rgba(0,0,0,0));
    }
    .expand-btn {
      position: absolute;
      left: 0.6rem;
      top: 0.6rem;
      padding: 0.35rem 0.55rem;
      border-radius: 0.7rem;
      font-size: 0.75rem;
      line-height: 1rem;
      font-weight: 600;
      border: 1px solid var(--border);
      background: rgba(226, 232, 240, 0.06);
      color: var(--fg);
      cursor: pointer;
      user-select: none;
    }
    .expand-btn:hover { background: rgba(226, 232, 240, 0.10); }
    html[data-theme="light"] .expand-btn { background: rgba(255, 255, 255, 0.70); }

    .admonition {
      border: 1px solid var(--border);
      background: var(--panel);
      border-radius: 0.9rem;
      padding: 0.85rem 1rem;
      margin: 1rem 0;
    }
    .admonition > .admonition-title {
      font-weight: 700;
      margin-bottom: 0.5rem;
      color: var(--fg);
    }
    .admonition.note { border-left: 4px solid rgba(59,130,246,0.65); }
    .admonition.tip { border-left: 4px solid rgba(34,197,94,0.65); }
    .admonition.warning { border-left: 4px solid rgba(250,204,21,0.65); }
    .admonition.danger { border-left: 4px solid rgba(244,63,94,0.65); }

    :root {
      color-scheme: dark;
      --fg: rgba(226, 232, 240, 0.88);
      --muted: rgba(226, 232, 240, 0.68);
      --panel: rgba(15, 23, 42, 0.55);
      --border: rgba(148, 163, 184, 0.16);
      --shadow: 0 10px 30px rgba(0, 0, 0, 0.35);

      --a1: rgba(59,130,246,1);   /* blue */
      --a2: rgba(168,85,247,1);   /* violet */
      --a3: rgba(34,197,94,1);    /* green */
      --a4: rgba(20,184,166,1);   /* teal */
      --a5: rgba(244,63,94,1);    /* rose */
      --a6: rgba(250,204,21,1);   /* amber */
    }

    [data-theme="light"] {
      color-scheme: light;
      --fg: rgba(15, 23, 42, 0.92);
      --muted: rgba(15, 23, 42, 0.72);
      --panel: rgba(248, 250, 252, 0.96);
      --border: rgba(15, 23, 42, 0.14);
      --shadow: 0 10px 25px rgba(2, 6, 23, 0.08);

      --a1: rgba(37, 99, 235, 1);
      --a2: rgba(124, 58, 237, 1);
      --a3: rgba(22, 163, 74, 1);
      --a4: rgba(13, 148, 136, 1);
      --a5: rgba(225, 29, 72, 1);
      --a6: rgba(202, 138, 4, 1);
    }

    .doc-bg {
      background: radial-gradient(1200px 700px at 20% -10%, rgba(59,130,246,0.18), transparent 60%),
                  radial-gradient(900px 600px at 80% 10%, rgba(168,85,247,0.14), transparent 55%),
                  radial-gradient(1000px 700px at 40% 110%, rgba(34,197,94,0.10), transparent 60%),
                  linear-gradient(180deg, #030712 0%, #020617 55%, #030712 100%);
    }

    .glass {
      background: var(--panel);
      border: 1px solid var(--border);
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
      -webkit-backdrop-filter: blur(10px);
      color: var(--fg);
      position: relative;
      contain: none;
    }

    html[data-theme="light"] .glass {
      background: rgba(241, 245, 249, 0.92);
      backdrop-filter: none;
      -webkit-backdrop-filter: none;
    }

    body {
      color: var(--fg);
      background: #020617;
    }

    html[data-theme="light"] body {
      background: #e2e8f0; /* slate-200 */
    }

    html[data-theme="light"] .doc-bg {
      background: radial-gradient(1200px 700px at 25% -10%, rgba(37,99,235,0.12), transparent 60%),
                  radial-gradient(900px 600px at 80% 0%, rgba(124,58,237,0.10), transparent 55%),
                  linear-gradient(180deg, #f1f5f9 0%, #e2e8f0 60%, #f1f5f9 100%);
    }

    .doc-subtitle { color: rgba(226, 232, 240, 0.70); }
    html[data-theme="light"] .doc-subtitle { color: var(--muted); }

    .brand-badge {
      background: rgba(226, 232, 240, 0.08);
      border: 1px solid rgba(226, 232, 240, 0.10);
      color: var(--fg);
    }
    html[data-theme="light"] .brand-badge {
      background: rgba(255, 255, 255, 0.70);
      border: 1px solid rgba(15, 23, 42, 0.10);
      color: var(--fg);
    }

    .theme-btn {
      border: 1px solid rgba(226, 232, 240, 0.18);
      background: rgba(226, 232, 240, 0.10);
      color: var(--fg);
    }
    .theme-btn:hover { background: rgba(226, 232, 240, 0.16); }
    html[data-theme="light"] .theme-btn {
      border: 1px solid rgba(15, 23, 42, 0.10);
      background: rgba(255, 255, 255, 0.70);
      color: var(--fg);
    }
    html[data-theme="light"] .theme-btn:hover { background: rgba(255, 255, 255, 0.90); }

    #searchOverlay {
      position: fixed;
      top: 72px;
      left: 50%;
      transform: translateX(-50%);
      z-index: 9999;
      width: min(860px, calc(100vw - 24px));
      display: none;
    }
    #searchOverlay[data-open="1"] { display: block; }
    .search-bar {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.75rem;
      border-radius: 1rem;
    }
    .search-input {
      flex: 1;
      border-radius: 0.9rem;
      border: 1px solid var(--border);
      background: rgba(255, 255, 255, 0.06);
      color: var(--fg);
      padding: 0.6rem 0.75rem;
      font-size: 0.95rem;
      outline: none;
    }
    html[data-theme="light"] .search-input {
      background: rgba(255, 255, 255, 0.80);
    }
    .search-meta {
      color: var(--muted);
      font-size: 0.85rem;
      white-space: nowrap;
      padding: 0 0.25rem;
    }
    mark.search-mark {
      background: rgba(250, 204, 21, 0.30);
      border: 1px solid rgba(250, 204, 21, 0.32);
      color: inherit;
      padding: 0.02rem 0.12rem;
      border-radius: 0.25rem;
    }
    html[data-theme="light"] mark.search-mark {
      background: rgba(234, 179, 8, 0.22);
      border-color: rgba(234, 179, 8, 0.28);
    }
    mark.search-mark.search-active {
      background: rgba(59, 130, 246, 0.28);
      border-color: rgba(59, 130, 246, 0.34);
    }
    html[data-theme="light"] mark.search-mark.search-active {
      background: rgba(37, 99, 235, 0.18);
      border-color: rgba(37, 99, 235, 0.22);
    }

    /* Pygments codehilite token colors — dark (Monokai-inspired) */
    .highlight .hll { background-color: #49483e; }
    .highlight .c, .highlight .ch, .highlight .cm, .highlight .cp,
    .highlight .cpf, .highlight .c1, .highlight .cs { color: #959077; }
    .highlight .k, .highlight .kc, .highlight .kd, .highlight .kp,
    .highlight .kr, .highlight .kt { color: #66d9ef; }
    .highlight .kn { color: #ff4689; }
    .highlight .o, .highlight .ow { color: #ff4689; }
    .highlight .n, .highlight .nb, .highlight .ni, .highlight .nl,
    .highlight .nn, .highlight .nv, .highlight .bp,
    .highlight .vc, .highlight .vg, .highlight .vi, .highlight .vm,
    .highlight .py, .highlight .fm { color: #f8f8f2; }
    .highlight .na, .highlight .nc, .highlight .nd, .highlight .ne,
    .highlight .nf, .highlight .nx { color: #a6e22e; }
    .highlight .nt { color: #ff4689; }
    .highlight .no { color: #66d9ef; }
    .highlight .s, .highlight .sa, .highlight .sb, .highlight .sc,
    .highlight .dl, .highlight .sd, .highlight .s2, .highlight .sh,
    .highlight .si, .highlight .sx, .highlight .sr, .highlight .s1,
    .highlight .ss { color: #e6db74; }
    .highlight .se { color: #ae81ff; }
    .highlight .m, .highlight .mb, .highlight .mf, .highlight .mh,
    .highlight .mi, .highlight .mo, .highlight .il { color: #ae81ff; }
    .highlight .l, .highlight .ld { color: #ae81ff; }
    .highlight .p, .highlight .pm { color: #f8f8f2; }
    .highlight .w { color: #f8f8f2; }
    .highlight .err { color: #ed007e; }
    .highlight .g, .highlight .ge, .highlight .gr, .highlight .gh,
    .highlight .gs, .highlight .gt, .highlight .gu,
    .highlight .gi, .highlight .gd, .highlight .go, .highlight .gp,
    .highlight .ges, .highlight .esc, .highlight .x { color: #f8f8f2; }
    .highlight .ge { font-style: italic; }
    .highlight .gs { font-weight: bold; }
    .highlight .gd { color: #ff4689; }
    .highlight .gi { color: #a6e22e; }
    .highlight .gp { color: #ff4689; font-weight: bold; }

    /* Pygments codehilite token colors — light (GitHub-inspired) */
    html[data-theme="light"] .highlight .hll { background-color: #ffffcc; }
    html[data-theme="light"] .highlight .c, html[data-theme="light"] .highlight .ch,
    html[data-theme="light"] .highlight .cm, html[data-theme="light"] .highlight .cp,
    html[data-theme="light"] .highlight .cpf, html[data-theme="light"] .highlight .c1,
    html[data-theme="light"] .highlight .cs { color: #6a737d; font-style: italic; }
    html[data-theme="light"] .highlight .k, html[data-theme="light"] .highlight .kc,
    html[data-theme="light"] .highlight .kd, html[data-theme="light"] .highlight .kp,
    html[data-theme="light"] .highlight .kr, html[data-theme="light"] .highlight .kt,
    html[data-theme="light"] .highlight .kn { color: #d73a49; }
    html[data-theme="light"] .highlight .o,
    html[data-theme="light"] .highlight .ow { color: #d73a49; }
    html[data-theme="light"] .highlight .n, html[data-theme="light"] .highlight .nb,
    html[data-theme="light"] .highlight .ni, html[data-theme="light"] .highlight .nl,
    html[data-theme="light"] .highlight .nn, html[data-theme="light"] .highlight .nv,
    html[data-theme="light"] .highlight .bp, html[data-theme="light"] .highlight .vc,
    html[data-theme="light"] .highlight .vg, html[data-theme="light"] .highlight .vi,
    html[data-theme="light"] .highlight .vm, html[data-theme="light"] .highlight .py { color: #24292e; }
    html[data-theme="light"] .highlight .na { color: #005cc5; }
    html[data-theme="light"] .highlight .nc, html[data-theme="light"] .highlight .nd,
    html[data-theme="light"] .highlight .ne, html[data-theme="light"] .highlight .nf,
    html[data-theme="light"] .highlight .nx, html[data-theme="light"] .highlight .fm { color: #6f42c1; }
    html[data-theme="light"] .highlight .nt { color: #22863a; }
    html[data-theme="light"] .highlight .no { color: #005cc5; }
    html[data-theme="light"] .highlight .s, html[data-theme="light"] .highlight .sa,
    html[data-theme="light"] .highlight .sb, html[data-theme="light"] .highlight .sc,
    html[data-theme="light"] .highlight .dl, html[data-theme="light"] .highlight .sd,
    html[data-theme="light"] .highlight .s2, html[data-theme="light"] .highlight .sh,
    html[data-theme="light"] .highlight .si, html[data-theme="light"] .highlight .sx,
    html[data-theme="light"] .highlight .sr, html[data-theme="light"] .highlight .s1,
    html[data-theme="light"] .highlight .ss, html[data-theme="light"] .highlight .se { color: #032f62; }
    html[data-theme="light"] .highlight .m, html[data-theme="light"] .highlight .mb,
    html[data-theme="light"] .highlight .mf, html[data-theme="light"] .highlight .mh,
    html[data-theme="light"] .highlight .mi, html[data-theme="light"] .highlight .mo,
    html[data-theme="light"] .highlight .il { color: #005cc5; }
    html[data-theme="light"] .highlight .l,
    html[data-theme="light"] .highlight .ld { color: #005cc5; }
    html[data-theme="light"] .highlight .p,
    html[data-theme="light"] .highlight .pm { color: #24292e; }
    html[data-theme="light"] .highlight .w { color: #24292e; }
    html[data-theme="light"] .highlight .err { color: #cb2431; }
    html[data-theme="light"] .highlight .g, html[data-theme="light"] .highlight .ge,
    html[data-theme="light"] .highlight .gr, html[data-theme="light"] .highlight .gh,
    html[data-theme="light"] .highlight .gs, html[data-theme="light"] .highlight .gt,
    html[data-theme="light"] .highlight .gu, html[data-theme="light"] .highlight .esc,
    html[data-theme="light"] .highlight .x { color: #24292e; }
    html[data-theme="light"] .highlight .gd { color: #cb2431; }
    html[data-theme="light"] .highlight .gi { color: #22863a; }
    html[data-theme="light"] .highlight .gp { color: #005cc5; font-weight: bold; }
    html[data-theme="light"] .highlight .go { color: #6a737d; }

    /* nicer scrollbars (webkit only) */
    #toc::-webkit-scrollbar, article pre::-webkit-scrollbar { height: 10px; width: 10px; }
    #toc::-webkit-scrollbar-thumb, article pre::-webkit-scrollbar-thumb {
      background: rgba(148, 163, 184, 0.24);
      border-radius: 9999px;
      border: 2px solid rgba(2,6,23,0.65);
    }
    #toc::-webkit-scrollbar-thumb:hover, article pre::-webkit-scrollbar-thumb:hover { background: rgba(148, 163, 184, 0.34); }

    #toc ul { list-style: none; padding-left: 0; margin: 0.25rem 0 0; }
    #toc li { margin: 0.125rem 0; }
    #toc a {
      display: block;
      padding: 0.35rem 0.5rem;
      border-radius: 0.6rem;
      color: var(--muted);
      text-decoration: none;
    }
    #toc a:hover { background: rgba(20, 184, 166, 0.12); color: var(--fg); }
    #toc a.toc-active {
      background: rgba(20, 184, 166, 0.14);
      border: 1px solid rgba(20, 184, 166, 0.28);
      font-weight: 600;
    }
    #toc .toc > ul { margin-top: 0.25rem; }

    .toc-section {
      margin-top: 0.25rem;
    }
    .toc-section-header {
      display: flex;
      align-items: center;
      gap: 0.25rem;
    }
    .toc-toggle {
      width: 1.5rem;
      height: 1.5rem;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      border-radius: 0.5rem;
      border: 1px solid transparent;
      background: transparent;
      color: var(--muted);
      cursor: pointer;
      user-select: none;
      font-weight: 700;
      line-height: 1;
    }
    .toc-toggle:hover {
      background: rgba(226, 232, 240, 0.08);
      border-color: var(--border);
      color: var(--fg);
    }
    .toc-children {
      margin-left: 0.75rem;
      border-left: 1px solid rgba(148, 163, 184, 0.14);
      padding-left: 0.5rem;
      margin-top: 0.15rem;
    }
    .toc-collapsed .toc-children { display: none; }

    .toc-mark {
      background: rgba(250, 204, 21, 0.22);
      border: 1px solid rgba(250, 204, 21, 0.22);
      color: inherit;
      padding: 0.02rem 0.18rem;
      border-radius: 0.25rem;
    }
    html[data-theme="light"] .toc-mark {
      background: rgba(234, 179, 8, 0.24);
      border-color: rgba(234, 179, 8, 0.30);
    }

    .toc-title { color: var(--fg); }
    .toc-subtitle { color: var(--muted); }
    html[data-theme="light"] .toc-title { color: rgba(15, 23, 42, 0.92); }
    html[data-theme="light"] .toc-subtitle { color: rgba(15, 23, 42, 0.72); }

    article { color: var(--fg); line-height: 1.75; font-size: 1.0625rem; max-width: clamp(72ch, 82vw, 96ch); margin-left: auto; margin-right: auto; }
    article p { color: var(--fg); margin: 1rem 0; }
    article li { color: var(--fg); margin: 0.25rem 0; }
    article ul, article ol { padding-left: 1.5rem; margin: 0.5rem 0; }
    article ul { list-style-type: disc; }
    article ol { list-style-type: decimal; }
    article hr {
      border: none;
      height: 1px;
      background: var(--border);
      margin: 2rem 0;
    }
    article a { color: var(--a1); text-decoration: underline; text-underline-offset: 3px; }
    article a:hover { color: var(--a4); }
    article strong { color: rgba(248, 250, 252, 0.98); font-weight: 700; }
    html[data-theme="light"] article strong { color: rgba(15, 23, 42, 0.98); }
    article em { color: var(--muted); font-style: italic; }
    article .headerlink {
      opacity: 0;
      transition: opacity 0.15s;
      text-decoration: none;
      margin-left: 0.3rem;
      color: var(--muted);
    }
    article h1:hover .headerlink,
    article h2:hover .headerlink,
    article h3:hover .headerlink,
    article h4:hover .headerlink { opacity: 0.6; }
    article .headerlink:hover { opacity: 1; }

    article h1 {
      font-size: 2rem;
      line-height: 2.5rem;
      margin: 0 0 1rem;
      background: linear-gradient(90deg, var(--a4), var(--a1), var(--a2));
      -webkit-background-clip: text;
      background-clip: text;
      color: transparent;
      letter-spacing: -0.02em;
    }
    article h2 {
      font-size: 1.5rem;
      line-height: 2rem;
      margin: 2.25rem 0 0.75rem;
      color: var(--fg);
      position: relative;
      padding-top: 0.2rem;
    }
    article h2:before {
      content: '';
      display: block;
      height: 2px;
      width: 2.5rem;
      margin-bottom: 0.7rem;
      background: linear-gradient(90deg, rgba(168,85,247,0.95), rgba(59,130,246,0.75), rgba(20,184,166,0.75));
      border-radius: 9999px;
    }
    article h3 { font-size: 1.25rem; line-height: 1.75rem; margin: 1.75rem 0 0.5rem; color: var(--a4); }
    article h4 { font-size: 1.125rem; line-height: 1.75rem; margin: 1.25rem 0 0.5rem; color: var(--muted); font-weight: 600; }

    article code {
      font-family: "Cascadia Mono", "Cascadia Mono PL", Consolas, "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, "Liberation Mono", "Courier New", monospace;
      font-size: 0.95em;
      background: rgba(148, 163, 184, 0.12);
      border: 1px solid rgba(148, 163, 184, 0.18);
      color: var(--fg);
      padding: 0.12rem 0.35rem;
      border-radius: 0.45rem;
    }
    html[data-theme="light"] article code {
      background: rgba(148, 163, 184, 0.14);
      border: 1px solid rgba(148, 163, 184, 0.22);
      color: rgba(15, 23, 42, 0.92);
    }
    article pre {
      background: rgba(2, 6, 23, 0.85);
      border: 1px solid rgba(59, 130, 246, 0.18);
      border-radius: 0.9rem;
      padding: 1rem;
      overflow: auto;
      white-space: pre;
      tab-size: 4;
      -moz-tab-size: 4;
      font-variant-ligatures: none;
      font-feature-settings: "liga" 0, "calt" 0;
      letter-spacing: 0;
      font-family: "Cascadia Mono", "Cascadia Mono PL", Consolas, "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, "Liberation Mono", "Courier New", monospace;
      font-kerning: none;
      line-height: 1.52;
    }
    html[data-theme="light"] article pre {
      background: #f6f8fa;
      border: 1px solid rgba(15, 23, 42, 0.12);
      color: #24292e;
    }
    html[data-theme="light"] article pre code {
      color: #24292e;
    }
    html[data-theme="light"] .highlight,
    html[data-theme="light"] .highlight pre,
    html[data-theme="light"] .highlight pre code {
      color: #24292e !important;
      background: #f6f8fa !important;
    }
    html[data-theme="light"] .highlight {
      border-radius: 0.9rem;
    }
    article pre code {
      background: transparent;
      border: none;
      padding: 0;
      white-space: pre;
      tab-size: 4;
      -moz-tab-size: 4;
      font-variant-ligatures: none;
      font-feature-settings: "liga" 0, "calt" 0;
      letter-spacing: 0;
      font-family: inherit;
      font-kerning: none;
    }

    /* python-markdown codehilite wrapper */
    .highlight pre {
      white-space: pre;
      tab-size: 4;
      -moz-tab-size: 4;
      font-variant-ligatures: none;
      font-feature-settings: "liga" 0, "calt" 0;
      letter-spacing: 0;
      font-family: "Cascadia Mono", "Cascadia Mono PL", Consolas, "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, "Liberation Mono", "Courier New", monospace;
      font-kerning: none;
      line-height: 1.52;
    }

    .highlight pre *, article pre * {
      font-family: inherit;
      font-variant-ligatures: none;
      font-feature-settings: "liga" 0, "calt" 0;
      letter-spacing: 0;
    }

    /* highlight.js may add .hljs and nested spans; force mono to prevent glyph fallback */
    .hljs, .hljs * {
      font-family: "Cascadia Mono", "Cascadia Mono PL", Consolas, "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, "Liberation Mono", "Courier New", monospace !important;
      font-variant-ligatures: none !important;
      font-feature-settings: "liga" 0, "calt" 0 !important;
      letter-spacing: 0 !important;
      font-kerning: none !important;
    }

    /* --- highlight.js CSS fallback theme (when CDN CSS is blocked) --- */
    .hljs {
      display: block;
      overflow-x: auto;
      padding: 0;
      color: rgba(226, 232, 240, 0.92);
      background: transparent;
    }
    html[data-theme="light"] .hljs {
      color: #24292e;
    }
    html[data-theme="light"] .hljs-comment,
    html[data-theme="light"] .hljs-quote {
      color: #6a737d;
    }
    html[data-theme="light"] .hljs-keyword,
    html[data-theme="light"] .hljs-selector-tag,
    html[data-theme="light"] .hljs-subst {
      color: #d73a49;
    }
    html[data-theme="light"] .hljs-string,
    html[data-theme="light"] .hljs-doctag,
    html[data-theme="light"] .hljs-regexp {
      color: #032f62;
    }
    html[data-theme="light"] .hljs-title,
    html[data-theme="light"] .hljs-section,
    html[data-theme="light"] .hljs-selector-id,
    html[data-theme="light"] .hljs-selector-class {
      color: #6f42c1;
    }
    html[data-theme="light"] .hljs-number,
    html[data-theme="light"] .hljs-literal,
    html[data-theme="light"] .hljs-symbol,
    html[data-theme="light"] .hljs-bullet {
      color: #005cc5;
    }
    html[data-theme="light"] .hljs-attr,
    html[data-theme="light"] .hljs-attribute,
    html[data-theme="light"] .hljs-variable,
    html[data-theme="light"] .hljs-template-variable,
    html[data-theme="light"] .hljs-type {
      color: #005cc5;
    }
    html[data-theme="light"] .hljs-built_in,
    html[data-theme="light"] .hljs-builtin-name {
      color: #e36209;
    }
    html[data-theme="light"] .hljs-meta,
    html[data-theme="light"] .hljs-meta-keyword,
    html[data-theme="light"] .hljs-meta-string {
      color: #6a737d;
    }
    .hljs-comment,
    .hljs-quote {
      color: rgba(148, 163, 184, 0.80);
      font-style: italic;
    }
    .hljs-keyword,
    .hljs-selector-tag,
    .hljs-subst {
      color: rgba(168, 85, 247, 0.95);
      font-weight: 600;
    }
    .hljs-string,
    .hljs-doctag,
    .hljs-regexp {
      color: rgba(34, 197, 94, 0.95);
    }
    .hljs-title,
    .hljs-section,
    .hljs-selector-id,
    .hljs-selector-class {
      color: rgba(59, 130, 246, 0.95);
      font-weight: 600;
    }
    .hljs-number,
    .hljs-literal,
    .hljs-symbol,
    .hljs-bullet {
      color: rgba(250, 204, 21, 0.95);
    }
    .hljs-attr,
    .hljs-attribute,
    .hljs-variable,
    .hljs-template-variable,
    .hljs-type {
      color: rgba(94, 234, 212, 0.95);
    }
    .hljs-built_in,
    .hljs-builtin-name {
      color: rgba(244, 63, 94, 0.95);
    }
    .hljs-meta,
    .hljs-meta-keyword,
    .hljs-meta-string {
      color: rgba(203, 213, 225, 0.95);
    }
    .hljs-emphasis { font-style: italic; }
    .hljs-strong { font-weight: 700; }
    .hljs-addition { background: rgba(34, 197, 94, 0.12); }
    .hljs-deletion { background: rgba(244, 63, 94, 0.12); }
    /* --- end highlight.js fallback theme --- */

    article pre, article pre code, .highlight pre, .highlight pre code {
      font-family: "Cascadia Mono", "Cascadia Mono PL", Consolas, "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, "Liberation Mono", "Courier New", monospace !important;
      font-variant-ligatures: none !important;
      font-feature-settings: "liga" 0, "calt" 0 !important;
      letter-spacing: 0 !important;
      font-kerning: none !important;
    }

    /* ASCII/box-drawing diagrams: try harder to keep CJK glyphs monospaced on Windows/Chrome */
    .ascii-diagram,
    .ascii-diagram code {
      font-family: "D2Coding", "NanumGothicCoding", "Cascadia Mono", "Cascadia Mono PL", Consolas, "JetBrains Mono", "GulimChe", "DotumChe", "MS Gothic", ui-monospace, SFMono-Regular, Menlo, Monaco, "Liberation Mono", "Courier New", monospace !important;
      font-variant-ligatures: none !important;
      font-feature-settings: "liga" 0, "calt" 0 !important;
      letter-spacing: 0 !important;
      font-kerning: none !important;
      white-space: pre !important;
      tab-size: 4;
      -moz-tab-size: 4;
      text-rendering: optimizeSpeed;
    }

    /* Mermaid blocks (offline lite: shown as plain text) */
    .mermaid {
      white-space: pre;
      tab-size: 4;
      -moz-tab-size: 4;
      font-variant-ligatures: none;
      font-feature-settings: "liga" 0, "calt" 0;
      letter-spacing: 0;
      font-family: "Cascadia Mono", "Cascadia Mono PL", Consolas, "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, "Liberation Mono", "Courier New", monospace;
      font-kerning: none;
      line-height: 1.35;
      background: rgba(2, 6, 23, 0.85);
      border: 1px solid rgba(59, 130, 246, 0.18);
      border-radius: 0.9rem;
      padding: 1rem;
      overflow: auto;
      margin: 1rem 0;
      color: rgba(226, 232, 240, 0.92);
    }
    html[data-theme="light"] .mermaid {
      background: rgba(255, 255, 255, 0.98);
      border: 1px solid rgba(15, 23, 42, 0.18);
      color: rgba(15, 23, 42, 0.92);
    }

    .mermaid svg {
      display: block;
      margin: 0.25rem auto;
    }

    /* Ensure labels remain visible regardless of mermaid theme defaults */
    .mermaid svg text,
    .mermaid svg .label text,
    .mermaid svg .edgeLabel,
    .mermaid svg .edgeLabel text {
      fill: currentColor !important;
      color: currentColor !important;
    }

    .mermaid-fallback {
      white-space: pre;
      tab-size: 4;
      -moz-tab-size: 4;
      font-variant-ligatures: none;
      font-feature-settings: "liga" 0, "calt" 0;
      letter-spacing: 0;
      font-family: "Cascadia Mono", "Cascadia Mono PL", Consolas, "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, "Liberation Mono", "Courier New", monospace;
      line-height: 1.35;
      background: rgba(2, 6, 23, 0.85);
      border: 1px solid rgba(244, 63, 94, 0.22);
      border-radius: 0.9rem;
      padding: 1rem;
      overflow: auto;
      margin: 1rem 0;
    }

    .mermaid-error {
      border: 1px solid rgba(244, 63, 94, 0.22);
      background: rgba(244, 63, 94, 0.04);
      border-radius: 0.9rem;
      padding: 0.85rem;
      margin: 1rem 0;
    }
    .mermaid-error-title {
      font-weight: 800;
      font-size: 0.9rem;
      color: rgba(254, 226, 226, 0.92);
      margin-bottom: 0.5rem;
    }
    html[data-theme="light"] .mermaid-error-title {
      color: rgba(190, 18, 60, 0.92);
    }
    .mermaid-error details {
      margin-top: 0.6rem;
    }
    .mermaid-error summary {
      cursor: pointer;
      user-select: none;
      color: var(--muted);
      font-size: 0.85rem;
    }
    .mermaid-error pre.mermaid-error-msg {
      margin-top: 0.5rem;
      white-space: pre-wrap;
      word-break: break-word;
      background: rgba(2, 6, 23, 0.55);
      border: 1px solid rgba(148, 163, 184, 0.18);
      border-radius: 0.75rem;
      padding: 0.6rem 0.7rem;
      color: rgba(226, 232, 240, 0.92);
      overflow: auto;
    }
    html[data-theme="light"] .mermaid-error pre.mermaid-error-msg {
      background: rgba(15, 23, 42, 0.06);
      color: rgba(15, 23, 42, 0.88);
    }

    /* Copy button for code blocks */
    .codewrap { position: relative; }
    .copy-btn {
      position: absolute;
      top: 0.6rem;
      right: 0.6rem;
      padding: 0.35rem 0.55rem;
      border-radius: 0.7rem;
      font-size: 0.75rem;
      line-height: 1rem;
      font-weight: 600;
      border: 1px solid var(--border);
      background: rgba(226, 232, 240, 0.06);
      color: var(--fg);
      cursor: pointer;
      user-select: none;
    }
    .copy-btn:hover { background: rgba(226, 232, 240, 0.10); }
    html[data-theme="light"] .copy-btn {
      background: rgba(255, 255, 255, 0.70);
    }

    .lang-label {
      position: absolute;
      top: 0.6rem;
      right: 4.2rem;
      padding: 0.2rem 0.5rem;
      border-radius: 0.5rem;
      font-size: 0.7rem;
      line-height: 1rem;
      font-weight: 600;
      color: rgba(148, 163, 184, 0.7);
      user-select: none;
      pointer-events: none;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }
    html[data-theme="light"] .lang-label {
      color: rgba(100, 116, 139, 0.8);
    }

    article table {
      width: 100%;
      border-collapse: collapse;
      margin: 1.25rem 0;
      display: block;
      overflow-x: auto;
      -webkit-overflow-scrolling: touch;
    }
    article th, article td {
      border: 1px solid rgba(148, 163, 184, 0.18);
      padding: 0.65rem 0.85rem;
      vertical-align: top;
    }
    article th { background: rgba(59, 130, 246, 0.10); font-weight: 600; color: rgba(224, 231, 255, 0.98); }
    article tbody tr:nth-child(even) { background: rgba(148, 163, 184, 0.04); }
    article tbody tr:hover { background: rgba(148, 163, 184, 0.08); }
    html[data-theme="light"] article th {
      background: rgba(37, 99, 235, 0.08);
      color: rgba(15, 23, 42, 0.92);
    }
    html[data-theme="light"] article td {
      border-color: rgba(15, 23, 42, 0.10);
    }
    html[data-theme="light"] article th {
      border-color: rgba(15, 23, 42, 0.12);
    }
    html[data-theme="light"] article tbody tr:nth-child(even) { background: rgba(15, 23, 42, 0.03); }
    html[data-theme="light"] article tbody tr:hover { background: rgba(15, 23, 42, 0.05); }
    article blockquote {
      border-left: 3px solid rgba(99, 102, 241, 0.55);
      padding: 0.6rem 0.9rem;
      margin: 1rem 0;
      background: rgba(99, 102, 241, 0.06);
      border-radius: 0 0.6rem 0.6rem 0;
      color: var(--muted);
    }
    html[data-theme="light"] article blockquote {
      border-left-color: rgba(99, 102, 241, 0.55);
      background: rgba(99, 102, 241, 0.06);
      color: var(--muted);
    }

    /* Back to Top button */
    .back-to-top {
      position: fixed;
      right: 18px;
      bottom: 64px;
      z-index: 9998;
      width: 2.5rem;
      height: 2.5rem;
      border-radius: 9999px;
      border: 1px solid rgba(226, 232, 240, 0.18);
      background: rgba(2, 6, 23, 0.75);
      color: var(--fg);
      font-size: 1.2rem;
      line-height: 1;
      cursor: pointer;
      display: none;
      align-items: center;
      justify-content: center;
      box-shadow: var(--shadow);
      transition: opacity 0.2s;
    }
    .back-to-top:hover { background: rgba(2, 6, 23, 0.88); }
    html[data-theme="light"] .back-to-top {
      background: rgba(255, 255, 255, 0.92);
      color: rgba(15, 23, 42, 0.92);
      border: 1px solid rgba(15, 23, 42, 0.14);
    }

    /* Lightbox */
    .lightbox-overlay {
      position: fixed;
      inset: 0;
      z-index: 10000;
      background: rgba(0, 0, 0, 0.85);
      display: none;
      align-items: center;
      justify-content: center;
      cursor: zoom-out;
    }
    .lightbox-overlay img {
      max-width: 92vw;
      max-height: 92vh;
      border-radius: 0.75rem;
      box-shadow: 0 20px 60px rgba(0,0,0,0.5);
    }
    html[data-theme="light"] .lightbox-overlay {
      background: rgba(255, 255, 255, 0.90);
    }
    article img {
      cursor: zoom-in;
    }

    /* TOC progress */
    .toc-progress {
      font-size: 0.7rem;
      color: var(--muted);
      padding: 0.2rem 0.5rem;
      margin-bottom: 0.25rem;
    }

    .topbar {
      background: var(--panel);
      border-bottom: 1px solid var(--border);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      overflow: visible;
    }

    /* Print styles */
    @media print {
      header.topbar, .theme-fab, #toast, #tocDrawer,
      aside, .copy-btn, .expand-btn, .lang-label,
      .back-to-top, .lightbox-overlay, #searchOverlay { display: none !important; }
      article { max-width: 100% !important; margin: 0 !important; padding: 0 !important; }
      article pre, .highlight pre {
        white-space: pre-wrap !important;
        word-break: break-word !important;
        border: 1px solid #ccc !important;
        background: #f6f8fa !important;
        color: #24292e !important;
      }
      .codewrap.collapsed {
        max-height: none !important;
        overflow: visible !important;
        mask-image: none !important;
        -webkit-mask-image: none !important;
      }
      article table { display: table !important; }
      a[href]::after { content: none !important; }
      .mermaid svg { max-width: 100% !important; }
    }
  </style>
</head>

<body class="doc-bg min-h-screen">
  <header class="topbar sticky top-0 z-50">
    <div class="mx-auto max-w-screen-xl px-4 sm:px-6 lg:px-8 py-3 flex items-center justify-between">
      <div class="flex items-center gap-3">
        <div class="brand-badge h-9 w-9 rounded-xl grid place-items-center font-bold">MD</div>
        <div class="leading-tight">
          <div class="doc-subtitle text-xs">Documentation</div>
          <div class="text-sm font-semibold">%%TITLE%%</div>
        </div>
      </div>
      <div class="flex items-center gap-3">
        <button id="btnToc" class="theme-btn sm:hidden inline-flex items-center gap-2 rounded-xl px-3 py-2 text-xs font-medium">TOC</button>
        <button id="btnSearch" class="theme-btn inline-flex items-center gap-2 rounded-xl px-3 py-2 text-xs font-medium">Search</button>
        <button id="btnAutoFold" class="theme-btn inline-flex items-center gap-2 rounded-xl px-3 py-2 text-xs font-medium">AutoFold</button>
        <button id="btnFold" class="theme-btn inline-flex items-center gap-2 rounded-xl px-3 py-2 text-xs font-medium">Fold</button>
      </div>
    </div>
  </header>

  <div id="searchOverlay" aria-hidden="true">
    <div class="glass search-bar">
      <input id="searchInput" class="search-input" type="search" placeholder="Search in document... (Esc to close)" autocomplete="off" />
      <span id="searchCount" class="search-meta">0 / 0</span>
      <button id="btnSearchPrev" class="theme-btn rounded-xl px-3 py-2 text-xs font-medium" type="button">Prev</button>
      <button id="btnSearchNext" class="theme-btn rounded-xl px-3 py-2 text-xs font-medium" type="button">Next</button>
      <button id="btnSearchClose" class="theme-btn rounded-xl px-3 py-2 text-xs font-medium" type="button">Close</button>
    </div>
  </div>

  <!-- Mobile TOC Drawer -->
  <div id="tocDrawer" class="drawer-hidden fixed inset-0 z-40">
    <div id="tocBackdrop" class="drawer-backdrop fixed inset-0"></div>
    <div class="fixed right-0 top-0 drawer-panel">
      <div class="p-4 border-b" style="border-color: var(--border);">
        <div class="flex items-center justify-between">
          <div>
            <div class="toc-title text-sm font-semibold">목차</div>
            <div class="toc-subtitle text-xs mt-0.5">Heading 기반 자동 생성</div>
          </div>
          <div class="flex items-center gap-2">
            <button id="btnAutoFoldMobile" class="theme-btn rounded-xl px-3 py-2 text-xs font-medium">AutoFold</button>
            <button id="btnTocClose" class="theme-btn rounded-xl px-3 py-2 text-xs font-medium">Close</button>
          </div>
        </div>
        <div class="mt-3">
          <input id="tocSearchMobile" type="search" placeholder="Search..." class="w-full rounded-xl border border-slate-200/10 bg-white/5 px-3 py-2 text-sm" />
        </div>
      </div>
      <nav id="tocMobile" class="h-[calc(100vh-6rem)] overflow-auto p-3 text-sm">
        %%TOC_HTML%%
      </nav>
    </div>
  </div>

  <div class="mx-auto max-w-screen-xl px-4 sm:px-6 lg:px-8 py-8">
    <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
      <aside class="hidden lg:block lg:col-span-3">
        <div class="sticky top-24">
          <div class="glass rounded-2xl">
            <div class="px-4 py-3 border-b border-slate-200/10">
              <div class="toc-title text-sm font-semibold">목차</div>
              <div class="toc-subtitle text-xs mt-0.5">Heading 기반 자동 생성</div>
            </div>
            <div class="px-3 pt-3">
              <input id="tocSearch" type="search" placeholder="Search..." class="w-full rounded-xl border border-slate-200/10 bg-white/5 px-3 py-2 text-sm" />
            </div>
            <nav id="toc" class="h-[calc(100vh-10rem)] overflow-auto p-3 text-sm">
              %%TOC_HTML%%
            </nav>
          </div>
        </div>
      </aside>

      <main class="lg:col-span-9">
        <article class="glass rounded-2xl px-6 sm:px-8 py-8">
          %%BODY_HTML%%
        </article>
      </main>
    </div>
  </div>

  <div id="toast" role="status" aria-live="polite"></div>
  <button id="btnBackToTop" type="button" class="back-to-top" aria-label="Back to top">&#8593;</button>
  <div id="lightbox" class="lightbox-overlay"><img id="lightboxImg" src="" alt="" /></div>

  <script>
    // Shared utilities (used by multiple script blocks)
    function _showToast(msg) {
      var el = document.getElementById('toast');
      if (!el) return;
      el.textContent = msg;
      el.style.display = 'block';
      clearTimeout(el._t);
      el._t = setTimeout(function () { el.style.display = 'none'; }, 1200);
    }
    function _getAutoFold() {
      try {
        var v = localStorage.getItem('toc_autofold');
        if (v === null) return true;
        return v === '1';
      } catch (e) { return true; }
    }
  </script>

  <button id="btnThemeFab" type="button" class="theme-fab">Theme</button>

  <script>
    (function () {
      var root = document.documentElement;
      var btnFab = document.getElementById('btnThemeFab');
      var btnAutoFold = document.getElementById('btnAutoFold');
      var btnAutoFoldMobile = document.getElementById('btnAutoFoldMobile');
      if (!btnFab) return;

      var getAutoFold = _getAutoFold;
      var showToast = _showToast;

      function setAutoFold(v) {
        try { localStorage.setItem('toc_autofold', v ? '1' : '0'); } catch (e) {}
        if (btnAutoFold) {
          btnAutoFold.textContent = v ? 'AutoFold: On' : 'AutoFold: Off';
        }
        if (btnAutoFoldMobile) {
          btnAutoFoldMobile.textContent = v ? 'AutoFold: On' : 'AutoFold: Off';
        }
      }

      function renderMermaid(mode) {
        try {
          if (!window.mermaid) return;

          var theme = (mode === 'light') ? 'default' : 'dark';
          window.mermaid.initialize({ startOnLoad: false, theme: theme });

          var nodes = document.querySelectorAll('.mermaid');
          for (var i = 0; i < nodes.length; i++) {
            var n = nodes[i];
            if (!n.dataset.src) n.dataset.src = n.textContent || '';
          }

          function toErrString(err) {
            try {
              if (!err) return 'unknown error';
              if (typeof err === 'string') return err;
              if (err.message) return String(err.message);
              return JSON.stringify(err);
            } catch (e) {
              return 'unknown error';
            }
          }

          function replaceWithError(node, src, errStr) {
            var wrap = document.createElement('div');
            wrap.className = 'mermaid-error';

            var title = document.createElement('div');
            title.className = 'mermaid-error-title';
            title.textContent = 'Mermaid parse failed';
            wrap.appendChild(title);

            var pre = document.createElement('pre');
            pre.className = 'mermaid-fallback';
            pre.textContent = src || '';
            wrap.appendChild(pre);

            var details = document.createElement('details');
            details.open = true;
            var summary = document.createElement('summary');
            summary.textContent = 'Show error';
            details.appendChild(summary);

            var msg = document.createElement('pre');
            msg.className = 'mermaid-error-msg';
            msg.textContent = errStr || '';
            details.appendChild(msg);
            wrap.appendChild(details);

            node.parentNode.replaceChild(wrap, node);

            try {
              console.error('[Mermaid parse failed]', errStr);
              console.error('[Mermaid source]', src);
            } catch (e) {}
          }

          // Render node-by-node so we can show per-diagram errors.
          (async function () {
            for (var k = 0; k < nodes.length; k++) {
              var el = nodes[k];
              var src = el.dataset.src || '';

              try {
                // Validate
                if (window.mermaid.parse) {
                  var parsed = window.mermaid.parse(src);
                  if (parsed && typeof parsed.then === 'function') {
                    await parsed;
                  }
                }

                // Render
                if (window.mermaid.render) {
                  var id = 'mmd-' + k + '-' + Date.now();
                  var out = window.mermaid.render(id, src);
                  if (out && typeof out.then === 'function') {
                    out = await out;
                  }

                  var svg = '';
                  if (typeof out === 'string') svg = out;
                  else if (out && out.svg) svg = out.svg;
                  else svg = '';

                  if (!svg) throw new Error('mermaid.render produced empty output');
                  el.innerHTML = svg;
                } else {
                  // Fallback to init API if render API is not available
                  el.textContent = src;
                  el.removeAttribute('data-processed');
                  window.mermaid.init(undefined, [el]);
                }
              } catch (err) {
                replaceWithError(el, src, toErrString(err));
              }
            }
          })();
        } catch (e2) {}
      }

      function apply(mode) {
        root.dataset.theme = mode;
        if (mode === 'light') {
          document.body.classList.remove('doc-bg');
        } else {
          document.body.classList.add('doc-bg');
        }
        if (btnFab) btnFab.textContent = (mode === 'light') ? 'Theme: Light' : 'Theme: Dark';
        renderMermaid(mode);
        setAutoFold(getAutoFold());
      }

      var saved = null;
      try { saved = localStorage.getItem('doc_theme'); } catch (e) {}
      var mode = saved;
      if (!mode) {
        try {
          mode = (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) ? 'light' : 'dark';
        } catch (e) {
          mode = 'dark';
        }
      }
      apply(mode);

      function toggleTheme() {
        mode = (mode === 'dark') ? 'light' : 'dark';
        try { localStorage.setItem('doc_theme', mode); } catch (e) {}
        apply(mode);
        showToast(mode === 'light' ? 'Light mode' : 'Dark mode');
      }

      if (btnFab) btnFab.addEventListener('click', toggleTheme);

      if (btnAutoFold) {
        setAutoFold(getAutoFold());
        btnAutoFold.addEventListener('click', function () {
          var next = !getAutoFold();
          setAutoFold(next);
          showToast(next ? 'AutoFold On' : 'AutoFold Off');
        });
      }

      if (btnAutoFoldMobile) {
        setAutoFold(getAutoFold());
        btnAutoFoldMobile.addEventListener('click', function () {
          var next = !getAutoFold();
          setAutoFold(next);
          showToast(next ? 'AutoFold On' : 'AutoFold Off');
        });
      }
    })();
  </script>

  <script>
    document.addEventListener('DOMContentLoaded', function () {
      var btnSearch = document.getElementById('btnSearch');
      var overlay = document.getElementById('searchOverlay');
      var input = document.getElementById('searchInput');
      var countEl = document.getElementById('searchCount');
      var btnPrev = document.getElementById('btnSearchPrev');
      var btnNext = document.getElementById('btnSearchNext');
      var btnClose = document.getElementById('btnSearchClose');
      var article = document.querySelector('article');
      if (!overlay || !input || !article) return;

      var marks = [];
      var hitMarks = [];
      var activeHit = -1;
      var isComposing = false;
      var applyTimer = null;

      function isEditableTarget(t) {
        if (!t) return false;
        var tag = (t.tagName || '').toLowerCase();
        if (tag === 'input' || tag === 'textarea' || tag === 'select') return true;
        if (t.isContentEditable) return true;
        return false;
      }

      function setOpen(v) {
        overlay.dataset.open = v ? '1' : '0';
        overlay.setAttribute('aria-hidden', v ? 'false' : 'true');
        if (v) {
          try { input.focus(); input.select(); } catch (e) {}
        }
      }

      function clearMarks() {
        try {
          for (var i = 0; i < marks.length; i++) {
            var m = marks[i];
            if (!m || !m.parentNode) continue;
            var txt = document.createTextNode(m.textContent || '');
            m.parentNode.replaceChild(txt, m);
          }
          marks = [];
          hitMarks = [];
          activeHit = -1;
          if (countEl) countEl.textContent = '0 / 0';
        } catch (e) {}
      }

      function normalizeQuery(q) {
        var s = String(q || '').trim();
        try {
          if (s && s.normalize) s = s.normalize('NFC');
        } catch (e) {}
        return s;
      }

      function collectTextNodes() {
        var walker = document.createTreeWalker(
          article,
          NodeFilter.SHOW_TEXT,
          {
            acceptNode: function (node) {
              try {
                if (!node || !node.parentNode) return NodeFilter.FILTER_REJECT;
                var s = String(node.nodeValue || '');
                if (!s || !s.trim()) return NodeFilter.FILTER_REJECT;
                var p = node.parentNode;
                var tag = (p.tagName || '').toLowerCase();
                if (tag === 'script' || tag === 'style') return NodeFilter.FILTER_REJECT;
                if (tag === 'code' || tag === 'pre') return NodeFilter.FILTER_REJECT;
                if (p.closest && p.closest('pre, code, .mermaid, .highlight')) return NodeFilter.FILTER_REJECT;
                return NodeFilter.FILTER_ACCEPT;
              } catch (e) {
                return NodeFilter.FILTER_REJECT;
              }
            }
          },
          false
        );
        var nodes = [];
        var n = null;
        while ((n = walker.nextNode())) nodes.push(n);
        return nodes;
      }

      function highlight(q) {
        clearMarks();
        q = normalizeQuery(q);
        if (!q) return;

        var qLower = q.toLowerCase();
        var nodes = collectTextNodes();
        if (!nodes || nodes.length === 0) return;

        // Search across the whole article text so matches spanning multiple text nodes
        // still work (common in HTML due to inline elements).
        var full = '';
        var meta = [];
        var off = 0;
        for (var i = 0; i < nodes.length; i++) {
          var t = String(nodes[i].nodeValue || '');
          meta.push({ node: nodes[i], start: off, len: t.length });
          full += t;
          off += t.length;
        }
        try { if (full && full.normalize) full = full.normalize('NFC'); } catch (e) {}
        var fullLower = full.toLowerCase();

        var hits = [];
        var pos = 0;
        while (true) {
          var at = fullLower.indexOf(qLower, pos);
          if (at < 0) break;
          hits.push({ start: at, end: at + q.length });
          pos = at + q.length;
        }
        if (hits.length === 0) {
          if (countEl) countEl.textContent = '0 / 0';
          return;
        }

        hitMarks = [];
        for (var z = 0; z < hits.length; z++) hitMarks.push([]);

        function wrapInNode(node, a, b, hitIndex) {
          try {
            // Split into [0..a)[a..b)[b..]
            var mid = node;
            if (a > 0) mid = node.splitText(a);
            var after = mid;
            if ((b - a) < mid.nodeValue.length) after = mid.splitText(b - a);
            var m = document.createElement('mark');
            m.className = 'search-mark';
            m.textContent = mid.nodeValue;
            mid.parentNode.replaceChild(m, mid);
            marks.push(m);
            try {
              if (typeof hitIndex === 'number' && hitIndex >= 0) {
                m.dataset.hitIndex = String(hitIndex);
                if (hitMarks[hitIndex]) hitMarks[hitIndex].push(m);
              }
            } catch (e2) {}
          } catch (e) {}
        }

        // Apply from end to start so Text.splitText offsets remain valid.
        for (var h = hits.length - 1; h >= 0; h--) {
          var s = hits[h].start;
          var e = hits[h].end;
          for (var j = meta.length - 1; j >= 0; j--) {
            var item = meta[j];
            var ns = item.start;
            var ne = item.start + item.len;
            if (e <= ns || s >= ne) continue;
            var localStart = Math.max(0, s - ns);
            var localEnd = Math.min(item.len, e - ns);
            if (localEnd > localStart) {
              wrapInNode(item.node, localStart, localEnd, h);
            }
          }
        }

        // Wrapping is applied in reverse order; sort marks back into document order
        // so navigation starts from the top.
        try {
          marks.sort(function (a, b) {
            if (a === b) return 0;
            var pos = a.compareDocumentPosition(b);
            if (pos & Node.DOCUMENT_POSITION_FOLLOWING) return -1;
            if (pos & Node.DOCUMENT_POSITION_PRECEDING) return 1;
            return 0;
          });
        } catch (e) {}

        // Sort segments inside each hit to keep active styling consistent.
        try {
          for (var hh = 0; hh < hitMarks.length; hh++) {
            hitMarks[hh].sort(function (a, b) {
              if (a === b) return 0;
              var pos = a.compareDocumentPosition(b);
              if (pos & Node.DOCUMENT_POSITION_FOLLOWING) return -1;
              if (pos & Node.DOCUMENT_POSITION_PRECEDING) return 1;
              return 0;
            });
          }
        } catch (e2) {}

        if (hits.length > 0) {
          setActive(0);
        } else {
          if (countEl) countEl.textContent = '0 / 0';
        }
      }

      function setActive(idx) {
        if (!hitMarks || hitMarks.length === 0) return;
        if (idx < 0) idx = hitMarks.length - 1;
        if (idx >= hitMarks.length) idx = 0;

        try {
          if (activeHit >= 0 && hitMarks[activeHit] && hitMarks[activeHit].length) {
            for (var i = 0; i < hitMarks[activeHit].length; i++) {
              hitMarks[activeHit][i].classList.remove('search-active');
            }
          }
        } catch (e) {}

        activeHit = idx;
        var arr = hitMarks[activeHit] || [];
        for (var j = 0; j < arr.length; j++) {
          try { arr[j].classList.add('search-active'); } catch (e2) {}
        }
        var el = arr.length ? arr[0] : null;
        if (el) {
          try { el.scrollIntoView({ block: 'center', behavior: 'smooth' }); } catch (e3) { try { el.scrollIntoView(true); } catch (e4) {} }
        }
        if (countEl) countEl.textContent = String(activeHit + 1) + ' / ' + String(hitMarks.length);
      }

      function next() { if (hitMarks.length) setActive(activeHit + 1); }
      function prev() { if (hitMarks.length) setActive(activeHit - 1); }

      function closeAndClear() {
        setOpen(false);
        input.value = '';
        clearMarks();
      }

      if (btnSearch) {
        btnSearch.addEventListener('click', function () {
          setOpen(true);
        });
      }
      if (btnClose) btnClose.addEventListener('click', closeAndClear);
      if (btnNext) btnNext.addEventListener('click', next);
      if (btnPrev) btnPrev.addEventListener('click', prev);

      function scheduleApply() {
        try { if (applyTimer) clearTimeout(applyTimer); } catch (e) {}
        applyTimer = setTimeout(function () {
          try {
            // Defer again to ensure input.value is updated (IME / composition timing).
            if (window.requestAnimationFrame) {
              window.requestAnimationFrame(function () {
                setTimeout(function () { highlight(input.value); }, 0);
              });
            } else {
              setTimeout(function () { highlight(input.value); }, 0);
            }
          } catch (e2) {
            highlight(input.value);
          }
        }, 0);
      }

      input.addEventListener('compositionstart', function () {
        isComposing = true;
        scheduleApply();
      });
      input.addEventListener('compositionupdate', function () {
        // Update while composing so multi-char Korean queries are searchable immediately.
        scheduleApply();
      });
      input.addEventListener('compositionend', function () {
        isComposing = false;
        scheduleApply();
      });

      input.addEventListener('beforeinput', function () {
        scheduleApply();
      });

      input.addEventListener('input', function (e) {
        // Do not block while composing: Chrome IME may keep isComposing true until commit.
        scheduleApply();
      });

      input.addEventListener('keyup', function (e) {
        if (!e) return;
        if (e.key === 'Enter' || e.key === 'Escape') return;
        scheduleApply();
      });

      input.addEventListener('keydown', function (e) {
        if (!e) return;
        if (e.key === 'Enter') {
          e.preventDefault();
          if (e.shiftKey) prev();
          else next();
        } else if (e.key === 'Escape') {
          e.preventDefault();
          closeAndClear();
        }
      });

      document.addEventListener('keydown', function (e) {
        if (!e) return;
        if (isEditableTarget(e.target) && e.target !== input) return;

        if ((e.ctrlKey || e.metaKey) && (e.key === 'k' || e.key === 'K')) {
          e.preventDefault();
          setOpen(true);
          return;
        }
        if (!e.ctrlKey && !e.metaKey && e.key === '/' && e.target !== input) {
          e.preventDefault();
          setOpen(true);
          return;
        }
        if (e.key === 'Escape' && overlay.dataset.open === '1') {
          e.preventDefault();
          closeAndClear();
          return;
        }
        if (overlay.dataset.open === '1') {
          if (e.key === 'F3') {
            e.preventDefault();
            if (e.shiftKey) prev();
            else next();
          }
        }
      });
    });
  </script>

  <script>
    document.addEventListener('DOMContentLoaded', function () {
      var btnToc = document.getElementById('btnToc');
      var drawer = document.getElementById('tocDrawer');
      var backdrop = document.getElementById('tocBackdrop');
      var btnClose = document.getElementById('btnTocClose');

      function openDrawer() {
        if (!drawer) return;
        drawer.classList.remove('drawer-hidden');
      }
      function closeDrawer() {
        if (!drawer) return;
        drawer.classList.add('drawer-hidden');
      }

      if (btnToc) btnToc.addEventListener('click', openDrawer);
      if (btnClose) btnClose.addEventListener('click', closeDrawer);
      if (backdrop) backdrop.addEventListener('click', closeDrawer);
      document.addEventListener('keydown', function (e) {
        if (e && e.key === 'Escape') closeDrawer();
      });
    });
  </script>

  <script>
    document.addEventListener('DOMContentLoaded', function () {
      var btnFold = document.getElementById('btnFold');

      function getFoldMinLinesDefault() {
        return %%COLLAPSE_MIN_LINES%%;
      }

      function getFoldMinLines() {
        try {
          var v = localStorage.getItem('doc_collapse_min_lines');
          if (v === null || v === undefined || v === '') return getFoldMinLinesDefault();
          var n = parseInt(v, 10);
          if (!isFinite(n) || n < 0) return getFoldMinLinesDefault();
          return n;
        } catch (e) {
          return getFoldMinLinesDefault();
        }
      }

      function setFoldMinLines(n) {
        try { localStorage.setItem('doc_collapse_min_lines', String(n)); } catch (e) {}
        if (btnFold) btnFold.textContent = (n <= 0) ? 'Fold: Off' : ('Fold: ' + n);
      }

      function isExpanded(codeId) {
        try { return localStorage.getItem('code_expanded_' + codeId) === '1'; } catch (e) { return false; }
      }

      function setExpanded(codeId, v) {
        try { localStorage.setItem('code_expanded_' + codeId, v ? '1' : '0'); } catch (e) {}
      }

      function getCodeText(pre) {
        var code = pre.querySelector('code');
        if (code) return code.textContent || '';
        return pre.textContent || '';
      }

      async function copyText(text) {
        if (navigator.clipboard && navigator.clipboard.writeText) {
          await navigator.clipboard.writeText(text);
          return;
        }
        // Fallback: execCommand
        var ta = document.createElement('textarea');
        ta.value = text;
        ta.setAttribute('readonly', '');
        ta.style.position = 'fixed';
        ta.style.top = '-1000px';
        ta.style.left = '-1000px';
        document.body.appendChild(ta);
        ta.select();
        try { document.execCommand('copy'); } finally { document.body.removeChild(ta); }
      }

      var showToast = _showToast;

      function addCodeUX(pre) {
        if (!pre) return;
        // Mermaid blocks are rendered separately
        if (pre.classList.contains('mermaid') || pre.classList.contains('mermaid-fallback')) return;
        if (pre.dataset && pre.dataset.copyBound === '1') return;
        pre.dataset.copyBound = '1';

        var block = pre;
        try {
          var hl = pre.closest ? pre.closest('.highlight') : null;
          if (hl) block = hl;
        } catch (e) {}

        var wrapper = block.parentElement;
        if (!wrapper || !wrapper.classList || !wrapper.classList.contains('codewrap')) {
          wrapper = document.createElement('div');
          wrapper.className = 'codewrap';
          block.parentNode.insertBefore(wrapper, block);
          wrapper.appendChild(block);
        }

        // Stable id per page render (used for persistence)
        var codeId = '';
        try {
          if (!wrapper.dataset.codeId) wrapper.dataset.codeId = String(addCodeUX._idx++);
          codeId = wrapper.dataset.codeId;
        } catch (e) {
          codeId = String(addCodeUX._idx++);
        }

        var btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'copy-btn';
        btn.textContent = 'Copy';
        btn.addEventListener('click', async function () {
          var text = getCodeText(pre);
          try {
            await copyText(text);
            showToast('Copied');
          } catch (e) {
            showToast('Copy failed');
          }
        });

        wrapper.appendChild(btn);

        // Language label
        try {
          var codeEl = pre.querySelector('code');
          if (codeEl) {
            var cls = codeEl.className || '';
            var m = cls.match(/language-(\S+)/);
            if (m && m[1]) {
              var lang = m[1].toLowerCase();
              if (lang !== 'text' && lang !== 'nohighlight') {
                var lbl = document.createElement('span');
                lbl.className = 'lang-label';
                lbl.textContent = lang;
                wrapper.appendChild(lbl);
              }
            }
          }
        } catch (e) {}

        // Collapse long code blocks
        function applyFold() {
          var minLines = getFoldMinLines();
          var text = getCodeText(pre);
          var lines = text.split(/\r?\n/);
          var count = lines.length;
          var shouldFold = (minLines > 0) && (count >= minLines);

          var exp = wrapper.querySelector('.expand-btn');
          if (!shouldFold) {
            wrapper.classList.remove('collapsed');
            if (exp && exp.parentNode) exp.parentNode.removeChild(exp);
            return;
          }

          if (!exp) {
            exp = document.createElement('button');
            exp.type = 'button';
            exp.className = 'expand-btn';
            exp.textContent = 'Expand';
            exp.addEventListener('click', function () {
              var isCollapsed = wrapper.classList.contains('collapsed');
              if (isCollapsed) {
                wrapper.classList.remove('collapsed');
                exp.textContent = 'Collapse';
                setExpanded(codeId, true);
              } else {
                wrapper.classList.add('collapsed');
                exp.textContent = 'Expand';
                setExpanded(codeId, false);
              }
            });
            wrapper.appendChild(exp);
          }

          // Restore persisted state
          if (isExpanded(codeId)) {
            wrapper.classList.remove('collapsed');
            exp.textContent = 'Collapse';
          } else {
            wrapper.classList.add('collapsed');
            exp.textContent = 'Expand';
          }
        }

        applyFold();
        wrapper._applyFold = applyFold;
      }

      addCodeUX._idx = 0;

      var pres = document.querySelectorAll('article pre');
      for (var i = 0; i < pres.length; i++) {
        addCodeUX(pres[i]);
      }

      function applyFoldAll() {
        try {
          var wraps = document.querySelectorAll('.codewrap');
          for (var i = 0; i < wraps.length; i++) {
            var w = wraps[i];
            if (w && w._applyFold) w._applyFold();
          }
        } catch (e) {}
      }

      // Fold button behavior
      if (btnFold) {
        var cur = getFoldMinLines();
        setFoldMinLines(cur);
        btnFold.addEventListener('click', function () {
          var steps = [0, 20, 35, 60];
          var v = getFoldMinLines();
          var idx = 0;
          for (var i = 0; i < steps.length; i++) {
            if (steps[i] === v) { idx = i; break; }
          }
          var next = steps[(idx + 1) % steps.length];
          setFoldMinLines(next);
          applyFoldAll();
        });
      }
    });
  </script>

  <script>
    document.addEventListener('DOMContentLoaded', function () {
      function enhanceToc(tocId, storageKey) {
        var toc = document.getElementById(tocId);
        if (!toc) return;

        var links = toc.querySelectorAll('a[href^="#"]');
        if (!links || links.length === 0) return;

        // Build sections based on heading level inferred from href targets.
        // python-markdown TOC renders nested <ul>, but link ordering is still linear.
        var items = [];
        for (var i = 0; i < links.length; i++) items.push(links[i]);

        // Determine level via closest LI nesting depth, fallback to text indent.
        function levelOf(a) {
          var li = a.closest('li');
          var lvl = 1;
          while (li) {
            var parentUl = li.parentElement;
            if (!parentUl || parentUl.tagName !== 'UL') break;
            var parentLi = parentUl.closest('li');
            if (!parentLi) break;
            lvl += 1;
            li = parentLi;
          }
          // toc_depth starts at 2-4, so treat lvl==1 as H2.
          return lvl;
        }

        var root = document.createElement('div');
        var currentSection = null;
        var sectionIdx = -1;

        // load collapsed state
        var collapsed = {};
        try {
          var raw = localStorage.getItem(storageKey);
          if (raw) collapsed = JSON.parse(raw) || {};
        } catch (e) {}

        function save() {
          try { localStorage.setItem(storageKey, JSON.stringify(collapsed)); } catch (e) {}
        }

        for (var k = 0; k < items.length; k++) {
          var a = items[k];
          var lvl = levelOf(a);

          if (lvl === 1) {
            sectionIdx += 1;
            currentSection = document.createElement('div');
            currentSection.className = 'toc-section';
            currentSection.dataset.sectionIndex = String(sectionIdx);

            var header = document.createElement('div');
            header.className = 'toc-section-header';

            var btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'toc-toggle';
            btn.textContent = (collapsed[sectionIdx] ? '▸' : '▾');

            var children = document.createElement('div');
            children.className = 'toc-children';

            if (collapsed[sectionIdx]) currentSection.classList.add('toc-collapsed');

            btn.addEventListener('click', function (ev) {
              var sec = ev.currentTarget._sec;
              var idx = Number(sec.dataset.sectionIndex);
              var isCollapsed = sec.classList.contains('toc-collapsed');
              if (isCollapsed) {
                sec.classList.remove('toc-collapsed');
                collapsed[idx] = false;
                ev.currentTarget.textContent = '▾';
              } else {
                sec.classList.add('toc-collapsed');
                collapsed[idx] = true;
                ev.currentTarget.textContent = '▸';
              }
              save();
            });
            btn._sec = currentSection;

            header.appendChild(btn);
            header.appendChild(a.cloneNode(true));
            currentSection.appendChild(header);
            currentSection.appendChild(children);
            root.appendChild(currentSection);
          } else {
            if (!currentSection) {
              // No H2 encountered yet; create a dummy section.
              currentSection = document.createElement('div');
              currentSection.className = 'toc-section';
              currentSection.dataset.sectionIndex = '0';
              var children0 = document.createElement('div');
              children0.className = 'toc-children';
              currentSection.appendChild(children0);
              root.appendChild(currentSection);
            }
            var ch = currentSection.querySelector('.toc-children');
            var itemWrap = document.createElement('div');
            // visually indent: lvl 2->H3, lvl3->H4
            itemWrap.style.marginLeft = (lvl === 2 ? '0.25rem' : '1.0rem');
            itemWrap.appendChild(a.cloneNode(true));
            ch.appendChild(itemWrap);
          }
        }

        // replace content
        toc.innerHTML = '';
        toc.appendChild(root);
      }

      enhanceToc('toc', 'toc_collapsed_desktop');
      enhanceToc('tocMobile', 'toc_collapsed_mobile');
    });
  </script>

  <script>
    document.addEventListener('DOMContentLoaded', function () {
      // Tag ASCII/box-drawing diagrams so we can apply a stricter font stack.
      (function () {
        try {
          var boxRe = /[┌┐└┘├┤┬┴┼│─═╔╗╚╝╠╣╦╩╬┃━]/;
          var pres = document.querySelectorAll('article pre, .highlight pre');
          for (var i = 0; i < pres.length; i++) {
            var pre = pres[i];
            var text = (pre.textContent || '');
            if (boxRe.test(text)) {
              pre.classList.add('ascii-diagram');
              var code = pre.querySelector('code');
              if (code) code.classList.add('ascii-diagram');
            }
          }
        } catch (e) {}
      })();

      function attachSearch(inputId, tocId) {
        var input = document.getElementById(inputId);
        var toc = document.getElementById(tocId);
        if (!input || !toc) return;

        var isSyncing = false;
        function syncSearchValue(v) {
          if (isSyncing) return;
          isSyncing = true;
          try {
            var otherId = (inputId === 'tocSearch') ? 'tocSearchMobile' : 'tocSearch';
            var other = document.getElementById(otherId);
            if (other && other.value !== v) {
              other.value = v;
            }
          } catch (e) {}
          isSyncing = false;
        }

        function escapeHtml(s) {
          return String(s)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
        }

        function escapeRegex(s) {
          return String(s).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        }

        function normalize(s) {
          return String(s || '').toLowerCase().trim();
        }

        function apply() {
          var q = normalize(input.value);
          syncSearchValue(input.value);
          var links = toc.querySelectorAll('a[href^="#"]');
          for (var i = 0; i < links.length; i++) {
            var a = links[i];
            if (!a.dataset.origText) {
              a.dataset.origText = a.textContent || '';
            }
            var orig = a.dataset.origText;
            var text = normalize(orig);
            var match = (!q || text.indexOf(q) >= 0);
            a.style.display = match ? '' : 'none';

            if (!q) {
              a.innerHTML = escapeHtml(orig);
            } else if (match) {
              var re = new RegExp('(' + escapeRegex(q) + ')', 'ig');
              a.innerHTML = escapeHtml(orig).replace(re, '<mark class="toc-mark">$1</mark>');
            }
          }

          // When searching, expand all sections so matches are visible.
          try {
            var sections = toc.querySelectorAll('.toc-section');
            for (var j = 0; j < sections.length; j++) {
              if (q) sections[j].classList.remove('toc-collapsed');
            }
          } catch (e) {}
        }

        input.addEventListener('input', apply);
        apply();
      }

      attachSearch('tocSearch', 'toc');
      attachSearch('tocSearchMobile', 'tocMobile');
    });
  </script>

  <script>
    document.addEventListener('DOMContentLoaded', function () {
      var toc = document.getElementById('toc');
      var tocMobile = document.getElementById('tocMobile');
      if (!toc && !tocMobile) return;

      var getAutoFold = _getAutoFold;

      function collectLinks(container) {
        if (!container) return [];
        var list = container.querySelectorAll('a[href^="#"]');
        return list ? Array.prototype.slice.call(list) : [];
      }

      var tocLinks = collectLinks(toc);
      var tocLinksMobile = collectLinks(tocMobile);
      var allLinks = tocLinks.concat(tocLinksMobile);
      if (!allLinks || allLinks.length === 0) return;

      var linkById = {};
      for (var i = 0; i < allLinks.length; i++) {
        var href = allLinks[i].getAttribute('href') || '';
        if (href.charAt(0) === '#') {
          var id = href.slice(1);
          if (!linkById[id]) linkById[id] = [];
          linkById[id].push(allLinks[i]);
        }
      }

      var headings = document.querySelectorAll('article h1, article h2, article h3, article h4');
      if (!headings || headings.length === 0) return;

      function setActive(id) {
        for (var j = 0; j < allLinks.length; j++) {
          allLinks[j].classList.remove('toc-active');
        }
        var arr = linkById[id];
        if (arr && arr.length) {
          for (var x = 0; x < arr.length; x++) {
            arr[x].classList.add('toc-active');
            try {
              var sec = arr[x].closest('.toc-section');
              if (sec) sec.classList.remove('toc-collapsed');
            } catch (e) {}
          }

          // Auto-collapse non-active sections (only if not searching)
          try {
            var q1 = '';
            var q2 = '';
            var s1 = document.getElementById('tocSearch');
            var s2 = document.getElementById('tocSearchMobile');
            if (s1) q1 = String(s1.value || '').trim();
            if (s2) q2 = String(s2.value || '').trim();
            var isSearching = (q1.length > 0) || (q2.length > 0);

            if (getAutoFold() && !isSearching) {
              function collapseOthers(container, activeLink) {
                if (!container || !activeLink) return;
                var activeSec = activeLink.closest('.toc-section');
                var secs = container.querySelectorAll('.toc-section');
                for (var i = 0; i < secs.length; i++) {
                  if (secs[i] === activeSec) secs[i].classList.remove('toc-collapsed');
                  else secs[i].classList.add('toc-collapsed');
                }
              }

              collapseOthers(toc, arr[0]);
              if (tocMobile && arr.length > 1) {
                collapseOthers(tocMobile, arr[arr.length - 1]);
              } else {
                collapseOthers(tocMobile, arr[0]);
              }
            }
          } catch (e) {}

          try {
            var a = arr[0];
            var container = toc || tocMobile;
            if (container) {
              var top = a.offsetTop - container.clientHeight / 2;
              if (top < 0) top = 0;
              container.scrollTop = top;
            }
          } catch (e) {}
        }
      }

      var currentId = '';

      if (typeof IntersectionObserver !== 'undefined') {
        var observer = new IntersectionObserver(function (entries) {
          var best = null;
          for (var k = 0; k < entries.length; k++) {
            var ent = entries[k];
            if (ent.isIntersecting) {
              if (!best || ent.intersectionRatio > best.intersectionRatio) {
                best = ent;
              }
            }
          }
          if (best && best.target && best.target.id && best.target.id !== currentId) {
            currentId = best.target.id;
            setActive(currentId);
          }
        }, { rootMargin: '-20% 0px -70% 0px', threshold: [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1] });

        for (var h = 0; h < headings.length; h++) {
          if (headings[h].id) observer.observe(headings[h]);
        }
      } else {
        function onScroll() {
          var bestId = '';
          var bestTop = -Infinity;
          for (var h2 = 0; h2 < headings.length; h2++) {
            var el = headings[h2];
            if (!el.id) continue;
            var rect = el.getBoundingClientRect();
            if (rect.top <= 120 && rect.top > bestTop) {
              bestTop = rect.top;
              bestId = el.id;
            }
          }
          if (bestId && bestId !== currentId) {
            currentId = bestId;
            setActive(currentId);
          }
        }
        window.addEventListener('scroll', onScroll, { passive: true });
        onScroll();
      }

      if (location.hash && location.hash.length > 1) {
        var initId = location.hash.slice(1);
        if (linkById[initId]) {
          currentId = initId;
          setActive(currentId);
        }
      }
    });
  </script>

  <script>
    document.addEventListener('DOMContentLoaded', function () {
      // A-1: Back to Top button
      var btnTop = document.getElementById('btnBackToTop');
      if (btnTop) {
        window.addEventListener('scroll', function () {
          if (window.scrollY > 200) {
            btnTop.style.display = 'flex';
          } else {
            btnTop.style.display = 'none';
          }
        }, { passive: true });
        btnTop.addEventListener('click', function () {
          window.scrollTo({ top: 0, behavior: 'smooth' });
        });
      }

      // A-2: Lightbox for images
      var lightbox = document.getElementById('lightbox');
      var lightboxImg = document.getElementById('lightboxImg');
      if (lightbox && lightboxImg) {
        document.querySelectorAll('article img').forEach(function (img) {
          img.addEventListener('click', function () {
            lightboxImg.src = img.src;
            lightboxImg.alt = img.alt || '';
            lightbox.style.display = 'flex';
          });
        });
        lightbox.addEventListener('click', function () {
          lightbox.style.display = 'none';
          lightboxImg.src = '';
        });
        document.addEventListener('keydown', function (e) {
          if (e.key === 'Escape' && lightbox.style.display === 'flex') {
            lightbox.style.display = 'none';
            lightboxImg.src = '';
          }
        });
      }

      // A-3: TOC progress indicator
      (function () {
        var toc = document.getElementById('toc');
        var tocMobile = document.getElementById('tocMobile');
        if (!toc && !tocMobile) return;

        var headings = document.querySelectorAll('article h2, article h3, article h4');
        var total = headings.length;
        if (total === 0) return;

        function addProgress(container) {
          if (!container) return null;
          var el = document.createElement('div');
          el.className = 'toc-progress';
          el.textContent = '0 / ' + total;
          container.parentNode.insertBefore(el, container);
          return el;
        }

        var progEl = addProgress(toc);
        var progElMobile = addProgress(tocMobile);

        function updateProgress() {
          var idx = 0;
          for (var i = 0; i < headings.length; i++) {
            var rect = headings[i].getBoundingClientRect();
            if (rect.top <= 150) idx = i + 1;
          }
          var text = idx + ' / ' + total;
          if (progEl) progEl.textContent = text;
          if (progElMobile) progElMobile.textContent = text;
        }

        window.addEventListener('scroll', updateProgress, { passive: true });
        updateProgress();
      })();

      // A-4: Code block double-click to select all
      document.querySelectorAll('article pre, .highlight pre').forEach(function (pre) {
        pre.addEventListener('dblclick', function (e) {
          // Don't interfere with button clicks
          if (e.target.closest('button')) return;
          try {
            var sel = window.getSelection();
            var range = document.createRange();
            var code = pre.querySelector('code') || pre;
            range.selectNodeContents(code);
            sel.removeAllRanges();
            sel.addRange(range);
          } catch (err) {}
        });
      });
    });
  </script>
</body>
</html>
"""


def _auto_fence_ascii_diagrams(md_text: str) -> str:
    box_chars = set("┌┐└┘├┤┬┴┼│─═╔╗╚╝╠╣╦╩╬┃━▲▼◀▶")
    lines = md_text.splitlines()
    out: list[str] = []

    def is_diagram_line(s: str) -> bool:
        if not s.strip():
            return False
        hit = any(ch in box_chars for ch in s)
        if not hit:
            return False
        # Avoid wrapping mermaid fenced blocks or already fenced code
        return True

    i = 0
    in_fence = False
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()

        if stripped.startswith("```"):
            # NOTE: Fence detection is intentionally simple.
            # - Only supports triple-backtick fences (not ~~~)
            # - Does not attempt to handle nested/imbalanced fences inside fenced blocks
            in_fence = not in_fence
            out.append(line)
            i += 1
            continue

        if in_fence:
            out.append(line)
            i += 1
            continue

        if is_diagram_line(line):
            start = i
            j = i
            while j < len(lines):
                if is_diagram_line(lines[j]) or lines[j].strip() == "":
                    j += 1
                    continue
                break

            block = lines[start:j]
            while block and block[0].strip() == "":
                block.pop(0)
            while block and block[-1].strip() == "":
                block.pop()

            # Require at least 2 non-empty lines with box chars to avoid
            # false positives on single occurrences like "│" in prose.
            diagram_line_count = sum(1 for b in block if any(ch in box_chars for ch in b))
            if diagram_line_count >= 2:
                out.append("```text")
                out.extend(block)
                out.append("```")
            else:
                out.extend(block)
            i = j
            continue

        out.append(line)
        i += 1

    return "\n".join(out) + ("\n" if md_text.endswith("\n") else "")


def _inline_mermaid_fences(md_text: str) -> str:
    """Convert ```mermaid fences into raw HTML blocks before markdown conversion.

    This prevents codehilite/pygments from tokenizing mermaid source (which can
    destroy whitespace and make it impossible to reconstitute the original diagram).
    """

    # Keep it simple: only support triple-backtick fences.
    lines = md_text.splitlines()
    out: list[str] = []
    i = 0
    in_mermaid = False
    buf: list[str] = []

    # Replace unsafe label text inside node definitions to avoid mermaid grammar conflicts.
    # Examples we need to handle (from mermaid.md):
    #   EL[ebest_live.py\nrun_...\n(orchestrator)]
    #   tee[tee (--tee/--no-tee)]
    #   B[Unpack tick_norm depth -> raw keys (best-effort)]
    # Match at token boundaries only (avoid rewriting inside already-quoted labels).
    # FIX: 레이블이 이미 "..." 또는 '...' 로 감싸인 경우(예: ID["label"])를
    # 다시 처리하지 않도록 lookahead로 제외한다.
    # 패턴: ID[ 다음이 " 또는 ' 로 시작하면 이미 따옴표 처리 완료 → 스킵.
    node_pat_sq  = re.compile(r'(?P<prefix>^|[^A-Za-z0-9_"\'])'
                               r'(?P<id>[A-Za-z_][A-Za-z0-9_]*)'
                               r'\[(?!["\'])(?P<label>[^\]]*?)\]')
    node_pat_par = re.compile(r'(?P<prefix>^|[^A-Za-z0-9_"\'])'
                               r'(?P<id>[A-Za-z_][A-Za-z0-9_]*)'
                               r'\((?!["\'])(?P<label>[^\)]*?)\)')
    node_pat_cur = re.compile(r'(?P<prefix>^|[^A-Za-z0-9_"\'])'
                               r'(?P<id>[A-Za-z_][A-Za-z0-9_]*)'
                               r'\{(?!["\'])(?P<label>[^\}]*?)\}')

    def _sanitize_label(label: str) -> str:
        s = label
        # \n → <br/>
        s = s.replace("\\n", "<br/>")
        s = s.replace("->", "→")
        # " → ' (레이블이 ["label"] 큰따옴표로 감싸이므로 내부 " 는 ' 로)
        s = s.replace('"', "'")
        # -- → — (mermaid 링크 구문과 충돌 방지)
        s = s.replace("--", "—")
        # [ ] → 유니코드 전각 대괄호 ［ ］ (U+FF3B, U+FF3D)
        # &#91;/&#93; HTML 엔티티는 safe_src.replace("&","&amp;") 에 의해
        # &amp;#91; 로 이중인코딩되어 브라우저가 &#91; 리터럴로 표시하므로 사용 금지.
        # 유니코드 전각 대괄호는 인코딩 문제 없이 mermaid 레이블에 안전하게 표시됨.
        s = s.replace("]", "］")
        s = s.replace("[", "［")
        return s

    def sanitize_mermaid_line(line: str) -> str:
        # Mermaid does not reliably support "\\n" escapes inside labels; use <br/>.
        line = line.replace("\\n", "<br/>")

        # Sanitize edge labels written as |label| (flowchart links)
        # FIX: |"label"| 형태(이미 큰따옴표로 감싸진 엣지 레이블)는 그대로 유지.
        # |label| 형태(따옴표 없는 것)만 _sanitize_label 처리.
        # 이유: _sanitize_label의 " → ' 변환이 |"label"| → |'label'| 로 만들어
        # mermaid 파서가 ' 를 구분자로 오해하여 파싱 오류 발생.
        def _edge_label_repl(m: re.Match) -> str:
            inner = m.group(1)
            # 이미 큰따옴표로 감싸진 경우: |"label"| → 그대로 유지
            if inner.startswith('"') and inner.endswith('"'):
                return f"|{inner}|"
            # 따옴표 없는 경우만 sanitize
            return "|" + _sanitize_label(inner) + "|"

        line = re.sub(r"\|([^|]+)\|", _edge_label_repl, line)

        # quadrantChart 전용: title/x-axis/y-axis/quadrant-N 라인 처리.
        # ★ diagram_type 체크 필수 — xychart-beta 등 다른 다이어그램의
        #   x-axis/y-axis 문법은 완전히 달라 이 처리를 적용하면 파싱 오류 발생.
        _s_stripped = line.lstrip()
        _indent_qc  = line[: len(line) - len(_s_stripped)]
        _cur_diagram_type = getattr(sanitize_mermaid_line, '_diagram_type', '')

        if _cur_diagram_type == 'quadrantChart':
            # title: 그대로 유지 (lexer가 [^\n]* 로 읽어 한글 OK)
            if _s_stripped.startswith('title '):
                return line

            # x-axis / y-axis: 각 라벨을 따옴표로 감싸기
            # quadrantChart x-axis 문법: X_AXIS STR --> STR | X_AXIS STR
            for _kw in ('x-axis', 'y-axis'):
                if _s_stripped.startswith(_kw):
                    _rest = _s_stripped[len(_kw):].strip()
                    if '-->' in _rest:
                        _left, _right = _rest.split('-->', 1)
                        _left  = _left.strip().strip('"').strip("'")
                        _right = _right.strip().strip('"').strip("'")
                        return f'{_indent_qc}{_kw} "{_left}" --> "{_right}"'
                    else:
                        _label = _rest.strip().strip('"').strip("'")
                        return f'{_indent_qc}{_kw} "{_label}"' if _label else line

            # quadrant-N: 라벨을 따옴표로 감싸기
            _qn_m = re.match(r'(quadrant-[1-4])\s+(.*)', _s_stripped)
            if _qn_m:
                _qn   = _qn_m.group(1)
                _qlbl = _qn_m.group(2).strip().strip('"').strip("'")
                return f'{_indent_qc}{_qn} "{_qlbl}"'

        # Special-case subgraph headers: keep syntax but quote the label if it uses bracket label forms.
        s = line.lstrip()
        if s.startswith("subgraph "):
            # 이미 큰따옴표로 감싸진 레이블 subgraph ID["label"] 은 그대로 유지 (재처리 금지)
            if re.match(r'^\s*subgraph\s+[A-Za-z_][A-Za-z0-9_]*\["', line):
                return line
            # 따옴표 없는 레이블을 큰따옴표로 감싸기
            m_sq = re.match(r"^(\s*)subgraph\s+([A-Za-z_][A-Za-z0-9_]*)\[([^\"'].*)\]\s*$", line)
            if m_sq:
                indent, sid, label = m_sq.groups()
                return f'{indent}subgraph {sid}["{_sanitize_label(label)}"]'
            m_par = re.match(r"^(\s*)subgraph\s+([A-Za-z_][A-Za-z0-9_]*)\(([^\"'].*)\)\s*$", line)
            if m_par:
                indent, sid, label = m_par.groups()
                return f'{indent}subgraph {sid}["{_sanitize_label(label)}"]'
            m_cur = re.match(r"^(\s*)subgraph\s+([A-Za-z_][A-Za-z0-9_]*)\{([^\"'].*)\}\s*$", line)
            if m_cur:
                indent, sid, label = m_cur.groups()
                return f'{indent}subgraph {sid}["{_sanitize_label(label)}"]'
            return line

        def repl_sq(m: re.Match) -> str:
            prefix = m.group('prefix')
            node_id = m.group('id')
            label = _sanitize_label(m.group('label'))
            return f"{prefix}{node_id}[\"{label}\"]"

        def repl_par(m: re.Match) -> str:
            prefix = m.group('prefix')
            node_id = m.group('id')
            label = _sanitize_label(m.group('label'))
            return f"{prefix}{node_id}(\"{label}\")"

        def repl_cur(m: re.Match) -> str:
            prefix = m.group('prefix')
            node_id = m.group('id')
            label = _sanitize_label(m.group('label'))
            return f"{prefix}{node_id}{{\"{label}\"}}"

        # sequenceDiagram / xychart 라인은 node_pat 적용 금지.
        # sequenceDiagram 메시지(E->>D: text)에 node_pat_par 가 적용되면
        # enc_out(60) → enc_out("60") 처럼 함수 호출 표현이 노드 레이블로 오변환됨.
        # xychart 는 x-axis/y-axis 배열 구문이 node_pat 과 충돌 가능.
        skip_node_pat = getattr(sanitize_mermaid_line, '_diagram_type', '') in (
            'sequenceDiagram', 'xychart-beta', 'quadrantChart'
        )
        if not skip_node_pat:
            # ★ BUG FIX: node_pat_par/cur 는 ["..."] 내부까지 재처리한다.
            # 예: G["분류 헤드 Linear(64→32)"] → Linear(64→32) 를 별개 노드로 오인해
            #     G["분류 헤드 Linear('64→32')"] 로 변환 → Mermaid parse failed.
            #
            # 해결: 이미 ["..."] 로 감싸진 구간을 플레이스홀더로 보호 후 node_pat 적용,
            #       이후 원래 텍스트로 복원한다.
            _ph_map: dict[str, str] = {}
            _ph_counter = [0]

            def _protect_quoted_label(m: re.Match) -> str:
                key = f"\x00PH{_ph_counter[0]}\x00"
                _ph_map[key] = m.group(0)
                _ph_counter[0] += 1
                return key

            # ["..."] 구간 보호 (내부에 "] 가 없는 단순 구간만 대상)
            protected = re.sub(r'\["[^"]*?"\]', _protect_quoted_label, line)

            protected = node_pat_sq.sub(repl_sq, protected)
            protected = node_pat_par.sub(repl_par, protected)
            protected = node_pat_cur.sub(repl_cur, protected)

            # 플레이스홀더 복원
            for key, val in _ph_map.items():
                protected = protected.replace(key, val)

            line = protected
        return line

    while i < len(lines):
        line = lines[i]
        if not in_mermaid:
            # Accept ```mermaid with trailing spaces and any casing.
            if re.match(r"^```mermaid\s*$", line.strip(), flags=re.IGNORECASE):
                in_mermaid = True
                buf = []
                raw_buf = []   # 원본 라인 보존 (fallback용)
                sanitize_mermaid_line._diagram_type = ''  # 블록 시작 시 초기화
                i += 1
                continue
            out.append(line)
            i += 1
            continue

        # in mermaid fence
        if line.strip() == "```":
            _diagram_type = getattr(sanitize_mermaid_line, '_diagram_type', '')
            _is_quadrant  = (_diagram_type == 'quadrantChart')

            # sanitize_mermaid_line 이 '' 를 반환한 라인 제거
            src = "\n".join(ln for ln in buf if ln != '')
            safe_src = src.replace("&", "&amp;")
            # 후처리: 이미 ["label"] 로 감싸인 노드 레이블 내부의 " 를 ' 로 교체.
            # node_pat_sq lookahead 가 건너뛴 케이스 대응.
            # xychart x-axis/y-axis 배열 형태 ["a","b","c"] 는 제외.
            # 탐욕적 매칭(.*?)으로 ["...최초..."] 전체를 캡처한다.
            def _fix_inner_quotes(line: str) -> str:
                stripped = line.lstrip()
                # quadrantChart 의 x-axis/y-axis/quadrant-N/title 라인:
                # sanitize_mermaid_line 에서 따옴표 배치 완료 → 재처리 금지.
                if _is_quadrant and (
                        stripped.startswith('x-axis') or
                        stripped.startswith('y-axis') or
                        stripped.startswith('title ') or
                        re.match(r'quadrant-[1-4]\b', stripped)):
                    return line
                # xychart x-axis/y-axis: ["a","b","c"] 배열 구문 보호.
                # _fix_inner_quotes 가 배열 내부 " → ' 로 바꾸면 파싱 실패.
                if stripped.startswith('x-axis') or stripped.startswith('y-axis'):
                    return line
                # ["..."] 패턴을 탐욕적으로 찾아 내부 " → '
                # 단, --> |"label"| 엣지 레이블의 경우도 보호 필요
                result = []
                pos = 0
                while pos < len(line):
                    # [" 시작 탐색
                    open_idx = line.find('["', pos)
                    if open_idx == -1:
                        result.append(line[pos:])
                        break
                    result.append(line[pos:open_idx + 2])  # [" 포함
                    inner_start = open_idx + 2
                    # "] 종료 탐색 — 가장 마지막 "] 사용 (탐욕적)
                    close_idx = line.find('"]', inner_start)
                    if close_idx == -1:
                        result.append(line[inner_start:])
                        break
                    inner = line[inner_start:close_idx]
                    # 내부 " → ' (단, HTML 엔티티 &quot; 는 보호)
                    inner = inner.replace('"', "'")
                    result.append(inner + '"]')
                    pos = close_idx + 2
                return ''.join(result)

            safe_src = '\n'.join(_fix_inner_quotes(fl) for fl in safe_src.split('\n'))
            out.append(f"<div class=\"mermaid\">{safe_src}</div>")
            in_mermaid = False
            buf = []
            i += 1
            continue

        # 첫 번째 라인(다이어그램 타입 선언)을 감지해 node_pat 스킵 여부 결정.
        # e.g. "sequenceDiagram", "xychart-beta", "quadrantChart" 등.
        stripped_line = line.strip()
        if not buf:  # 블록의 첫 번째 라인
            diagram_type = stripped_line.split()[0] if stripped_line else ''
            sanitize_mermaid_line._diagram_type = diagram_type
        raw_buf.append(line)
        buf.append(sanitize_mermaid_line(line))
        i += 1

    # Unterminated fence: fall back to original text
    if in_mermaid:
        out.append("```mermaid")
        out.extend(buf)

    return "\n".join(out) + ("\n" if md_text.endswith("\n") else "")


def _normalize_diagram_codeblocks(md_text: str) -> str:
    """Normalize whitespace/unicode inside box-drawing diagram code blocks.

    This targets cases where the MD itself looks misaligned due to tabs,
    non-breaking spaces, or full-width spaces inside the diagram.
    """

    box_chars = set("┌┐└┘├┤┬┴┼│─═╔╗╚╝╠╣╦╩╬┃━▲▼◀▶")
    lines = md_text.splitlines()
    out: list[str] = []

    i = 0
    in_fence = False
    fence_lang = ""
    fence_has_box = False

    def normalize_line(s: str) -> str:
        s = unicodedata.normalize("NFKC", s)
        s = s.replace("\t", "    ")
        s = s.replace("\u00a0", " ")
        s = s.replace("\u3000", "  ")
        return s

    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()

        if stripped.startswith("```"):
            if not in_fence:
                in_fence = True
                fence_lang = stripped[3:].strip().lower()
                fence_has_box = False
                out.append(line)
            else:
                in_fence = False
                fence_lang = ""
                fence_has_box = False
                out.append(line)
            i += 1
            continue

        if in_fence:
            if any(ch in box_chars for ch in line):
                fence_has_box = True
            # Only normalize if this looks like a diagram fence.
            if fence_lang in ("", "text") or fence_has_box:
                out.append(normalize_line(line))
            else:
                out.append(line)
            i += 1
            continue

        out.append(line)
        i += 1

    return "\n".join(out) + ("\n" if md_text.endswith("\n") else "")


def _tag_fenced_diagram_blocks_as_text(md_text: str) -> str:
    """If a fenced code block has no language and contains box-drawing chars,
    tag it as ```text to avoid unwanted syntax highlighting and font fallback."""

    box_chars = set("┌┐└┘├┤┬┴┼│─═╔╗╚╝╠╣╦╩╬┃━▲▼◀▶")
    lines = md_text.splitlines()
    out: list[str] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()

        if stripped.startswith("```"):
            fence = stripped
            lang = fence[3:].strip()
            if lang:
                out.append(line)
                i += 1
                continue

            # No language specified: look ahead until closing fence
            j = i + 1
            has_box = False
            while j < len(lines):
                if lines[j].lstrip().startswith("```"):
                    break
                if any(ch in box_chars for ch in lines[j]):
                    has_box = True
                j += 1

            if has_box:
                # Preserve original indentation
                prefix = line[: len(line) - len(stripped)]
                out.append(prefix + "```text")
            else:
                out.append(line)

            i += 1
            continue

        out.append(line)
        i += 1

    return "\n".join(out) + ("\n" if md_text.endswith("\n") else "")


def markdown_to_tailwind_html(md_text: str, title: str = "Document", config: RenderConfig | None = None) -> str:
    config = config or RenderConfig()
    md_text = _auto_fence_ascii_diagrams(md_text)
    md_text = _tag_fenced_diagram_blocks_as_text(md_text)
    md_text = _normalize_diagram_codeblocks(md_text)
    md_text = _inline_mermaid_fences(md_text)
    md = markdown.Markdown(
        extensions=[
            "fenced_code",
            "tables",
            "toc",
            "codehilite",
            "admonition",
            "attr_list",
            "md_in_html",
        ],
        extension_configs={
            "codehilite": {
                "css_class": "highlight",
                "linenums": False,
            },
            "toc": {
                "permalink": True,
                "toc_depth": "2-4",
            },
        },
    )

    html_body = md.convert(md_text)
    toc_html = getattr(md, "toc", "") or ""

    # Convert ```mermaid blocks rendered by markdown/codehilite into <div class="mermaid">...</div>
    # Typical output: <pre><code class="language-mermaid">...</code></pre>
    def _mermaid_repl(m: re.Match) -> str:
        inner = m.group(1)
        src = html.unescape(inner)
        return f"<div class=\"mermaid\">{src}</div>"

    html_body = re.sub(
        r"<pre><code class=\"language-mermaid\">([\s\S]*?)</code></pre>",
        _mermaid_repl,
        html_body,
        flags=re.IGNORECASE,
    )

    doc_html = HTML_TEMPLATE
    doc_html = doc_html.replace("%%TITLE%%", title)
    doc_html = doc_html.replace("%%TOC_HTML%%", toc_html)
    doc_html = doc_html.replace("%%BODY_HTML%%", html_body)
    doc_html = doc_html.replace("%%COLLAPSE_MIN_LINES%%", str(int(config.collapse_codeblock_min_lines)))
    doc_html = doc_html.replace("%%MERMAID_SANITIZE_MODE%%", str(config.mermaid_sanitize_mode))

    if bool(config.embed_assets):
        doc_html = embed_assets_into_html(doc_html, Path(config.assets_dir))
    return doc_html


def run_gui() -> int:
    try:
        from PySide6.QtWidgets import (
            QApplication,
            QWidget,
            QVBoxLayout,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QPushButton,
            QFileDialog,
            QMessageBox,
            QCheckBox,
            QSpinBox,
            QComboBox,
        )
        from PySide6.QtCore import QSettings, QUrl
        from PySide6.QtGui import QDesktopServices
    except Exception as e:
        raise SystemExit(
            "PySide6 is not available. Install it (pip install PySide6) or run without --gui.\n"
            f"Details: {e}"
        )

    app = QApplication.instance() or QApplication(sys.argv)

    class _DropWidget(QWidget):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.setAcceptDrops(True)
            self._on_drop = None

        def dragEnterEvent(self, event):
            if event.mimeData().hasUrls():
                event.acceptProposedAction()

        def dropEvent(self, event):
            urls = event.mimeData().urls()
            if urls and self._on_drop:
                path = urls[0].toLocalFile()
                if path:
                    self._on_drop(path)

    w = _DropWidget()
    w.setWindowTitle("MD \u2192 HTML Converter")
    w.setMinimumWidth(620)

    settings = QSettings("Transformer", "MD_to_HTML")

    root = QVBoxLayout(w)
    root.setContentsMargins(12, 12, 12, 12)
    root.setSpacing(10)

    def row(label_text: str):
        box = QHBoxLayout()
        box.setSpacing(8)
        lab = QLabel(label_text)
        lab.setMinimumWidth(120)
        box.addWidget(lab)
        return box

    in_edit = QLineEdit("")
    in_edit.setPlaceholderText("*.md")
    out_edit = QLineEdit("")
    out_edit.setPlaceholderText("*.html")
    title_edit = QLineEdit("")

    r1 = row("Input (.md)")
    r1.addWidget(in_edit)
    btn_in = QPushButton("Browse")
    r1.addWidget(btn_in)
    root.addLayout(r1)

    r2 = row("Output (.html)")
    r2.addWidget(out_edit)
    btn_out = QPushButton("Browse")
    r2.addWidget(btn_out)
    root.addLayout(r2)

    r3 = row("Title")
    r3.addWidget(title_edit)
    root.addLayout(r3)

    spin_collapse = QSpinBox()
    spin_collapse.setRange(0, 9999)
    spin_collapse.setValue(COLLAPSE_CODEBLOCK_MIN_LINES)
    spin_collapse.setSingleStep(5)
    spin_collapse.setToolTip("0 = Off")

    combo_sanitize = QComboBox()
    combo_sanitize.addItems(["auto", "on", "off"])
    combo_sanitize.setCurrentText(MERMAID_SANITIZE_MODE)

    r4 = row("Options")

    chk_embed = QCheckBox("Embed assets")
    chk_embed.setChecked(False)
    assets_edit = QLineEdit(str(ASSETS_DIR))
    assets_edit.setPlaceholderText("assets")
    assets_edit.setMinimumWidth(140)

    r4.addSpacing(10)
    r4.addWidget(QLabel("Fold min lines"))
    r4.addWidget(spin_collapse)

    r4.addSpacing(10)
    r4.addWidget(QLabel("Mermaid sanitize"))
    r4.addWidget(combo_sanitize)

    r4.addSpacing(10)
    r4.addWidget(chk_embed)
    r4.addWidget(QLabel("Assets dir"))
    r4.addWidget(assets_edit)
    r4.addStretch(1)
    root.addLayout(r4)

    action_row = QHBoxLayout()
    action_row.addStretch(1)

    btn_open = QPushButton("Open output")
    btn_open.setEnabled(False)
    action_row.addWidget(btn_open)

    btn_folder = QPushButton("Open folder")
    btn_folder.setEnabled(False)
    action_row.addWidget(btn_folder)

    btn_run = QPushButton("Generate")
    btn_run.setDefault(True)
    btn_run.setMinimumHeight(34)
    action_row.addWidget(btn_run)

    root.addLayout(action_row)

    last_generated = {"path": None}

    out_manually_set = {"value": False}

    def suggested_out_path(in_text: str) -> str:
        try:
            p = Path((in_text or "").strip())
            if p.name:
                return str(p.with_suffix('.html'))
        except Exception:
            pass
        return ""

    def maybe_update_out_from_in():
        if out_manually_set["value"]:
            return
        sug = suggested_out_path(in_edit.text())
        if sug:
            out_edit.setText(sug)

    def browse_in():
        p, _ = QFileDialog.getOpenFileName(w, "Select markdown", str(Path.cwd()), "Markdown (*.md);;All files (*.*)")
        if p:
            in_edit.setText(p)
            try:
                if not title_edit.text().strip():
                    title_edit.setText(Path(p).stem)
            except Exception:
                pass
            try:
                maybe_update_out_from_in()
            except Exception:
                pass

    def _handle_drop(path: str):
        in_edit.setText(path)
        try:
            if not title_edit.text().strip():
                title_edit.setText(Path(path).stem)
        except Exception:
            pass
        try:
            maybe_update_out_from_in()
        except Exception:
            pass

    w._on_drop = _handle_drop

    def browse_out():
        try:
            base = Path(in_edit.text().strip())
            default_path = base.with_suffix('.html') if base.name else (Path.cwd() / 'output.html')
        except Exception:
            default_path = Path.cwd() / 'output.html'
        p, _ = QFileDialog.getSaveFileName(w, "Save HTML", str(default_path), "HTML (*.html)")
        if p:
            out_edit.setText(p)
            out_manually_set["value"] = True

    def open_output():
        try:
            p = last_generated.get("path")
            if not p:
                return
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))
        except Exception:
            return

    def open_folder():
        try:
            p = last_generated.get("path")
            if not p:
                return
            folder = Path(str(p)).parent
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
        except Exception:
            return

    def run():
        nonlocal w
        in_raw = in_edit.text().strip()
        if not in_raw:
            QMessageBox.critical(w, "Error", "Please select an input markdown file (*.md).")
            return

        in_path = Path(in_raw)
        out_raw = out_edit.text().strip()
        out_path = Path(out_raw) if out_raw else in_path.with_suffix('.html')
        if not in_path.exists() or not in_path.is_file():
            QMessageBox.critical(w, "Error", f"Input file not found:\n{in_path}")
            return

        render_config = RenderConfig(
            collapse_codeblock_min_lines=int(spin_collapse.value()),
            mermaid_sanitize_mode=str(combo_sanitize.currentText() or "auto"),
            embed_assets=bool(chk_embed.isChecked()),
            assets_dir=Path((assets_edit.text() or "assets").strip() or "assets"),
        )

        # Show progress
        btn_run.setEnabled(False)
        btn_run.setText("Generating...")
        app.processEvents()

        try:
            md_text = in_path.read_text(encoding="utf-8")
        except Exception as e:
            btn_run.setEnabled(True)
            btn_run.setText("Generate")
            QMessageBox.critical(w, "Error", f"Failed to read input:\n{e}")
            return

        title = title_edit.text().strip() or in_path.stem
        try:
            out_html = markdown_to_tailwind_html(md_text, title=title, config=render_config)
            out_path.write_text(out_html, encoding="utf-8")
        except Exception as e:
            btn_run.setEnabled(True)
            btn_run.setText("Generate")
            QMessageBox.critical(w, "Error", f"Failed to generate HTML:\n{e}")
            return

        btn_run.setEnabled(True)
        btn_run.setText("Generate")

        # Persist GUI state
        try:
            settings.setValue("in_path", str(in_path))
            settings.setValue("out_path", str(out_path))
            settings.setValue("title", str(title_edit.text()))
            settings.setValue("collapse_min_lines", int(spin_collapse.value()))
            settings.setValue("mermaid_sanitize", str(combo_sanitize.currentText()))
            settings.setValue("embed_assets", 1 if chk_embed.isChecked() else 0)
            settings.setValue("assets_dir", str(assets_edit.text()))
        except Exception:
            pass

        last_generated["path"] = str(out_path)
        btn_open.setEnabled(True)
        btn_folder.setEnabled(True)

        QMessageBox.information(w, "Done", f"Generated:\n{out_path}")

    btn_in.clicked.connect(browse_in)
    btn_out.clicked.connect(browse_out)
    btn_run.clicked.connect(run)
    btn_open.clicked.connect(open_output)
    btn_folder.clicked.connect(open_folder)

    def on_out_edited(_text: str):
        # Any manual edit to output path should stop auto-following the input path.
        if out_edit.hasFocus():
            out_manually_set["value"] = True

    def on_in_edited(_text: str):
        maybe_update_out_from_in()
        try:
            if not title_edit.text().strip():
                p = Path(in_edit.text().strip())
                if p.name:
                    title_edit.setText(p.stem)
        except Exception:
            pass

    out_edit.textEdited.connect(on_out_edited)
    in_edit.textChanged.connect(on_in_edited)

    # Restore previous session
    try:
        prev_collapse = int(settings.value("collapse_min_lines", COLLAPSE_CODEBLOCK_MIN_LINES) or COLLAPSE_CODEBLOCK_MIN_LINES)
        prev_sanitize = str(settings.value("mermaid_sanitize", MERMAID_SANITIZE_MODE) or MERMAID_SANITIZE_MODE)
        prev_embed = int(settings.value("embed_assets", 0) or 0)
        prev_assets = str(settings.value("assets_dir", str(ASSETS_DIR)) or str(ASSETS_DIR))

        # Keep input/output fields empty on launch so placeholders (*.md/*.html) are visible.
        # (We still persist these values on Generate, but we don't auto-restore them.)
        try:
            in_edit.setText("")
            out_edit.setText("")
            title_edit.setText("")
            out_manually_set["value"] = False
        except Exception:
            pass
        spin_collapse.setValue(prev_collapse)
        if prev_sanitize:
            combo_sanitize.setCurrentText(prev_sanitize)
        chk_embed.setChecked(bool(prev_embed))
        if prev_assets:
            assets_edit.setText(prev_assets)
    except Exception:
        pass

    w.show()
    # Center on screen
    try:
        screen = app.primaryScreen().availableGeometry()
        w.move(
            (screen.width() - w.frameGeometry().width()) // 2,
            (screen.height() - w.frameGeometry().height()) // 2,
        )
    except Exception:
        pass
    return app.exec()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gui",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--embed-assets",
        dest="embed_assets",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--assets-dir",
        dest="assets_dir",
        default=str(ASSETS_DIR),
    )
    parser.add_argument(
        "--in",
        dest="in_path",
        default="TFT_DUAL_MODEL_DESIGN_GUIDE_v2.1.md",
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        default=None,
    )
    parser.add_argument(
        "--title",
        dest="title",
        default=None,
    )
    parser.add_argument(
        "--collapse-min-lines",
        dest="collapse_min_lines",
        type=int,
        default=COLLAPSE_CODEBLOCK_MIN_LINES,
    )
    parser.add_argument(
        "--mermaid-sanitize",
        dest="mermaid_sanitize",
        choices=["auto", "on", "off"],
        default=MERMAID_SANITIZE_MODE,
    )
    args = parser.parse_args()

    # Default to GUI unless --cli is provided.
    if not bool(args.cli):
        raise SystemExit(run_gui())

    in_path = Path(args.in_path)
    out_path = Path(args.out_path) if args.out_path else in_path.with_suffix('.html')

    if not in_path.exists() or not in_path.is_file():
        raise SystemExit(f"Input file not found: {in_path}")

    if args.collapse_min_lines is not None and int(args.collapse_min_lines) > 0:
        collapse_min_lines = int(args.collapse_min_lines)
    else:
        collapse_min_lines = COLLAPSE_CODEBLOCK_MIN_LINES

    render_config = RenderConfig(
        collapse_codeblock_min_lines=collapse_min_lines,
        mermaid_sanitize_mode=str(args.mermaid_sanitize or MERMAID_SANITIZE_MODE),
        embed_assets=bool(args.embed_assets),
        assets_dir=Path(str(args.assets_dir or "assets").strip() or "assets"),
    )

    md_text = in_path.read_text(encoding="utf-8")
    title = args.title if args.title is not None else in_path.stem
    out_html = markdown_to_tailwind_html(md_text, title=title, config=render_config)
    out_path.write_text(out_html, encoding="utf-8")

    print(str(out_path))
