import gradio as gr
from src.registry import load_registry
from src.lesson.controller import LessonController


LANGS = load_registry()

_BAR_SHELL = """
<div style="background:#e0e0e0;border-radius:8px;height:36px;
            position:relative;overflow:hidden;margin:4px 0;">
  <div id="qbar-fill"
       style="width:0%;height:100%;background:hsl(0,80%,42%);
              transition:width 0.25s ease,background 0.25s ease;"></div>
  <div style="position:absolute;left:90%;top:0;width:3px;height:100%;
              background:rgba(0,0,0,0.35);"></div>
  <span id="qbar-label"
        style="position:absolute;inset:0;display:flex;align-items:center;
               justify-content:center;font-weight:bold;font-size:15px;color:#111;">
    0%
  </span>
</div>
"""

_STATUS_SHELL = """
<div id="status-box"
     style="font-size:26px;font-weight:bold;text-align:center;
            padding:12px 0;min-height:48px;">
  &nbsp;
</div>
"""

_STATUS_JS = """(txt) => {
    const el = document.getElementById('status-box');
    if (el) el.innerHTML = txt || '&nbsp;';
    return txt;
}"""

_BAR_JS = """(conf) => {
    const fill  = document.getElementById('qbar-fill');
    const label = document.getElementById('qbar-label');
    if (!fill || !label) return conf;
    const p = parseFloat(conf) || 0;
    const hue = Math.round(p * 120);               // 0=red → 120=green
    fill.style.width      = Math.round(p * 100) + '%';
    fill.style.background = 'hsl(' + hue + ',80%,42%)';
    label.textContent     = Math.round(p * 100) + '%';
    return conf;
}"""


def build_app():
    controller = LessonController(LANGS)

    with gr.Blocks(title="Sign Language Tutor") as demo:
        gr.Markdown(
            "# Sign Language Tutor\n"
            "Learn fingerspelling in multiple sign languages. "
            "This lab teaches **fingerspelling** — one component of sign language."
        )

        with gr.Row():
            lang = gr.Dropdown(
                choices=[(l.name, l.code) for l in LANGS.values()],
                value=list(LANGS.keys())[0],
                label="Language",
            )
            lesson = gr.Dropdown(
                choices=[("Alphabet", "alphabet")],
                value="alphabet",
                label="Lesson",
            )
            letter_display = gr.Textbox(label="Current letter", interactive=False)
            progress_dots = gr.HTML(value="●○○○○○○○○○○", label="Progress")

        gr.Markdown("**Click the Record button (⏺) on the webcam below to start signing.**")

        with gr.Row():
            with gr.Column(scale=1):
                cam = gr.Image(
                    sources=["webcam"],
                    streaming=True,
                    label="Live feed — press ⏺ to start",
                )
            with gr.Column(scale=1):
                target = gr.Image(
                    label="Sign this letter",
                    interactive=False,
                )
                # Static bar shell — rendered once, never replaced
                gr.HTML(value=_BAR_SHELL, label="Quality")
                # Hidden number carries the value; JS updates the bar in-place
                quality_val = gr.Number(value=0.0, visible=False)
                # Static status shell — JS updates text in-place, no flicker
                gr.HTML(value=_STATUS_SHELL)
                status = gr.Textbox(value="", visible=False)
                with gr.Row():
                    skip_btn = gr.Button("Skip")
                    next_btn = gr.Button("Next letter", interactive=False, variant="primary")

        letter_idx = gr.State(0)
        passed = gr.State(False)  # latches True once the letter is passed

        def on_frame(frame, lang_code, idx, already_passed):
            if frame is None:
                return 0.0, gr.update(), "", idx, already_passed
            try:
                result = controller.process_frame(frame, lang_code)
                quality = result.get("confidence", 0.0)
                completed = already_passed or result.get("completed", False)
                status_text = (
                    "✅ PASSED! Proceed to the next letter."
                    if completed
                    else result.get("status", "").replace("**", "")
                )
                return quality, gr.update(interactive=completed), status_text, idx, completed
            except Exception as e:
                return 0.0, gr.update(), f"Error: {e}", idx, already_passed

        cam.stream(
            on_frame,
            inputs=[cam, lang, letter_idx, passed],
            outputs=[quality_val, next_btn, status, letter_idx, passed],
            stream_every=0.2,
        )

        # JS-only handlers: update DOM directly — no HTML replacement, no flicker
        quality_val.change(
            fn=None,
            inputs=[quality_val],
            outputs=[quality_val],
            js=_BAR_JS,
        )
        status.change(
            fn=None,
            inputs=[status],
            outputs=[status],
            js=_STATUS_JS,
        )

        def on_switch_language(lang_code, idx):
            controller.switch_language(lang_code)
            letter = controller.get_letter(lang_code, idx)
            ref = controller.get_reference_image(lang_code, idx)
            total = len(controller._registry[lang_code].classes)
            return (
                letter,
                ref,
                f"●{'○' * (total - 1)}",
            )

        lang.change(
            on_switch_language,
            inputs=[lang, letter_idx],
            outputs=[letter_display, target, progress_dots],
        )

        def on_next(idx, lang_code):
            lang_obj = controller._registry[lang_code]
            new_idx = (idx + 1) % len(lang_obj.classes)
            controller.set_target(new_idx)
            letter = lang_obj.classes[new_idx]
            ref = controller.get_reference_image(lang_code, new_idx)
            total = len(lang_obj.classes)
            dots = "●" * (new_idx + 1) + "○" * (total - new_idx - 1)
            return new_idx, letter, ref, dots, gr.update(interactive=False), False, ""

        next_btn.click(
            on_next,
            inputs=[letter_idx, lang],
            outputs=[letter_idx, letter_display, target, progress_dots, next_btn, passed, status],
        )

        skip_btn.click(
            on_next,
            inputs=[letter_idx, lang],
            outputs=[letter_idx, letter_display, target, progress_dots, next_btn, passed, status],
        )

        demo.load(
            on_switch_language,
            inputs=[lang, letter_idx],
            outputs=[letter_display, target, progress_dots],
        )

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.launch(server_name="0.0.0.0", server_port=7860)
