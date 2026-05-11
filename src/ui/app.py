import gradio as gr
from src.registry import load_registry
from src.lesson.controller import LessonController


LANGS = load_registry()


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

        with gr.Row():
            with gr.Column(scale=1):
                cam = gr.Image(
                    sources=["webcam"],
                    streaming=True,
                    label="Live feed",
                )
            with gr.Column(scale=1):
                target = gr.Image(
                    label="Sign this letter",
                    interactive=False,
                )
                light = gr.HTML(value="", label="Score")
                status = gr.Markdown()
                with gr.Row():
                    skip_btn = gr.Button("Skip")
                    next_btn = gr.Button("Next letter")

        # State tracking (Gradio doesn't have built-in state in streaming)
        # We use gr.State for the current letter index
        letter_idx = gr.State(0)

        def on_frame(frame, lang_code, idx):
            if frame is None:
                return frame, "", "", idx
            result = controller.process_frame(frame, lang_code)
            light_html = controller._scorer.render_light(result["light"]) if controller._scorer else ""
            return (
                result["annotated"],
                light_html,
                result["status"],
                idx,
            )

        cam.stream(
            lambda frame, lang_code, idx: on_frame(frame, lang_code, idx),
            inputs=[cam, lang, letter_idx],
            outputs=[cam, light, status, letter_idx],
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
            lang = controller._registry[lang_code]
            new_idx = (idx + 1) % len(lang.classes)
            controller.set_target(new_idx)
            letter = lang.classes[new_idx]
            ref = controller.get_reference_image(lang_code, new_idx)
            total = len(lang.classes)
            dots = "●" * (new_idx + 1) + "○" * (total - new_idx - 1)
            return new_idx, letter, ref, dots

        next_btn.click(
            on_next,
            inputs=[letter_idx, lang],
            outputs=[letter_idx, letter_display, target, progress_dots],
        )

        def on_skip(idx, lang_code):
            return on_next(idx, lang_code)

        skip_btn.click(
            on_skip,
            inputs=[letter_idx, lang],
            outputs=[letter_idx, letter_display, target, progress_dots],
        )

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.launch(server_name="0.0.0.0", server_port=7860)
