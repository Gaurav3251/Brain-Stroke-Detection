import threading
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import sys
import os
from pathlib import Path
from PIL import Image, ImageTk

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from brainstroke.model_io import load_models
from brainstroke.inference import predict_single_image
from brainstroke.core.config import INFER_DIR


MODEL_CHOICES = [
    "ensemble",
    "sg_densenet201_se",
    "sg_mobilenet_v2",
    "sg_resnet50",
    "sg_xception",
    "sg_convnext_small",
    "all",
]


def main():
    root = tk.Tk()
    root.title("Brain Stroke Detection")
    root.geometry("980x760")

    img_var = tk.StringVar()
    model_var = tk.StringVar(value="ensemble")
    status_var = tk.StringVar(value="Load models to begin.")
    visual_title_var = tk.StringVar(value="Visual review will appear here after inference.")

    models = {"value": None}
    preview_ref = {"image": None}

    def show_visual_review(image_path: str):
        stem = Path(image_path).stem
        review_path = INFER_DIR / f"inference_{stem}_explainability.png"
        if not review_path.exists():
            visual_title_var.set("Visual review output was not generated.")
            preview_label.configure(image="", text="No visual review image found.")
            preview_ref["image"] = None
            return

        img = Image.open(review_path)
        resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
        img.thumbnail((900, 430), resample)
        photo = ImageTk.PhotoImage(img)
        preview_ref["image"] = photo
        visual_title_var.set("Input CT, Segmentation, and GradCAM | Model Output: " + model_var.get())
        preview_label.configure(image=photo, text="")


    def load_all_models():
        try:
            status_var.set("Loading models...")
            root.update_idletasks()
            models["value"] = load_models()
            status_var.set("Models loaded. Ready.")
        except Exception as e:
            status_var.set("Failed to load models")
            messagebox.showerror("Load Error", str(e))

    def select_image():
        path = filedialog.askopenfilename(
            title="Select CT image",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp"), ("All files", "*.*")],
        )
        if path:
            img_var.set(path)

    def run_inference():
        if not img_var.get():
            messagebox.showwarning("Missing image", "Please select an image first.")
            return
        if models["value"] is None:
            messagebox.showwarning("Models not loaded", "Models are not loaded yet.")
            return

        def _run():
            try:
                status_var.set("Running inference...")
                predict_single_image(img_var.get(), models["value"], model_choice=model_var.get())
                root.after(0, lambda: show_visual_review(img_var.get()))
                status_var.set("Done. Visual review updated below. Outputs saved in artifacts/ui/inference.")
            except Exception as e:
                status_var.set("Inference failed")
                messagebox.showerror("Inference Error", str(e))

        threading.Thread(target=_run, daemon=True).start()

    tk.Label(root, text="Brain Stroke Detection", font=("Segoe UI", 14, "bold")).pack(pady=10)

    frm = tk.Frame(root)
    frm.pack(pady=5)

    tk.Label(frm, text="Selected image:").grid(row=0, column=0, sticky="w")
    tk.Entry(frm, textvariable=img_var, width=55).grid(row=1, column=0, padx=5)
    tk.Button(frm, text="Browse", command=select_image).grid(row=1, column=1)

    tk.Label(frm, text="Model:").grid(row=2, column=0, sticky="w", pady=(10, 0))
    ttk.Combobox(
        frm,
        textvariable=model_var,
        values=MODEL_CHOICES,
        state="readonly",
        width=24,
    ).grid(row=3, column=0, sticky="w", padx=5)

    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=10)
    tk.Button(btn_frame, text="Load Models", command=load_all_models).pack(side=tk.LEFT, padx=5)
    tk.Button(btn_frame, text="Run Inference", command=run_inference).pack(side=tk.LEFT, padx=5)

    tk.Label(root, textvariable=status_var, fg="#444").pack(pady=5)

    visual_frame = tk.LabelFrame(root, text="Visual Review", padx=10, pady=10)
    visual_frame.pack(fill=tk.BOTH, expand=True, padx=14, pady=10)
    tk.Label(visual_frame, textvariable=visual_title_var, font=("Segoe UI", 11, "bold")).pack(pady=(0, 8))
    preview_label = tk.Label(visual_frame, text="No inference output yet.", bg="#f5f5f5", relief=tk.GROOVE)
    preview_label.pack(fill=tk.BOTH, expand=True)

    root.mainloop()


if __name__ == "__main__":
    main()
