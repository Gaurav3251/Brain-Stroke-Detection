import threading
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from brainstroke.inference import load_models, predict_single_image


def main():
    root = tk.Tk()
    root.title("Brain Stroke Detection")
    root.geometry("520x260")

    img_var = tk.StringVar()
    model_var = tk.StringVar(value="ensemble")
    status_var = tk.StringVar(value="Load models to begin.")

    models = {"value": None}

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
                status_var.set("Done. Check outputs/inference.")
            except Exception as e:
                status_var.set("Inference failed")
                messagebox.showerror("Inference Error", str(e))

        threading.Thread(target=_run, daemon=True).start()

    tk.Label(root, text="Brain Stroke Detection", font=("Segoe UI", 14, "bold")).pack(pady=10)

    frm = tk.Frame(root)
    frm.pack(pady=5)

    tk.Label(frm, text="Selected image:").grid(row=0, column=0, sticky="w")
    tk.Entry(frm, textvariable=img_var, width=45).grid(row=1, column=0, padx=5)
    tk.Button(frm, text="Browse", command=select_image).grid(row=1, column=1)

    tk.Label(frm, text="Model:").grid(row=2, column=0, sticky="w", pady=(10, 0))
    ttk.Combobox(
        frm,
        textvariable=model_var,
        values=["ensemble", "densenet121", "efficientnet_b4", "swin_unet", "all"],
        state="readonly",
        width=20,
    ).grid(row=3, column=0, sticky="w", padx=5)

    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=10)
    tk.Button(btn_frame, text="Load Models", command=load_all_models).pack(side=tk.LEFT, padx=5)
    tk.Button(btn_frame, text="Run Inference", command=run_inference).pack(side=tk.LEFT, padx=5)

    tk.Label(root, textvariable=status_var, fg="#444").pack(pady=5)

    root.mainloop()


if __name__ == "__main__":
    main()
