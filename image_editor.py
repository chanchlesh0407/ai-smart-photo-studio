
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import threading


# ============================================================
# OPTIONAL AI BACKGROUND REMOVAL
# ============================================================

REMBG_AVAILABLE = False
rembg_remove = None
rembg_session = None

try:

    from rembg import remove, new_session

    rembg_remove = remove

    REMBG_AVAILABLE = True

except ImportError:

    REMBG_AVAILABLE = False


# ============================================================
# IMAGE EDITOR
# ============================================================

class ImageEditor:

    def __init__(
        self,
        root,
        image_path=None
    ):

        self.root = root

        self.root.title(
            "AI Smart Photo Studio - Image Editor"
        )

        self.root.geometry(
            "1250x780"
        )

        self.root.minsize(
            1000,
            650
        )

        self.root.configure(
            bg="#17181c"
        )


        # ====================================================
        # IMAGE DATA
        # ====================================================

        self.original_image = None

        self.current_image = None

        self.before_image = None

        self.display_photo = None

        self.image_path = None


        # ====================================================
        # SLIDER VALUES
        # ====================================================

        self.brightness_value = 0

        self.contrast_value = 1.0

        self.sharpness_value = 0


        # ====================================================
        # UI STATE
        # ====================================================

        self.processing = False

        self.show_before = False


        # ====================================================
        # CREATE UI
        # ====================================================

        self.create_ui()


        # ====================================================
        # LOAD IMAGE
        # ====================================================

        if image_path:

            self.load_image(
                image_path
            )


    # ========================================================
    # CREATE UI
    # ========================================================

    def create_ui(self):

        # ----------------------------------------------------
        # HEADER
        # ----------------------------------------------------

        header = tk.Frame(
            self.root,
            bg="#17181c",
            height=75
        )

        header.pack(
            fill="x"
        )

        header.pack_propagate(
            False
        )


        title = tk.Label(
            header,
            text="AI SMART PHOTO STUDIO",
            font=(
                "Segoe UI",
                22,
                "bold"
            ),
            fg="white",
            bg="#17181c"
        )

        title.pack(
            side="left",
            padx=25,
            pady=15
        )


        subtitle = tk.Label(
            header,
            text="AI Image Enhancement & Editing",
            font=(
                "Segoe UI",
                10
            ),
            fg="#9aa0a6",
            bg="#17181c"
        )

        subtitle.pack(
            side="left",
            padx=5,
            pady=22
        )


        # ----------------------------------------------------
        # MAIN AREA
        # ----------------------------------------------------

        main = tk.Frame(
            self.root,
            bg="#17181c"
        )

        main.pack(
            fill="both",
            expand=True,
            padx=18,
            pady=(0, 15)
        )


        # ====================================================
        # IMAGE AREA
        # ====================================================

        image_container = tk.Frame(
            main,
            bg="#24262d"
        )

        image_container.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(0, 12)
        )


        self.image_label = tk.Label(
            image_container,
            text=(
                "No image loaded\n\n"
                "Click  Open Image  to begin"
            ),
            font=(
                "Segoe UI",
                15
            ),
            fg="#8f949c",
            bg="#24262d"
        )

        self.image_label.pack(
            fill="both",
            expand=True,
            padx=15,
            pady=15
        )


        # ====================================================
        # CONTROL PANEL
        # ====================================================

        controls = tk.Frame(
            main,
            bg="#202228",
            width=315
        )

        controls.pack(
            side="right",
            fill="y"
        )

        controls.pack_propagate(
            False
        )


        # ====================================================
        # FILE
        # ====================================================

        self.section(
            controls,
            "FILE"
        )


        self.button(
            controls,
            "📂  Open Image",
            self.open_image
        )


        self.button(
            controls,
            "💾  Save Image",
            self.save_image
        )


        # ====================================================
        # AI
        # ====================================================

        self.section(
            controls,
            "AI TOOLS"
        )


        self.button(
            controls,
            "✨  AI Auto Enhance",
            self.auto_enhance
        )


        self.button(
            controls,
            "🧹  AI Remove Background",
            self.remove_background
        )


        # ====================================================
        # ADJUSTMENTS
        # ====================================================

        self.section(
            controls,
            "ADJUSTMENTS"
        )


        # Brightness
        self.slider(
            controls,
            "Brightness",
            -100,
            100,
            0,
            self.change_brightness
        )


        # Contrast
        self.slider(
            controls,
            "Contrast",
            50,
            200,
            100,
            self.change_contrast
        )


        # Sharpness
        self.slider(
            controls,
            "Sharpness",
            0,
            100,
            0,
            self.change_sharpness
        )


        # ====================================================
        # QUICK EFFECTS
        # ====================================================

        self.section(
            controls,
            "QUICK EFFECTS"
        )


        effects = tk.Frame(
            controls,
            bg="#202228"
        )

        effects.pack(
            fill="x",
            padx=15
        )


        self.small_button(
            effects,
            "🔍 Sharpen",
            self.sharpen
        )


        self.small_button(
            effects,
            "🌫 Denoise",
            self.denoise
        )


        self.small_button(
            effects,
            "⚫ B&W",
            self.grayscale
        )


        self.small_button(
            effects,
            "🌁 Blur",
            self.blur
        )


        # ====================================================
        # RESIZE
        # ====================================================

        self.section(
            controls,
            "RESIZE"
        )


        resize_frame = tk.Frame(
            controls,
            bg="#202228"
        )

        resize_frame.pack(
            fill="x",
            padx=15
        )


        tk.Label(
            resize_frame,
            text="Width",
            fg="#b9bdc5",
            bg="#202228",
            font=("Segoe UI", 9)
        ).grid(
            row=0,
            column=0,
            padx=3
        )


        self.width_entry = tk.Entry(
            resize_frame,
            width=8,
            bg="#30333a",
            fg="white",
            insertbackground="white",
            relief="flat"
        )

        self.width_entry.grid(
            row=0,
            column=1,
            padx=3
        )


        tk.Label(
            resize_frame,
            text="Height",
            fg="#b9bdc5",
            bg="#202228",
            font=("Segoe UI", 9)
        ).grid(
            row=0,
            column=2,
            padx=3
        )


        self.height_entry = tk.Entry(
            resize_frame,
            width=8,
            bg="#30333a",
            fg="white",
            insertbackground="white",
            relief="flat"
        )

        self.height_entry.grid(
            row=0,
            column=3,
            padx=3
        )


        self.button(
            controls,
            "↔  Resize Image",
            self.resize_image
        )


        # ====================================================
        # BEFORE / AFTER
        # ====================================================

        self.section(
            controls,
            "COMPARISON"
        )


        self.button(
            controls,
            "↔  Show Before",
            self.toggle_before
        )


        self.button(
            controls,
            "↩  Reset",
            self.reset_image
        )


        # ====================================================
        # STATUS
        # ====================================================

        self.status = tk.Label(
            controls,
            text="Ready",
            font=(
                "Segoe UI",
                9
            ),
            fg="#8ab4f8",
            bg="#202228",
            wraplength=270,
            justify="left"
        )

        self.status.pack(
            fill="x",
            padx=18,
            pady=12
        )


        # ====================================================
        # AI STATUS
        # ====================================================

        if REMBG_AVAILABLE:

            ai_text = (
                "● AI Background Removal: Ready"
            )

            ai_color = "#81c995"

        else:

            ai_text = (
                "● AI Background Removal: Not installed"
            )

            ai_color = "#f28b82"


        self.ai_status = tk.Label(
            controls,
            text=ai_text,
            font=(
                "Segoe UI",
                8
            ),
            fg=ai_color,
            bg="#202228",
            wraplength=270
        )

        self.ai_status.pack(
            padx=15,
            pady=(0, 12)
        )


    # ========================================================
    # SECTION LABEL
    # ========================================================

    def section(
        self,
        parent,
        text
    ):

        label = tk.Label(
            parent,
            text=text,
            font=(
                "Segoe UI",
                9,
                "bold"
            ),
            fg="#8ab4f8",
            bg="#202228",
            anchor="w"
        )

        label.pack(
            fill="x",
            padx=15,
            pady=(13, 5)
        )


    # ========================================================
    # BUTTON
    # ========================================================

    def button(
        self,
        parent,
        text,
        command
    ):

        btn = tk.Button(
            parent,
            text=text,
            command=command,
            font=(
                "Segoe UI",
                9
            ),
            bg="#30333a",
            fg="white",
            activebackground="#41454d",
            activeforeground="white",
            relief="flat",
            bd=0,
            cursor="hand2",
            pady=7
        )

        btn.pack(
            fill="x",
            padx=15,
            pady=3
        )


    # ========================================================
    # SMALL BUTTON
    # ========================================================

    def small_button(
        self,
        parent,
        text,
        command
    ):

        btn = tk.Button(
            parent,
            text=text,
            command=command,
            font=(
                "Segoe UI",
                8
            ),
            bg="#30333a",
            fg="white",
            activebackground="#41454d",
            relief="flat",
            bd=0,
            cursor="hand2",
            pady=6
        )

        btn.pack(
            side="left",
            fill="x",
            expand=True,
            padx=2
        )


    # ========================================================
    # SLIDER
    # ========================================================

    def slider(
        self,
        parent,
        text,
        minimum,
        maximum,
        default,
        command
    ):

        frame = tk.Frame(
            parent,
            bg="#202228"
        )

        frame.pack(
            fill="x",
            padx=15,
            pady=2
        )


        label = tk.Label(
            frame,
            text=text,
            font=(
                "Segoe UI",
                8
            ),
            fg="#b9bdc5",
            bg="#202228"
        )

        label.pack(
            anchor="w"
        )


        scale = tk.Scale(
            frame,
            from_=minimum,
            to=maximum,
            orient="horizontal",
            resolution=1,
            showvalue=True,
            bg="#202228",
            fg="white",
            highlightthickness=0,
            troughcolor="#3b3e46",
            activebackground="#8ab4f8",
            command=command
        )

        scale.set(
            default
        )

        scale.pack(
            fill="x"
        )


        if text == "Brightness":

            self.brightness_scale = scale

        elif text == "Contrast":

            self.contrast_scale = scale

        elif text == "Sharpness":

            self.sharpness_scale = scale


    # ========================================================
    # OPEN IMAGE
    # ========================================================

    def open_image(self):

        path = filedialog.askopenfilename(

            title="Open Image",

            filetypes=[
                (
                    "Image Files",
                    "*.jpg *.jpeg *.png *.bmp *.webp"
                )
            ]
        )


        if not path:

            return


        self.load_image(
            path
        )


    # ========================================================
    # LOAD IMAGE
    # ========================================================

    def load_image(
        self,
        path
    ):

        image = cv2.imread(
            path,
            cv2.IMREAD_COLOR
        )


        if image is None:

            messagebox.showerror(
                "Error",
                "Could not load the image."
            )

            return


        self.original_image = (
            image.copy()
        )

        self.current_image = (
            image.copy()
        )

        self.before_image = (
            image.copy()
        )

        self.image_path = path


        height, width = (
            image.shape[:2]
        )


        self.width_entry.delete(
            0,
            tk.END
        )

        self.width_entry.insert(
            0,
            str(width)
        )


        self.height_entry.delete(
            0,
            tk.END
        )

        self.height_entry.insert(
            0,
            str(height)
        )


        self.reset_sliders()


        self.update_display()


        self.set_status(
            f"Loaded: {os.path.basename(path)}\n"
            f"Resolution: {width} × {height}"
        )


    # ========================================================
    # RESET SLIDERS
    # ========================================================

    def reset_sliders(self):

        self.brightness_scale.set(
            0
        )

        self.contrast_scale.set(
            100
        )

        self.sharpness_scale.set(
            0
        )


    # ========================================================
    # UPDATE DISPLAY
    # ========================================================

    def update_display(
        self,
        image=None
    ):

        if image is None:

            image = (
                self.current_image
            )


        if image is None:

            return


        # ----------------------------------------------------
        # BGR -> RGB
        # ----------------------------------------------------

        if len(image.shape) == 2:

            rgb = cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2RGB
            )

        else:

            rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )


        # ----------------------------------------------------
        # PIL
        # ----------------------------------------------------

        pil = Image.fromarray(
            rgb
        )


        # ----------------------------------------------------
        # Fit preview
        # ----------------------------------------------------

        max_width = 850

        max_height = 650


        pil.thumbnail(
            (
                max_width,
                max_height
            ),
            Image.Resampling.LANCZOS
        )


        self.display_photo = (
            ImageTk.PhotoImage(
                pil
            )
        )


        self.image_label.config(
            image=self.display_photo,
            text=""
        )


    # ========================================================
    # CHECK IMAGE
    # ========================================================

    def has_image(self):

        if self.current_image is None:

            messagebox.showwarning(
                "No Image",
                "Please open an image first."
            )

            return False

        return True


    # ========================================================
    # STATUS
    # ========================================================

    def set_status(
        self,
        text
    ):

        self.status.config(
            text=text
        )


    # ========================================================
    # BRIGHTNESS
    # ========================================================

    def change_brightness(
        self,
        value
    ):

        if self.original_image is None:

            return


        self.brightness_value = (
            float(value)
        )


        self.apply_adjustments()


    # ========================================================
    # CONTRAST
    # ========================================================

    def change_contrast(
        self,
        value
    ):

        if self.original_image is None:

            return


        self.contrast_value = (
            float(value) / 100.0
        )


        self.apply_adjustments()


    # ========================================================
    # SHARPNESS
    # ========================================================

    def change_sharpness(
        self,
        value
    ):

        if self.original_image is None:

            return


        self.sharpness_value = (
            float(value)
        )


        self.apply_adjustments()


    # ========================================================
    # APPLY SLIDER ADJUSTMENTS
    # ========================================================

    def apply_adjustments(self):

        if self.original_image is None:

            return


        image = (
            self.original_image.copy()
        )


        # ----------------------------------------------------
        # Brightness
        # ----------------------------------------------------

        brightness = (
            self.brightness_value
        )


        # ----------------------------------------------------
        # Contrast
        # ----------------------------------------------------

        contrast = (
            self.contrast_value
        )


        image = cv2.convertScaleAbs(
            image,
            alpha=contrast,
            beta=brightness
        )


        # ----------------------------------------------------
        # Sharpness
        # ----------------------------------------------------

        sharpness = (
            self.sharpness_value
        )


        if sharpness > 0:

            amount = (
                sharpness / 100.0
            )


            blurred = cv2.GaussianBlur(
                image,
                (0, 0),
                1.2
            )


            image = cv2.addWeighted(
                image,
                1.0 + amount,
                blurred,
                -amount,
                0
            )


        self.current_image = (
            image
        )


        self.update_display()


    # ========================================================
    # AUTO ENHANCE
    # ========================================================

    def auto_enhance(self):

        if not self.has_image():

            return


        self.before_image = (
            self.current_image.copy()
        )


        image = (
            self.current_image.copy()
        )


        # ----------------------------------------------------
        # LAB contrast enhancement
        # ----------------------------------------------------

        lab = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2LAB
        )


        l, a, b = cv2.split(
            lab
        )


        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )


        l = clahe.apply(
            l
        )


        lab = cv2.merge(
            (
                l,
                a,
                b
            )
        )


        enhanced = cv2.cvtColor(
            lab,
            cv2.COLOR_LAB2BGR
        )


        # ----------------------------------------------------
        # Mild color correction
        # ----------------------------------------------------

        enhanced = cv2.convertScaleAbs(
            enhanced,
            alpha=1.04,
            beta=2
        )


        # ----------------------------------------------------
        # High-quality mild sharpening
        # ----------------------------------------------------

        blurred = cv2.GaussianBlur(
            enhanced,
            (0, 0),
            1.0
        )


        enhanced = cv2.addWeighted(
            enhanced,
            1.12,
            blurred,
            -0.12,
            0
        )


        self.current_image = (
            enhanced
        )


        self.update_display()


        self.set_status(
            "AI Auto Enhance applied\n"
            "• Local contrast improved\n"
            "• Details enhanced\n"
            "• Image sharpened"
        )


    # ========================================================
    # SHARPEN
    # ========================================================

    def sharpen(self):

        if not self.has_image():

            return


        self.before_image = (
            self.current_image.copy()
        )


        image = (
            self.current_image
        )


        blurred = cv2.GaussianBlur(
            image,
            (0, 0),
            1.0
        )


        result = cv2.addWeighted(
            image,
            1.35,
            blurred,
            -0.35,
            0
        )


        self.current_image = (
            result
        )


        self.update_display()


        self.set_status(
            "Image sharpened"
        )


    # ========================================================
    # DENOISE
    # ========================================================

    def denoise(self):

        if not self.has_image():

            return


        self.before_image = (
            self.current_image.copy()
        )


        result = (
            cv2.fastNlMeansDenoisingColored(
                self.current_image,
                None,
                7,
                7,
                7,
                21
            )
        )


        self.current_image = (
            result
        )


        self.update_display()


        self.set_status(
            "Noise reduction applied"
        )


    # ========================================================
    # GRAYSCALE
    # ========================================================

    def grayscale(self):

        if not self.has_image():

            return


        self.before_image = (
            self.current_image.copy()
        )


        gray = cv2.cvtColor(
            self.current_image,
            cv2.COLOR_BGR2GRAY
        )


        self.current_image = cv2.cvtColor(
            gray,
            cv2.COLOR_GRAY2BGR
        )


        self.update_display()


        self.set_status(
            "Grayscale effect applied"
        )


    # ========================================================
    # BLUR
    # ========================================================

    def blur(self):

        if not self.has_image():

            return


        self.before_image = (
            self.current_image.copy()
        )


        self.current_image = cv2.GaussianBlur(
            self.current_image,
            (7, 7),
            0
        )


        self.update_display()


        self.set_status(
            "Soft blur applied"
        )


    # ========================================================
    # RESIZE
    # ========================================================

    def resize_image(self):

        if not self.has_image():

            return


        try:

            width = int(
                self.width_entry.get()
            )

            height = int(
                self.height_entry.get()
            )


            if (
                width <= 0
                or
                height <= 0
            ):

                raise ValueError


        except ValueError:

            messagebox.showerror(
                "Invalid Size",
                "Enter valid width and height."
            )

            return


        self.before_image = (
            self.current_image.copy()
        )


        self.current_image = (
            cv2.resize(
                self.current_image,
                (
                    width,
                    height
                ),
                interpolation=cv2.INTER_LANCZOS4
            )
        )


        self.update_display()


        self.set_status(
            f"Image resized to "
            f"{width} × {height}"
        )


    # ========================================================
    # AI BACKGROUND REMOVAL
    # ========================================================

    def remove_background(self):

        if not self.has_image():

            return


        if not REMBG_AVAILABLE:

            messagebox.showinfo(
                "AI Background Removal",
                "The AI background-removal engine "
                "is not installed.\n\n"
                "Install it with:\n\n"
                "pip install rembg\n\n"
                "Then restart the editor."
            )

            return


        if self.processing:

            return


        self.processing = True


        self.set_status(
            "AI is removing the background...\n"
            "Please wait."
        )


        thread = threading.Thread(
            target=self._background_worker,
            daemon=True
        )


        thread.start()


    # ========================================================
    # BACKGROUND REMOVAL WORKER
    # ========================================================

    def _background_worker(self):

        try:

            # ------------------------------------------------
            # Convert OpenCV -> PIL
            # ------------------------------------------------

            rgb = cv2.cvtColor(
                self.current_image,
                cv2.COLOR_BGR2RGB
            )


            pil_image = Image.fromarray(
                rgb
            )


            # ------------------------------------------------
            # Create reusable session
            # ------------------------------------------------

            global rembg_session


            if rembg_session is None:

                from rembg import new_session

                rembg_session = new_session(
                    "u2net"
                )


            # ------------------------------------------------
            # Remove background
            # ------------------------------------------------

            output = rembg_remove(
                pil_image,
                session=rembg_session
            )


            # ------------------------------------------------
            # Convert result
            # ------------------------------------------------

            if output.mode != "RGBA":

                output = output.convert(
                    "RGBA"
                )


            rgba = np.array(
                output
            )


            # ------------------------------------------------
            # Save temporary alpha image
            # ------------------------------------------------

            bgr = cv2.cvtColor(
                rgba,
                cv2.COLOR_RGBA2BGRA
            )


            # ------------------------------------------------
            # Display against a checkerboard-like
            # neutral background while preserving alpha
            # ------------------------------------------------

            alpha = bgr[:, :, 3]


            background = np.full(
                (
                    bgr.shape[0],
                    bgr.shape[1],
                    3
                ),
                245,
                dtype=np.uint8
            )


            foreground = bgr[:, :, :3]


            alpha_float = (
                alpha.astype(
                    np.float32
                ) / 255.0
            )


            alpha_float = (
                alpha_float[:, :, None]
            )


            composite = (
                foreground
                * alpha_float
                +
                background
                * (
                    1
                    -
                    alpha_float
                )
            )


            composite = np.clip(
                composite,
                0,
                255
            ).astype(
                np.uint8
            )


            self.root.after(
                0,
                lambda: self.background_finished(
                    composite
                )
            )


        except Exception as e:

            self.root.after(
                0,
                lambda: self.background_error(
                    str(e)
                )
            )


    # ========================================================
    # BACKGROUND SUCCESS
    # ========================================================

    def background_finished(
        self,
        image
    ):

        self.processing = False

        self.before_image = (
            self.current_image.copy()
        )

        self.current_image = (
            image
        )


        self.update_display()


        self.set_status(
            "AI Background Removal complete\n"
            "Subject isolated successfully."
        )


    # ========================================================
    # BACKGROUND ERROR
    # ========================================================

    def background_error(
        self,
        error
    ):

        self.processing = False


        self.set_status(
            "Background removal failed."
        )


        messagebox.showerror(
            "AI Background Removal Error",
            error
        )


    # ========================================================
    # BEFORE / AFTER
    # ========================================================

    def toggle_before(self):

        if self.current_image is None:

            return


        if self.before_image is None:

            return


        self.show_before = (
            not self.show_before
        )


        if self.show_before:

            self.update_display(
                self.before_image
            )

            self.set_status(
                "Showing BEFORE image"
            )

        else:

            self.update_display(
                self.current_image
            )

            self.set_status(
                "Showing AFTER image"
            )


    # ========================================================
    # RESET
    # ========================================================

    def reset_image(self):

        if self.original_image is None:

            return


        self.current_image = (
            self.original_image.copy()
        )


        self.before_image = (
            self.original_image.copy()
        )


        self.show_before = False


        self.reset_sliders()


        height, width = (
            self.current_image.shape[:2]
        )


        self.width_entry.delete(
            0,
            tk.END
        )

        self.width_entry.insert(
            0,
            str(width)
        )


        self.height_entry.delete(
            0,
            tk.END
        )

        self.height_entry.insert(
            0,
            str(height)
        )


        self.update_display()


        self.set_status(
            "Image restored to original"
        )


    # ========================================================
    # SAVE IMAGE
    # ========================================================

    def save_image(self):

        if not self.has_image():

            return


        path = filedialog.asksaveasfilename(

            title="Save Edited Image",

            defaultextension=".jpg",

            filetypes=[
                (
                    "JPEG Image",
                    "*.jpg"
                ),
                (
                    "PNG Image",
                    "*.png"
                ),
                (
                    "WebP Image",
                    "*.webp"
                )
            ]
        )


        if not path:

            return


        extension = (
            os.path.splitext(path)[1]
            .lower()
        )


        image = (
            self.current_image
        )


        # ----------------------------------------------------
        # PNG supports alpha
        # ----------------------------------------------------

        if (
            extension == ".png"
            and
            len(image.shape) == 3
            and
            image.shape[2] == 4
        ):

            success = cv2.imwrite(
                path,
                image
            )


        else:

            # JPEG does not support alpha
            if (
                len(image.shape) == 3
                and
                image.shape[2] == 4
            ):

                image = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGRA2BGR
                )


            success = cv2.imwrite(
                path,
                image
            )


        if success:

            self.set_status(
                f"Saved:\n"
                f"{os.path.basename(path)}"
            )


            messagebox.showinfo(
                "Saved",
                "Edited image saved successfully."
            )


        else:

            messagebox.showerror(
                "Save Error",
                "Could not save the image."
            )


# ============================================================
# MAIN
# ============================================================

def main():

    root = tk.Tk()

    app = ImageEditor(
        root
    )

    root.mainloop()


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()

