"""
Standalone Dataset Setup tool for the Underwater Image Visibility Analyzer.

Lets you browse a folder of JPEG images (or a random sample), score each
image's visibility (0-100), and automatically copies images into
category folders (excellent/good/fair/poor/zero) while writing a
labels.csv file usable for training.

This is a self-contained extraction of the "Dataset Setup" tab from gui.py
and has no dependency on the visibility_analyzer package, PyTorch, or pandas.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import csv
import glob
import random
import shutil
from datetime import datetime
from PIL import Image, ImageTk


class DatasetSetupApp:
    """Standalone application for scoring images and building a training dataset."""

    def __init__(self, root):
        self.root = root
        self.root.title("Visibility Dataset Setup Tool")
        self.root.geometry("1100x750")

        # Configuration variables
        self.source_images_folder = tk.StringVar()
        self.dataset_output_folder = tk.StringVar()
        self.sample_size = tk.IntVar(value=100)

        # Scoring state
        self.current_image_index = 0
        self.image_files = []
        self.processed_images = {}  # image_path -> score
        self.current_score = tk.IntVar(value=50)
        self.preview_image = None
        self.selected_image_path = ""
        self.zoom_factor = 1.0

        self.setup_gui()

    def setup_gui(self):
        """Build the dataset setup layout directly in the root window."""
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel - Configuration and controls
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=1)

        # Configuration section
        config_frame = ttk.LabelFrame(left_frame, text="Setup Configuration", padding=10)
        config_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(config_frame, text="Source Images Folder:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(config_frame, textvariable=self.source_images_folder, width=40).grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(config_frame, text="Browse", command=self.browse_source_images).grid(row=0, column=2, pady=2)

        ttk.Label(config_frame, text="Dataset Output Folder:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(config_frame, textvariable=self.dataset_output_folder, width=40).grid(row=1, column=1, padx=5, pady=2)
        ttk.Button(config_frame, text="Browse", command=self.browse_dataset_output).grid(row=1, column=2, pady=2)

        ttk.Label(config_frame, text="Sample Size (for large datasets):").grid(row=2, column=0, sticky=tk.W, pady=2)
        sample_frame = ttk.Frame(config_frame)
        sample_frame.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)

        sample_entry = ttk.Entry(sample_frame, textvariable=self.sample_size, width=10)
        sample_entry.pack(side=tk.LEFT)
        ttk.Label(sample_frame, text="images (0 = all)").pack(side=tk.LEFT, padx=(5, 0))

        action_frame = ttk.Frame(config_frame)
        action_frame.grid(row=3, column=0, columnspan=3, pady=10)

        setup_row = ttk.Frame(action_frame)
        setup_row.pack(fill=tk.X, pady=2)

        ttk.Button(setup_row, text="📁 Create Dataset Structure",
                   command=self.create_dataset_structure).pack(side=tk.LEFT, padx=5)
        ttk.Button(setup_row, text="💾 Finalize CSV",
                   command=self.export_dataset_csv).pack(side=tk.RIGHT, padx=5)

        load_row = ttk.Frame(action_frame)
        load_row.pack(fill=tk.X, pady=2)

        ttk.Button(load_row, text="🔍 Load All Images",
                   command=self.load_source_images).pack(side=tk.LEFT, padx=5)
        ttk.Button(load_row, text="🎲 Random Sample",
                   command=self.load_random_sample,
                   style="Accent.TButton").pack(side=tk.LEFT, padx=5)

        self.progress_info = ttk.Label(config_frame, text="Ready to load images... (JPEG files only)")
        self.progress_info.grid(row=4, column=0, columnspan=3, sticky=tk.W, pady=5)

        stats_frame = ttk.Frame(config_frame)
        stats_frame.grid(row=5, column=0, columnspan=3, sticky=tk.W, pady=2)

        ttk.Label(stats_frame, text="💡 Tip:", font=("TkDefaultFont", 8, "bold")).pack(side=tk.LEFT)
        ttk.Label(stats_frame, text="For datasets >1000 images, use Random Sample for faster scoring",
                  font=("TkDefaultFont", 8), foreground="blue").pack(side=tk.LEFT, padx=(2, 0))

        # Scoring section
        scoring_frame = ttk.LabelFrame(left_frame, text="Image Scoring", padding=10)
        scoring_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(scoring_frame, text="Visibility Score:").pack(anchor=tk.W)
        score_frame = ttk.Frame(scoring_frame)
        score_frame.pack(fill=tk.X, pady=5)

        ttk.Label(score_frame, text="Poor (0)").pack(side=tk.LEFT)
        self.score_scale = tk.Scale(score_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                     variable=self.current_score, length=200)
        self.score_scale.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=10)
        ttk.Label(score_frame, text="Excellent (100)").pack(side=tk.RIGHT)

        # Prevent scale widget from interfering with keyboard shortcuts
        self.score_scale.bind("<KeyPress-space>", lambda e: "break")
        self.score_scale.bind("<KeyPress-Return>", lambda e: "break")

        self.score_display = ttk.Label(scoring_frame, text="Score: 50", font=("TkDefaultFont", 12, "bold"))
        self.score_display.pack(pady=5)

        # Quick score buttons - Ordered by quality
        quick_score_frame = ttk.Frame(scoring_frame)
        quick_score_frame.pack(fill=tk.X, pady=5)

        row1_frame = ttk.Frame(quick_score_frame)
        row1_frame.pack(fill=tk.X, pady=2)

        ttk.Button(row1_frame, text="Zero (0)",
                   command=lambda: self.quick_score_and_next(0)).pack(side=tk.LEFT, padx=2)
        ttk.Button(row1_frame, text="Poor (10)",
                   command=lambda: self.quick_score_and_next(10)).pack(side=tk.LEFT, padx=2)
        ttk.Button(row1_frame, text="Too Low (20)",
                   command=lambda: self.quick_score_and_next(20)).pack(side=tk.LEFT, padx=2)
        ttk.Button(row1_frame, text="Barely Fair (41)",
                   command=lambda: self.quick_score_and_next(41)).pack(side=tk.LEFT, padx=2)
        ttk.Button(row1_frame, text="Fair (50)",
                   command=lambda: self.quick_score_and_next(50)).pack(side=tk.LEFT, padx=2)

        row2_frame = ttk.Frame(quick_score_frame)
        row2_frame.pack(fill=tk.X, pady=2)

        ttk.Button(row2_frame, text="Good (70)",
                   command=lambda: self.quick_score_and_next(70)).pack(side=tk.LEFT, padx=2)
        ttk.Button(row2_frame, text="Pretty Good (79)",
                   command=lambda: self.quick_score_and_next(79)).pack(side=tk.LEFT, padx=2)
        ttk.Button(row2_frame, text="Excellent (90)",
                   command=lambda: self.quick_score_and_next(90)).pack(side=tk.LEFT, padx=2)
        ttk.Button(row2_frame, text="Perfect (100)",
                   command=lambda: self.quick_score_and_next(100)).pack(side=tk.LEFT, padx=2)

        nav_frame = ttk.Frame(scoring_frame)
        nav_frame.pack(fill=tk.X, pady=10)

        ttk.Button(nav_frame, text="⬅ Previous", command=self.previous_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="✅ Score & Next", command=self.score_and_next,
                   style="Accent.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="⏭ Skip", command=self.next_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="➡ Next", command=self.next_image).pack(side=tk.RIGHT, padx=5)

        # Status section
        status_frame = ttk.LabelFrame(left_frame, text="Progress Status", padding=10)
        status_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.dataset_status = scrolledtext.ScrolledText(status_frame, height=8, width=40)
        self.dataset_status.pack(fill=tk.BOTH, expand=True)

        # Right panel - Image viewer
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=2)

        viewer_frame = ttk.LabelFrame(right_frame, text="Image Viewer", padding=10)
        viewer_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        image_container = ttk.Frame(viewer_frame)
        image_container.pack(fill=tk.BOTH, expand=True)

        canvas_frame = ttk.Frame(image_container)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.image_canvas = tk.Canvas(canvas_frame, bg='lightgray', width=500, height=400)

        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.image_canvas.yview)
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.image_canvas.xview)

        self.image_canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        self.image_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        image_info_frame = ttk.Frame(viewer_frame)
        image_info_frame.pack(fill=tk.X, pady=5)

        self.image_info_label = ttk.Label(image_info_frame, text="No image loaded", font=("TkDefaultFont", 10))
        self.image_info_label.pack(side=tk.LEFT)

        zoom_frame = ttk.Frame(image_info_frame)
        zoom_frame.pack(side=tk.RIGHT)

        ttk.Button(zoom_frame, text="🔍+", command=self.zoom_in).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="🔍-", command=self.zoom_out).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="🔄", command=self.reset_zoom).pack(side=tk.LEFT, padx=2)

        # Bind events
        self.current_score.trace_add('write', self.update_score_display)
        self.image_canvas.bind("<Button-1>", self.on_image_click)
        self.image_canvas.bind("<MouseWheel>", self.on_mouse_wheel)

        # Keyboard shortcuts for faster workflow
        self.root.bind("<KeyPress-Left>", self.on_key_left)
        self.root.bind("<KeyPress-Right>", self.on_key_right)
        self.root.bind("<KeyPress-space>", self.on_key_space)
        self.root.bind("<KeyPress-Return>", self.on_key_enter)
        self.root.bind("<KeyPress-0>", self.on_key_0)
        self.root.bind("<KeyPress-1>", self.on_key_1)
        self.root.bind("<KeyPress-2>", self.on_key_2)
        self.root.bind("<KeyPress-3>", self.on_key_3)
        self.root.bind("<KeyPress-4>", self.on_key_4)
        self.root.bind("<KeyPress-5>", self.on_key_5)
        self.root.bind("<KeyPress-6>", self.on_key_6)
        self.root.bind("<KeyPress-7>", self.on_key_7)
        self.root.bind("<KeyPress-8>", self.on_key_8)
        self.root.bind("<KeyPress-plus>", self.on_key_plus)
        self.root.bind("<KeyPress-minus>", self.on_key_minus)

        # Keyboard shortcut help
        shortcut_frame = ttk.LabelFrame(left_frame, text="⌨️ Keyboard Shortcuts", padding=5)
        shortcut_frame.pack(fill=tk.X, padx=5, pady=5)

        shortcuts_text = """← → : Previous/Next Image    Space/Enter : Score & Next
0: Zero (0)  1: Poor (10)  4: Fair (50)  5: Good (70)  7: Excellent (90)
2: Too Low (20)  3: Barely Fair (41)  6: Pretty Good (79)  8: Perfect (100)
+/- : Zoom In/Out"""

        ttk.Label(shortcut_frame, text=shortcuts_text, font=("TkDefaultFont", 8),
                  foreground="gray").pack()

        self.update_score_display()

    # ---------------------------------------------------------------
    # Keyboard event handlers
    # ---------------------------------------------------------------
    def on_key_left(self, event):
        self.previous_image()
        return "break"

    def on_key_right(self, event):
        self.next_image()
        return "break"

    def on_key_space(self, event):
        if self.image_files:
            self.score_and_next()
        return "break"

    def on_key_enter(self, event):
        if self.image_files:
            self.score_and_next()
        return "break"

    def on_key_0(self, event):
        if self.image_files:
            self.quick_score_and_next(0)
        return "break"

    def on_key_1(self, event):
        if self.image_files:
            self.quick_score_and_next(10)
        return "break"

    def on_key_2(self, event):
        if self.image_files:
            self.quick_score_and_next(20)
        return "break"

    def on_key_3(self, event):
        if self.image_files:
            self.quick_score_and_next(41)
        return "break"

    def on_key_4(self, event):
        if self.image_files:
            self.quick_score_and_next(50)
        return "break"

    def on_key_5(self, event):
        if self.image_files:
            self.quick_score_and_next(70)
        return "break"

    def on_key_6(self, event):
        if self.image_files:
            self.quick_score_and_next(79)
        return "break"

    def on_key_7(self, event):
        if self.image_files:
            self.quick_score_and_next(90)
        return "break"

    def on_key_8(self, event):
        if self.image_files:
            self.quick_score_and_next(100)
        return "break"

    def on_key_plus(self, event):
        if self.image_files:
            self.zoom_in()
        return "break"

    def on_key_minus(self, event):
        if self.image_files:
            self.zoom_out()
        return "break"

    # ---------------------------------------------------------------
    # Folder/browse helpers
    # ---------------------------------------------------------------
    def browse_source_images(self):
        folder = filedialog.askdirectory(
            title="Select Source Images Folder",
            initialdir=self.source_images_folder.get() or os.path.expanduser("~")
        )
        if folder:
            self.source_images_folder.set(folder)
            self.log_dataset_message(f"📁 Source folder selected: {folder}")

    def browse_dataset_output(self):
        folder = filedialog.askdirectory(
            title="Select Dataset Output Folder",
            initialdir=self.dataset_output_folder.get() or os.path.expanduser("~")
        )
        if folder:
            self.dataset_output_folder.set(folder)
            self.log_dataset_message(f"📂 Output folder selected: {folder}")

    # ---------------------------------------------------------------
    # Image loading
    # ---------------------------------------------------------------
    def load_source_images(self):
        source_folder = self.source_images_folder.get()
        if not source_folder:
            messagebox.showerror("Error", "Please select a source images folder first.")
            return

        if not os.path.exists(source_folder):
            messagebox.showerror("Error", "Source folder does not exist.")
            return

        try:
            jpeg_extensions = ['.jpg', '.jpeg']
            self.image_files = []

            self.log_dataset_message("🔍 Scanning for JPEG images...")
            for ext in jpeg_extensions:
                pattern = os.path.join(source_folder, "**", f"*{ext}")
                self.image_files.extend(glob.glob(pattern, recursive=True))
                pattern = os.path.join(source_folder, "**", f"*{ext.upper()}")
                self.image_files.extend(glob.glob(pattern, recursive=True))

            self.image_files = sorted(list(set(self.image_files)))

            if not self.image_files:
                messagebox.showwarning("No Images", "No JPEG image files found in the selected folder.")
                return

            if len(self.image_files) > 1000:
                if not messagebox.askyesno(
                    "Large Dataset Warning",
                    f"Found {len(self.image_files)} images! This may be slow to navigate.\n\n"
                    f"Consider using 'Random Sample' for faster scoring.\n\n"
                    f"Continue loading all images?"
                ):
                    return

            self.current_image_index = 0
            self.processed_images = {}

            if self.dataset_output_folder.get():
                self.initialize_csv_file()

            self.log_dataset_message(f"🖼️ Loaded {len(self.image_files)} images")
            self.update_progress_info()
            self.display_current_image()

        except Exception as e:
            self.log_dataset_message(f"❌ Error loading images: {str(e)}")
            messagebox.showerror("Error", f"Failed to load images: {str(e)}")

    def load_random_sample(self):
        source_folder = self.source_images_folder.get()
        if not source_folder:
            messagebox.showerror("Error", "Please select a source images folder first.")
            return

        if not os.path.exists(source_folder):
            messagebox.showerror("Error", "Source folder does not exist.")
            return

        sample_size = self.sample_size.get()
        if sample_size <= 0:
            messagebox.showerror("Error", "Sample size must be greater than 0.")
            return

        try:
            jpeg_extensions = ['.jpg', '.jpeg']
            all_image_files = []

            self.log_dataset_message("🔍 Scanning for JPEG images...")
            for ext in jpeg_extensions:
                pattern = os.path.join(source_folder, "**", f"*{ext}")
                all_image_files.extend(glob.glob(pattern, recursive=True))
                pattern = os.path.join(source_folder, "**", f"*{ext.upper()}")
                all_image_files.extend(glob.glob(pattern, recursive=True))

            all_image_files = list(set(all_image_files))

            if not all_image_files:
                messagebox.showwarning("No Images", "No JPEG image files found in the selected folder.")
                return

            self.log_dataset_message(f"📊 Found {len(all_image_files)} total JPEG images")

            if sample_size > len(all_image_files):
                if messagebox.askyesno(
                    "Sample Size Too Large",
                    f"Requested sample size ({sample_size}) is larger than available images ({len(all_image_files)}).\n\n"
                    f"Load all {len(all_image_files)} images instead?"
                ):
                    sample_size = len(all_image_files)
                else:
                    return

            random.seed()  # Use current time as seed for true randomness
            self.image_files = random.sample(all_image_files, sample_size)
            self.image_files.sort()

            self.current_image_index = 0
            self.processed_images = {}

            if self.dataset_output_folder.get():
                self.initialize_csv_file()

            self.log_dataset_message(f"🎲 Randomly selected {len(self.image_files)} images from {len(all_image_files)} total")
            self.update_progress_info()
            self.display_current_image()

        except Exception as e:
            self.log_dataset_message(f"❌ Error loading random sample: {str(e)}")
            messagebox.showerror("Error", f"Failed to load random sample: {str(e)}")

    # ---------------------------------------------------------------
    # Image display
    # ---------------------------------------------------------------
    def display_current_image(self):
        if not self.image_files or self.current_image_index >= len(self.image_files):
            return

        try:
            image_path = self.image_files[self.current_image_index]
            self.selected_image_path = image_path

            pil_image = Image.open(image_path)

            canvas_width = self.image_canvas.winfo_width() or 500
            canvas_height = self.image_canvas.winfo_height() or 400

            display_size = self.calculate_display_size(pil_image.size, canvas_width, canvas_height)
            display_size = (int(display_size[0] * self.zoom_factor),
                             int(display_size[1] * self.zoom_factor))

            display_image = pil_image.resize(display_size, Image.Resampling.LANCZOS)
            self.preview_image = ImageTk.PhotoImage(display_image)

            self.image_canvas.delete("all")

            canvas_center_x = canvas_width // 2
            canvas_center_y = canvas_height // 2
            self.image_canvas.create_image(canvas_center_x, canvas_center_y,
                                            image=self.preview_image, anchor=tk.CENTER)

            self.image_canvas.configure(scrollregion=self.image_canvas.bbox("all"))

            filename = os.path.basename(image_path)
            progress_text = f"Image {self.current_image_index + 1} of {len(self.image_files)}: {filename}"
            self.image_info_label.config(text=progress_text)

            if image_path in self.processed_images:
                score = self.processed_images[image_path]
                self.current_score.set(score)
                self.log_dataset_message(f"🎯 Previously scored: {filename} = {score}")
            else:
                self.current_score.set(50)

        except Exception as e:
            self.log_dataset_message(f"❌ Error displaying image: {str(e)}")

    def calculate_display_size(self, image_size, canvas_width, canvas_height):
        img_width, img_height = image_size

        scale_x = (canvas_width - 20) / img_width
        scale_y = (canvas_height - 20) / img_height

        scale = min(scale_x, scale_y, 1.0)

        return (int(img_width * scale), int(img_height * scale))

    # ---------------------------------------------------------------
    # Scoring
    # ---------------------------------------------------------------
    def set_quick_score(self, score):
        self.current_score.set(score)

    def quick_score_and_next(self, score):
        if not self.image_files:
            return
        self.current_score.set(score)
        self.score_and_next()

    def update_score_display(self, *args):
        score = self.current_score.get()

        if score >= 90:
            category = "Excellent"
            color = "darkgreen"
        elif score >= 70:
            category = "Good"
            color = "green"
        elif score >= 40:
            category = "Fair"
            color = "orange"
        elif score > 5:
            category = "Poor"
            color = "red"
        else:
            category = "Zero"
            color = "darkred"

        self.score_display.config(text=f"Score: {score} ({category})", foreground=color)

    def previous_image(self):
        if self.image_files and self.current_image_index > 0:
            self.current_image_index -= 1
            self.display_current_image()
            self.update_progress_info()

    def next_image(self):
        if self.image_files and self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.display_current_image()
            self.update_progress_info()

    def score_and_next(self):
        if not self.image_files:
            return

        try:
            image_path = self.image_files[self.current_image_index]
            score = self.current_score.get()
            filename = os.path.basename(image_path)

            self.log_dataset_message(f"🎯 Applying score {score} to image {self.current_image_index + 1}/{len(self.image_files)}")

            self.processed_images[image_path] = score

            self.copy_image_to_category(image_path, score)

            if score >= 90:
                category = "excellent"
            elif score >= 70:
                category = "good"
            elif score >= 40:
                category = "fair"
            elif score > 0:
                category = "poor"
            else:
                category = "zero"

            csv_filename = f"{category}/{filename}"
            self.append_to_csv(csv_filename, score)

            self.update_progress_info()

            self.score_display.config(background="lightgreen")
            self.root.after(300, lambda: self.score_display.config(background="SystemButtonFace"))

            self.log_dataset_message(f"✅ Scored: {filename} = {score} (saved to CSV)")

            self.next_image()

        except Exception as e:
            self.log_dataset_message(f"❌ Error scoring image: {str(e)}")
            messagebox.showerror("Error", f"Failed to score image: {str(e)}")

    def copy_image_to_category(self, image_path, score):
        output_folder = self.dataset_output_folder.get()
        if not output_folder:
            messagebox.showerror("Error", "Please select a dataset output folder.")
            return

        if score >= 90:
            category = "excellent"
        elif score >= 70:
            category = "good"
        elif score >= 40:
            category = "fair"
        elif score > 0:
            category = "poor"
        else:
            category = "zero"

        category_folder = os.path.join(output_folder, "images", category)
        os.makedirs(category_folder, exist_ok=True)

        filename = os.path.basename(image_path)
        dest_path = os.path.join(category_folder, filename)

        counter = 1
        original_dest = dest_path
        while os.path.exists(dest_path):
            name, ext = os.path.splitext(original_dest)
            dest_path = f"{name}_{counter}{ext}"
            counter += 1

        shutil.copy2(image_path, dest_path)
        self.log_dataset_message(f"📁 Copied to {category}: {filename}")

    # ---------------------------------------------------------------
    # CSV handling
    # ---------------------------------------------------------------
    def export_dataset_csv(self):
        if not self.processed_images:
            messagebox.showwarning("No Data", "No images have been scored yet.")
            return

        output_folder = self.dataset_output_folder.get()
        if not output_folder:
            messagebox.showerror("Error", "Please select a dataset output folder.")
            return

        try:
            csv_path = os.path.join(output_folder, "labels.csv")

            if os.path.exists(csv_path):
                with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)
                    existing_data = list(reader)

                existing_data.sort(key=lambda x: x['filename'])

                with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                    fieldnames = ['filename', 'visibility_score', 'evaluator_count', 'std_dev']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(existing_data)

                self.log_dataset_message(f"💾 Finalized CSV with {len(existing_data)} entries (sorted by filename)")
                messagebox.showinfo("Export Complete",
                                     f"CSV file already exists with real-time updates!\n"
                                     f"Finalized {len(existing_data)} scored images at:\n{csv_path}\n\n"
                                     f"Your progress was automatically saved during scoring.")

            else:
                csv_data = []
                for image_path, score in self.processed_images.items():
                    filename = os.path.basename(image_path)

                    if score >= 90:
                        category = "excellent"
                    elif score >= 70:
                        category = "good"
                    elif score >= 40:
                        category = "fair"
                    elif score > 0:
                        category = "poor"
                    else:
                        category = "zero"

                    csv_filename = f"{category}/{filename}"
                    csv_data.append({
                        'filename': csv_filename,
                        'visibility_score': score,
                        'evaluator_count': 1,
                        'std_dev': 0.0
                    })

                csv_data.sort(key=lambda x: x['filename'])

                with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                    fieldnames = ['filename', 'visibility_score', 'evaluator_count', 'std_dev']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(csv_data)

                self.log_dataset_message(f"💾 Created CSV with {len(csv_data)} entries")
                messagebox.showinfo("Export Complete",
                                     f"Successfully created CSV with {len(csv_data)} scored images at:\n{csv_path}")

        except Exception as e:
            self.log_dataset_message(f"❌ Error finalizing CSV: {str(e)}")
            messagebox.showerror("Export Error", f"Failed to finalize CSV: {str(e)}")

    def initialize_csv_file(self):
        output_folder = self.dataset_output_folder.get()
        if not output_folder:
            return None

        csv_path = os.path.join(output_folder, "labels.csv")

        try:
            if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
                with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                    fieldnames = ['filename', 'visibility_score', 'evaluator_count', 'std_dev']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()

                self.log_dataset_message(f"📄 Initialized labels.csv at: {csv_path}")

            return csv_path

        except Exception as e:
            self.log_dataset_message(f"❌ Error initializing CSV: {str(e)}")
            return None

    def append_to_csv(self, filename, score):
        output_folder = self.dataset_output_folder.get()
        if not output_folder:
            return

        csv_path = os.path.join(output_folder, "labels.csv")

        try:
            if not os.path.exists(csv_path):
                self.initialize_csv_file()

            existing_entries = set()
            if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
                with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)
                    existing_entries = {row['filename'] for row in reader}

            if filename not in existing_entries:
                with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                    fieldnames = ['filename', 'visibility_score', 'evaluator_count', 'std_dev']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                    writer.writerow({
                        'filename': filename,
                        'visibility_score': score,
                        'evaluator_count': 1,
                        'std_dev': 0.0
                    })

                self.log_dataset_message(f"💾 Added to CSV: {filename} = {score}")
            else:
                self.update_csv_entry(filename, score)

        except Exception as e:
            self.log_dataset_message(f"❌ Error appending to CSV: {str(e)}")

    def update_csv_entry(self, filename, new_score):
        output_folder = self.dataset_output_folder.get()
        if not output_folder:
            return

        csv_path = os.path.join(output_folder, "labels.csv")

        try:
            with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                rows = list(reader)

            for row in rows:
                if row['filename'] == filename:
                    row['visibility_score'] = new_score
                    break

            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['filename', 'visibility_score', 'evaluator_count', 'std_dev']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            self.log_dataset_message(f"🔄 Updated CSV: {filename} = {new_score}")

        except Exception as e:
            self.log_dataset_message(f"❌ Error updating CSV entry: {str(e)}")

    # ---------------------------------------------------------------
    # Dataset structure
    # ---------------------------------------------------------------
    def create_dataset_structure(self):
        if self.dataset_output_folder.get():
            dataset_path = self.dataset_output_folder.get()
        else:
            parent_dir = filedialog.askdirectory(
                title="Select Parent Directory for Training Dataset",
                initialdir=os.path.expanduser("~")
            )

            if not parent_dir:
                return

            dataset_path = os.path.join(parent_dir, "training_dataset")

        try:
            folders_to_create = [
                "images",
                "images/excellent",
                "images/good",
                "images/fair",
                "images/poor",
                "images/very_poor",
                "validation",
                "validation/images"
            ]

            for folder in folders_to_create:
                folder_path = os.path.join(dataset_path, folder)
                os.makedirs(folder_path, exist_ok=True)
                self.log_dataset_message(f"✅ Created folder: {folder}")

            labels_csv_path = os.path.join(dataset_path, "labels.csv")
            sample_labels_content = """filename,visibility_score,evaluator_count,std_dev
# Sample entries - replace with your actual data
# img001.jpg,78.5,12,8.2
# img002.jpg,45.3,15,12.1
# img003.jpg,92.1,10,5.5
"""

            with open(labels_csv_path, 'w', encoding='utf-8') as f:
                f.write(sample_labels_content)
            self.log_dataset_message("✅ Created sample labels.csv")

            val_labels_csv_path = os.path.join(dataset_path, "validation", "labels.csv")
            with open(val_labels_csv_path, 'w', encoding='utf-8') as f:
                f.write(sample_labels_content)
            self.log_dataset_message("✅ Created validation labels.csv")

            readme_path = os.path.join(dataset_path, "README.md")
            readme_content = """# Training Dataset Structure

This folder contains the expected structure for training the underwater image visibility model.

## Folder Structure:
```
training_dataset/
├── images/
│   ├── excellent/          # Visibility scores 90-100
│   ├── good/              # Visibility scores 70-89
│   ├── fair/              # Visibility scores 40-69
│   ├── poor/              # Visibility scores 15-39
│   └── very_poor/         # Visibility scores 0-14
├── labels.csv             # Image filename -> MOS score mapping
└── validation/            # Hold-out validation set
    ├── images/
    └── labels.csv
```

## Instructions:

1. **Organize Images by Quality:**
   - Place excellent images (visibility 90-100) in `images/excellent/`
   - Place good images (visibility 70-89) in `images/good/`
   - Place fair images (visibility 40-69) in `images/fair/`
   - Place poor images (visibility 15-39) in `images/poor/`
   - Place very poor images (visibility 0-14) in `images/very_poor/`

2. **Update labels.csv:**
   - Edit the `labels.csv` file with your actual image filenames and scores
   - Format: `filename,visibility_score,evaluator_count,std_dev`
   - Use Mean Opinion Score (MOS) from human evaluators

3. **Validation Set:**
   - Place 20% of your images in `validation/images/`
   - Update `validation/labels.csv` accordingly

4. **Quality Control:**
   - Remove images with high standard deviation (>20)
   - Ensure balanced representation across all visibility ranges
   - Have 10-30 evaluators rate each image on 0-100 scale
"""

            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(readme_content)
            self.log_dataset_message("✅ Created README.md with instructions")

            self.dataset_output_folder.set(dataset_path)

            self.log_dataset_message("🎉 Training dataset structure created successfully!")
            self.log_dataset_message(f"📁 Location: {dataset_path}")

            messagebox.showinfo(
                "Dataset Structure Created",
                f"Training dataset structure created successfully!\n\n"
                f"Location: {dataset_path}\n\n"
                f"Next steps:\n"
                f"1. Load images above and score them\n"
                f"2. Or manually add images to category folders"
            )

        except Exception as e:
            error_msg = f"Error creating dataset structure: {str(e)}"
            self.log_dataset_message(f"❌ {error_msg}")
            messagebox.showerror("Error", error_msg)

    # ---------------------------------------------------------------
    # Zoom / misc
    # ---------------------------------------------------------------
    def zoom_in(self):
        self.zoom_factor = min(self.zoom_factor * 1.2, 5.0)
        self.display_current_image()

    def zoom_out(self):
        self.zoom_factor = max(self.zoom_factor / 1.2, 0.1)
        self.display_current_image()

    def reset_zoom(self):
        self.zoom_factor = 1.0
        self.display_current_image()

    def on_image_click(self, event):
        pass

    def on_mouse_wheel(self, event):
        if event.delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    def log_dataset_message(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        full_message = f"[{timestamp}] {message}\n"

        self.dataset_status.insert(tk.END, full_message)
        self.dataset_status.see(tk.END)
        self.root.update_idletasks()

    def update_progress_info(self):
        if not self.image_files:
            return

        total_images = len(self.image_files)
        scored_images = len(self.processed_images)
        current_index = self.current_image_index + 1

        progress_pct = (scored_images / total_images) * 100 if total_images > 0 else 0

        progress_text = (f"Progress: {scored_images}/{total_images} images scored ({progress_pct:.1f}%) | "
                          f"Current: {current_index}/{total_images} | ✅ CSV auto-saved")
        self.progress_info.config(text=progress_text)


def main():
    """Main entry point for the standalone dataset setup tool."""
    root = tk.Tk()

    try:
        root.tk.call("source", "azure.tcl")
        root.tk.call("set_theme", "light")
    except Exception:
        pass  # Use default theme if azure theme not available

    app = DatasetSetupApp(root)

    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")

    root.mainloop()


if __name__ == "__main__":
    main()
