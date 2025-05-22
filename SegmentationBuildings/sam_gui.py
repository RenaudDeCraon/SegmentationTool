"""
GUI for SAM Building Segmentation with Parameter Presets
A user-friendly interface for segmenting buildings from TIFF images using Meta's SAM model.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import threading
import numpy as np
import rasterio
from rasterio.plot import reshape_as_image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

class SAMSegmentationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SAM Building Segmentation Tool")
        self.root.geometry("750x600")
        self.root.resizable(True, True)
        
        # Variables
        self.input_image_path = tk.StringVar()
        self.output_directory = tk.StringVar()
        self.output_filename = tk.StringVar(value="building_segmentation")
        self.model_type = tk.StringVar(value="vit_b")
        self.checkpoint_path = tk.StringVar()
        self.score_thresh = tk.DoubleVar(value=0.85)
        self.min_area = tk.IntVar(value=500)
        self.max_area = tk.IntVar(value=50000)
        self.min_aspect = tk.DoubleVar(value=0.5)
        self.max_aspect = tk.DoubleVar(value=2.0)
        
        # Parameter presets
        self.presets = {
            "Urban": {
                "score_thresh": 0.85,
                "min_area": 500,
                "max_area": 55000,
                "min_aspect": 0.55,
                "max_aspect": 2.15
            },
            "Rural": {
                "score_thresh": 0.775,
                "min_area": 1000,
                "max_area": 100000,
                "min_aspect": 0.45,
                "max_aspect": 3.0
            },
            "Industrial": {
                "score_thresh": 0.8,
                "min_area": 2000,
                "max_area": 300000,
                "min_aspect": 0.35,
                "max_aspect": 5.5
            }
        }
        
        # Set default checkpoint path if it exists
        if os.path.exists("sam_vit_b_01ec64.pth"):
            self.checkpoint_path.set("sam_vit_b_01ec64.pth")
        
        self.setup_scrollable_ui()
        
    def setup_scrollable_ui(self):
        # Create main canvas and scrollbar
        self.canvas = tk.Canvas(self.root, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        # Configure scrollable frame
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        # Create window in canvas
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        
        # Configure canvas scrolling
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Pack canvas and scrollbar
        self.canvas.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=10)
        self.scrollbar.pack(side="right", fill="y", padx=(0, 10), pady=10)
        
        # Bind mousewheel to canvas
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.root.bind("<MouseWheel>", self._on_mousewheel)
        
        # Bind canvas resize
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        
        # Set up the UI content
        self.setup_ui_content()
        
        # Configure focus and key bindings for scrolling
        self.canvas.focus_set()
        self.root.bind("<Up>", lambda e: self.canvas.yview_scroll(-1, "units"))
        self.root.bind("<Down>", lambda e: self.canvas.yview_scroll(1, "units"))
        self.root.bind("<Prior>", lambda e: self.canvas.yview_scroll(-1, "pages"))
        self.root.bind("<Next>", lambda e: self.canvas.yview_scroll(1, "pages"))
        
    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
    def _on_canvas_configure(self, event):
        # Update the canvas window width to match canvas width
        canvas_width = event.width
        self.canvas.itemconfig(self.canvas_window, width=canvas_width)
        
    def setup_ui_content(self):
        # Title
        title_label = ttk.Label(self.scrollable_frame, text="SAM Building Segmentation Tool", 
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Input Image Section
        input_frame = ttk.LabelFrame(self.scrollable_frame, text="Input Image", padding=10)
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(input_frame, text="Select TIFF Image:").pack(anchor=tk.W)
        input_path_frame = ttk.Frame(input_frame)
        input_path_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Entry(input_path_frame, textvariable=self.input_image_path, width=60).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(input_path_frame, text="Browse", command=self.browse_input_image).pack(side=tk.RIGHT, padx=(5, 0))
        
        # SAM Model Section
        model_frame = ttk.LabelFrame(self.scrollable_frame, text="SAM Model Configuration", padding=10)
        model_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Model type
        model_type_frame = ttk.Frame(model_frame)
        model_type_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(model_type_frame, text="Model Type:").pack(side=tk.LEFT)
        model_combo = ttk.Combobox(model_type_frame, textvariable=self.model_type, 
                                  values=["vit_b", "vit_h"], state="readonly", width=10)
        model_combo.pack(side=tk.LEFT, padx=(10, 0))
        
        # Checkpoint path
        ttk.Label(model_frame, text="Model Checkpoint:").pack(anchor=tk.W, pady=(10, 0))
        checkpoint_frame = ttk.Frame(model_frame)
        checkpoint_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Entry(checkpoint_frame, textvariable=self.checkpoint_path, width=60).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(checkpoint_frame, text="Browse", command=self.browse_checkpoint).pack(side=tk.RIGHT, padx=(5, 0))
        
        # Output Section
        output_frame = ttk.LabelFrame(self.scrollable_frame, text="Output Configuration", padding=10)
        output_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Output directory
        ttk.Label(output_frame, text="Output Directory:").pack(anchor=tk.W)
        output_dir_frame = ttk.Frame(output_frame)
        output_dir_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Entry(output_dir_frame, textvariable=self.output_directory, width=60).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(output_dir_frame, text="Browse", command=self.browse_output_directory).pack(side=tk.RIGHT, padx=(5, 0))
        
        # Output filename
        filename_frame = ttk.Frame(output_frame)
        filename_frame.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(filename_frame, text="Output Filename (without extension):").pack(side=tk.LEFT)
        ttk.Entry(filename_frame, textvariable=self.output_filename, width=30).pack(side=tk.RIGHT)
        
        # Parameter Presets Section
        preset_frame = ttk.LabelFrame(self.scrollable_frame, text="Parameter Presets", padding=10)
        preset_frame.pack(fill=tk.X, pady=(0, 10))
        
        preset_info = ttk.Label(preset_frame, text="Quick parameter settings for different area types:", 
                               font=("Arial", 9, "italic"))
        preset_info.pack(anchor=tk.W, pady=(0, 10))
        
        preset_buttons_frame = ttk.Frame(preset_frame)
        preset_buttons_frame.pack(fill=tk.X)
        
        # Urban preset button
        urban_btn = ttk.Button(preset_buttons_frame, text="Urban Areas\n(Dense Buildings)", 
                              command=lambda: self.apply_preset("Urban"))
        urban_btn.pack(side=tk.LEFT, padx=(0, 10), fill=tk.X, expand=True)
        
        # Rural preset button
        rural_btn = ttk.Button(preset_buttons_frame, text="Rural Areas\n(Scattered Buildings)", 
                              command=lambda: self.apply_preset("Rural"))
        rural_btn.pack(side=tk.LEFT, padx=(0, 10), fill=tk.X, expand=True)
        
        # Industrial preset button
        industrial_btn = ttk.Button(preset_buttons_frame, text="Industrial Areas\n(Large Buildings)", 
                                   command=lambda: self.apply_preset("Industrial"))
        industrial_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Current preset display
        self.current_preset_var = tk.StringVar(value="Custom Settings")
        preset_status = ttk.Label(preset_frame, textvariable=self.current_preset_var, 
                                 font=("Arial", 9, "bold"), foreground="blue")
        preset_status.pack(pady=(10, 0))
        
        # Parameters Section
        params_frame = ttk.LabelFrame(self.scrollable_frame, text="Segmentation Parameters", padding=10)
        params_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Create a grid for parameters
        params_grid = ttk.Frame(params_frame)
        params_grid.pack(fill=tk.X)
        
        # Score threshold
        ttk.Label(params_grid, text="Score Threshold:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        score_scale = ttk.Scale(params_grid, from_=0.1, to=1.0, variable=self.score_thresh, 
                               orient=tk.HORIZONTAL, length=200, command=self.on_parameter_change)
        score_scale.grid(row=0, column=1, sticky=tk.W)
        self.score_label = ttk.Label(params_grid, text="0.85")
        self.score_label.grid(row=0, column=2, padx=(10, 0))
        
        # Min area
        ttk.Label(params_grid, text="Min Area (pixels):").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        min_area_entry = ttk.Entry(params_grid, textvariable=self.min_area, width=10)
        min_area_entry.grid(row=1, column=1, sticky=tk.W, pady=(10, 0))
        min_area_entry.bind('<KeyRelease>', self.on_parameter_change)
        
        # Max area
        ttk.Label(params_grid, text="Max Area (pixels):").grid(row=2, column=0, sticky=tk.W, padx=(0, 10), pady=(5, 0))
        max_area_entry = ttk.Entry(params_grid, textvariable=self.max_area, width=10)
        max_area_entry.grid(row=2, column=1, sticky=tk.W, pady=(5, 0))
        max_area_entry.bind('<KeyRelease>', self.on_parameter_change)
        
        # Min aspect ratio
        ttk.Label(params_grid, text="Min Aspect Ratio:").grid(row=3, column=0, sticky=tk.W, padx=(0, 10), pady=(5, 0))
        min_aspect_entry = ttk.Entry(params_grid, textvariable=self.min_aspect, width=10)
        min_aspect_entry.grid(row=3, column=1, sticky=tk.W, pady=(5, 0))
        min_aspect_entry.bind('<KeyRelease>', self.on_parameter_change)
        
        # Max aspect ratio
        ttk.Label(params_grid, text="Max Aspect Ratio:").grid(row=4, column=0, sticky=tk.W, padx=(0, 10), pady=(5, 0))
        max_aspect_entry = ttk.Entry(params_grid, textvariable=self.max_aspect, width=10)
        max_aspect_entry.grid(row=4, column=1, sticky=tk.W, pady=(5, 0))
        max_aspect_entry.bind('<KeyRelease>', self.on_parameter_change)
        
        # Progress Section
        progress_frame = ttk.LabelFrame(self.scrollable_frame, text="Progress", padding=10)
        progress_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.progress_var = tk.StringVar(value="Ready to segment...")
        self.progress_label = ttk.Label(progress_frame, textvariable=self.progress_var)
        self.progress_label.pack(anchor=tk.W)
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.progress_bar.pack(fill=tk.X, pady=(5, 0))
        
        # Buttons
        button_frame = ttk.Frame(self.scrollable_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.segment_button = ttk.Button(button_frame, text="Start Segmentation", 
                                        command=self.start_segmentation, style="Accent.TButton")
        self.segment_button.pack(side=tk.LEFT)
        
        ttk.Button(button_frame, text="Reset", command=self.reset_form).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(button_frame, text="Exit", command=self.root.quit).pack(side=tk.RIGHT)
        
        # Scroll instruction
        scroll_info = ttk.Label(self.scrollable_frame, 
                               text="Use mouse wheel, arrow keys, or Page Up/Down to scroll", 
                               font=("Arial", 8, "italic"))
        scroll_info.pack(pady=(10, 20))
        
        # Configure the score threshold label to update
        self.score_thresh.trace('w', self.update_score_label)
        self.update_score_label()
        
    def apply_preset(self, preset_name):
        """Apply parameter preset for different area types"""
        preset = self.presets[preset_name]
        
        self.score_thresh.set(preset["score_thresh"])
        self.min_area.set(preset["min_area"])
        self.max_area.set(preset["max_area"])
        self.min_aspect.set(preset["min_aspect"])
        self.max_aspect.set(preset["max_aspect"])
        
        self.current_preset_var.set(f"Current: {preset_name} Preset")
        self.update_score_label()
        
        # Show info about the applied preset
        messagebox.showinfo("Preset Applied", 
                           f"{preset_name} area parameters applied!\n\n"
                           f"Score Threshold: {preset['score_thresh']}\n"
                           f"Min Area: {preset['min_area']:,} pixels\n"
                           f"Max Area: {preset['max_area']:,} pixels\n"
                           f"Min Aspect Ratio: {preset['min_aspect']}\n"
                           f"Max Aspect Ratio: {preset['max_aspect']}")
    
    def on_parameter_change(self, *args):
        """Called when parameters are manually changed"""
        self.current_preset_var.set("Custom Settings")
        
    def update_score_label(self, *args):
        """Update the score threshold display"""
        self.score_label.config(text=f"{self.score_thresh.get():.2f}")
        
    def browse_input_image(self):
        filename = filedialog.askopenfilename(
            title="Select Input Image",
            filetypes=[
                ("TIFF files", "*.tif *.tiff"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.input_image_path.set(filename)
            
    def browse_checkpoint(self):
        filename = filedialog.askopenfilename(
            title="Select SAM Checkpoint",
            filetypes=[
                ("PyTorch files", "*.pth"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.checkpoint_path.set(filename)
            
    def browse_output_directory(self):
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_directory.set(directory)
            
    def reset_form(self):
        self.input_image_path.set("")
        self.output_directory.set("")
        self.output_filename.set("building_segmentation")
        self.score_thresh.set(0.85)
        self.min_area.set(500)
        self.max_area.set(50000)
        self.min_aspect.set(0.5)
        self.max_aspect.set(2.0)
        self.progress_var.set("Ready to segment...")
        self.current_preset_var.set("Custom Settings")
        
    def validate_inputs(self):
        if not self.input_image_path.get():
            messagebox.showerror("Error", "Please select an input image.")
            return False
            
        if not os.path.exists(self.input_image_path.get()):
            messagebox.showerror("Error", "Input image file does not exist.")
            return False
            
        if not self.checkpoint_path.get():
            messagebox.showerror("Error", "Please select a SAM checkpoint file.")
            return False
            
        if not os.path.exists(self.checkpoint_path.get()):
            messagebox.showerror("Error", "Checkpoint file does not exist.")
            return False
            
        if not self.output_directory.get():
            messagebox.showerror("Error", "Please select an output directory.")
            return False
            
        if not os.path.exists(self.output_directory.get()):
            messagebox.showerror("Error", "Output directory does not exist.")
            return False
            
        if not self.output_filename.get().strip():
            messagebox.showerror("Error", "Please provide an output filename.")
            return False
            
        return True
        
    def start_segmentation(self):
        if not self.validate_inputs():
            return
            
        # Disable the button and start progress
        self.segment_button.config(state="disabled")
        self.progress_bar.start()
        
        # Run segmentation in a separate thread
        thread = threading.Thread(target=self.run_segmentation)
        thread.daemon = True
        thread.start()
        
    def run_segmentation(self):
        try:
            self.update_progress("Loading image...")
            
            # Load image
            with rasterio.open(self.input_image_path.get()) as dataset:
                meta = dataset.meta.copy()
                img_array = dataset.read()
                img = reshape_as_image(img_array)
                
                if img.shape[2] > 3:
                    img = img[:, :, :3]
                elif img.shape[2] == 1:
                    img = np.repeat(img, 3, axis=2)
            
            # Normalize image
            if img.dtype != np.uint8:
                img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
            
            img_rgb = img[:, :, :3].copy()
            
            self.update_progress("Loading SAM model...")
            
            # Load SAM model
            sam = sam_model_registry[self.model_type.get()](checkpoint=self.checkpoint_path.get())
            mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=32,
                pred_iou_thresh=0.9,
                stability_score_thresh=0.95,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=self.min_area.get(),
            )
            
            self.update_progress("Generating masks...")
            
            # Generate masks
            masks = mask_generator.generate(img_rgb)
            
            self.update_progress(f"Processing {len(masks)} masks...")
            
            # Build label map
            H, W = img.shape[:2]
            label_map = np.zeros((H, W), dtype=np.uint8)
            
            building_count = 0
            for mask_data in masks:
                mask = mask_data['segmentation']
                score = mask_data['predicted_iou']
                bbox = mask_data['bbox']
                
                if score < self.score_thresh.get():
                    continue
                
                area = mask.sum()
                x, y, w, h = bbox
                aspect = w / h if h > 0 else 0
                
                if not (self.min_area.get() <= area <= self.max_area.get() and 
                       self.min_aspect.get() <= aspect <= self.max_aspect.get()):
                    continue
                
                label_map[mask] = 1
                building_count += 1
            
            self.update_progress(f"Found {building_count} buildings. Saving results...")
            
            # Prepare output paths
            base_filename = self.output_filename.get().strip()
            output_dir = self.output_directory.get()
            
            label_path = os.path.join(output_dir, f"{base_filename}_labels.tiff")
            overlay_path = os.path.join(output_dir, f"{base_filename}_overlay.tiff")
            
            # Save label map
            label_meta = meta.copy()
            label_meta.update({'count': 1, 'dtype': 'uint8', 'nodata': 0})
            
            with rasterio.open(label_path, 'w', **label_meta) as dst:
                dst.write(label_map[np.newaxis, :, :])
            
            # Save overlay
            overlay = img_rgb.copy()
            overlay[label_map == 1] = [0, 255, 0]  # Green for buildings
            
            overlay_meta = meta.copy()
            overlay_meta.update({'count': 3, 'dtype': 'uint8'})
            
            with rasterio.open(overlay_path, 'w', **overlay_meta) as dst:
                dst.write(overlay.transpose(2, 0, 1))
            
            self.update_progress(f"Segmentation complete! Found {building_count} buildings.")
            
            # Show success message
            self.root.after(0, lambda: messagebox.showinfo(
                "Success", 
                f"Segmentation completed successfully!\n\n"
                f"Found {building_count} buildings\n"
                f"Results saved to:\n"
                f"• {label_path}\n"
                f"• {overlay_path}"
            ))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Segmentation failed:\n{str(e)}"))
            self.update_progress("Error occurred during segmentation.")
            
        finally:
            # Re-enable button and stop progress
            self.root.after(0, self.segmentation_finished)
            
    def update_progress(self, message):
        self.root.after(0, lambda: self.progress_var.set(message))
        
    def segmentation_finished(self):
        self.segment_button.config(state="normal")
        self.progress_bar.stop()

def main():
    root = tk.Tk()
    app = SAMSegmentationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()