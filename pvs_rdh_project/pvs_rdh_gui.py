# GUI Application for PVS-RDH
# User-friendly interface for the Pixel Value Splitting RDH algorithm

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import cv2
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import os
from pvs_rdh_implementation import PVS_RDH

class PVS_RDH_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PVS-RDH: Pixel Value Splitting Reversible Data Hiding")
        self.root.geometry("1200x800")
        
        # Variables
        self.original_image = None
        self.embedded_image = None
        self.recovered_image = None
        self.embedding_info = None
        self.pvs_algorithm = PVS_RDH()
        
        # Create GUI
        self.create_widgets()
        
    def create_widgets(self):
        # Main notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 1: Embedding
        self.embed_frame = ttk.Frame(notebook)
        notebook.add(self.embed_frame, text="Embedding")
        self.create_embedding_tab()
        
        # Tab 2: Extraction
        self.extract_frame = ttk.Frame(notebook)
        notebook.add(self.extract_frame, text="Extraction")
        self.create_extraction_tab()
        
        # Tab 3: Analysis
        self.analysis_frame = ttk.Frame(notebook)
        notebook.add(self.analysis_frame, text="Analysis")
        self.create_analysis_tab()
        
    def create_embedding_tab(self):
        # Left panel: Controls
        left_frame = ttk.Frame(self.embed_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Image loading
        ttk.Label(left_frame, text="1. Load Image", font=('Arial', 12, 'bold')).pack(anchor=tk.W, pady=(0,5))
        ttk.Button(left_frame, text="Browse Image", command=self.load_image).pack(fill=tk.X, pady=2)
        
        self.image_info_label = ttk.Label(left_frame, text="No image loaded", foreground="gray")
        self.image_info_label.pack(anchor=tk.W, pady=2)
        
        # Watermark input
        ttk.Separator(left_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        ttk.Label(left_frame, text="2. Enter Watermark", font=('Arial', 12, 'bold')).pack(anchor=tk.W, pady=(0,5))
        
        self.watermark_text = tk.Text(left_frame, height=4, width=30)
        self.watermark_text.pack(fill=tk.X, pady=2)
        self.watermark_text.insert('1.0', "Hello PVS-RDH!")
        
        # Algorithm parameters
        ttk.Separator(left_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        ttk.Label(left_frame, text="3. Parameters", font=('Arial', 12, 'bold')).pack(anchor=tk.W, pady=(0,5))
        
        params_frame = ttk.Frame(left_frame)
        params_frame.pack(fill=tk.X)
        
        ttk.Label(params_frame, text="K1 (offset):").grid(row=0, column=0, sticky=tk.W)
        self.k1_var = tk.StringVar(value="4")
        ttk.Entry(params_frame, textvariable=self.k1_var, width=5).grid(row=0, column=1, padx=5)
        
        ttk.Label(params_frame, text="K2 (offset):").grid(row=1, column=0, sticky=tk.W)
        self.k2_var = tk.StringVar(value="6")
        ttk.Entry(params_frame, textvariable=self.k2_var, width=5).grid(row=1, column=1, padx=5)
        
        ttk.Label(left_frame, text="Note: K1 + K2 must equal 10", font=('Arial', 8), foreground="gray").pack(anchor=tk.W, pady=2)
        
        # Embedding button
        ttk.Separator(left_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        self.embed_button = ttk.Button(left_frame, text="🔐 Embed Watermark", command=self.embed_watermark)
        self.embed_button.pack(fill=tk.X, pady=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(left_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=2)
        
        # Results
        ttk.Separator(left_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        ttk.Label(left_frame, text="Results", font=('Arial', 12, 'bold')).pack(anchor=tk.W)
        
        self.results_text = tk.Text(left_frame, height=8, width=30, state=tk.DISABLED)
        self.results_text.pack(fill=tk.X, pady=2)
        
        # Save button
        self.save_button = ttk.Button(left_frame, text="💾 Save Embedded Image", command=self.save_embedded_image, state=tk.DISABLED)
        self.save_button.pack(fill=tk.X, pady=5)
        
        # Right panel: Image display
        right_frame = ttk.Frame(self.embed_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Image canvas
        self.embed_canvas_frame = ttk.Frame(right_frame)
        self.embed_canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create figure for image display
        self.embed_fig, self.embed_axes = plt.subplots(1, 2, figsize=(10, 5))
        self.embed_axes[0].set_title("Original Image")
        self.embed_axes[1].set_title("Embedded Image")
        
        self.embed_canvas = FigureCanvasTkAgg(self.embed_fig, self.embed_canvas_frame)
        self.embed_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def create_extraction_tab(self):
        # Left panel: Controls
        left_frame = ttk.Frame(self.extract_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Embedded image loading
        ttk.Label(left_frame, text="1. Load Embedded Image", font=('Arial', 12, 'bold')).pack(anchor=tk.W, pady=(0,5))
        ttk.Button(left_frame, text="Browse Embedded Image", command=self.load_embedded_image).pack(fill=tk.X, pady=2)
        
        self.embedded_info_label = ttk.Label(left_frame, text="No embedded image loaded", foreground="gray")
        self.embedded_info_label.pack(anchor=tk.W, pady=2)
        
        # Load embedding info
        ttk.Separator(left_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        ttk.Label(left_frame, text="2. Load Embedding Info", font=('Arial', 12, 'bold')).pack(anchor=tk.W, pady=(0,5))
        ttk.Button(left_frame, text="Browse Info File", command=self.load_embedding_info).pack(fill=tk.X, pady=2)
        
        self.info_status_label = ttk.Label(left_frame, text="No info file loaded", foreground="gray")
        self.info_status_label.pack(anchor=tk.W, pady=2)
        
        # Extract button
        ttk.Separator(left_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        self.extract_button = ttk.Button(left_frame, text="🔓 Extract Watermark", command=self.extract_watermark, state=tk.DISABLED)
        self.extract_button.pack(fill=tk.X, pady=5)
        
        # Extraction results
        ttk.Separator(left_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        ttk.Label(left_frame, text="Extracted Watermark", font=('Arial', 12, 'bold')).pack(anchor=tk.W)
        
        self.extracted_text = tk.Text(left_frame, height=4, width=30, state=tk.DISABLED)
        self.extracted_text.pack(fill=tk.X, pady=2)
        
        ttk.Label(left_frame, text="Recovery Status", font=('Arial', 12, 'bold')).pack(anchor=tk.W, pady=(10,5))
        
        self.recovery_status = ttk.Label(left_frame, text="Not tested", foreground="gray")
        self.recovery_status.pack(anchor=tk.W, pady=2)
        
        # Save recovered image
        self.save_recovered_button = ttk.Button(left_frame, text="💾 Save Recovered Image", command=self.save_recovered_image, state=tk.DISABLED)
        self.save_recovered_button.pack(fill=tk.X, pady=5)
        
        # Right panel: Image display
        right_frame = ttk.Frame(self.extract_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Image canvas for extraction
        self.extract_canvas_frame = ttk.Frame(right_frame)
        self.extract_canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.extract_fig, self.extract_axes = plt.subplots(1, 2, figsize=(10, 5))
        self.extract_axes[0].set_title("Embedded Image")
        self.extract_axes[1].set_title("Recovered Image")
        
        self.extract_canvas = FigureCanvasTkAgg(self.extract_fig, self.extract_canvas_frame)
        self.extract_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def create_analysis_tab(self):
        # Analysis controls
        control_frame = ttk.Frame(self.analysis_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(control_frame, text="📊 Analyze Current Images", command=self.analyze_images).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="📈 Capacity Analysis", command=self.capacity_analysis).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="📋 Generate Report", command=self.generate_report).pack(side=tk.LEFT, padx=5)
        
        # Analysis canvas
        self.analysis_canvas_frame = ttk.Frame(self.analysis_frame)
        self.analysis_canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Load image
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    # Try PIL
                    pil_image = Image.open(file_path).convert('L')
                    image = np.array(pil_image)
                
                self.original_image = image
                self.image_info_label.config(text=f"Loaded: {os.path.basename(file_path)} ({image.shape})")
                
                # Display image
                self.embed_axes[0].clear()
                self.embed_axes[0].imshow(image, cmap='gray')
                self.embed_axes[0].set_title("Original Image")
                self.embed_axes[0].axis('off')
                self.embed_canvas.draw()
                
                # Enable embedding if watermark is present
                if self.watermark_text.get('1.0', tk.END).strip():
                    self.embed_button.config(state=tk.NORMAL)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def embed_watermark(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Please load an image first")
            return
        
        watermark = self.watermark_text.get('1.0', tk.END).strip()
        if not watermark:
            messagebox.showerror("Error", "Please enter a watermark")
            return
        
        try:
            K1 = int(self.k1_var.get())
            K2 = int(self.k2_var.get())
            
            if K1 + K2 != 10:
                messagebox.showerror("Error", "K1 + K2 must equal 10")
                return
                
        except ValueError:
            messagebox.showerror("Error", "K1 and K2 must be integers")
            return
        
        # Disable button and start progress
        self.embed_button.config(state=tk.DISABLED)
        self.progress.start()
        
        # Run embedding in separate thread
        def embed_thread():
            try:
                # Initialize algorithm
                self.pvs_algorithm = PVS_RDH(K1=K1, K2=K2)
                
                # Convert watermark to binary
                watermark_bits = ''.join(format(ord(c), '08b') for c in watermark)
                
                # Embed
                self.embedded_image, self.embedding_info = self.pvs_algorithm.embed_watermark(self.original_image, watermark_bits)
                
                # Update GUI in main thread
                self.root.after(0, self.embedding_complete)
                
            except Exception as e:
                self.root.after(0, lambda: self.embedding_error(str(e)))
        
        threading.Thread(target=embed_thread, daemon=True).start()
    
    def embedding_complete(self):
        self.progress.stop()
        self.embed_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.NORMAL)
        
        # Display embedded image
        self.embed_axes[1].clear()
        self.embed_axes[1].imshow(self.embedded_image, cmap='gray')
        self.embed_axes[1].set_title("Embedded Image")
        self.embed_axes[1].axis('off')
        self.embed_canvas.draw()
        
        # Update results
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete('1.0', tk.END)
        
        results = f"""Embedding Results:
━━━━━━━━━━━━━━━━━━
✅ Success!

📊 Statistics:
• Embedding pairs: {self.embedding_info['embedding_pairs']}
• Auxiliary data: {self.embedding_info['auxiliary_data_size']} bits
• Watermark length: {self.embedding_info['watermark_length']} bits
• Payload: {self.embedding_info['watermark_length'] / (self.original_image.shape[0] * self.original_image.shape[1]):.4f} bpp

🎯 Quality Metrics:
• PSNR: {self.embedding_info['psnr']:.2f} dB
• MSE: {self.embedding_info['mse']:.2f}
"""
        
        self.results_text.insert('1.0', results)
        self.results_text.config(state=tk.DISABLED)
        
        messagebox.showinfo("Success", "Watermark embedded successfully!")
    
    def embedding_error(self, error_msg):
        self.progress.stop()
        self.embed_button.config(state=tk.NORMAL)
        messagebox.showerror("Embedding Error", f"Failed to embed watermark: {error_msg}")
    
    def save_embedded_image(self):
        if self.embedded_image is None:
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Embedded Image",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                cv2.imwrite(file_path, self.embedded_image)
                
                # Also save embedding info
                info_path = file_path.rsplit('.', 1)[0] + '_info.json'
                import json
                
                # Convert numpy arrays to lists for JSON serialization
                info_copy = self.embedding_info.copy()
                info_copy['embedding_pairs_list'] = [list(pair) for pair in info_copy['embedding_pairs_list']]
                info_copy['auxiliary_data'] = [list(pair) for pair in info_copy['auxiliary_data']]
                
                with open(info_path, 'w') as f:
                    json.dump(info_copy, f, indent=2)
                
                messagebox.showinfo("Success", f"Embedded image and info saved:\n{file_path}\n{info_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save: {str(e)}")
    
    def load_embedded_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Embedded Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    pil_image = Image.open(file_path).convert('L')
                    image = np.array(pil_image)
                
                self.embedded_image = image
                self.embedded_info_label.config(text=f"Loaded: {os.path.basename(file_path)} ({image.shape})")
                
                # Display image
                self.extract_axes[0].clear()
                self.extract_axes[0].imshow(image, cmap='gray')
                self.extract_axes[0].set_title("Embedded Image")
                self.extract_axes[0].axis('off')
                self.extract_canvas.draw()
                
                self.check_extraction_ready()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load embedded image: {str(e)}")
    
    def load_embedding_info(self):
        file_path = filedialog.askopenfilename(
            title="Select Embedding Info File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                import json
                with open(file_path, 'r') as f:
                    self.embedding_info = json.load(f)
                
                # Convert lists back to tuples
                self.embedding_info['embedding_pairs_list'] = [tuple(pair) for pair in self.embedding_info['embedding_pairs_list']]
                self.embedding_info['auxiliary_data'] = [tuple(pair) for pair in self.embedding_info['auxiliary_data']]
                
                self.info_status_label.config(text=f"Loaded: {os.path.basename(file_path)}")
                self.check_extraction_ready()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load info file: {str(e)}")
    
    def check_extraction_ready(self):
        if self.embedded_image is not None and self.embedding_info is not None:
            self.extract_button.config(state=tk.NORMAL)
    
    def extract_watermark(self):
        if self.embedded_image is None or self.embedding_info is None:
            messagebox.showerror("Error", "Please load both embedded image and info file")
            return
        
        try:
            # Extract watermark
            watermark_bits, self.recovered_image = self.pvs_algorithm.extract_watermark(self.embedded_image, self.embedding_info)
            
            # Convert binary to text
            extracted_text = ''
            for i in range(0, len(watermark_bits), 8):
                if i + 8 <= len(watermark_bits):
                    byte = watermark_bits[i:i+8]
                    if len(byte) == 8:
                        extracted_text += chr(int(byte, 2))
            
            # Display extracted text
            self.extracted_text.config(state=tk.NORMAL)
            self.extracted_text.delete('1.0', tk.END)
            self.extracted_text.insert('1.0', extracted_text)
            self.extracted_text.config(state=tk.DISABLED)
            
            # Display recovered image
            self.extract_axes[1].clear()
            self.extract_axes[1].imshow(self.recovered_image, cmap='gray')
            self.extract_axes[1].set_title("Recovered Image")
            self.extract_axes[1].axis('off')
            self.extract_canvas.draw()
            
            # Check perfect recovery (if original image is available)
            if self.original_image is not None:
                diff = np.sum(np.abs(self.original_image.astype(int) - self.recovered_image.astype(int)))
                if diff == 0:
                    self.recovery_status.config(text="✅ Perfect recovery", foreground="green")
                else:
                    self.recovery_status.config(text=f"⚠️ Recovery error: {diff} pixels", foreground="orange")
            else:
                self.recovery_status.config(text="✅ Extraction complete", foreground="green")
            
            self.save_recovered_button.config(state=tk.NORMAL)
            messagebox.showinfo("Success", "Watermark extracted successfully!")
            
        except Exception as e:
            messagebox.showerror("Extraction Error", f"Failed to extract watermark: {str(e)}")
    
    def save_recovered_image(self):
        if self.recovered_image is None:
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Recovered Image",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                cv2.imwrite(file_path, self.recovered_image)
                messagebox.showinfo("Success", f"Recovered image saved: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save: {str(e)}")
    
    def analyze_images(self):
        if self.original_image is None or self.embedded_image is None:
            messagebox.showwarning("Warning", "Please load and embed images first")
            return
        
        # Create analysis plots
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        
        # Original vs Embedded
        axes[0,0].imshow(self.original_image, cmap='gray')
        axes[0,0].set_title("Original Image")
        axes[0,0].axis('off')
        
        axes[0,1].imshow(self.embedded_image, cmap='gray')
        axes[0,1].set_title("Embedded Image")
        axes[0,1].axis('off')
        
        # Difference
        diff_image = np.abs(self.original_image.astype(int) - self.embedded_image.astype(int))
        axes[0,2].imshow(diff_image, cmap='hot')
        axes[0,2].set_title(f"Difference (max: {diff_image.max()})")
        axes[0,2].axis('off')
        
        # Histograms
        axes[1,0].hist(self.original_image.flatten(), bins=50, alpha=0.7, color='blue', label='Original')
        axes[1,0].hist(self.embedded_image.flatten(), bins=50, alpha=0.7, color='red', label='Embedded')
        axes[1,0].set_title("Pixel Value Distribution")
        axes[1,0].legend()
        
        # Group-A analysis
        group_A_orig, _ = self.pvs_algorithm.pixel_value_splitting(self.original_image)
        group_A_embed, _ = self.pvs_algorithm.pixel_value_splitting(self.embedded_image)
        
        axes[1,1].hist(group_A_orig.flatten(), bins=30, alpha=0.7, color='blue', label='Original Group-A')
        axes[1,1].hist(group_A_embed.flatten(), bins=30, alpha=0.7, color='red', label='Embedded Group-A')
        axes[1,1].set_title("Group-A Distribution")
        axes[1,1].legend()
        
        # Quality metrics over different capacities (if available)
        if hasattr(self, 'capacity_data'):
            axes[1,2].plot(self.capacity_data['lengths'], self.capacity_data['psnr'], 'b-o')
            axes[1,2].set_xlabel("Watermark Length")
            axes[1,2].set_ylabel("PSNR (dB)")
            axes[1,2].set_title("Quality vs Capacity")
        else:
            axes[1,2].text(0.5, 0.5, "Run Capacity Analysis", ha='center', va='center', transform=axes[1,2].transAxes)
            axes[1,2].set_title("Capacity Analysis")
        
        plt.tight_layout()
        
        # Clear previous analysis canvas and add new plot
        for widget in self.analysis_canvas_frame.winfo_children():
            widget.destroy()
        
        canvas = FigureCanvasTkAgg(fig, self.analysis_canvas_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw()
    
    def capacity_analysis(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        messagebox.showinfo("Info", "Running capacity analysis... This may take a moment.")
        
        # Run capacity analysis in separate thread
        def analysis_thread():
            try:
                lengths = range(50, 500, 50)
                psnr_values = []
                
                for length in lengths:
                    # Generate random watermark
                    test_watermark = ''.join(np.random.choice(['0', '1'], length))
                    
                    try:
                        pvs_test = PVS_RDH()
                        embedded, info = pvs_test.embed_watermark(self.original_image, test_watermark)
                        metrics = pvs_test.calculate_metrics(self.original_image, embedded)
                        psnr_values.append(metrics['psnr'])
                    except:
                        break
                
                self.capacity_data = {'lengths': list(lengths[:len(psnr_values)]), 'psnr': psnr_values}
                
                # Update GUI
                self.root.after(0, lambda: messagebox.showinfo("Success", "Capacity analysis complete! Check the Analysis tab."))
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Capacity analysis failed: {str(e)}"))
        
        threading.Thread(target=analysis_thread, daemon=True).start()
    
    def generate_report(self):
        if self.original_image is None or self.embedded_image is None:
            messagebox.showwarning("Warning", "Please load and embed images first")
            return
        
        report_text = f"""
PVS-RDH Analysis Report
======================

Image Information:
• Original size: {self.original_image.shape}
• Pixel value range: {self.original_image.min()} - {self.original_image.max()}

Embedding Parameters:
• K1 (positive offset): {self.embedding_info['K1']}
• K2 (negative offset): {self.embedding_info['K2']}

Embedding Statistics:
• Embedding pairs found: {self.embedding_info['embedding_pairs']}
• Auxiliary data size: {self.embedding_info['auxiliary_data_size']} bits
• Watermark length: {self.embedding_info['watermark_length']} bits
• Payload: {self.embedding_info['watermark_length'] / (self.original_image.shape[0] * self.original_image.shape[1]):.4f} bpp

Quality Metrics:
• PSNR: {self.embedding_info['psnr']:.2f} dB
• MSE: {self.embedding_info['mse']:.2f}

Algorithm Performance:
• Embedding pairs utilization: {(self.embedding_info['watermark_length'] / self.embedding_info['embedding_pairs']) * 100:.1f}%
• Auxiliary data overhead: {(self.embedding_info['auxiliary_data_size'] / self.embedding_info['watermark_length']) * 100:.1f}%

Conclusion:
The PVS-RDH algorithm successfully embedded {self.embedding_info['watermark_length']} bits with good quality preservation (PSNR > 40dB is considered excellent for steganography).
"""
        
        # Show report in new window
        report_window = tk.Toplevel(self.root)
        report_window.title("PVS-RDH Analysis Report")
        report_window.geometry("600x500")
        
        text_widget = tk.Text(report_window, wrap=tk.WORD, padx=10, pady=10)
        text_widget.pack(fill=tk.BOTH, expand=True)
        text_widget.insert('1.0', report_text)
        text_widget.config(state=tk.DISABLED)
        
        # Save button
        def save_report():
            file_path = filedialog.asksaveasfilename(
                title="Save Report",
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            if file_path:
                try:
                    with open(file_path, 'w') as f:
                        f.write(report_text)
                    messagebox.showinfo("Success", f"Report saved: {file_path}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save report: {str(e)}")
        
        ttk.Button(report_window, text="Save Report", command=save_report).pack(pady=10)

def main():
    root = tk.Tk()
    app = PVS_RDH_GUI(root)
    
    # Add menu bar
    menubar = tk.Menu(root)
    root.config(menu=menubar)
    
    # File menu
    file_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="File", menu=file_menu)
    file_menu.add_command(label="Load Image", command=app.load_image)
    file_menu.add_command(label="Load Embedded Image", command=app.load_embedded_image)
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=root.quit)
    
    # Help menu
    help_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Help", menu=help_menu)
    help_menu.add_command(label="About", lambda: messagebox.showinfo("About", "PVS-RDH: Pixel Value Splitting Reversible Data Hiding\n\nBased on the research paper by Meenpal et al.\n\nImplementation for educational purposes."))
    
    root.mainloop()

if __name__ == "__main__":
    main()
