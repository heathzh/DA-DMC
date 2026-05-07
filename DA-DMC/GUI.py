import tkinter as tk
from tkinter import ttk
from tkinter import PhotoImage
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib
from PIL import Image

# load the model and standardizer
model_path = "C:/NN model/test/my_model_saved_model.keras"
scaler_path = "C:/NN model/test/scaler.pkl"
model = load_model(model_path)
scaler = joblib.load(scaler_path)

def predict(model, inputs):
    """Use the model to make predictions on the input data"""
    predictions = model.predict(inputs)
    predicted_classes = np.argmax(predictions, axis=1)
    return predicted_classes

def calculate_and_plot(a_value, b_value):
    # Generate variable ranges
    c_start, c_step, c_end = 250, 10, 350  
    d_start, d_step, d_end = 130, 10, 230
    
    c_values = np.arange(c_start, c_end + c_step, c_step)
    d_values = np.arange(d_start, d_end + d_step, d_step)
    
    # Generate all combinations
    C, D = np.meshgrid(c_values, d_values)
    C_flat = C.flatten()
    D_flat = D.flatten()
    
    # Construct input data with fixed a and b values
    inputs = np.column_stack((C_flat, np.full(C_flat.shape, a_value), 
                             np.full(D_flat.shape, b_value), D_flat))
    
    # Data standardization
    new_data = scaler.transform(inputs)
    
    # Model prediction
    predictions = predict(model, new_data)
    
    # Assign colors to each point based on prediction values
    colors = []
    for pred in predictions:
        if pred == 0:
            colors.append('blue')  # 0:1,represented by blue
        elif pred == 1:
            colors.append('green')  # 1:1,represented by green
        elif pred == 2:
            colors.append('orange')  # 1:2,represented by orange
        elif pred == 3:
            colors.append('black')  # 1:3,represented by black
        else:
            colors.append('red')  # 1:N,represented by red
    
    # Draw scatter plot
    fig, ax1 = plt.subplots(figsize=(4.5, 3))
    ax1.scatter(C_flat, D_flat, c=colors)  
    ax1.set_xlabel('Edge Length (μm)')
    ax1.set_ylabel('Height (μm)')
    plt.subplots_adjust(left=0.15, bottom=0.15,right=0.75 )
    
    # Insert colorbar
    image_path = 'C:/NN model/test/colorbar.png'
    image = Image.open(image_path)
    image = np.array(image)
    ax2 = fig.add_axes([0.03, 0.05, 0.93, 0.93], anchor='NE', zorder=-1)
    ax2.imshow(image)
    ax2.axis('off')  
    return fig

def calculate_w_values(a_value, b_value):
    w1 = 0.8 * a_value - 14
    w2 = 0.8 * b_value - 14
    return w1, w2

def on_calculate_clicked():
    a_value = float(a_value_entry.get())
    b_value = float(b_value_entry.get())
    w1, w2 = calculate_w_values(a_value, b_value)
    results_w1_var = tk.StringVar()
    results_w2_var = tk.StringVar()
    results_w1_var.set(f"{w1:.2f}")
    results_w2_var.set(f"{w2:.2f}")
    fig = calculate_and_plot(a_value, b_value)
    
    # Clear previous figures to avoid overlapping
    for widget in middle_frame.winfo_children():
        widget.destroy()

    # Display the generated figure in the middle frame
    canvas = FigureCanvasTkAgg(fig, master=middle_frame)
    canvas.draw()
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill=tk.BOTH, expand=True)
    
    # Display prediction data at the bottom left of the GUI
    results_label = tk.Label(left_frame, textvariable=results_w1_var)
    results_label.grid(column=0, row=6, sticky=tk.W, padx=5, pady=5)
    results_label = tk.Label(left_frame, textvariable=results_w2_var)
    results_label.grid(column=0, row=8, sticky=tk.W, padx=5, pady=5)

window = tk.Tk()
window.title("DA-DMC GUI")

# Set window size
window.geometry("1250x300")  
window.resizable(False, False)  

# Divide into three sections
left_frame = ttk.Frame(window, width=200, height=300)
middle_frame = ttk.Frame(window, width=450, height=300)
right_frame = ttk.Frame(window, width=600, height=300)

# Configure layout
left_frame.grid(row=0, column=0, sticky="nsew")
middle_frame.grid(row=0, column=1, sticky="nsew")
right_frame.grid(row=0, column=2, sticky="nsew")

# Set frame sizes
left_frame.grid_propagate(False)
middle_frame.grid_propagate(False)
right_frame.grid_propagate(False)

# Configure column weights to ensure proportional space allocation
window.grid_columnconfigure(0, weight=4)
window.grid_columnconfigure(1, weight=11)
window.grid_columnconfigure(2, weight=12)

# Configure row weight
window.grid_rowconfigure(0, weight=1)

#  Left side
a_value_label = ttk.Label(left_frame, text="Enter r1:")
a_value_label.grid(column=0, row=0, sticky=tk.W, padx=5, pady=5)

a_value_entry = ttk.Entry(left_frame)
a_value_entry.grid(column=0, row=1, sticky=tk.EW, padx=5, pady=5)

b_value_label = ttk.Label(left_frame, text="Enter r2:")
b_value_label.grid(column=0, row=2, sticky=tk.W, padx=5, pady=5)

b_value_entry = ttk.Entry(left_frame)
b_value_entry.grid(column=0, row=3, sticky=tk.EW, padx=5, pady=5)

calculate_button = ttk.Button(left_frame, text="Calculate", command=on_calculate_clicked)
calculate_button.grid(column=0, row=4, sticky=tk.EW, padx=5, pady=5)

results1_label = ttk.Label(left_frame, text="Output w1:")
results1_label.grid(column=0, row=5, sticky=tk.W, padx=5, pady=5)

results2_label = ttk.Label(left_frame, text="Output w2:")
results2_label.grid(column=0, row=7, sticky=tk.W, padx=5, pady=5)

# Right side
try:
    image_right = PhotoImage(file="C:/NN model/test/label.png")
    image_label_right = ttk.Label(right_frame, image=image_right)
    image_label_right.grid(column=0, row=0, sticky=tk.NSEW, padx=0, pady=0)
except:
    no_image_label = ttk.Label(right_frame, text="Image not available")
    no_image_label.grid(column=0, row=0, sticky=tk.NSEW, padx=0, pady=0)

window.mainloop()