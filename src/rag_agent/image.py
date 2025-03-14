import tkinter as tk
from PIL import Image as PILImage, ImageTk
import io    

def show_graph(graph, title: str):
    root = tk.Tk()
    root.title(title)
    
    # Get the image in memory buffer
    img_bytes = graph.get_graph().draw_mermaid_png()
    img_buffer = io.BytesIO()
    img_buffer.write(img_bytes)
    # graph.get_graph().draw_mermaid_png(img_buffer)

    img_buffer.seek(0)
    
    img = PILImage.open(img_buffer)
    #img.show() This does NOT block
    img = ImageTk.PhotoImage(img)
    panel = tk.Label(root, image=img)
    panel.pack(side="bottom", fill="both", expand="yes")
    root.mainloop()# This blocks