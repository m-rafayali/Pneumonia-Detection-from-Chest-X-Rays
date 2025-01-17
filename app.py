# 1
import gradio as gr
import os
import torch
from model import create_model
from typing import Dict

# setup class names
with open("class_names.txt", "r") as f:
    class_names = [food_name.strip() for food_name in f.readlines()]

# model and transforms
model, model_transforms = create_model(num_classes=2)

# load the saved weights
model.load_state_dict(
    torch.load(f="resnet.pth", map_location=torch.device("cpu"))
)

# predict function
def predict(img):
    try:
        # Check if image is received
        if img is None:
            return "No image uploaded!"

        # Model loading (assuming effnetb2 is defined in create_model)
        model, model_transforms = create_model(num_classes=2)
        model.load_state_dict(
            torch.load(f="resnet.pth", map_location=torch.device("cpu"))
        )

        img = model_transforms(img).unsqueeze(0)  # Add batch dim
        model.eval()
        with torch.inference_mode():
            pred_logit = model(img)
            pred_probs = torch.softmax(pred_logit, dim=1)
            pred_labels_and_probs = {
                class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))
            }

        return pred_labels_and_probs
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Gradio app setup
title = "Pneumonia Detection"
# description = "Classify images into 10 CIFAR-10 classes using a ResNet50 model. Quick, accurate, and a great demonstration of computer vision in action."
description = "Classify chest X-ray images into two categories: Normal and Pneumonia using a deep learning model."

# Example list
example_list = [["examples/" + example] for example in os.listdir("examples")]

# Create the Gradio demo
demo = gr.Interface(
    fn=predict,  # maps input to output
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2, label="Predictions"),
    examples=example_list,
    title=title,
    description=description
)

# Launch the app
demo.launch(
    debug=False
    # Prints error locally like in Google Colab
    # Generate link publicly, like share with public
)
