import torch
from transformers import SamModel, SamProcessor, BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr

# Load BLIP for Caption Generation
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load SAM for Segmentation
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
sam_model = SamModel.from_pretrained("facebook/sam-vit-base")

# Load YOLOv5 for Object Detection
yolo_model = torch.hub.load("ultralytics/yolov5", "yolov5s")

# Load CLIP for Zero-shot Classification
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def generate_caption(image):
    """
    Generate a caption for the image using BLIP.
    """
    inputs = blip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        caption = blip_model.generate(**inputs)
    return blip_processor.decode(caption[0], skip_special_tokens=True)


def object_detection_with_bounding_boxes(image_path):
    """
    Perform object detection using YOLOv5 and return the image with bounding boxes.
    """
    results = yolo_model(image_path)
    detections = results.pandas().xyxy[0]  # Bounding boxes

    # Load the original image
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # Draw bounding boxes with labels
    for _, row in detections.iterrows():
        # Bounding box coordinates
        xmin, ymin, xmax, ymax = row["xmin"], row["ymin"], row["xmax"], row["ymax"]
        confidence = row["confidence"]
        label = f"{row['name']} ({confidence:.2f})"

        # Draw rectangle and label
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
        draw.text((xmin, ymin - 10), label, fill="red")

    # Save and display the image
    detected_image_path = "detected_image.png"
    image.save(detected_image_path)
    return detected_image_path, detections[["name", "confidence", "xmin", "ymin", "xmax", "ymax"]]


def segment_image_with_boundary(image_path, object_name="object"):
    """
    Perform segmentation using SAM, guided by YOLOv5 object detection results.
    """
    image = Image.open(image_path).convert("RGB")
    resized_image = image.resize((1024, 1024))  # Resize to SAM input size

    detections = object_detection_with_bounding_boxes(image_path)[1]
    bounding_box = None

    for _, row in detections.iterrows():
        if object_name.lower() in row["name"].lower():
            bounding_box = [row["xmin"], row["ymin"], row["xmax"], row["ymax"]]
            break

    if bounding_box is None:
        return f"No '{object_name}' found in the image."

    # Scale bounding box for resized image
    scale_x = 1024 / image.width
    scale_y = 1024 / image.height
    bounding_box = [int(bounding_box[0] * scale_x), int(bounding_box[1] * scale_y),
                    int(bounding_box[2] * scale_x), int(bounding_box[3] * scale_y)]

    # Perform segmentation with SAM
    inputs = sam_processor(images=resized_image, return_tensors="pt")
    with torch.no_grad():
        outputs = sam_model(pixel_values=inputs["pixel_values"])

    masks = outputs.pred_masks.cpu().numpy().squeeze()

    # Visualize segmentation
    plt.figure(figsize=(10, 10))
    plt.imshow(resized_image)
    plt.imshow(masks[0], alpha=0.5, cmap="jet")  # Overlay segmentation boundary
    plt.axis("off")
    plt.title(f"Segmentation for '{object_name}'")
    plt.savefig("segmented_image.png")
    plt.close()

    return "segmented_image.png"


def process_query(image, query):
    """
    Process the user query and dynamically decide the task to perform.
    """
    image_path = "uploaded_image.png"
    image.save(image_path)

    query_lower = query.lower()
    if "caption" in query_lower or "describe" in query_lower or "scene" in query_lower:
        caption = generate_caption(image)
        return f"Scene Description: {caption}"
    elif "segment" in query_lower or "highlight" in query_lower:
        object_name = query.split("segment")[-1].strip() or "object"
        segmented_image_path = segment_image_with_boundary(image_path, object_name)
        return segmented_image_path
    elif "detect" in query_lower or "object" in query_lower:
        detected_image_path, detections = object_detection_with_bounding_boxes(image_path)
        return detected_image_path
    elif "classify" in query_lower:
        return classify_image(image, query.split("classify")[-1].strip())
    else:
        return "Query not recognized. Please ask to 'caption', 'segment', 'detect', or 'classify'."


# Gradio Interface
gr.Interface(
    fn=process_query,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Textbox(lines=2, placeholder="Enter your query (e.g., 'Describe the scene', 'Detect objects', 'Segment the apple')", label="Query")
    ],
    outputs="image",
    title="VisionIQ: Intelligent Multimodal Assistant for Complex Visual Scene Understanding and Reasoning",
    description="Upload an image and ask a query to perform scene description, object detection (with bounding boxes), segmentation (with boundaries), or classification."
).launch()
