import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import matplotlib.pyplot as plt

# Step 1: Load the BLIP-2 Model and Processor
model_name = "Salesforce/blip2-flan-t5-xl"
processor = BlipProcessor.from_pretrained(model_name)
model = BlipForConditionalGeneration.from_pretrained(model_name)

# Step 2: Define Image Captioning Function
def generate_caption(image_path):
    """Generates a caption for a given image using BLIP-2."""
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    # Generate caption
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=50, num_beams=5, num_return_sequences=1)
    
    # Decode the generated caption
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

# Step 3: Evaluate the Model
# Function to compute BLEU and CIDEr scores
from nltk.translate.bleu_score import sentence_bleu
from pycocoevalcap.cider.cider import Cider

def evaluate_model(image_paths, ground_truths):
    """Evaluate BLIP-2 on BLEU and CIDEr metrics."""
    bleu_scores = []
    cider_scorer = Cider()

    for img_path, gt in zip(image_paths, ground_truths):
        generated_caption = generate_caption(img_path)
        print(f"Image: {img_path}\nGenerated: {generated_caption}\nGround Truth: {gt}\n")

        # BLEU Score
        bleu_scores.append(sentence_bleu([gt.split()], generated_caption.split()))

    # CIDEr requires a corpus of references
    results = {"image1": [generate_caption(p) for p in image_paths]}
    cider_score, _ = cider_scorer.compute_score({"image1": ground_truths}, results)

    return {
        "BLEU": sum(bleu_scores) / len(bleu_scores),
        "CIDEr": cider_score,
    }

# Step 4: Visualization
def visualize_caption(image_path, caption):
    """Displays the image with its generated caption."""
    image = Image.open(image_path).convert("RGB")
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.title(caption, fontsize=14, wrap=True)
    plt.show()

# Step 5: Example Usage
if __name__ == "__main__":
    # Example image path (Replace with your image path)
    image_path = "example_image.jpg"

    # Generate caption
    caption = generate_caption(image_path)
    print(f"Generated Caption: {caption}")

    # Visualize caption
    visualize_caption(image_path, caption)

    # Example evaluation (Replace with actual image paths and ground truths)
    image_paths = ["example_image1.jpg", "example_image2.jpg"]
    ground_truths = ["A group of people in a crowded marketplace.", "A dog running in a park."]

    scores = evaluate_model(image_paths, ground_truths)
    print(f"BLEU Score: {scores['BLEU']:.2f}, CIDEr Score: {scores['CIDEr']:.2f}")
