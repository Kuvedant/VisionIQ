import torch
from transformers import FlamingoProcessor, FlamingoForConditionalGeneration
from PIL import Image
import matplotlib.pyplot as plt

# Step 1: Load the Flamingo Model and Processor
model_name = "openai/flamingo-3b"
processor = FlamingoProcessor.from_pretrained(model_name)
model = FlamingoForConditionalGeneration.from_pretrained(model_name)

# Step 2: Define a Function for Multimodal Question Answering
def multimodal_question_answering(image_path, question):
    """Generate an answer to a multimodal question using Flamingo."""
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[question], images=image, return_tensors="pt", padding=True)

    # Generate the answer
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=50, num_beams=5)
    
    # Decode the generated answer
    answer = processor.decode(outputs[0], skip_special_tokens=True)
    return answer

# Step 3: Fine-Tuning 
def fine_tune_flamingo(dataset_loader, epochs=3, learning_rate=5e-5):
    """Fine-tune Flamingo on a custom dataset."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for batch in dataset_loader:
            images, questions, answers = batch  # Replace with actual dataset logic
            inputs = processor(text=questions, images=images, return_tensors="pt", padding=True)
            labels = processor(text=answers, return_tensors="pt", padding=True).input_ids

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

# Step 4: Visualization
def visualize_question_answer(image_path, question, answer):
    """Displays the image along with the question and generated answer."""
    image = Image.open(image_path).convert("RGB")
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"Q: {question}\nA: {answer}", fontsize=14, wrap=True)
    plt.show()

# Step 5: Example Usage
if __name__ == "__main__":
    # Example image path (Replace with your image path)
    image_path = "example_image.jpg"

    # Example question
    question = "Which of the objects on the table could be used for outdoor activities?"

    # Generate the answer
    answer = multimodal_question_answering(image_path, question)
    print(f"Generated Answer: {answer}")

    # Visualize the result
    visualize_question_answer(image_path, question, answer)

    # Note: Fine-tuning example is not executed here. It requires a custom dataset and setup.
