# predict.py
import argparse
import torch
from model import load_checkpoint
from utils import process_image

def predict(image_path, checkpoint_path, top_k, category_names, gpu):
    # Load the model checkpoint
    model = load_checkpoint(checkpoint_path)

    # Process the input image
    input_tensor = process_image(image_path)

    # Move model to GPU if available
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    input_tensor = input_tensor.to(device)

    # Set the model to evaluation mode
    model.eval()

    # Perform inference
    with torch.no_grad():
        output = torch.exp(model(input_tensor))

    # Get the top K probabilities and classes
    probs, classes = output.topk(top_k)

    # Convert indices to class labels
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    classes = [idx_to_class[idx] for idx in classes[0].tolist()]

    # Print results
    print("Top", top_k, "classes:")
    print("Probabilities:", probs[0].tolist())
    print("Class indices:", classes)

    # Convert indices to class names if category_names is provided
    if category_names:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
        class_names = [cat_to_name[class_idx] for class_idx in classes]
        print("Class names:", class_names)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict flower name from an image.")
    parser.add_argument("image_path", help="Path to the input image.")
    parser.add_argument("checkpoint_path", help="Path to the model checkpoint.")
    parser.add_argument("--top_k", type=int, default=1, help="Return top K most likely classes.")
    parser.add_argument("--category_names", help="Path to a mapping of categories to real names.")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference.")

    args = parser.parse_args()

    predict(args.image_path, args.checkpoint_path, args.top_k, args.category_names, args.gpu)
