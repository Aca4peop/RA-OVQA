import open_clip
import numpy as np


disortions = {
    "Blurring": "The image looks blurry and out of focus.",
    "Deformation": "Objects are warped and deformed.",
    "Fog": "The image has foggy or hazy.",
    "Noise": "The image has noise.",
    "Overexposure": "The image has overexposed areas.",
    "Underexposure": "The image has underexposed areas.",
    "Ringing": "There are halo like ripples.",
    "Blockness": "The image shows square block patterns, like heavy compression artifacts.",
    "Color distortion": "The image has unnatural colors that are too dark or too light.",
    "Discontinuities": "There are abrupt changes in brightness or color.",
    "Missing objects": "Some parts of the scene are missing.",
    "Seams": "There are seams or stitching lines.",
    "Ghosting": "Faint duplicate shadows of objects appear, like a double image.",
}


if __name__ == "__main__":
    model, _, preprocess = open_clip.create_model_and_transforms(
        "convnext_base_w", pretrained="laion2b_s13b_b82k_augreg"
    )
    tokenizer = open_clip.get_tokenizer("convnext_base_w")
    texts = list(disortions.values())
    text = tokenizer(texts)
    text_features = model.encode_text(text)
    text_features = text_features.detach().numpy()
    np.save("dis_text_features.npy", text_features)
