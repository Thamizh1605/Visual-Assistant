from transformers import ViltProcessor, ViltForQuestionAnswering,BlipProcessor,BlipForQuestionAnswering
from PIL import Image
import torch


def answer_question(image_path: str, question: str):
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    # Open local image
    img = Image.open(image_path).convert("RGB")

    # Prepare inputs
    encoding = processor(img, question, return_tensors="pt")

    # Forward pass
    with torch.no_grad():
        outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    answer = model.config.id2label[idx]
    return answer

def answer_question_Blip(image_path:str,question:str):

    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    image = Image.open(image_path)
    inputs = processor(image, question, return_tensors="pt")
    out = model.generate(**inputs, max_length=10)
    answer = processor.tokenizer.decode(out[0], skip_special_tokens=True)
    return answer
