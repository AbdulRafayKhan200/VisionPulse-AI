from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import io
import os
import uvicorn

app = FastAPI(title="VisionPulse AI")

# Model Initialization
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
model.eval()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...), prompt: str = Form("object")):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Process prompts
        text_queries = [p.strip() for p in prompt.split(",") if p.strip()]
        
        inputs = processor(text=text_queries, images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        target_sizes = torch.tensor([image.size[::-1]])
        
        # Updated post-processing using processor's image processor helper directly
        results = processor.image_processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=0.1
        )[0]
        
        boxes, scores, labels = results["boxes"], results["scores"], results["labels"]
        
        detections = []
        for box, score, label in zip(boxes, scores, labels):
            box_coords = [round(i, 2) for i in box.tolist()]
            detections.append({
                "box": box_coords,
                "score": round(score.item(), 3),
                "label": text_queries[label.item()]
            })
            
        return JSONResponse(content={"status": "success", "detections": detections, "image_size": [image.width, image.height]})
    
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)