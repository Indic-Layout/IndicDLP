import cv2
import os
import argparse
from doclayout_yolo import YOLOv10

def main(args):
    print("Running model inference...")

    model = YOLOv10(args.model_path)

    det_res = model.predict(
        args.image_path,
        imgsz=1024,
        conf=0.3,
        iou=0.8,
        device="cuda:0"
    )
    print("Model classes:", model.names)

    # Save annotated output
    annotated_frame = det_res[0].plot(pil=True, line_width=5, font_size=20)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    cv2.imwrite(args.output_path, annotated_frame)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run layout prediction using DocLayout-YOLOv10")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the model .pt file")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the input image")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the annotated output image")

    args = parser.parse_args()
    main(args)
