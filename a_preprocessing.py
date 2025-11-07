import cv2
import numpy as np
import os
import argparse

def preprocess_image(img_path, s_thresh=40, v_thresh=50, kernel_size=5):
    img = cv2.imread(img_path)
    if img is None:
        return None, None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    mask = (s > s_thresh) & (v > v_thresh)
    mask = mask.astype(np.uint8) * 255
    if kernel_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    result = cv2.bitwise_and(img, img, mask=mask)
    return mask, result

def process_folder(input_dir, output_dir, s_thresh=40, v_thresh=50, kernel_size=5):
    os.makedirs(output_dir, exist_ok=True)
    for name in os.listdir(input_dir):
        lower = name.lower()
        if lower.endswith(".jpg") or lower.endswith(".jpeg") or lower.endswith(".png"):
            in_path = os.path.join(input_dir, name)
            mask, result = preprocess_image(in_path, s_thresh, v_thresh, kernel_size)
            if result is not None:
                base, ext = os.path.splitext(name)
                out_img = os.path.join(output_dir, f"prep_{base}{ext}")
                out_mask = os.path.join(output_dir, f"mask_{base}.png")
                cv2.imwrite(out_img, result)
                cv2.imwrite(out_mask, mask)
    print("Preprocessing completed.")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="images")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--s_thresh", type=int, default=40)
    parser.add_argument("--v_thresh", type=int, default=50)
    parser.add_argument("--kernel", type=int, default=5)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    process_folder(args.input_dir, args.output_dir, args.s_thresh, args.v_thresh, args.kernel)
