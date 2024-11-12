import sys
import os
import cv2
import pandas as pd

# Add the src directory to the sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# main.py
from src.video_reader import capture_frame
from src.ocr_processor import preprocess_frame, crop_frame
from src.frame_template_matcher import process_image

# def template_match_and_mask(frame):
#     templates_dir = "templates/guns/"
#     masked_frame = frame.copy()
    
#     for template_name in os.listdir(templates_dir):
#         template_path = os.path.join(templates_dir, template_name)
#         template = cv2.imread(template_path, 0)
#         if template is None:
#             continue
        
#         res = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
#         min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
#         if max_val > 0.50:
#             top_left = max_loc
#             h, w = template.shape
#             bottom_right = (top_left[0] + w, top_left[1] + h)
#             cv2.rectangle(masked_frame, top_left, bottom_right, (0, 0, 0), -1)
    
#     return masked_frame

def main():
    video_path = "C:/Users/liebe/Videos/4K Video Downloader+/OPTIC TEXAS VS LA THEIVES GRAND FINALS ($50K KAYSAN LAN) BLACK OPS 6.mp4"
    start_frame = (60 * (4*60 + 20)) 
    end_frame = (60 * (4*60 + 40))
    
    results = []

    for frame_number in range(start_frame, end_frame + 1, 12):
        # Step 1: Capture frame
        frame = capture_frame(video_path, frame_number)
        if frame is None:
            print(f"No frame captured at frame {frame_number}. Skipping.")
            continue
        
        
        # Step 2: Define region of interest (ROI)
        top_left = (18, 374)
        bottom_right = (300, 395)
        cropped_image = crop_frame(frame, top_left, bottom_right)
        
        # Step 3: Preprocess frame
        processed_frame = preprocess_frame(cropped_image)
        
        pct_black = 100 * (processed_frame == 0).sum() / processed_frame.size
        pct_white = 100 * (processed_frame == 255).sum() / processed_frame.size
        
        if pct_white > 50 or pct_black > 95:
            print(f"Skipping frame {frame_number} due to high pct of white or black pixels")
            continue

        ## show frame
        # cv2.imshow("Captured Frame", processed_frame)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        # # Step 4: Template match and mask
        # masked_frame = template_match_and_mask(processed_frame)
        
        text = process_image(processed_frame)
        print(f"Frame {frame_number}: {text}")
        results.append({"frame_number": frame_number, "text": text})
    
    df = pd.DataFrame(results)
    print(df)
    
    ## end video
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
