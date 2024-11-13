import sys
import os
import cv2
import pandas as pd

# Add the src directory to the sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.video_reader import capture_frame
from src.ocr_processor import preprocess_frame, crop_frame
from src.frame_template_matcher import read_killfeed_frame



def main():
    video_path = "C:/Users/liebe/Videos/4K Video Downloader+/OPTIC TEXAS VS LA THEIVES GRAND FINALS ($50K KAYSAN LAN) BLACK OPS 6.mp4"
    start_frame = (60 * (0*60 + 4)) 
    end_frame = (60 * (8*60 + 35))
    
    
    results = []

    for frame_number in range(start_frame, end_frame + 1, 30):
        ROI_TOP_LEFT = (18, 374)
        ROI_BOTTOM_RIGHT = (300, 395)
        frame = capture_frame(video_path, frame_number)
        if frame is None:
            print(f"No frame captured at frame {frame_number}. Skipping.")
            continue
        
        cropped_image = crop_frame(frame, ROI_TOP_LEFT, ROI_BOTTOM_RIGHT)
        
        pct_black = 100 * (cropped_image == 0).sum() / cropped_image.size
        pct_white = 100 * (cropped_image == 255).sum() / cropped_image.size
        
        if pct_white > 50 or pct_black > 95:
            print(f"Skipping frame {frame_number} due to high pct of white or black pixels")
            continue
        
        processed_frame = preprocess_frame(frame)
        ## we have 3 rows to the killfeed, so we need to read 3 frames
        ## the next row is 25 pixels above the current row
        for i in range(3):
            text = read_killfeed_frame(processed_frame, bottom_right=ROI_BOTTOM_RIGHT, top_left=ROI_TOP_LEFT)
            results.append({"frame_number": frame_number, "row":i, "text": text})
            ROI_TOP_LEFT = (ROI_TOP_LEFT[0], ROI_TOP_LEFT[1] - 25)
            ROI_BOTTOM_RIGHT = (ROI_BOTTOM_RIGHT[0], ROI_BOTTOM_RIGHT[1] - 25)
        
        
        # text = read_killfeed_frame(processed_frame, bottom_right=ROI_BOTTOM_RIGHT, top_left=ROI_TOP_LEFT)
        # results.append({"frame_number": frame_number, "text": text})
    
    df = pd.DataFrame(results)
    ## write to csv
    df.to_csv("killfeed.csv", index=False)
    print(df)
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
