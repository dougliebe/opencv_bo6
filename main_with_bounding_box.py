import sys
import os
import cv2

# Add the src directory to the sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# main_with_bounding_box.py
from src.video_reader import capture_frame
from src.ocr_processor import preprocess_frame

def click_event(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        if len(points) == 2:
            cv2.destroyAllWindows()

def template_match_and_mask(frame):
    templates_dir = "templates/guns/"
    masked_frame = frame.copy()
    
    for template_name in os.listdir(templates_dir):
        template_path = os.path.join(templates_dir, template_name)
        template = cv2.imread(template_path, 0)
        if template is None:
            continue
        
        res = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        if max_val > 0.50:
            top_left = max_loc
            h, w = template.shape
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv2.rectangle(masked_frame, top_left, bottom_right, (0, 0, 0), -1)
    
    return masked_frame

def main():
    video_path = "C:/Users/liebe/Videos/4K Video Downloader+/OPTIC TEXAS VS LA THEIVES GRAND FINALS ($50K KAYSAN LAN) BLACK OPS 6.mp4"
    start_frame = (60 * (4*60 + 20)) 
    end_frame = (60 * (4*60 + 40))
    
    global points
    points = []

    for frame_number in range(start_frame, end_frame + 1, 60):
        # Step 1: Capture frame
        frame = capture_frame(video_path, frame_number)
        if frame is None:
            print(f"No frame captured at frame {frame_number}. Skipping.")
            continue
        
        # Step 2: Show frame and get bounding box
        cv2.imshow("Frame", frame)
        cv2.setMouseCallback("Frame", click_event)
        cv2.waitKey(0)
        
        if len(points) == 2:
            top_left, bottom_right = points
            cropped_image = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            ## process frame
            processed_frame = preprocess_frame(cropped_image)
            
            # Step 3: Template match and mask
            masked_frame = template_match_and_mask(processed_frame)
            
            # Save cropped image
            save_path = f"templates/guns/frame_{frame_number}.png"
            cv2.imwrite(save_path, masked_frame)
            print(f"Saved cropped image to {save_path}")
        
        points = []  # Reset points for the next frame
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()