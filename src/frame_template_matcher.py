import cv2
import numpy as np
import video_reader
import ocr_processor
import os

def crop_and_save_images(image, coordinates, save_paths):
    # print(f"Image size: {image.shape}")
    for (top_left, bottom_right), save_path in zip(coordinates, save_paths):
        
        cropped_image = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        
        ## if the percent white is greater than 70%, invert the image
        pct_white = 100 * (cropped_image == 255).sum() / cropped_image.size
        if pct_white > 70:
            cropped_image = cv2.bitwise_not(cropped_image)
        
        ## create bounding box and crop each image again
        dilated_image = cv2.dilate(cropped_image, np.ones((5, 5), np.uint8), iterations=1)
        contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue  # Skip if no contours found
        
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            cropped_image = cropped_image[y:y+h, x:x+w]
            
        cv2.imwrite(save_path, cropped_image)

def template_match(image, templates):
    results = []
    for template_path in templates:
        template = cv2.imread(template_path, 0)
        dilated_template = cv2.dilate(template, np.ones((2, 2), np.uint8), iterations=1)
        
        res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        res_dilated = cv2.matchTemplate(image, dilated_template, cv2.TM_CCOEFF_NORMED)
        min_val_dilated, max_val_dilated, min_loc_dilated, max_loc_dilated = cv2.minMaxLoc(res_dilated)
        
        if max_val_dilated > max_val:
            results.append((template_path, max_val_dilated, max_loc_dilated))
        else:
            results.append((template_path, max_val, max_loc))
    return results

def update_cropped_scoreboard_names(image):
    if image is None:
        print("Failed to capture or preprocess frame.")
        return
    # print(f"Image size: {image.shape}")
    top_left_x, top_left_y = 892, 65
    bottom_right_x, bottom_right_y = 1041, 148
    width = bottom_right_x - top_left_x
    height = (bottom_right_y - top_left_y) // 4

    coordinates = [
        ((top_left_x, top_left_y), (bottom_right_x, top_left_y + height)),
        ((top_left_x, top_left_y + height), (bottom_right_x, top_left_y + 2 * height)),
        ((top_left_x, top_left_y + 2 * height), (bottom_right_x, top_left_y + 3 * height)),
        ((top_left_x, top_left_y + 3 * height), (bottom_right_x, bottom_right_y))
    ]
    save_paths = ["templates/scoreboard_names/crop1.png", "templates/scoreboard_names/crop2.png", "templates/scoreboard_names/crop3.png", "templates/scoreboard_names/crop4.png"]
    crop_and_save_images(image, coordinates, save_paths)

    top_left_x, top_left_y = 76, 67
    bottom_right_x, bottom_right_y = 177, 147
    width = bottom_right_x - top_left_x
    height = (bottom_right_y - top_left_y) // 4

    coordinates = [
        ((top_left_x, top_left_y), (bottom_right_x, top_left_y + height)),
        ((top_left_x, top_left_y + height), (bottom_right_x, top_left_y + 2 * height)),
        ((top_left_x, top_left_y + 2 * height), (bottom_right_x, top_left_y + 3 * height)),
        ((top_left_x, top_left_y + 3 * height), (bottom_right_x, bottom_right_y))
    ]
    save_paths_2 = ["templates/scoreboard_names/crop5.png", "templates/scoreboard_names/crop6.png", "templates/scoreboard_names/crop7.png", "templates/scoreboard_names/crop8.png"]
    crop_and_save_images(image, coordinates, save_paths_2)

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

def read_killfeed_frame(image, top_left=(300, 395), bottom_right=(18, 374)):
    # print(f"Image size: {image.shape}")
    # image = video_reader.capture_image_frame(image_path)
    
    ## crop to the killfeed region using provided top_left and bottom_right
    search_region = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    
    ## mask out the gun
    search_region = template_match_and_mask(search_region)
    
    ## scrunch the image to make it easier to match
    search_region = cv2.resize(search_region, (search_region.shape[1] - 25, search_region.shape[0]), interpolation=cv2.INTER_LINEAR)

    ## save paths = all files in ./templates/scoreboard_names/ with png
    save_paths = ["templates/scoreboard_names/" + file for file in os.listdir("templates/scoreboard_names/") if file.endswith(".png")]

    results = template_match(search_region, save_paths)
    
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    if len(sorted_results) < 2 or sorted_results[0][1] <= 0.6 or sorted_results[1][1] <= 0.6:
        return 'no matches'

    first_four_max_val = max(results[:4], key=lambda x: x[1])
    second_four_max_val = max(results[4:], key=lambda x: x[1])

    first_four_max_name = first_four_max_val[0]
    second_four_max_name = second_four_max_val[0]
    first_four_max_name = os.path.basename(first_four_max_name)
    second_four_max_name = os.path.basename(second_four_max_name)

    first_four_x_coord = first_four_max_val[2]
    second_four_x_coord = second_four_max_val[2]

    first_four_max_val = first_four_max_val[1]
    second_four_max_val = second_four_max_val[1]

    if second_four_x_coord > first_four_x_coord:
        print(f"{first_four_max_name} killed {second_four_max_name} with confidence {first_four_max_val:.2f} / {second_four_max_val:.2f}")
    else:
        print(f"{second_four_max_name} killed {first_four_max_name} with confidence {second_four_max_val:.2f} / {first_four_max_val:.2f}")

    if second_four_x_coord > first_four_x_coord:
        return f"{first_four_max_name} killed {second_four_max_name}"
    else:
        return f"{second_four_max_name} killed {first_four_max_name}"

    # for result in results:
    #     print(f"Template: {result[0]}, Max Value: {result[1]}, Location: {result[2]}")

if __name__ == "__main__":
    image_path = "frame_15600.png"
    image = video_reader.capture_image_frame(image_path)
    ## preprocess frame
    processed_frame = ocr_processor.preprocess_frame(image)
    
    # print size of image
    # print(f"Image size: {processed_frame.shape}")
    
    read_killfeed_frame(processed_frame)