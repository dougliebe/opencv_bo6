import cv2
import numpy as np
import video_reader
import ocr_processor

def crop_and_save_images(image, coordinates, save_paths, team):
    for (top_left, bottom_right), save_path in zip(coordinates, save_paths):
        cropped_image = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        
        ## if the percent white is greater than 70%, invert the image
        pct_white = 100 * (cropped_image == 255).sum() / cropped_image.size
        if pct_white > 70:
            cropped_image = cv2.bitwise_not(cropped_image)
        
        ## create bounding box and crop each image again
        dilated_image = cv2.dilate(cropped_image, np.ones((5, 5), np.uint8), iterations=1)
        contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            # ## plot image with bounding box
            # cv2.rectangle(cropped_image, (x, y), (x+w, y+h), (255, 255, 0), 2)
            # cv2.imshow("Bounding Box", cropped_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            cropped_image = cropped_image[y:y+h, x:x+w]
            
            # ## create an image with team text only on it, then add it to the right of the image
            # team_image = np.zeros((cropped_image.shape[0], 30, 3), dtype=np.uint8)
            # cv2.putText(team_image, team, (1, team_image.shape[0] // 2 + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
            # # print shapes of cropped_image and team_image
            # print(f"Cropped Image shape: {cropped_image.shape}, Team Image shape: {team_image.shape}")
            # # cv2.imshow("Team Image", team_image)
            # # cv2.waitKey(0)
            # # cv2.destroyAllWindows()
            # if len(cropped_image.shape) == 2:  # if cropped_image is grayscale
            #     cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2BGR)
            # cropped_image = np.hstack((team_image, cropped_image))
            # ## add 5px of black space to the right of the image
            # cropped_image = cv2.copyMakeBorder(cropped_image, 0, 0, 30, 10, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            
        cv2.imwrite(save_path, cropped_image)

def template_match(image, templates):
    results = []
    for template_path in templates:
        template = cv2.imread(template_path, 0)
        ## print sizes of image and template
        # print(f"Image size: {image.shape}, Template size: {template.shape}")
        res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        results.append((template_path, max_val, max_loc))
    return results

def process_image(image):
    # image = video_reader.capture_image_frame(image_path)
    if image is None:
        print("Failed to capture or preprocess frame.")
        return

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
    save_paths = ["crop1.png", "crop2.png", "crop3.png", "crop4.png"]
    crop_and_save_images(image, coordinates, save_paths, team='[OG]')

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
    save_paths_2 = ["crop5.png", "crop6.png", "crop7.png", "crop8.png"]
    crop_and_save_images(image, coordinates, save_paths_2, team='[100T]')

    save_paths.extend(save_paths_2)

    top_left = (18, 374)
    bottom_right = (300, 395)
    search_region = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    search_region = cv2.resize(search_region, (search_region.shape[1] - 25, search_region.shape[0]), interpolation=cv2.INTER_LINEAR)

    results = template_match(search_region, save_paths)

    first_four_max_val = max(results[:4], key=lambda x: x[1])
    second_four_max_val = max(results[4:], key=lambda x: x[1])

    first_four_max_name = first_four_max_val[0]
    second_four_max_name = second_four_max_val[0]

    first_four_x_coord = first_four_max_val[2]
    second_four_x_coord = second_four_max_val[2]

    first_four_max_val = first_four_max_val[1]
    second_four_max_val = second_four_max_val[1]

    # if second_four_x_coord > first_four_x_coord:
    #     print(f"{first_four_max_name} killed {second_four_max_name}")
    # else:
    #     print(f"{second_four_max_name} killed {first_four_max_name}")
    if second_four_x_coord > first_four_x_coord:
        return f"{first_four_max_name} killed {second_four_max_name}"
    else:
        return f"{second_four_max_name} killed {first_four_max_name}"

    # for result in results:
    #     print(f"Template: {result[0]}, Max Value: {result[1]}, Location: {result[2]}")

if __name__ == "__main__":
    image_path = "images\OPTIC TEXAS VS LA THEIVES GRAND FINALS ($50K KAYSAN LAN) BLACK OPS 6 31-55 screenshot.png"
    image = video_reader.capture_image_frame(image_path)
    ## preprocess frame
    processed_frame = ocr_processor.preprocess_frame(image)
    
    process_image(image)