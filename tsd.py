# libraries needed
import cv2 as cv
import numpy as np
import argparse
import time


#pop up window
def show_img(window_name, img, adjust=False):
    """3 arguments: window name, source images, boolean to adjust to screen size"""
    if adjust:
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    else:
        cv.namedWindow(window_name)

    cv.imshow(window_name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()

# Filter contours based on area and aspect ratio
def is_valid_contour(contour, min_area, max_area, min_aspect_ratio, max_aspect_ratio):
    area = cv.contourArea(contour)
    if area < min_area or area > max_area:
        return False
    x, y, w, h = cv.boundingRect(contour)
    aspect_ratio = w / float(h)
    if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
        return False
    return True

#Shape based detection start
def image_preprocessing(image):

    fixed_width=150
    scale_factor = fixed_width/image.shape[1] 
    upscaled_image = cv.resize(image, None,fx=scale_factor,fy=scale_factor, interpolation=cv.INTER_AREA)

    #Denoising
    denoised_img = cv.fastNlMeansDenoisingColored(upscaled_image, None, 10, 10, 7, 21)

    #Sharpening
    kernel = np.array([[0, -1, 0], 
                   [-1, 5,-1], 
                   [0, -1, 0]])
    sharpened_image = cv.filter2D(denoised_img, -1, kernel)

    return sharpened_image, scale_factor

def detect_edges(image):
    """
    Detect edges in an image using a pre-trained structured edge detection model.
    Returns:
        numpy.ndarray: Edge map.
    """

    if image is None:
        raise ValueError("Failed to load the image.")

    image=image.astype(np.float32)/255

    # Load the pre-trained model
    model_path = 'model.yml'
    model = cv.ximgproc.createStructuredEdgeDetection(model_path)

    # Detect edges using the pre-trained model
    edges = model.detectEdges(image)

    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv.dilate(edges, kernel, iterations=1)
    eroded_edges = cv.erode(dilated_edges, kernel, iterations=1)
    
    return eroded_edges

# canny edge detection
def canny_edge_detection(image):
    #Denoising
    img = cv.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 100,200)
    ###edges = cv.GaussianBlur(edges, (5, 5), 0)
    return edges

def detect_circles(scale_factor,image,pre_image, dp=1, min_dist=100, param1=150, param2=20, min_radius=10, max_radius=100):
    """
    Detect circles in an edge-detected image using Hough Circle Transform with adjustable parameters.
    
    Parameters:
    - img: Original image (for drawing detected circles)
    - edges: Edge-detected image (for circle detection)
    - dp: Inverse ratio of accumulator resolution to image resolution (higher values -> smaller circles detected)
    - min_dist: Minimum distance between detected circle centers
    - param1: Higher threshold for Canny edge detection (lower = more sensitive to edges)
    - param2: Accumulator threshold for circle detection (lower = more circles, higher = stricter)
    - min_radius: Minimum radius of detected circles
    - max_radius: Maximum radius of detected circles
    
    Returns:
    - Image with detected circles drawn, or None if no circles are found.
    """
    edge_image=detect_edges(pre_image)
    # Convert edges to 8-bit single-channel (required by HoughCircles)
    edges_8bit = np.uint8(edge_image * 255)

    # Detect circles using Hough Transform with tunable parameters
    circles = cv.HoughCircles(edges_8bit, 
                              cv.HOUGH_GRADIENT, 
                              dp=dp, 
                              minDist=min_dist, 
                              param1=param1, 
                              param2=param2, 
                              minRadius=min_radius, 
                              maxRadius=max_radius)

    if circles is None:
        return None  # Return None if no circles are found

    # Convert the (x, y, radius) to integers
    circles = np.uint16(np.around(circles))
    
    # Find the largest circle based on radius
    largest_circle = max(circles[0, :], key=lambda x: x[2])
    center_x, center_y, radius = largest_circle
    
    # Copy the original image to draw the circle
    output_img = image.copy()

    ori_x = int(center_x / scale_factor)
    ori_y = int(center_y / scale_factor)
    ori_radius = int(radius / scale_factor)
    
    circle_contour = np.array([[ori_x - ori_radius, ori_y- ori_radius],
                                [ori_x + ori_radius, ori_y - ori_radius],
                                [ori_x + ori_radius, ori_y + ori_radius],
                                [ori_x - ori_radius, ori_y + ori_radius]], dtype=np.int32)
    circle_contour = np.array((circle_contour * scale_factor), dtype=np.int32)
    
    return circle_contour

def detect_triangles(pre_image):
    # Perform Canny edge detection
    tri_edges = canny_edge_detection(pre_image)
    
    # Find contours from the edge-detected image
    contours, _ = cv.findContours(tri_edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Initialize variables to store the largest triangle contour
    largest_area = 0
    largest_triangle_contour = None

    # Define area and aspect ratio thresholds
    min_area = 500
    max_area = 0.25 * (min(pre_image.shape[:2]) ** 2)
    min_aspect_ratio = 1 / 1.9
    max_aspect_ratio = 1.9

    # Iterate over all detected contours
    for contour in contours:
        # Apply Douglas-Peucker algorithm to simplify the contour
        epsilon = 0.03 * cv.arcLength(contour, True)  # Precision parameter
        approx_contour = cv.approxPolyDP(contour, epsilon, True)

        # Check if the simplified contour has 3 vertices (i.e., forms a triangle)
        if len(approx_contour) == 3:
            # Validate the contour using the is_valid_contour function
            if is_valid_contour(approx_contour, min_area, max_area, min_aspect_ratio, max_aspect_ratio):
                area = cv.contourArea(approx_contour)

                # Check if this triangle has the largest area found so far
                if area > largest_area:
                    largest_area = area
                    largest_triangle_contour = approx_contour

    if largest_triangle_contour is not None:
            # Create contour for the triangle
            triangle_contour = np.array(largest_triangle_contour, dtype=np.int32)
            # Return contour and the bounding box
            return triangle_contour
        
    else:
        return None
    
# combine the results of 2 shape detectors
def integrate_circle_tri(tri_cnt, circle_cnt):
    if circle_cnt is not None and tri_cnt is not None:
        # compare the area
        if cv.contourArea(circle_cnt) >= cv.contourArea(tri_cnt):
            output = circle_cnt
        else:
            output = tri_cnt

    elif circle_cnt is not None and tri_cnt is None:
        output = circle_cnt

    elif circle_cnt is None and tri_cnt is not None:
        output = tri_cnt

    else:
        output = np.array([])

    return output

#Integrate two shape detection
def integrate_shape_detections(pre_image):
    # Detect circles and triangles
    circle_contour = detect_circles(scale_factor,image, pre_image)
    triangle_contour = detect_triangles(pre_image)

    # Integrate results based on area
    final_contour = integrate_circle_tri(triangle_contour, circle_contour)

    output_image = pre_image.copy()

    if len(final_contour) > 0:
        return final_contour 
    else:
        return None

#Color Based Detection Start
def image_enhancement(image):

    fixed_width=150
    scale_factor = fixed_width/image.shape[1] 
    resized_image = cv.resize(image, None,fx=scale_factor,fy=scale_factor, interpolation=cv.INTER_AREA)

    # denoising
    img = cv.fastNlMeansDenoisingColored(resized_image, None, 10, 10, 7, 21)
    
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv_img)
    gamma = 0.5
    saturation_scale = 1.5
    value_scale = v / 255.0
    
    # Adjust saturation
    s_adjusted = np.clip(s * saturation_scale, 0, 255).astype(np.uint8)
    
    # Apply gamma correction to Value channel
    v_adjusted = np.clip(cv.pow(v * value_scale, gamma) * 255.0, 0, 255).astype(np.uint8)
    
    hsv_adjusted = cv.merge([h, s_adjusted, v_adjusted])
    enhanced_img = cv.cvtColor(hsv_adjusted, cv.COLOR_HSV2BGR)
   
    return enhanced_img , scale_factor

def color_segmentation(image):
    
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV) 
    
    lower_red1 = np.array([0, 100, 90])
    upper_red1 = np.array([18, 255, 255])
    lower_red2 = np.array([144, 100, 90])
    upper_red2 = np.array([180, 255, 255])
    mask_red1 = cv.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv.bitwise_or(mask_red1, mask_red2)

    lower_yellow = np.array([90, 102, 102])
    upper_yellow = np.array([126, 255, 255])
    yellow_mask = cv.inRange(hsv, lower_yellow, upper_yellow)
    
    lower_blue = np.array([16, 90, 102])
    upper_blue = np.array([32, 255, 255])
    blue_mask = cv.inRange(hsv, lower_blue, upper_blue)

    mask = cv.bitwise_or(red_mask, blue_mask)
    mask = cv.bitwise_or(mask, yellow_mask)

    #ADD BLACK COLOR
    
    segmented_image = cv.bitwise_and(image, image, mask=mask)
    return segmented_image, mask

def color_detection(image):
    
    segmented_image, mask = color_segmentation(image)
    
    ret, binary_image = cv.threshold(mask, 175, 255, cv.THRESH_BINARY)
    
    #Morphological Transform
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    cleaned_image = cv.morphologyEx(binary_image, cv.MORPH_CLOSE, kernel)
    cleaned_image = cv.morphologyEx(binary_image, cv.MORPH_OPEN, kernel)
    
    contours, _ = cv.findContours(cleaned_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    min_area = 800
    max_area = 0.25 * (min(image.shape[:2]) ** 2)
    min_aspect_ratio = 1/1.9
    max_aspect_ratio = 1.9
    
    largest_contour= None
    
    valid_contours = []
    for contour in contours:
        if is_valid_contour(contour, min_area, max_area, min_aspect_ratio, max_aspect_ratio):
            valid_contours.append(contour)
    
    if valid_contours:
        largest_contour = max(valid_contours, key=cv.contourArea)
        x, y, w, h = cv.boundingRect(largest_contour)

    if largest_contour is not None:
        largest_contour = np.array(largest_contour, dtype=np.int32)
        return largest_contour
    else:
        return None
    
def integrate_edge_color(output1, output2):
    if output1 is None and output2 is None:
        return np.array([])
    
    if output1 is None or output2 is None or (isinstance(output1, np.ndarray) and len(output1) == 0) or (isinstance(output2, np.ndarray) and len(output2) == 0):
        return output1 if output1 is not None else output2

    # If both outputs are valid contours, compare their areas
    if isinstance(output1, np.ndarray) and isinstance(output2, np.ndarray):
        if len(output1) > 0 and len(output2) > 0:
            area1 = cv.contourArea(output1)
            area2 = cv.contourArea(output2)
            return output1 if area1 > area2 else output2



# Main operations
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--fn", required=True, help="filename")
args = vars(ap.parse_args())

# Operation start
# Read image
start_time = time.time()
image = cv.imread(cv.samples.findFile(args["fn"]))
image_copy = image.copy()

pre_image, scale_factor = image_preprocessing(image)
shape_cnt  = integrate_shape_detections(pre_image)
enh_img, scale_factor = image_enhancement(image)
color_cnt = color_detection(enh_img)
final_output = integrate_edge_color(color_cnt,shape_cnt)

# Take the execution time
print(f"The execution time of this pipeline: {(time.time()-start_time):.3f}s")

if len(final_output) == 0:
    print("no detection!")
    show_img("no detection", image)
else:
    x, y, w, h = cv.boundingRect(final_output)
    cv.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
    show_img("results", image_copy)