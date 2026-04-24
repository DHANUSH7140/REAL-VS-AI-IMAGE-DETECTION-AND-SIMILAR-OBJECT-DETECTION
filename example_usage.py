"""
Example Usage: Similar Object Detection Module Integration

This script demonstrates how to run the newly integrated Got-Chu
similarity detector independently, passing an image and an ROI bounding box.
It will output an annotated image with the detected similar objects highlighted.

Usage:
    python example_usage.py
"""
import os
import cv2
import sys

# Ensure the root path is accessible to import the wrapper service
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'real-vs-ai-detector')))

from services.similar_detector_api import detect_similar_objects

def create_dummy_image(path):
    """Creates a dummy test image with some repeating patterns."""
    import numpy as np
    # Create a 512x512 gray image
    img = np.ones((512, 512, 3), dtype=np.uint8) * 200
    # Draw three distinct blue squares (our "similar objects")
    cv2.rectangle(img, (50, 50), (150, 150), (255, 0, 0), -1)
    cv2.rectangle(img, (250, 50), (350, 150), (255, 0, 0), -1)
    cv2.rectangle(img, (150, 300), (250, 400), (255, 0, 0), -1)
    # Draw some noise/distractors
    cv2.circle(img, (400, 400), 50, (0, 255, 0), -1)
    cv2.imwrite(path, img)
    return path

def main():
    test_image_path = "test_similarity.jpg"
    
    print("1. Creating a sample test image with repeating patterns...")
    create_dummy_image(test_image_path)
    print(f"   -> Saved test image to {test_image_path}\n")

    # Define our Region of Interest (ROI) bounding box [x1, y1, x2, y2]
    # We select the first blue square from (50, 50) to (150, 150)
    roi_bbox = [50, 50, 150, 150]
    
    print(f"2. Running Similar Object Detection on ROI: {roi_bbox}...")
    
    # Call the isolated Wrapper API
    result = detect_similar_objects(test_image_path, roi_bbox)
    
    if result.get("error"):
        print(f"❌ Error: {result['error']}")
    else:
        count = result.get("count", 0)
        boxes = result.get("boxes", [])
        print(f"✅ Success! Detected {count} similar objects.")
        print("   Bounding Boxes:", boxes)
        
        # In a web app, the 'annotated_image_b64' can be passed directly to an <img> tag:
        # e.g., <img src="{{ result.annotated_image_b64 }}">
        if result.get("annotated_image_b64"):
            print("   -> Generated base64 annotated image successfully.")

    # Cleanup
    if os.path.exists(test_image_path):
        os.remove(test_image_path)

if __name__ == "__main__":
    main()
