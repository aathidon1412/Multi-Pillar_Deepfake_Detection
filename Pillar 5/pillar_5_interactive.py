import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import math
import os
import argparse

def calculate_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return (px, py)

# Global variables for drawing
drawing = False
ix, iy = -1, -1
user_lines = []
img_copy = None

def draw_line(event, x, y, flags, param):
    global ix, iy, drawing, img_copy, user_lines
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(img_copy, (ix, iy), (x, y), (0, 255, 255), 2)
        cv2.imshow('Draw Shadows', img_copy)
        user_lines.append([ix, iy, x, y])

def analyze_interactive(image_path, output_dir="."):
    global img_copy, user_lines
    os.makedirs(output_dir, exist_ok=True)
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
        
    vis_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Scale down for viewing if it's too big for the screen
    screen_height = 800
    scale = 1.0
    if img.shape[0] > screen_height:
        scale = screen_height / img.shape[0]
        img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))

    img_copy = img.copy()
    
    print("--------------------------------------------------")
    print("INTERACTIVE MODE:")
    print("1. Click and drag to draw lines along the TRUE cast shadows in the image.")
    print("2. Draw at least 2 lines.")
    print("3. Press ENTER when you are done.")
    print("--------------------------------------------------")
    
    cv2.namedWindow('Draw Shadows')
    cv2.setMouseCallback('Draw Shadows', draw_line)
    cv2.imshow('Draw Shadows', img_copy)
    
    # Wait for enter key
    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == 13: # Enter key
            break
            
    cv2.destroyAllWindows()
    
    if len(user_lines) < 2:
        print("You must draw at least 2 lines! Exiting.")
        return
        
    # Rescale lines back to original image size
    lines = []
    for line in user_lines:
        lines.append([int(coord / scale) for coord in line])
        
    line_data = []
    for i, line in enumerate(lines):
        x1, y1, x2, y2 = line
        m = float('inf') if x2 - x1 == 0 else (y2 - y1) / (x2 - x1)
        line_data.append({'id': i, 'line': line, 'm': m})
        
    intersections = []
    intersection_data = []
    
    for i in range(len(line_data)):
        for j in range(i + 1, len(line_data)):
            l1 = line_data[i]
            l2 = line_data[j]
            pt = calculate_intersection(l1['line'], l2['line'])
            if pt is not None:
                intersections.append(pt)
                intersection_data.append({
                    'Line Pair ID': f"{l1['id']}-{l2['id']}",
                    'Slope m1': round(l1['m'], 4) if not math.isinf(l1['m']) else 'inf',
                    'Slope m2': round(l2['m'], 4) if not math.isinf(l2['m']) else 'inf',
                    'Intersection (X, Y)': (round(pt[0], 2), round(pt[1], 2))
                })
                
    if not intersections:
        print("No intersections found among lines.")
        return
        
    centroid_x = np.median([pt[0] for pt in intersections])
    centroid_y = np.median([pt[1] for pt in intersections])
    
    distances = []
    for d in intersection_data:
        pt = d['Intersection (X, Y)']
        dist = math.sqrt((pt[0] - centroid_x)**2 + (pt[1] - centroid_y)**2)
        d['Distance from Centroid (px)'] = round(dist, 2)
        distances.append(dist)
        
    # With manual lines, variance should be very low if authentic
    variance_metric = np.mean(distances) if distances else 0.0
    
    threshold = 45.0
    if variance_metric < threshold:
        verdict = "AUTHENTIC PHYSICS"
    else:
        verdict = "PHYSICS ANOMALY (AI GENERATED)"
        
    # Generate Artifacts
    df = pd.DataFrame(intersection_data)
    df.to_csv(os.path.join(output_dir, "pillar_5_interactive_table.csv"), index=False)
    
    telemetry = {
        "pillar": "Pillar 5 - Interactive Physics",
        "vectors_analyzed": len(lines),
        "geometric_variance_score": round(float(variance_metric), 3),
        "estimated_vanishing_point": [round(float(centroid_x), 2), round(float(centroid_y), 2)],
        "verdict": verdict
    }
    with open(os.path.join(output_dir, "pillar_5_interactive_telemetry.json"), 'w') as f:
        json.dump(telemetry, f, indent=4)
        
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(vis_img)
    
    for l_data in line_data:
        x1, y1, x2, y2 = l_data['line']
        ax.plot([x1, x2], [y1, y2], color='cyan', linewidth=3)
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ext = max(vis_img.shape[0], vis_img.shape[1]) * 2
        if math.isinf(l_data['m']):
            ray_x1, ray_y1 = mid_x, mid_y - ext
            ray_x2, ray_y2 = mid_x, mid_y + ext
        else:
            angle = math.atan(l_data['m'])
            ray_x1 = mid_x - ext * math.cos(angle)
            ray_y1 = mid_y - ext * math.sin(angle)
            ray_x2 = mid_x + ext * math.cos(angle)
            ray_y2 = mid_y + ext * math.sin(angle)
        ax.plot([ray_x1, ray_x2], [ray_y1, ray_y2], color='green', linestyle=':', alpha=0.5)

    ax.scatter([centroid_x], [centroid_y], color='red', s=100, zorder=5)
    
    # Don't restrict the view tightly so we can see the vanishing point
    bounds_x = [0, vis_img.shape[1], centroid_x]
    bounds_y = [0, vis_img.shape[0], centroid_y]
    ax.set_xlim([min(bounds_x) - 100, max(bounds_x) + 100])
    ax.set_ylim([max(bounds_y) + 100, min(bounds_y) - 100])
    
    textstr = '\n'.join((
        f'Interactive Variance: {variance_metric:.2f} px',
        f'Threshold: {threshold:.2f} px',
        f'Verdict: {verdict}'
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props)
            
    ax.set_title("Interactive Shadow Forensics")
    ax.axis('off')
    
    plot_path = os.path.join(output_dir, "pillar_5_interactive_plot.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nAnalysis complete! Verdict: {verdict} (Variance: {variance_metric:.2f} px)")
    print(f"Results saved to {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default=".")
    args = parser.parse_args()
    analyze_interactive(args.image, args.output)
