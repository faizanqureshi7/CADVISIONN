import cv2
import numpy as np
from PIL import Image
import google.generativeai as genai
import os
from dotenv import load_dotenv
import textwrap


def create_summary_visualization(img1, aligned_img2, highlighted):
    """
    Create a 1x3 visualization with proper dimension handling.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = aligned_img2.shape[:2]
    h3, w3 = highlighted.shape[:2]

    max_h = max(h1, h2, h3)
    max_w = max(w1, w2, w3)

    img1_resized = cv2.resize(img1, (max_w, max_h)) if (h1, w1) != (max_h, max_w) else img1.copy()
    img2_resized = cv2.resize(aligned_img2, (max_w, max_h)) if (h2, w2) != (max_h, max_w) else aligned_img2.copy()
    img3_resized = cv2.resize(highlighted, (max_w, max_h)) if (h3, w3) != (max_h, max_w) else highlighted.copy()

    composite = np.ones((max_h, max_w * 3, 3), dtype=np.uint8) * 255
    composite[0:max_h, 0:max_w] = img1_resized
    composite[0:max_h, max_w:2*max_w] = img2_resized
    composite[0:max_h, 2*max_w:3*max_w] = img3_resized

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 3

    labels = [
        ("Original Drawing", (10, 50)),
        ("Revised Drawing", (max_w + 10, 50)),
        ("Changes Highlighted", (2*max_w + 10, 50))
    ]

    for label, (x, y) in labels:
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.rectangle(composite, (x-5, y-text_h-10), (x+text_w+5, y+5), (255, 255, 255), -1)
        cv2.rectangle(composite, (x-5, y-text_h-10), (x+text_w+5, y+5), (0, 0, 0), 2)
        cv2.putText(composite, label, (x, y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    return composite


def generate_summary_with_gemini(composite_image, api_key=None):
    """
    Generate summary using Gemini Vision API with better error handling.
    """
    if api_key is None:
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return "⚠️ GEMINI_API_KEY not found in .env file. Please add:\nGEMINI_API_KEY=your_key_here"

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')

        image_rgb = cv2.cvtColor(composite_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        prompt = (
            "You are analyzing a CAD/technical drawing revision comparison. The image shows three views:\n\n"
            "**Left**: Original drawing\n"
            "**Middle**: Revised drawing  \n"
            "**Right**: Highlighted changes (Green=additions, Red=deletions, Yellow=modifications)\n\n"
            "Provide a concise, engineering-focused summary (150-250 words) covering:\n\n"
            "1. **Major Structural Changes**: Dimensions, shapes, component positions\n"
            "2. **Text/Label Changes**: Dimension values, part numbers, annotations\n"
            "3. **Color Analysis**:\n   - Red: Removed elements\n   - Green: Added elements\n   - Yellow: Modified/overlapping regions\n\n"
            "Be specific and quantitative where possible. Focus on actionable engineering insights."
        )

        print("   🤖 Sending request to Gemini...")
        response = model.generate_content([prompt, pil_image])

        if response and getattr(response, 'text', None):
            print("   ✅ Summary generated successfully")
            return response.text
        else:
            return "⚠️ Gemini returned empty response. Please check API quota and image quality."

    except Exception as e:
        error_msg = str(e)
        if "API_KEY_INVALID" in error_msg:
            return "❌ Invalid API key. Please check your GEMINI_API_KEY in .env"
        elif "QUOTA_EXCEEDED" in error_msg:
            return "❌ API quota exceeded. Please check your Gemini API usage limits."
        elif "SAFETY" in error_msg:
            return "⚠️ Content filtered by safety settings. Try adjusting image content."
        else:
            return f"❌ Error generating summary: {error_msg}"


def analyze_change_types(highlighted):
    """
    Analyze the types of changes based on color (improved color detection).
    """
    hsv = cv2.cvtColor(highlighted, cv2.COLOR_BGR2HSV)

    red_mask1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
    red_mask2 = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    green_mask = cv2.inRange(hsv, (40, 100, 100), (80, 255, 255))

    yellow_mask = cv2.inRange(hsv, (20, 100, 100), (40, 255, 255))

    total_pixels = highlighted.shape[0] * highlighted.shape[1]
    red_pixels = np.sum(red_mask > 0)
    green_pixels = np.sum(green_mask > 0)
    yellow_pixels = np.sum(yellow_mask > 0)

    return {
        "deletions_percent": (red_pixels / total_pixels) * 100,
        "additions_percent": (green_pixels / total_pixels) * 100,
        "modifications_percent": (yellow_pixels / total_pixels) * 100,
        "total_change_percent": ((red_pixels + green_pixels + yellow_pixels) / total_pixels) * 100,
        "deletions_pixels": int(red_pixels),
        "additions_pixels": int(green_pixels),
        "modifications_pixels": int(yellow_pixels)
    }


def generate_change_summary(img1, aligned_img2, highlighted, api_key=None):
    """
    Generate complete summary with error handling.
    """
    print("   📊 Creating summary visualization...")
    composite = create_summary_visualization(img1, aligned_img2, highlighted)

    print("   🔍 Analyzing change types...")
    stats = analyze_change_types(highlighted)

    print("   🤖 Generating AI summary...")
    ai_summary = generate_summary_with_gemini(composite, api_key)

    return {
        "ai_summary": ai_summary,
        "statistics": stats,
        "composite_image": composite
    }


def create_summary_panel(summary_data, width=1800, height=1200):
    """
    Create text panel with automatic sizing.
    """
    panel = np.ones((height, width, 3), dtype=np.uint8) * 255

    font = cv2.FONT_HERSHEY_SIMPLEX
    title_font_scale = 1.8
    text_font_scale = 0.9
    line_spacing = 45
    margin_left = 40
    margin_right = 40
    y_pos = 80

    cv2.putText(panel, "CAD REVISION ANALYSIS REPORT", (margin_left, y_pos),
                font, title_font_scale, (0, 0, 128), 3, cv2.LINE_AA)
    y_pos += 70

    cv2.line(panel, (margin_left, y_pos), (width - margin_right, y_pos), (0, 0, 0), 2)
    y_pos += 60

    stats = summary_data["statistics"]
    cv2.putText(panel, "CHANGE STATISTICS:", (margin_left, y_pos),
                font, text_font_scale * 1.2, (0, 0, 128), 2, cv2.LINE_AA)
    y_pos += 50

    stats_lines = [
        f"Total Changes: {stats['total_change_percent']:.2f}%",
        f"Additions (Green): {stats['additions_percent']:.2f}%",
        f"Deletions (Red): {stats['deletions_percent']:.2f}%",
        f"Modifications (Yellow): {stats['modifications_percent']:.2f}%"
    ]

    for line in stats_lines:
        cv2.putText(panel, line, (margin_left + 20, y_pos),
                    font, text_font_scale, (30, 30, 30), 2, cv2.LINE_AA)
        y_pos += line_spacing

    y_pos += 40

    cv2.putText(panel, "AI ANALYSIS:", (margin_left, y_pos),
                font, text_font_scale * 1.2, (0, 0, 128), 2, cv2.LINE_AA)
    y_pos += 60

    ai_text = summary_data["ai_summary"]
    ai_text = ai_text.replace("###", "").replace("##", "").replace("**", "")

    usable_width = width - margin_left - margin_right
    max_chars_per_line = int(usable_width / 13)

    wrapped_lines = []
    for line in ai_text.split('\n'):
        line = line.strip()
        if line:
            if len(line) > max_chars_per_line:
                wrapped_lines.extend(textwrap.wrap(line, width=max_chars_per_line, break_long_words=False))
            else:
                wrapped_lines.append(line)
        else:
            wrapped_lines.append("")

    for line in wrapped_lines:
        if y_pos > height - 80:
            cv2.putText(panel, "... (see full report in text file)",
                        (margin_left, y_pos),
                        font, text_font_scale * 0.8, (128, 128, 128), 2, cv2.LINE_AA)
            break

        if line.strip():
            is_header = line[0].isdigit() or (len(line) > 2 and line[1] == '.')
            color = (0, 0, 128) if is_header else (30, 30, 30)
            thickness = 3 if is_header else 2

            cv2.putText(panel, line, (margin_left + 20, y_pos),
                        font, text_font_scale, color, thickness, cv2.LINE_AA)
            y_pos += line_spacing + (10 if is_header else 0)
        else:
            y_pos += line_spacing // 2

    return panel


