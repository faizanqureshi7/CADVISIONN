import cv2
import numpy as np
from PIL import Image
import os
from dotenv import load_dotenv
import textwrap
from typing import Dict, Any, Optional
import base64
from io import BytesIO


def create_summary_visualization(img1: np.ndarray, 
                                 aligned_img2: np.ndarray, 
                                 highlighted: np.ndarray) -> np.ndarray:
    """
    Create a 1x3 visualization with proper dimension handling.
    
    Args:
        img1: Original image
        aligned_img2: Aligned revised image
        highlighted: Highlighted changes image
        
    Returns:
        Composite visualization image
    """
    try:
        # Get dimensions
        try:
            h1, w1 = img1.shape[:2]
            h2, w2 = aligned_img2.shape[:2]
            h3, w3 = highlighted.shape[:2]
        except Exception as e:
            raise RuntimeError(f"Failed to get image dimensions: {e}")

        try:
            max_h = max(h1, h2, h3)
            max_w = max(w1, w2, w3)
        except Exception as e:
            raise RuntimeError(f"Failed to calculate max dimensions: {e}")

        # Resize images to match
        try:
            img1_resized = cv2.resize(img1, (max_w, max_h)) if (h1, w1) != (max_h, max_w) else img1.copy()
            img2_resized = cv2.resize(aligned_img2, (max_w, max_h)) if (h2, w2) != (max_h, max_w) else aligned_img2.copy()
            img3_resized = cv2.resize(highlighted, (max_w, max_h)) if (h3, w3) != (max_h, max_w) else highlighted.copy()
        except Exception as e:
            raise RuntimeError(f"Failed to resize images: {e}")

        # Create composite
        try:
            composite = np.ones((max_h, max_w * 3, 3), dtype=np.uint8) * 255
            composite[0:max_h, 0:max_w] = img1_resized
            composite[0:max_h, max_w:2*max_w] = img2_resized
            composite[0:max_h, 2*max_w:3*max_w] = img3_resized
        except Exception as e:
            raise RuntimeError(f"Failed to create composite image: {e}")

        # Add labels
        try:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5
            thickness = 3

            labels = [
                ("Original Drawing", (10, 50)),
                ("Revised Drawing", (max_w + 10, 50)),
                ("Changes Highlighted", (2*max_w + 10, 50))
            ]

            for label, (x, y) in labels:
                try:
                    (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
                    cv2.rectangle(composite, (x-5, y-text_h-10), (x+text_w+5, y+5), (255, 255, 255), -1)
                    cv2.rectangle(composite, (x-5, y-text_h-10), (x+text_w+5, y+5), (0, 0, 0), 2)
                    cv2.putText(composite, label, (x, y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
                except Exception as e:
                    print(f"Warning: Failed to add label '{label}': {e}")
                    continue
        except Exception as e:
            print(f"Warning: Failed to add labels to composite: {e}")

        return composite
        
    except Exception as e:
        print(f"Critical error in create_summary_visualization: {e}")
        raise


def numpy_to_base64_data_url(img: np.ndarray, format: str = "PNG") -> str:
    """
    Convert numpy array to base64 data URL.
    
    Args:
        img: Numpy array image (BGR format)
        format: Image format (PNG, JPEG)
        
    Returns:
        Base64 data URL string (e.g., "data:image/png;base64,...")
    """
    try:
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(img_rgb)
        
        # Save to bytes buffer
        buffered = BytesIO()
        pil_image.save(buffered, format=format)
        
        # Encode to base64
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Create data URL
        mime_type = f"image/{format.lower()}"
        data_url = f"data:{mime_type};base64,{img_base64}"
        
        return data_url
        
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        raise


def generate_summary_with_gpt5(img1: np.ndarray,
                                img2: np.ndarray,
                                api_key: Optional[str] = None) -> str:
    """
    Generate summary using OpenAI GPT-5 with exact same logic as provided code.
    
    Args:
        img1: Original image (numpy array) - Image A
        img2: Revised image (numpy array) - Image B
        api_key: Optional OpenAI API key (loads from .env if not provided)
        
    Returns:
        Generated HTML summary text or error message
    """
    try:
        # Load API key
        if api_key is None:
            try:
                load_dotenv()
                api_key = os.getenv("OPENAI_API_KEY")
            except Exception as e:
                print(f"Warning: Failed to load .env file: {e}")

        if not api_key:
            return "⚠️ OPENAI_API_KEY not found in .env file. Please add:\nOPENAI_API_KEY=your_key_here"

        # Import OpenAI
        try:
            from openai import OpenAI
        except ImportError:
            return "❌ OpenAI library not installed. Run: pip install openai"

        # Initialize client
        try:
            client = OpenAI(api_key=api_key)
        except Exception as e:
            return f"❌ Failed to initialize OpenAI client: {e}"

        if not client:
            return "❌ OpenAI client not initialized - cannot perform comparison"

        # Convert images to base64 data URLs
        try:
            print("   🔄 Converting images to base64 data URLs...")
            data_url_a = numpy_to_base64_data_url(img1, format="PNG")
            data_url_b = numpy_to_base64_data_url(img2, format="PNG")
        except Exception as e:
            return f"❌ Failed to convert images: {e}"

        # Exact same prompt from your code
        prompt_text = """You are a CAD image-diff assistant. 
Compare Image A and Image B and summarize all geometric and annotation differences (including measurement and reading changes).
Return the summary as an HTML fragment only (no markdown, no explanation, no extra text). 
If there are no visible differences, return exactly: No differences found
HTML requirements:
- Return a single top-level container (e.g. <div> ... </div>).
- Structure the output with bold category headings followed by indented bullet lists.
- Provide a bullet list of changes using <ul><li>...</li></ul> when differences exist. Make the difference type bold.
- For each change include short location, and values.
- Keep the HTML minimal and valid."""

        try:
            print("   🤖 Starting GPT-5 image comparison...")
            print("   📤 Sending request to GPT-5...")
            
            # Use responses.create with exact same structure
            text_response = client.responses.create(
                model="gpt-5-mini",
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt_text},
                            {"type": "input_image", "image_url": data_url_a},
                            {"type": "input_image", "image_url": data_url_b},
                        ],
                    }
                ],
            )
            
        except AttributeError:
            # Fallback to chat.completions if responses.create doesn't exist
            print("   🔄 Fallback to chat.completions API...")
            try:
                text_response = client.chat.completions.create(
                    model="gpt-4o",  # Use latest available model
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt_text
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": data_url_a,
                                        "detail": "high"
                                    }
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": data_url_b,
                                        "detail": "high"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=2000,
                    temperature=0.3
                )
                
                # Extract summary from chat response
                if text_response and text_response.choices:
                    context = text_response.choices[0].message.content
                    
                    if not context:
                        print("⚠️  No text summary found in GPT response")
                        return "⚠️ GPT returned empty response"
                    
                    print("   ✅ Summary generated successfully")
                    return context
                else:
                    return "⚠️ GPT returned empty response"
                    
            except Exception as e:
                error_msg = str(e)
                if "invalid_api_key" in error_msg.lower():
                    return "❌ Invalid API key. Please check your OPENAI_API_KEY in .env"
                elif "insufficient_quota" in error_msg.lower():
                    return "❌ API quota exceeded. Please check your OpenAI account billing."
                elif "rate_limit" in error_msg.lower():
                    return "⚠️ Rate limit reached. Please wait a moment and try again."
                else:
                    return f"❌ GPT-5 comparison failed: {error_msg}"
        
        except Exception as e:
            error_msg = str(e)
            if "invalid_api_key" in error_msg.lower():
                return "❌ Invalid API key. Please check your OPENAI_API_KEY in .env"
            elif "insufficient_quota" in error_msg.lower():
                return "❌ API quota exceeded. Please check your OpenAI account billing."
            elif "rate_limit" in error_msg.lower():
                return "⚠️ Rate limit reached. Please wait a moment and try again."
            else:
                return f"❌ GPT-5 comparison failed: {error_msg}"

        # Extract text output - exact same logic
        try:
            summary = []
            for item in text_response.output:
                if item.type == "message":
                    for content in item.content:
                        if content.type == "output_text":
                            summary.append(content.text)

            context = "\n".join(summary)

            if not context:
                print("⚠️  No text summary found in GPT-5 response")
                return "⚠️ GPT-5 returned empty response"
                
            print("   ✅ Summary generated successfully")
            return context
            
        except Exception as e:
            print(f"Warning: Failed to extract response: {e}")
            return f"❌ Error extracting response: {e}"
            
    except Exception as e:
        print(f"Critical error in generate_summary_with_gpt5: {e}")
        return f"❌ Unexpected error: {e}"


def generate_summary_with_gemini(composite_image: np.ndarray, 
                                 api_key: Optional[str] = None) -> str:
    """
    Generate summary using Gemini Vision API with better error handling.
    (Keep this as fallback)
    """
    try:
        if api_key is None:
            try:
                load_dotenv()
                api_key = os.getenv("GEMINI_API_KEY")
            except Exception as e:
                print(f"Warning: Failed to load .env file: {e}")

        if not api_key:
            return "⚠️ GEMINI_API_KEY not found in .env file. Please add:\nGEMINI_API_KEY=your_key_here"

        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
        except Exception as e:
            return f"❌ Failed to configure Gemini API: {e}"

        try:
            image_rgb = cv2.cvtColor(composite_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
        except Exception as e:
            return f"❌ Failed to convert image for API: {e}"

        prompt = ("""You are analyzing a CAD drawing revision comparison.
**Left**: Original | **Middle**: Revised | **Right**: Highlighted changes
Provide ONLY two sections with subsections:
## Structural/Geometric Changes
For each distinct change, create a subsection:
                  
### [Component/Area Name]
Detailed description of what changed and why it matters.If no changes, write 'None detected.'
## Textual/Annotation Changes
For each category, create a subsection:
                  
### [Category Name]
Detailed description of the text changes.If no changes, write 'None detected.'
                  
RULES:
- Use ## for subsection headings
- Each subsection = descriptive sentences with specific details
- Include component IDs, measurements, and locations when visible
- Do NOT invent details not visible in the images
""")

        try:
            print("   🤖 Sending request to Gemini...")
            response = model.generate_content([prompt, pil_image])
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

        try:
            if response and getattr(response, 'text', None):
                print("   ✅ Summary generated successfully")
                return response.text
            else:
                return "⚠️ Gemini returned empty response. Please check API quota and image quality."
        except Exception as e:
            return f"❌ Error extracting response: {e}"
            
    except Exception as e:
        print(f"Critical error in generate_summary_with_gemini: {e}")
        return f"❌ Unexpected error: {e}"


def analyze_change_types(highlighted: np.ndarray) -> Dict[str, float]:
    """
    Analyze the types of changes based on color (improved color detection).
    
    Args:
        highlighted: Highlighted changes image
        
    Returns:
        Dictionary containing change statistics
    """
    try:
        # Convert to HSV
        try:
            hsv = cv2.cvtColor(highlighted, cv2.COLOR_BGR2HSV)
        except Exception as e:
            raise RuntimeError(f"Failed to convert image to HSV: {e}")

        # Create color masks
        try:
            red_mask1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
            red_mask2 = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)

            green_mask = cv2.inRange(hsv, (40, 100, 100), (80, 255, 255))
            yellow_mask = cv2.inRange(hsv, (20, 100, 100), (40, 255, 255))
        except Exception as e:
            raise RuntimeError(f"Failed to create color masks: {e}")

        # Calculate statistics
        try:
            total_pixels = highlighted.shape[0] * highlighted.shape[1]
            if total_pixels == 0:
                raise ValueError("Image has zero pixels")
                
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
        except Exception as e:
            raise RuntimeError(f"Failed to calculate statistics: {e}")
            
    except Exception as e:
        print(f"Critical error in analyze_change_types: {e}")
        raise


def generate_change_summary(img1: np.ndarray, 
                           aligned_img2: np.ndarray, 
                           highlighted: np.ndarray, 
                           api_key: Optional[str] = None,
                           use_gpt5: bool = True) -> Dict[str, Any]:
    """
    Generate complete summary with error handling.
    
    Args:
        img1: Original image
        aligned_img2: Aligned revised image
        highlighted: Highlighted changes image
        api_key: Optional API key (OpenAI or Gemini)
        use_gpt5: If True, use GPT-5; if False, use Gemini
        
    Returns:
        Dictionary containing AI summary, statistics, and composite image
    """
    try:
        print("   📊 Creating summary visualization...")
        try:
            composite = create_summary_visualization(img1, aligned_img2, highlighted)
        except Exception as e:
            raise RuntimeError(f"Failed to create visualization: {e}")

        print("   🔍 Analyzing change types...")
        try:
            stats = analyze_change_types(highlighted)
        except Exception as e:
            raise RuntimeError(f"Failed to analyze changes: {e}")

        print("   🤖 Generating AI summary...")
        try:
            if use_gpt5:
                # Use GPT-5 with two separate images
                ai_summary = generate_summary_with_gpt5(img1, aligned_img2, api_key)
            else:
                # Fallback to Gemini with composite
                ai_summary = generate_summary_with_gemini(composite, api_key)
        except Exception as e:
            print(f"Warning: AI summary generation failed: {e}")
            ai_summary = f"❌ AI summary unavailable: {e}"

        return {
            "ai_summary": ai_summary,
            "statistics": stats,
            "composite_image": composite
        }
        
    except Exception as e:
        print(f"Critical error in generate_change_summary: {e}")
        raise


def create_summary_panel(summary_data: Dict[str, Any], 
                        width: int = 1800, 
                        height: int = 1200) -> np.ndarray:
    """
    Create text panel with automatic sizing.
    
    Args:
        summary_data: Dictionary containing AI summary and statistics
        width: Panel width in pixels
        height: Panel height in pixels
        
    Returns:
        Summary panel image
    """
    try:
        # Create blank panel
        try:
            panel = np.ones((height, width, 3), dtype=np.uint8) * 255
        except Exception as e:
            raise RuntimeError(f"Failed to create panel: {e}")

        # Setup fonts and layout
        try:
            font = cv2.FONT_HERSHEY_SIMPLEX
            title_font_scale = 1.8
            text_font_scale = 0.9
            line_spacing = 45
            margin_left = 40
            margin_right = 40
            y_pos = 80
        except Exception as e:
            raise RuntimeError(f"Failed to setup layout parameters: {e}")

        # Add title
        try:
            cv2.putText(panel, "CAD REVISION ANALYSIS REPORT", (margin_left, y_pos),
                        font, title_font_scale, (0, 0, 128), 3, cv2.LINE_AA)
            y_pos += 70

            cv2.line(panel, (margin_left, y_pos), (width - margin_right, y_pos), (0, 0, 0), 2)
            y_pos += 60
        except Exception as e:
            print(f"Warning: Failed to add title: {e}")

        # Add statistics
        try:
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
                try:
                    cv2.putText(panel, line, (margin_left + 20, y_pos),
                                font, text_font_scale, (30, 30, 30), 2, cv2.LINE_AA)
                    y_pos += line_spacing
                except Exception as e:
                    print(f"Warning: Failed to add stats line '{line}': {e}")
                    continue

            y_pos += 40
        except Exception as e:
            print(f"Warning: Failed to add statistics section: {e}")

        # Add AI analysis
        try:
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
                    try:
                        if len(line) > max_chars_per_line:
                            wrapped_lines.extend(textwrap.wrap(line, width=max_chars_per_line, break_long_words=False))
                        else:
                            wrapped_lines.append(line)
                    except Exception as e:
                        print(f"Warning: Failed to wrap line: {e}")
                        wrapped_lines.append(line)
                else:
                    wrapped_lines.append("")

            for line in wrapped_lines:
                try:
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
                except Exception as e:
                    print(f"Warning: Failed to add text line: {e}")
                    continue
        except Exception as e:
            print(f"Warning: Failed to add AI analysis section: {e}")

        return panel
        
    except Exception as e:
        print(f"Critical error in create_summary_panel: {e}")
        raise

