import os
import sys
import time
import uuid
import json
import requests
import cv2
import numpy as np
import concurrent.futures
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from google import genai

# UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
ENV_PATH = os.path.join(PROJECT_DIR, '.env')
load_dotenv(ENV_PATH)

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
STATIC_FOLDER = os.path.join(BASE_DIR, 'static')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

PAGE_ID = os.getenv('FACEBOOK_PAGE_ID')
PAGE_TOKEN = os.getenv('PAGE_ACCESS_TOKEN')
IG_ID = os.getenv('INSTAGRAM_ACCOUNT_ID')
GEMINI_KEY = os.getenv('GEMINI_API_KEY')

ai_client = genai.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None

LOGO_PATH = os.path.join(STATIC_FOLDER, 'logo.png')
PODIUM_BG_PATH = os.path.join(STATIC_FOLDER, 'podium_bg.jpg')

def remove_background_ai_precision(img_path):
    orig_pil = Image.open(img_path)
    if orig_pil.mode == 'RGBA':
        extrema = orig_pil.getextrema()
        if len(extrema) >= 4 and extrema[3][0] < 200:
            bbox = orig_pil.getbbox()
            if bbox and bbox[2] - bbox[0] > 50 and bbox[3] - bbox[1] > 50:
                return orig_pil.crop(bbox)

    img = cv2.imread(img_path)
    if img is None:
        return orig_pil.convert('RGBA')

    h, w = img.shape[:2]
    # Fast resize for GrabCut to guarantee sub-second execution
    max_dim = 650
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img_small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        img_small = img.copy()

    sh, sw = img_small.shape[:2]
    try:
        mask = np.zeros((sh, sw), np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect = (3, 3, sw - 6, int(sh * 0.92))
        cv2.grabCut(img_small, mask, rect, bgdModel, fgdModel, 2, cv2.GC_INIT_WITH_RECT)
        mask_fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_fg)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask_clean = np.where(labels == largest_label, 255, 0).astype('uint8')
        else:
            mask_clean = mask_fg

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
        mask_blur = cv2.GaussianBlur(mask_clean, (5, 5), 1.0)

        # Scale mask back to original resolution
        if scale != 1.0:
            mask_blur = cv2.resize(mask_blur, (w, h), interpolation=cv2.INTER_LINEAR)
            mask_clean = cv2.resize(mask_clean, (w, h), interpolation=cv2.INTER_NEAREST)

        x, y, cw, ch = cv2.boundingRect(mask_clean)
        if cw > 30 and ch > 30:
            b, g, r = cv2.split(img)
            rgba = cv2.merge([r, g, b, mask_blur])
            crop = rgba[y:y+ch, x:x+cw]
            return Image.fromarray(crop)
    except Exception as e:
        print(f"Fast Cutout notice: {e}")

    return orig_pil.convert('RGBA')

def analyze_with_ai(prod_name, reg_price, off_price, style_prompt=''):
    def _call_gemini():
        if not ai_client:
            return None
        prompt = f"""You are the creative director for BioGlow, a premier authentic skincare brand in Bangladesh.
Product: '{prod_name}'. Regular price: {reg_price} TK, Offer price: {off_price} TK.
User Custom Style Command: '{style_prompt if style_prompt else "Clean luxury studio"}'

Return ONLY a valid JSON object with:
1. "product_title": Clean official product title (max 28 characters).
2. "product_subtitle": Target audience / skin type (max 35 characters).
3. "tagline": Clean size or benefit tag (max 38 characters).
4. "bullets": A list of exactly 5 feature badges:
   - "icon": 2-4 uppercase characters for icon badge (e.g. '48H', 'B3+B5', 'SPF', 'OIL', '5-S', 'DERM')
   - "heading": 3-5 word benefit heading (max 26 characters)
   - "detail": 5-8 word scientific benefit detail (max 48 characters)
5. "caption": Persuasive natural Bengali sales copy adhering to BioGlow rules:
   - Engaging hook on skincare pain points
   - Introduce product as authentic solution
   - 5 key benefits with bullet emojis
   - Regular price {reg_price} TK (~~{reg_price} TK~~) and Offer price {off_price} TK
   - BioGlow 100% original guarantee
   - NEVER mention country of origin or where collected/sourced from
   - Clear call to action (inbox with name, address, phone; nationwide cash on delivery)
   - 6-8 relevant hashtags including #BioGlow
"""
        resp = ai_client.models.generate_content(
            model='gemini-3.5-flash-lite',
            contents=prompt,
            config=dict(response_mime_type='application/json')
        )
        return json.loads(resp.text)

    # Fast ThreadPool with strict 4.0 second timeout to guarantee Gunicorn never times out
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_call_gemini)
            res = future.result(timeout=4.0)
            if res and isinstance(res, dict) and 'product_title' in res:
                return res
    except Exception as e:
        print(f"Gemini call timed out or error ({e}), deploying instant authentic BioGlow template")

    # Instant Fallback Template (Always 0.001s, 100% reliable)
    clean_tag = prod_name.replace(' ', '').replace('-', '')
    return {
        "product_title": prod_name[:28],
        "product_subtitle": "Authentic Skincare Essentials",
        "tagline": "Daily Face & Body Care | 100% Genuine",
        "bullets": [
            {"icon": "100%", "heading": "100% Genuine", "detail": "Direct authentic import guaranteed"},
            {"icon": "CARE", "heading": "Deep Skin Care", "detail": "Nourishes and protects skin barrier"},
            {"icon": "DERM", "heading": "Dermatologist Tested", "detail": "Gentle on sensitive & all skin types"},
            {"icon": "BEST", "heading": "Best Offer Price", "detail": f"Offer price now {off_price} TK"},
            {"icon": "FAST", "heading": "Nationwide Delivery", "detail": "Cash on delivery across Bangladesh"}
        ],
        "caption": f"""✨ আপনার ত্বকের যত্নে ও সুরক্ষায় BioGlow নিয়ে এলো ১০০% অথেনটিক "{prod_name}"! 🌿

আপনার ত্বকের স্বাস্থ্য ও স্বাভাবিক উজ্জ্বলতা ধরে রাখতে এটি হতে পারে আপনার নিত্যদিনের বিশ্বস্ত সঙ্গী। কোনো চিটচিটে ভাব ছাড়াই ত্বকে জোগায় গভীর পুষ্টি ও আর্দ্রতা।

🌸 কেন এটি আপনার স্কিনকেয়ারের জন্য সেরা?
✅ ত্বককে রাখে দীর্ঘস্থায়ী হাইড্রেটেড, মসৃণ ও প্রাণবন্ত
✅ ক্ষতিগ্রস্ত স্কিন ব্যারিয়ার মেরামত ও সুরক্ষিত রাখতে সাহায্য করে
✅ সব ধরনের ত্বকের জন্য নিরাপদ ও ডার্মাটোলজিক্যালি পরীক্ষিত
✅ ১০০% অরিজিনাল এবং প্রিমিয়াম কোয়ালিটি নিশ্চয়তা

📦 রেগুলার প্রাইজ: ~~{reg_price} TK~~
🔥 ধামাকা অফার প্রাইজ: {off_price} TK মাত্র!

💯 BioGlow গ্যারান্টি: আমরা প্রদান করি শতভাগ আসল স্কিনকেয়ার প্রোডাক্ট। নকল প্রমাণে মানিব্যাক গ্যারান্টি!

🛒 অর্ডার করতে এখনই আপনার নাম, পূর্ণ ঠিকানা এবং মোবাইল নম্বর লিখে আমাদের পেজে ইনবক্স করুন।
🚚 সমগ্র বাংলাদেশে ক্যাশ অন ডেলিভারি সুবিধা রয়েছে!

#BioGlow #AuthenticSkincare #{clean_tag} #SkinCareRoutine #BeautyBD #SkincareBangladesh"""
    }

def generate_poster(product_cutout, ai_data, reg_price, off_price, style_prompt=''):
    W, H = 1080, 1080
    prompt_lower = (style_prompt or '').lower()

    if ('podium' in prompt_lower or 'marble' in prompt_lower) and os.path.exists(PODIUM_BG_PATH):
        canvas = Image.open(PODIUM_BG_PATH).convert('RGBA').resize((W, H), Image.Resampling.LANCZOS)
        base_y = 818
        target_h = 550
        prod_x = 715
    else:
        # Default / Custom clean studio with ambient customized glow
        canvas = Image.new('RGBA', (W, H), (252, 253, 254, 255))
        
        # Determine ambient glow tint from prompt
        glow_color = (232, 246, 240, 170) # default soft mint/sage
        if 'rose' in prompt_lower or 'pink' in prompt_lower:
            glow_color = (255, 235, 240, 170)
        elif 'gold' in prompt_lower or 'warm' in prompt_lower or 'sun' in prompt_lower:
            glow_color = (255, 248, 225, 180)
        elif 'blue' in prompt_lower or 'water' in prompt_lower:
            glow_color = (225, 242, 254, 180)

        glow = Image.new('RGBA', (700, 700), (0, 0, 0, 0))
        ImageDraw.Draw(glow).ellipse([60, 60, 640, 640], fill=glow_color)
        canvas.alpha_composite(glow.filter(ImageFilter.GaussianBlur(65)), (510, 310))
        base_y = 860
        target_h = 580
        prod_x = 710

    # Logo & Watermark
    if os.path.exists(LOGO_PATH):
        logo_img = Image.open(LOGO_PATH).convert('RGBA')
        logo_tight = logo_img.crop(logo_img.getbbox())
        logo_w = 175
        logo_h = int(logo_tight.height * (logo_w / logo_tight.width))
        logo_placed = logo_tight.resize((logo_w, logo_h), Image.Resampling.LANCZOS)

        wm_w = int(W * 0.44)
        wm_h = int(logo_tight.height * (wm_w / logo_tight.width))
        wm = logo_tight.resize((wm_w, wm_h), Image.Resampling.LANCZOS)
        r, g, b, a = wm.split()
        wm.putalpha(a.point(lambda p: int(p * 0.065)))
        canvas.alpha_composite(wm, ((W - wm_w) // 2, (H - wm_h) // 2))
        canvas.alpha_composite(logo_placed, (60, 48))
    else:
        logo_h = 40

    scale = target_h / max(product_cutout.height, 1)
    target_w = int(product_cutout.width * scale)
    scaled_prod = product_cutout.resize((target_w, target_h), Image.Resampling.LANCZOS)
    prod_y = base_y - target_h

    amb_shadow = Image.new('RGBA', (target_w + 100, 60), (0, 0, 0, 0))
    ImageDraw.Draw(amb_shadow).ellipse([15, 10, target_w + 85, 50], fill=(25, 45, 50, 75))
    canvas.alpha_composite(amb_shadow.filter(ImageFilter.GaussianBlur(14)), (prod_x - 50, base_y - 25))

    cont_shadow = Image.new('RGBA', (target_w + 20, 24), (0, 0, 0, 0))
    ImageDraw.Draw(cont_shadow).ellipse([8, 4, target_w + 12, 18], fill=(15, 25, 30, 150))
    canvas.alpha_composite(cont_shadow.filter(ImageFilter.GaussianBlur(4)), (prod_x - 10, base_y - 10))

    canvas.alpha_composite(scaled_prod, (prod_x, prod_y))

    # Typography
    try:
        f_title = ImageFont.truetype(r'C:\Windows\Fonts\segoeuib.ttf', 35)
        f_subtitle = ImageFont.truetype(r'C:\Windows\Fonts\segoeuib.ttf', 20)
        f_regular = ImageFont.truetype(r'C:\Windows\Fonts\segoeui.ttf', 16)
        f_bold = ImageFont.truetype(r'C:\Windows\Fonts\segoeuib.ttf', 17)
        f_badge_header = ImageFont.truetype(r'C:\Windows\Fonts\segoeuib.ttf', 15)
        f_price_old = ImageFont.truetype(r'C:\Windows\Fonts\segoeui.ttf', 22)
        f_price_new = ImageFont.truetype(r'C:\Windows\Fonts\segoeuib.ttf', 46)
        f_price_tk = ImageFont.truetype(r'C:\Windows\Fonts\segoeuib.ttf', 30)
        f_icon = ImageFont.truetype(r'C:\Windows\Fonts\segoeuib.ttf', 13)
        f_icon_sm = ImageFont.truetype(r'C:\Windows\Fonts\segoeuib.ttf', 10)
    except:
        f_title = f_subtitle = f_regular = f_bold = f_badge_header = f_price_old = f_price_new = f_price_tk = f_icon = f_icon_sm = ImageFont.load_default()

    cetaphil_blue = (0, 79, 159, 255)
    cetaphil_green = (46, 139, 87, 255)
    brand_navy = (16, 45, 84, 255)
    sub_gray = (75, 85, 95, 255)

    draw = ImageDraw.Draw(canvas)
    title_x = 60
    title_y = 48 + logo_h + 18
    
    title_display = ai_data.get('product_title', 'Skincare Product')
    subtitle_display = ai_data.get('product_subtitle', 'Authentic Skincare Essentials')
    tagline_display = ai_data.get('tagline', 'Daily Face & Body Care | 100% Genuine')

    draw.text((title_x, title_y), title_display, fill=brand_navy, font=f_title)
    draw.text((title_x, title_y + 44), subtitle_display, fill=cetaphil_blue, font=f_subtitle)
    draw.text((title_x, title_y + 74), tagline_display, fill=sub_gray, font=f_regular)

    line_y = title_y + 104
    draw.line([(title_x, line_y), (title_x + 400, line_y)], fill=(200, 220, 215, 255), width=2)

    bullets = ai_data.get('bullets', [])
    start_by = line_y + 18
    badge_radius = 23
    badge_x = title_x + badge_radius

    for i, b in enumerate(bullets[:5]):
        badge_text = b.get('icon', 'BIO')
        heading = b.get('heading', 'Skincare Benefit')
        detail = b.get('detail', 'Nourishes & protects skin')

        cy = start_by + (i * 67) + badge_radius
        draw.ellipse([badge_x - badge_radius, cy - badge_radius, badge_x + badge_radius, cy + badge_radius],
                     outline=cetaphil_green, width=2, fill=(240, 250, 245, 255))
        
        lines = badge_text.split('\n')
        if len(lines) == 1:
            bbox_b = draw.textbbox((0, 0), lines[0], font=f_icon)
            draw.text((badge_x - (bbox_b[2]-bbox_b[0])//2, cy - (bbox_b[3]-bbox_b[1])//2 - 2), lines[0], fill=cetaphil_green, font=f_icon)
        else:
            b1 = draw.textbbox((0, 0), lines[0], font=f_icon_sm)
            b2 = draw.textbbox((0, 0), lines[1], font=f_icon_sm)
            draw.text((badge_x - (b1[2]-b1[0])//2, cy - 13), lines[0], fill=cetaphil_green, font=f_icon_sm)
            draw.text((badge_x - (b2[2]-b2[0])//2, cy + 1), lines[1], fill=brand_navy, font=f_icon_sm)
        
        text_start_x = badge_x + badge_radius + 18
        draw.text((text_start_x, cy - 18), heading, fill=brand_navy, font=f_bold)
        draw.text((text_start_x, cy + 5), detail, fill=sub_gray, font=f_regular)

    # Price Badge
    price_box_x = title_x
    price_box_y = H - 150
    box_w = 330
    box_h = 95
    p_shadow = Image.new('RGBA', (box_w + 20, box_h + 20), (0, 0, 0, 0))
    ImageDraw.Draw(p_shadow).rounded_rectangle([10, 10, box_w + 10, box_h + 10], radius=12, fill=(0, 25, 40, 80))
    canvas.alpha_composite(p_shadow.filter(ImageFilter.GaussianBlur(8)), (price_box_x - 10, price_box_y - 5))

    draw.rounded_rectangle([price_box_x, price_box_y, price_box_x + box_w, price_box_y + box_h], radius=12, fill=(11, 79, 150, 255))
    draw.text((price_box_x + 20, price_box_y + 10), 'BEST PRICE', fill=(140, 230, 255, 255), font=f_badge_header)

    old_price_text = f'{reg_price} TK'
    old_price_x = price_box_x + 205
    old_price_y = price_box_y + 9
    draw.text((old_price_x, old_price_y), old_price_text, fill=(200, 220, 240, 220), font=f_price_old)
    bbox_old = draw.textbbox((old_price_x, old_price_y), old_price_text, font=f_price_old)
    strike_y = (bbox_old[1] + bbox_old[3]) // 2
    draw.line([(bbox_old[0] - 2, strike_y), (bbox_old[2] + 2, strike_y)], fill=(255, 90, 90, 255), width=2)

    draw.text((price_box_x + 20, price_box_y + 32), str(off_price), fill=(255, 255, 255, 255), font=f_price_new)
    draw.text((price_box_x + 155, price_box_y + 44), 'TK', fill=(255, 255, 255, 255), font=f_price_tk)

    return canvas.convert('RGB')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/generate', methods=['POST'])
def api_generate():
    try:
        t_start = time.time()
        file = request.files.get('image')
        prod_name = request.form.get('product_name', 'Skincare Product')
        reg_price = request.form.get('regular_price', '0')
        off_price = request.form.get('offer_price', '0')
        style_prompt = request.form.get('style_prompt', '')
        
        if not file:
            return jsonify({'success': False, 'error': 'No image provided'}), 400

        ext = os.path.splitext(file.filename)[1] or '.png'
        uid = uuid.uuid4().hex[:8]
        raw_path = os.path.join(UPLOAD_FOLDER, f'raw_{uid}{ext}')
        file.save(raw_path)

        cutout = remove_background_ai_precision(raw_path)
        ai_data = analyze_with_ai(prod_name, reg_price, off_price, style_prompt=style_prompt)
        poster_rgb = generate_poster(cutout, ai_data, reg_price, off_price, style_prompt=style_prompt)
        
        poster_filename = f'poster_{uid}.jpg'
        poster_path = os.path.join(STATIC_FOLDER, poster_filename)
        poster_rgb.save(poster_path, 'JPEG', quality=95)

        total_elapsed = round(time.time() - t_start, 2)
        print(f"API /generate SUCCESS for '{prod_name}' in {total_elapsed}s")

        return jsonify({
            'success': True,
            'poster_url': f'/static/{poster_filename}',
            'poster_path': poster_path,
            'caption': ai_data.get('caption', '')
        })
    except Exception as e:
        print(f"API /generate Exception: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/publish', methods=['POST'])
def api_publish():
    try:
        data = request.get_json() or {}
        poster_path = data.get('poster_path')
        caption = data.get('caption')
        is_scheduled = data.get('is_scheduled', False)
        schedule_time = data.get('schedule_time')

        if not poster_path or not os.path.exists(poster_path):
            return jsonify({'success': False, 'error': 'Poster file not found'}), 400

        # 1. Facebook Page (Instant OR Scheduled)
        fb_url = f"https://graph.facebook.com/v21.0/{PAGE_ID}/photos"
        fb_payload = {
            "caption": caption,
            "access_token": PAGE_TOKEN
        }

        if is_scheduled and schedule_time:
            fb_payload["published"] = "false"
            fb_payload["scheduled_publish_time"] = int(schedule_time)

        with open(poster_path, "rb") as f:
            fb_resp = requests.post(
                fb_url,
                data=fb_payload,
                files={"source": f}
            ).json()

        if "id" not in fb_resp:
            return jsonify({'success': False, 'error': f"Facebook error: {fb_resp}"}), 500

        fb_id = fb_resp["id"]
        post_id = fb_resp.get("post_id", fb_id)

        # If scheduled on Facebook
        if is_scheduled:
            return jsonify({
                'success': True,
                'is_scheduled': True,
                'scheduled_time': schedule_time,
                'fb_post_id': post_id
            })

        # 2. Instagram (Instant)
        p_info = requests.get(
            f"https://graph.facebook.com/v21.0/{fb_id}",
            params={"fields": "images", "access_token": PAGE_TOKEN}
        ).json()

        ig_permalink = "https://www.instagram.com/bioglowbd/"
        if "images" in p_info and len(p_info["images"]) > 0:
            cdn_url = p_info["images"][0]["source"]
            cont_resp = requests.post(
                f"https://graph.facebook.com/v21.0/{IG_ID}/media",
                data={"image_url": cdn_url, "caption": caption, "access_token": PAGE_TOKEN}
            ).json()
            
            creation_id = cont_resp.get("id")
            if creation_id:
                time.sleep(3)
                pub_resp = requests.post(
                    f"https://graph.facebook.com/v21.0/{IG_ID}/media_publish",
                    data={"creation_id": creation_id, "access_token": PAGE_TOKEN}
                ).json()
                
                media_id = pub_resp.get("id")
                if media_id:
                    link_resp = requests.get(
                        f"https://graph.facebook.com/v21.0/{media_id}",
                        params={"fields": "permalink", "access_token": PAGE_TOKEN}
                    ).json()
                    ig_permalink = link_resp.get("permalink", ig_permalink)

        return jsonify({
            'success': True,
            'is_scheduled': False,
            'fb_post_id': post_id,
            'fb_url': f'https://www.facebook.com/{PAGE_ID}',
            'ig_url': ig_permalink
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"BioGlow Cloud App Server running on port {port} with Instant Speed & Fast Timeout")
    app.run(host='0.0.0.0', port=port, debug=False)
