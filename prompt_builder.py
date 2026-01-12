"""
Prompt Builder Module
プロンプト生成ロジックを提供
"""

import json
from typing import Optional


# ============================================
# 固定英語ブロック（定数定義）
# ============================================

CHARACTER_LOCK = """
[CHARACTER CONSISTENCY]
- Maintain the exact same character design, proportions, and art style across all variations.
- Preserve facial features, body type, clothing details, and color scheme.
- Keep the overall mood and atmosphere consistent.
- Do not alter the character's identity or recognizable traits.
""".strip()

BANNER_LAYOUT = """
[BANNER LAYOUT RULES]
- Reserve clear space for text placement (top 20% or bottom 20% of the image).
- Ensure sufficient padding and margins around key visual elements.
- Maintain visual hierarchy with focal point clearly defined.
- Leave negative space for potential overlay text or logos.
- Design with banner aspect ratio in mind (wide format preferred).
""".strip()

TEXT_RULES = """
[TEXT RENDERING RULES]
- Render the following text EXACTLY as specified, character by character: {text_verbatim}
- Do not paraphrase, translate, or modify the text in any way.
- Ensure text is clearly legible with appropriate font size and contrast.
- Position text prominently within the designated text area.
- Use clean, readable typography that matches the overall design aesthetic.
""".strip()

EDIT_ONLY = """
[EDIT RESTRICTIONS]
- ONLY modify the specified element: {edit_target}
- Preserve ALL other elements exactly as they appear in the original image.
- Maintain the same composition, layout, and positioning.
- Keep unchanged elements pixel-perfect where possible.
- The edit should blend seamlessly with the existing design.
""".strip()

VARIATION = """
[VARIATION RULES]
- Create variations along a SINGLE axis: {variation_axis}
- Variation details: {variation_details}
- Keep ALL other aspects identical across variations.
- Each variation should be distinctly different along the specified axis.
- Maintain consistency in quality, style, and overall aesthetic.
""".strip()


# ============================================
# HEX to 英語色名変換
# ============================================

# 基本色のマッピング（近似色も含む）
COLOR_NAMES = {
    # Reds
    (255, 0, 0): "pure red",
    (220, 20, 60): "crimson",
    (178, 34, 34): "firebrick red",
    (255, 99, 71): "tomato red",
    (255, 69, 0): "orange-red",
    (233, 69, 96): "vibrant coral pink",
    (255, 87, 51): "vibrant orange-red",
    
    # Pinks
    (255, 192, 203): "soft pink",
    (255, 105, 180): "hot pink",
    (255, 20, 147): "deep pink",
    (219, 112, 147): "dusty rose",
    (255, 182, 193): "light pink",
    
    # Oranges
    (255, 165, 0): "bright orange",
    (255, 140, 0): "dark orange",
    (255, 127, 80): "coral orange",
    (255, 200, 100): "warm golden orange",
    
    # Yellows
    (255, 255, 0): "bright yellow",
    (255, 215, 0): "gold",
    (255, 223, 0): "golden yellow",
    (250, 250, 210): "light goldenrod",
    (255, 250, 205): "lemon chiffon",
    
    # Greens
    (0, 255, 0): "pure green",
    (0, 128, 0): "forest green",
    (34, 139, 34): "forest green",
    (50, 205, 50): "lime green",
    (144, 238, 144): "light green",
    (0, 100, 0): "dark green",
    (51, 255, 87): "vibrant lime green",
    (46, 139, 87): "sea green",
    
    # Blues
    (0, 0, 255): "pure blue",
    (0, 0, 139): "dark blue",
    (0, 191, 255): "deep sky blue",
    (30, 144, 255): "dodger blue",
    (70, 130, 180): "steel blue",
    (135, 206, 235): "sky blue",
    (173, 216, 230): "light blue",
    (51, 87, 255): "vivid royal blue",
    (15, 52, 96): "deep navy blue",
    (22, 33, 62): "dark midnight blue",
    
    # Purples
    (128, 0, 128): "purple",
    (148, 0, 211): "dark violet",
    (138, 43, 226): "blue-violet",
    (75, 0, 130): "indigo",
    (238, 130, 238): "violet",
    (221, 160, 221): "plum",
    (83, 52, 131): "deep royal purple",
    
    # Browns
    (139, 69, 19): "saddle brown",
    (160, 82, 45): "sienna brown",
    (210, 180, 140): "tan",
    (222, 184, 135): "burlywood",
    (245, 222, 179): "wheat",
    
    # Grays
    (128, 128, 128): "medium gray",
    (169, 169, 169): "dark gray",
    (192, 192, 192): "silver gray",
    (211, 211, 211): "light gray",
    (105, 105, 105): "dim gray",
    (243, 243, 243): "off-white gray",
    
    # Blacks & Whites
    (0, 0, 0): "pure black",
    (255, 255, 255): "pure white",
    (245, 245, 245): "white smoke",
    (250, 250, 250): "snow white",
    
    # Teals & Cyans
    (0, 255, 255): "cyan",
    (0, 139, 139): "dark cyan",
    (0, 128, 128): "teal",
    (32, 178, 170): "light sea green",
    (64, 224, 208): "turquoise",
}


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """HEX色コードをRGBタプルに変換"""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 3:
        hex_color = ''.join([c * 2 for c in hex_color])
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def color_distance(c1: tuple[int, int, int], c2: tuple[int, int, int]) -> float:
    """2つのRGB色間のユークリッド距離を計算"""
    return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5


def hex_to_color_name(hex_color: str) -> str:
    """
    HEX色コードを英語の色名に変換
    
    Args:
        hex_color: HEX形式の色コード（例: "#FF5733"）
    
    Returns:
        英語の色名（例: "vibrant orange-red"）
    """
    try:
        rgb = hex_to_rgb(hex_color)
    except (ValueError, IndexError):
        return "undefined color"
    
    # 完全一致を探す
    if rgb in COLOR_NAMES:
        return COLOR_NAMES[rgb]
    
    # 最も近い色を探す
    min_distance = float('inf')
    closest_name = "custom color"
    
    for known_rgb, name in COLOR_NAMES.items():
        dist = color_distance(rgb, known_rgb)
        if dist < min_distance:
            min_distance = dist
            closest_name = name
    
    # 色相に基づく修飾子を追加
    r, g, b = rgb
    
    # 明度判定
    brightness = (r + g + b) / 3
    if brightness > 200:
        modifier = "light "
    elif brightness < 80:
        modifier = "dark "
    else:
        modifier = ""
    
    # 彩度が低い場合
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    if max_val - min_val < 30:
        if brightness > 200:
            return "off-white"
        elif brightness < 50:
            return "near-black"
        else:
            return f"{modifier}gray"
    
    return f"{modifier}{closest_name}"


def convert_palette_to_description(palette: str) -> str:
    """
    カンマ区切りのHEX色リストを英語の色名リストに変換
    
    Args:
        palette: カンマ区切りのHEX色（例: "#FF5733,#33FF57,#3357FF"）
    
    Returns:
        色名のリスト（例: "vibrant orange-red, vibrant lime green, vivid royal blue"）
    """
    if not palette:
        return ""
    
    colors = [c.strip() for c in palette.split(',') if c.strip()]
    color_names = [hex_to_color_name(c) for c in colors]
    return ", ".join(color_names)


def convert_palette_dict_to_description(palette: dict) -> str:
    """
    辞書形式のパレットを英語の色名リストに変換
    
    Args:
        palette: 色の辞書（例: {"primary": "#FF5733", "secondary": "#33FF57"}）
    
    Returns:
        色名のリスト（例: "Primary: vibrant orange-red, Secondary: vibrant lime green"）
    """
    if not palette:
        return ""
    
    descriptions = []
    for role, hex_color in palette.items():
        color_name = hex_to_color_name(hex_color)
        descriptions.append(f"{role.capitalize()}: {color_name}")
    
    return ", ".join(descriptions)


# ============================================
# プロンプト生成関数
# ============================================

def build_generate_prompt(payload: dict) -> str:
    """
    Generate用プロンプトを構築
    
    Args:
        payload: 入力データ
            - template_id: テンプレートID
            - intent_ja: 日本語の意図説明
            - quality: 品質設定
            - palette: カラーパレット（カンマ区切りHEXまたは辞書）
            - variation_axis: バリエーション軸（オプション）
            - variation_details: バリエーション詳細（オプション）
            - text_verbatim: 表示テキスト（オプション）
            - layout_preference: レイアウト設定（オプション）
    
    Returns:
        完全なプロンプト文字列
    """
    prompt_parts = []
    
    # 1. キャラクター一貫性ルール
    prompt_parts.append(CHARACTER_LOCK)
    prompt_parts.append("")
    
    # 2. バナーレイアウトルール
    prompt_parts.append(BANNER_LAYOUT)
    prompt_parts.append("")
    
    # 3. テキスト描画ルール（text_verbatimがある場合）
    text_verbatim = payload.get("text_verbatim")
    if text_verbatim:
        # JSONエスケープで安全に処理
        escaped_text = json.dumps(text_verbatim, ensure_ascii=False)
        text_rules = TEXT_RULES.format(text_verbatim=escaped_text)
        prompt_parts.append(text_rules)
        prompt_parts.append("")
    
    # 4. バリエーションルール（variation_axisがある場合）
    variation_axis = payload.get("variation_axis")
    if variation_axis:
        variation_details = payload.get("variation_details", "as specified")
        variation_block = VARIATION.format(
            variation_axis=variation_axis,
            variation_details=variation_details
        )
        prompt_parts.append(variation_block)
        prompt_parts.append("")
    
    # 5. カラーパレット
    palette = payload.get("palette")
    if palette:
        if isinstance(palette, dict):
            palette_desc = convert_palette_dict_to_description(palette)
        else:
            palette_desc = convert_palette_to_description(palette)
        
        prompt_parts.append(f"[COLOR PALETTE]")
        prompt_parts.append(f"Use the following color scheme: {palette_desc}")
        prompt_parts.append("")
    
    # 6. レイアウト設定
    layout_preference = payload.get("layout_preference")
    if layout_preference:
        layout_map = {
            "text_centered": "Center-aligned text with balanced composition",
            "spacious": "Generous whitespace with minimalist element placement",
            "layout_dense": "Information-rich layout with efficient space usage",
            "minimal": "Ultra-minimal design with essential elements only",
            "asymmetric": "Dynamic asymmetric composition with visual tension"
        }
        layout_desc = layout_map.get(layout_preference, layout_preference)
        prompt_parts.append(f"[LAYOUT PREFERENCE]")
        prompt_parts.append(f"Layout style: {layout_desc}")
        prompt_parts.append("")
    
    # 7. 品質設定
    quality = payload.get("quality", "medium")
    quality_map = {
        "low": "Quick draft quality, focus on composition",
        "medium": "Standard quality with good detail",
        "high": "High-fidelity final quality with maximum detail"
    }
    quality_desc = quality_map.get(quality, quality_map["medium"])
    prompt_parts.append(f"[QUALITY]")
    prompt_parts.append(f"Target quality: {quality_desc}")
    prompt_parts.append("")
    
    # 8. ユーザーの意図（日本語）
    intent_ja = payload.get("intent_ja", "")
    if intent_ja:
        prompt_parts.append("[USER REQUEST]")
        prompt_parts.append(f"Create a banner based on this description: {intent_ja}")
        prompt_parts.append("")
    
    # 9. テンプレートID（参照用）
    template_id = payload.get("template_id", "unknown")
    prompt_parts.append(f"[TEMPLATE: {template_id}]")
    
    return "\n".join(prompt_parts)


def build_edit_prompt(payload: dict) -> str:
    """
    Edit用プロンプトを構築
    
    Args:
        payload: 入力データ
            - template_id: テンプレートID
            - edit_only: 編集対象（color/text/background/add_element）
            - palette: 新しいカラーパレット（オプション）
            - intent_ja: 日本語の編集意図
            - text_verbatim: 新しいテキスト（オプション）
    
    Returns:
        Edit用プロンプト文字列
    """
    prompt_parts = []
    
    # 1. 編集制限ルール
    edit_only = payload.get("edit_only", "color")
    edit_target_map = {
        "color": "COLOR SCHEME ONLY - change colors while preserving all shapes, text, and composition",
        "text": "TEXT CONTENT ONLY - replace text while preserving all visual elements and layout",
        "background": "BACKGROUND ONLY - modify background while preserving all foreground elements",
        "add_element": "ADD NEW ELEMENTS - insert new decorative elements without removing existing ones"
    }
    edit_target = edit_target_map.get(edit_only, edit_only)
    
    edit_block = EDIT_ONLY.format(edit_target=edit_target)
    prompt_parts.append(edit_block)
    prompt_parts.append("")
    
    # 2. 編集タイプ別の追加ルール
    if edit_only == "color":
        palette = payload.get("palette")
        if palette:
            if isinstance(palette, dict):
                palette_desc = convert_palette_dict_to_description(palette)
            else:
                palette_desc = convert_palette_to_description(palette)
            
            prompt_parts.append("[NEW COLOR SCHEME]")
            prompt_parts.append(f"Apply the following colors: {palette_desc}")
            prompt_parts.append("- Maintain color harmony and contrast ratios")
            prompt_parts.append("- Preserve the overall visual balance")
            prompt_parts.append("")
    
    elif edit_only == "text":
        text_verbatim = payload.get("text_verbatim")
        if text_verbatim:
            escaped_text = json.dumps(text_verbatim, ensure_ascii=False)
            prompt_parts.append("[NEW TEXT CONTENT]")
            prompt_parts.append(f"Replace existing text with: {escaped_text}")
            prompt_parts.append("- Match the original font style and size as closely as possible")
            prompt_parts.append("- Maintain the same text positioning and alignment")
            prompt_parts.append("")
    
    elif edit_only == "background":
        prompt_parts.append("[BACKGROUND MODIFICATION]")
        prompt_parts.append("- Seamlessly replace or modify the background")
        prompt_parts.append("- Ensure foreground elements remain crisp and unaffected")
        prompt_parts.append("- Maintain appropriate contrast with foreground")
        prompt_parts.append("")
        
        palette = payload.get("palette")
        if palette:
            if isinstance(palette, dict):
                palette_desc = convert_palette_dict_to_description(palette)
            else:
                palette_desc = convert_palette_to_description(palette)
            prompt_parts.append(f"New background colors: {palette_desc}")
            prompt_parts.append("")
    
    elif edit_only == "add_element":
        prompt_parts.append("[ELEMENT ADDITION]")
        prompt_parts.append("- Add new decorative elements that complement the existing design")
        prompt_parts.append("- Ensure new elements blend naturally with the current style")
        prompt_parts.append("- Do not obscure or interfere with existing important elements")
        prompt_parts.append("")
    
    # 3. ユーザーの編集意図（日本語）
    intent_ja = payload.get("intent_ja", "")
    if intent_ja:
        prompt_parts.append("[EDIT REQUEST]")
        prompt_parts.append(f"Apply this modification: {intent_ja}")
        prompt_parts.append("")
    
    # 4. テンプレートID（参照用）
    template_id = payload.get("template_id", "unknown")
    prompt_parts.append(f"[TEMPLATE: {template_id}]")
    
    return "\n".join(prompt_parts)


def build_prompt(payload: dict, action: str = "generate") -> str:
    """
    アクションに応じたプロンプトを構築
    
    Args:
        payload: 入力データ
        action: "generate" または "edit"
    
    Returns:
        完全なプロンプト文字列
    """
    if action == "edit":
        return build_edit_prompt(payload)
    else:
        return build_generate_prompt(payload)


# ============================================
# 参照画像プロンプト追加
# ============================================

# 参照画像タイプ別のプロンプトテンプレート
IMAGE_REFERENCE_PROMPTS = {
    "character": """
[CHARACTER REFERENCE]
- Use the provided character image as a design reference.
- Match the character's appearance, proportions, and distinctive features.
- Adapt the character to the banner context while maintaining recognizability.
- Preserve the art style and visual quality of the reference.
""".strip(),
    
    "color": """
[COLOR & ATMOSPHERE REFERENCE]
- Match the color palette and atmosphere from the provided reference image.
- Extract and apply the dominant colors, tones, and mood.
- Maintain similar lighting conditions and color temperature.
- Use the reference as a style guide for overall visual harmony.
""".strip(),
    
    "material": """
[MATERIAL REFERENCE]
- Incorporate visual elements from the provided material reference.
- Use textures, patterns, or decorative elements as inspiration.
- Blend the referenced materials naturally into the banner design.
- Maintain consistency with the overall design aesthetic.
""".strip()
}


def add_image_references(prompt: str, ref_images: list[dict]) -> str:
    """
    プロンプトに画像参照情報を追加
    
    Args:
        prompt: 元のプロンプト
        ref_images: 参照画像のリスト
            各要素: {"type": "character"|"color"|"material", "b64": str, "name": str}
    
    Returns:
        参照情報が追加されたプロンプト
    """
    if not ref_images:
        return prompt
    
    # タイプ別にグループ化
    grouped = {
        "character": [],
        "color": [],
        "material": []
    }
    
    for img in ref_images:
        img_type = img.get("type", "material")
        if img_type in grouped:
            grouped[img_type].append(img)
    
    # 参照セクションを構築
    reference_parts = []
    
    for img_type, images in grouped.items():
        if images:
            # タイプ別のプロンプトテンプレートを追加
            template = IMAGE_REFERENCE_PROMPTS.get(img_type, "")
            if template:
                reference_parts.append(template)
            
            # 画像数を記載
            count = len(images)
            if count == 1:
                reference_parts.append(f"- {count} {img_type} reference image provided.")
            else:
                reference_parts.append(f"- {count} {img_type} reference images provided.")
            reference_parts.append("")
    
    if not reference_parts:
        return prompt
    
    # プロンプトの先頭に参照情報を追加
    reference_section = "\n".join(reference_parts)
    
    return f"{reference_section}\n{prompt}"


def get_reference_images_for_api(ref_images: list[dict]) -> list[dict]:
    """
    API呼び出し用の画像参照リストを生成
    
    Args:
        ref_images: 参照画像のリスト
    
    Returns:
        API用の画像リスト（base64データ含む）
    """
    api_images = []
    
    for img in ref_images:
        api_images.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img['b64']}"
            }
        })
    
    return api_images


# ============================================
# テスト用
# ============================================

if __name__ == "__main__":
    # Generate テスト
    generate_payload = {
        "template_id": "t01",
        "intent_ja": "猫キャラのバナー",
        "quality": "low",
        "palette": "#FF5733,#33FF57,#3357FF,#F3F3F3",
        "variation_axis": "構図",
        "variation_details": "正面/横向き/後ろ姿"
    }
    
    print("=" * 60)
    print("GENERATE PROMPT:")
    print("=" * 60)
    print(build_generate_prompt(generate_payload))
    print()
    
    # Edit テスト
    edit_payload = {
        "template_id": "t09",
        "edit_only": "color",
        "palette": "#FF6B6B,#4ECDC4,#45B7D1,#96CEB4",
        "intent_ja": "暖色系に変更"
    }
    
    print("=" * 60)
    print("EDIT PROMPT:")
    print("=" * 60)
    print(build_edit_prompt(edit_payload))
