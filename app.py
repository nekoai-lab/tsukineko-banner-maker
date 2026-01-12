# -*- coding: utf-8 -*-
"""
tsukineko-banner-maker
Streamlit UI for AI-powered banner generation
"""

import os
import base64
import io
import json
from datetime import datetime
from typing import Optional
from pathlib import Path

import streamlit as st
import yaml
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, BadRequestError, APIError
from PIL import Image

from prompt_builder import build_generate_prompt, build_edit_prompt, add_image_references

# 参照画像ライブラリのパス
SAVED_REFERENCES_DIR = Path("assets/saved_references")
SAVED_REFERENCES_JSON = Path("saved_references.json")

# 環境変数の読み込み
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenAI クライアント初期化
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# コスト管理定数
FINAL_QUALITY_DAILY_LIMIT = 5

# ページ設定
st.set_page_config(
    page_title="Tsukineko Banner Maker",
    page_icon=":cat:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    .main-header {
        background: linear-gradient(90deg, #e94560, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 1rem;
    }
    .template-description {
        background: rgba(255, 255, 255, 0.05);
        border-left: 4px solid #e94560;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #e94560, #ff6b6b);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border-radius: 8px;
        width: 100%;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(233, 69, 96, 0.4);
    }
    .color-palette-container {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
    }
    /* ラベルテキストを白に */
    .stSelectbox label,
    .stTextArea label,
    .stTextInput label,
    .stRadio label,
    .stCheckbox label,
    .stFileUploader label,
    .stColorPicker label,
    .stExpander summary,
    .stMarkdown p,
    .stMarkdown strong,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown p {
        color: white !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: white !important;
    }
    /* サイドバー全体のテキストを黒文字に */
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stExpander summary,
    section[data-testid="stSidebar"] .stExpander p,
    section[data-testid="stSidebar"] div[data-testid="stText"] {
        color: black !important;
    }
    /* ラジオボタン・チェックボックスのオプションテキスト（メインエリア） */
    .main .stRadio div[role="radiogroup"] label,
    .main .stRadio div[role="radiogroup"] label span,
    .main .stRadio div[role="radiogroup"] label p,
    .main .stCheckbox span,
    .main .stSelectbox div[data-baseweb="select"] span,
    /* バリエーション軸のラジオボタン（メインエリア） */
    section.main .stRadio label,
    section.main .stRadio label span,
    section.main .stRadio p,
    [data-testid="stMainBlockContainer"] .stRadio label,
    [data-testid="stMainBlockContainer"] .stRadio label span,
    [data-testid="stMainBlockContainer"] .stRadio div[role="radiogroup"] label,
    [data-testid="stMainBlockContainer"] .stRadio div[role="radiogroup"] label span,
    [data-testid="stMainBlockContainer"] .stRadio div[role="radiogroup"] p,
    .stRadio[data-testid="stRadio"] label span,
    div[data-baseweb="radio"] label {
        color: white !important;
    }
    /* サイドバー内のラジオボタンは黒文字 */
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label,
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label span,
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label p {
        color: black !important;
    }
    /* st.info(), st.warning(), st.error() のテキストを白に */
    .stAlert [data-testid="stMarkdownContainer"] p {
        color: white !important;
    }
    /* サイドバーのアラートは黒文字 */
    section[data-testid="stSidebar"] .stAlert p {
        color: black !important;
    }
    /* ポップオーバー内のテキストを黒に */
    [data-testid="stPopover"] p,
    [data-testid="stPopover"] strong,
    [data-testid="stPopover"] th,
    [data-testid="stPopover"] td,
    [data-testid="stPopover"] .stMarkdown,
    div[data-baseweb="popover"] p,
    div[data-baseweb="popover"] strong,
    div[data-baseweb="popover"] th,
    div[data-baseweb="popover"] td {
        color: black !important;
    }
    /* Manage Library内のExpander内テキストを黒に */
    section[data-testid="stSidebar"] .stExpander [data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] .stExpander [data-testid="stMarkdownContainer"] strong {
        color: black !important;
    }
    .generated-image {
        border-radius: 12px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    .cost-warning {
        background: rgba(255, 193, 7, 0.15);
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .cost-counter {
        display: flex;
        justify-content: space-between;
        padding: 0.25rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# テンプレート読み込み
# ============================================
@st.cache_data
def load_templates():
    """templates.yaml を読み込む"""
    with open("templates.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("templates", []), data.get("field_definitions", {})


# ============================================
# Session State 初期化
# ============================================
def init_session_state():
    """セッション状態の初期化"""
    if "last_generated_images" not in st.session_state:
        st.session_state.last_generated_images = []
    
    # 品質別の生成回数カウンター
    if "generation_count" not in st.session_state:
        st.session_state.generation_count = {
            "draft": 0,
            "standard": 0,
            "final": 0
        }
    
    # 旧形式からの移行（後方互換性）
    if isinstance(st.session_state.generation_count, int):
        old_count = st.session_state.generation_count
        st.session_state.generation_count = {
            "draft": 0,
            "standard": old_count,
            "final": 0
        }
    
    if "form_data" not in st.session_state:
        st.session_state.form_data = {}
    if "last_prompt" not in st.session_state:
        st.session_state.last_prompt = ""
    if "is_generating" not in st.session_state:
        st.session_state.is_generating = False
    if "cost_acknowledged" not in st.session_state:
        st.session_state.cost_acknowledged = False
    
    # 参照画像
    if "ref_images" not in st.session_state:
        st.session_state.ref_images = {
            "character": [],
            "color": [],
            "material": []
        }


def get_total_generation_count() -> int:
    """合計生成回数を取得"""
    counts = st.session_state.generation_count
    return counts["draft"] + counts["standard"] + counts["final"]


def increment_generation_count(quality: str):
    """生成回数をインクリメント"""
    quality_key_map = {
        "low": "draft",
        "medium": "standard",
        "high": "final"
    }
    key = quality_key_map.get(quality, "standard")
    st.session_state.generation_count[key] += 1


# ============================================
# 参照画像ライブラリ管理
# ============================================
def load_saved_references() -> dict:
    """保存された参照画像の情報を読み込む"""
    if not SAVED_REFERENCES_JSON.exists():
        return {"references": [], "max_count": 5}
    
    try:
        with open(SAVED_REFERENCES_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"references": [], "max_count": 5}


def save_references_json(data: dict):
    """参照画像の情報を保存"""
    with open(SAVED_REFERENCES_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def add_saved_reference(name: str, image_data: bytes, ref_type: str = "character") -> bool:
    """
    参照画像を保存（最大3つまで）
    
    Args:
        name: 表示名
        image_data: 画像のバイトデータ
        ref_type: タイプ（character, color, material）
    
    Returns:
        成功したかどうか
    """
    data = load_saved_references()
    
    # 最大数チェック
    if len(data["references"]) >= data["max_count"]:
        return False
    
    # 新しいIDを生成
    existing_ids = [r["id"] for r in data["references"]]
    for i in range(1, 100):
        new_id = f"ref_{i:03d}"
        if new_id not in existing_ids:
            break
    
    # フォルダがなければ作成
    SAVED_REFERENCES_DIR.mkdir(parents=True, exist_ok=True)
    
    # 画像を保存
    image_path = SAVED_REFERENCES_DIR / f"{new_id}.png"
    
    # PIL Imageに変換して保存
    try:
        img = Image.open(io.BytesIO(image_data))
        # リサイズ（大きすぎる場合）
        max_size = 512
        if img.width > max_size or img.height > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        img.save(image_path, "PNG")
    except Exception as e:
        st.error(f"Failed to save image: {e}")
        return False
    
    # JSONに追加
    data["references"].append({
        "id": new_id,
        "name": name,
        "type": ref_type,
        "filename": f"{new_id}.png"
    })
    save_references_json(data)
    
    return True


def delete_saved_reference(ref_id: str) -> bool:
    """保存された参照画像を削除"""
    data = load_saved_references()
    
    # 該当の参照を探す
    ref_to_delete = None
    for ref in data["references"]:
        if ref["id"] == ref_id:
            ref_to_delete = ref
            break
    
    if not ref_to_delete:
        return False
    
    # ファイルを削除
    image_path = SAVED_REFERENCES_DIR / ref_to_delete["filename"]
    if image_path.exists():
        image_path.unlink()
    
    # JSONから削除
    data["references"] = [r for r in data["references"] if r["id"] != ref_id]
    save_references_json(data)
    
    return True


def get_saved_reference_image(ref_id: str) -> Optional[tuple[Image.Image, str]]:
    """
    保存された参照画像を取得
    
    Returns:
        (PIL Image, base64文字列) または None
    """
    data = load_saved_references()
    
    for ref in data["references"]:
        if ref["id"] == ref_id:
            image_path = SAVED_REFERENCES_DIR / ref["filename"]
            if image_path.exists():
                img = Image.open(image_path)
                # base64に変換
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                return img, b64
    
    return None


# ============================================
# 参照画像分析（Vision API使用）
# ============================================
def analyze_reference_images(ref_images: list[dict]) -> str:
    """
    参照画像をVision APIで分析し、詳細な特徴を抽出
    
    Args:
        ref_images: 参照画像リスト [{"type": "character"|"color"|"material", "b64": str, "name": str}]
    
    Returns:
        分析結果のテキスト（プロンプトに追加用）
    """
    if not client or not ref_images:
        return ""
    
    # 画像タイプ別の分析プロンプト
    analysis_prompts = {
        "character": """Analyze this character reference image and describe:
1. Character's distinctive visual features (hair color/style, eye color, facial features)
2. Outfit/clothing details (colors, style, accessories)
3. Art style (anime, realistic, chibi, etc.)
4. Overall color palette used for the character
Be concise but specific. Output in English.""",
        
        "background": """Analyze this background/scene reference image and describe:
1. Scene type (indoor/outdoor, natural/urban, fantasy/realistic)
2. Key visual elements (buildings, nature, objects)
3. Lighting and atmosphere (time of day, mood, weather)
4. Color palette and overall aesthetic
Be concise but specific. Output in English.""",
        
        "style": """Analyze this style reference image and describe:
1. Dominant colors and color harmony
2. Art style and visual aesthetic
3. Textures, patterns, and decorative elements
4. Overall mood and atmosphere to replicate
Be concise but specific. Output in English."""
    }
    
    analysis_results = []
    
    for img in ref_images:
        img_type = img.get("type", "style")
        b64_data = img.get("b64", "")
        
        if not b64_data:
            continue
        
        # 分析用プロンプト
        system_prompt = analysis_prompts.get(img_type, analysis_prompts["style"])
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # コスト効率のためminiを使用
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": system_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{b64_data}",
                                    "detail": "low"  # コスト削減
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            
            analysis = response.choices[0].message.content
            
            # タイプ別にフォーマット
            type_labels = {
                "character": "CHARACTER FEATURES",
                "background": "BACKGROUND & SCENE",
                "style": "STYLE & ATMOSPHERE"
            }
            label = type_labels.get(img_type, "REFERENCE")
            analysis_results.append(f"[{label} FROM REFERENCE IMAGE]\n{analysis}")
            
        except Exception as e:
            # 分析失敗時はスキップ（画像生成は続行）
            error_msg = str(e)
            st.warning(f"Reference image analysis skipped: {error_msg}")
            # デバッグ用：エラー詳細をexpanderで表示
            with st.expander("Error details", expanded=False):
                st.code(error_msg)
            continue
    
    if analysis_results:
        return "\n\n".join(analysis_results)
    return ""


# ============================================
# ファイル名生成
# ============================================
def generate_filename(template_id: str, quality: str, output_format: str, index: int = 0) -> str:
    """
    ファイル名を生成
    形式: {template_id}_{quality}_{timestamp}_{index}.{format}
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{template_id}_{quality}_{timestamp}_{index:02d}.{output_format}"


# ============================================
# 画像生成処理
# ============================================
def generate_images_with_progress(
    prompt: str,
    model: str,
    n: int,
    size: str,
    quality: str,
    background: str,
    output_format: str,
    progress_callback=None
) -> list[dict]:
    """
    OpenAI APIで画像を生成（プログレス付き）
    
    Returns:
        生成された画像のリスト [{"image": PIL.Image, "b64": str}, ...]
    """
    if not client:
        raise ValueError("OpenAI APIキーが設定されていません")
    
    # サイズの変換（autoの場合はデフォルト）
    api_size = size if size != "auto" else "1024x1024"
    
    # 品質の変換
    quality_map = {"low": "low", "medium": "medium", "high": "high"}
    api_quality = quality_map.get(quality, "medium")
    
    images = []
    
    # 1枚ずつ生成してプログレス更新
    for i in range(n):
        if progress_callback:
            progress_callback(i, n, f"Generating image {i + 1}/{n}...")
        
        # API呼び出し（1枚ずつ）
        # gpt-image-1.5 では response_format は使わない
        if model.startswith("gpt-image"):
            response = client.images.generate(
                model=model,
                prompt=prompt,
                n=1,
                size=api_size,
                quality=api_quality
            )
        else:
            response = client.images.generate(
                model=model,
                prompt=prompt,
                n=1,
                size=api_size,
                quality=api_quality,
                response_format="b64_json"
            )
        
        # 結果を処理
        for data in response.data:
            # gpt-image-1.5 はb64_json、他のモデルも同様
            b64_data = getattr(data, 'b64_json', None)
            if b64_data:
                image_bytes = base64.b64decode(b64_data)
            elif hasattr(data, 'url') and data.url:
                # URLから取得
                import urllib.request
                with urllib.request.urlopen(data.url) as resp:
                    image_bytes = resp.read()
                b64_data = base64.b64encode(image_bytes).decode('utf-8')
            else:
                raise ValueError("No image data in response")
            
            image = Image.open(io.BytesIO(image_bytes))
            images.append({
                "image": image,
                "b64": b64_data
            })
        
        if progress_callback:
            progress_callback(i + 1, n, f"Image {i + 1}/{n} complete")
    
    return images


def generate_images(
    prompt: str,
    model: str,
    n: int,
    size: str,
    quality: str,
    background: str,
    output_format: str
) -> list[dict]:
    """
    OpenAI APIで画像を生成（一括）
    
    Returns:
        生成された画像のリスト [{"image": PIL.Image, "b64": str}, ...]
    """
    if not client:
        raise ValueError("OpenAI APIキーが設定されていません")
    
    # サイズの変換（autoの場合はデフォルト）
    api_size = size if size != "auto" else "1024x1024"
    
    # 品質の変換
    quality_map = {"low": "low", "medium": "medium", "high": "high"}
    api_quality = quality_map.get(quality, "medium")
    
    # API呼び出し
    # gpt-image-1.5 では response_format は使わない
    if model.startswith("gpt-image"):
        response = client.images.generate(
            model=model,
            prompt=prompt,
            n=n,
            size=api_size,
            quality=api_quality
        )
    else:
        response = client.images.generate(
            model=model,
            prompt=prompt,
            n=n,
            size=api_size,
            quality=api_quality,
            response_format="b64_json"
        )
    
    # 結果を処理
    images = []
    for data in response.data:
        # gpt-image-1.5 はb64_json、他のモデルも同様
        b64_data = getattr(data, 'b64_json', None)
        if b64_data:
            image_bytes = base64.b64decode(b64_data)
        elif hasattr(data, 'url') and data.url:
            # URLから取得
            import urllib.request
            with urllib.request.urlopen(data.url) as resp:
                image_bytes = resp.read()
            b64_data = base64.b64encode(image_bytes).decode('utf-8')
        else:
            raise ValueError("No image data in response")
        
        image = Image.open(io.BytesIO(image_bytes))
        images.append({
            "image": image,
            "b64": b64_data
        })
    
    return images


def edit_image(
    base_image: bytes,
    prompt: str,
    model: str,
    size: str,
    quality: str
) -> list[dict]:
    """
    OpenAI APIで画像を編集
    
    Returns:
        編集された画像のリスト [{"image": PIL.Image, "b64": str}, ...]
    """
    if not client:
        raise ValueError("OpenAI APIキーが設定されていません")
    
    # サイズの変換
    api_size = size if size != "auto" else "1024x1024"
    
    # 品質の変換
    quality_map = {"low": "low", "medium": "medium", "high": "high"}
    api_quality = quality_map.get(quality, "medium")
    
    # 画像をPNG形式に変換
    img = Image.open(io.BytesIO(base_image))
    
    # RGBA形式に変換（透過対応）
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    
    # バイト列に変換
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    # API呼び出し
    # gpt-image-1.5 では response_format は使わない
    if model.startswith("gpt-image"):
        response = client.images.edit(
            model=model,
            image=img_byte_arr,
            prompt=prompt,
            n=1,
            size=api_size
        )
    else:
        response = client.images.edit(
            model=model,
            image=img_byte_arr,
            prompt=prompt,
            n=1,
            size=api_size,
            response_format="b64_json"
        )
    
    # 結果を処理
    images = []
    for data in response.data:
        # gpt-image-1.5 はb64_json、他のモデルも同様
        b64_data = getattr(data, 'b64_json', None)
        if b64_data:
            image_bytes = base64.b64decode(b64_data)
        elif hasattr(data, 'url') and data.url:
            # URLから取得
            import urllib.request
            with urllib.request.urlopen(data.url) as resp:
                image_bytes = resp.read()
            b64_data = base64.b64encode(image_bytes).decode('utf-8')
        else:
            raise ValueError("No image data in response")
        
        image = Image.open(io.BytesIO(image_bytes))
        images.append({
            "image": image,
            "b64": b64_data
        })
    
    return images


def image_to_bytes(image: Image.Image, output_format: str) -> bytes:
    """PIL ImageをバイトデータとMIMEタイプに変換"""
    img_byte_arr = io.BytesIO()
    
    # フォーマット変換
    if output_format == "jpeg":
        # JPEGはRGBAをサポートしないのでRGBに変換
        if image.mode == "RGBA":
            image = image.convert("RGB")
        image.save(img_byte_arr, format="JPEG", quality=95)
    elif output_format == "webp":
        image.save(img_byte_arr, format="WEBP", quality=95)
    else:  # png
        image.save(img_byte_arr, format="PNG")
    
    img_byte_arr.seek(0)
    return img_byte_arr.getvalue()


# ============================================
# コスト管理
# ============================================
def check_final_quality_limit() -> tuple[bool, int]:
    """
    Final品質の生成制限をチェック
    
    Returns:
        (制限超過しているか, 現在の回数)
    """
    final_count = st.session_state.generation_count["final"]
    is_exceeded = final_count >= FINAL_QUALITY_DAILY_LIMIT
    return is_exceeded, final_count


def render_cost_warning(quality: str, n: int) -> bool:
    """
    コスト警告を表示し、生成可能かを返す
    
    Returns:
        生成可能な場合True
    """
    can_generate = True
    
    # Final品質の警告
    if quality == "high":
        is_exceeded, final_count = check_final_quality_limit()
        
        if is_exceeded:
            st.error(
                f"Final品質の1日の生成上限（{FINAL_QUALITY_DAILY_LIMIT}回）に達しました"
            )
            st.info("Draft または Standard 品質をお使いください。")
            return False
        
        # 警告表示
        st.warning(
            f"[コスト高注意] Final品質は高コストです。"
            f"今日のFinal生成回数: {final_count}/{FINAL_QUALITY_DAILY_LIMIT} 回"
        )
        
        # 確認チェックボックス
        cost_acknowledged = st.checkbox(
            "コストを理解しました",
            value=st.session_state.cost_acknowledged,
            key="cost_checkbox"
        )
        st.session_state.cost_acknowledged = cost_acknowledged
        
        if not cost_acknowledged:
            st.info("生成するには上のチェックボックスにチェックを入れてください")
            can_generate = False
    
    # 3枚以上生成時の確認
    if n >= 3:
        st.info(f"{n}枚の画像を生成します。生成には時間がかかる場合があります。")
    
    return can_generate


# ============================================
# 参照画像アップロード
# ============================================
def process_uploaded_image(uploaded_file) -> dict:
    """
    アップロードされた画像を処理してbase64エンコード
    
    Returns:
        {"name": str, "b64": str, "image": PIL.Image}
    """
    image_bytes = uploaded_file.getvalue()
    b64_data = base64.b64encode(image_bytes).decode('utf-8')
    
    # サムネイル用にPIL Imageも作成
    image = Image.open(io.BytesIO(image_bytes))
    
    return {
        "name": uploaded_file.name,
        "b64": b64_data,
        "image": image
    }


def render_reference_library() -> list[dict]:
    """
    参照画像ライブラリをレンダリング
    保存された参照画像から選択、または新規追加
    
    Returns:
        選択された参照画像のリスト [{"type": str, "b64": str, "name": str}, ...]
    """
    ref_images = []
    saved_data = load_saved_references()
    saved_refs = saved_data.get("references", [])
    max_count = saved_data.get("max_count", 5)
    
    # --- 参照画像の選択 ---
    st.markdown("#### Use Reference")
    
    # 選択肢を構築
    options = ["None (参照しない)"]
    option_ids = [None]
    
    for ref in saved_refs:
        options.append(f"{ref['name']} ({ref['type']})")
        option_ids.append(ref["id"])
    
    selected_index = st.radio(
        "Select reference image",
        options=range(len(options)),
        format_func=lambda x: options[x],
        index=0,
        key="ref_selection",
        label_visibility="collapsed"
    )
    
    # 選択された参照画像を取得
    selected_id = option_ids[selected_index]
    if selected_id:
        result = get_saved_reference_image(selected_id)
        if result:
            img, b64 = result
            # 選択された画像をプレビュー
            st.image(img, width=100, caption=options[selected_index])
            
            # 参照情報を取得
            ref_info = next((r for r in saved_refs if r["id"] == selected_id), None)
            if ref_info:
                ref_images.append({
                    "type": ref_info.get("type", "character"),
                    "b64": b64,
                    "name": ref_info["name"]
                })
    
    st.divider()
    
    # --- 参照画像ライブラリ管理 ---
    with st.expander(f"Manage Library ({len(saved_refs)}/{max_count})", expanded=False):
        
        # 保存済み画像の表示
        if saved_refs:
            st.markdown("**Saved References:**")
            for ref in saved_refs:
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col1:
                    # サムネイル表示
                    result = get_saved_reference_image(ref["id"])
                    if result:
                        img, _ = result
                        st.image(img, width=60)
                
                with col2:
                    st.markdown(f"**{ref['name']}**")
                    st.caption(f"Type: {ref['type']}")
                
                with col3:
                    if st.button("Delete", key=f"del_{ref['id']}", type="secondary"):
                        if delete_saved_reference(ref["id"]):
                            st.success(f"Deleted: {ref['name']}")
                            st.rerun()
            
            st.divider()
        
        # 新規追加
        if len(saved_refs) < max_count:
            st.markdown("**Add New Reference:**")
            
            new_name = st.text_input(
                "Name",
                placeholder="e.g. Tsukineko, Background A",
                key="new_ref_name"
            )
            
            # Type選択とヘルプ
            type_col, help_col = st.columns([4, 1])
            
            with type_col:
                type_options = {
                    "character": "キャラクター",
                    "background": "背景",
                    "style": "スタイル"
                }
                new_type = st.selectbox(
                    "Type",
                    options=list(type_options.keys()),
                    format_func=lambda x: type_options[x],
                    index=0,
                    key="new_ref_type"
                )
            
            with help_col:
                st.markdown("")  # スペーサー
                with st.popover("?"):
                    st.markdown("""
**Type の説明**

| Type | 用途 |
|------|------|
| **キャラクター** | キャラの外見参照（髪色、服装、アートスタイル） |
| **背景** | 背景/シーン参照（風景、建物、雰囲気） |
| **スタイル** | 色味/雰囲気/素材をまとめて参照 |
""")
            
            new_file = st.file_uploader(
                "Image",
                type=["png", "jpg", "jpeg"],
                key="new_ref_file"
            )
            
            if st.button("Save Reference", type="primary", disabled=not (new_name and new_file)):
                if new_name and new_file:
                    if add_saved_reference(new_name, new_file.getvalue(), new_type):
                        st.success(f"Saved: {new_name}")
                        st.rerun()
                    else:
                        st.error("Failed to save (max limit reached)")
        else:
            st.info(f"Library full ({max_count}/{max_count}). Delete to add more.")
    
    # 選択結果のサマリー
    if ref_images:
        st.success(f"Using: {ref_images[0]['name']}")
    
    return ref_images


# ============================================
# サイドバー（共通設定）
# ============================================
def render_sidebar():
    """サイドバーの共通設定をレンダリング"""
    with st.sidebar:
        st.markdown("## Settings")
        st.divider()
        
        # APIキー確認
        if OPENAI_API_KEY:
            st.success("API Key configured")
        else:
            st.error("API Key not set")
            st.info("Set `OPENAI_API_KEY` in `.env` file")
        
        st.divider()
        
        # モデル選択
        model = st.selectbox(
            "Model",
            options=["gpt-image-1.5", "dall-e-3", "dall-e-2"],
            index=0,
            help="Select the model for image generation"
        )
        
        # 品質設定
        quality_options = {
            "Draft (low)": "low",
            "Standard (medium)": "medium",
            "Final (high)": "high"
        }
        quality_label = st.selectbox(
            "Quality",
            options=list(quality_options.keys()),
            index=1,
            help="Select generation quality (higher = more time and cost)"
        )
        quality = quality_options[quality_label]
        
        # Final品質選択時の警告（サイドバー内）
        if quality == "high":
            is_exceeded, final_count = check_final_quality_limit()
            remaining = FINAL_QUALITY_DAILY_LIMIT - final_count
            if remaining <= 2:
                st.warning(f"Final remaining: {remaining}")
        
        # サイズ設定
        size_options = {
            "1024x1024 (Square)": "1024x1024",
            "1536x1024 (Landscape)": "1536x1024",
            "1024x1536 (Portrait)": "1024x1536",
            "auto": "auto"
        }
        size_label = st.selectbox(
            "Size",
            options=list(size_options.keys()),
            index=0,
            help="Select image size"
        )
        size = size_options[size_label]
        
        # 背景設定
        background_options = {
            "auto": "auto",
            "opaque": "opaque",
            "transparent": "transparent"
        }
        background_label = st.selectbox(
            "Background",
            options=list(background_options.keys()),
            index=0,
            help="Background transparency setting"
        )
        background = background_options[background_label]
        
        # 出力形式
        output_format = st.selectbox(
            "Output Format",
            options=["png", "webp", "jpeg"],
            index=0,
            help="File format for saving"
        )
        
        st.divider()
        
        # デバッグモード
        show_prompt = st.checkbox(
            "Show Prompt",
            value=False,
            help="Display the prompt used for generation"
        )
        
        st.divider()
        
        # 生成統計（品質別）
        st.markdown("### Generation Stats")
        
        counts = st.session_state.generation_count
        
        # 品質別カウンター表示
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Draft", counts["draft"], help="Low quality (low cost)")
        with col2:
            st.metric("Standard", counts["standard"], help="Standard quality")
        with col3:
            st.metric("Final", counts["final"], delta=None, help="High quality (high cost)")
        
        # 合計
        total = get_total_generation_count()
        st.caption(f"Total generations: **{total}**")
        
        # Final品質の残り回数
        final_remaining = FINAL_QUALITY_DAILY_LIMIT - counts["final"]
        if final_remaining < FINAL_QUALITY_DAILY_LIMIT:
            if final_remaining > 0:
                st.progress(
                    final_remaining / FINAL_QUALITY_DAILY_LIMIT,
                    text=f"Final remaining: {final_remaining}/{FINAL_QUALITY_DAILY_LIMIT}"
                )
            else:
                st.error(f"Final limit reached ({FINAL_QUALITY_DAILY_LIMIT})")
        
        st.divider()
        
        # 参照画像ライブラリ
        st.markdown("### Reference Library")
        
        ref_images = render_reference_library()
        
        return {
            "model": model,
            "quality": quality,
            "size": size,
            "background": background,
            "output_format": output_format,
            "show_prompt": show_prompt,
            "ref_images": ref_images
        }


# ============================================
# 動的フィールドレンダリング
# ============================================
def render_field(field_name: str, template: dict, field_defs: dict) -> any:
    """フィールドを動的にレンダリング"""
    defaults = template.get("defaults", {})
    required_fields = template.get("required", [])
    is_required = field_name in required_fields
    
    field_def = field_defs.get(field_name, {})
    label = field_def.get("label", field_name)
    if is_required:
        label = f"{label} *"
    
    # intent_ja: テキストエリア（日本語説明）
    if field_name == "intent_ja":
        return st.text_area(
            label,
            placeholder=field_def.get("placeholder", "Describe the banner you want to create"),
            height=120,
            help="Describe the banner content in detail"
        )
    
    # text_verbatim: テキストエリア（完全一致用）
    elif field_name == "text_verbatim":
        return st.text_area(
            label,
            placeholder=field_def.get("placeholder", "Enter the exact text to display"),
            height=80,
            help="Enter the exact text to be rendered on the banner"
        )
    
    # variation_axis: ラジオボタン（日本語化 + ポップアップ）
    elif field_name == "variation_axis":
        axis_options = {
            "なし": None,
            "カラー": "palette",
            "構図": "composition",
            "表情": "expression",
            "季節": "season"
        }
        # 内部値 → 日本語表示名（逆引き用）
        value_to_label = {v: k for k, v in axis_options.items()}
        
        default_axis = defaults.get("variation_axis")
        default_label = value_to_label.get(default_axis, "なし")
        default_index = list(axis_options.keys()).index(default_label) if default_label in axis_options else 0
        
        # ラベルとヘルプボタンを横並び
        label_col, help_col = st.columns([4, 1])
        with label_col:
            st.markdown(f"**{label}**")
        with help_col:
            with st.popover("?"):
                st.markdown("""
**バリエーション軸の説明**

| オプション | 説明 |
|-----------|------|
| **なし** | バリエーションなし（単一生成） |
| **カラー** | 色味を変えたバリエーション |
| **構図** | アングル・配置を変えたバリエーション |
| **表情** | キャラクターの表情を変えたバリエーション |
| **季節** | 季節感を変えたバリエーション |
""")
        
        selected = st.radio(
            label,
            options=list(axis_options.keys()),
            index=default_index,
            horizontal=True,
            label_visibility="collapsed"
        )
        return axis_options[selected]
    
    # variation_details: テキスト入力（variation_axis選択時のみ）
    elif field_name == "variation_details":
        # variation_axisがform_dataにあるか確認
        axis = st.session_state.form_data.get("variation_axis")
        if axis:
            default_details = defaults.get("variation_details", [])
            default_text = "\n".join(default_details) if isinstance(default_details, list) else ""
            return st.text_area(
                label,
                value=default_text,
                height=100,
                help="Enter variation details, one per line"
            )
        return None
    
    # palette: 4つのカラーピッカー
    elif field_name == "palette":
        st.markdown(f"**{label}**")
        cols = st.columns(4)
        colors = {}
        color_names = ["Primary", "Secondary", "Accent", "Extra"]
        default_colors = ["#e94560", "#533483", "#0f3460", "#16213e"]
        
        for i, (col, name) in enumerate(zip(cols, color_names)):
            with col:
                colors[name.lower()] = st.color_picker(
                    name,
                    value=default_colors[i]
                )
        return colors
    
    # base_image: ファイルアップロード（Edit系のみ）
    elif field_name == "base_image":
        uploaded = st.file_uploader(
            label,
            type=["png", "jpg", "jpeg", "webp"],
            help="Upload the base image to edit"
        )
        
        # 前回生成した画像を使用するオプション
        if st.session_state.last_generated_images:
            use_last = st.checkbox(
                "Use previously generated image",
                value=False,
                help="Use the last generated image as base"
            )
            if use_last:
                # 最初の画像を使用
                return st.session_state.last_generated_images[0]
        
        if uploaded:
            return uploaded.getvalue()
        return None
    
    # edit_only: セレクトボックス
    elif field_name == "edit_only":
        edit_options = {
            "color": "color",
            "text": "text",
            "background": "background",
            "add_element": "add_element"
        }
        default_edit = defaults.get("edit_only", "color")
        default_index = 0
        for i, v in enumerate(edit_options.values()):
            if v == default_edit:
                default_index = i
                break
        
        selected = st.selectbox(
            label,
            options=list(edit_options.keys()),
            index=default_index,
            help="Select the element to edit"
        )
        return edit_options[selected]
    
    # layout_preference: セレクトボックス（日本語化）
    elif field_name == "layout_preference":
        # 日本語表示名 → 内部値
        layout_options = {
            "テキスト中央": "text_centered",
            "余白重視": "spacious",
            "情報量重視": "layout_dense",
            "ミニマル": "minimal",
            "非対称": "asymmetric"
        }
        # 内部値 → 日本語表示名（逆引き用）
        value_to_label = {v: k for k, v in layout_options.items()}
        
        default_layout = defaults.get("layout_preference", "text_centered")
        default_label = value_to_label.get(default_layout, "テキスト中央")
        default_index = list(layout_options.keys()).index(default_label) if default_label in layout_options else 0
        
        # ラベルとヘルプボタンを横並び
        label_col, help_col = st.columns([4, 1])
        with label_col:
            st.markdown(f"**{label}**")
        with help_col:
            with st.popover("?"):
                st.markdown("""
**レイアウト設定の説明**

| オプション | 説明 |
|-----------|------|
| **テキスト中央** | テキストを中央に配置、バランスの取れた構図 |
| **余白重視** | 余白を多めに取り、ミニマルな要素配置 |
| **情報量重視** | 情報を詰め込んだ、効率的なスペース使用 |
| **ミニマル** | 必要最小限の要素のみ、超シンプル |
| **非対称** | 動的な非対称構図、視覚的なインパクト |
""")
        
        selected = st.selectbox(
            label,
            options=list(layout_options.keys()),
            index=default_index,
            label_visibility="collapsed"
        )
        return layout_options[selected]
    
    # size_set: サイズプリセット表示
    elif field_name == "size_set":
        st.markdown(f"**{label}**")
        size_set = defaults.get("size_set", [])
        if size_set:
            for size_info in size_set:
                st.caption(f"- {size_info.get('name', 'Custom')}: {size_info.get('width', 0)}x{size_info.get('height', 0)}")
        return size_set
    
    # その他のフィールド
    else:
        return st.text_input(label, help=f"Enter value for {field_name}")


# ============================================
# 画像結果表示
# ============================================
def display_generated_images(
    images: list[dict],
    template_id: str,
    quality: str,
    output_format: str
):
    """生成された画像を表示"""
    st.markdown("### Generated Images")
    
    # グリッド表示
    cols_per_row = min(len(images), 3)
    cols = st.columns(cols_per_row)
    
    for i, img_data in enumerate(images):
        with cols[i % cols_per_row]:
            # 画像表示
            st.image(img_data["image"], use_container_width=True)
            
            # ファイル名生成
            filename = generate_filename(template_id, quality, output_format, i)
            
            # ダウンロードボタン
            img_bytes = image_to_bytes(img_data["image"], output_format)
            
            mime_types = {
                "png": "image/png",
                "webp": "image/webp",
                "jpeg": "image/jpeg"
            }
            
            st.download_button(
                label="Download",
                data=img_bytes,
                file_name=filename,
                mime=mime_types.get(output_format, "image/png"),
                key=f"download_{i}_{datetime.now().timestamp()}"
            )


# ============================================
# メインエリア
# ============================================
def render_main_area(templates: list, field_defs: dict, settings: dict):
    """メインエリアをレンダリング"""
    
    # ヘッダー
    st.markdown('<h1 class="main-header">Tsukineko Banner Maker</h1>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #a0a0a0;'>AI-Powered Banner Generation Tool</p>", unsafe_allow_html=True)
    
    st.divider()
    
    # テンプレート選択
    template_options = {f"{t['label']} ({t['id']})": t for t in templates}
    
    # ラベルとヘルプボタン
    label_col, help_col = st.columns([6, 1])
    with label_col:
        st.markdown("**Select Template**")
    with help_col:
        with st.popover("?"):
            st.markdown("""
**テンプレート一覧**

| テンプレート | 説明 |
|-------------|------|
| **キャラ統一バナー（3案）** | 構図違いの3パターン生成 |
| **文字入りバナー** | テキストを正確に描画 |
| **季節感バリエーション** | 春夏秋冬の4パターン |
| **色違いバナー（3案）** | 配色違いの3パターン |
| **サイズ展開テンプレ** | 複数サイズを一括生成 |
| **シンプルロゴバナー** | 余白多めのミニマル |
| **情報詰め込み型** | 情報量の多いバナー |
| **ミニマルデザイン** | 要素を最小限に |
| **色調整** | 既存画像の色味変更 |
| **テキスト差替え** | 文字部分のみ編集 |
| **背景入替え** | 背景を別の画像に |
| **要素追加** | 画像に要素を追加 |
""")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_label = st.selectbox(
            "Select Template",
            options=list(template_options.keys()),
            label_visibility="collapsed"
        )
    
    selected_template = template_options[selected_label]
    
    with col2:
        action_badge = "Generate" if selected_template["action"] == "generate" else "Edit"
        st.markdown(f"""
        <div style='background: rgba(233, 69, 96, 0.2); padding: 0.5rem 1rem; border-radius: 8px; text-align: center; margin-top: 1.7rem;'>
            <span style='color: #ff6b6b; font-weight: 600;'>{action_badge}</span>
            <span style='color: #a0a0a0; margin-left: 0.5rem;'>x {selected_template['n']} images</span>
        </div>
        """, unsafe_allow_html=True)
    
    # テンプレート説明
    st.markdown(f"""
    <div class="template-description">
        <strong>{selected_template['label']}</strong><br>
        <span style='color: #a0a0a0;'>{selected_template['description']}</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # 動的フィールド表示
    st.markdown("### Input Fields")
    
    form_data = {}
    fields = selected_template.get("fields", [])
    
    # variation_axis を先に処理（variation_details の表示制御のため）
    if "variation_axis" in fields:
        form_data["variation_axis"] = render_field("variation_axis", selected_template, field_defs)
        st.session_state.form_data["variation_axis"] = form_data["variation_axis"]
    
    # 残りのフィールドを処理
    for field_name in fields:
        if field_name == "variation_axis":
            continue  # 既に処理済み
        
        value = render_field(field_name, selected_template, field_defs)
        if value is not None:
            form_data[field_name] = value
    
    st.divider()
    
    # コスト警告とチェック
    n = selected_template["n"]
    can_generate = render_cost_warning(settings["quality"], n)
    
    # アクションボタン
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if selected_template["action"] == "generate":
            button_label = f"Generate Banner ({n} images)"
        else:
            button_label = "Edit Image"
        
        # APIキーチェック + コスト確認
        button_disabled = not OPENAI_API_KEY or not can_generate
        
        if st.button(
            button_label,
            use_container_width=True,
            type="primary",
            disabled=button_disabled
        ):
            # ペイロード作成
            payload = {
                "template_id": selected_template["id"],
                "quality": settings["quality"],
                **form_data
            }
            
            # プロンプト生成
            if selected_template["action"] == "generate":
                prompt = build_generate_prompt(payload)
            else:
                prompt = build_edit_prompt(payload)
            
            # 参照画像がある場合はVision APIで分析してプロンプトに追加
            ref_images = settings.get("ref_images", [])
            ref_analysis = ""
            if ref_images:
                with st.spinner("Analyzing reference images..."):
                    ref_analysis = analyze_reference_images(ref_images)
                
                if ref_analysis:
                    # 分析結果をプロンプトの先頭に追加
                    prompt = f"{ref_analysis}\n\n---\n\n{prompt}"
            
            st.session_state.last_prompt = prompt
            
            # デバッグモード: プロンプト表示
            if settings["show_prompt"]:
                with st.expander("Generated Prompt", expanded=True):
                    st.code(prompt, language="text")
                if ref_images and ref_analysis:
                    st.success(f"Reference images analyzed: {len(ref_images)} image(s)")
            
            # 生成処理
            try:
                st.session_state.is_generating = True
                
                if selected_template["action"] == "generate":
                    # 3枚以上の場合はプログレスバー表示
                    if n >= 3:
                        progress_bar = st.progress(0, text="Preparing...")
                        status_text = st.empty()
                        
                        def update_progress(current, total, message):
                            progress_bar.progress(current / total, text=message)
                            status_text.text(message)
                        
                        images = generate_images_with_progress(
                            prompt=prompt,
                            model=settings["model"],
                            n=n,
                            size=settings["size"],
                            quality=settings["quality"],
                            background=settings["background"],
                            output_format=settings["output_format"],
                            progress_callback=update_progress
                        )
                        
                        progress_bar.progress(1.0, text="Complete!")
                        status_text.empty()
                    else:
                        # 2枚以下は通常処理
                        with st.spinner("Generating images..."):
                            images = generate_images(
                                prompt=prompt,
                                model=settings["model"],
                                n=n,
                                size=settings["size"],
                                quality=settings["quality"],
                                background=settings["background"],
                                output_format=settings["output_format"]
                            )
                else:
                    # Edit処理
                    with st.spinner("Editing image..."):
                        base_image = form_data.get("base_image")
                        if not base_image:
                            st.error("Please upload a base image")
                            st.session_state.is_generating = False
                            st.stop()
                        
                        # base_imageがbytesでない場合（PIL Imageの場合）
                        if isinstance(base_image, Image.Image):
                            img_byte_arr = io.BytesIO()
                            base_image.save(img_byte_arr, format='PNG')
                            base_image = img_byte_arr.getvalue()
                        
                        images = edit_image(
                            base_image=base_image,
                            prompt=prompt,
                            model=settings["model"],
                            size=settings["size"],
                            quality=settings["quality"]
                        )
                
                # 結果を保存
                st.session_state.last_generated_images = [img["image"] for img in images]
                
                # 生成回数をインクリメント
                increment_generation_count(settings["quality"])
                
                # コスト確認フラグをリセット
                st.session_state.cost_acknowledged = False
                
                st.session_state.is_generating = False
                
                # 成功メッセージ
                st.success(f"{len(images)} image(s) generated successfully!")
                
                # 結果表示
                display_generated_images(
                    images=images,
                    template_id=selected_template["id"],
                    quality=settings["quality"],
                    output_format=settings["output_format"]
                )
            
            except RateLimitError as e:
                st.session_state.is_generating = False
                st.error("API rate limit reached")
                with st.expander("Error Details"):
                    st.code(str(e))
            
            except BadRequestError as e:
                st.session_state.is_generating = False
                st.error("Request error")
                with st.expander("Error Details"):
                    st.code(str(e))
            
            except APIError as e:
                st.session_state.is_generating = False
                st.error("API error occurred")
                with st.expander("Error Details"):
                    st.code(str(e))
            
            except Exception as e:
                st.session_state.is_generating = False
                st.error("Unexpected error occurred")
                with st.expander("Error Details"):
                    st.code(f"{type(e).__name__}: {str(e)}")
    
    # 前回生成した画像の表示（ボタンクリック外）
    if st.session_state.last_generated_images and not st.session_state.is_generating:
        st.divider()
        st.markdown("### Previous Results")
        cols = st.columns(min(len(st.session_state.last_generated_images), 3))
        for i, img in enumerate(st.session_state.last_generated_images):
            with cols[i % len(cols)]:
                st.image(img, use_container_width=True)


# ============================================
# メイン実行
# ============================================
def main():
    """メインエントリーポイント"""
    init_session_state()
    
    # テンプレート読み込み
    try:
        templates, field_defs = load_templates()
    except FileNotFoundError:
        st.error("templates.yaml not found")
        st.stop()
    except yaml.YAMLError as e:
        st.error(f"templates.yaml parse error: {e}")
        st.stop()
    
    # サイドバー
    settings = render_sidebar()
    
    # メインエリア
    render_main_area(templates, field_defs, settings)


if __name__ == "__main__":
    main()
