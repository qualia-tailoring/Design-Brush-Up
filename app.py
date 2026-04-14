"""
Maison Qualia — Design Brushup（Streamlit版）
pip install streamlit openai pillow pillow-heif
実行: streamlit run app.py
"""

import io, base64, json, re, os
import streamlit as st
from PIL import Image
from pillow_heif import register_heif_opener
import openai

register_heif_opener()  # HEIC対応

# ── クライアント初期化 ────────────────────────────────────
try:
    openai_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    openai_key = os.environ.get("OPENAI_API_KEY", "")

openai_client = openai.OpenAI(api_key=openai_key)

# ── Prompts ───────────────────────────────────────────────
ANALYZE_SYSTEM = """You are a fashion design AI assistant for Maison Qualia.
Analyze the uploaded image and respond ONLY in valid JSON (no markdown fences):
{
  "isDesign": true/false,
  "reason": "explanation if not a design image",
  "category": "e.g. women's wear / accessories",
  "elements": ["silhouette", "color", "detail"],
  "textFound": "any text written on the image, or null",
  "suggestedPrompt": "base DALL-E prompt for fashion illustration refinement"
}
If the image is clearly not fashion-related (food, documents, screenshots), set isDesign=false."""

REFINE_SYSTEM = """You are a fashion design prompt engineer for Maison Qualia.
Convert user instructions into an optimized DALL-E 3 prompt.

Output style must always follow this exact visual standard:
- Fashion illustration showing BOTH front bodice view (with female figure) AND back bodice view (flat/technical) side by side
- Stylish editorial sketch style: loose expressive line art with selective watercolor or marker color fills
- Fine crosshatching and hatching for fabric texture and volume
- Bold confident outlines with delicate interior detail lines
- Color swatches shown between the two views
- White background, high contrast, fashion week presentation quality
- NOT flat vector — expressive, hand-crafted illustrator quality

Respond ONLY in valid JSON (no markdown fences):
{
  "dallePrompt": "full optimized prompt string",
  "summary": "日本語で変更内容の簡潔な説明"
}"""

# ── Helpers ───────────────────────────────────────────────
def parse_json(text: str) -> dict:
    cleaned = re.sub(r"```json|```", "", text).strip()
    return json.loads(cleaned)

def handle_openai_error(e: Exception) -> str:
    msg = str(e)
    if "insufficient_quota" in msg or "quota" in msg.lower():
        return "現在サービスの利用上限に達しています。しばらく経ってから再度お試しください。"
    if "rate_limit" in msg:
        return "アクセスが集中しています。少し待ってから再度お試しください。"
    if "invalid_api_key" in msg:
        return "サービスの設定に問題があります。管理者にお問い合わせください。"
    return f"エラーが発生しました。再度お試しください。（{msg[:80]}）"

def img_to_base64(img: Image.Image) -> tuple[str, str]:
    if img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")
    img.thumbnail((1024, 1024))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode(), "image/jpeg"

def img_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def analyze_image(img: Image.Image) -> dict:
    b64, media = img_to_base64(img)
    for attempt in range(3):
        try:
            msg = openai_client.chat.completions.create(
                model="gpt-4o",
                max_tokens=1000,
                messages=[{
                    "role": "system",
                    "content": ANALYZE_SYSTEM
                }, {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{media};base64,{b64}"}},
                        {"type": "text", "text": "Analyze this image."}
                    ]
                }]
            )
            return parse_json(msg.choices[0].message.content)
        except openai.InternalServerError:
            if attempt == 2:
                raise
            import time; time.sleep(2)

def refine_prompt(instruction: str, analysis: dict) -> dict:
    msg = openai_client.chat.completions.create(
        model="gpt-4o",
        max_tokens=1000,
        messages=[{
            "role": "system",
            "content": REFINE_SYSTEM
        }, {
            "role": "user",
            "content": f"Design context: {json.dumps(analysis)}\nUser instruction: {instruction}"
        }]
    )
    return parse_json(msg.choices[0].message.content)

def generate_image(prompt: str) -> Image.Image:
    response = openai_client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        n=1,
        size="1024x1024",
        style="vivid",
        quality="standard",
        response_format="b64_json",
    )
    b64 = response.data[0].b64_json
    return Image.open(io.BytesIO(base64.b64decode(b64)))

# ── UI ────────────────────────────────────────────────────
st.set_page_config(page_title="Maison Qualia — Design Brushup", layout="centered")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400&display=swap');
  .mq-title { font-family: 'Cormorant Garamond', serif; letter-spacing: 0.25em; font-size: 1.4rem; color: #2C2C2C; }
  .mq-sub   { font-size: 0.6rem; letter-spacing: 0.2em; color: #8A8680; }
  .mq-tag   { display:inline-block; font-size:0.6rem; letter-spacing:0.1em; padding:2px 10px;
               border:1px solid #E8E4DF; color:#8A8680; margin-right:6px; text-transform:uppercase; }
  .stButton>button { background:#1A1A1A; color:#fff; border:none; letter-spacing:0.15em;
                     font-size:0.7rem; padding:10px 0; width:100%; }
  .stButton>button:hover { background:#333; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="mq-title">MAISON QUALIA</div>', unsafe_allow_html=True)
st.markdown('<div class="mq-sub">DESIGN BRUSHUP</div>', unsafe_allow_html=True)
st.markdown("---")

# ── セッション初期化 ──────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "rotation" not in st.session_state:
    st.session_state.rotation = 0
if "base_idx" not in st.session_state:
    st.session_state.base_idx = 0
if "active_idx" not in st.session_state:
    st.session_state.active_idx = 0

MAX_BRUSHUP = 3

# ── アップロード ──────────────────────────────────────────
if not st.session_state.history:
    uploaded = st.file_uploader("デザイン画をアップロード", type=["png","jpg","jpeg","webp","heic","heif"])
    if uploaded:
        img = Image.open(uploaded)
        if img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")
        with st.spinner("デザイン画を解析中..."):
            try:
                result = analyze_image(img)
            except Exception as e:
                st.error(handle_openai_error(e))
                st.stop()

        if not result.get("isDesign"):
            st.error(f"デザイン画として認識できませんでした。\n理由：{result.get('reason','')}")
        else:
            st.session_state.analysis = result
            st.session_state.history = [{"img": img, "summary": "アップロード原画"}]
            st.rerun()

# ── メイン画面 ────────────────────────────────────────────
if st.session_state.history:
    analysis      = st.session_state.analysis
    history       = st.session_state.history
    labels        = ["Original"] + [f"Rev.{i}" for i in range(1, len(history))]
    brushup_count = len(history) - 1

    # タグ
    tags = [analysis.get("category","")]+( analysis.get("elements") or [])[:3]
    st.markdown(" ".join(f'<span class="mq-tag">{t}</span>' for t in tags if t), unsafe_allow_html=True)
    st.markdown("")

    # 履歴セレクター
    if len(history) > 1:
        idx = st.select_slider("バージョン", options=range(len(history)),
                               format_func=lambda i: labels[i],
                               value=st.session_state.active_idx)
        st.session_state.active_idx = idx
    else:
        idx = 0
        st.session_state.active_idx = 0
    active = history[idx]

    # 回転ボタン（オリジナルのみ）
    if idx == 0:
        col_r1, col_r2, col_r3 = st.columns([1, 1, 4])
        with col_r1:
            if st.button("↺ 90°"):
                st.session_state.rotation = (st.session_state.rotation + 90) % 360
                st.rerun()
        with col_r2:
            if st.button("↻ リセット"):
                st.session_state.rotation = 0
                st.rerun()
        with col_r3:
            st.caption(f"回転: {st.session_state.rotation}°　※全画面は画像右上のアイコンから")
        display_img = active["img"].rotate(-st.session_state.rotation, expand=True)
    else:
        display_img = active["img"]

    # メイン画像
    st.image(display_img, use_container_width=True)

    # キャプション＋ブラッシュアップ情報
    col1, col2 = st.columns([4, 1])
    with col1:
        st.caption(active["summary"])
        if idx > 0:
            st.markdown(f'<div style="font-size:0.6rem;color:#8A8680;margin-top:4px">ベース：{labels[idx-1]}　→　{labels[idx]}</div>',
                        unsafe_allow_html=True)
    with col2:
        st.download_button("SAVE", data=img_to_bytes(display_img),
                           file_name=f"mq-{'original' if idx==0 else f'rev{idx}'}.png",
                           mime="image/png")

    st.markdown("---")

    # ブラッシュアップ上限
    if brushup_count >= MAX_BRUSHUP:
        st.warning(f"ブラッシュアップは{MAX_BRUSHUP}回までです。NEW ＋ から新しいデザインを始めてください。")
        if st.button("NEW ＋"):
            st.session_state.history  = []
            st.session_state.analysis = None
            st.session_state.rotation = 0
            st.session_state.base_idx = 0
            st.rerun()
    else:
        # ベース画像選択
        st.markdown('<div style="font-size:0.65rem;letter-spacing:0.12em;color:#8A8680;margin-bottom:6px">BASE IMAGE FOR BRUSHUP</div>', unsafe_allow_html=True)
        if len(history) > 1:
            base_idx = st.select_slider("ベース画像", options=range(len(history)),
                                        format_func=lambda i: labels[i], value=len(history)-1)
        else:
            base_idx = 0
        st.session_state.base_idx = base_idx
        st.markdown(f'<div style="font-size:0.6rem;color:#8A8680;margin-bottom:12px">残り {MAX_BRUSHUP - brushup_count} 回</div>',
                    unsafe_allow_html=True)

        st.markdown("---")

        # 指示入力
        st.markdown('<div style="font-size:0.65rem;letter-spacing:0.12em;color:#8A8680;margin-bottom:8px">BRUSHUP INSTRUCTION</div>', unsafe_allow_html=True)
        instruction = st.text_area("修正指示", value="",
                                   placeholder="例：袖をバルーン袖に変更、カラーはモノトーンで",
                                   label_visibility="collapsed", height=100)

        col_btn, col_new = st.columns([3, 1])
        with col_btn:
            if st.button("BRUSHUP", disabled=not instruction.strip()):
                with st.spinner("デザインを生成中..."):
                    try:
                        ctx = {**analysis, "base_version": labels[st.session_state.base_idx]}
                        refined = refine_prompt(instruction, ctx)
                        new_img = generate_image(refined["dallePrompt"])
                    except Exception as e:
                        st.error(handle_openai_error(e))
                        st.stop()
                st.session_state.history.append({
                    "img": new_img,
                    "summary": refined.get("summary", instruction),
                    "base": labels[st.session_state.base_idx],
                    "prompt": refined.get("dallePrompt", "")
                })
                st.session_state.active_idx = len(st.session_state.history) - 1
                st.rerun()
        with col_new:
            if st.button("NEW ＋"):
                st.session_state.history  = []
                st.session_state.analysis = None
                st.session_state.rotation = 0
                st.session_state.base_idx = 0
                st.rerun()
