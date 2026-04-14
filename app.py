"""
Maison Qualia — Design Brushup（Streamlit版）
pip install streamlit anthropic openai pillow
実行: streamlit run app.py
"""

import io, base64, json, re, os
import streamlit as st
from PIL import Image
import openai

# ── クライアント初期化 ────────────────────────────────────
openai_client = openai.OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY", ""))
)

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

def img_to_base64(img: Image.Image) -> tuple[str, str]:
    if img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")
    img.thumbnail((1024, 1024))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode(), "image/jpeg"

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

def img_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

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
    st.session_state.history = []   # [{img, summary}]
if "analysis" not in st.session_state:
    st.session_state.analysis = None

# ── アップロード ──────────────────────────────────────────
if not st.session_state.history:
    uploaded = st.file_uploader("デザイン画をアップロード", type=["png","jpg","jpeg","webp","heic","heif"])
    if uploaded:
        img = Image.open(uploaded)
        if img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")
        with st.spinner("デザイン画を解析中..."):
            result = analyze_image(img)

        if not result.get("isDesign"):
            st.error(f"デザイン画として認識できませんでした。\n理由：{result.get('reason','')}")
        else:
            st.session_state.analysis = result
            st.session_state.history = [{"img": img, "summary": "アップロード原画"}]
            st.rerun()

# ── メイン画面 ────────────────────────────────────────────
if st.session_state.history:
    analysis = st.session_state.analysis
    history  = st.session_state.history

    # タグ
    tags = [analysis.get("category","")] + (analysis.get("elements") or [])[:3]
    st.markdown(" ".join(f'<span class="mq-tag">{t}</span>' for t in tags if t), unsafe_allow_html=True)
    st.markdown("")

    labels = ["Original"] + [f"Rev.{i}" for i in range(1, len(history))]
    # 履歴セレクター（2件以上ある時だけ表示）
    if len(history) > 1:
        idx = st.select_slider(
            "バージョン",
            options=range(len(history)),
            format_func=lambda i: labels[i],
            value=len(history) - 1
        )
    else:
        idx = 0
    active = history[idx]

    # メイン画像
    st.image(active["img"], use_container_width=True)
    col1, col2 = st.columns([4,1])
    with col1:
        st.caption(active["summary"])
    with col2:
        st.download_button("SAVE", data=img_to_bytes(active["img"]),
                           file_name=f"mq-{'original' if idx==0 else f'rev{idx}'}.png",
                           mime="image/png")

    st.markdown("---")

    # 指示入力
    st.markdown('<div style="font-size:0.65rem;letter-spacing:0.12em;color:#8A8680;margin-bottom:8px">BRUSHUP INSTRUCTION</div>', unsafe_allow_html=True)

    default_text = analysis.get("textFound") or ""
    instruction = st.text_area("修正指示", value=default_text,
                               placeholder="例：袖をバルーン袖に変更、カラーはモノトーンで",
                               label_visibility="collapsed", height=100)

    col_btn, col_new = st.columns([3,1])
    with col_btn:
        if st.button("BRUSHUP", disabled=not instruction.strip()):
            with st.spinner("デザインを生成中..."):
                refined = refine_prompt(instruction, analysis)
                new_img = generate_image(refined["dallePrompt"])
            st.session_state.history.append({
                "img": new_img,
                "summary": refined.get("summary", instruction)
            })
            st.rerun()
    with col_new:
        if st.button("NEW ＋"):
            st.session_state.history = []
            st.session_state.analysis = None
            st.rerun()
