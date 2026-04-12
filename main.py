"""
Maison Qualia — Design Brushup Backend
pip install fastapi uvicorn openai anthropic python-multipart pillow
実行: uvicorn main:app --reload --port 8000
"""

import base64, io, os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import anthropic
import openai

# ── 環境変数 ──────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY    = os.environ.get("OPENAI_API_KEY", "")

anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
openai_client    = openai.OpenAI(api_key=OPENAI_API_KEY)

# ── App ───────────────────────────────────────────────────
app = FastAPI(title="MQ Design Brushup API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 本番時は claude.ai ドメイン等に絞る
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Prompts ───────────────────────────────────────────────
ANALYZE_SYSTEM = """You are a fashion design AI assistant for Maison Qualia.
Analyze the uploaded image and respond ONLY in valid JSON (no markdown fences):
{
  "isDesign": true/false,
  "reason": "explanation if not a design image",
  "category": "e.g. women's wear / accessories",
  "elements": ["silhouette", "color", "detail", ...],
  "textFound": "any text written on the image, or null",
  "suggestedPrompt": "base DALL-E prompt for fashion illustration refinement"
}
If the image is clearly not fashion-related (food, documents, screenshots), set isDesign=false."""

REFINE_SYSTEM = """You are a fashion design prompt engineer for Maison Qualia.
Convert user instructions into an optimized DALL-E 3 prompt.
Output style must always be: fashion illustration, flat vector, clean line art, minimal color palette, editorial fashion sketch, white background.
Respond ONLY in valid JSON (no markdown fences):
{
  "dallePrompt": "full optimized prompt string",
  "summary": "日本語で変更内容の簡潔な説明"
}"""

# ── Helpers ───────────────────────────────────────────────
def img_to_base64(file_bytes: bytes, content_type: str) -> tuple[str, str]:
    """画像をリサイズしてbase64に変換"""
    img = Image.open(io.BytesIO(file_bytes))
    img.thumbnail((1024, 1024))
    buf = io.BytesIO()
    fmt = "JPEG" if content_type == "image/jpeg" else "PNG"
    media = "image/jpeg" if fmt == "JPEG" else "image/png"
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode(), media

def parse_json_response(text: str) -> dict:
    import json, re
    cleaned = re.sub(r"```json|```", "", text).strip()
    return json.loads(cleaned)

# ── Routes ────────────────────────────────────────────────

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """画像を解析してデザイン画か判定 + 要素抽出"""
    raw = await file.read()
    b64, media = img_to_base64(raw, file.content_type)

    msg = anthropic_client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1000,
        system=ANALYZE_SYSTEM,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media, "data": b64}},
                {"type": "text", "text": "Analyze this image."}
            ]
        }]
    )
    result = parse_json_response(msg.content[0].text)
    return JSONResponse(result)


@app.post("/refine-prompt")
async def refine_prompt(
    instruction: str = Form(...),
    analysis_context: str = Form(...),
):
    """ユーザー指示をDALL-E用プロンプトに最適化"""
    msg = anthropic_client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1000,
        system=REFINE_SYSTEM,
        messages=[{
            "role": "user",
            "content": f"Design context: {analysis_context}\nUser instruction: {instruction}"
        }]
    )
    result = parse_json_response(msg.content[0].text)
    return JSONResponse(result)


@app.post("/generate")
async def generate(prompt: str = Form(...)):
    """DALL-E 3で画像生成"""
    try:
        response = openai_client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size="1024x1024",
            style="vivid",
            quality="standard",
            response_format="b64_json",   # URLではなくbase64で受け取る
        )
        b64 = response.data[0].b64_json
        return JSONResponse({"image": f"data:image/png;base64,{b64}"})
    except openai.BadRequestError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}

# index.html を同じフォルダから配信（APIルートより後に置く）
app.mount("/", StaticFiles(directory=".", html=True), name="static")
