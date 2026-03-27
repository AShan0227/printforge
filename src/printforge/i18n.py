"""Internationalization support for PrintForge v2.2.

Simple key-value translation with fallback to English.
"""

from typing import Dict, Optional

TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "en": {
        "app.name": "PrintForge",
        "app.tagline": "One photo to 3D print",
        "gen.started": "Generation started",
        "gen.processing": "Processing your image...",
        "gen.done": "3D model generated successfully!",
        "gen.failed": "Generation failed",
        "gen.multi_view": "Multi-view mode: 4 angles for better detail",
        "auth.registered": "Account created! Save your API key.",
        "auth.key_warning": "This key won't be shown again",
        "quota.remaining": "{remaining} generations remaining",
        "quota.exhausted": "Quota exhausted. Upgrade your plan.",
        "share.created": "Share link created",
        "share.embed_hint": "Copy the embed code to add to your website",
        "error.generic": "Something went wrong. Please try again.",
        "error.network": "Network error. Check your connection.",
        "material.pla": "PLA — Easy to print, great for prototyping",
        "material.petg": "PETG — Strong and chemical resistant",
        "material.abs": "ABS — Tough, needs enclosure",
    },
    "zh": {
        "app.name": "烟戏",
        "app.tagline": "一张照片，变成3D打印",
        "gen.started": "开始生成",
        "gen.processing": "正在处理你的图片...",
        "gen.done": "3D模型生成成功！",
        "gen.failed": "生成失败",
        "gen.multi_view": "多视图模式：4个角度获得更多细节",
        "auth.registered": "账号创建成功！请保存你的API密钥。",
        "auth.key_warning": "此密钥不会再次显示",
        "quota.remaining": "剩余 {remaining} 次生成配额",
        "quota.exhausted": "配额已用完，请升级套餐。",
        "share.created": "分享链接已创建",
        "share.embed_hint": "复制嵌入代码添加到你的网站",
        "error.generic": "出了点问题，请重试。",
        "error.network": "网络错误，请检查连接。",
        "material.pla": "PLA — 易打印，适合原型制作",
        "material.petg": "PETG — 强韧，耐化学腐蚀",
        "material.abs": "ABS — 坚韧，需要密封仓",
    },
    "ja": {
        "app.name": "PrintForge",
        "app.tagline": "写真一枚で3Dプリント",
        "gen.started": "生成開始",
        "gen.done": "3Dモデル生成完了！",
        "gen.failed": "生成失敗",
    },
}

_current_locale = "en"


def set_locale(locale: str):
    global _current_locale
    if locale in TRANSLATIONS:
        _current_locale = locale


def t(key: str, locale: Optional[str] = None, **kwargs) -> str:
    """Translate a key. Falls back to English if not found."""
    loc = locale or _current_locale
    text = TRANSLATIONS.get(loc, {}).get(key)
    if text is None:
        text = TRANSLATIONS.get("en", {}).get(key, key)
    if kwargs:
        text = text.format(**kwargs)
    return text


def get_supported_locales() -> list:
    return list(TRANSLATIONS.keys())
