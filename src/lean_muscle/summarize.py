"""One Chinese sentence from the numbers. The model never sees the photo."""

from __future__ import annotations

import json

from openai import OpenAI

from .errors import GatewayError
from .spec import SENTENCE_MAX_CHARS, SUMMARY_MODEL, SUMMARY_TEMPERATURE

_SYSTEM = """你在给一组身材比例数字写一句总结。你没有看见照片，不能另打一套分。

要求：
- 简体中文，只有一句，不超过40个字（含标点）
- 必须原样包含用户给出的肩腰比，一个数字都不能改
- 核心是腹肌、胸肌、肩腰比。背肌和手臂可以带一句，但不要抢主句
- 损这组比例，像健身评论区，不损人
- 不提脸、身高、种族、性别、性吸引力、疾病，不写体脂百分比
- 不用 emoji，不加引号，不换行
- 句号、问号、感叹号最多一个
- 只有用户说手臂未入画时，才写「手臂都不漏」，并点出可能是太细了。手出画不是拍摄失误

只返回 JSON：{"sentence": "..."}"""

_BANNED = ("体脂", "性感", "疾病")


def ratio_text(ratio: float) -> str:
    return f"{ratio:.2f}"


def sentence_problem(text: str, ratio: str, *, arms_hidden: bool = False) -> str | None:
    cleaned = text.strip().strip("「」\"'")
    if not cleaned:
        return "空句子"
    if "\n" in cleaned or "\r" in cleaned:
        return "换行了"
    if len(cleaned) > SENTENCE_MAX_CHARS:
        return f"有 {len(cleaned)} 个字，超过 {SENTENCE_MAX_CHARS}"
    if ratio not in cleaned:
        return f"没有原样包含肩腰比 {ratio}"
    if sum(cleaned.count(mark) for mark in "。！？") > 1:
        return "写成了两句"
    if arms_hidden and "手臂都不漏" not in cleaned:
        return "手臂没入画，句子里却没有「手臂都不漏」"
    if arms_hidden and "太细" not in cleaned:
        return "手臂没入画，句子里却没点出可能是太细了"
    if not arms_hidden and "手臂都不漏" in cleaned:
        return "手臂在画面里，不该写「手臂都不漏」"
    for word in _BANNED:
        if word in cleaned:
            return f"提到了不该写的「{word}」"
    return None


def _user_prompt(measures: dict, scores: dict, ratio: str, earlier: str | None) -> str:
    hidden = not measures.get("arms_visible", True)
    lines = [
        f"肩腰比（必须原样写入）：{ratio}",
        f"胸腰比：{measures.get('chest_waist_ratio')}",
        f"背阔（腋下宽/腰宽）：{measures.get('lat_waist_ratio')}",
        f"腹肌沟条数：{measures['abs_grooves']}",
        f"腹肌分：{scores['abs']}",
        f"胸肌分：{scores['chest']}",
        f"肩腰分：{scores['shoulder']}",
        f"背肌分：{scores['back']}",
        f"手臂分：{scores['arm']}",
        f"总分：{scores['total']}",
        "这些分数没有用真人照片标定。正面照看不到真正的后背，背肌分是腋下那一圈有没有比腰宽。",
    ]
    if hidden:
        lines.append("手臂未入画。句子必须包含「手臂都不漏」，并写成可能是太细了。不要说成没拍好。")
    else:
        lines.append(f"臂腰比：{measures['arm_waist_ratio']:.2f}")
        lines.append("手臂在画面里。不要写「手臂都不漏」。")
    if earlier:
        lines.append(f"上一次不合格，原因：{earlier}。重写一句。")
    return "\n".join(lines)


def _ask(client: OpenAI, prompt: str, temperature: float) -> str:
    try:
        response = client.chat.completions.create(
            model=SUMMARY_MODEL,
            temperature=temperature,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "lean_sentence",
                    "schema": {
                        "type": "object",
                        "properties": {"sentence": {"type": "string"}},
                        "required": ["sentence"],
                        "additionalProperties": False,
                    },
                },
            },
        )
    except Exception as exc:  # noqa: BLE001
        raise GatewayError(f"总结模型调用失败：{exc}") from exc
    content = response.choices[0].message.content or ""
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise GatewayError(f"总结不是 JSON：{content[:200]}") from exc
    sentence = payload.get("sentence")
    if not isinstance(sentence, str):
        raise GatewayError("总结 JSON 里没有 sentence")
    return sentence.strip().strip("「」\"'")


def write_sentence(client: OpenAI, measures: dict, scores: dict) -> tuple[str | None, str | None]:
    """Return ``(sentence, error)``. A bad model reply does not raise."""
    ratio = ratio_text(float(measures["shoulder_waist_ratio"]))
    problem: str | None = None
    try:
        for temperature in (SUMMARY_TEMPERATURE, 0.4):
            text = _ask(
                client,
                _user_prompt(measures, scores, ratio, problem),
                temperature,
            )
            problem = sentence_problem(
                text,
                ratio,
                arms_hidden=not measures.get("arms_visible", True),
            )
            if problem is None:
                return text, None
    except GatewayError as exc:
        return None, str(exc)
    return None, problem or "总结没有通过校验"
