"""Conservative speech-act guard; questions are context, not affirmative evidence."""

import re


def explicit_persistence_request(text: str) -> bool:
    if question_only(text) or re.search(r"(?:不要|无需|不用|不必).{0,20}(?:记住|保存|长期)", text):
        return False
    return bool(
        re.search(
            r"请.{0,80}(?:记住|保存|长期记忆)|(?:存入|保存到|加入)(?:长期)?记忆|(?:作为|设为)长期规则|跨会话记住|长期保存",
            text,
        )
    )


def question_only(text: str) -> bool:
    # English statements followed by questions must remain usable evidence.
    # Split full stops while preserving decimals and common abbreviations.
    clauses = []
    start = 0
    for boundary in re.finditer(r"[。！？!?；;\n]+|\.(?=\s+[A-Za-z\"“])", text):
        if boundary.group() == "." and re.search(
            r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|e\.g|i\.e)$|\b(?:[A-Z]\.)*[A-Z]$",
            text[: boundary.start()],
        ):
            continue
        clauses.append(text[start : boundary.end()].strip())
        start = boundary.end()
    clauses.append(text[start:].strip())
    clauses = [part for part in clauses if part]
    if not clauses:
        return False

    def question(clause):
        value = clause.rstrip("。；;！! ")
        return bool(
            value.endswith(("?", "？", "吗", "么", "呢"))
            or re.search(r"是不是|是否|能否|有没有|为什么|怎么办|谁是|什么是", value)
            or re.match(
                r"(?i)^(?:is|are|was|were|does|do|did|can|could|who|what|why|where|when|how)\b", value
            )
        )

    return all(question(clause) for clause in clauses)
