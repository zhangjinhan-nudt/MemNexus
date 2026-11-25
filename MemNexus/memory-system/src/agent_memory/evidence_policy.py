"""Conservative speech-act guard; questions are context, not affirmative evidence."""

import re

_QUESTION_HEAD = re.compile(
    r"(?i)^(?:is|are|was|were|does|do|did|can|could|who|whom|whose|which|what|why|where|when|how)\b"
)
_DECLARATIVE_START = re.compile(
    r"^(?:(?i:i|you|he|she|it|we|they|there|this|that|these|those)"
    r"(?:['’](?i:m|re|s|ve|d|ll))?\s+\S+"
    r"|(?i:the|a|an|my|your|his|her|its|our|their)\s+\S+\s+\S+"
    r"|[A-Z][\w'’.-]*\s+\S+)"
)
_SCOPE_HEAD = re.compile(r"(?i)^(?:if|unless|whether|provided|assuming|supposing)\b")
# These are grammatical scope operators, not entities, answer types or benchmark
# phrases. A clipped positive substring cannot silently discard its operator.
_SCOPE_OPERATOR = re.compile(
    r"(?i)\b(?:if|unless|whether|not(?!\s+only\b)|never|no|without|cannot)\b|\b\w+n['’]t\b"
)
_CONDITIONAL_OPERATOR = re.compile(r"(?i)\b(?:if|unless|whether)\b")
_SHARED_PREDICATE_HEAD = re.compile(
    r"(?i)^(?:am|is|are|was|were|have|has|had|do|does|did|can|could|will|would|shall|should|may|might|must)\b"
)


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    spans = []
    start = 0
    for boundary in re.finditer(r"[。！？!?；;\n]+|\.(?=\s+[A-Za-z\"“])", text):
        if boundary.group() == "." and re.search(
            r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|e\.g|i\.e)$|\b(?:[A-Z]\.)*[A-Z]$", text[:boundary.start()]
        ):
            continue
        spans.append((start, boundary.end()))
        start = boundary.end()
    spans.append((start, len(text)))
    return [(left, right) for left, right in spans if text[left:right].strip()]


def _clause_spans(text: str) -> list[tuple[int, int]]:
    """Split only explicit independent-clause boundaries, retaining offsets.

    A terminal request must not erase a preceding independent statement. In
    contrast, a fronted if/when clause cannot be promoted by cutting its comma.
    This deliberately leaves uncertain syntax together for the existing reader.
    """
    spans = []
    for left, right in _sentence_spans(text):
        cursor = left
        for boundary in re.finditer(r",\s*(?:(?:and|but|yet|so)\s+)?|\s+(?:and|but|yet|so)\s+",
                                    text[left:right], re.IGNORECASE):
            cut, following = left + boundary.start(), left + boundary.end()
            prefix = text[cursor:cut].strip()
            suffix = text[following:right].strip()
            coordinated = bool(re.search(r"\b(?:and|but|yet|so)\b", boundary.group(), re.IGNORECASE))
            independent_prefix = (not _SCOPE_HEAD.match(prefix) and not _QUESTION_HEAD.match(prefix)
                                  and _DECLARATIVE_START.match(prefix))
            head = _QUESTION_HEAD.match(suffix)
            question_suffix = head and (suffix.endswith(('?', '？')) or
                head.group().lower() in {'who', 'whom', 'whose', 'which', 'what', 'why', 'where', 'when', 'how'} or
                _DECLARATIVE_START.match(suffix[head.end():].lstrip()))
            independent_suffix = (question_suffix or (coordinated and not head and
                (_DECLARATIVE_START.match(suffix) or
                 (_SHARED_PREDICATE_HEAD.match(suffix) and not _CONDITIONAL_OPERATOR.search(prefix)))))
            # Consequential coordination requires an explicit independent subject.
            # Do not split purpose clauses ("so that ...") or carry a premise
            # out of a genuinely conditional first clause.
            consequential = bool(re.search(r"\bso\b", boundary.group(), re.IGNORECASE))
            if consequential and (_CONDITIONAL_OPERATOR.search(prefix)
                                  or not _DECLARATIVE_START.match(suffix)
                                  or re.match(r"(?i)^that\b", suffix)):
                continue
            if independent_prefix and independent_suffix:
                spans.append((cursor, cut))
                cursor = following
        spans.append((cursor, right))
    return [(left, right) for left, right in spans if text[left:right].strip()]


def source_assertion_spans(text: str) -> list[tuple[int, int]]:
    """Original offsets of statements, including qualified/negative statements.

    These spans describe speech acts, not truth or entailment. Call
    ``assertion_supported`` on a selected quote before using a clipped span.
    """
    return [(left, right) for left, right in _clause_spans(text)
            if not question_only(text[left:right].strip())]


def source_assertion_scopes(text: str) -> list[dict]:
    """Non-textual provenance: source offsets and grammatical operator ranges.

    Consumers may translate every offset into a displayed fragment's coordinate
    system, including negative offsets for a clipped qualifier. No raw neighbor
    content is carried into API responses or receipts by this representation.
    """
    scopes = []
    padding = " \t\r\n.,;:!?。；：！？"
    for left, right in source_assertion_spans(text):
        while left < right and text[left] in padding:
            left += 1
        while right > left and text[right - 1] in padding:
            right -= 1
        if left == right:
            continue
        operators = [
            {'start': left + match.start(), 'end': left + match.end(),
             'kind': 'conditional' if _CONDITIONAL_OPERATOR.fullmatch(match.group()) else 'negative'}
            for match in _SCOPE_OPERATOR.finditer(text[left:right])
        ]
        scopes.append({'start': left, 'end': right, 'operators': operators})
    return scopes


def assertion_span_supported(scopes: list[dict], start: int, end: int) -> bool:
    """Apply the same source-scope policy to offsets without carrying source text."""
    if end <= start:
        return False
    for scope in scopes:
        if min(end, scope['end']) <= max(start, scope['start']):
            continue
        # Later negative adjuncts do not negate an earlier main clause. An
        # if/unless condition can govern either side and cannot be discarded.
        operators = (operator for operator in scope['operators']
                     if operator['start'] < end or operator['kind'] == 'conditional')
        if all(start <= operator['start'] and end >= operator['end'] for operator in operators):
            return True
    return False


def assertion_supported(source_text: str, quote: str, *,
                        start: int | None = None, end: int | None = None) -> bool:
    """Whether a literal quote preserves an asserting source's speech-act scope.

    Broad quotes may include context plus an assertion. A subquote is evaluated
    only at its own original location, never licensed by another sentence in the
    event. All ambiguous literal occurrences must qualify unless exact source
    coordinates disambiguate them. This is not a claim-entailment classifier.
    """
    if not quote or (start is None) != (end is None):
        return False
    if start is not None:
        if start < 0 or end <= start or source_text[start:end] != quote:
            return False
        occurrences = [(start, end)]
    else:
        occurrences = []
        offset = source_text.find(quote)
        while offset >= 0:
            occurrences.append((offset, offset + len(quote)))
            offset = source_text.find(quote, offset + 1)
    if not occurrences:
        return False
    scopes = source_assertion_scopes(source_text)
    return all(assertion_span_supported(scopes, left, right) for left, right in occurrences)


def listed_label_supported(text: str, quote: str, predicate: str, value: str, *,
                           start: int | None = None, end: int | None = None) -> bool:
    """A verbatim named outline label is data, not its proposed factual answer."""
    if quote.strip() != value.strip() or not re.search(
        r'\b(?:titles?|topics?|headings?|names?)\b|标题|主题|话题|名称',
        predicate.replace('_', ' '), re.IGNORECASE,
    ):
        return False
    if (start is None) != (end is None):
        return False
    if start is not None:
        if start < 0 or end <= start or text[start:end] != quote:
            return False
        locations = [(start, end)]
    else:
        locations = [(match.start(), match.end()) for match in re.finditer(re.escape(quote), text)]
    for quote_start, quote_end in locations:
        line_start = text.rfind('\n', 0, quote_start) + 1
        line_end = text.find('\n', quote_end)
        line = text[line_start:line_end if line_end >= 0 else len(text)].strip()
        if re.match(r'^(?:[-*+]\s|\d+[.)]\s|#{1,6}\s|\||(?:topic|title|heading|name)\s*\d*\s*:)',
                    line, re.IGNORECASE):
            continue
        if (line == quote and len(text.splitlines()) > 1
                and re.search(r'\b(?:titles?|topics?|headings?)\b|标题|主题|话题', predicate, re.IGNORECASE)
                and re.match(r'^(?:how|where|when|what)\s+to\b', line, re.IGNORECASE)
                and '?' not in line and '？' not in line):
            continue
        return False
    return bool(locations)


def _fronted_declarative(clause: str) -> bool:
    """Recognize an explicit adjunct/main-clause boundary, not just a WH word.

    ``When I arrived, I called`` asserts a fact, unlike ``When did I arrive``
    or ``When I arrived, who called``. Without a comma or recognizable subject
    starts, keep the existing conservative classification of ambiguous WH text.
    This is speech-act recognition, not a parser or a claim-truth test.
    """
    match = re.match(r"(?i)^(?:when|where)\s+(.+?),\s*(.+)$", clause)
    if not match:
        return False
    return all(not _QUESTION_HEAD.match(part) and _DECLARATIVE_START.match(part)
               for part in match.groups())


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
    clauses = [text[left:right].strip() for left, right in _sentence_spans(text)]
    if not clauses:
        return False

    def question(clause):
        value = clause.rstrip("。；;！! ")
        return bool(
            value.endswith(("?", "？", "吗", "么", "呢"))
            or re.search(r"是不是|是否|能否|有没有|为什么|怎么办|谁是|什么是", value)
            or (_QUESTION_HEAD.match(value) and not _fronted_declarative(value))
        )

    return all(question(clause) for clause in clauses)


def source_sentence_spans(text: str) -> list[dict]:
    """Context boundaries for reader interpretations, without a truth label.

    Include interrogatives and hypotheses. A consumer may interpret a premise
    inside them only while retaining the surrounding sentence, not by clipping
    away its operators. Offsets can be transported without leaking source text.
    """
    result = []
    for left, right in _sentence_spans(text):
        while left < right and text[left].isspace():
            left += 1
        while right > left and text[right - 1].isspace():
            right -= 1
        if left < right:
            result.append({'start': left, 'end': right})
    return result

