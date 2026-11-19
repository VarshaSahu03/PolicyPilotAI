
import re

# ==========================================
# 1. BAD WORDS
# ==========================================

BAD_WORDS = [
    "fuck",
    "shit",
    "bitch"
]


# ==========================================
# 2. PROMPT INJECTION PATTERNS
# ==========================================

INJECTION_PATTERNS = [
    r"\bignore\b.{0,50}\b(instructions?|rules?|guidelines?|system prompt)\b",
    r"\bdisregard\b.{0,50}\b(instructions?|rules?|policy)\b",
    r"\bforget\b.{0,30}\b(previous|all|your)\b.{0,20}\b(instructions?|rules?)\b",
    r"\byou are now\b",
    r"\bpretend (to be|you are)\b",
    r"\breveal\b.{0,30}\b(system prompt|hidden instructions?)\b",
    r"\bshow\b.{0,30}\b(system prompt|hidden instructions?)\b",
    r"\bdeveloper message\b",
    r"\bsystem prompt\b",
    r"\bact as\b.{0,30}\b(unrestricted|jailbroken|unfiltered)\b"
]


# ==========================================
# 3. SENSITIVE INFORMATION PATTERNS
# ==========================================

SENSITIVE_PATTERNS = [
    r"\bapi[_ -]?key\b",
    r"\bsecret[_ -]?key\b",
    r"\baccess[_ -]?token\b",
    r"\bprivate[_ -]?key\b",
]


# ==========================================
# 4. HARMFUL REQUEST PATTERNS
# ==========================================

HARMFUL_REQUEST_PATTERNS = [
    r"\b(hack|crack|steal|guess)\b.{0,40}\b(password|credential|account)\b",
    r"\b(password|credential|account)\b.{0,40}\b(hack|crack|steal)\b"
]


# ==========================================
# 5. INPUT GUARDRAIL
# ==========================================

def check_input_guardrail(question: str):

    if not isinstance(question, str) or not question.strip():
        return False, "Question is empty. Please type a question."

    q = question.lower().strip()

    # A. Length validation
    if len(question) > 500:
        return False, "Question is too long. Please ask a shorter question."

    # B. Prompt injection detection
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, q):
            return False, (
                "I cannot ignore my instructions or reveal internal prompts. "
                "Please ask a company-policy-related question."
            )

    # C. Abusive language detection
    for word in BAD_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", q):
            return False, (
                "Please use respectful language and ask your question politely."
            )

    # D. Harmful request detection
    for pattern in HARMFUL_REQUEST_PATTERNS:
        if re.search(pattern, q):
            return False, (
                "I cannot help with hacking passwords or accessing accounts. "
                "Please ask a company-policy-related question."
            )

    return True, "OK"


# ==========================================
# 6. OUTPUT GUARDRAIL
# ==========================================

def check_output_guardrail(answer: str):

    if not isinstance(answer, str) or len(answer.strip()) < 5:
        return False, (
            "Sorry, I could not find an answer in the company policy documents."
        )

    answer_low = answer.lower()

    for pattern in SENSITIVE_PATTERNS:
        if re.search(pattern, answer_low):
            return False, (
                "Sorry, I cannot share confidential credentials or secret keys."
            )

    return True, answer
