import re

# Configuration

MAX_QUERY_LENGTH = 500
MIN_WORDS = 3

DOMAIN_KEYWORDS = [
    "employee",
    "leave",
    "policy",
    "salary",
    "attendance",
    "holiday",
    "vpn",
    "benefits",
    "security",
    "reimbursement",
    "learning",
    "performance"
]

PROMPT_INJECTION_PATTERNS = [

    "ignore previous instructions",

    "ignore all previous",

    "forget previous instructions",

    "forget everything",

    "system prompt",

    "developer message",

    "act as",

    "pretend to be",

    "jailbreak",

    "override",

    "bypass",

    "disable safety",

    "reveal prompt"
]

BANNED_WORDS = [

    "hack",

    "malware",

    "virus",

    "exploit",

    "password"
]

# Empty Query

def validate_empty(question):

    if len(question.strip()) == 0:
        return False, "Please enter a question."

    return True, ""

# Length

def validate_length(question):

    if len(question) > MAX_QUERY_LENGTH:
        return False, "Question is too long."

    return True, ""

# Minimum words

def validate_min_words(question):

    if len(question.split()) < MIN_WORDS:
        return False, "Please ask a more descriptive question."

    return True, ""

# Prompt Injection

def detect_prompt_injection(question):

    q = question.lower()

    for attack in PROMPT_INJECTION_PATTERNS:

        if attack in q:
            return False, "Potential prompt injection detected."

    return True, ""

# Restricted words

def detect_banned_words(question):

    q = question.lower()

    for word in BANNED_WORDS:

        if word in q:

            return False, f"Restricted keyword detected: {word}"

    return True, ""

# Domain Check

def validate_domain(question):

    q = question.lower()

    for keyword in DOMAIN_KEYWORDS:

        if keyword in q:

            return True, ""

    return False, (
        "This application only answers questions related "
        "to the uploaded Employee Handbook."
    )

# Run all guardrails

def run_input_guardrails(question):

    validators = [

        validate_empty,

        validate_length,

        validate_min_words,

        detect_prompt_injection,

        detect_banned_words,

        validate_domain
    ]

    for validator in validators:

        valid, message = validator(question)

        if not valid:
            return valid, message

    return True, ""