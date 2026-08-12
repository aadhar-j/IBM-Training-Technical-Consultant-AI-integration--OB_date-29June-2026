from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os
import re
# CONFIGURATION

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MAX_QUERY_LENGTH = 500
MIN_WORDS = 3

SIMILARITY_THRESHOLD = 0.40

# PROMPT INJECTION PATTERNS

PROMPT_INJECTION_PATTERNS = [

    "ignore previous instructions",

    "ignore all previous instructions",

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

    "reveal your prompt",

    "show hidden prompt"

]

# RESTRICTED WORDS

BANNED_WORDS = [

    "hack",

    "malware",

    "virus",

    "exploit",

    "password",

    "phishing",

    "ransomware"

]

# DOMAIN DESCRIPTION

DOMAIN_DESCRIPTION = [
     "refund",
    "return",
    "replacement",
    "exchange",
    "damaged product",
    "defective product",
    "broken item",
    "wrong item",
    "missing item",
    "return request",
    "pickup",
    "warranty",
    "return eligibility",
    "refund status",
    "partial refund",
    "refund delay",

    "delivery",
    "shipment",
    "courier",
    "tracking",
    "delayed delivery",
    "late delivery",
    "delivery delay",
    "reschedule delivery",
    "package missing",
    "package lost",
    "not delivered",
    "wrong address",
    "tracking update",
    "shipping issue",

    "payment failed",
    "payment declined",
    "payment rejected",
    "duplicate payment",
    "double charge",
    "charged twice",
    "invoice",
    "billing issue",
    "transaction failed",
    "payment status",
    "bank verification",
    "wallet balance",
    "refund to source",

    "order status",
    "order cancellation",
    "cancel order",
    "modify order",
    "wrong order",
    "order update",
    "order confirmation",
    "placed order",
    "pending order",
    "order processing",

    "login issue",
    "unable to login",
    "account locked",
    "password reset",
    "forgot password",
    "verification failed",
    "account access",
    "profile update",
    "account suspended",
    "otp issue",
    "authentication",

    "app crash",
    "application crash",
    "website down",
    "page not loading",
    "loading issue",
    "technical error",
    "server error",
    "bug",
    "cache issue",
    "clear cache",
    "reinstall app",
    "refresh page",
    "system issue",
    "feature not working",

    "discount not applied",
    "coupon not working",
    "promo code",
    "cashback",
    "reward points",
    "offer expired",
    "offer unavailable",
    "discount issue",
    "promotion",
    "voucher",
    "coupon redemption",

    "complaint",
    "issue",
    "problem",
    "unable",
    "failed",
    "rejected",
    "damaged",
    "defective",
    "broken",
    "missing",
    "wrong",
    "delayed",
    "refund",
    "return",
    "replacement",
    "cancel",
    "escalate",
    "support",
    "help",
    "resolution",
    "compensation"
]

# LOAD EMBEDDING MODEL

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# CREATE DOMAIN EMBEDDING
# (Only once)

domain_embeddings = [embedding_model.embed_query(DESCRIPTION) for DESCRIPTION in DOMAIN_DESCRIPTION]

# EMPTY QUESTION

def validate_empty(question):

    if len(question.strip()) == 0:

        return False, "Please enter a question."

    return True, ""


# LENGTH CHECK

def validate_length(question):

    if len(question) > MAX_QUERY_LENGTH:

        return False, "Question is too long."

    return True, ""


# MINIMUM WORDS

def validate_min_words(question):

    if len(question.split()) < MIN_WORDS:

        return False, "Please ask a more descriptive question."

    return True, ""


# PROMPT INJECTION DETECTION

def detect_prompt_injection(question):

    q = question.lower()

    for pattern in PROMPT_INJECTION_PATTERNS:

        if pattern in q:

            return False, "Prompt Injection Detected."

    return True, ""


# RESTRICTED WORDS

def detect_banned_words(question):

    q = question.lower()

    for word in BANNED_WORDS:

        if word in q:

            return False, f"Restricted keyword detected: {word}"

    return True, ""


# SEMANTIC DOMAIN VALIDATION

def semantic_domain_validation(question):

    question_embedding = embedding_model.embed_query(
        question
    )

    similarities = [
        cosine_similarity([question_embedding], [domain_embedding])[0][0]
        for domain_embedding in domain_embeddings
        ]

    similarity = max(similarities)

    print("=" * 50)
    print("Semantic Similarity :", similarity)
    print("=" * 50)

    if similarity >= SIMILARITY_THRESHOLD:

        return True, ""

    return False, "Question is outside the supported domain."


# MAIN GUARDRAIL FUNCTION

def run_input_guardrails(question):

    validators = [

        validate_empty,

        validate_length,

        validate_min_words,

        detect_prompt_injection,

        detect_banned_words,

        semantic_domain_validation

    ]

    for validator in validators:

        status, message = validator(question)
        if not status:
            return status, message

    return True, ""


EMAIL_PATTERN = r"\S+@\S+\.\S+"
PHONE_PATTERN = r"\b\d{10}\b"
AADHAR_PATTERN = r"\b\d{4}\s?\d{4}\s?\d{4}\b"
PAN_PATTERN = r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"

def mask_pii(question):

    original_question = question

    question = re.sub(
        EMAIL_PATTERN,
        "[EMAIL]",
        question
    )

    question = re.sub(
        PHONE_PATTERN,
        "[PHONE]",
        question
    )

    question = re.sub(
        AADHAR_PATTERN,
        "[AADHAR]",
        question
    )

    question = re.sub(
        PAN_PATTERN,
        "[PAN]",
        question
    )

    pii_found = original_question != question

    return pii_found, question