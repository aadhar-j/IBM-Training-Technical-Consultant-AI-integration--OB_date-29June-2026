from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------
# Semantic CACHING
# ----------------------------------------------------------

QUERY_CACHE = []
CACHE_SIMILARITY_THRESHOLD = 0.95

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

def check_query_cache(query):

    question_embedding = embedding_model.embed_query(query)

    for item in QUERY_CACHE:

        similarity = cosine_similarity(
            [question_embedding],
            [item["embedding"]]
        )[0][0]

        if similarity >= CACHE_SIMILARITY_THRESHOLD:

            print("=" * 60)
            print("CACHE HIT")
            print("Similarity :", similarity)
            print("=" * 60)

            return True, item["answer"]

    return False, question_embedding


def save_query_cache(question, embedding, answer):

    QUERY_CACHE.append(
        {
            "question": question,
            "embedding": embedding,
            "answer": answer
        }
    )