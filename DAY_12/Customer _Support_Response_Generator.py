import os
import streamlit as st
from dotenv import load_dotenv
import pandas as pd
from rank_bm25 import BM25Okapi

# =====================================
# Load Environment Variables
# =====================================

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

# =====================================
# Streamlit UI
# =====================================

st.set_page_config(page_title="Customer support system")
st.title("📚 Customer Support using BM25")

# ----------------------------------------

df = pd.read_excel(r"C:\Users\AadharJain\Desktop\IBM_Training-main\DAY_12\Complaint Dataset.xlsx")
print(df.columns.to_list())

ALLOWED_CATEGORIES = [

    "Strict",

    "Friendly",

    "All"

]

# ============================================================
# 6. TOKENIZATION
# ============================================================

def tokenize_text(text):
    return re.findall(r"\b\w+\b",text.lower())


# ============================================================
# 7. CREATE CHUNKS
# ============================================================

def create_chunks(pages_text, category, chunk_size=500, chunk_overlap=100):
    chunks = []

    for page in pages_text:
        text = page["text"]
        page_number = page["page"]
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            if chunk_text.strip():
                chunks.append({

                    "id":                    len(chunks),
                    "text":                  chunk_text,
                    "category":              category
                })

            start = (end - chunk_overlap)

    return chunks

# ============================================================
# LLM-BASED CATEGORY CLASSIFICATION
# ============================================================

def classify_query_with_llm(query):
    prompt = f"""
                You are a query classification system.
                Classify the user's question into exactly
                ONE of the following categories:

                Strict Policy Mode
                Friendly Tone Mode
                Fallback Mode

                Category definitions:


                Strict:
                Questions that contain basic enquiries, easy to resolve
                or easily resolvable.


                Friendly:
                Questions that have a tone of urgency, needs immediate resolution, or
                incase of a repeated question.


                All:
                Use this when the question does not clearly
                belong to one category or requires information
                from multiple categories.


                User Question:   {query}

                Return ONLY valid JSON in this format:
                {{
                    "category": "Strict"
                }}

                Do not include any additional text.
            """


    response = client.chat.completions.create(

        model="gpt-4o-mini",


        messages=[

            {

                "role": "system",
                "content":"You classify user queries into predefined categories."

            },

            {

                "role":"user",
                "content":prompt

            }

        ],

            temperature=0
        )


    content = (response.choices[0].message.content)
    try:
        result = json.loads(content)
        category = result.get(
            "category",
            "All"
        )

        if category not in ALLOWED_CATEGORIES:
            category = "All"

        return category

    except:
        return "All"


# 10. CREATE BM25 INDEX
# ============================================================

def create_bm25_index(chunks):
    tokenized_documents = [tokenize_text(chunk["text"])     for chunk in chunks]

    return BM25Okapi(tokenized_documents)


#  ============================================================
# 12. METADATA FILTERING
# ============================================================

def filter_chunks_by_category(chunks,category):
    if category == "All":
        return chunks
    return [chunk       for chunk in chunks     if chunk["category"] == category]


# ============================================================
# 14. BM25 RETRIEVAL
# ============================================================

def keyword_retrieval(query, bm25, chunks, top_k=10):
    tokenized_query = tokenize_text(query)
    scores = bm25.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[-min(top_k,len(chunks)):][::-1]

    results = []

    for index in top_indices:
        results.append({
            "chunk_id": chunks[index]["id"],
            "text":     chunks[index]["text"],
            "category": chunks[index]["category"],
            "bm25_score": float(scores[index])
        })

    return results

# -----------------Generating answer--------------------
# ------------------------------------------------------

def generate_answer(query, retrieved_chunks):
    context = ""
    for chunk in retrieved_chunks:
        context += f"""
                    Category: {chunk["category"]}
                    Content:  {chunk["text"]}

                """

    prompt = f"""
            You are a helpful document question-answering assistant.
            Answer the user's question using ONLY 
            the information in the provided context.
            Do not use outside knowledge.
            If the answer cannot be found in the context, say:

            "I could not find the answer in the uploaded documents."

    Context: {context}
    User Question: {query}
    Answer: """

    response = client.chat.completions.create(

        model="gpt-4o-mini",
        messages=[

            {
                "role": "system",
                "content": "Answer using only the retrieved document context."
            },

            {
                "role": "user",
                "content": prompt
            }

        ],

        temperature=0
    )

    return (response.choices[0].message.content)

# CREATE INPUT BOX

query = st.text_input("What's your enquiry:")
if query:
    with st.spinner("LLM is identifying the document category..."):
        detected_category = (classify_query_with_llm(query))
    st.info(f"LLM detected category: "f"**{detected_category}**")
   
    filtered_chunks = (filter_chunks_by_category(all_chunks, detected_category))
    if not filtered_chunks:
        st.warning("No matching documents found.")
        st.stop()
        st.write(f"Searching "f"{len(filtered_chunks)} chunks "f"after metadata filtering.")
   
        with st.spinner("Creating filtered search indexes..."):
            filtered_bm25 = (create_bm25_index(filtered_chunks))     

        # ====================================================
        # BM25 SEARCH
        # ====================================================

        keyword_results = (keyword_retrieval(
                                query,
                                filtered_bm25,
                                filtered_chunks,
                                top_k=10
                            ))

        # ====================================================
        # GENERATE ANSWER
        # ====================================================

        with st.spinner("Generating answer..."):
            answer = generate_answer(query, keyword_results)

        st.subheader("Answer")
        st.write(answer)

        #  ====================================================
        # DISPLAY SOURCES
        # ====================================================

        st.subheader("Retrieved Sources")
        for i, chunk in enumerate(keyword_results):
            with st.expander(f"Result {i + 1}"):
                st.write(chunk["text"])
                st.write(
                    f"Category: "
                    f"{chunk['category']}"
                )