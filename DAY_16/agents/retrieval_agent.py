retriever=None
def set_retriever(r):
    global retriever; 
    retriever=r
def retrieval_agent(state):
    state['documents']=retriever.invoke(state['question'])
    return state
