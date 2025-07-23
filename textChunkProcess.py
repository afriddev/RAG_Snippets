from langchain.text_splitter import CharacterTextSplitter

raw_text = """
your pdf text will be here and we need more cleaner text for efficient search on vector

"""
clean_text = " ".join(raw_text.splitlines()) 
text_splitter = CharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10,
    separator=""
)
chunks = text_splitter.split_text(clean_text)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:\n{chunk}\n{'-'*40}")
