import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import HTMLSectionSplitter, RecursiveCharacterTextSplitter

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Initialisez un index FAISS correctement configuré
dimension = 768  # Correspond à la taille des vecteurs produits par le modèle d'embeddings
index = faiss.IndexFlatL2(dimension)  # Crée un index FAISS basé sur la distance L2

url = "https://en.wikipedia.org/wiki/2023_Cricket_World_Cup"
loader = AsyncHtmlLoader(url)
html_data = loader.load()

# Fixed-sized by character chunking
# text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
#
# hmtl2text = Html2TextTransformer()
# html_data_transformed = hmtl2text.transform_documents(html_data)
#
# print(html_data_transformed[0].page_content)
# chunks=text_splitter.create_documents([html_data_transformed[0].page_content])
#
# print(f"The number of chunks is {len(chunks)}")
# print(f" -- chunk 4 ending -- \n {chunks[4].page_content[-200:]}")
# print(f"-- chunk 5 beginning -- \n {chunks[5].page_content[:200]}")


# Specialized (adaptative) chunking
sections_to_split_on = [("h1", "Header 1"), ("h2", "Header 2"), ("table", "Table"), ("p", "Paragraph")]
html_section_splitter = HTMLSectionSplitter(sections_to_split_on)
specialized_chunks = html_section_splitter.split_text(html_data[0].page_content)

print(f"The number of adaptative chunks is {len(specialized_chunks)}")

# Chaining recursive fixed-sized character chunking with adaptative chunking
recursive_text_splitter = RecursiveCharacterTextSplitter(["\n\n", "\n", "."], chunk_size=1000, chunk_overlap=100)
recursive_chunks = recursive_text_splitter.split_documents(specialized_chunks)
print(f"The number of recursive chunks is {len(recursive_chunks)}")

# hf_embeddings = embeddings.embed_documents([chunk.page_content for chunk in recursive_chunks])
# print(f"embeddings dimensions {len(hf_embeddings[0])}")

# Instantiate the FAISS object
vectorstore = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)
vectorstore.add_documents(recursive_chunks)

# vectorstore = FAISS.from_documents(recursive_chunks, embeddings)
vectorstore.save_local("indexes", "cricket_world_cup_index")

print(f"The number of indexed chunks {vectorstore.index.ntotal}")

saved_vectorstore = FAISS.load_local(
    folder_path="indexes",
    embeddings=embeddings,
    index_name="cricket_world_cup_index",
    allow_dangerous_deserialization=True
)

query = "Who won the 2023 World Cup final match?"

docs = saved_vectorstore.similarity_search(query, k=2)

for i, doc in enumerate(docs):
    print(f"Page {i + 1}:\n{doc.page_content}\n{'-' * 40}")
