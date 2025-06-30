from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer

url = "https://en.wikipedia.org/wiki/2023_Cricket_World_Cup"
loader = AsyncHtmlLoader(url)
data = loader.load()

hmtl2text = Html2TextTransformer()
html_data_transformed = hmtl2text.transform_documents(data)

print(html_data_transformed[0].page_content)
