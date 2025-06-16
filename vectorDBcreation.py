import fitz # PyMuPDF
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

def extract_text_from_pdf(pdf_path):
  text = ""
  try:
    doc = fitz.open(pdf_path)
    
    for page in doc:
        text += page.get_text()
    return text
  except Exception as ex:
    print(ex)
  finally:
    return text



try:
  data = extract_text_from_pdf("sdlc.pdf")
  
  
  # Split text into chunks
  #text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=50)
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=1000,
      chunk_overlap=50,
      separators=["\n\n", "\n", ".", " ", ""]
  )
  texts = text_splitter.split_text(data)
  
  # Use sentence-transformers model for embedding
  embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
  
  # Create Chroma vector store with local persistence
  persist_directory = 'chroma_db_sdlc_pdf'
  db = Chroma.from_texts(texts, embedding, persist_directory=persist_directory)
  db.persist()
except Exception as ex:
    print(ex)
finally:
  print("end")