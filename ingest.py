"""
Doküman yükleme ve vektör veritabanı oluşturma scripti.
docs/ klasöründeki PDF ve TXT dosyalarını yükler, böler ve ChromaDB'ye kaydeder.
"""

import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

def load_documents(docs_dir: str = "docs"):
    """docs/ klasöründeki tüm PDF ve TXT dosyalarını yükler."""
    documents = []
    docs_path = Path(docs_dir)
    
    if not docs_path.exists():
        raise FileNotFoundError(f"{docs_dir} klasörü bulunamadı!")
    
    # PDF dosyalarını yükle
    pdf_files = list(docs_path.glob("*.pdf"))
    for pdf_file in pdf_files:
        print(f"PDF yükleniyor: {pdf_file.name}")
        loader = PyPDFLoader(str(pdf_file))
        documents.extend(loader.load())
    
    # TXT dosyalarını yükle
    txt_files = list(docs_path.glob("*.txt"))
    for txt_file in txt_files:
        print(f"TXT yükleniyor: {txt_file.name}")
        loader = TextLoader(str(txt_file), encoding='utf-8')
        documents.extend(loader.load())
    
    if not documents:
        raise ValueError(f"{docs_dir} klasöründe PDF veya TXT dosyası bulunamadı!")
    
    print(f"Toplam {len(documents)} doküman yüklendi.")
    return documents

def split_documents(documents, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Dokümanları belirtilen boyutlarda parçalara böler."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Dokümanlar {len(chunks)} parçaya bölündü.")
    return chunks

def create_vectorstore(chunks, persist_directory: str = "./chroma_db", api_key: str = None):
    """Vektörleri oluşturur ve ChromaDB'ye kaydeder."""
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API Key gerekli! Lütfen OPENAI_API_KEY ortam değişkenini ayarlayın veya api_key parametresini verin.")
    
    # OpenAI embeddings oluştur
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    
    # ChromaDB'ye kaydet
    print(f"Vektörler oluşturuluyor ve {persist_directory} klasörüne kaydediliyor...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    print(f"Vektör veritabanı başarıyla oluşturuldu: {persist_directory}")
    return vectorstore

def main():
    """Ana fonksiyon: Tüm işlemleri sırayla yürütür."""
    print("=" * 50)
    print("Doküman Yükleme ve Vektör Veritabanı Oluşturma")
    print("=" * 50)
    
    # --- EKLENEN KISIM: API Key'i Kullanıcıdan İste ---
    import getpass
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  Ortam değişkenlerinde API Key bulunamadı.")
        # getpass şifre girerken ekranda görünmesini engeller (yıldız bile çıkmaz, güvenlidir)
        # Eğer getpass sorun olursa düz input() da kullanabilirsin.
        try:
            api_key = getpass.getpass("👉 Lütfen OpenAI API Key'inizi yapıştırıp Enter'a basın: ")
        except:
            api_key = input("👉 Lütfen OpenAI API Key'inizi yapıştırıp Enter'a basın: ")
            
    if not api_key or not api_key.startswith("sk-"):
        print("\n❌ Geçersiz veya boş API Key! Program sonlandırılıyor.")
        return
    # ----------------------------------------------------

    try:
        # 1. Dokümanları yükle
        documents = load_documents("docs")
        
        # 2. Dokümanları parçalara böl
        chunks = split_documents(documents, chunk_size=1000, chunk_overlap=200)
        
        # 3. Vektör veritabanını oluştur (API Key'i buraya gönderiyoruz)
        vectorstore = create_vectorstore(chunks, persist_directory="./chroma_db", api_key=api_key)
        
        print("\n✅ İşlem başarıyla tamamlandı!")
        print(f"📁 Vektör veritabanı oluşturuldu: ./chroma_db")
        
    except Exception as e:
        print(f"\n❌ Hata: {str(e)}")
        raise

if __name__ == "__main__":
    main()


