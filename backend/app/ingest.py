"""
Enhanced ingestion for STEP textbook with concurrent image description.
"""
import fitz
import os
import time
import base64
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

PDF_PATH = "/Users/akhil/Downloads/STEP-textbook.pdf"
CHROMA_PATH = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SKIP_CHAPTERS = {
    "Title Page", "Copyright Page", "Contents", "Preface", "Cover",
    "About the Editors", "Contributing Authors", "Associate Authors",
    "Faculty Advisors", "General Acknowledgments", "Special Acknowledgments",
    "How to Contribute", "Photo Acknowledgments", "Index",
    "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M",
    "N", "O", "P", "R", "S", "T", "U", "V", "Z",
}


def build_chapter_map(toc, total_pages):
    chapter_map = {}
    current_chapter = "Unknown Chapter"
    toc_entries = sorted([(entry[2], entry[1]) for entry in toc])
    if not toc_entries:
        return {p: current_chapter for p in range(1, total_pages + 1)}
    entry_idx = 0
    for p in range(1, total_pages + 1):
        while entry_idx < len(toc_entries) and p >= toc_entries[entry_idx][0]:
            current_chapter = toc_entries[entry_idx][1]
            entry_idx += 1
        chapter_map[p] = current_chapter
    return chapter_map


def extract_images_from_page(page, page_num, chapter):
    images = []
    try:
        image_list = page.get_images(full=True)
    except Exception:
        return images
    for img_idx, img in enumerate(image_list):
        xref = img[0]
        try:
            base_image = page.parent.extract_image(xref)
            if base_image:
                image_bytes = base_image["image"]
                if len(image_bytes) < 20000:
                    continue
                img_b64 = base64.b64encode(image_bytes).decode("utf-8")
                ext = base_image.get("ext", "png")
                images.append({
                    "base64": img_b64, "ext": ext,
                    "page": page_num, "chapter": chapter,
                    "index": img_idx, "size_kb": len(image_bytes) // 1024,
                })
        except Exception:
            continue
    return images


def describe_single_image(img_data, client):
    """Describe one image using google.genai REST API. Returns (img_data, description) or (img_data, None)."""
    try:
        from google.genai.types import Part, Content

        img_bytes = base64.b64decode(img_data["base64"])

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                Part.from_bytes(
                    data=img_bytes,
                    mime_type=f"image/{img_data['ext']}",
                ),
                (
                    f"This image is from a medical textbook (First Aid for USMLE Step 1), "
                    f"section '{img_data['chapter']}', page {img_data['page']}. "
                    "Describe this image/diagram/figure in detail for a medical student. "
                    "Include all labels, arrows, pathways, and key relationships. "
                    "If a clinical photo, describe findings. If a flowchart, describe each step."
                ),
            ],
        )

        if response and response.text:
            return (img_data, response.text)
    except Exception as e:
        err = str(e)[:80]
        print(f"    [SKIP] page {img_data['page']}: {err}")

    return (img_data, None)


def ingest_pdf():
    if not os.path.exists(PDF_PATH):
        print(f"Error: PDF not found at {PDF_PATH}")
        return

    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("Cleared old ChromaDB data.")

    print(f"Opening PDF: {PDF_PATH}")
    doc = fitz.open(PDF_PATH)
    total_pages = len(doc)
    toc = doc.get_toc()
    chapter_map = build_chapter_map(toc, total_pages)

    # --- Phase 1: Text ---
    print("\n=== Phase 1: Text Extraction ===")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    text_documents = []
    all_images = []

    for i in range(total_pages):
        page = doc[i]
        page_num = i + 1
        chapter = chapter_map.get(page_num, "Unknown Chapter")
        if chapter in SKIP_CHAPTERS:
            continue

        text = page.get_text()
        if text.strip():
            chunks = text_splitter.split_text(text)
            for chunk in chunks:
                clean = chunk.strip()
                if len(clean) < 50:
                    continue
                contextualized = f"[Section: {chapter} | Page {page_num}]\n\n{clean}"
                text_documents.append(Document(
                    page_content=contextualized,
                    metadata={"page": page_num, "chapter": chapter, "source": "textbook", "type": "text"},
                ))

        page_images = extract_images_from_page(page, page_num, chapter)
        all_images.extend(page_images)

    print(f"Extracted {len(text_documents)} text chunks")
    print(f"Found {len(all_images)} images/diagrams to describe")

    # --- Phase 2: Image Descriptions (concurrent) ---
    image_documents = []
    if all_images:
        print(f"\n=== Phase 2: Image Description ({len(all_images)} images, 5 workers) ===")

        from google import genai
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

        success = 0
        skipped = 0
        completed = 0

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(describe_single_image, img, client): img
                for img in all_images
            }
            for future in as_completed(futures):
                completed += 1
                pct = (completed / len(all_images)) * 100
                img_data, description = future.result()

                if description:
                    contextualized = (
                        f"[Section: {img_data['chapter']} | Page {img_data['page']} | "
                        f"Figure/Diagram]\n\n{description}"
                    )
                    image_documents.append(Document(
                        page_content=contextualized,
                        metadata={
                            "page": img_data["page"], "chapter": img_data["chapter"],
                            "source": "textbook", "type": "figure",
                        },
                    ))
                    success += 1
                    print(f"  [{completed}/{len(all_images)}] ({pct:.0f}%) "
                          f"page {img_data['page']} {img_data['chapter']} ✓")
                else:
                    skipped += 1
                    if completed % 20 == 0:
                        print(f"  [{completed}/{len(all_images)}] ({pct:.0f}%) ... {skipped} skipped so far")

        print(f"\nImage results: {success} described, {skipped} skipped")

    # --- Phase 3: Embed ---
    all_documents = text_documents + image_documents
    print(f"\n=== Phase 3: Embedding {len(all_documents)} total chunks ===")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    batch_size = 200
    for i in range(0, len(all_documents), batch_size):
        batch = all_documents[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(all_documents) + batch_size - 1) // batch_size
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} chunks)")
        vectorstore.add_documents(batch)

    print(f"\n✅ Ingestion complete!")
    print(f"   Text chunks: {len(text_documents)}")
    print(f"   Image descriptions: {len(image_documents)}")
    print(f"   Total: {len(all_documents)}")


if __name__ == "__main__":
    ingest_pdf()
