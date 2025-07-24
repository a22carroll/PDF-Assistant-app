import tkinter as tk
from tkinter import filedialog, messagebox
from app.pdf_parser import parse_pdf
from app.vector_store import chunk_text, embed_chunks, store_embeddings
from app.rag_engine import answer_query

class PDFAssistantApp:
    def __init__(self, root):
        self.root = root
        root.title("PDF Assistant")
        root.geometry("600x400")

        self.upload_button = tk.Button(root, text="Upload PDF", command=self.upload_pdf)
        self.upload_button.pack(pady=10)

        self.query_entry = tk.Entry(root, width=80)
        self.query_entry.pack(pady=10)

        self.ask_button = tk.Button(root, text="Ask", command=self.ask_question)
        self.ask_button.pack(pady=5)

        self.result_text = tk.Text(root, height=15, wrap=tk.WORD)
        self.result_text.pack(pady=10)

    def upload_pdf(self):
        path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if not path:
            return

        try:
            text = parse_pdf(path)
            chunks = chunk_text(text)
            embeddings = embed_chunks(chunks)
            store_embeddings(chunks, embeddings)
            messagebox.showinfo("Success", "PDF uploaded and processed!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process PDF:\n{e}")

    def ask_question(self):
        query = self.query_entry.get()
        if not query.strip():
            return

        try:
            answer = answer_query(query)
            self.result_text.delete("1.0", tk.END)
            self.result_text.insert(tk.END, answer)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to get answer:\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = PDFAssistantApp(root)
    root.mainloop()
