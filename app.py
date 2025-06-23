import os
import uuid
import chardet
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler, filters,
    CallbackContext
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

from langchain_together import ChatTogether
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import asyncio
from langchain.prompts import PromptTemplate

# Путь к локальной базе
CHROMA_PATH = "chroma_db"

# Инициализация эмбеддингов и хранилища
embedding_model = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_model)

llm = ChatTogether(
    model="mistralai/Mistral-7B-Instruct-v0.1",
    together_api_key="3af52120bb258bd70f2c3a692ee58d9adb37e7052a166383a8444e468cdb5e2e",  # сюда вставь свой ключ
    temperature=0.4
)

template = """
Ты — ассистент, который отвечает на вопросы, используя только приведённые документы. Если ответа нет явно, попробуй понять намёки, метафоры и описания. Если всё равно не получается — скажи "ничего не нашёл".

Контекст:
===========
{context}
===========

Вопрос: {question}

Ответ:
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True,
    chain_type_kwargs={
        "prompt": prompt,
        "document_variable_name": "context"
    }
)

# Обработка текстовых файлов
async def handle_file(update: Update, context: CallbackContext):
    user_id = str(update.message.from_user.id)
    doc = update.message.document

    if not doc.file_name.endswith(".txt"):
        await update.message.reply_text("Пока поддерживаются только .txt файлы.")
        return

    # Отправим уведомление, что начинается обработка
    status_message = await update.message.reply_text("⏳ Подождите, файл обрабатывается...")

    file = await doc.get_file()
    file_path = f"temp_{uuid.uuid4()}.txt"
    await file.download_to_drive(file_path)

    # Определим кодировку
    with open(file_path, "rb") as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        encoding = result["encoding"] or "utf-8"  # fallback

    try:
        text = raw_data.decode(encoding)
    except UnicodeDecodeError:
        await status_message.edit_text(f"❌ Не удалось декодировать файл (кодировка: {encoding}).")
        os.remove(file_path)
        return

    os.remove(file_path)

    # Чанкинг
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.create_documents([text], metadatas=[{"user_id": user_id}])

    # Сохраняем в Chroma
    vectorstore.add_documents(chunks)

    await status_message.edit_text(f"✅ Файл обработан и добавлен в базу ({len(chunks)} чанков).")
# Обработка запроса
async def handle_query(update: Update, context: CallbackContext):
    query = update.message.text
    user_id = str(update.message.from_user.id)

    qa_chain.retriever.search_kwargs["filter"] = {"user_id": user_id}

    print("Using filter:", qa_chain.retriever.search_kwargs.get("filter"))
    print("QA chain input keys:", qa_chain.input_keys)

    result = await asyncio.to_thread(lambda: qa_chain.invoke({"query": query}))

    answer = result["result"]
    source_docs = result.get("source_documents", [])

    citations = "\n\n---\n".join(
        f"📄 {doc.page_content[:300]}..." for doc in source_docs if doc.metadata.get("user_id") == user_id
    )

    if not citations.strip():
        await update.message.reply_text("Не могу найти ответ на этот вопрос.")
        return

    final_response = f"🧠 Ответ:\n{answer}\n\n📚 Источники:\n{citations}"
    await update.message.reply_text(final_response[:4096])

# Функция удаления документов по user_id
async def handle_delete(update: Update, context: CallbackContext):
    user_id = str(update.message.from_user.id)
    # Удаляем документы с метадатой user_id из векторного хранилища
    try:
        vectorstore._collection.delete(where={"user_id": user_id})
        await update.message.reply_text("Ваши документы удалены из базы.")
    except Exception as e:
        await update.message.reply_text(f"Ошибка при удалении: {e}")

# Команда /help
async def handle_help(update: Update, context: CallbackContext):
    help_text = """
📌 <b>Инструкция по использованию:</b>

1. 📄 Отправь <b>.txt-файл</b> — он будет проанализирован и добавлен в базу знаний.
2. ❓ Задай вопрос — бот найдёт ответ на основе загруженных файлов.
3. 🗑️ Команда <b>/delete</b> удалит все твои загруженные документы.
4. ℹ️ Команда <b>/help</b> — покажет эту справку.

Формат: только .txt, язык: русский.

Пример:
— Загрузи «Мастер и Маргарита.txt»
— Спроси: <i>«Сколько лет Маргарите?»</i>

🔐 Все данные обрабатываются индивидуально и привязаны к твоему Telegram ID.
"""
    await update.message.reply_text(help_text, parse_mode="HTML")

def main():
    TOKEN = "7759875570:AAFM7nXGtnsOfpGCZEc3n8Dc912NKTmKm5w"
    app = Application.builder().token(TOKEN).build()

    app.add_handler(MessageHandler(filters.Document.TEXT, handle_file))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_query))
    app.add_handler(CommandHandler("start", lambda u, c: u.message.reply_text("Пришли .txt файл или задай вопрос! Чтобы узнать подробности, вызови /help.")))
    app.add_handler(CommandHandler("delete", handle_delete))
    app.add_handler(CommandHandler("help", handle_help))

    print("RAG-бот запущен...")
    app.run_polling()

if __name__ == "__main__":
    main()