import os
import uuid
import chardet
import logging
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

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    filename='bot.log',  
    filemode='a'
)
logger = logging.getLogger(__name__)

# Путь к локальной базе
CHROMA_PATH = "chroma_db"

# Инициализация эмбеддингов, хранилища и llm
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
    logger.info(f"User {user_id} uploaded file: {doc.file_name}")

    if not doc.file_name.endswith(".txt"):
        logger.warning(f"User {user_id} uploaded unsupported file type.")
        await update.message.reply_text("Пока поддерживаются только .txt файлы.")
        return

    # Отправим уведомление, что начинается обработка
    status_message = await update.message.reply_text("⏳ Подождите, файл обрабатывается...")

    file = await doc.get_file()
    file_path = f"temp_{uuid.uuid4()}.txt"
    await file.download_to_drive(file_path)
    logger.info(f"Downloaded file to {file_path}")

    # Определим кодировку
    with open(file_path, "rb") as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        encoding = result["encoding"] or "utf-8"  # fallback

    try:
        text = raw_data.decode(encoding)
    except UnicodeDecodeError:
        logger.error(f"Decoding error for user {user_id} with encoding {encoding}")
        await status_message.edit_text(f"❌ Не удалось декодировать файл (кодировка: {encoding}).")
        os.remove(file_path)
        return

    os.remove(file_path)
    logger.info(f"File {file_path} removed after decoding.")

    # Чанкинг
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.create_documents([text], metadatas=[{"user_id": user_id}])
    logger.info(f"Created {len(chunks)} chunks for user {user_id}")

    # Сохраняем в Chroma
    vectorstore.add_documents(chunks)
    logger.info(f"Chunks added to vectorstore for user {user_id}")

    await status_message.edit_text(f"✅ Файл обработан и добавлен в базу ({len(chunks)} чанков).")

# Обработка запроса
async def handle_query(update: Update, context: CallbackContext):
    query = update.message.text
    user_id = str(update.message.from_user.id)
    logger.info(f"User {user_id} submitted query: {query}")

    qa_chain.retriever.search_kwargs["filter"] = {"user_id": user_id}

    result = await asyncio.to_thread(lambda: qa_chain.invoke({"query": query}))

    answer = result["result"]
    source_docs = result.get("source_documents", [])

    citations = "\n\n---\n".join(
        f"📄 {doc.page_content[:300]}..." for doc in source_docs if doc.metadata.get("user_id") == user_id
    )

    if not citations.strip():
        logger.warning(f"No relevant sources found for user {user_id}'s query.")
        await update.message.reply_text("Не могу найти ответ на этот вопрос.")
        return

    final_response = f"Ответ:\n{answer}\n\n📚 Источники:\n{citations}"
    logger.info(f"Sending answer to user {user_id}")
    await update.message.reply_text(final_response[:4096])

# Функция удаления документов по user_id
async def handle_delete(update: Update, context: CallbackContext):
    user_id = str(update.message.from_user.id)
    logger.info(f"User {user_id} requested deletion of their documents.")
    # Удаляем документы с метадатой user_id из векторного хранилища
    try:
        vectorstore._collection.delete(where={"user_id": user_id})
        logger.info(f"Documents deleted for user {user_id}")
        await update.message.reply_text("Ваши документы удалены из базы.")
    except Exception as e:
        logger.error(f"Error deleting documents for user {user_id}: {e}")
        await update.message.reply_text(f"Ошибка при удалении: {e}")

# Команда /help
async def handle_help(update: Update, context: CallbackContext):
    user_id = str(update.message.from_user.id)
    logger.info(f"User {user_id} requested help.")
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

    logger.info("RAG Telegram bot started...")
    print("RAG-бот запущен...")
    app.run_polling()

if __name__ == "__main__":
    asyncio.run(main())