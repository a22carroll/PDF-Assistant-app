from openai import OpenAI
from app.vector_store import get_context

client = OpenAI()

def answer_query(query):
    context = get_context(query)
    prompt = f"Use the following context to answer the question:\n\n{'\n---\n'.join(context)}\n\nQuestion: {query}"

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content