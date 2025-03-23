from langchain.prompts import (
     PromptTemplate,
     SystemMessagePromptTemplate,
     HumanMessagePromptTemplate,
     ChatPromptTemplate,
 )

review_system_template_str = """Your job is to use cutomer
 reviews to answer questions about their experience at a
 shop. Use the following context to answer questions.
 Be as detailed as possible, but don't make up any information
 that's not from the context. If you don't know an answer, say
 you don't know.

 {context}
 """

review_system_prompt = SystemMessagePromptTemplate(
     prompt=PromptTemplate(
         input_variables=["context"], template=review_system_template_str
     )
 )

review_human_prompt = HumanMessagePromptTemplate(
     prompt=PromptTemplate(
         input_variables=["question"], template="{question}"
     )
 )

messages = [review_system_prompt, review_human_prompt]
review_prompt_template = ChatPromptTemplate(
     input_variables=["context", "question"],
     messages=messages,
 )

def test_template_components():
    context = "I had a great stay!"
    question = "Did anyone have a positive experience?"
    messages = review_prompt_template.format_messages(context=context, question=question)
    assert any(m.type == "system" and context in m.content for m in messages)
    assert any(m.type == "human" and question in m.content for m in messages)
