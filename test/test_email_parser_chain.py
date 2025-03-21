import pytest, json
from datetime import datetime
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage, ToolCall
from schema import EmailModel
from rag_agent import EmailRAG
from data.sample_emails import EMAILS

@pytest.mark.asyncio
async def test_email_parser_chain(rag):
    email = await rag.ParseEmail(EMAILS[0])
    assert email
    assert email.entity_name == 'Occupational Safety and Health Administration (OSHA)'
    