import os,pytest, re
from uuid_extensions import uuid7, uuid7str
from langchain_core.runnables import RunnableConfig
from src.models import ChatMessage
pytest_plugins = ('pytest_asyncio',)

@pytest.mark.asyncio(loop_scope="function")
async def test_hospital_waiting_time(HealthcareRAGFixture):
    config = RunnableConfig(run_name=test_hospital_waiting_time.__name__, thread_id=uuid7str())
    lc_ai_message = await HealthcareRAGFixture.ChatAgent(config, [("human", "What is the wait time at Wallace-Hamilton?")])
    assert lc_ai_message
    ai_message = ChatMessage.from_langchain(lc_ai_message)
    #print(f"ai_message: {ai_message}")
    assert not ai_message.tool_calls
    assert ai_message.content
    print(f"test_hospital_waiting_time: {ai_message.content}")
    assert ai_message.content 
    assert "The current wait time at Wallace-Hamilton is " in ai_message.content
    regex = r"\b(\d+)(\s)+(hours?|minutes?)+"
    waitingtime = re.search(regex, ai_message.content).groups()
    assert waitingtime
    assert len(waitingtime[0])

@pytest.mark.asyncio(loop_scope="function")
async def test_hospital_shortest_wait_time(HealthcareRAGFixture):
    config = RunnableConfig(run_name=test_hospital_shortest_wait_time.__name__, thread_id=uuid7str())
    lc_ai_message = await HealthcareRAGFixture.ChatAgent(config, [("human", "Which hospital has the shortest wait time?")])
    assert lc_ai_message
    ai_message = ChatMessage.from_langchain(lc_ai_message)
    #print(f"ai_message: {ai_message}")
    assert not ai_message.tool_calls
    assert ai_message.content
    print(f"test_hospital_shortest_wait_time: {ai_message.content}")
    assert ai_message.content
    assert " wait time " in ai_message.content
    regex = r"\b(\d+)(\s)+(hours?|minutes?)+"
    waitingtime = re.search(regex, ai_message.content).groups()
    assert waitingtime
    assert len(waitingtime[0])

@pytest.mark.asyncio(loop_scope="function")
async def test_hospital_patient_reviews(HealthcareRAGFixture):
    config = RunnableConfig(run_name=test_hospital_patient_reviews.__name__, thread_id=uuid7str())
    lc_ai_message = await HealthcareRAGFixture.ChatAgent(config, [("human", "What have patients said about their quality of rest during their stay?")])
    assert lc_ai_message
    ai_message = ChatMessage.from_langchain(lc_ai_message)
    #print(f"ai_message: {ai_message}")
    assert not ai_message.tool_calls
    assert ai_message.content
    """
    ================================== Ai Message ==================================
    Name: Healthcare ReAct Agent

    According to patient reviews, several patients have mentioned that they had difficulty getting a good rest during their stay at the hospital. Some common issues included noise levels in shared rooms and uncomfortable beds. For example, 
    Tyler Sanders DVM from Pugh-Rogers hospital stated that the noise level in the shared rooms made it hard for him to rest and recover. Ryan Espinoza from Rose Inc hospital also mentioned that the noisy neighbors in the shared room affected his ability to rest. 
    Caleb Coleman from Richardson-Powell hospital said that the noise level in the ward was disruptive, affecting his ability to rest. Tammy Benson DVM from Garcia Ltd hospital had a different issue, stating that the uncomfortable beds made it difficult for her to get a good night's sleep. 
    Overall, patients have expressed that their quality of rest was compromised due to various factors such as noise levels and uncomfortable beds during their stay at the hospital.
    test_hospital_patient_reviews: According to patient reviews, several patients have mentioned that they had difficulty getting a good rest during their stay at the hospital. Some common issues included noise levels in shared rooms and uncomfortable beds. 
    For example, Tyler Sanders DVM from Pugh-Rogers hospital stated that the noise level in the shared rooms made it hard for him to rest and recover. Ryan Espinoza from Rose Inc hospital also mentioned that the noisy neighbors in the shared room affected his ability to rest. 
    Caleb Coleman from Richardson-Powell hospital said that the noise level in the ward was disruptive, affecting his ability to rest. Tammy Benson DVM from Garcia Ltd hospital had a different issue, stating that the uncomfortable beds made it difficult for her to get a good night's sleep. 
    Overall, patients have expressed that their quality of rest was compromised due to various factors such as noise levels and uncomfortable beds during their stay at the hospital.
    """
    print(f"test_hospital_patient_reviews: {ai_message.content}")
    assert ai_message.content
    assert "noise level" in ai_message.content
    assert "had difficulty getting a good rest" in ai_message.content
    assert "hard for him to rest and recover" in ai_message.content
    assert "uncomfortable beds" in ai_message.content
    assert "a good night's sleep" in ai_message.content

@pytest.mark.asyncio(loop_scope="function")
async def test_hospital_cypher_query(HealthcareRAGFixture):
    config = RunnableConfig(run_name=test_hospital_cypher_query.__name__, thread_id=uuid7str())
    lc_ai_message = await HealthcareRAGFixture.ChatAgent(config, [("human","Which physician has treated the most patients covered by Cigna?")])
    assert lc_ai_message
    ai_message = ChatMessage.from_langchain(lc_ai_message)
    #print(f"ai_message: {ai_message}")
    assert not ai_message.tool_calls
    assert ai_message.content
    """
    ================================== Ai Message ==================================
    Name: Healthcare ReAct Agent

    According to our records, Kayla Lawson has treated the most patients covered by Cigna, with a total of 10 patients under her care.
    test_hospital_cypher_query: According to our records, Kayla Lawson has treated the most patients covered by Cigna, with a total of 10 patients under her care.
    """
    print(f"test_hospital_cypher_query: {ai_message.content}")
    assert ai_message.content
    assert "has treated the most patients covered by Cigna" in ai_message.content
    assert "a total of " in ai_message.content
    patients = re.search(r"(\d+)", ai_message.content)
    assert patients
    assert len(patients[0])

@pytest.mark.asyncio(loop_scope="function")
async def test_hospital_specific_patient_review(HealthcareRAGFixture):
    config = RunnableConfig(run_name=test_hospital_specific_patient_review.__name__, thread_id=uuid7str())
    lc_ai_message = await HealthcareRAGFixture.ChatAgent(config, [("human","Query the graph database to show me the reviews written by patient 7674")])
    assert lc_ai_message
    ai_message = ChatMessage.from_langchain(lc_ai_message)
    #print(f"ai_message: {ai_message}")
    assert not ai_message.tool_calls
    assert ai_message.content
    """
    ================================== Ai Message ==================================
    Name: Healthcare ReAct Agent

    Based on the query of the graph database, Patient 7674 wrote a review stating that they received exceptional care from Dr. Sandra Porter at Jones, Brown and Murray hospital, but had a negative experience with the billing process, finding it confusing and frustrating, and wished for clearer communication about costs.
    test_hospital_specific_patient_review: Based on the query of the graph database, Patient 7674 wrote a review stating that they received exceptional care from Dr. Sandra Porter at Jones, Brown and Murray hospital, but had a negative experience with the billing process, finding it confusing and frustrating, and wished for clearer communication about costs.
    """
    print(f"test_hospital_specific_patient_review: {ai_message.content}")
    assert ai_message.content
    assert "Patient 7674 wrote " in ai_message.content
    assert "confusing and frustrating" in ai_message.content
    assert "clearer communication" in ai_message.content.lower()
    assert "Jones, Brown" in ai_message.content
    assert "Sandra Porter" in ai_message.content