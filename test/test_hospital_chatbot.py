import os,pytest, re
from uuid_extensions import uuid7, uuid7str
from langchain_core.runnables import RunnableConfig
from src.models import ChatMessage
pytest_plugins = ('pytest_asyncio',)

@pytest.mark.asyncio(loop_scope="function")
async def test_hospital_waiting_time(HealthcareRAGFixture):
    config = RunnableConfig(run_name=test_hospital_waiting_time.__name__, thread_id=uuid7str())
    lc_ai_message = await HealthcareRAGFixture.ChatAgent(config, "What is the wait time at Wallace-Hamilton?")
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
    lc_ai_message = await HealthcareRAGFixture.ChatAgent(config, "Which hospital has the shortest wait time?")
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
    lc_ai_message = await HealthcareRAGFixture.ChatAgent(config, "What have patients said about their quality of rest during their stay?")
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
    ---
    Patients from four different hospitals (Pugh-Rogers, Rose Inc, Garcia Ltd, and Richardson-Powell) have reported difficulties with getting a good quality of rest during their stay. Specifically, three patients (Tyler Sanders DVM at Pugh-Rogers, Ryan Espinoza at Rose Inc, and Caleb Coleman at Richardson-Powell) mentioned that the noise level in their shared rooms or wards was disruptive and affected their ability to rest. Another patient, Tammy Benson DVM at Garcia Ltd, reported that the uncomfortable beds made it difficult for her to get a good night's sleep. All of these patients emphasized the importance of rest and recovery during their hospital stay, suggesting that hospitals should consider ways to minimize noise levels and improve the comfort of their beds to better support patient care.
    test_hospital_patient_reviews: Patients from four different hospitals (Pugh-Rogers, Rose Inc, Garcia Ltd, and Richardson-Powell) have reported difficulties with getting a good quality of rest during their stay. Specifically, three patients (Tyler Sanders DVM at Pugh-Rogers, Ryan Espinoza at Rose Inc, and Caleb Coleman at Richardson-Powell) mentioned that the noise level in their shared rooms or wards was disruptive and affected their ability to rest. Another patient, Tammy Benson DVM at Garcia Ltd, reported that the uncomfortable beds made it difficult for her to get a good night's sleep. All of these patients emphasized the importance of rest and recovery during their hospital stay, suggesting that hospitals should consider ways to minimize noise levels and improve the comfort of their beds to better support patient care.
    """
    print(f"test_hospital_patient_reviews: {ai_message.content}")
    assert ai_message.content
    assert "noise levels" in ai_message.content
    assert "shared rooms" in ai_message.content
    assert "disruptive" in ai_message.content

@pytest.mark.asyncio(loop_scope="function")
async def test_hospital_cypher_query(HealthcareRAGFixture):
    config = RunnableConfig(run_name=test_hospital_cypher_query.__name__, thread_id=uuid7str())
    lc_ai_message = await HealthcareRAGFixture.ChatAgent(config, "Which physician has treated the most patients covered by Cigna?")
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
    lc_ai_message = await HealthcareRAGFixture.ChatAgent(config, "Query the graph database to show me the reviews written by patient 7674")
    assert lc_ai_message
    ai_message = ChatMessage.from_langchain(lc_ai_message)
    #print(f"ai_message: {ai_message}")
    assert not ai_message.tool_calls
    assert ai_message.content
    """
    ================================== Ai Message ==================================
    Name: Healthcare ReAct Agent

    Based on the query results, patient 7674 wrote a review stating that they received exceptional care from Dr. Sandra Porter at Jones, Brown and Murray hospital, but had a negative experience with the billing process, finding it confusing and frustrating, and wished for clearer communication about costs.
    test_hospital_specific_patient_review: Based on the query results, patient 7674 wrote a review stating that they received exceptional care from Dr. Sandra Porter at Jones, Brown and Murray hospital, but had a negative experience with the billing process, finding it confusing and frustrating, and wished for clearer communication about costs.
    ---    
    The reviews written by patient 7674 are:

    * "The hospital provided exceptional care, but they found the billing process to be confusing and frustrating, and wished for clearer communication about costs."

    This review suggests that while the patient was satisfied with the medical care they received, they had difficulties with the administrative aspects of their visit.
    test_hospital_specific_patient_review: The reviews written by patient 7674 are:

    * "The hospital provided exceptional care, but they found the billing process to be confusing and frustrating, and wished for clearer communication about costs."

    This review suggests that while the patient was satisfied with the medical care they received, they had difficulties with the administrative aspects of their visit.
    """
    print(f"test_hospital_specific_patient_review: {ai_message.content}")
    assert ai_message.content
    assert "exceptional care" in ai_message.content.lower()
    assert "billing process" in ai_message.content.lower()
    assert "confusing and frustrating" in ai_message.content
    assert "clearer communication" in ai_message.content.lower()

@pytest.mark.asyncio(loop_scope="function")
async def test_hospital_top_payer_query(HealthcareRAGFixture):
    config = RunnableConfig(run_name=test_hospital_top_payer_query.__name__, thread_id=uuid7str())
    lc_ai_message = await HealthcareRAGFixture.ChatAgent(config, "Which payer provides the most coverage in terms of total billing amount?")
    assert lc_ai_message
    ai_message = ChatMessage.from_langchain(lc_ai_message)
    assert not ai_message.tool_calls
    assert ai_message.content
    """
    ================================== Ai Message ==================================
    Name: Healthcare ReAct Agent

    The payer that provides the most coverage in terms of total billing amount is UnitedHealthcare, with a total billed amount of $52,221,646.74.
    """
    print(f"test_hospital_top_payer_query: {ai_message.content}")
    assert ai_message.content
    assert "UnitedHealthcare" in ai_message.content
    money_regex = re.compile('|'.join([
        r'\b\$?(\d*\.\d{1,2})',  # e.g., $.50, .50, $1.50, $.5, .5
        r'\b\$?(\d+)',           # e.g., $500, $5, 500, 5
        r'\b\$?(\d+\.?)',         # e.g., $5.
        r'\b\$?(\d{1,3},\d{3})*',           # e.g., $123,456, $1,234, $12,345
        r'\b\$?(\d{1,3},\d{3})*\.?',         # e.g., $123,456.5, $1,234, $12,345
        r'\b\$?(\d{1,3},\d{3})*\.\d{1,2}',         # e.g., $123,456.5, $1,234, $12,345
    ]))
    assert re.match(money_regex, ai_message.content)
