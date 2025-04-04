import os,pytest, re
from dotenv import load_dotenv
pytest_plugins = ('pytest_asyncio',)
load_dotenv()

@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.skip(reason="Needs billing; https://github.com/langchain-ai/langchain/issues/30547")
async def test_hospital_waiting_time(HealthcareRAGFixture):
    response = await HealthcareRAGFixture.ainvoke({"input": "What is the wait time at Wallace-Hamilton?"})
    result = response.get('output')
    assert result
    assert "The current wait time at Wallace-Hamilton is " in result
    waitingtime = re.search(r"(\d+\.*\d+)", result)
    assert waitingtime
    assert len(waitingtime[0])

@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.skip(reason="Needs billing; https://github.com/langchain-ai/langchain/issues/30547")
async def test_hospital_shortest_wait_time(HealthcareRAGFixture):
    response = await HealthcareRAGFixture.ainvoke({"input": "Which hospital has the shortest wait time?"})
    result = response.get('output')
    assert result
    assert "The hospital with the shortest wait time is " in result
    waitingtime = re.search(r"(\d+\.*\d+)", result)
    assert waitingtime
    assert len(waitingtime[0])

@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.skip(reason="Needs billing; https://github.com/langchain-ai/langchain/issues/30547")
async def test_hospital_patient_reviews(HealthcareRAGFixture):
    response = await HealthcareRAGFixture.ainvoke({"input": "What have patients said about their quality of rest during their stay?"})
    result = response.get('output')
    assert result
    assert "Patients have mentioned" in result
    assert "constant interruptions for routine checks" in result
    assert "noise level at night" in result
    assert "a good night's sleep during their stay" in result
    assert "quality of rest" in result

@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.skip(reason="Needs billing; https://github.com/langchain-ai/langchain/issues/30547")
async def test_hospital_cypher_query(HealthcareRAGFixture):
    response = await HealthcareRAGFixture.ainvoke({"input": "Which physician has treated the most patients covered by Cigna?"})
    result = response.get('output')
    assert result
    assert "The physician who has treated the most patients covered by Cigna is " in result
    assert "has treated a total of " in result
    patients = re.search(r"(\d+)", result)
    assert patients
    assert len(patients[0])

@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.skip(reason="Needs billing; https://github.com/langchain-ai/langchain/issues/30547")
async def test_hospital_specific_patient_review(HealthcareRAGFixture):
    response = await HealthcareRAGFixture.ainvoke({"input": "Query the graph database to show me the reviews written by patient 7674"})
    result = response.get('output')
    assert result
    assert "Patient 7674 wrote the following review: " in result
