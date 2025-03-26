import os,pytest, re
from dotenv import load_dotenv
pytest_plugins = ('pytest_asyncio',)
load_dotenv()

@pytest.mark.asyncio
@pytest.mark.skip(reason="Needs billing")
async def test_hospital_waiting_time(HealthcareRAG):
    response = await HealthcareRAG.ainvoke({"input": "What is the wait time at Wallace-Hamilton?"})
    result = response.get('output')
    assert result
    assert "The current wait time at Wallace-Hamilton is " in result
    waitingtime = re.search(r"(\d+\.*\d+)", result)
    assert waitingtime
    assert len(waitingtime[0])

@pytest.mark.asyncio
@pytest.mark.skip(reason="Needs billing")
async def test_hospital_shortest_wait_time(HealthcareRAG):
    response = await HealthcareRAG.ainvoke({"input": "Which hospital has the shortest wait time?"})
    result = response.get('output')
    assert result
    assert "The hospital with the shortest wait time is " in result
    waitingtime = re.search(r"(\d+\.*\d+)", result)
    assert waitingtime
    assert len(waitingtime[0])
