import pytest, re
from uuid_extensions import uuid7, uuid7str
from langchain_core.runnables import RunnableConfig
from src.Healthcare.Tools import HealthcareCypher
pytest_plugins = ('pytest_asyncio',)

@pytest.mark.asyncio(loop_scope="function")
async def test_hospital_visit_duration():
    config = RunnableConfig(run_name=test_hospital_visit_duration.__name__, thread_id=uuid7str())
    query = "What is the average visit duration for emergency visits in North Carolina?"
    response = await HealthcareCypher.ainvoke(query, config)
    result = response.get('result')
    assert result
    assert "The average visit duration for emergency visits in North Carolina is" in result
    duration = re.search(r"(\d+\.\d+)", result)
    assert duration
    assert len(duration[0])

@pytest.mark.asyncio(loop_scope="function")
async def test_hospital_percent_increase():
    config = RunnableConfig(run_name=test_hospital_percent_increase.__name__, thread_id=uuid7str())
    query = "Which state had the largest percent increase in Medicaid visits from 2022 to 2023?"
    response = await HealthcareCypher.ainvoke(query, config)
    result = response.get('result')
    print(f"test_hospital_percent_increase: {result}")
    assert result
    #assert "The state with the largest percent increase in Medicaid visits from 2022 to 2023 is" in result
    #assert "with a percent increase of" in result
    #percent = re.search(r"(\d+\.\d+)%", result)
    #assert percent
    #assert len(percent[0])
