import os,pytest, vertexai, re
pytest_plugins = ('pytest_asyncio',)

@pytest.mark.asyncio(loop_scope="function")
#@pytest.mark.skip(reason="https://github.com/langchain-ai/langchain/issues/30547")
async def test_hospital_visit_duration():
    query = "What is the average visit duration for emergency visits in North Carolina?"
    from src.Healthcare.HospitalCypherChain import hospital_cypher_chain
    response = await hospital_cypher_chain.ainvoke(query)
    result = response.get('result')
    assert result
    assert "The average visit duration for emergency visits in North Carolina is" in result
    duration = re.search(r"(\d+\.\d+)", result)
    assert duration
    assert len(duration[0])

@pytest.mark.asyncio(loop_scope="function")
#@pytest.mark.skip(reason="https://github.com/langchain-ai/langchain/issues/30547")
async def test_hospital_percent_increase():
    query = "Which state had the largest percent increase in Medicaid visits from 2022 to 2023?"
    from src.Healthcare.HospitalCypherChain import hospital_cypher_chain
    response = await hospital_cypher_chain.ainvoke(query)
    result = response.get('result')
    assert result
    assert "The state with the largest percent increase in Medicaid visits from 2022 to 2023 is" in result
    assert "with a percent increase of" in result
    percent = re.search(r"(\d+\.\d+)%", result)
    assert percent
    assert len(percent[0])
