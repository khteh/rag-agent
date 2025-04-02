import os,pytest, vertexai, re
from dotenv import load_dotenv
pytest_plugins = ('pytest_asyncio',)
load_dotenv()

@pytest.mark.asyncio
@pytest.mark.skip(reason="https://github.com/langchain-ai/langchain/issues/30547")
async def test_hospital_visit_duration():
    #vertexai.init(project=os.environ.get("GOOGLE_CLOUD_PROJECT"), location=os.environ.get("GOOGLE_CLOUD_LOCATION"))
    query = "What is the average visit duration for emergency visits in North Carolina?"
    from Healthcare.HospitalCypherChain import hospital_cypher_chain
    response = await hospital_cypher_chain.ainvoke(query)
    result = response.get('result')
    assert result
    assert "The average visit duration for emergency visits in North Carolina is" in result
    duration = re.search(r"(\d+\.\d+)", result)
    assert duration
    assert len(duration[0])

@pytest.mark.asyncio
@pytest.mark.skip(reason="https://github.com/langchain-ai/langchain/issues/30547")
async def test_hospital_percent_increase():
    #vertexai.init(project=os.environ.get("GOOGLE_CLOUD_PROJECT"), location=os.environ.get("GOOGLE_CLOUD_LOCATION"))
    query = "Which state had the largest percent increase in Medicaid visits from 2022 to 2023?"
    from Healthcare.HospitalCypherChain import hospital_cypher_chain
    response = await hospital_cypher_chain.ainvoke(query)
    result = response.get('result')
    assert result
    assert "The state with the largest percent increase in Medicaid visits from 2022 to 2023 is" in result
    assert "with a percent increase of" in result
    percent = re.search(r"(\d+\.\d+)%", result)
    assert percent
    assert len(percent[0])
