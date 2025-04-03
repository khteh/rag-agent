import os,pytest, vertexai
from dotenv import load_dotenv
pytest_plugins = ('pytest_asyncio',)
load_dotenv()

@pytest.mark.asyncio(loop_scope="function")
#@pytest.mark.skip(reason="Needs billing")
async def test_hospital_review_chain():
    #vertexai.init(project=os.environ.get("GOOGLE_CLOUD_PROJECT"), location=os.environ.get("GOOGLE_CLOUD_LOCATION"))
    query = """What have patients said about hospital efficiency?
            Mention details from specific reviews."""
    from Healthcare.HospitalReviewChain import reviews_vector_chain
    response = await reviews_vector_chain.ainvoke(query)
    result = response.get('result')
    assert result
    assert "hospital" in result
    assert "patient" in result
    assert "communication" in result
    #assert "bureaucratic" in result
    assert "efficient" in result
