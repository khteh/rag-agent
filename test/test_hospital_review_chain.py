import os,pytest, vertexai
pytest_plugins = ('pytest_asyncio',)

@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.skip(reason="https://github.com/langchain-ai/langchain/issues/30687 https://github.com/neo4j/neo4j/issues/13636")
async def test_hospital_review_chain():
    query = """What have patients said about hospital efficiency?
            Mention details from specific reviews."""
    from Healthcare.HospitalReviewChain import reviews_vector_chain
    response = await reviews_vector_chain.ainvoke(query)
    result = response.get('result')
    assert result
    assert "hospital" in result
    assert "patient" in result
    assert "lack of clear communication" in result
    assert "inefficiencies" in result
    assert "difficulties" in result
