import os,pytest, vertexai
pytest_plugins = ('pytest_asyncio',)

@pytest.mark.asyncio(loop_scope="function")
async def test_hospital_review_chain():
    query = """What have patients said about hospital efficiency?
            Mention details from specific reviews."""
    from src.Healthcare.HospitalReviewChain import reviews_vector_chain
    response = await reviews_vector_chain.ainvoke(query)
    result = response.get('result')
    print(f"test_hospital_review_chain: {result}")
    assert result
    assert "hospital" in result
    assert "patient" in result
    assert "lack of clear communication" in result
    assert "inefficiencies" in result
    assert "difficulties" in result
