import os,pytest, vertexai
pytest_plugins = ('pytest_asyncio',)

@pytest.mark.asyncio(loop_scope="function")
async def test_hospital_review_chain():
    query = """What have patients said about hospital efficiency?
            Mention details from specific reviews."""
    from src.Healthcare.HospitalReviewChain import reviews_vector_chain
    response = await reviews_vector_chain.ainvoke(query)
    """
    response: {'query': 'What have patients said about hospital efficiency?\n Mention details from specific reviews.', 
               'result': 'Patients haven\'t directly mentioned hospital efficiency in their reviews. However, some comments can be indirectly related to efficiency. 
                         For example, Gary Cook mentioned that the doctors at Jordan Inc "seemed rushed during consultations", which could imply that the hospital\'s scheduling or workflow might be inefficient, 
                         leading to doctors having limited time with patients.\n\nOn the other hand, other reviews have focused on aspects like facilities, staff attitude, medical care quality, and amenities, without explicitly discussing efficiency. 
                         Therefore, it\'s difficult to draw a conclusion about hospital efficiency based on these reviews alone.'}    
    """
    result = response.get('result')
    print(f"test_hospital_review_chain: response: {response}, type: {type(response)}, result: {result}")
    assert result
    assert "hospital" in result
    assert "patient" in result
    assert "lack of clear communication" in result
    assert "inefficiencies" in result
    assert "difficulties" in result
