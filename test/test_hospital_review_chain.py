import pytest
from uuid_extensions import uuid7, uuid7str
from langchain_core.runnables import RunnableConfig
from src.Healthcare.Tools import HealthcareReview
pytest_plugins = ('pytest_asyncio',)

@pytest.mark.asyncio(loop_scope="function")
async def test_hospital_review_chain():
    config = RunnableConfig(run_name=test_hospital_review_chain.__name__, thread_id=uuid7str())
    query = """What have patients said about hospital efficiency?
            Mention details from specific reviews."""
    response = await HealthcareReview.ainvoke(query, config)
    """
    response: {'query': 'What have patients said about hospital efficiency?\n Mention details from specific reviews.', 
               'result': 'Patients haven\'t directly mentioned hospital efficiency in their reviews. However, some comments can be indirectly related to efficiency. 
                         For example, Gary Cook mentioned that the doctors at Jordan Inc "seemed rushed during consultations", which could imply that the hospital\'s scheduling or workflow might be inefficient, 
                         leading to doctors having limited time with patients.\n\nOn the other hand, other reviews have focused on aspects like facilities, staff attitude, medical care quality, and amenities, without explicitly discussing efficiency. 
                         Therefore, it\'s difficult to draw a conclusion about hospital efficiency based on these reviews alone.'}    
    Patients haven\'t directly mentioned hospital efficiency in their reviews. However, one patient, Gary Cook, mentioned that the doctors at Jordan Inc "seemed rushed during consultations", which could be related to hospital efficiency. This suggests that the hospital may have a high volume of patients or limited staff, leading to doctors feeling rushed. But it\'s not a direct comment on the overall efficiency of the hospital. \n\n
    No other reviews mention anything related to hospital efficiency, such as wait times, admission processes, or discharge procedures.                         
    """
    result = response.get('result')
    print(f"{test_hospital_review_chain.__name__}: response: {response}, type: {type(response)}, result: {result}")
    assert result
    assert "Gary Cook" in result
    assert "Jordan Inc" in result
    assert "seemed rushed during consultations" in result
