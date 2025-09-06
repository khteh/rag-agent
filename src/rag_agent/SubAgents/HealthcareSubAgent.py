from src.Healthcare.Tools import HealthcareReview, HealthcareCypher
healthcare_sub_agent_prompt = """You are a helpful healthcare AI assistant."""
healthcare_sub_agent = {
    "name": "healthcare-agent",
    "description": "Used to extract useful information from email given as input and determine if the email needs escalation based on the escalation criteria provided as input.",
    "prompt": healthcare_sub_agent_prompt,
    "tools": [HealthcareReview, HealthcareCypher],
}