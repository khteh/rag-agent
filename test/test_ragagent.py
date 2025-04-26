import os,pytest, re
from uuid_extensions import uuid7, uuid7str
from langchain_core.runnables import RunnableConfig
from src.models import ChatMessage
pytest_plugins = ('pytest_asyncio',)

@pytest.mark.asyncio(loop_scope="function")
async def test_ragagent_agent_blogs(RAGAgentFixture):
    config = RunnableConfig(run_name=test_ragagent_agent_blogs.__name__, thread_id=uuid7str())
    lc_ai_message = await RAGAgentFixture.ChatAgent(config, ("What is task decomposition?\n" "What is the standard method for Task Decomposition?\n" "Once you get the answer, look up common extensions of that method."))
    assert lc_ai_message
    ai_message = ChatMessage.from_langchain(lc_ai_message)
    """
    Task decomposition is the process of breaking down complex tasks or projects into smaller, more manageable subtasks. This technique is used to simplify work, improve efficiency, and increase productivity.

    The standard method for task decomposition is often referred to as the "Divide and Conquer" approach. However, a more structured method is the SMART (Specific, Measurable, Achievable, Relevant, Time-bound) criteria, which helps ensure that each subtask is well-defined and achievable.

    To further decompose tasks, common extensions of this method include:

    1. Mind Mapping: A visual technique used to create a diagram of related ideas, concepts, and tasks.
    2. Gantt Charts: A type of bar chart that illustrates a project schedule, helping to break down tasks into smaller components and track progress over time.
    3. Work Breakdown Structure (WBS): A hierarchical decomposition of the work to be done, often used in project management to organize and structure tasks.
    4. Pomodoro Technique: A time management method that involves breaking down work into short, focused intervals (typically 25 minutes) called "Pomodoros," separated by brief breaks.

    These extensions can help individuals and teams effectively decompose tasks, prioritize their work, and manage their time more efficiently.
    test_ragagent_agent_blogs: Task decomposition is the process of breaking down complex tasks or projects into smaller, more manageable subtasks. This technique is used to simplify work, improve efficiency, and increase productivity.

    The standard method for task decomposition is often referred to as the "Divide and Conquer" approach. However, a more structured method is the SMART (Specific, Measurable, Achievable, Relevant, Time-bound) criteria, which helps ensure that each subtask is well-defined and achievable.

    To further decompose tasks, common extensions of this method include:

    1. Mind Mapping: A visual technique used to create a diagram of related ideas, concepts, and tasks.
    2. Gantt Charts: A type of bar chart that illustrates a project schedule, helping to break down tasks into smaller components and track progress over time.
    3. Work Breakdown Structure (WBS): A hierarchical decomposition of the work to be done, often used in project management to organize and structure tasks.
    4. Pomodoro Technique: A time management method that involves breaking down work into short, focused intervals (typically 25 minutes) called "Pomodoros," separated by brief breaks.

    These extensions can help individuals and teams effectively decompose tasks, prioritize their work, and manage their time more efficiently.
    """
    assert not ai_message.tool_calls
    assert ai_message.content
    #print(f"test_ragagent_agent_blogs: {ai_message.content}")
    assert "breaking down complex tasks" in ai_message.content
    assert "smaller, more manageable sub" in ai_message.content

@pytest.mark.asyncio(loop_scope="function")
async def test_ragagent_mlflow_blogs(RAGAgentFixture):
    config = RunnableConfig(run_name=test_ragagent_mlflow_blogs.__name__, thread_id=uuid7str())
    lc_ai_message = await RAGAgentFixture.ChatAgent(config, ("What is MLFlow?\n""Where can you run it?"))
    assert lc_ai_message
    ai_message = ChatMessage.from_langchain(lc_ai_message)
    #print(f"ai_message: {ai_message}")
    """
    MLFlow is an open-source platform for managing the end-to-end machine learning lifecycle. It provides a comprehensive framework for data scientists and engineers to develop, deploy, and manage machine learning models. MLFlow allows users to track experiments, manage models, and deploy them in various environments.

    MLFlow can be run in several environments, including:

    1. Local Machine: MLFlow can be installed on a local machine using pip, the Python package manager.
    2. Cloud Platforms: MLFlow is supported on major cloud platforms such as Amazon Web Services (AWS), Microsoft Azure, Google Cloud Platform (GCP), and IBM Cloud.
    3. Containerization: MLFlow can be run in containerized environments like Docker, allowing for easy deployment and management of machine learning models.
    4. Kubernetes: MLFlow supports deployment on Kubernetes, a popular container orchestration platform.
    5. Hadoop and Spark: MLFlow integrates with Apache Hadoop and Apache Spark, enabling users to manage machine learning workflows on big data platforms.

    Overall, MLFlow provides a flexible and scalable solution for managing machine learning workflows across various environments.
    test_ragagent_mlflow_blogs: MLFlow is an open-source platform for managing the end-to-end machine learning lifecycle. It provides a comprehensive framework for data scientists and engineers to develop, deploy, and manage machine learning models. MLFlow allows users to track experiments, manage models, and deploy them in various environments.

    MLFlow can be run in several environments, including:

    1. Local Machine: MLFlow can be installed on a local machine using pip, the Python package manager.
    2. Cloud Platforms: MLFlow is supported on major cloud platforms such as Amazon Web Services (AWS), Microsoft Azure, Google Cloud Platform (GCP), and IBM Cloud.
    3. Containerization: MLFlow can be run in containerized environments like Docker, allowing for easy deployment and management of machine learning models.
    4. Kubernetes: MLFlow supports deployment on Kubernetes, a popular container orchestration platform.
    5. Hadoop and Spark: MLFlow integrates with Apache Hadoop and Apache Spark, enabling users to manage machine learning workflows on big data platforms.

    Overall, MLFlow provides a flexible and scalable solution for managing machine learning workflows across various environments.
    """
    assert not ai_message.tool_calls
    assert ai_message.content
    #print(f"test_ragagent_mlflow_blogs: {ai_message.content}")
    assert "machine learning" in ai_message.content
    assert "end-to-end machine learning lifecycle" in ai_message.content
    assert "local machine" in ai_message.content.lower()
    assert "cloud Platforms" in ai_message.content.lower()
    assert "container" in ai_message.content.lower()
    assert "kubernetes" in ai_message.content.lower()
