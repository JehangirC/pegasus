from evaluator.llms.vertexai_llm import VertexAILLM

llm = VertexAILLM(model_name="gemini-2.0-flash-001", project_id="testing-ragas", location="europe-west1")
response = llm.generate("Write a short story about a cat.")
print(response)