from evaluator.llms.vertexai_llm import VertexAILLM

llm = VertexAILLM(model_name="gemini-1.5-flash", project_id="testing-ragas", location="europe-west2")
response = llm.generate("Write a short story about a cat.")
print(response)