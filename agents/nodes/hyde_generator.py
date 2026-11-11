"""
agents/nodes/hyde_generator.py — Hypothetical Document Embedding Generator (Node 5)
=====================================================================================

STEP-BY-STEP IMPLEMENTATION PLAN:
-----------------------------------
Step 1:  Import portkey_client, embedder service, and AgentState

Step 2:  Write hyde_generator_node(state: AgentState) → dict:

            --- What is HyDE? ---
            Instead of embedding the question ("What is the accuracy?"),
            we ask the LLM to write a FAKE ideal answer paragraph,
            then embed THAT. The fake answer looks more like actual paper text,
            so it retrieves better matches.

            --- Implementation ---
            - Extract state["rewritten_query"]
            - Build HyDE prompt:
                "Write a one-paragraph excerpt from a research paper that would
                 perfectly answer the following question. Write it as if it were
                 actual text from the paper's Results or Discussion section.
                 Question: {rewritten_query}"
            - Call portkey_client.chat_completion()
            - Store the generated text in state["hyde_document"]
            - Embed the hyde_document using embedder.embed_text()
            - Store embedding for use in retriever_node
            - Append "🧠 HyDE document generated" to state["steps"]
            - Return updated state dict

EXAMPLE:
    Query:  "What accuracy did the model achieve?"
    HyDE:   "The proposed ResNet-50 model achieved a top-1 accuracy of 94.3%
              on the ImageNet validation set, outperforming the baseline by 2.1%..."
"""

