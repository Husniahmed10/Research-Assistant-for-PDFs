"""NeMo initialization for input/output-only validation, without dialogue search."""

from nemoguardrails.embeddings.index import EmbeddingsIndex


class GuardrailsOnlyIndex(EmbeddingsIndex):
    """Dialogue/intent retrieval is disabled; no extra embedding model is needed.

    NeMo initializes its dialogue indexes even for rails-only calls. These nodes
    use explicit custom actions and static refusal messages, never intent search.
    """

    async def add_items(self, items):
        pass

    async def search(self, text, max_results):
        raise RuntimeError("Dialogue search is disabled for the guardrail-only runtime")


def init(app):
    app.register_embedding_search_provider("guardrails_only", GuardrailsOnlyIndex)
