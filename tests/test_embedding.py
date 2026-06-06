"""Tests for embedding model and Qdrant operations.

These tests require Qdrant to be running. Mark with pytest marker.
"""

import pytest

COLLECTION_NAME = "test_internal_docs"


@pytest.fixture
def embedding_model():
    """Load the bi-encoder model."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("multi-qa-mpnet-base-cos-v1")


def test_embedding_dimensions(embedding_model):
    """Embedding produces 768-dim vector."""
    embedding = embedding_model.encode("How does OAuth work?")
    assert len(embedding) == 768


def test_embedding_is_float_list(embedding_model):
    """Embedding is a list of floats."""
    embedding = embedding_model.encode("Test text")
    assert all(isinstance(x, float) for x in embedding.tolist())


def test_similar_texts_closer(embedding_model):
    """Similar texts have higher cosine similarity."""
    from numpy import dot
    from numpy.linalg import norm

    query = embedding_model.encode("OAuth token refresh")
    relevant = embedding_model.encode(
        "To refresh an OAuth token, call POST /oauth/token with the refresh_token grant type."
    )
    unrelated = embedding_model.encode(
        "The deployment pipeline uses Docker containers on Kubernetes."
    )

    def cosine(a, b):
        return dot(a, b) / (norm(a) * norm(b))

    sim_relevant = cosine(query, relevant)
    sim_unrelated = cosine(query, unrelated)

    assert sim_relevant > sim_unrelated
    assert sim_relevant > 0.5
    assert sim_unrelated < 0.5


@pytest.mark.skipif(
    "not config.getoption('--run-qdrant-tests')",
    reason="Qdrant tests require running Qdrant instance",
)
class TestQdrantIntegration:
    """Integration tests requiring a running Qdrant instance."""

    @pytest.fixture(autouse=True)
    async def setup_collection(self):
        from qdrant_client import AsyncQdrantClient, models

        client = AsyncQdrantClient(url="http://localhost:6333")
        try:
            await client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=768,
                    distance=models.Distance.COSINE,
                ),
            )
        except Exception:
            pass
        yield
        # Cleanup
        try:
            await client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        await client.close()

    async def test_upsert_and_search(self, embedding_model):
        """Points are upserted and searchable."""
        from qdrant_client import AsyncQdrantClient, models
        import uuid

        client = AsyncQdrantClient(url="http://localhost:6333")

        # Create a test point
        embedding = embedding_model.encode("OAuth token refresh endpoint").tolist()
        point = models.PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "url": "https://example.com/oauth",
                "chunk_index": 0,
                "content": "The OAuth token refresh endpoint...",
            },
        )
        await client.upsert(collection_name=COLLECTION_NAME, points=[point])

        # Search
        query_vector = embedding_model.encode("How do I refresh tokens?").tolist()
        results = await client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=1,
        )

        assert len(results.points) >= 1
        assert results.points[0].payload["url"] == "https://example.com/oauth"
        assert results.points[0].score is not None

        await client.close()
