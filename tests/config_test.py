from adv_rag import QualityConfig, MediumConfig, FastConfig


def test_quality_config():
    assert QualityConfig.chunk_size_tokens < MediumConfig.chunk_size_tokens
    assert "large" in QualityConfig.embedding_model_name


def test_medium_config():
    assert MediumConfig.chunk_size_tokens == 400
    assert "base" in MediumConfig.embedding_model_name


def test_fast_config():
    assert FastConfig.chunk_size_tokens >= 600
    assert "small" in FastConfig.embedding_model_name
