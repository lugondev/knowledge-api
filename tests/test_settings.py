from kbase.settings import Settings


def test_api_keys_parse_into_key_to_tenant():
    s = Settings.from_env({"KB_API_KEYS": "aaa:acme, bbb:globex"})
    assert s.api_keys == {"aaa": "acme", "bbb": "globex"}


def test_unset_api_keys_is_empty_not_a_wildcard():
    # An empty map is what makes every request a 401. It must never
    # degrade into "no keys configured means anyone may call".
    s = Settings.from_env({})
    assert s.api_keys == {}


def test_malformed_key_entry_is_ignored_not_guessed():
    s = Settings.from_env({"KB_API_KEYS": "nocolon, ccc:initech"})
    assert s.api_keys == {"ccc": "initech"}


def test_defaults():
    s = Settings.from_env({})
    assert s.database_url.startswith("sqlite+aiosqlite://")
    assert s.max_upload_bytes == 20_000_000
    assert s.docs_enabled is True


def test_docs_disabled_by_false_string():
    assert Settings.from_env({"KB_DOCS": "false"}).docs_enabled is False


def test_check_names_every_missing_requirement():
    problems = Settings.from_env({}).check()
    joined = " ".join(problems)
    assert "KB_API_KEYS" in joined
    assert "KB_EMBED_BASE_URL" in joined
    assert "KB_EMBED_MODEL" in joined


def test_check_passes_on_a_complete_environment():
    s = Settings.from_env(
        {
            "KB_API_KEYS": "aaa:acme",
            "KB_EMBED_BASE_URL": "http://localhost:1234/v1",
            "KB_EMBED_MODEL": "text-embedding-3-small",
        }
    )
    assert s.check() == []


def test_embed_dim_defaults_to_zero_when_unset():
    s = Settings.from_env({})
    assert s.embed_dim == 0


def test_a_non_numeric_embed_dim_is_a_problem_not_a_guess():
    s = Settings.from_env({"KB_EMBED_DIM": "large"})
    assert any("KB_EMBED_DIM" in p for p in s.check())


def test_postgres_without_a_dimension_warns_that_it_will_scan():
    s = Settings.from_env(
        {
            "KB_API_KEYS": "k:acme",
            "KB_DATABASE_URL": "postgresql+asyncpg://u:p@h/db",
            "KB_EMBED_BASE_URL": "http://e/v1",
            "KB_EMBED_MODEL": "m",
        }
    )
    assert s.check() == []
    assert any("scan" in w for w in s.warnings())


def test_a_dimension_over_the_hnsw_ceiling_warns_but_starts():
    s = Settings.from_env(
        {
            "KB_API_KEYS": "k:acme",
            "KB_DATABASE_URL": "postgresql+asyncpg://u:p@h/db",
            "KB_EMBED_BASE_URL": "http://e/v1",
            "KB_EMBED_MODEL": "m",
            "KB_EMBED_DIM": "3072",
        }
    )
    assert s.check() == []
    assert any("2000" in w for w in s.warnings())
