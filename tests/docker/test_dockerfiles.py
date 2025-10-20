import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def read_dockerfile(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def extract_copy_sources(contents: str):
    for line in contents.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = re.match(r"^COPY\s+([^\s]+)\s+(.*)$", stripped, re.IGNORECASE)
        if match:
            sources = match.group(1).split()
            for src in sources:
                yield src.rstrip("/")


def test_production_dockerfile_sets_app_env():
    contents = read_dockerfile("Dockerfile")
    assert "ENV APP_ENV=prod" in contents


def test_development_dockerfile_sets_app_env():
    contents = read_dockerfile("Dockerfile.dev")
    assert "ENV APP_ENV=dev" in contents


def test_dockerfiles_define_model_version():
    prod = read_dockerfile("Dockerfile")
    dev = read_dockerfile("Dockerfile.dev")
    assert "ENV MODEL_VERSION=" in prod
    assert "ENV MODEL_VERSION=" in dev


def test_production_copy_sources_exist():
    contents = read_dockerfile("Dockerfile")
    missing = [
        src for src in extract_copy_sources(contents)
        if not (REPO_ROOT / src).exists()
    ]
    assert not missing, f"Dockerfile COPY sources missing from repo: {missing}"


def test_development_copy_sources_exist():
    contents = read_dockerfile("Dockerfile.dev")
    missing = [
        src for src in extract_copy_sources(contents)
        if not (REPO_ROOT / src).exists()
    ]
    assert not missing, f"Dockerfile.dev COPY sources missing from repo: {missing}"


def test_requirements_cover_web_stack():
    requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "fastapi" in requirements.lower()
    assert "uvicorn" in requirements.lower()


def test_uvicorn_entrypoint_module_exists():
    contents = read_dockerfile("Dockerfile")
    cmd_line = next(
        (line for line in contents.splitlines() if line.strip().startswith("CMD")), ""
    )
    assert "app.main:app" in cmd_line
    assert (REPO_ROOT / "app" / "main.py").exists()


def test_models_directory_tracked_even_without_artifacts():
    models_dir = REPO_ROOT / "models"
    assert models_dir.exists() and models_dir.is_dir()


def test_dev_dockerfile_does_not_copy_app_code():
    contents = read_dockerfile("Dockerfile.dev")
    copy_lines = [
        line for line in contents.splitlines() if line.strip().upper().startswith("COPY")
    ]
    assert all("app/" not in line.split() for line in copy_lines), "Dev Dockerfile should rely on bind mount for app/ code"


def test_dockerignore_allows_app_and_src():
    dockerignore = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    assert "app/" not in dockerignore, "app/ must be included in build context"
    assert "src/" not in dockerignore, "src/ must be included in build context"
