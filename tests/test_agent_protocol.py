from __future__ import annotations

from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def _resolve_doc_link(source_path: str, link_target: str) -> Path | None:
    if link_target.startswith(("http://", "https://", "mailto:")):
        return None
    if link_target.startswith("#"):
        return None

    target_without_anchor = link_target.split("#", 1)[0]
    if target_without_anchor == "":
        return None

    if target_without_anchor.startswith("/"):
        return Path(target_without_anchor)

    return (REPO_ROOT / source_path).parent.joinpath(target_without_anchor).resolve()


def test_agent_protocol_surfaces_exist() -> None:
    required_paths = (
        "AGENTS.md",
        "docs/agent/README.md",
        "docs/agent/playbooks/task-a-block3-change.md",
        "docs/agent/playbooks/doc-contract-sync.md",
        "docs/agent/playbooks/verification-and-review.md",
        "tasks/task_A/AGENTS.md",
    )

    for relative_path in required_paths:
        assert (REPO_ROOT / relative_path).exists(), relative_path


def test_repo_agents_protocol_contains_required_guidance() -> None:
    text = _read("AGENTS.md")

    assert "Explore -> Plan -> Execute -> Verify -> Sync docs/manifests" in text
    assert "Source-of-truth order" in text
    assert "Allowed commands" in text
    assert "Default verification commands" in text
    assert "large rerun" in text
    assert "external download" in text
    assert "destructive git" in text
    assert "frozen scientific contracts" in text
    assert "tasks/task_A/AGENTS.md" in text
    assert "docs/agent/README.md" in text


def test_task_a_agents_protocol_captures_authority_chain_and_resume_boundary() -> None:
    text = _read("tasks/task_A/AGENTS.md")

    assert "docs/task_A_spec.md" in text
    assert "docs/task_A_block3_redesign_v1_1.md" in text
    assert "tasks/task_A/README.md" in text
    assert "block3_execution_runbook.md" in text
    assert "prepare" in text
    assert "run_block0" in text
    assert "run_block1" in text
    assert "run_block2" in text
    assert "run_block3" in text
    assert "package_results" in text
    assert "--resume" in text
    assert "Block 3 has no checkpoint/resume path" in text
    assert "packet-local" in text


def test_playbooks_use_standard_sections() -> None:
    required_markers = (
        "## Trigger",
        "## Read First",
        "## Allowed Commands",
        "## Required Verification",
        "## Required Updates",
        "## Stop And Ask The User",
    )
    content_paths = (
        "docs/agent/playbooks/task-a-block3-change.md",
        "docs/agent/playbooks/doc-contract-sync.md",
        "docs/agent/playbooks/verification-and-review.md",
    )

    for relative_path in content_paths:
        text = _read(relative_path)
        for marker in required_markers:
            assert marker in text, f"{marker} missing from {relative_path}"


def test_agent_entrypoints_link_to_protocol_docs() -> None:
    readme_text = _read("README.md")
    docs_index_text = _read("docs/index.md")
    task_a_readme_text = _read("tasks/task_A/README.md")
    playbook_index_text = _read("docs/agent/README.md")

    assert "Agent Collaboration" in readme_text
    assert "AGENTS.md" in readme_text

    assert "## Agent Collaboration" in docs_index_text
    assert "agent/README.md" in docs_index_text

    assert "Agent-first" in task_a_readme_text
    assert "tasks/task_A/AGENTS.md" in task_a_readme_text

    assert "task-a-block3-change.md" in playbook_index_text
    assert "doc-contract-sync.md" in playbook_index_text
    assert "verification-and-review.md" in playbook_index_text


def test_agent_protocol_links_resolve_locally() -> None:
    protocol_docs = (
        "AGENTS.md",
        "docs/agent/README.md",
        "docs/agent/playbooks/task-a-block3-change.md",
        "docs/agent/playbooks/doc-contract-sync.md",
        "docs/agent/playbooks/verification-and-review.md",
        "tasks/task_A/AGENTS.md",
        "README.md",
        "docs/index.md",
        "tasks/task_A/README.md",
    )

    for relative_path in protocol_docs:
        text = _read(relative_path)
        for link_target in MARKDOWN_LINK_RE.findall(text):
            resolved = _resolve_doc_link(relative_path, link_target)
            if resolved is None:
                continue
            assert resolved.exists(), f"{relative_path} -> {link_target}"
