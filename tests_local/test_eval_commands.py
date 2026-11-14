"""评测命令的验收退出语义：没有基线或模式不符时不能以退出0冒充通过。"""
import json

from scripts.eval_retrieval import choose_baseline, evaluation_exit_code


def test_full_acceptance_requires_baseline_but_baseline_recording_is_allowed():
    report = dict(results=[dict(mode="hybrid_qdrant", error=None)], source_changed=False, acceptance=None)
    comparison = dict(holdout_regressions=[])
    assert evaluation_exit_code(report, comparison, split="all", label="candidate") == 2
    assert evaluation_exit_code(report, comparison, split="all", label="baseline") == 0
    assert evaluation_exit_code(report, comparison, split="dev", label="candidate") == 0


def test_degraded_or_regressed_run_never_reports_success():
    report = dict(results=[dict(mode="lexical", error=None)], source_changed=False, acceptance=dict(passed=True))
    assert evaluation_exit_code(report, dict(holdout_regressions=[]), split="all", label="candidate") == 2
    report["results"][0]["mode"] = "hybrid_qdrant"
    assert evaluation_exit_code(report, dict(holdout_regressions=["H1"]), split="all", label="candidate") == 2


def test_auto_baseline_skips_dev_only_or_different_corpus(tmp_path):
    scope = dict(dataset_sha256="d", v2_sha256="v", corpus_id="c", embedding_id="e")
    full = dict(label="baseline", scope=scope, summary={s: {"count": n} for s, n in (("dev", 80), ("holdout", 40), ("v2", 36))},
                results=[dict(mode="hybrid_qdrant", error=None)] * 156)
    dev = {**full, "summary": {**full["summary"], "holdout": {"count": 0}}}
    other = {**full, "scope": {**scope, "corpus_id": "other"}}
    paths = []
    for n, report in enumerate((dev, other, full)):
        path = tmp_path / f"{n}.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        paths.append(path)
    assert choose_baseline(paths, {"scope": scope}) == paths[-1]
