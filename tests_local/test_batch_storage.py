"""批次中断恢复：已完成项可安全重放，未发布目录不进入资料库。"""
import json
import pytest
from local_rag.batch_storage import update_reviewed_record, publish_document


def test_review_update_can_resume_after_first_record(tmp_path):
    record = tmp_path / 'record.json'
    record.write_text('{"status":"DATE_REVIEW_REQUIRED"}', encoding='utf-8')
    backup = tmp_path / 'before.json'
    before = record.read_bytes()
    new = dict(status='PARSED')
    update_reviewed_record(record, before, new, backup)
    update_reviewed_record(record, before, new, backup)
    assert backup.read_bytes() == before
    assert json.loads(record.read_text()) == new


def test_review_update_refuses_unexpected_current_record(tmp_path):
    record = tmp_path / 'record.json'
    record.write_text('{"status":"OTHER"}')
    with pytest.raises(ValueError):
        update_reviewed_record(record, b'{"status":"OLD"}', {'status': 'NEW'}, tmp_path / 'before.json')


def test_document_publish_is_repeatable_without_overwrite(tmp_path):
    target, stage = tmp_path / 'library/doc', tmp_path / 'batch/staging'
    files = {'source.html': b'raw', 'articles.json': b'[]', 'record.json': b'{}'}
    publish_document(target, stage, files)
    publish_document(target, stage, files)
    assert (target / 'source.html').read_bytes() == b'raw'
    with pytest.raises(ValueError):
        publish_document(target, stage, dict(files, **{'source.html': b'changed'}))


def test_unpublished_stage_can_resume(tmp_path):
    target, stage = tmp_path / 'library/doc', tmp_path / 'batch/staging'
    stage.mkdir(parents=True)
    (stage / 'source.html').write_bytes(b'raw')
    publish_document(target, stage, {'source.html': b'raw', 'record.json': b'{}'})
    assert (target / 'record.json').exists()
    assert not stage.exists()
