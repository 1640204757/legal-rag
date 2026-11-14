"""重放边界：没有成功真实网络证据、原始字节被改或模型输入不同都不能冒充复验。"""
import asyncio

import pytest

from scripts.replay_material_reading import ReplayModel, verify_real_source


@pytest.mark.parametrize('row',[
    dict(split='dev',checks={'completed':True},network_chat=[],raw_answer='假回答'),
    dict(split='holdout',checks={'completed':True},network_chat=[],raw_answer='隐私题'),
])
def test_non_real_or_private_source_is_rejected(row):
    with pytest.raises(ValueError):verify_real_source(row)


def test_changed_input_cannot_reuse_a_real_raw_answer():
    messages=[dict(role='user',content='原始输入')]
    model=ReplayModel(dict(model_messages=messages,raw_answer='已保存的原始输出'))
    assert asyncio.run(model.chat(messages))=='已保存的原始输出'
    with pytest.raises(RuntimeError,match='输入不同'):
        asyncio.run(model.chat([dict(role='user',content='已改变的输入')]))
