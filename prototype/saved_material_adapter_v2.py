"""真实材料适配修订版：在V1独立原文核验上补完整条号检查。

V1源码已绑定首轮协议，不覆盖它；主入口 build_source_catalog。
副作用、输入输出及原文限制沿用V1，不联网、不改材料或候选。
"""
import re

from .saved_material_adapter import adapt_trace, read_bound
from .saved_material_adapter import build_source_catalog as _build_v1


HEADER=re.compile(r'^第[〇零一二三四五六七八九十百千万两0-9]+条(?:之[〇零一二三四五六七八九十百千万两0-9]+)?(?=\s|$)')


def build_source_catalog(root,source_id,bindings):
    """正文首部完整条号必须与目录条号相等，防止主条吞掉“之一”等插入后缀。"""
    rows,provenance=_build_v1(root,source_id,bindings)
    for row in rows:
        number=row['id'].split('/',1)[1]
        matched=HEADER.match(row['text'].lstrip())
        if matched is None or matched[0]!=number:
            raise ValueError('正文完整条号与目录身份不符：'+row['id'])
    provenance['article_identity_check']='EXACT_HEADER_INCLUDING_INSERTION_SUFFIX'
    return rows,provenance
