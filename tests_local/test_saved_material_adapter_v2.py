"""插入条号前缀碰撞回归：V1已绑定首轮协议，修复版独立保留。"""
import importlib.util
import json
import test_saved_material_adapter as original_tests
from test_saved_material_adapter import sha


class ExactArticleTests(original_tests.AdapterTests):
    def setUp(self):
        super().setUp()
        spec=importlib.util.find_spec('prototype.saved_material_adapter_v2')
        self.assertIsNotNone(spec)
        from prototype import saved_material_adapter_v2
        self.api=saved_material_adapter_v2

    def test_inserted_article_cannot_impersonate_parent_article(self):
        self.article['text']='第一条之一 真实的插入条文。'
        self.raw='<html><body>第一条之一 真实的插入条文。</body></html>'.encode()
        self.articles=json.dumps([self.article],ensure_ascii=False).encode()
        self.record.update(raw_sha256=sha(self.raw),parsed_sha256=sha(self.articles))
        self.write_source()
        with self.assertRaises(ValueError): self.build()

    def test_complete_inserted_article_number_is_supported(self):
        self.article.update(article_no='第一条之一',text='第一条之一 真实的插入条文。')
        self.raw='<html><body>第一条之一 真实的插入条文。</body></html>'.encode()
        self.articles=json.dumps([self.article],ensure_ascii=False).encode()
        self.record.update(raw_sha256=sha(self.raw),parsed_sha256=sha(self.articles))
        self.write_source()
        self.assertEqual(self.build()[0][0]['id'],'test-law/第一条之一')
