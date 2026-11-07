import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from src.runtime import Runtime


class ReleaseStatusTests(unittest.TestCase):
    def test_missing_failed_or_stale_evaluation_cannot_promote_an_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = Runtime(root / 'index.sqlite3', root)
            physical = {'build_id': 'current', 'release_ready': True, 'gates': {'blockers': []}}
            self.assertFalse(runtime._qualify_release(physical)['release_ready'])
            suite = root / 'resources/evaluation.json'
            suite.parent.mkdir()
            suite.write_text('{"cases": ["frozen"]}')
            report_path = root / 'data/evaluation/release.json'
            report_path.parent.mkdir(parents=True)
            report = dict(passed=True, split='heldout', mode='hybrid', build_id='current',
                          suite_sha256=hashlib.sha256(suite.read_bytes()).hexdigest())
            report_path.write_text(json.dumps(report))
            self.assertTrue(runtime._qualify_release(physical)['release_ready'])
            for field, value in [('passed', False), ('split', 'development'), ('build_id', 'previous'), ('suite_sha256', 'old')]:
                report_path.write_text(json.dumps(dict(report, **{field: value})))
                self.assertFalse(runtime._qualify_release(physical)['release_ready'], field)


if __name__ == '__main__':
    unittest.main()
