"""Exercise rejection, restoration and no-overwrite behavior with synthetic fixtures."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checker = load('check_repository')
restorer = load('restore_artifacts')


class RepositoryCheckTests(unittest.TestCase):
    def test_rejects_credential_without_printing_value(self):
        token = ('hf_' + 'A' * 32).encode()
        issues = checker.check_file('example.py', token)
        self.assertTrue(issues)
        self.assertNotIn(token.decode(), str(issues))

    def test_rejects_data_and_logs_but_allows_provenance(self):
        for name in ['data/processed/mimic_training/example.csv',
                     'data/simulation/tested/example.csv', 'logs/example.out',
                     'checkpoints/example.pt', 'clinical_histories.txt']:
            self.assertTrue(checker.check_file(name, b'test'), name)
        self.assertFalse(checker.check_file('configs/run.json', b'{}'))
        self.assertFalse(checker.check_file('data/manifests/retired_artifacts.csv', b'header\n'))

    def test_notebook_output_and_malformed_notebooks(self):
        nb = {'cells':[{'cell_type':'code','source':['1+1'],
                        'outputs':[{'output_type':'stream','text':'2'}],'execution_count':1}]}
        self.assertTrue(checker.check_file('a.ipynb', json.dumps(nb).encode()))
        nb['cells'][0].update(outputs=[], execution_count=None)
        self.assertFalse(checker.check_file('a.ipynb', json.dumps(nb).encode()))
        self.assertTrue(checker.check_file('a.ipynb', b'{broken'))

    def test_size_limit(self):
        with patch.object(checker, 'LIMIT', 8):
            self.assertTrue(checker.check_file('a.txt', b'123456789'))


class RestorationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.repo, self.source, self.dest = (base / n for n in ('repo','source','dest'))
        for directory in (self.repo,self.source,self.dest):
            directory.mkdir()
        self.rows = []
        for i in range(2):
            name = f'data/simulation/tested/fixture_{i}.csv'
            payload = f'sequence,label\nABC,{i}\n'.encode()
            path=self.source/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)
            self.rows.append({'group':'synthetic-data','path':name,'bytes':str(len(payload)),
                              'sha256':hashlib.sha256(payload).hexdigest(),'git_blob':'unused','source_commit':'unused'})
        self.write_manifest()
        patcher=patch.object(restorer,'ROOT',self.repo)
        patcher.start();self.addCleanup(patcher.stop)

    def write_manifest(self):
        manifest=self.repo/'data/manifests/retired_artifacts.csv'
        manifest.parent.mkdir(parents=True,exist_ok=True)
        with manifest.open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(self.rows[0]));writer.writeheader();writer.writerows(self.rows)

    def test_dry_run_restore_and_repeat(self):
        self.assertEqual(restorer.restore('synthetic-data',self.source,self.dest,True)['written'],0)
        self.assertEqual(list(self.dest.iterdir()),[])
        self.assertEqual(restorer.restore('synthetic-data',self.source,self.dest,False)['written'],2)
        for row in self.rows:
            self.assertEqual((self.source/row['path']).read_bytes(),(self.dest/row['path']).read_bytes())
        self.assertEqual(restorer.restore('synthetic-data',self.source,self.dest,False)['written'],0)

    def test_existing_conflict_prevents_all_writes(self):
        target=self.dest/self.rows[1]['path'];target.parent.mkdir(parents=True);target.write_bytes(b'preserve me')
        with self.assertRaisesRegex(ValueError,'Refusing to replace'):
            restorer.restore('synthetic-data',self.source,self.dest,False)
        self.assertFalse((self.dest/self.rows[0]['path']).exists())
        self.assertEqual(target.read_bytes(),b'preserve me')

    def test_corrupt_source_prevents_all_writes(self):
        (self.source/self.rows[1]['path']).write_bytes(b'bad')
        with self.assertRaisesRegex(ValueError,'checksum mismatch'):
            restorer.restore('synthetic-data',self.source,self.dest,False)
        self.assertEqual(list(self.dest.iterdir()),[])

    def test_clinical_paths_traversal_and_symlink_escape(self):
        for name in ['data/processed/mimic_training/a.csv','../outside.csv','/tmp/outside.csv']:
            with self.assertRaises(ValueError):
                restorer.validate_record(dict(self.rows[0],path=name))
        outside=self.dest.parent/'outside';outside.mkdir()
        (self.dest/'data').symlink_to(outside,target_is_directory=True)
        with self.assertRaises(ValueError):
            restorer.restore('synthetic-data',self.source,self.dest,False)

    def test_local_git_restore_and_wrong_object_rejected(self):
        subprocess.run(['git','init','-q',str(self.source)],check=True)
        subprocess.run(['git','-C',str(self.source),'add','.'],check=True)
        subprocess.run(['git','-C',str(self.source),'-c','user.name=Test','-c','user.email=test@example.invalid',
                        'commit','-qm','synthetic fixture'],check=True)
        row=self.rows[0]
        row['source_commit']=subprocess.check_output(['git','-C',str(self.source),'rev-parse','HEAD']).decode().strip()
        row['git_blob']=subprocess.check_output(['git','-C',str(self.source),'rev-parse','HEAD:'+row['path']]).decode().strip()
        with patch.object(restorer,'ROOT',self.source):
            self.assertEqual(restorer.read_artifact(row,None),(self.source/row['path']).read_bytes())
            with self.assertRaisesRegex(ValueError,'Git object mismatch'):
                restorer.read_artifact(dict(row,git_blob='0'*40),None)


if __name__ == '__main__':
    unittest.main()
