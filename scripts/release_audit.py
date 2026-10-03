"""Check publishable content, local links and exported results without network access."""
import json
import re
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.decision_engine import threshold_table

def run():
    checks=[]
    def record(component,evidence): checks.append({'component':component,'status':'PASS','evidence':evidence})
    snapshot=json.loads((ROOT/'site/data.json').read_text())
    source=json.loads((ROOT/'artifacts/tables/run_summary.json').read_text())
    assert snapshot['summary']==source
    clients=pd.DataFrame(snapshot['clients']).sort_values('id')
    saved=pd.read_csv(ROOT/'artifacts/tables/test_predictions.csv').sort_values('ID')
    np.testing.assert_array_equal(clients.id,saved.ID)
    np.testing.assert_allclose(clients.score,saved.score,atol=1e-14)
    np.testing.assert_array_equal(clients.event,saved.event)
    assert clients.id.is_unique and all(len(d)==5 for d in clients.drivers)
    result=threshold_table(clients.event,clients.score.to_numpy(),[source['review_threshold']]).iloc[0]
    assert int(result.tp)==source['test_metrics']['tp'] and int(result.fn)==source['test_metrics']['fn']
    record('Public snapshot integrity',f'{len(clients)} scores, events and driver records match saved holdout outputs; policy counts match')
    paths=[ROOT/'README.md',*sorted((ROOT/'docs').glob('*.md')),*sorted((ROOT/'reports').glob('*.md'))]
    links=0
    for path in paths:
        text=path.read_text(encoding='utf-8-sig')
        assert '\ufffd' not in text and 'â†' not in text and 'Ã—' not in text, path
        for target in re.findall(r'\]\(([^)]+)\)',text):
            target=target.split('#',1)[0]
            if target and not target.startswith(('http://','https://','mailto:')):
                assert (path.parent/target).resolve().exists(),f'{path.relative_to(ROOT)}: missing {target}'
                links+=1
    record('Markdown presentation',f'{links} local document/image links verified; UTF-8 characters checked')
    html=(ROOT/'site/index.html').read_text()
    for filename in ['app.js','style.css','favicon.svg','data.json','.nojekyll']:
        assert (ROOT/'site'/filename).exists()
    for asset in re.findall(r"(?:image\(|src=)[\"'](.*?\.png)[\"']",(ROOT/'site/app.js').read_text()):
        assert (ROOT/'site/assets'/asset).exists(),asset
    assert 'rel="noopener noreferrer"' in html and 'not bank underwriting advice' in html
    record('Public demo assets and scope','Static entry points, analytical images and educational/public-data scope present')
    # Git candidate listing honors .gitignore; before init, examine only the intentional source directories.
    if (ROOT/'.git').exists():
        result=subprocess.run(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=ROOT,capture_output=True,check=True)
        candidates=set(filter(None,result.stdout.decode().split('\0')))
    else:
        candidates={str(p.relative_to(ROOT)).replace('\\','/') for d in ['src','app','scripts','site','tests','sql','docs','reports','config','notebooks','artifacts/figures','artifacts/tables','.github']
            for p in (ROOT/d).rglob('*') if p.is_file() and p.suffix not in {'.pyc','.duckdb','.joblib'}}
        candidates-= {'artifacts/tables/test_predictions.csv','artifacts/tables/shap_local.csv','artifacts/tables/pytest_output.txt',
                     'artifacts/tables/streamlit_server_output.txt','artifacts/tables/pytest_results.xml'}
    large=[]
    for name in candidates:
        assert not name.startswith(('.venv/','data/raw/','data/processed/','artifacts/models/','artifacts/jupyter/','pytest-cache-files-'))
        p=ROOT/name
        if p.stat().st_size>15*1024*1024: large.append(name)
        if p.suffix in {'.py','.md','.json','.ipynb','.yaml','.yml','.html','.js','.css','.txt'}:
            content=p.read_text(encoding='utf-8-sig')
            # Match credential-shaped values, never generic variable names or authentication documentation.
            patterns=[r'gh[pousr]_[A-Za-z0-9]{30,}',r'github_pat_[A-Za-z0-9_]{30,}',r'AKIA[A-Z0-9]{16}',
                      r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----']
            assert not any(re.search(pattern,content) for pattern in patterns),f'Credential-shaped value in {name}'
            assert not re.search(r'C:[/\\]+Users[/\\]+',content,re.I),f'Local user path in {name}'
    assert not large,large
    record('Upload hygiene',f'{len(candidates)} candidate files checked: no environment/raw-model artifacts, credential-shaped values, local user paths or files above size limit')
    ignore=(ROOT/'.gitignore').read_text(encoding='utf-8-sig')
    assert all(value in ignore for value in ['.venv/','artifacts/models/*','data/raw/*','data/processed/*'])
    record('Licensing and exclusions','MIT license, dataset NOTICE and source/artifact ignore rules checked')
    assert (ROOT/'LICENSE').exists() and (ROOT/'NOTICE').exists()
    summary={'checks':checks}
    (ROOT/'artifacts/tables/release_audit.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    report='# Publication checks\n\n| Component | Status | Evidence |\n| --- | --- | --- |\n'
    report+='\n'.join(f"| {r['component']} | {r['status']} | {r['evidence']} |" for r in checks)
    report+='\n\nThis scan checks intentional upload files and saved-result consistency. Browser rendering, slider/scenario behavior and HTTP deployment verification are recorded separately in the browser and deployment audit artifacts. It is not a security certification.\n'
    (ROOT/'reports/publication_audit.md').write_text(report,encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': run()
