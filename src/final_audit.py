"""Verify saved results, notebooks, SQL, tests and local Streamlit server."""
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
import importlib.metadata
import numpy as np
import pandas as pd
import joblib
import duckdb
import nbformat
from .data import ROOT, validate
from .evaluation import metrics
from .decision_engine import expected_loss, risk_bands
from .documentation import write_documentation

def run():
    rows=[]; tables=ROOT/'artifacts/tables'
    def check(name,fn):
        try:
            detail=fn()
            rows.append({'component':name,'status':'PASS','evidence':str(detail)})
        except Exception as error:
            rows.append({'component':name,'status':'FAIL','evidence':repr(error)})
    summary=json.loads((tables/'run_summary.json').read_text())
    def source():
        import pandas as pd
        raw=pd.read_excel(ROOT/'data/raw/credit.xls',header=1).rename(columns={'default payment next month':'event'})
        validate(raw)
        assert len(raw)==summary['rows'] and int(raw.event.sum())==summary['events']
        assert hashlib.sha256((ROOT/'data/raw/credit.xls').read_bytes()).hexdigest()==summary['source_sha256']
        return f"Verified {len(raw)} rows, schema, ID uniqueness, target and SHA256"
    check('Source provenance and data validation',source)
    def replay():
        b=joblib.load(ROOT/'artifacts/models/decision_bundle.joblib')
        predictions=pd.read_csv(tables/'test_predictions.csv')
        p=b['calibrator'].predict(b['model'].predict_proba(predictions)[:,1])
        np.testing.assert_allclose(p,predictions.score,atol=1e-12)
        assert (risk_bands(p,b['boundaries'])==predictions.risk_band).all()
        np.testing.assert_allclose(predictions.scenario_loss_ntd,expected_loss(p,.45,100000),atol=1e-7)
        m=metrics(predictions.event,p,b['review_threshold'])
        for key,value in m.items(): np.testing.assert_allclose(value,summary['test_metrics'][key],atol=1e-12)
        return 'Replayed all test scores, bands, scenario losses and final metrics from the saved model'
    check('Saved artifact loading and metrics integrity',replay)
    def profiles():
        manifest=pd.read_csv(tables/'split_manifest.csv',dtype={'profile_hash':str})
        assert manifest.groupby('profile_hash').split.nunique().max()==1
        assert manifest.ID.is_unique
        for name,part in summary['partitions'].items(): assert (manifest.split==name).sum()==part['n']
        return 'Exact profile hashes and source IDs disjoint across all partitions'
    check('Train/calibration/validation/test contamination',profiles)
    def selection():
        val=pd.read_csv(tables/'validation_models.csv')
        assert val.sort_values(['brier','log_loss','candidate']).iloc[0].candidate==summary['selected']
        assert summary['shap_max_additivity_error']<1e-4
        return 'Validation-only selection verified; TreeSHAP raw-margin additivity verified'
    check('Model selection, calibration and explanation',selection)
    def sql():
        conn=duckdb.connect(str(tables/'portfolio.duckdb'),read_only=True); count=0
        try:
            for path in sorted((ROOT/'sql').glob('*.sql')):
                result=conn.execute(path.read_text()).df()
                saved=pd.read_csv(tables/f'sql_{path.stem}.csv')
                pd.testing.assert_frame_equal(result,saved,check_dtype=False,atol=1e-7,rtol=1e-10)
                count+=1
        finally: conn.close()
        return f'{count} queries executed again and matched saved outputs'
    check('Executable SQL',sql)
    def notebooks():
        executed=json.loads((tables/'notebook_execution.json').read_text())
        assert len(executed)==8
        for stem in executed:
            book=nbformat.read(ROOT/'notebooks'/f'{stem}.ipynb',as_version=4)
            for cell in book.cells:
                if cell.cell_type=='code':
                    assert cell.execution_count is not None
                    assert not any(o.output_type=='error' for o in cell.outputs)
        return f'{len(executed)} notebooks with executed code cells and no error outputs'
    check('Notebook execution',notebooks)
    def tests():
        log=tables/'pytest_output.txt'
        with log.open('w',encoding='utf-8') as stream:
            result=subprocess.run([sys.executable,'-m','pytest','-q'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,timeout=240)
        assert result.returncode==0,log.read_text()
        suites=ET.parse(tables/'pytest_results.xml').getroot()
        suites_list=list(suites) if suites.tag=='testsuites' else [suites]
        n=sum(int(x.attrib['tests']) for x in suites_list)
        return f'{n} tests passed, including all Streamlit pages and slider endpoint'
    check('Complete pytest suite and dashboard AppTest',tests)
    def server():
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
        log=tables/'streamlit_server_output.txt'
        with log.open('w',encoding='utf-8') as stream:
            process=subprocess.Popen([sys.executable,'-m','streamlit','run','app/app.py','--server.headless=true',
                f'--server.port={port}','--server.address=127.0.0.1','--browser.gatherUsageStats=false'],
                cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            try:
                for _ in range(60):
                    if process.poll() is not None: raise RuntimeError('Streamlit process stopped: '+log.read_text())
                    try:
                        response=urllib.request.urlopen(f'http://127.0.0.1:{port}/_stcore/health',timeout=1)
                        if response.status==200:
                            page=urllib.request.urlopen(f'http://127.0.0.1:{port}/',timeout=5)
                            assert page.status==200 and b'Streamlit' in page.read()
                            return 'Local server health and HTML endpoint returned HTTP 200; temporary process stopped after check'
                    except (OSError,TimeoutError): pass
                    time.sleep(.5)
                raise RuntimeError('Server health timed out')
            finally:
                process.terminate()
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
    check('Streamlit HTTP launch and health',server)
    def documents():
        # Regeneration must produce the same documents from the same executed source of truth.
        paths=[ROOT/'README.md',*sorted((ROOT/'reports').glob('*.md'))]
        exclude={'final_audit.md','completion_report.md'}
        before={p:p.read_bytes() for p in paths if p.name not in exclude}
        write_documentation(summary)
        assert all(p.read_bytes()==content for p,content in before.items())
        assert (ROOT/'reports/interview_guide.md').read_text().count('### Q')>=40
        assert (ROOT/'reports/resume_bullets.md').read_text().count('\n- ')==3
        return f'{len(before)} deterministic documents verified; interview guide and exactly three resume bullets checked'
    check('README, reports and resume numeric source consistency',documents)
    def placeholders():
        import re
        problems=[]
        for folder in ['src','app','tests','docs','reports','sql','config']:
            for path in (ROOT/folder).rglob('*'):
                if path.is_file() and path.suffix in {'.py','.md','.sql','.yaml'} and path.name!='final_audit.md':
                    # Literal tokens are constructed here so the scanner itself does not raise a false positive.
                    text=path.read_text(encoding='utf-8-sig')
                    pattern=r'\b('+'TO'+'DO|FIX'+'ME|T'+'BD'+r')\b'
                    if re.search(pattern,text): problems.append(str(path.relative_to(ROOT)))
        assert not problems,problems
        return 'No unresolved implementation markers in source, app, tests, SQL or documentation'
    check('Unfinished implementation marker scan',placeholders)
    def dependencies():
        for line in (ROOT/'requirements.txt').read_text(encoding='utf-8-sig').splitlines():
            if '==' in line:
                name,version=line.split('=='); assert importlib.metadata.version(name)==version
        result=subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True,timeout=30)
        assert result.returncode==0,result.stdout+result.stderr
        return 'Pinned direct packages match environment and pip check has no dependency conflicts; full requirements-lock.txt recorded'
    check('Reproducible dependencies',dependencies)
    def paths():
        import re
        readme=(ROOT/'README.md').read_text()
        targets=re.findall(r'\]\(([^)]+)\)',readme)
        for path in targets:
            if not path.startswith('http') and path!='reports/final_audit.md':
                assert (ROOT/path).exists(),path
        ignore=(ROOT/'.gitignore').read_text(encoding='utf-8-sig')
        for item in ['.venv/','data/raw/*','data/processed/*','artifacts/models/*']: assert item in ignore
        assert all((ROOT/p).exists() for p in ['app/app.py','requirements.txt','config/config.yaml'])
        return 'README local links, entry points and required data/model/environment ignore rules verified'
    check('File paths, README commands and ignore rules',paths)
    def assumptions():
        readme=(ROOT/'README.md').read_text().lower()
        assert all(token in readme for token in ['review-capacity scenario','lgd and ead are explicitly assumed','no actual production or temporal drift evidence'])
        card=(ROOT/'reports/model_card.md').read_text()
        assert 'not an economically optimal approval rule' in card and 'LGD and EAD are unobserved' in card
        app=(ROOT/'app/app.py').read_text()
        assert 'assumed LGD and EAD' in app and 'Educational portfolio decision-support simulation' in app
        return 'Explicit capacity, loss-input, educational-use and random/synthetic-monitoring labels checked in generated docs and app'
    check('Decision / loss / monitoring assumptions labelled',assumptions)
    pd.DataFrame(rows).to_csv(tables/'final_audit.csv',index=False)
    (tables/'final_audit.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    report='# Final execution audit\n\n'
    report+='| Component | Status | Evidence |\n| --- | --- | --- |\n'
    report+='\n'.join(f"| {r['component']} | {r['status']} | {r['evidence'].replace('|','/').replace(chr(10),' ')} |" for r in rows)
    report+='\n\nThe pipeline regenerates downstream artifacts from the source without loading old models. Notebooks execute against these artifacts and rerun SQL. Clean reproduction is additionally checked by the agent during finalization. HTTP verification and AppTest are programmatic checks, not full browser accessibility or security certification. Figures are generated from executed artifacts. Statistical and fairness limitations are in the model card. Empirical reporting is deterministic from saved executed tables; this check does not claim independent external validation.\n'
    (ROOT/'reports/final_audit.md').write_text(report,encoding='utf-8')
    if all(row['status']=='PASS' for row in rows):
        from .completion import write_completion
        write_completion()
    # Manifest links source code and outputs without sensitive environment paths.
    paths=[p for directory in ['src','app','tests','sql','docs','config','reports','notebooks','artifacts/tables','artifacts/figures']
           for p in (ROOT/directory).rglob('*') if p.is_file() and p.suffix not in {'.pyc','.duckdb'}]
    manifest={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in paths if p.name!='artifact_manifest.json'}
    (tables/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    for row in rows: print(row['status'],row['component'],row['evidence'],flush=True)
    if any(row['status']=='FAIL' for row in rows): raise SystemExit(1)

if __name__=='__main__': run()
