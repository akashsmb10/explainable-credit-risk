"""Optional Playwright QA for the static explorer; no training or real decisions."""
import argparse
import functools
import http.server
import json
from pathlib import Path
import sys
import threading
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]

class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self,*args): pass

def check(base_url=None):
    server=None
    if not base_url:
        handler=functools.partial(QuietHandler,directory=str(ROOT/'site'))
        server=http.server.ThreadingHTTPServer(('127.0.0.1',0),handler)
        threading.Thread(target=server.serve_forever,daemon=True).start()
        base_url=f'http://127.0.0.1:{server.server_port}/'
    checks=[]; errors=[]
    try:
        with sync_playwright() as p:
            edge=Path('C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe')
            options={'headless':True}
            if edge.exists(): options['executable_path']=str(edge)
            browser=p.chromium.launch(**options)
            page=browser.new_page(viewport={'width':1440,'height':1080},device_scale_factor=1)
            page.on('pageerror',lambda error:errors.append(str(error)))
            page.goto(base_url,wait_until='networkidle',timeout=60000)
            page.locator('#content').wait_for(state='visible')
            assert 'Credit risk, made explainable.' in page.locator('h1').inner_text()
            expected=json.loads((ROOT/'artifacts/tables/run_summary.json').read_text())
            actual=page.evaluate('D.summary')
            assert actual['source_sha256']==expected['source_sha256']
            assert actual['test_metrics']==expected['test_metrics']
            checks.append('Loaded snapshot matches executed source hash and test metrics')
            page.screenshot(path=str(ROOT/'site/assets/dashboard-preview.png'),full_page=True)
            for name in ['overview','distribution','applicant','performance','explainability','decisions','loss']:
                page.locator(f'button[data-page="{name}"]').click()
                page.wait_for_function('(name)=>document.querySelector(`button[data-page="${name}"]`).getAttribute("aria-current")==="page"',arg=name)
                assert page.locator('h1').count()==1
                assert page.locator('body').evaluate('(e)=>e.scrollWidth <= window.innerWidth')
                checks.append(f'Desktop page rendered: {name}')
            page.locator('button[data-page="decisions"]').click()
            page.wait_for_selector('#threshold')
            frozen=page.evaluate('decision(D.summary.review_threshold)')
            assert frozen['tp']==expected['test_metrics']['tp']
            assert frozen['fn']==expected['test_metrics']['fn']
            for value in [0,1]:
                page.locator('#threshold').evaluate('(e,v)=>{e.value=v;e.dispatchEvent(new Event("input",{bubbles:true}));}',value)
                actual=page.evaluate('(v)=>decision(v)',value)
                assert actual['review_rate']==(1 if value==0 else 0)
                assert 'NaN' not in page.locator('#decision-output').inner_text()
            page.locator('#reset-threshold').click()
            assert abs(float(page.locator('#threshold').input_value())-expected['review_threshold'])<1e-14
            checks.append('Frozen policy counts, threshold endpoints and exact reset passed')
            page.locator('button[data-page="loss"]').click()
            page.wait_for_selector('#ead-a')
            assert '9,000 NTD' in page.locator('#loss-output').inner_text()
            assert '18,000 NTD' in page.locator('#loss-output').inner_text()
            page.locator('#ead-a').fill('-1')
            assert 'Enter finite nonnegative exposures' in page.locator('#loss-output').inner_text()
            page.locator('#ead-a').fill('0')
            assert '0 NTD' in page.locator('#loss-output').inner_text()
            checks.append('Loss scaling, zero exposure and invalid negative-exposure handling passed')
            page.set_viewport_size({'width':390,'height':844})
            for name in ['overview','distribution','applicant','performance','explainability','decisions','loss']:
                page.locator(f'button[data-page="{name}"]').click()
                page.wait_for_function('(name)=>document.querySelector(`button[data-page="${name}"]`).getAttribute("aria-current")==="page"',arg=name)
                assert page.locator('body').evaluate('(e)=>e.scrollWidth <= window.innerWidth'),name
                checks.append(f'Mobile page fits viewport: {name}')
            assert not errors,errors
            page.locator('button[data-page="overview"]').click()
            page.screenshot(path=str(ROOT/'artifacts/figures/demo-mobile.png'),full_page=True)
            browser.close()
    finally:
        if server: server.shutdown();server.server_close()
    result={'status':'PASS','target':'public deployment' if base_url.startswith('https://') else 'local demo',
            'checks':checks,'page_errors':errors}
    (ROOT/'artifacts/tables/browser_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--url');args=parser.parse_args()
    check(args.url)
