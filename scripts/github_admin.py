"""Authenticated repository administration. Credentials stay in process memory."""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import requests

REPOSITORY='akashsmb10/explainable-credit-risk'
ROOT=Path(__file__).resolve().parents[1]

def session():
    env=os.environ.copy(); env['GIT_TERMINAL_PROMPT']='0'
    result=subprocess.run(['git','-c','credential.interactive=never','credential','fill'],
        input='protocol=https\nhost=github.com\n\n',capture_output=True,text=True,env=env,timeout=30)
    if result.returncode:
        raise RuntimeError('No usable GitHub credential is available through Git Credential Manager. Sign in before publishing.')
    values=dict(line.split('=',1) for line in result.stdout.splitlines() if '=' in line)
    if not values.get('password'): raise RuntimeError('GitHub credential is unavailable')
    client=requests.Session()
    client.headers.update({'Authorization':'Bearer '+values['password'],'Accept':'application/vnd.github+json',
                           'X-GitHub-Api-Version':'2022-11-28'})
    return client

def request(client,method,path,**kwargs):
    response=client.request(method,'https://api.github.com'+path,timeout=45,**kwargs)
    if response.status_code>=400:
        message=response.json().get('message','GitHub API request failed')
        raise RuntimeError(f'{method} {path}: HTTP {response.status_code}: {message}')
    return response.json() if response.content else {}

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=['status','enable-pages','pages-status','metadata','verify-public','actions-status','ci-diagnostics'])
    args=parser.parse_args(); client=session(); base='/repos/'+REPOSITORY
    if args.action=='status':
        repo=request(client,'GET',base)
        print(json.dumps({k:repo.get(k) for k in ['full_name','private','default_branch','permissions','size','html_url']},indent=2))
    elif args.action=='enable-pages':
        current=client.get('https://api.github.com'+base+'/pages',timeout=45)
        if current.status_code==404:
            value=request(client,'POST',base+'/pages',json={'source':{'branch':'gh-pages','path':'/'},'build_type':'legacy'})
        elif current.status_code==200:
            value=request(client,'PUT',base+'/pages',json={'source':{'branch':'gh-pages','path':'/'},'build_type':'legacy'})
        else: raise RuntimeError(f'Pages access returned HTTP {current.status_code}')
        print(json.dumps({k:value.get(k) for k in ['status','html_url','source','build_type']},indent=2))
    elif args.action=='pages-status':
        page=request(client,'GET',base+'/pages')
        build=client.get('https://api.github.com'+base+'/pages/builds/latest',timeout=45)
        latest=build.json() if build.status_code==200 else {}
        value={'status':page.get('status'),'html_url':page.get('html_url'),'source':page.get('source'),
               'latest_build_status':latest.get('status'),'latest_commit':latest.get('commit'),
               'build_error':latest.get('error',{}).get('message')}
        print(json.dumps(value,indent=2))
    elif args.action=='metadata':
        repo=request(client,'PATCH',base,json={
            'description':'Explainable retail credit-risk analytics with calibrated ML, SHAP, SQL segmentation, decision simulations, and an interactive dashboard.',
            'homepage':'https://akashsmb10.github.io/explainable-credit-risk/'})
        request(client,'PUT',base+'/topics',json={'names':['credit-risk','explainable-ai','xgboost','shap','streamlit','risk-analytics','machine-learning','sql','portfolio-project','model-calibration']})
        print(json.dumps({k:repo.get(k) for k in ['html_url','homepage','description']},indent=2))
    elif args.action=='actions-status':
        value=request(client,'GET',base+'/actions/runs?per_page=5')
        print(json.dumps([{'name':r['name'],'status':r['status'],'conclusion':r['conclusion'],
                           'sha':r['head_sha'],'html_url':r['html_url']} for r in value['workflow_runs']],indent=2))
    elif args.action=='ci-diagnostics':
        runs=request(client,'GET',base+'/actions/runs?per_page=10')['workflow_runs']
        run=next(r for r in runs if r['name']=='Verify credit-risk project')
        jobs=request(client,'GET',base+f"/actions/runs/{run['id']}/jobs")['jobs']
        print(json.dumps([{'name':j['name'],'conclusion':j['conclusion'],'steps':[
            {'name':st['name'],'conclusion':st['conclusion']} for st in j.get('steps',[])]} for j in jobs],indent=2))
        for j in jobs:
            if j['conclusion']=='failure':
                response=client.get('https://api.github.com'+base+f"/actions/jobs/{j['id']}/logs",timeout=45)
                if response.status_code==200:
                    print('\n'.join(response.text.splitlines()[-90:]))
    elif args.action=='verify-public':
        page=request(client,'GET',base+'/pages')
        url=page.get('html_url')
        expected_url='https://akashsmb10.github.io/explainable-credit-risk/'
        if url!=expected_url: raise RuntimeError('Unexpected public Pages URL; publication is not verified')
        # Never send the API's Authorization header to the public hosting domain.
        response=requests.get(url,timeout=45)
        if response.status_code!=200 or 'Explainable Credit Risk' not in response.text:
            raise RuntimeError(f'Public demo is not ready: HTTP {response.status_code}')
        payload=requests.get(url+'data.json',timeout=45)
        payload.raise_for_status(); saved=json.loads((ROOT/'site/data.json').read_text())
        if payload.json()!=saved: raise RuntimeError('Public snapshot differs from reviewed local snapshot')
        for name in ['app.js','style.css','favicon.svg','assets/calibration.png','assets/shap_summary.png']:
            asset=requests.get(url+name,timeout=45)
            if asset.status_code!=200: raise RuntimeError(f'Public asset unavailable: {name}')
        value={'repository_url':'https://github.com/'+REPOSITORY,'demo_url':url,'verified_live':True,
               'verified_utc':datetime.now(timezone.utc).isoformat(),
               'source_sha256':saved['summary']['source_sha256'],'snapshot_clients':len(saved['clients']),
               'hosting':'GitHub Pages','demo_type':'static frozen-results explorer; not a hosted Streamlit process'}
        (ROOT/'config/publication.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
        (ROOT/'artifacts/tables/deployment_audit.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
        print(json.dumps(value,indent=2))

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    try: main()
    except (RuntimeError,requests.RequestException) as error:
        print(str(error),file=sys.stderr); raise SystemExit(1)
