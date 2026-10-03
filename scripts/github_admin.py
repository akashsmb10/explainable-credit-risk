"""Authenticated repository administration. Credentials stay in process memory."""
import argparse
import json
import os
import subprocess
import sys
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
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=['status','enable-pages','pages-status','metadata'])
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

if __name__=='__main__':
    try: main()
    except (RuntimeError,requests.RequestException) as error:
        print(str(error),file=sys.stderr); raise SystemExit(1)
