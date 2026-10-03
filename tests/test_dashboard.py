from streamlit.testing.v1 import AppTest
from src.data import ROOT

def test_all_dashboard_pages():
    app=AppTest.from_file(str(ROOT/'app/app.py'),default_timeout=60).run()
    assert not app.exception
    for page in app.sidebar.radio[0].options:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, (page,[x.message for x in app.exception])
    # Exercise the empty-acceptance endpoint explicitly.
    app.sidebar.radio[0].set_value('Decision Threshold Simulator').run()
    app.slider[0].set_value(0.).run()
    assert not app.exception

def test_app_representative_explanation():
    import json
    s=json.loads((ROOT/'artifacts/tables/run_summary.json').read_text())
    app=AppTest.from_file(str(ROOT/'app/app.py'),default_timeout=60).run()
    app.sidebar.radio[0].set_value('Applicant Risk Explorer').run()
    app.selectbox[0].set_value(s['representative_ids']['borderline']).run()
    assert not app.exception
