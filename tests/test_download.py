"""A fresh clone must create ignored data directories before first acquisition."""
import hashlib
import io
import zipfile
from src import data

def test_first_download_creates_missing_raw_directory(tmp_path,monkeypatch):
    source=(data.ROOT/'data/raw/credit.xls').read_bytes()
    archive=io.BytesIO()
    with zipfile.ZipFile(archive,'w') as writer:
        writer.writestr('default of credit card clients.xls',source)
    payload=archive.getvalue()
    new_root=tmp_path/'fresh-clone'
    monkeypatch.setattr(data,'ROOT',new_root)
    monkeypatch.setattr(data.urllib.request,'urlopen',lambda *args,**kwargs:io.BytesIO(payload))
    frame,checksum=data.load_data()
    assert (new_root/'data/raw/credit.xls').read_bytes()==source
    assert len(frame)>0 and frame.ID.is_unique
    assert checksum==hashlib.sha256(source).hexdigest()
