import pytest
from github import Github

g = Github("anilbhatt1", "github$123")

repo = g.get_repo("anilbhatt1/S1_dvc_emlo")
contents = repo.get_contents("")
content_lst = []
file_lst = []
while contents:
    file_content = contents.pop(0)
    if file_content.type == "dir":
        contents.extend(repo.get_contents(file_content.path))
    else:
        content_lst.append(file_content)
##
for file in content_lst:
    item = str(file)
    file_nm = item.split('/')[-1].split('"')[-2]
    file_lst.append(file_nm)

def test_data_zip():
    if 'data.zip' in file_lst:
        stat = 'Found'
    else:
        stat = 'Not Found'
    assert stat == 'Not Found'

def test_model_pth():
    if 'model.pth' in file_lst:
        stat = 'Found'
    else:
        stat = 'Not Found'
    assert stat == 'Not Found'

def test_data_dvc():
    if 'data.dvc' in file_lst:
        stat = 'Found'
    else:
        stat = 'Not Found'
    assert stat == 'Found'

def test_model_dvc():
    if 'model.pth.dvc' in file_lst:
        stat = 'Found'
    else:
        stat = 'Not Found'
    assert stat == 'Found'
