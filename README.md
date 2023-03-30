# Breaking Artificial Intelligence

텐서플로 2와 머신러닝으로 시작하는 자연어처리 라는 책을 통한 인공지능 스터디.

## Purpose

    인공지능 관련 프로젝트를 여러개 진행하면서 공식 문서와 구글링으로 얻은 지식으로는 한계가 있어서
    전문적이며 깊이 있는 지식이 필요하여 기초부터 탄탄하게 다시 공부하고자 함.

## How to ?

    "텐서플로2와 머신러닝으로 시작하는 자연어처리" 라는 서적을 통해 진행.

## Enviroment
- python 3.9.10
- TensorFlow 2.11.*

## VENV GUIDE
create env : 

    py -[python-version] -m venv [folder-name]
    ex >  py -3.8 -m venv env

use project env(package) : 

    .\[folder-name]\Scripts\activate
    ex > .\env\Scripts\activate
    [!] 만약 보안오류 발생 시 관리자 권한으로 진행.
    [!] 그래도 안된다면 다음 명령어를 통해 권한 부여
    > Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
    [!] 최초 env 진입 시 pip 업그레이드 해주세요.

use local env(package) : 

    .\[folder-name]\Scripts\deactivate || deactivate
    ex > .\env\Scripts\deactivate || deactivate