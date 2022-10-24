# Contributing

해당 레퍼지토리에 기여할 때 누가 어떤 내용에 기여했는지 구분하기 위해 아래의 규칙을 지키면 좋을 거 같습니다.

## 풀 리퀘스트 프로세스

1. `main`브랜치에 되도록 바로 커밋하지 말기.
2. 환경 변수가 추가되거나, 포트가 변경되거나, 파일 경로가 변경되는 등 프로젝트를 구동하기 위한 절차가 변경된 경우, README.md 를 변경하기.
3. 눈갱 방지를 위해 최소한 코드 에디터가 제공하는 [코드 정리(Linting)](https://code.visualstudio.com/docs/python/linting) 기능을 사용하여 *(VSCode 의 경우 cmd+k+f)* [PEP8](https://peps.python.org/pep-0008/)을 준수하려고 노력해 보기.
4. git 커밋에도 컨벤션 사용해보기! 근데 단순 문자로 하면 칙칙하니 이모티콘으로 해보기.
    - 🐛 : Bugfix, 버그 패치
    - ✨ : Feat, 기능 추가
    - 🎨 : Style & Typo, 기능 변경 없이 간단한 변수명, 파일명, 경로변경 등의 작업
    - 🔧 : Refactor, 기능 변경 없이 레거시를 리팩터링하는 거대한 작업
    - 📝 : Docs, 기능 변경 없이 문서 및 주석 수정
    - 🎉 : First commit

```text
<타입 이모지> <한글 한줄설명> (이슈링크)
# 한 줄 띄우기
<커밋에 대한 본문(선택 사항) 한글 설명>
🎨 여러개를 한번에 커밋하는 경우 사용하고 이렇게 설명을 달아줘도 좋음
```

`example`
```text
✨ EDA 기능 추가 (#102)

EDA 코드를 추가함
```

### 참고

- [CONTRIBUTING.md template](https://gist.github.com/PurpleBooth/b24679402957c63ec426)
- [VSCode Linting](https://code.visualstudio.com/docs/python/linting)
- [Python PEP8](https://peps.python.org/pep-0008/)
- [Git commit convention](https://www.conventionalcommits.org/ko/v1.0.0/)
- [Git commit convention reference: FastAPI Template project](https://github.com/tiangolo/full-stack-fastapi-postgresql)
- [Sementic versioning](https://semver.org/lang/ko/)
- [Repository convention reference: SSAFY BookFlex project](https://github.com/glenn93516/BookFlex)