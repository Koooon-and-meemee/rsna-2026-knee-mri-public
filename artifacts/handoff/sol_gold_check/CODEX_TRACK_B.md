# 코덱스 지시문 — Track B (비판적 교정)

**Track A 가 끝난 뒤, 새 세션에서 돌린다.** A 와 같은 세션이면 안 된다.

---

너는 근골격계 영상의학 전문의로서, 이전 판독자가 무릎 MRI 판독문에서 뽑아둔
구조화된 소견을 감사한다. 네 출력이 이미지 모델의 학습 라벨이 되므로
속도보다 정확도가 중요하다.

## 작업

1. `/home/sangwon-park/kaggle/artifacts/handoff/sol_gold_check/prompt_critique.txt` 를
   읽는다. 라벨링 규칙 전문과, 기존 라벨을 어떻게 다뤄야 하는지가 들어 있다.
   **그 지시를 그대로 따른다.**

2. `gold58_with_fable.json` 의 58건을 처리한다. 각 항목은 판독문과 이전 판독자의
   예측을 함께 담고 있다.

3. 결과를 `labels_gold58_sol_critique.csv` 로 쓴다. 형식은 prompt_critique.txt 에
   지정된 대로.

4. 채점한다:

       cd /home/sangwon-park/kaggle/artifacts/handoff/sol_gold_check
       uv run python compare.py pseudo_student_gold.csv labels_gold58_sol_critique.csv --names student sol_critique

## 이 트랙의 함정 — 반드시 읽을 것

이 트랙은 **구조상 오염돼 있다.** 모델에게 기존 숫자를 보여주면 그쪽으로 끌려간다.
이건 추측이 아니라 이 프로젝트가 측정한 사실이다: 이전 판독자는 프롬프트에
ACL 유병률 0.41 을 주입받고 0.29~0.40 을 출력했고, 아무 숫자도 못 받은 다른
라벨러는 같은 코퍼스에서 0.154 를 냈다.

그러니 **각 소견마다 판독문만 보고 네 판단을 먼저 세운 뒤에** 기존 값을 확인해라.
기존 값은 증거이지 출발점이 아니다.

그리고 기존 라벨은 소견마다 신뢰도가 다르다. 실제 방사선의 라벨 58건으로 채점한 값:

    믿을 만함                    믿기 어려움
    ACL              0.996       Synovitis        0.751
    MCL              0.983       Lateral OA       0.829
    Medial Meniscus  0.963       Effusion         0.847
    Medial OA        0.945       Contusion        0.872

왼쪽에서 이견이 나면 네가 틀렸을 가능성이 더 크다 — 판독문에 명시적 근거가 있을 때만
뒤집어라. 오른쪽에서는 기존 값에 정보가 거의 없으니 네 판단이 지배해야 한다.
특히 Synovitis 는 기존 라벨러가 거의 찍기 수준이다.

## 반드시 지킬 것

- 이 디렉터리 밖의 파일을 만들거나 고치지 말 것. **`src/make_batches.py` 절대 실행 금지** —
  기존 배치를 지우고 번호를 재부여해 확보된 라벨 1,000건이 조용히 어긋난다.
- `gold58_truth.csv` 는 채점용 정답이다. 라벨링 중에 열지 마라.

## 끝나고 보고할 것

1. `compare.py` 출력 전문
2. **기존 라벨과 가장 크게 갈린 판독문 2~3건**과 각각 한 줄 이유.
   3,400건을 마저 다시 라벨링할지 결정하는 데는 총점보다 이 목록이 더 쓸모 있다.
3. 네가 뒤집은 것이 주로 어느 소견에 몰렸는지
