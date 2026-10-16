# 코덱스 지시문 — Track A (블라인드 재추출)

**이 트랙을 먼저, 단독 세션에서 돌린다.** Track B 와 같은 세션에서 이어 하지 말 것 —
Track B 는 기존 라벨을 보여주므로, 한 번 보고 나면 블라인드가 오염된다.

---

너는 근골격계 영상의학 전문의로서 무릎 MRI 판독문에서 구조화된 소견을 추출한다.
네 출력이 이미지 모델의 학습 라벨이 되므로 속도보다 정확도가 중요하다.

## 작업

1. `/home/sangwon-park/kaggle/artifacts/handoff/sol_gold_check/prompt_blind.txt` 를 읽는다.
   그 파일에 라벨링 규칙 전문(12개 소견 정의, 부정 표현 처리, 확률 등급화 규칙 등)이
   들어 있다. **그 지시를 그대로 따른다.**

2. `gold58_reports.json` 의 58개 판독문 전부를 라벨링한다. 스페인어·터키어·영어 등이
   섞여 있으며, 번역하지 말고 각 언어 그대로 읽는다.

3. 결과를 `labels_gold58_sol.csv` 로 쓴다. 컬럼은 `id` 와 12개 소견,
   확률은 소수점 2자리, id 는 입력과 정확히 같게, 58행 전부 있어야 한다.
   `Baker's` 헤더는 CSV 인용 처리할 것.

4. 채점한다:

       cd /home/sangwon-park/kaggle/artifacts/handoff/sol_gold_check
       uv run python compare.py pseudo_student_gold.csv labels_gold58_sol.csv --names student sol

## 반드시 지킬 것

- **이 디렉터리 밖의 파일을 만들거나 고치지 말 것.** 특히 `src/make_batches.py` 는
  절대 실행하지 마라 — 기존 배치를 지우고 번호를 다시 매겨서, 이미 확보된
  라벨 1,000건이 조용히 어긋난다.
- `gold58_truth.csv` 는 **채점용 정답이다. 라벨링 중에 열어보지 마라.** 열면 이 측정
  전체가 무의미해진다.
- `pseudo_fable_gold.csv` 와 `gold58_with_fable.json` 도 **열지 마라.** 기존 라벨러의
  답이며, 보는 순간 이 트랙은 블라인드가 아니게 된다.

## 끝나고 보고할 것

1. `compare.py` 출력 전문 (소견별 표 + MACRO + bootstrap 판정)
2. 판독문이 애매해서 확률을 중간값으로 둘 수밖에 없었던 케이스 2~3개와 그 이유
3. 12개 소견 중 판독문이 아예 언급하지 않는 비율이 특히 높았던 소견

넘어야 할 선은 macro AUC **0.9004**(기존 라벨러 fable). 다만 총점보다 소견별을 봐라 —
기존 라벨러는 ACL 0.996, MCL 0.983 으로 위쪽엔 남은 여지가 없고, 약한 쪽은
**Synovitis 0.751, Lateral OA 0.829, Effusion 0.847** 이다. 여유는 전부 거기에 있다.
