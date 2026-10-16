# Sol 5.6 (extra high) vs Fable — 투트랙 라벨 추출 검증

RSNA Knee 프로젝트, gold58(진짜 방사선의 라벨 있는 58 스터디)의 리포트를 Sol에게
두 가지 방식으로 먹여서 Fable보다 나은 teacher가 될 수 있는지 확인.

배경: 이미지 모델 앙상블 성능이 teacher(Fable, gold 대비 macro-AUC ~0.900)에
막혀있다는 게 확정됨(2026-08-26). teacher를 바꾸면 이 천장이 움직일 수 있음 —
단, 58개로는 AUC 갭 ~0.04는 돼야 유의미(compare.py 참고). Fable은 재기동 불가라
Sol로 투트랙 진행.

## Track A — 블라인드 재추출 (`prompt_blind.txt`)
Fable 결과를 보여주지 않고 리포트만으로 완전 독립 라벨링. 앵커링 없는 가장
깨끗한 비교. 입력: `gold58_reports.json` (`{id, report}`).
출력 지정 경로: `labels_gold58_sol.csv`.

## Track B — 비판적 교정 (`prompt_critique.txt`)
리포트 + Fable 예측(`gold58_with_fable.json`, `{id, report, fable_prediction}`)을
같이 주고, **먼저 자기 판단을 세운 뒤에** Fable 값과 비교해서 틀린 곳만 고치도록
지시. Fable의 구체적 실패 패턴(별 근거 없이 특정 finding에 확신 과다/과소 등)을
잡아낼 수 있음. 0.15 이상 차이나면 `flagged=1`로 표시하게 해뒀음 — 어디서
갈라지는지 나중에 보기 쉽게.

**주의(앵커링 리스크)**: `src/fable_prompt.py`에 이미 기록된 실패 사례 참고 —
Fable 자체도 "예시로 준 prior 숫자"에 끌려간 전례가 있음(ACL 0.41 주입 →
0.29-0.40으로 수렴, 진짜 corpus 비율은 ~0.20). Track B는 같은 함정에 빠질 수
있으니, **Track A(블라인드)가 주 지표, Track B는 진단/보정 참고용**으로 취급할 것.
두 트랙 결과가 갈리면 Track A를 신뢰.

## 공통 파일
- `gold58_truth.csv` — 채점용 정답(경쟁 train.csv의 실제 라벨 컬럼, Fable/Sol/Qwen
  누구의 출력도 아님).
- `pseudo_fable_gold.csv` — Fable이 이 58개에 낸 예측(gold58 macro-AUC 0.9004,
  Track B의 입력이기도 함).
- `pseudo_student_gold.csv`/`.parquet` — **실제 배포된 teacher**(Fable의 1,000개
  라벨로 LoRA 증류한 Qwen3.5-9B student, `pseudo_student_all.parquet`이 진짜
  이미지 모델 학습에 쓰이는 4,407개 전량 라벨의 gold58 서브셋)의 예측.
  gold58 macro-AUC 0.8993 — Fable과 사실상 동일(증류 성공 확인됨). 이미지
  모델 앙상블이 막혀있는 진짜 천장은 이 숫자임.
- `compare.py` — 두 teacher를 gold58로 paired bootstrap 비교하는 스크립트.
  `uv run python compare.py pseudo_student_gold.csv <sol_output>.csv --names student sol`
  (fable과 비교하려면 `pseudo_fable_gold.csv`로 교체 — 둘 다 0.900 근처라
  결과는 거의 같을 것)
- `labels.py` — LABELS/CRITERIA 정의 원본(프롬프트가 참조하는 정의 소스).

## 다음 단계 (맥 쪽에서)
1. Track A: `prompt_blind.txt` + `gold58_reports.json` → Sol.
2. Track B: `prompt_critique.txt` + `gold58_with_fable.json` → Sol.
3. 각 출력 CSV를 `id,{columns}[,flagged]` 형식으로 저장 (컬럼 순서는
   `labels.py`의 LABELS).
4. `compare.py`로 fable vs Track A, fable vs Track B 각각 paired 비교.
   macro-AUC 갭 ~0.04 이상이어야 58개 샘플로 유의미(compare.py 상단 룰).
