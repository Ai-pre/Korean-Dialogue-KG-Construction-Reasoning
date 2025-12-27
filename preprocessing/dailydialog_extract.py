import os
import json
import time
from openai import OpenAI


# =========================
# 1) EVENT PROMPT (V3)
# =========================
EVENT_PROMPT = """
You are an advanced Korean SNS dialogue event extractor.

GOAL:
Extract key events from the conversation. Each event must contain:
- a natural event_sentence (speaker identity implicit, using expressions like “한 참여자는”, “다른 친구는”, “상대방은”)
- a causal explanation referencing previous dialogue context.

EVENT SENTENCE RULES:
- Never use “발화자”.
- Use natural and varied Korean expressions for the actor.
- Must summarize the action or intention clearly.

CAUSE RULES:
- Must incorporate conversational context.
- Avoid generic or shallow reasons.
- Provide meaningful human-level reasoning.

OUTPUT FORMAT:
{
  "events": [
    {
      "id": "E1",
      "event_sentence": "...",
      "event_cause": "..."
    }
  ]
}

STRICT:
- JSON only.
"""

# =========================
# 2) TRIPLE PROMPT (V3)
# =========================
TRIPLE_PROMPT_V4 = """
You are an advanced Korean commonsense reasoning model following the ATOMIC framework.
Your job is to generate 9 high-quality ATOMIC relations for the given event.

INPUT EVENT:
- event_sentence: 자연스러운 사건 설명
- event_cause: 사건이 발생한 이유 또는 대화 흐름

IMPORTANT RELATION RULES (STRICT):

1) xIntent  
- 행위자가 왜 그런 행동/말을 했는지 목적 또는 숨겨진 이유  
- 감정 금지  
- 1문장

2) xNeed  
- 사건이 일어나기 *전에* 충족되어야 했던 조건  
- 사건 내용 반복 금지  
- 감정/욕구 금지  
- 1문장

3) xAttr  
- 사건으로부터 추론되는 행위자의 성격/특성  
- 1문장

4) xEffect  
- 사건 이후 행위자에게 일어나는 상태 변화 (감정 금지)  
- “무엇을 하게 된다”, “상태가 어떻게 변한다"와 같은 형태  
- 감정이 포함되면 안 됨  
- 1문장

5) xReact  
- 행위자의 감정만 표현  
- "기뻤다 / 불안했다 / 민망함을 느꼈다" 등  
- 감정 외 요소 금지  
- 1문장

6) xWant  
- 사건 직후 행위자가 원하는 것  
- 1문장

7) oEffect  
- 주변 사람들이 사건으로 인해 겪는 상태 변화  
- 감정 금지  
- 1문장

8) oReact  
- 주변 사람들이 느끼는 감정  
- 1문장

9) oWant  
- 주변 사람들이 통상적으로 바라게 되는 후속 행동/상태  
- 1문장

OUTPUT FORMAT (STRICT JSON):
{
  "triples": [
    {"relation": "xIntent", "tail": "...", "event_id": "E1", "head": "..."},
    ...
  ]
}

REQUIREMENTS:
- 모든 tail은 자연스러운 한국어 ‘완전한 문장’이어야 함.
- 감정이 허용된 relation(xReact, oReact) 외에는 감정 단어 사용 금지.
- xEffect와 oEffect는 반드시 actor vs others를 구분할 것.
- 반복, 모호한 단어, 단일명사 출력 금지.
"""



# ===========================================================
# 3) Extract N events (V3 optimized)
# ===========================================================
def extract_events(client, convo_text, n_events):
    prompt = EVENT_PROMPT + f"\nN = {n_events}\n===DIALOG===\n{convo_text}"

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0.1,
            )
            data = json.loads(resp.choices[0].message.content.strip())
            if "events" in data:
                return data
        except Exception as e:
            print(f"[Retry {attempt+1}] event error:", e)
            time.sleep(1)

    return {"events": []}


# ===========================================================
# 4) Generate triples for each event
# ===========================================================
def generate_triples(client, event):
    head = event["event_sentence"]
    cause = event["event_cause"]

    input_block = f"event_sentence: {head}\nevent_cause: {cause}"

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": TRIPLE_PROMPT_V4 + "\n" + input_block}],
                max_tokens=2048,
                temperature=0.1,
            )

            data = json.loads(resp.choices[0].message.content.strip())

            # Validate 9 triples exist
            if "triples" in data and len(data["triples"]) == 9:
                return data["triples"]

        except Exception as e:
            print("[Triple Retry]", e)
            time.sleep(1)

    print("⚠️ Triple fallback for:", head)
    return []



# ===========================================================
# 5) MAIN: 모든 파일 처리 + events.json / triples.json 따로 저장
# ===========================================================
def process_dataset(src_dir, out_event_json, out_triple_json, limit=1000):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    files = sorted(os.listdir(src_dir))[:limit]

    # ➤ 초기 빈 리스트 파일 생성 (JSON 배열 시작)
    with open(out_event_json, "w", encoding="utf-8") as f:
        f.write("[\n")

    with open(out_triple_json, "w", encoding="utf-8") as f:
        f.write("[\n")

    first_event = True
    first_triple = True

    total = len(files)

    for idx, fname in enumerate(files, start=1):
        if not fname.endswith(".txt"):
            continue

        print(f"[{idx}/{total}] Processing {fname} ...")

        path = os.path.join(src_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            convo = f.read()

        n_lines = len([l for l in convo.split("\n") if l.strip()])
        n = 3 if n_lines <= 5 else 4 if n_lines <= 12 else 5

        # ---------------------------
        # 1) Extract events
        # ---------------------------
        events_data = extract_events(client, convo, n)
        events_data["filename"] = fname

        # ---- events.json에 즉시 append ----
        with open(out_event_json, "a", encoding="utf-8") as f:
            if not first_event:
                f.write(",\n")
            f.write(json.dumps(events_data, ensure_ascii=False, indent=2))
            f.flush()

        first_event = False

        # ---------------------------
        # 2) Generate triples for each event
        # ---------------------------
        triple_list = []

        for ev in events_data["events"]:
            triples = generate_triples(client, ev)
            for t in triples:
                t["event_id"] = ev["id"]
                t["head"] = ev["event_sentence"]
                t["filename"] = fname
                triple_list.append(t)

        triple_block = {"filename": fname, "triples": triple_list}

        # ---- triples.json에 즉시 append ----
        with open(out_triple_json, "a", encoding="utf-8") as f:
            if not first_triple:
                f.write(",\n")
            f.write(json.dumps(triple_block, ensure_ascii=False, indent=2))
            f.flush()

        first_triple = False

    # ---------------------------
    # JSON 배열 닫기
    # ---------------------------
    with open(out_event_json, "a", encoding="utf-8") as f:
        f.write("\n]\n")

    with open(out_triple_json, "a", encoding="utf-8") as f:
        f.write("\n]\n")

    print("DONE:", out_event_json, out_triple_json)



def main():
    SRC_DIR = "/home/jaesang/kg_project/data/dialog/src"
    OUT_EVENTS = "/home/jaesang/kg_project/output/dialog_events.json"
    OUT_TRIPLES = "/home/jaesang/kg_project/output/dialog_triples.json"

    # 🔥 output 폴더 자동 생성
    os.makedirs(os.path.dirname(OUT_EVENTS), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_TRIPLES), exist_ok=True)

    process_dataset(
        src_dir=SRC_DIR,
        out_event_json=OUT_EVENTS,
        out_triple_json=OUT_TRIPLES,
        limit=1000
    )


if __name__ == "__main__":
    main()
