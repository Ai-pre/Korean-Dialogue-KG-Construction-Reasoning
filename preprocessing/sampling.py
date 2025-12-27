import json
import random

# ==== 1) JSON 파일 로드 ====
path = "/home/jaesang/kg_project/output/dialog_triples.json"    # 파일명 맞게 수정
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

# ==== 2) 모든 triples 평탄화 ====
all_triples = []
for item in data:
    for tri in item["triples"]:
        # 필요한 필드만 추출해서 저장
        cleaned = {
            "head": tri["head"],
            "relation": tri["relation"],
            "tail": tri["tail"]
        }
        all_triples.append(cleaned)

print(f"총 트리플 수: {len(all_triples)}")

# ==== 3) 랜덤 100개 샘플링 ====
sample_size = 300
sampled = random.sample(all_triples, min(sample_size, len(all_triples)))

# ==== 4) 평가용 JSON으로 저장 ====
output_path = "/home/jaesang/kg_project/output/sampled_triples_100.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(sampled, f, indent=2, ensure_ascii=False)

print(f"완료! head/relation/tail만 포함된 300개 샘플 저장됨 → {output_path}")
