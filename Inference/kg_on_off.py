import os
import torch
import time
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ==========================================
# 1. 환경 설정 및 모델 로드
# ==========================================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
MODEL_NAME = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
KG_FILE_PATH = "/home/jaesang/kg_project/output/graph_data.pkl" # KG 파일 경로

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device: {DEVICE}", flush=True)

# RTX 5000용 4-bit 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

print("[INFO] 모델 로드 중 (4-bit Quantization)...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config, 
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    print("[INFO] 모델 로드 완료!")
except Exception as e:
    print(f"[Critical Error] 모델 로드 실패: {e}")
    exit()

# ==========================================
# 2. KG 데이터 검색 함수 (Real KG)
# ==========================================
def load_real_kg(query):
    try:
        with open(KG_FILE_PATH, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"[Error] KG 파일({KG_FILE_PATH}) 로드 실패: {e}")
        return []

    # 역방향 매핑 (ID -> Text)
    id2node = {v: k for k, v in data['node2id'].items()}
    id2rel = {v: k for k, v in data['rel2id'].items()}
    
    # 관계명 한글 번역 (선택적)
    rel_map_kr = {
        'xIntent': '의도/이유', 'xNeed': '필요 조건', 'xEffect': '결과/영향',
        'xWant': '원하는 것', 'xReact': '감정 반응', 'xAttr': '특성',
        'oEffect': '타인에게 미치는 영향', 'oReact': '타인의 반응', 'oWant': '타인이 원하는 것'
    }

    # 질문 검색 (간단한 키워드 매칭)
    query_tokens = query.split()
    relevant_head_ids = []
    
    # 쿼리 내 단어가 포함된 노드 찾기
    for text, idx in data['node2id'].items():
        if any(token in text for token in query_tokens if len(token) > 1):
            relevant_head_ids.append(idx)

    if not relevant_head_ids:
        return []

    # 연결된 지식 추출
    edge_index = data['edge_index'].cpu()
    edge_type = data['edge_type'].cpu()
    found_facts = []
    
    for head_id in relevant_head_ids[:5]: # 상위 5개 노드만 탐색
        mask = (edge_index[0] == head_id)
        connected_tails = edge_index[1][mask]
        connected_rels = edge_type[mask]
        
        head_text = id2node[head_id]
        
        for t_id, r_id in zip(connected_tails, connected_rels):
            tail_text = id2node[t_id.item()]
            raw_rel = id2rel[r_id.item()]
            rel_kr = rel_map_kr.get(raw_rel, raw_rel)
            
            fact = f"- 상황: [{head_text}] -> {rel_kr}: [{tail_text}]"
            found_facts.append(fact)
            
            if len(found_facts) >= 10: break
        if len(found_facts) >= 10: break

    return found_facts

# ==========================================
# 3. 프롬프트 함수들
# ==========================================
def prompt_no_kg(query):
    return [
        {"role": "system", "content": "당신은 공감 능력이 뛰어난 심리 상담가입니다."},
        {"role": "user", "content": f"다음 발화를 듣고 화자의 감정과 상태를 추론해 주세요.\n\n발화: {query}"}
    ]

def prompt_with_kg(query, kg_facts):
    kg_text = "\n".join(kg_facts)
    return [
        {"role": "system", "content": "당신은 상식 지식(Commonsense Knowledge)을 활용해 심층 분석하는 심리 상담가입니다."},
        {"role": "user", "content": f"다음 상식 지식을 참고하여 화자의 속마음과 상황을 구체적으로 분석해 주세요.\n\n[참고 지식]\n{kg_text}\n\n[사용자 발화]\n{query}"}
    ]

# ==========================================
# 4. 생성 함수
# ==========================================
def generate(prompt):
    # 토크나이징
    if isinstance(prompt, list):
        inputs = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
    else:
        inputs = tokenizer(str(prompt), return_tensors="pt").to(model.device)
        
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.eos_token_id

    # 생성 파라미터
    gen_config = {
        "max_new_tokens": 512,  # 답변 길이 넉넉하게
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.2,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "early_stopping": True   # <--- 추가: 적당히 다 했으면 멈춰라
    }

    start_time = time.time()
    try:
        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_config)
    except Exception as e:
        return f"에러 발생: {e}"

    print(f"[Debug] 생성 소요 시간: {time.time() - start_time:.2f}초")
    
    new_tokens = outputs[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

# ==========================================
# 5. 메인 실행부
# ==========================================
if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # ----------------------------------------------------
    # [입력] 여기 질문을 바꿔가며 테스트 해보세요!
    # ----------------------------------------------------
    # 데이터셋에 '배가 아프다' 관련 내용이 있으므로 잘 될 겁니다.
    query = "배가 너무 아파서 조퇴하고 싶어." 
    
    print(f"\n[사용자 발화] {query}")
    print("=" * 60)

    # ----------------------------------------------------
    # Case 1: KG 없이 (No KG)
    # ----------------------------------------------------
    print("\n>>> [Case 1] KG 없이 생성 중 (일반 모드)...")
    prompt_off = prompt_no_kg(query)
    result_off = generate(prompt_off)
    print(f"\n[결과 (KG Off)]:\n{result_off}")
    print("-" * 60)

    # ----------------------------------------------------
    # Case 2: Real KG 포함 (With KG)
    # ----------------------------------------------------
    print("\n>>> [Case 2] Real KG 검색 및 생성 중 (지식 활용 모드)...")
    
    # 1. 지식 검색
    kg_facts = load_real_kg(query)
    
    print(f"[검색된 지식 개수]: {len(kg_facts)}")
    if kg_facts:
        print("[지식 예시 (상위 3개)]:")
        for f in kg_facts[:3]:
            print("  " + f)
    else:
        print("[주의] 관련된 지식을 찾지 못했습니다. (더미 데이터 없이 진행)")
        kg_facts = ["관련된 상식 정보 없음."]

    # 2. 생성
    prompt_on = prompt_with_kg(query, kg_facts)
    result_on = generate(prompt_on)
    print(f"\n[결과 (KG On)]:\n{result_on}")
    print("=" * 60)