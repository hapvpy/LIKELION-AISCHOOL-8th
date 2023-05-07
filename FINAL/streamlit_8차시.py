# 사진 옆 재료버튼 -> 링크연결 (챗봇형식0)


## 라이브러리 임포트
import streamlit as st 

import requests
from bs4 import BeautifulSoup

import pandas as pd
import numpy as np
import copy
import json

from ast import literal_eval

import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from transformers import pipeline
from transformers import GPT2TokenizerFast
from PIL import Image
from tqdm import tqdm

import pickle
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

import sklearn.datasets as datasets
import sklearn.manifold as manifold

import openai
import os
import io
import sys
from dotenv import load_dotenv
from streamlit_chat import message



## 파일 및 API 가져오기
# app.py 파일이 위치한 경로 가져오기
APP_DIR = os.path.dirname(os.path.abspath(__file__))
# app.py에서 만든 데이터프레임 불러오기
LAST_DF_PATH = os.path.join(APP_DIR, '..', 'last_df.pkl')
df = pd.read_pickle(LAST_DF_PATH)
df['feature'] = df['재료']
openai.api_key = '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'

cur_os = sys.platform



## 모델 선언
# mps_device = torch.device('mps')
model = SentenceTransformer('BM-K/KoSimCSE-roberta-multitask')



## 상위 5개 항목 출력(링크로 중복제거-민규님)
def get_query_sim_top_k(query, model, df):
    "쿼리와 데이터 간의 코사인 유사도를 측정하고 유사한 순위 5개 반환"
    query_encode = model.encode(query)
    cos_scores = util.pytorch_cos_sim(query_encode, df['ko-sroberta-multitask-feature6'])[0]
    top_results = torch.topk(cos_scores, k=1)
    return top_results

query = '고기 쪽파'
top_result = get_query_sim_top_k(query, model, df)


# df.iloc[top_result[1].numpy(), :][['요리', '종류', '재료']]



## 메세지 프롬프트 생성
# intent에서 의도 파악하고 recom 혹은 desc 로 판단되는 것.
msg_prompt = {
    'recom' : {
                'system' : "당신은 사용자의 질문에 따라 레시피를 추천하는 유용한 도우미입니다.", 
                'user' : "사용자에게 레시피를 추천할 때, '👨🏻‍🍳 그럼요!'으로 시작하는 간단한 인사말 1문장을 작성하세요.", 
              },
    'desc' : {
                'system' : "You are a helpful assistant who kindly answers.", 
                'user' : "Please write a simple greeting starting with 'of course' to explain the recipes to the user.", 
              },
    'intent' : {
                'system' : "You are a helpful assistant who understands the intent of the user's question.",
                'user' : "Which category does the sentence below belong to: 'description', 'recommended', 'search'? Show only categories. \n context:"
                }
}

user_msg_history = []



## OpenAI API와 GPT-3 모델을 사용하여 msg에 대한 응답 생성
# 이전 대화내용을 고려하여 대화 생성.
def get_chatgpt_msg(msg):
    completion = openai.ChatCompletion.create(
                    model='gpt-3.5-turbo',
                    messages=msg
                    )
    return completion['choices'][0]['message']['content']



## intent와 사용자 쿼리를 바탕으로 prompt 생성
# 적절한 초기 메세지 생성, 사용자와의 자연스러운 대화구성 가능.
def set_prompt(intent, query, msg_prompt_init, model):
    '''prompt 형태를 만들어주는 함수'''
    m = dict()
    # 검색 또는 추천이면
    if ('recom' in intent) or ('search' in intent):
        msg = msg_prompt_init['recom'] # 시스템 메세지를 가지고오고
    # 설명문이면
    elif 'desc' in intent:
        msg = msg_prompt_init['desc'] # 시스템 메세지를 가지고오고
    # intent 파악
    else:
        msg = msg_prompt_init['intent']
        msg['user'] += f' {query} \n A:'
    for k, v in msg.items():
        m['role'], m['content'] = k, v
    return [m]



## 입력된 텍스트에 대해 gpt 모델을 사용하여 응답 생성.
# 함수 내부에서 입력된 텍스트를 토큰화하고, 생성된 응답을 디코딩 하여 반환.
# 입력 텍스트에 대해서만 모델을 사용하여 응답 생성(이전 대화 고려 x)
def generate_answer(model, tokenizer, input_text, max_len=50):
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    # generate text until the specified length
    output = model.generate(input_ids=input_ids, max_length=max_len, do_sample=True, top_p=0.92, top_k=50)
    # decode the generated text
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response


import webbrowser

def user_interact(query, model, msg_prompt_init):
    # 1. 사용자의 의도를 파악
    user_intent = set_prompt('intent', query, msg_prompt_init, None)
    user_intent = get_chatgpt_msg(user_intent).lower()
    print("user_intent : ", user_intent)
    
    # 2. 사용자의 쿼리에 따라 prompt 생성    
    intent_data = set_prompt(user_intent, query, msg_prompt_init, model)
    intent_data_msg = get_chatgpt_msg(intent_data).replace("\n", "").strip()
    print("intent_data_msg : ", intent_data_msg)
    
# 3-1. 추천 또는 검색이면
    if ('recom' in user_intent) or ('search' in user_intent):
        recom_msg = str()
        # 기존에 메세지가 있으면 쿼리로 대체
        if (len(user_msg_history) > 0 ) and (user_msg_history[-1]['role'] == 'assistant'):
            query = user_msg_history[-1]['content']['feature']
        # 유사 아이템 가져오기
        #top_result = get_query_sim_top_k(query, model, movies_metadata, top_k=1 if 'recom' in user_intent else 3) # 추천 개수 설정하려면!
        top_result = get_query_sim_top_k(query, model, df)
        # 검색이면, 자기 자신의 컨텐츠는 제외
        top_index = top_result[1].numpy() if 'recom' in user_intent else top_result[1].numpy()[1:]
        # 장르, 제목, overview를 가져와서 출력
        r_set_d = df.iloc[top_index, :][['요리', '설명', '재료', '사진', '요리방법']]
        r_set_d = json.loads(r_set_d.to_json(orient="records"))
        for r in r_set_d:
            message(f" {intent_data_msg} \n{str(recom_msg)}")
            st.write("### 🍳추천메뉴: 【 " + r['요리'] + " 】")
            # '사진' 컬럼에서 이미지 불러오기
            img_url = r['사진']
            response = requests.get(img_url)
            image_bytes = io.BytesIO(response.content)
            image = Image.open(image_bytes)
            st.image(image, width=500)
            

            st.write("\n")

            # '설명' 출력
            message('👨‍🍳 간단하게 "' + r['요리'] + '" 소개를 해드릴게요. \n\n ' + r['설명'])
            st.write("\n")
            st.write("\n")
                            

            # 재료구매링크 출력
            message('재료가 궁금하신가요? \n 아래 재료를 클릭하면 구매 페이지로 넘어갑니다.')
                # st.markdown("<p style='color:blue; font-style: italic'> (재료를 클릭하면 구매 페이지로 넘어갑니다.)</p>", unsafe_allow_html=True)
            r_ingredients = r['재료'].split()
            button_html = ''
            for ing in r_ingredients:
                gs_url = f"https://m.gsfresh.com/shop/search/searchSect.gs?tq={ing}&mseq=S-11209-0301&keyword={ing}"
                button_html += f"""<a href="{gs_url}" target="_blank" style="text-decoration: none; color: white; background-color: #FF5733; padding: 6px 12px; border-radius: 5px; margin-right: 5px;">{ing}</a>"""
            st.markdown(button_html, unsafe_allow_html=True)



            # 레시피 토글로 출력
            with st.expander('##### 🍽️ 요리방법이 궁금하신가요?'):
                recipe_steps = r['요리방법'].replace('[', '').replace(']', '').replace("'", "").replace("\\r\\n", ' ').split(", ")
                for i, step in enumerate(recipe_steps):
                    step = step.strip()  
                    if step:  
                        st.write(f"{i+1}. {step}")



            
        user_msg_history.append({'role' : 'assistant', 'content' : f"{intent_data_msg} {str(recom_msg)}"})
       






        
    # 3-2. 설명이면
    elif 'desc' in user_intent:
        try:
            # 이전 메세지에 따라서 설명을 가져와야 하기 때문에 이전 메세지 컨텐츠를 가져옴
            prev_msg = next(filter(lambda x: x['role'] == 'user', reversed(user_msg_history)))
            top_result = get_query_sim_top_k(query, model, df)
            if len(top_result[1]) == 0:
                # 정확히 일치하는 결과가 없을 경우, 유사한 결과 중에서 가장 상위 1개를 가져옴
                top_result = get_query_sim_top_k(query, model, df, k=1)
            # feature가 상세 설명이라고 가정하고 해당 컬럼의 값을 가져옴
            desc = top_result[0]['overview']
            user_msg_history.append({'role' : 'assistant', 'content' : f"{intent_data_msg} {desc}"})
            return (f"\ndesc data : {intent_data_msg} \n{desc}\n")
        except (StopIteration, IndexError):
            user_msg_history.append({'role' : 'assistant', 'content' : "이전에 요청된 설명이 없습니다."})

    # 3-3. 추천 또는 검색이 아니면
    else:
        # 새로운 쿼리와 기존 대화를 모두 이용해 생성된 prompt에서 답변 생성
        answer = generate_answer(query, user_msg_history, model)
        user_msg_history.append({'role' : 'assistant', 'content' : answer})
        return answer



# query = "돼지고기와 양파가 들어간 음식을 추천해줘"
# user_interact(query, model, copy.deepcopy(msg_prompt))




# def generate_response(prompt, df=None, past=None):
#     if df is not None:
#         # 레시피 데이터를 이용한 prompt 생성
#         prompt = f"{prompt}\nRecipes:"
#         for i, row in df.iterrows():
#             prompt = f"{prompt}\n{i+1}. {row['title']}"
#     if past is not None:
#         # 이전 대화 기록을 이용한 prompt 생성
#         prompt = f"{prompt}\nPast messages:"
#         for msg in past:
#             prompt = f"{prompt}\n- {msg}"
    
#     completions = openai.Completion.create (
#         engine="text-davinci-003",
#         prompt=prompt,
#         max_tokens=1024,
#         stop=None,
#         temperature=0,
#         top_p=1,
#     )
#     message = completions["choices"][0]["text"].replace("\n", "")
#     return message







if __name__ == "__main__":
    st.title('🤖 레시피 추천 챗봇, [레챗!]')
    st.write("""
        가지고 있는 재료를 입력하거나 궁금한 레시피를 "🤖레쳇"에게 물어보세요!
        """)
    st.write('\n')
    
    # 챗봇 생성하기
    if not hasattr(st.session_state, 'generated'):
        st.session_state.generated = []

    if not hasattr(st.session_state, 'past'):
        st.session_state.past = []

    


    # 쿼리 변형하기
    query = None
    with st.form(key='my_form'):
        query = st.text_input(f"Your Query:")
        submitted = st.form_submit_button('Send')



    # 수행문 만들기
    if submitted and query:
        output = user_interact(query, model, msg_prompt)
        st.session_state.past.append(query)
        st.session_state.generated.append(output)


        
    # 출력하기
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')

