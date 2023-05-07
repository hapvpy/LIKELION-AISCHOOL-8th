# 토글로 정리된 페이지


# 패키지 import
import streamlit as st
import pandas as pd
import random
import time
import pickle
# 사진출력 패키지 import
import io
from PIL import Image
import requests


# 데이터 로드
# 파일 경로 설정
file_path = 'ko_sroberta_recipes.pkl'
# pkl 파일 불러오기
with open(file_path, 'rb') as f:
    df = pickle.load(f)

# 스트림릿 페이지 제목 설정하기
st.title('🤖 레시피 추천 챗봇, [레챗!]')
st.write('알레르기 식품, 요리 범주, 난이도, 소요시간을 선택하실 수 있습니다.')
st.write("\n")


## 알레르기

# 알레르기 항목 리스트
# allergies = ['난류', '갑각류', '우유', '견과류', '대두', '밀', '메밀', '땅콩', '쇠고기', '닭고기', '돼지고기', '생선', '조개류', '아황산류']
allergies = {
    '우유': ['우유', '치즈', '버터', '크림', '요거트', '아이스크림'],
    '난류': ['계란','달걀', '메렌지', '마요네즈'],
    '땅콩': ['땅콩', '피넛버터', '땅콩크림', '땅콩깨'],
    '견과류': ['아몬드', '땅콩','호두', '피스타치오', '브라질너트', '마카다미아너트', '잣'],
    '대두': ['대두', '콩', '미소', '순두부', '된장', '콩나물', '콩물', '두부','간장'],
    '밀': ['밀가루', '밀떡', '면류', '케이크', '쿠키', '파스타', '빵', '시리얼'],
    '갑각류': ['새우', '랍스타', '게', '대게', '꽃게', '홍합', '조개류'],
    '조개류': [ '굴', '홍합', '전복', '조개','소라'],
    '생선': ['고등어', '연어', '참치', '멸치', '광어', '붕어', '오징어', '문어'],
    '육류': ['돼지고기', '햄', '소시지', '베이컨', '삼겹살', '쇠고기'],
    '복숭아': ['복숭아', '자두', '망고', '모과', '사과', '배', '포도']
}
st.write("\n")
with st.expander('#### Q1. 알레르기가 있으신가요?'):
    # 체크박스로 선택받기
    st.write('##### Q1-1. 체크박스로 입력하기')
    
    cols = st.columns(2)
    selected_allergies = []
    for i, allergy in enumerate(allergies):
        if i % 2 == 0:
            checkbox_col = cols[0]
        else:
            checkbox_col = cols[1]
        selected = checkbox_col.checkbox(allergy, key=allergy)
        if selected:
            selected_allergies.append(allergy)

    # 기타 알레르기 입력 받기
    st.write("\n")
    st.write("\n")
    other_input = st.text_input('##### Q1-2. 직접 입력하기  ex) 복숭아, 수박 등', key='other_input')

    # 선택된 알레르기와 기타 알레르기 출력하기
    st.write('###### ⬇️ 선택하신 알레르기 항목')
    selected_allergies = [allergy for allergy in allergies if st.session_state.get(allergy)]
    if len(selected_allergies) == 0 and not other_input:
        st.write('알레르기가 없습니다.')
    else:
        allergy_list = ", ".join(selected_allergies)
        if other_input:
            allergy_list += ", " + other_input
        st.write(allergy_list)

    if any(selected_allergies) or other_input:
        # 선택된 알레르기와 입력받은 알레르기 가져오기
        selected_allergies = [allergy for allergy in allergies if st.session_state.get(allergy)]
        other_allergy = other_input.strip()

        # 포함하지 않는 데이터 추출하기
        # 알레르기 식품이 포함되지 않은 레시피 필터링
        # df_al = df[~df['재료'].str.contains('|'.join([f"^{x}\s|\s{x}\s|\s{x}$" for x in (selected_values + [other_allergy])]), regex=True)]
        tmp = df.copy()
        for a in selected_allergies:
            tmp = tmp.loc[~tmp['재료'].str.contains('|'.join(allergies[a]))]
        df_al = tmp.copy()
        # 기타 알러지 리스트 만들기
        other_allergies = [x.strip() for x in other_allergy.split(',') if x.strip()]
        # 기타 알러지가 포함된 데이터 제외하기
        for allergy in other_allergies:
            df_al = df_al[~df_al['재료'].str.contains(allergy)]
    else:
        df_al = df

with st.expander(" 알레르기 정보 확인하기"):
        st.markdown("<p style='color:red'> (일부 항목만 해당할 경우, 해당 항목을 직접 입력해주세요.)</p>", unsafe_allow_html=True)
        data = [
        ["체크 항목", "포함된 항목"],
        ['우유', '우유, 치즈, 버터, 크림, 요거트, 아이스크림'],
        ['난류', '계란, 달걀, 메렌지, 마요네즈'],
        ['땅콩', '땅콩, 피넛버터, 땅콩크림, 땅콩깨'],
        ['견과류', '아몬드, 땅콩, 호두, 피스타치오, 브라질너트, 마카다미아너트, 잣'],
        ['대두', '대두, 콩, 미소, 순두부, 된장, 콩나물, 콩물, 두부, 간장'],
        ['밀', '밀가루, 밀떡, 면류, 케이크, 쿠키, 파스타, 빵, 시리얼'],
        ['갑각류', '새우, 랍스타, 게, 대게, 꽃게, 홍합, 조개류'],
        ['생선', '고등어, 연어, 참치, 멸치, 광어, 붕어, 오징어, 문어'],
        ['육류', '돼지고기, 햄, 소시지, 베이컨, 삼겹살'],
        ['복숭아', '복숭아, 자두, 망고, 모과, 사과, 배, 포도']
    ]
        al_data = pd.DataFrame(data[1:], columns=data[0])
        st.write(al_data)



## 요리 범주 선택하기
st.write("\n")
st.write("\n")
st.write("\n")
menus = ['전체', '초대요리', '한식', '간식', '양식', '밑반찬', '채식', 
         '일식', '중식', '퓨전', '분식',    '안주', '베이킹', '다이어트', 
         '도시락', '키토', '오븐 요리', '메인요리', '간단요리']

with st.expander('#### Q2. 원하는 요리 범주가 있으신가요?'):
    cols = st.columns(4)
    selected_menus = []
    for i, menu in enumerate(menus):
        checkbox_col = cols[i % 4]
        selected = checkbox_col.checkbox(menu, key=menu)
        if selected:
            selected_menus.append(menu)

    if '전체' in selected_menus:
        # 모든 메뉴가 선택된 경우, 모든 레시피 데이터프레임 할당
        df_me = df_al.copy()
    else:
        # 선택된 메뉴 가져오기
        selected_menus = [menu for menu in selected_menus if menu != '전체']
        # 해당 종류가 포함된 레시피 필터링
        df_me = df_al[df_al['종류'].str.contains('|'.join(selected_menus))]
    



## 요리 난이도 선택
st.write("\n")
st.write("\n")
st.write("\n")
with st.expander('#### Q3. 원하는 요리 난이도가 있으신가요?'):
    levels = st.multiselect('',['초보자', '중급자', '고급자'])

    if levels:
        # 선택된 요리 난이도에 맞는 레시피 필터링
        filtered_df = df_me[df_me['난이도'].isin([1 if '초보자' in levels else 0,
                                                2 if '중급자' in levels else 0,
                                                3 if '고급자' in levels else 0])]
    else:
        # 체크박스에서 선택하지 않은 경우, 데이터프레임을 모두 출력
        filtered_df = df_me



# 희망 요리시간 입력
st.write("\n")   
st.write("\n")
st.write("\n")
with st.expander("#### Q4. 희망하는 요리시간이 있으신가요?"):
    time = st.text_input('희망하는 최대 소요시간을 입력해주세요. ex) 120 (분 단위 숫자로 입력)')
    last_df = filtered_df.copy()

    if time:
        # 입력값을 정수형으로 변환
        time = int(time)

        last_df = last_df[last_df['소요시간'] <= time]


st.write("\n")
st.write("\n")
# 저장 버튼 클릭 이벤트 핸들링
if st.button("저장"):
    with open('last_df.pkl', 'wb') as f:
        pickle.dump(last_df, f)
    st.write('저장되었습니다.')



