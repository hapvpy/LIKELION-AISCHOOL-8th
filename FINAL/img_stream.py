# 패키지 import
import streamlit as st
import pandas as pd
import random
# 사진출력 패키지 import
import io
from PIL import Image
import requests


# 데이터 로드
df = pd.read_csv("ziwon.csv")

# 스트림릿 페이지 제목 설정하기
st.title('레시피 추천 챗봇')
st.write('메뉴명, 범주, 요리난이도, 소요시간, 재료 정보를 확인하실 수 있습니다.')

# 챗봇 대화 기록을 저장할 리스트 선언
conversation = []

# 이전 대화 내용 출력하기
for message in reversed(conversation):
    if message["is_user"]:
        st.text_input("User input", value=message["text"], key=message["time"], disabled=True)
    else:
        st.write(message["text"])

# 알레르기 유발식품 입력
allergy = st.text_input('알레르기 식품을 입력하세요. (띄어쓰기로 구분해주세요.)')

if allergy:
    # 알레르기 식품을 리스트로 변환
    allergy_list = allergy.split(',')
    # 알레르기 식품이 포함되지 않은 레시피 필터링 (str.contains는 리스트 안받아서 리스트컴프리헨션으로 변경)
    df1 = df[~df['재료'].str.contains('|'.join([f"^{x}\s|\s{x}\s|\s{x}$" for x in allergy_list]))]
    
    ingredients = st.text_input('가지고 있는 재료를 입력하세요.  (띄어쓰기로 구분해주세요.)', key='ingredients_input')

    if ingredients:
        # 재료를 리스트로 변환
        ingredient_list = ingredients.split()

        # 재료가 포함된 레시피 필터링
        df2 = df1[df1['재료'].apply(lambda x: all(item.lower() in x.lower() for item in ingredient_list))]

        level = st.radio('원하는 요리 난이도를 선택하세요.', ['초보자', '중급자', '고급자'])

        if level:
            # 선택된 요리 난이도에 맞는 레시피 필터링
            if level == '초보자':
                filtered_df = df2[df2['난이도'] == 1]
            elif level == '중급자':
                filtered_df = df2[df2['난이도'] == 2]
            else:
                filtered_df = df2[df2['난이도'] == 3]
            
            time = st.text_input('희망하는 최대 소요시간을 입력해주세요.')

            if time:
                # 입력값을 정수형으로 변환
                time = int(time)

                last_df = filtered_df[filtered_df['소요시간'] <= time]

                # 필터링된 레시피 출력
                if not last_df.empty:
                    # 랜덤한 인덱스를 선택하여 해당 행을 출력
                    random_index = random.randint(0, last_df.shape[0] - 1)
                    random_recipe = last_df.iloc[random_index]
                    
                    # 랜덤으로 선택된 행의 '사진' 컬럼 값 가져오기
                    img_url = random_recipe['사진']
                    # URL에서 이미지 불러오기
                    response = requests.get(img_url)
                    image_bytes = io.BytesIO(response.content)
                    image = Image.open(image_bytes)
                    # 이미지 출력하기
                    
                    st.image(image, width=500)

                    st.write('# 🧑‍🍳 ', random_recipe['요리'])

                    st.write(random_recipe)

                else:
                    st.write("검색 결과가 없습니다.")


