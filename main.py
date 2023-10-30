import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.llms import CTransformers
import os
import OpenDartReader

import pandas as pd
from streamlit_extras.buy_me_a_coffee import button
import requests
import time
import yfinance as yf
from prophet import Prophet
from PIL import Image

def main():

    # while True:  # 무한 루프 시작
        # fabicon title Wide , normal  layout = 'wide',
        st.set_page_config(page_title='VendorReport', page_icon = 'BuyBeer_128.png', initial_sidebar_state = 'auto')
        button(username="damuljumong", floating=True, width=221)
        image = Image.open("BuyBeer_128.png")
        # Buy Me a Beer HTML 코드
        tossme_button = """        
        <div style="position: absolute; top: 1px; right: 5px;">
            <a href="https://toss.me/damulcandy" target="_blank">
                <img src="https://i.seadn.io/gae/GVOLxPFn_9SPxiCfgDWsHa0fzSqYQK2IFHVifxeXmCrA6o5srQw1fbEOzeLpCyjzU0svPxKA2NI_q0CYTJAiuECdteYwmncyIyVm?auto=format&dpr=1&w=10" alt="insert coin with toss" style="height: auto !important;width: auto !important;" >            
            </a>
        </div>
        """
        # Streamlit 앱에 HTML 삽입
        st.write(tossme_button, unsafe_allow_html=True)
    
        chat_model = ChatOpenAI()

        # api_key = 'API_KEY_DART'
        var_name = "API_KEY_DART"
        api_key = st.secrets.get(var_name, None)
        #api_key = os.environ.get(var_name, None)
        dart = OpenDartReader(api_key) 
    
        st.title('해외 업체 경영현황 보고서')
        stock_codes_input = "AAPL,HPE,DOX" # 애플 , HPE, Amdocs 
        symbol = "AAPL"

        year = 2023
        vendorinfo_records = []

        stock_codes_input = st.text_input('Vendor Stock Symbol을 입력하세요. 예) 애플 Stock Symbol : AAPL',key='stock_codes_input_2')
        if stock_codes_input:
            stock_codes = [code.strip() for code in stock_codes_input.split(',')]
        if st.button('업체 정보 요청'):
            with st.spinner('업체 정보 리포트 작성 중...'):
                for symbol in stock_codes:
                    # Yahoo Finance에서 주식 정보를 가져옵니다.
                    stock = yf.Ticker(symbol)
                    time.sleep(3)
                    # 기업 정보를 출력합니다.
                    st.write("---")
                    #st.write("stock.info all:", stock)
                    #st.write("stock.info all:", stock.info)

                    st.write("Company Name:", stock.info["longName"])
                    st.write("CEO name:", stock.info["companyOfficers"][0]["name"])
                    st.write("No of full Time Employees:", stock.info["fullTimeEmployees"])
                    st.write("Symbol:", stock.info["symbol"])
                    st.write("Industry:", stock.info["industry"])
                    st.write("Sector:", stock.info["sector"])
                    st.write("Website:", stock.info["website"]) # previousClose

                    st.write("Description:", stock.info["longBusinessSummary"])

                    st.write("Current Price:", stock.history(period="1d")["Close"].iloc[0])
                    # st.write("Previous Close:", stock.history(period="1d")["Close"].iloc[0])
                    st.write("Previous Close:", stock.info["previousClose"])
                    st.write("Open Price:", stock.history(period="1d")["Open"].iloc[0])
                    st.write("Day's Range:", stock.history(period="1d")["Low"].iloc[0], "-", stock.history(period="1d")["High"].iloc[0])
                    st.write("52 Week Range:", stock.info["dayLow"], "-", stock.info["dayHigh"])
                    
                    # 일일 주식 가격 데이터를 가져옵니다.
                    daily_prices = stock.history(period="1d")
                    st.write("Daily Stock Prices:",daily_prices)

                    # 초기 스케일 팩터 설정
                    if 'scale_factor' not in st.session_state:
                        st.session_state.scale_factor = 1.0

                    # 월간 주식 가격 데이터를 가져옵니다.
                    monthly_prices = stock.history(period="1mo")
                    st.write("Monthly Stock Prices:",monthly_prices)             

                    # 스케일 팩터를 사용하여 그래프 스케일 조정
                    scale_factor = st.slider("그래프 스케일 조정", 0.1, 10.0, st.session_state.scale_factor)
                    st.session_state.scale_factor = scale_factor  # 스케일 팩터 업데이트

                    # 그래프 스케일 조정
                    st.line_chart(monthly_prices[["Open", "High", "Low", "Close"]])
                    data = monthly_prices
                    data = data.reset_index()
                    # 'Date' 열에서 시간대 정보 제거
                    data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)
                    data ["ds"] = data["Date"]
                    data ["y"] = data["Close"]                            
                    data = data [["ds","y"]]
                    st.write("monthly_prices Stock Prices:",data)
                    # model = Prophet()
                    # 주말을 제외하고 1주일간의 예측을 수행하는 Prophet 모델 생성
                    model = Prophet(weekly_seasonality=False)
                    model.add_seasonality(
                        name='custom_weekly',
                        period=7,  # 1주일 주기
                        fourier_order=3,  # 주기성을 고려할 정도 (조절 가능)
                        prior_scale=0.1  # 하이퍼파라미터 (조절 가능)
                    )
                    model.fit(data)
                    
                    # future = model.make_future_dataframe(periods=24, freq='H')
                    # 내일을 예측하려면 1일 (하루)의 미래 데이터프레임 생성
                    future = model.make_future_dataframe(periods=7, freq='D')
                    forecast = model.predict(future)
                    # st.write("Forecast for Tomorrow:", forecast)
                    fig1 = model.plot(forecast)
                    # 실제 값 가져오기 (예제에서는 monthly_prices를 사용)
                    actual_data = data[['ds', 'y']]
                    # 실제 값 그래프에 추가
                    fig1.gca().plot(actual_data['ds'], actual_data['y'], 'r.', label='Actual')

                    # 그래프에 레전드 추가
                    fig1.gca().legend(["Prophet Forecast", "Actual"])

                    st.pyplot(fig1)
                    # 'ds'와 'yhat' 열의 이름 변경
                    forecast.rename(columns={'ds': 'Date', 'yhat': 'Close'}, inplace=True)

                    # 변경된 DataFrame 출력
                    st.write("Forecast for Tomorrow:", forecast[['Date', 'Close']].tail(4))

                    # 분기별 재무 정보 가져오기
                    quarterly_financials = stock.quarterly_financials
                    st.write("Quarterly Financial Statements:", quarterly_financials)

                    # Yahoo Finance에서 재무 정보를 가져옵니다.
                    financials = stock.financials
                    st.write("Financial Statements:",financials)
                    # P/E 비율 가져오기
                    pe_ratio = stock.info["trailingPE"]
                    st.write(f"P/E Ratio: {pe_ratio}")

                    # 배당 수익률 가져오기
                    dividend_yield = stock.info["trailingAnnualDividendYield"]
                    st.write(f"Dividend Yield: {dividend_yield * 100}%")

                    # # 주식의 52주 범위 가져오기
                    # fifty_two_week_range = stock.info["fiftyTwoWeekRange"]
                    # print(f"52-Week Range: {fifty_two_week_range}")

                    # 시가총액 가져오기
                    market_cap = stock.info["marketCap"]
                    st.write(f"Market Cap: ${market_cap / 10**9}B")

                    # 전일 종가 가져오기
                    previous_close = stock.history(period="1d")["Close"].values[0]
                    st.write(f"Previous Close: ${previous_close}")                    
#=================
        st.title('국내 업체 경영현황 보고서')

        stock_codes_input = "005930,072130,078000,069410" # 삼성전자 , 유엔젤, 텔코웨어 엔텔스 
        symbol = "005930"

        year = 2023
        financial_records = []
        vendorinfo_records = []

        stock_codes_input = st.text_input('업체 Stock code를 입력하세요. 예 삼성전자 Stock code 005930,072130,078000,069410',key='stock_codes_input_1')
        if stock_codes_input:
            stock_codes = [code.strip() for code in stock_codes_input.split(',')]

        content = st.text_input('인공지능이 분석할 업체명을 입력하세요.')

        if st.button('업체 분석 요청'):
            with st.spinner('업체 리포트 작성 중...'):
                for symbol in stock_codes:
                    # 회사명에  포함된 회사들에 대한 개황정보
                    vendorInfo = dart.company(symbol)
                    # excel_filename = f'./files/{symbol}_vendorInfo.xlsx'
                    # vendorInfo.to_excel(excel_filename, index=False)             
                    PeopleInfo = dart.report(symbol, '직원', year - 1)
                    # excel_filename = f'./files/{symbol}_PeopleInfo.xlsx'
                    # PeopleInfo.to_excel(excel_filename, index=False)     

                    fnInfo = dart.finstate(symbol, year -1,reprt_code ='11011') 
                    # excel_filename = f'./files/{symbol}_fnInfo.xlsx'
                    # fnInfo.to_excel(excel_filename, index=False) 
                    fnInfo_1Q = dart.finstate(symbol, year,reprt_code ='11013') # 1 분기
                    # excel_filename = f'./files/{symbol}_fnInfo_1Q.xlsx'
                    # fnInfo_1Q.to_excel(excel_filename, index=False)  
                    fnInfo_2Q = dart.finstate(symbol, year,reprt_code ='11012') # 2 분기
                    # excel_filename = f'./files/{symbol}_fnInfo_2Q.xlsx'
                    # fnInfo_2Q.to_excel(excel_filename, index=False)  
                    fnInfo_3Q = dart.finstate(symbol, year,reprt_code ='11014') # 3 분기
                    # excel_filename = f'./files/{symbol}_fnInfo_3Q.xlsx'
                    # fnInfo_3Q.to_excel(excel_filename, index=False)

                    # 선택할 행과 열 인덱스
                    # selected_rows = [0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

                    # # fnInfo 데이터프레임에서 선택된 행과 열 선택
                    # selected_vendorInfo = vendorInfo.iloc[selected_rows, [4, 9, 18, 15, 12]]
                    # selected_vendorInfo.columns = ['회사명', '회계항목', '20년', '21년', '22년']  # Rename columns

                    # # fnInfo_1Q, fnInfo_2Q, fnInfo_3Q에서 선택된 행과 열 선택
                    # selected_PeopleInfo = PeopleInfo.iloc[selected_rows, [12]]
                    # selected_PeopleInfo.columns = ['23년 1Q']  # Rename columns

                    previous_corp_name = None
                    previous_ceo_nm = None
                    previous_adres = None
                    previous_est_dt = None                

                    # sm, fo_bbm, sexdstn 값을 리스트로 저장
                    fo_bbm_list = PeopleInfo['fo_bbm']                    
                    sexdstn_list = PeopleInfo['sexdstn']
                    sm_list = PeopleInfo['sm']
                    for fo_bbm, sexdstn, sm in zip(                    
                        fo_bbm_list, 
                        sexdstn_list,
                        sm_list         
                    ):
                        corp_name = vendorInfo['corp_name']
                        ceo_nm = vendorInfo['ceo_nm']
                        adres = vendorInfo['adres']
                        est_dt = vendorInfo['est_dt']
                        fo_bbm_list = PeopleInfo['fo_bbm']                    
                        sexdstn_list = PeopleInfo['sexdstn']
                        sm_list = PeopleInfo['sm']
                        # corp_name이 이전과 동일한 경우에만 추가
                        if ( corp_name != previous_corp_name or ceo_nm != previous_ceo_nm or adres != previous_adres or est_dt != previous_est_dt ):
                            vendorinfo_records.append({
                                'corp_name': corp_name,
                                'ceo_nm': ceo_nm,
                                'adres': adres,
                                'est_dt': est_dt,
                                'Business': fo_bbm,
                                'sex': sexdstn,
                                'employees': sm     
                            })
                        else :
                            vendorinfo_records.append({
                                'Business': fo_bbm,
                                'sex': sexdstn,
                                'employees': sm     
                            })
                        # 이전 corp_name 업데이트
                        previous_corp_name = corp_name
                        previous_ceo_nm = ceo_nm
                        previous_adres = adres
                        previous_est_dt = est_dt                


                    # st.dataframe(vendorInfo,width=800)
                    # st.dataframe(PeopleInfo,width=800)
                    # 기존의 리스트를 데이터프레임으로 변환
                    vendorinfo_df = pd.DataFrame(vendorinfo_records)
                    # 열 이름을 변경
                    vendorinfo_df.columns = ['회사명', '대표', '주소', '설립일', '사업부', '성별', '종업원수']        
                    st.dataframe(vendorinfo_df,width=800)
                    vendorinfo_records = []

                    # 선택할 행과 열 인덱스
                    selected_rows = [0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

                    # fnInfo 데이터프레임에서 선택된 행과 열 선택
                    selected_fnInfo = fnInfo.iloc[selected_rows, [4, 9, 18, 15, 12]]
                    selected_fnInfo.columns = ['회사명', '회계항목', '20년', '21년', '22년']  # Rename columns

                    # fnInfo_1Q, fnInfo_2Q, fnInfo_3Q에서 선택된 행과 열 선택
                    selected_fnInfo_1Q = fnInfo_1Q.iloc[selected_rows, [12]]
                    selected_fnInfo_1Q.columns = ['23년 1Q']  # Rename columns

                    selected_fnInfo_2Q = fnInfo_2Q.iloc[selected_rows, [12]]
                    selected_fnInfo_2Q.columns = ['23년 2Q']  # Rename columns

                    # fnInfo_3Q 데이터프레임을 위한 초기화
                    selected_fnInfo_3Q = None

                    # fnInfo_3Q가 비어있지 않은 경우에만 처리
                    if not fnInfo_3Q.empty:
                        # 선택한 행의 인덱스 중에서 데이터프레임의 길이를 초과하는 인덱스를 필터링하여 출력
                        selected_rows = [index for index in selected_rows if index < len(fnInfo_3Q)]

                        # 선택한 행이 존재하는 경우에만 fnInfo_3Q에서 선택된 열 선택
                        if selected_rows:
                            selected_fnInfo_3Q = fnInfo_3Q.iloc[selected_rows, [12]]
                            selected_fnInfo_3Q.columns = ['23년 3Q']  # Rename columns

                    # 선택된 데이터프레임을 하나로 통합
                    combined_df = pd.concat([selected_fnInfo, selected_fnInfo_1Q, selected_fnInfo_2Q, selected_fnInfo_3Q], axis=1)

                    # 데이터프레임 출력
                    st.dataframe(combined_df, width=1200)

                    # Event 
                    # dart.event(corp, event, start=None, end=None)
                    # 조회가능한 주요사항 항목: 
                    # ['부도발생', '영업정지', '회생절차', '해산사유', '유상증자', '무상증자', '유무상증자', '감자', '관리절차개시', '소송', '해외상장결정', '해외상장폐지결정', '해외상장', '해외상장폐지', '전환사채발행', '신주인수권부사채발행', '교환사채발행', '관리절차중단', '조건부자본증권발행', '자산양수도', '타법인증권양도', '유형자산양도', '유형자산양수', '타법인증권양수', '영업양도', '영업양수', '자기주식취득신탁계약해지', '자기주식취득신탁계약체결', '자기주식처분', '자기주식취득', '주식교환', '회사분할합병', '회사분할', '회사합병', '사채권양수', '사채권양도결정']
                    vn_event =  ['부도발생', '영업정지', '회생절차', '해산사유', '유상증자', '무상증자', '유무상증자', '감자', '관리절차개시', '소송', '해외상장결정', '해외상장폐지결정', '해외상장', '해외상장폐지', '전환사채발행', '신주인수권부사채발행', '교환사채발행', '관리절차중단', '조건부자본증권발행', '자산양수도', '타법인증권양도', '유형자산양도', '유형자산양수', '타법인증권양수', '영업양도', '영업양수', '자기주식취득신탁계약해지', '자기주식취득신탁계약체결', '자기주식처분', '자기주식취득', '주식교환', '회사분할합병', '회사분할', '회사합병', '사채권양수', '사채권양도결정']
                    for each_event in vn_event:
                        issues = dart.event(symbol, each_event) #
                        if not issues.empty:
                            st.write(each_event)
                            st.dataframe(issues, width=1200)
            st.write("인공지능 ( Open AI )이 분석한 기업 정보를 알려드립니다")
            result = chat_model.predict(content + "을 분석해줘")    # OpenAI sknam

            st.write(result)

if __name__ == '__main__':
    main()
