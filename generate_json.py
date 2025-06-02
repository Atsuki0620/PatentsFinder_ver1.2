import streamlit as st

st.set_page_config(page_title="PatentsFinder マルチ方針テスト", layout="wide")

# サイドバーでモード選択
mode = st.sidebar.radio("実行方針を選択してください", 
    ["方針１：ボタン＋フラグ", 
     "方針２：プロンプト分岐", 
     "方針３：純チャット＋フォローアップ", 
     "方針４：タブ／ステッパーUI"])

# 共通の OpenAI API Key 入力
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.warning("まずは API キーを入力してください。")
    st.stop()

# LangChain LLM インスタンスは共通で生成
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, temperature=0.2)

# **方針１：ボタン＋状態フラグでステップ分割**
if mode == "方針１：ボタン＋フラグ":
    st.header("【方針1】ボタン＋州フラグによるステップ分割")
    #  ここに suggest_technologies(), generate_ipc_candidates(), finalize_json() などの
    #  フラグ管理付き関数を実装するブロックを入れる
    #  例：
    if "step1_done" not in st.session_state:
        st.session_state.step1_done = False

    # ステップ1：キーワード入力 + IPC候補提案
    if not st.session_state.step1_done:
        kw = st.text_input("ステップ1：調査したい技術領域やキーワードを入力")
        if st.button("IPC候補を提案"):
            # ここで IPC 候補を LLM から取得し、セッションに保存
            ipc_list = ["C02F 1/00", "G06N 3/04", "G01N 33/569"]  # 例として固定値
            st.session_state.ipc_candidates = ipc_list
            st.write(f"提案された IPC 候補: {ipc_list}")
            st.session_state.step1_done = True
            st.experimental_rerun()
    else:
        # ステップ2：countries/assignees/publication_from をまとめて入力
        cols = st.columns(3)
        with cols[0]:
            countries = st.text_input("対象国コード (カンマ区切り、例: JP,US,EP)")
        with cols[1]:
            assignees = st.text_input("出願人 (カンマ区切り、例: Panasonic, Sony)")
        with cols[2]:
            pub_from = st.text_input("公開日下限 (YYYY-MM-DD)")
        if st.button("JSON生成"):
            # ここで IPC 候補＋追加情報 をもとに最終 JSON を作成
            result = {
                "ipc_codes": st.session_state.ipc_candidates,
                "countries": [c.strip() for c in countries.split(",")],
                "assignees": [a.strip() for a in assignees.split(",")],
                "publication_from": pub_from
            }
            st.code(result, language="json")  # JSON をそのまま表示

# **方針２：プロンプト内分岐でマルチターンを１つの関数にまとめる**
elif mode == "方針２：プロンプト分岐":
    st.header("【方針2】プロンプト内マルチターン分岐で一括制御")
    #  ここでは、たとえば「generate_ipc_workflow()」のような関数を使い、
    #  st.session_state の内容次第でプロンプトを切り替える
    #  ユーザー入力フォームはひとつだけでOK
    user_input = st.text_input("調査したい技術領域や情報を入力してください")
    if st.button("実行"):
        # 以下はあくまでイメージ
        if "ipc_candidates" not in st.session_state:
            # プロンプトを「IPC候補を提案するモード」で呼び出す
            prompt = f"【候補モード】{user_input}"
            # ここで LLM を呼び、st.session_state.ipc_candidates に格納
            st.session_state.ipc_candidates = ["C02F 1/00", "G06N 3/04", "G01N 33/569"]
            st.write("IPC 候補を提案しました:", st.session_state.ipc_candidates)
        else:
            # 追加情報（countries/assignees/pub_from）を含む JSON 生成モード
            prompt = f"【JSON生成モード】IPC候補:{st.session_state.ipc_candidates}, 情報:{user_input}"
            # LLMを呼んで最終JSONを生成
            result_json = {
                "ipc_codes": st.session_state.ipc_candidates,
                "countries": ["JP", "US"],        # 仮の例
                "assignees": ["Panasonic"],       # 仮の例
                "publication_from": "2021-01-01"  # 仮の例
            }
            st.code(result_json, language="json")

# **方針３：純チャット対話フロー＋フォローアップ関数**
elif mode == "方針３：純チャット＋フォローアップ":
    st.header("【方針3】チャット対話＋フォローアップ関数")
    # こちらは st.chat_input() を使いながら、必要に応じて
    # 「追加情報をください」と LLM が質問し、その後で finalize_json() を呼ぶ流れ
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # チャット入力
    msg = st.text_input("チャットで調査を進める場合はここに入力")
    if st.button("送信"):
        # ユーザー発言を履歴に追加
        st.session_state.chat_history.append({"role": "user", "content": msg})
        # ここで LLM に会話履歴を渡して応答を得る
        # もし「追加情報をください」系のトリガーであれば、その後 finalize_json() を呼ぶ
        # 例として直接 JSON を出すケースを示す
        if "countries" in msg.lower():
            # countries入力と判定した仮の例
            # finalize_json() を実行
            final = {
                "ipc_codes": ["C02F 1/00", "G06N 3/04"],
                "countries": ["JP", "US"],
                "assignees": ["Sony"],
                "publication_from": "2021-01-01"
            }
            st.code(final, language="json")
        else:
            # まずは IPC候補を返す
            st.write("まずは関連 IPC 候補を提案します...")
            st.write("- C02F 1/00\n- G06N 3/04\n- G01N 33/569")

# **方針４：タブ／ステッパーUIでステージを分ける**
else:
    st.header("【方針4】タブ／ステッパーUIでステージを明確化")
    tab1, tab2, tab3 = st.tabs(["STEP1: キーワード入力", "STEP2: IPC候補提示", "STEP3: JSON生成"])
    with tab1:
        st.write("STEP1: 調査する技術キーワードを入力")
        kw = st.text_input("技術キーワードを入力してください")
        if st.button("STEP1実行"):
            # ここで「STEP2で IPC候補を表示」ためのフラグを立てる
            st.session_state.ipc_for_step2 = ["C02F 1/00", "G06N 3/04"]
            st.success("STEP1完了。STEP2タブに移動してください。")
    with tab2:
        st.write("STEP2: IPC候補を確認")
        if "ipc_for_step2" in st.session_state:
            st.write("以下が候補です:", st.session_state.ipc_for_step2)
        else:
            st.warning("まずは STEP1 を実行してください。")
        if st.button("STEP2実行"):
            st.session_state.step2_done = True
            st.success("STEP2完了。STEP3タブに移動してください。")
    with tab3:
        st.write("STEP3: 漏れなく countries, assignees, publication_from を入力")
        if "step2_done" in st.session_state:
            cols = st.columns(3)
            with cols[0]:
                countries = st.text_input("対象国コード (例: JP,US)")
            with cols[1]:
                assignees = st.text_input("出願人 (例: Panasonic)")
            with cols[2]:
                pub_from = st.text_input("公開日下限 (YYYY-MM-DD)")
            if st.button("STEP3実行"):
                result = {
                    "ipc_codes": st.session_state.ipc_for_step2,
                    "countries": [c.strip() for c in countries.split(",")],
                    "assignees": [a.strip() for a in assignees.split(",")],
                    "publication_from": pub_from
                }
                st.code(result, language="json")
        else:
            st.warning("まずは STEP2 を完了してください。")
