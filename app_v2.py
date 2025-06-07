# --------------------------------------------
# 1. 共通設定・インポート
# --------------------------------------------
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import re
import json
import pandas as pd
import numpy as np
from google.cloud import bigquery
from sklearn.metrics.pairwise import cosine_similarity
import os

# --- BigQuery/Embedding/類似度計算のための関数群 ---

# 設定（config.yamlの代替）
# BQ_PROJECT は認証後に上書きされる
BQ_PUBLIC_PROJECT = "patents-public-data"
BQ_DATASET = "patents"
BQ_TABLE = "publications"
BQ_LOCATION = "US"
BQ_LIMIT = 100
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")

# BigQueryから特許データを抽出
def search_patents_by_params(params: dict) -> pd.DataFrame:
    # 公開データセット参照用にBQ_PUBLIC_PROJECT, BQ_LOCATIONを利用
    client = bigquery.Client(project=BQ_PROJECT, credentials=GCP_CREDENTIALS, location=BQ_LOCATION)
    where = []
    if params.get("ipc_codes"):
        ipc_list = [f"'{c}'" for c in params["ipc_codes"]]
        where.append(f"ipc.code IN ({','.join(ipc_list)})")
    if params.get("countries"):
        if isinstance(params["countries"], list):
            countries = [f"'{c}'" for c in params["countries"]]
        else:
            countries = [f"'{params['countries']}'"]
        where.append(f"country_code IN ({','.join(countries)})")
    if params.get("assignees"):
        if isinstance(params["assignees"], list):
            assignees = [f"'{a}'" for a in params["assignees"]]
        else:
            assignees = [f"'{params['assignees']}'"]
        where.append(f"assignee IN ({','.join(assignees)})")
    if params.get("publication_from"):
        where.append(f"publication_date >= '{params['publication_from']}'")
    where_clause = " AND ".join(where) if where else "1=1"
    sql = f"""
        SELECT
            publication_number,
            (SELECT v.text FROM UNNEST(title_localized) AS v WHERE v.language='en' LIMIT 1) AS title,
            (SELECT v.text FROM UNNEST(abstract_localized) AS v WHERE v.language='en' LIMIT 1) AS abstract,
            publication_date,
            STRING_AGG(DISTINCT ipc.code, ',') AS ipc_codes,
            STRING_AGG(DISTINCT assignee_harmonized.name, ',') AS assignees
        FROM `{BQ_PUBLIC_PROJECT}.{BQ_DATASET}.{BQ_TABLE}` AS p
            LEFT JOIN UNNEST(p.ipc) AS ipc
            LEFT JOIN UNNEST(p.assignee_harmonized) AS assignee_harmonized
        WHERE {where_clause}
        GROUP BY publication_number, title, abstract, publication_date
        LIMIT {BQ_LIMIT}
    """
    df = client.query(sql).to_dataframe()
    return df

# 特許テキストをベクトル化（OpenAI API例）
def vectorize_texts(texts: list, openai_api_key: str) -> np.ndarray:
    import openai
    client = openai.OpenAI(api_key=openai_api_key)
    vectors = []
    for text in texts:
        resp = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
        vectors.append(resp.data[0].embedding)
    return np.array(vectors)

# クエリと特許ベクトルの類似度ランキング
def rank_by_similarity(query: str, patent_texts: list, openai_api_key: str) -> list:
    query_vec = vectorize_texts([query], openai_api_key)[0].reshape(1, -1)
    patent_vecs = vectorize_texts(patent_texts, openai_api_key)
    sims = cosine_similarity(query_vec, patent_vecs)[0]
    ranked_idx = np.argsort(sims)[::-1]
    return ranked_idx, sims

# --------------------------------------------
# 2. ページ設定・タイトル・説明
# --------------------------------------------
st.set_page_config(page_title="💡 PatentsFinder_ver1.2 チャット版", page_icon="🔍")
st.title("🔍 PatentsFinder_ver1.2 - チャットフロー実装 (方針3)")
st.write(
    """
    このチャット版アプリは、対話形式で IPC コードを提案し、
    その後ユーザーが「countries, assignees, publication_from」を入力すると
    自動的にJSON形式の検索パラメータを生成します。
    """
)

# --------------------------------------------
# 3. API キー入力
# --------------------------------------------
openai_api_key = st.text_input("OpenAI API Key", type="password")
openai_auth_ok = False
if openai_api_key:
    try:
        import openai
        openai.api_key = openai_api_key
        # v1.0.0以降の認証確認: embeddingエンドポイントでダミーリクエスト
        openai.embeddings.create(input="test", model="text-embedding-ada-002")
        st.success("OpenAI APIキーの認証に成功しました。")
        openai_auth_ok = True
    except Exception as e:
        st.error(f"OpenAI APIキーの認証に失敗しました: {e}")
else:
    st.info("まずは OpenAI API Key を入力してください。", icon="🗝️")
    st.stop()

gcp_json_str = st.text_area("Google Cloud サービスアカウントキー（JSONを貼り付け）", height=200)
gcp_auth_ok = False
if gcp_json_str:
    import io
    import json as _json
    try:
        gcp_info = _json.loads(gcp_json_str)
        from google.oauth2 import service_account
        GCP_CREDENTIALS = service_account.Credentials.from_service_account_info(gcp_info)
        # プロジェクトIDを取得
        BQ_PROJECT = gcp_info.get("project_id")
        # BigQueryクライアントで認証テスト
        from google.cloud import bigquery
        client = bigquery.Client(project=BQ_PROJECT, credentials=GCP_CREDENTIALS)
        client.query("SELECT 1").result()
        st.success("Google Cloud サービスアカウント認証に成功しました。")
        gcp_auth_ok = True
    except Exception as e:
        st.error(f"Google Cloud サービスアカウント認証に失敗しました: {e}")
else:
    st.info("BigQueryを利用するにはサービスアカウントキー（JSON）を貼り付けてください。", icon="🔑")
    st.stop()

if not (openai_auth_ok and gcp_auth_ok):
    st.stop()

# --------------------------------------------
# 4. LangChain の LLM インスタンス生成
# --------------------------------------------
llm = ChatOpenAI(
    model_name="gpt-4.1",
    openai_api_key=openai_api_key,
    temperature=0.2
)

# --------------------------------------------
# 5. セッションステート初期化
# --------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # 会話履歴
if "ipc_candidates" not in st.session_state:
    st.session_state.ipc_candidates = []  # 提案された IPC コード
if "expect_search_params" not in st.session_state:
    st.session_state.expect_search_params = False  # 検索パラメータ待ちフラグ
if "expect_tech_suggestion" not in st.session_state:
    st.session_state.expect_tech_suggestion = True  # 技術深掘り待ちフラグ
if "tech_suggested" not in st.session_state:
    st.session_state.tech_suggested = False  # 技術深掘り済みフラグ
if "ipc_codes" not in st.session_state:
    st.session_state.ipc_codes = []
if "countries" not in st.session_state:
    st.session_state.countries = []
if "assignees" not in st.session_state:
    st.session_state.assignees = []
if "publication_from" not in st.session_state:
    st.session_state.publication_from = ""
if "search_ready" not in st.session_state:
    st.session_state.search_ready = False

# --------------------------------------------
# 6. 関数定義: IPC コードを提案し、追加情報を促す質問をする
# --------------------------------------------
def generate_ipc_candidates(user_input: str):
    # 会話履歴にユーザー発言を追加
    st.session_state.messages.append({"role": "user", "content": user_input})

    # プロンプトを作成
    system_prompt = """
    あなたは「特許調査アシスタント」です。
    ユーザーが入力した技術領域やキーワードに基づき、関連性の高い IPC コードを 3～5 個程度、一言説明付きで提案してください。
    その後、次のように追加情報をリクエストしてください：
    「次に、対象国(countries)、出願人(assignees)、公開日下限(publication_from)を
    カンマ区切りで教えてください。例：JP, Sony, 2021-01-01」
    """
    lc_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ]

    # LLM コール
    response = llm(lc_messages)
    ai_content = response.content.strip()

    # AI 応答を会話履歴に追加・表示
    st.session_state.messages.append({"role": "assistant", "content": ai_content})
    with st.chat_message("assistant"):
        st.markdown(ai_content)

    # IPC コード部分を正規表現で抽出
    codes = re.findall(r"[A-Z]\d{2}[A-Z]?\s*\d{1,2}/\d{1,2}", ai_content)
    unique_codes = []
    for code in codes:
        code_clean = code.replace(" ", "")
        if code_clean not in unique_codes:
            unique_codes.append(code_clean)
    st.session_state.ipc_candidates = unique_codes
    st.session_state.ipc_codes = unique_codes  # IPCコードを検索用にもセット

    # 追加情報待ちフラグを立てる
    st.session_state.expect_search_params = True

# --------------------------------------------
# 7. 関数定義: 検索パラメータ用 JSON を生成する （LLM にパースを任せる版）
# --------------------------------------------
def finalize_search_parameters(user_input: str):
    """
    ユーザー入力（自由形式）を LLM に渡し、'countries', 'assignees', 'publication_from' を
    推測してもらい、さらに ipc_candidates と組み合わせて最終 JSON を生成・表示します。
    """
    # 会話履歴にユーザー発言を追加
    st.session_state.messages.append({"role": "user", "content": user_input})

    # ①：まず LLM に「パース用プロンプト」を生成
    parse_prompt = f"""
    次のユーザー入力を解析し、以下のキーを含む JSON オブジェクトを返してください。
    ・countries: 国コード (例: 日本→\"JP\", アメリカ→\"US\", 中国→\"CN\" 等)
    ・assignees: 出願人名 (そのまま文字列)
    ・publication_from: 公開日下限を \"YYYY-MM-DD\" 形式で指定 (例: \"2021年以降\"→\"2021-01-01\")
    余計な説明は一切不要で、必ず純粋に JSON オブジェクトだけを返してください。

    ユーザー入力:
    \"\"\"{user_input}\"\"\"
    """

    # LLM呼び出し（パース用）
    parse_response = llm([SystemMessage(content=parse_prompt)])
    try:
        # LLM の出力を JSON パースして dict に変換
        parsed = json.loads(parse_response.content.strip())
    except json.JSONDecodeError:
        # もし JSON 化に失敗したら、再度明確なフォーマットを要求
        follow_up = (
            "申し訳ありません。入力の形式がうまく解釈できませんでした。\n"
            "「countries, assignees, publication_from」をカンマ区切りで教えてください。"
            "例: JP, Sony, 2021-01-01"
        )
        st.session_state.messages.append({"role": "assistant", "content": follow_up})
        with st.chat_message("assistant"):
            st.markdown(follow_up)
        return

    # ②：JSON から各フィールドを取得
    countries = parsed.get("countries", [])
    assignees = parsed.get("assignees", [])
    publication_from = parsed.get("publication_from", "")

    # ③：既存の ipc_candidates と合成して最終出力を作る
    result = {
        "ipc_codes": st.session_state.ipc_codes,
        "countries": countries,
        "assignees": assignees,
        "publication_from": publication_from
    }
    json_result = json.dumps(result, ensure_ascii=False, indent=2)
    # ④：画面表示用メッセージ
    final_message = (
        "以下が最終的な検索条件です。\n"
        f"```json\n{json_result}"
    )
    st.session_state.messages.append({"role": "assistant", "content": final_message})
    with st.chat_message("assistant"):
        st.markdown(final_message)
    # パース待ちフラグをリセット
    st.session_state.expect_search_params = False
    # 検索条件が確定したのでsearch_readyをTrueに
    st.session_state.search_ready = True

# --------------------------------------------
# 8. 関数定義: 技術分野やサブトピックを提案する
# --------------------------------------------
def suggest_technologies(user_input: str):
    """
    ユーザー入力に対し、関連性の高い技術分野やサブトピックを自然文で 2～4 項目程度提案する。
    IPCコードには一切言及しない。
    """
    st.session_state.messages.append({"role": "user", "content": user_input})
    system_prompt = """
    あなたは「特許調査アシスタント」です。以下のルールに従って、ユーザーが入力した技術領域やキーワードに対し、
    関連性の高い技術分野やサブトピックを 2〜4 項、自然な日本語で提案してください。IPCコードや特許分類番号には言及せず、
    クリーンに技術的視点のみで回答します。

    例：ユーザーが「逆浸透膜の機械学習」と入力した場合、
     1. 逆浸透膜プロセスにおける運転条件最適化のための機械学習アルゴリズム開発
     2. センサー収集データを用いた膜汚染検知および予測技術
     3. AIを活用した膜製造工程での材料選定と品質管理
    """
    lc_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ]
    response = llm(lc_messages)
    ai_content = response.content.strip()
    st.session_state.messages.append({"role": "assistant", "content": ai_content})
    with st.chat_message("assistant"):
        st.markdown(ai_content)
    # 追加質問を促す
    follow_up = "上記の中で特に調査したい内容や、さらに具体的な技術テーマがあればご記入ください。"
    st.session_state.messages.append({"role": "assistant", "content": follow_up})
    with st.chat_message("assistant"):
        st.markdown(follow_up)
    st.session_state.expect_tech_suggestion = False
    st.session_state.tech_suggested = True

# --------------------------------------------
# 9. 会話履歴を画面に表示
# --------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --------------------------------------------
# 10. チャット入力フォーム
# --------------------------------------------
user_input = st.chat_input("入力してください…")

if user_input:
    if st.session_state.expect_tech_suggestion:
        # 技術深掘りステップ
        suggest_technologies(user_input)
    elif not st.session_state.ipc_candidates:
        # IPC候補生成ステップ
        generate_ipc_candidates(user_input)
    elif st.session_state.expect_search_params:
        # 検索パラメータ生成ステップ
        finalize_search_parameters(user_input)
    else:
        # 新規キーワード入力時のリセット
        st.session_state.ipc_candidates = []
        st.session_state.expect_search_params = False
        st.session_state.expect_tech_suggestion = True
        st.session_state.tech_suggested = False
        suggest_technologies(user_input)

# --- Streamlit UIの続き ---
# 検索パラメータJSONが生成されたら検索・ベクトル化・ランキング・表示
if st.session_state.get("search_ready", False):
    params = {
        "ipc_codes": st.session_state.ipc_codes,
        "countries": st.session_state.countries,
        "assignees": st.session_state.assignees,
        "publication_from": st.session_state.publication_from
    }
    st.markdown("### 特許データ検索・ベクトル化・類似度ランキング")
    if st.button("特許検索・類似度ランキング実行"):
        with st.spinner("BigQueryから特許データ抽出中..."):
            df = search_patents_by_params(params)
        if df.empty:
            st.warning("該当する特許が見つかりませんでした。")
        else:
            st.session_state["search_df"] = df  # ← セッションに保存
            st.success(f"{len(df)}件の特許を取得しました。ベクトル化・ランキングを実行します。")
            st.markdown("#### 取得特許一覧（検索条件に合致したもの）")
            st.dataframe(df)
    # --- ここからは常にセッションのdfを参照 ---
    df = st.session_state.get("search_df")
    if df is not None and not df.empty:
        st.markdown("#### 検索意図や追加クエリ（ベクトル類似度計算用）")
        st.info("この欄には『知りたい内容』『重視したい観点』『追加キーワード』などを自然文で入力してください。例：AIによる水質異常検知の最新技術 など")
        query_text = st.text_input("検索意図や追加クエリ（ベクトル類似度計算用）", key="query_text")
        if st.button("類似度ランキング実行", key="rank_button") and query_text:
            try:
                texts = df["abstract"].fillna("").tolist()
                if not any(texts):
                    st.warning("特許要約（abstract）が空のため、類似度ランキングを実行できません。")
                else:
                    idx, sims = rank_by_similarity(query_text, texts, openai_api_key)
                    df_ranked = df.iloc[idx].copy()
                    df_ranked["similarity"] = sims[idx]
                    st.session_state["df_ranked"] = df_ranked  # ランキング結果をセッションに保存
                    st.session_state["explanations"] = None  # 解説リセット
                    st.dataframe(df_ranked)
                    csv = df_ranked.to_csv(index=False).encode("utf-8-sig")
                    st.download_button("CSVダウンロード", csv, "results.csv", "text/csv", key="csv_download")
            except Exception as e:
                st.error(f"類似度ランキング処理中にエラーが発生しました: {e}")

        # --- ランキング結果があればN件解説UIを常に表示 ---
        df_ranked = st.session_state.get("df_ranked")
        if df_ranked is not None and not df_ranked.empty:
            st.markdown("#### 上位N件の特許を選択し、日本語で解説")
            n_max = min(10, len(df_ranked))
            if "topn" not in st.session_state:
                st.session_state["topn"] = min(3, n_max)
            n = st.number_input("解説したい上位件数 (N)", min_value=1, max_value=n_max, value=st.session_state["topn"], step=1, key="topn")
            if st.button("選択したN件を日本語で解説", key="explain_button"):
                topN_df = df_ranked.head(n)
                explanations = []
                import openai
                client = openai.OpenAI(api_key=openai_api_key)
                for i, row in topN_df.iterrows():
                    jp_prompt = (
                        "以下は特許の要約です。専門用語も分かりやすく、200字程度で日本語で解説してください。\n"
                        "---\n"
                        f"{row['abstract']}"
                    )
                    try:
                        response = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[{"role": "system", "content": jp_prompt}]
                        )
                        jp_summary = response.choices[0].message.content.strip()
                    except Exception as e:
                        jp_summary = f"要約生成エラー: {e}"
                    explanations.append({"title": row['title'], "summary": jp_summary})
                st.session_state["explanations"] = explanations
            # --- 解説結果があれば表示 ---
            explanations = st.session_state.get("explanations")
            if explanations:
                for i, ex in enumerate(explanations, 1):
                    st.markdown(f"**{i}件目: {ex['title']}**")
                    st.info(ex["summary"])
