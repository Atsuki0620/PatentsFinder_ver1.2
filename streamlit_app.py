# --------------------------------------------
# 1. 共通設定・インポート
# --------------------------------------------
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage

# --------------------------------------------
# 2. ページ設定・タイトル・説明
# --------------------------------------------
st.set_page_config(page_title="💡 PatentsFinder ver1.2", page_icon="🔍")
st.title("🔍 PatentsFinder ver1.2")
st.write("""
このアプリは、会話を通じて「特許調査に必要な IPC (International Patent Classification) コード」を絞り込むための AI アシスタントです。
1. 左のサイドバー（または下部）の入力欄に OpenAI API Key を入力  
2. 『調査したい技術領域やキーワード』を会話形式で入力すると、AI が関連する IPC コードを提案します。  
3. 提案された IPC コードを使って、実際の特許検索エンジンで調査できます。
""")

# --------------------------------------------
# 3. サイドバー or 最上部：API キー入力
# --------------------------------------------
# （※ 画面上部に直接置くか、st.sidebar を使うかは好みで変更可）
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("まずは OpenAI API Key を入力してください。", icon="🗝️")
    st.stop()

# --------------------------------------------
# 4. LangChain の LLM インスタンス生成
#    ── “AI の役割” を定義する system_prompt を含める
# --------------------------------------------
SYSTEM_PROMPT = """
あなたは特許調査クエリジェネレータです。ユーザーは技術領域やキーワード、日付、国を自由形式で一度に入力します。AIは以下の流れで進め、最終的にJSON形式の検索条件を出力します。不足があればまとめて質問し、情報が揃えば追加質問なしに完結します。

ユーザー入力の解析
publication_from（公開日下限）、countries（国コード）、keywords（キーワード）が含まれれば抽出
ユーザー指定のIPCコードがあればそのまま使用
指定がなければkeywordsから関連IPCコードを3～5個一言説明付きで自動生成し、焦点とするサブトピックを提案して確認

IPCコード提案
例：入力が「逆浸透膜の機械学習」なら
C02F 1/00：逆浸透膜製造・利用
G06N 3/04：機械学習知識処理
G01N 33/569：膜性能評価
「機械学習を用いた製造高性能化」「性能予測技術」のどちらか、あるいは別の方向性を尋ねる

公開日と国の確認
IPCが確定したら一度に「公開日下限（YYYY-MM-DD）と国コード（JP, US, CNなど）」をカンマ区切りで質問
キーワード再確認（必要時）
初回入力でkeywordsが不明瞭なら、IPC提案と併せて追加のキーワードをリスト形式で尋ねる
出力
以下の4項目が揃ったら追加質問なしにJSONコードブロックで出力し、最後に「以上が特許検索条件になります…」と付加
{
  "ipc_codes": ["C02F 1/00", "G06N 3/04", …],
  "publication_from": "YYYY-MM-DD",
  "countries": ["JP", "US", "CN"],
  "keywords": ["逆浸透膜", "機械学習", …]
}

■注意■
IPCコードは番号選択禁止。サブトピックで深堀する
最終出力以外はJSON以外の余計なテキスト禁止
publication_fromとcountriesは同時にまとめて質問
"""



llm = ChatOpenAI(
    model_name="gpt-3.5-turbo", 
    openai_api_key=openai_api_key,
    temperature=0.2  # IPC 絞り込みではあまりブレを持たせないためやや低め
)

# --------------------------------------------
# 5. session_state で会話履歴を管理
# --------------------------------------------
# {"role":"user"/"assistant"/"system", "content": "..."} のリストを保持
if "messages" not in st.session_state:
    # 初回起動時にシステムプロンプトを挿入しておく
    st.session_state.messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

# --------------------------------------------
# 6. 既存の会話履歴を表示（system は表示せず user/assistant のみ描画）
# --------------------------------------------
for msg in st.session_state.messages:
    if msg["role"] in ["user", "assistant"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# --------------------------------------------
# 7. ユーザー入力フォーム
# --------------------------------------------
if prompt := st.chat_input("調査したい技術領域・対象期間・出願国・キーワードを入力してください…"):

    # 7-1. ユーザー発言をセッションに追加・表示
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

     # 7-2. LangChain 用にメッセージを HumanMessage/AIMessage/SystemMessage に変換
    lc_messages = []
    for m in st.session_state.messages:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            lc_messages.append(AIMessage(content=m["content"]))
        elif m["role"] == "system":
            lc_messages.append(SystemMessage(content=m["content"]))

    # 7-3. LLM に問い合わせ（IPC コードを絞り込む役割のまま応答）
    # ※ LangChain v0.1 系では system メッセージを自動検出しない場合があるため、
    #    その際は ChatPromptTemplate を使って system と user を分離する必要があります。
    #    ここでは単純化して「lc_messages に system/user を混在させる形」を想定。
    response = llm(lc_messages)
    ai_content = response.content

    # 7-4. AI の応答を表示・履歴に追加
    with st.chat_message("assistant"):
        st.markdown(ai_content)
    st.session_state.messages.append({"role": "assistant", "content": ai_content})

# --------------------------------------------
# 8. （オプション）提案 IPC コードのみを抽出してサマリ表示
# --------------------------------------------
# AI の回答は Markdown 形式で IPC コードのリストを返す想定なので、
# 必要に応じて「コードだけ抜き出して右ペインに表示する」など実装可。
# 例：
if "assistant" in [m["role"] for m in st.session_state.messages]:
    # 最後の assist メッセージを解析して IPC コードを箇条書きだけ抽出し、横に表示するなど
    last_ai = st.session_state.messages[-1]["content"]
    # 簡易的に「行頭が数字. 〜」を正規表現で拾うなどしてコードを抽出できる
    # この部分は後段の “拡張ポイント” で紹介
    # st.sidebar.markdown("### 提案された IPC コード")
    # st.sidebar.markdown(抜き出したリストを Markdown 化して表示)
    pass
