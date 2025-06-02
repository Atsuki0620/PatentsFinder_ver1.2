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
        "ipc_codes": st.session_state.ipc_candidates,
        "countries": countries,
        "assignees": assignees,
        "publication_from": publication_from
    }
    json_result = json.dumps(result, ensure_ascii=False, indent=2)

    # ④：画面表示用メッセージ
    final_message = (
        "以下が最終的な検索条件です。\n"
        f"```json\n{json_result}\n```"
        "以上が特許検索条件になります。このまま検索エンジンに貼り付けて調査を開始してください。"
    )
    st.session_state.messages.append({"role": "assistant", "content": final_message})
    with st.chat_message("assistant"):
        st.markdown(final_message)

    # パース待ちフラグをリセット
    st.session_state.expect_search_params = False
