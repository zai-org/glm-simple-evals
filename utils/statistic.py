def count_reflection_actions(data, response_key):
    special_tokens = {
        "reflection": "<reflection>",
        "action": "<action>",
        "plan": "<plan>",
        "answer": "<answer>",
    }
    # Count the number of actions in the data
    special_tokens_count = {k: 0 for k in special_tokens}
    for row in data:
        response = row[response_key]
        for k, v in special_tokens.items():
            special_tokens_count[k] += response.count(v)
    average_special_tokens = {k: v / len(data) for k, v in special_tokens_count.items()}
    return average_special_tokens