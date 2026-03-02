from __future__ import annotations


def get_support_records(start_row_id: int) -> list[dict[str, object]]:
    records = [
        {
            "source_text": (
                "Greeting response policy: If the user says hi, hello, hey, good morning, "
                "good afternoon, or good evening, respond politely and briefly. Example "
                "response: Hello. How can I help you with the HR policy document today?"
            ),
            "source_column": "support_text",
        },
        {
            "source_text": (
                "Thanks response policy: If the user says thank you, thanks, or appreciate it, "
                "respond politely and briefly. Example response: You're welcome. Let me know "
                "if you want anything else from the HR policy."
            ),
            "source_column": "support_text",
        },
        {
            "source_text": (
                "Farewell response policy: If the user says bye, goodbye, see you, or take care, "
                "respond politely and briefly. Example response: Goodbye. If you need more help "
                "with the HR policy, feel free to return."
            ),
            "source_column": "support_text",
        },
    ]

    output: list[dict[str, object]] = []
    for offset, record in enumerate(records):
        output.append(
            {
                "row_id": start_row_id + offset,
                "source_text": record["source_text"],
                "source_column": record["source_column"],
            }
        )
    return output

