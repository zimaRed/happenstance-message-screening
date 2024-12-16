# Happenstance Message Screening Module

A module to screen messages received by Happenstance and evaluate the effectiveness of the message screening functions being used.

This module loads test cases from a JSON file and uses a provided message screening function to determine if the test messages are "requests" or not. It then evaluates the screening function against all test cases and prints out results, including failed cases and a final score.

---

## Functions

### `validate_query(message_text)`
Runs the message through a sequence of filters to validate it.

### `evaluate(message_screener)`
Evaluates a given screening function against a set of test cases.

### `keyword_filter(message_text)`
Checks a message for disallowed keywords.

### `regex_filter(message_text)`
Checks a message against certain regex patterns.

### `llm_filter(message_text)`
Uses an LLM-based filter to validate queries.

### `main()`
Runs the evaluation using the `validate_query` function.

---

## How to Run an Evaluation

1. Ensure all dependencies are installed.
2. Make sure message_screen_examples.json is up-to-date with test cases.
3. Run `python message_screener.py` in the CLI to evaluate the current version of the validate_query function.
